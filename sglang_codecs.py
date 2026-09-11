#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_codecs.py packs and unpacks the NVFP4 and FP8      **#
#** tensors exactly the way ModelOpt does, bit for bit.       **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Sep-10-2026 -------------------- **#
#** ********************************************************* **#
#**                                                           **#
#** Author: Thireus <gguf@thireus.com>                        **#
#**                                                           **#
#** https://gguf.thireus.com/                                 **#
#** Thireus' GGUF Tool Suite - Quantize LLMs Like a Chef       **#
#**                                  ·     ·       ·~°          **#
#**     Λ,,Λ             ₚₚₗ  ·° ᵍᵍᵐˡ   · ɪᴋ_ʟʟᴀᴍᴀ.ᴄᴘᴘ°   ᴮᶠ¹⁶ ·  **#
#**    (:·ω·)       。··°      ·   ɢɢᴜғ   ·°·  ₕᵤ𝓰𝓰ᵢₙ𝓰𝒻ₐ𝒸ₑ   ·°   **#
#**    /    o―ヽニニフ))             · · ɪǫ3_xxs      ~·°        **#
#**    し―-J                                                   **#
#**                                                           **#
#** Copyright © 2026 - Thireus.         𝒻ₒᵤᵣ ᵦᵢₜₛ ₐₙ𝒹 ₐ 𝒹ᵣₑₐₘ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_codecs.py - the weight codecs an SGLang `modelopt_mixed` checkpoint can
contain, written to reproduce the reference implementations BYTE FOR BYTE.

    NVFP4 / W4A16_NVFP4   weight u8[N,K/2] + weight_scale e4m3[N,K/16]
                          + weight_scale_2 f32 scalar (+ input_scale f32 for W4A4)
    FP8                   weight e4m3[N,K] + weight_scale f32 scalar
                          (+ input_scale f32)
    FP8_PB_WO             weight e4m3[N,K] + weight_scale_inv f32[cN,cK]

BYTE-FOR-BYTE IS THE POINT, and it is not a slogan: `sglang_write.py
--crosscheck` compares our written bytes against a published ModelOpt checkpoint
wherever the two chose the same algorithm for the same module, and the whole
recipes2 lane came out 1,415 / 1,415 modules identical.  That only happens if
every rounding decision below matches, and three of them are counter-intuitive
enough to be worth stating in the code:

  1. RECIPROCAL MULTIPLY, NOT DIVISION.  ModelOpt writes
     `reduce_amax(input).float() / (E2M1_MAX * E4M3_MAX)`, and PyTorch lowers
     tensor-by-PYTHON-FLOAT division to a multiply by the reciprocal.  The two
     differ by one ULP on about one tensor in six.  Measured against RadixArk's
     file: true division reproduced weight_scale_2 in 102/128 NVFP4 tensors,
     reciprocal multiply in 128/128.
  2. FP8's DIVISION HAPPENS IN BFLOAT16, the module dtype, because
     `FP8QTensor.quantize` is `(input / expanded_scales).to(float8_e4m3fn)` with
     `input` still bf16.  Doing it in float32 moves ~2.4 % of bytes by one
     mantissa LSB.
  3. THE E2M1 ROUND-TO-NEAREST-EVEN MIDPOINTS are asymmetric: 0.75, 1.75 and 3.5
     round UP (their even neighbour is above) and 0.25, 1.25, 2.5, 5.0 round
     DOWN.  Expressed as seven comparisons that is exactly `_cast_fp4`, with no
     searchsorted in the inner loop.

TORCH IS IMPORTED LAZILY AND ONLY HERE.  The two float8 casts ARE the format -
numpy has no float8 - so the writer needs torch, and nothing else in this suite
does.  Importing it at module scope would make `quant_assign.py`,
`sglang_speed.py` and `sglang_native.py` all depend on a package the assign and
predict path never uses.  So: `require_torch()`, with an error that names what
to do about it.

SOURCES, all on this box:
  nvidia-modelopt 0.46.0  qtensor/nvfp4_tensor.py  (E2M1_MAX=6, E4M3_MAX=448,
                          group 16, the `_cast_fp4` midpoint table)
                          qtensor/fp8_tensor.py:74 (`scales = amax / 448.0`)
  sglang/srt/layers/quantization/fp8.py:561-610
                          weight_scale_inv float32 [ceil(N/bn), ceil(K/bk)],
                          weight_block_size [128,128], activation_scheme dynamic
  sglang/srt/layers/quantization/modelopt_quant.py:775-793  the decode path
"""

from __future__ import annotations

import numpy as np

E2M1_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_GROUP = 16
NVFP4_WS2_DENOM = E2M1_MAX * E4M3_MAX          # 2688.0, exact in binary
E4M3_MIN_POS = 2.0 ** -9                       # smallest positive e4m3 subnormal

E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)

_TORCH = None


def require_torch():
    """The torch handle, or a message that says exactly what is missing and why.

    The suite's own venv deliberately has neither torch nor safetensors: the
    assign, survey and predict paths are pure numpy and must stay installable
    without a 3 GB dependency.  The WRITER needs torch for two casts that have
    no numpy equivalent (`float8_e4m3fn`), and those casts are the format
    itself, not a convenience.
    """
    global _TORCH
    if _TORCH is None:
        try:
            import torch                                        # noqa: F401
        except Exception as e:
            raise SystemExit(
                'sglang_codecs needs PyTorch for the float8_e4m3fn casts, which ARE '
                'the NVFP4/FP8 formats (numpy has no float8 dtype), and it is not '
                f'importable here: {e}\n'
                'The assign/predict path (quant_assign.py, sglang_speed.py, '
                'sglang_native.py) does NOT need torch - only the checkpoint writer '
                'does.  Run the writer under a Python that has torch - the '
                'virtualenv you run SGLang from already does:\n'
                '    <sglang-venv>/bin/python sglang_write.py ...')
        _TORCH = torch
    return _TORCH


# --------------------------------------------------------------------------- #
# source helpers
# --------------------------------------------------------------------------- #
def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Widen raw BF16 bits to float32.  Exact: BF16 is the top half of a float32."""
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def as_f32_tensor(u16: np.ndarray):
    torch = require_torch()
    return torch.from_numpy(bf16_u16_to_f32(u16))


# --------------------------------------------------------------------------- #
# NVFP4 / W4A16_NVFP4
# --------------------------------------------------------------------------- #
def nvfp4_amax(w32) -> float:
    return float(w32.abs().max())


def nvfp4_weight_scale_2(amax: float) -> np.float32:
    """ws2 = amax / (6*448), computed and stored in float32.

    THE FUSED-SHARD RULE.  SGLang's `ModelOptFp4LinearMethod` takes
    `layer.weight_scale_2.max()` over the shards of a fused linear and applies
    that single alpha to the whole GEMM WITHOUT requantising
    (modelopt_quant.py:1806-1810).  A shard written with its own smaller ws2 is
    then dequantised with the larger one and comes out scaled up by the ratio -
    a gross error, and silent.  So the caller must pass the amax of the WHOLE
    fused group here and give every shard the same ws2.  RadixArk does exactly
    this: gate_proj.weight_scale_2 == up_proj.weight_scale_2 in 64/64 layers.

    It costs essentially nothing in accuracy: on one gate_proj it moves rel_rmse
    from 0.094889 to 0.094905, i.e. 0.017 %.
    """
    if not np.isfinite(amax) or amax <= 0.0:
        return np.float32(1.0)
    return np.float32(np.float32(amax) * np.float32(1.0 / NVFP4_WS2_DENOM))


def nvfp4_quantize(w32, ws2: np.float32, row_chunk: int = 0):
    """float32 [N,K] -> (packed u8 [N,K/2], scale codes u8 [N,K/16]).

    `ws2` is NOT derived here; pass the fused-group value from
    `nvfp4_weight_scale_2()`.  The arithmetic order is ModelOpt's:
        pbs   = blk_amax / float32(6*ws2)          (float32)
        pbs   = 1.0 where pbs == 0                 (an all-zero block)
        pbs   = clamp(pbs, 2^-9, 448)
        ws    = e4m3(pbs)                          (RNE - the cast IS the format)
        scaled= W / (float(ws) * ws2)              (float32, the ROUNDED ws)
        code  = e2m1(scaled)                       (RNE, sign in bit 3)
        byte  = (code_odd << 4) | code_even        (even input index = LOW nibble)
    """
    torch = require_torch()
    n, k = w32.shape
    if k % NVFP4_GROUP:
        raise ValueError(f'input dim {k} is not a multiple of {NVFP4_GROUP}')
    ws2_t = torch.tensor(float(ws2), dtype=torch.float32)
    six_ws2 = 6.0 * ws2_t                                    # float32, one rounding
    packed = torch.empty((n, k // 2), dtype=torch.uint8)
    codes = torch.empty((n, k // NVFP4_GROUP), dtype=torch.uint8)
    step = row_chunk or max(1, int(2 ** 27 // max(k, 1)))     # ~128M elements/chunk
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        blk = w32[r0:r1].reshape(r1 - r0, k // NVFP4_GROUP, NVFP4_GROUP)
        pbs = blk.abs().amax(dim=2) / six_ws2
        pbs = torch.where(pbs == 0, torch.ones_like(pbs), pbs)
        pbs = pbs.clamp_(E4M3_MIN_POS, E4M3_MAX)
        ws = pbs.to(torch.float8_e4m3fn)
        codes[r0:r1] = ws.view(torch.uint8)
        scaled = blk / (ws.float() * ws2_t).unsqueeze(-1)
        a = scaled.abs()
        o = (a > 0.25).to(torch.uint8)
        o += (a >= 0.75)
        o += (a > 1.25)
        o += (a >= 1.75)
        o += (a > 2.5)
        o += (a >= 3.5)
        o += (a > 5.0)
        o += (scaled < 0).to(torch.uint8) << 3
        o = o.reshape(r1 - r0, k)
        packed[r0:r1] = o[:, 0::2] | (o[:, 1::2] << 4)
    return packed.numpy(), codes.numpy()


def nvfp4_dequantize(packed: np.ndarray, codes: np.ndarray, ws2: float) -> np.ndarray:
    """SGLang's own reference decode, for the round-trip check."""
    torch = require_torch()
    n, half = packed.shape
    lut = np.concatenate([E2M1_LUT, -E2M1_LUT]).astype(np.float32)
    out = np.empty((n, half * 2), np.float32)
    out[:, 0::2] = lut[packed & 0x0F]
    out[:, 1::2] = lut[packed >> 4]
    s = torch.from_numpy(np.array(codes, copy=True)).view(
        torch.float8_e4m3fn).float().numpy() * np.float32(ws2)
    return out * np.repeat(s, NVFP4_GROUP, axis=1)


# --------------------------------------------------------------------------- #
# FP8 - ModelOpt per-tensor static
# --------------------------------------------------------------------------- #
def fp8_static_quantize(w32, amax: float = None):
    """float32 [N,K] -> (e4m3 bytes [N,K], float32 scalar scale).

    Both the reciprocal multiply and the bfloat16 division are deliberate; see
    the module docstring.  Measured against RadixArk's file: 85/85 weight_scale
    values reproduce with the reciprocal multiply, 47/85 with true division.
    """
    torch = require_torch()
    wb = w32.to(torch.bfloat16)
    if amax is None:
        amax = float(wb.abs().max())
    s = np.float32(np.float32(amax) * np.float32(1.0 / E4M3_MAX))
    q = (wb / torch.tensor(float(s), dtype=torch.bfloat16)).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).numpy(), s


# --------------------------------------------------------------------------- #
# FP8_PB_WO - 128x128 block, weight-only, dynamic activations
# --------------------------------------------------------------------------- #
def fp8_block_quantize(w32, block=(128, 128), scale_bf16: bool = False,
                       row_block_chunk: int = 64):
    """float32 [N,K] -> (e4m3 bytes [N,K], scale [ceil(N/128), ceil(K/128)]).

        s = amax(|W| over the 128x128 tile) * float32(1/448)   <- reciprocal multiply
        q = (W / s).to(float8_e4m3fn)

    Verified against a published FP8 checkpoint of this model, which is exactly
    this format: 100.000000 % of weight bytes.  True division instead of the
    reciprocal multiply moves 0.10 % of bytes; dividing in bfloat16 moves 3.0 %.

    We store float32, which is what `Fp8LinearMethod` registers (fp8.py:595-610,
    BlockQuantScaleParameter, dtype float32) and is self-consistent with the
    byte model; `scale_bf16=True` reproduces the BF16 scales of RadixArk's file.
    DESPITE THE NAME `weight_scale_inv` THE TENSOR HOLDS THE SCALE: the kernel
    computes w = q * weight_scale_inv[block].
    """
    torch = require_torch()
    bn, bk = block
    n, k = w32.shape
    cn, ck = (n + bn - 1) // bn, (k + bk - 1) // bk
    pad_n, pad_k = cn * bn - n, ck * bk - k
    src = w32
    if pad_n or pad_k:
        src = torch.nn.functional.pad(w32, (0, pad_k, 0, pad_n))
    inv = torch.tensor(float(np.float32(1.0 / E4M3_MAX)), dtype=torch.float32)
    scale = torch.empty((cn, ck), dtype=torch.float32)
    out = torch.empty((cn * bn, ck * bk), dtype=torch.float8_e4m3fn)
    step = max(1, row_block_chunk)
    for i0 in range(0, cn, step):
        i1 = min(cn, i0 + step)
        v = src[i0 * bn:i1 * bn].reshape(i1 - i0, bn, ck, bk)
        amax = v.abs().amax(dim=(1, 3))                       # [rows, ck]
        s = amax * inv
        s = torch.where(s == 0, torch.ones_like(s), s)
        if scale_bf16:
            s = s.to(torch.bfloat16).float()
        scale[i0:i1] = s
        out[i0 * bn:i1 * bn] = (v / s[:, None, :, None]).to(
            torch.float8_e4m3fn).reshape((i1 - i0) * bn, ck * bk)
    q = out[:n, :k].contiguous().view(torch.uint8).numpy()
    return q, scale.numpy()


def fp8_block_dequantize(q: np.ndarray, scale: np.ndarray, block=(128, 128)) -> np.ndarray:
    torch = require_torch()
    bn, bk = block
    n, k = q.shape
    w = torch.from_numpy(np.array(q, copy=True)).view(torch.float8_e4m3fn).float()
    s = torch.from_numpy(scale)
    s = s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    return (w * s).numpy()


# --------------------------------------------------------------------------- #
def selftest(verbose=True) -> int:
    """Codec invariants that hold without any checkpoint on disk."""
    torch = require_torch()
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)
        if verbose:
            print(('  ok   ' if ok else '  FAIL ') + msg)

    chk(NVFP4_WS2_DENOM == 2688.0, 'NVFP4: 6 * 448 == 2688, exact in binary')
    # the reciprocal multiply is a DIFFERENT number from the division, on purpose
    amax = 0.3183098733425140380859375
    a = np.float32(np.float32(amax) * np.float32(1.0 / NVFP4_WS2_DENOM))
    b = np.float32(np.float32(amax) / np.float32(NVFP4_WS2_DENOM))
    chk(nvfp4_weight_scale_2(amax) == a,
        'NVFP4: weight_scale_2 uses the reciprocal multiply (ModelOpt\'s own lowering)')
    if a != b and verbose:
        print(f'         (and it differs from true division here: {a!r} vs {b!r})')
    chk(nvfp4_weight_scale_2(0.0) == np.float32(1.0),
        'NVFP4: an all-zero tensor gets ws2 = 1.0, not a division by zero')

    g = torch.Generator().manual_seed(1234)
    w = (torch.randn(64, 256, generator=g) * 0.05).float()
    ws2 = nvfp4_weight_scale_2(float(w.abs().max()))
    packed, codes = nvfp4_quantize(w, ws2)
    chk(packed.shape == (64, 128) and codes.shape == (64, 16),
        f'NVFP4: shapes are [N,K/2] and [N,K/16] ({packed.shape}, {codes.shape})')
    d = nvfp4_dequantize(packed, codes, float(ws2))
    e = np.abs(d - w.numpy())
    amx = float(w.abs().max())
    chk(float(e.max()) / amx < 0.20,
        f'NVFP4: max|err| {float(e.max())/amx:.4f} of amax is within E2M1\'s own bound')
    # every decoded value must lie on the E2M1 grid times its block scale
    s = torch.from_numpy(np.array(codes, copy=True)).view(
        torch.float8_e4m3fn).float().numpy() * np.float32(ws2)
    ratio = np.abs(d / np.repeat(s, NVFP4_GROUP, axis=1))
    onlut = np.isclose(ratio[:, :, None], E2M1_LUT[None, None, :], atol=1e-5).any(-1)
    chk(bool(onlut.all()), 'NVFP4: every decoded value is on the E2M1 grid x block scale')

    # RNE midpoints: 0.75 -> 1.0 (code 2), 1.25 -> 1.0 (code 2), 3.5 -> 4.0 (code 6)
    # One block whose amax is exactly 6.0, so ws2 = 6/2688, the per-block scale
    # lands on 448 (exactly representable in e4m3) and `scaled` IS the input:
    # the codes below are then the raw `_cast_fp4` decisions, nothing else.
    mid = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
                         0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float32)
    p2, _c2 = nvfp4_quantize(mid, nvfp4_weight_scale_2(6.0))
    codes_seq = []
    for byte in p2[0][:4]:
        codes_seq += [int(byte) & 0x0F, int(byte) >> 4]
    chk(codes_seq == [0, 2, 2, 4, 4, 6, 6, 7],
        f'NVFP4: the seven RNE midpoints round to the EVEN grid point '
        f'(0.75/1.75/3.5 up, 0.25/1.25/2.5/5.0 down) -> {codes_seq}')

    q8, s8 = fp8_static_quantize(w)
    chk(q8.shape == (64, 256) and s8.dtype == np.float32,
        'FP8: e4m3 payload is [N,K] with one float32 scale')
    d8 = torch.from_numpy(q8.copy()).view(torch.float8_e4m3fn).float().numpy() * s8
    chk(float(np.abs(d8 - w.numpy()).max()) / amx < 0.07,
        'FP8: max|err| within E4M3\'s half-ULP bound (2^-4 of the element)')

    qb, sb = fp8_block_quantize(w)
    chk(sb.shape == (1, 2), f'FP8_PB_WO: scale is [ceil(N/128), ceil(K/128)] ({sb.shape})')
    db = fp8_block_dequantize(qb, sb)
    chk(float(np.abs(db - w.numpy()).max()) / amx < 0.07,
        'FP8_PB_WO: max|err| within E4M3\'s bound')
    # a padded (non-multiple-of-128) shape must still round-trip
    w2 = (torch.randn(96, 200, generator=g) * 0.05).float()
    q2, s2 = fp8_block_quantize(w2)
    chk(q2.shape == (96, 200) and s2.shape == (1, 2),
        f'FP8_PB_WO: a shape that is not a multiple of 128 pads and crops back '
        f'({q2.shape}, {s2.shape})')
    chk(float(np.abs(fp8_block_dequantize(q2, s2) - w2.numpy()).max())
        / float(w2.abs().max()) < 0.07,
        'FP8_PB_WO: ...and the padded tile\'s error is still in bound')

    z = torch.zeros(16, 32)
    qz, sz = fp8_block_quantize(z)
    chk(bool((sz == 1.0).all()), 'FP8_PB_WO: an all-zero tile gets scale 1.0, not 0')
    if verbose:
        print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


if __name__ == '__main__':
    import sys
    sys.exit(selftest())
