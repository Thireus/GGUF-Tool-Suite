#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** fp4_rtn_metric.py estimates the SGLang quant types' error **#
#** by rounding to nearest, when no measured rows exist.      **#
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
#** Copyright © 2026 - Thireus.   𝓰ᵤₑₛₛᵢₙ𝓰, ᵦᵤₜ 𝓌ᵢₜₕ 𝒹ₑ𝒸ᵢₘₐₗₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# quant_assign.py needs two things before it can put a qtype in a recipe: a size
# (which comes from a tensors.<q>.map) and a degradation value (which comes from
# group0/kld_results.csv, produced by benchmark_each_tensor.sh running the whole
# model under llama-perplexity --kl-divergence). For a new qtype the second one
# costs a GPU campaign, and for nvfp4 there is not even a published shard set to
# benchmark.
#
# The two Blackwell FP4 types are the one case where that cost can be avoided
# honestly. Both are ROUND-TO-NEAREST and DETERMINISTIC, and both IGNORE the
# importance matrix - quantize_nvfp4() and quantize_mxfp4() in
# ggml/src/ggml-quants.c (llama.cpp b11654 / 1adda1caf) each open with
# GGML_UNUSED(quant_weights) and call the _ref row quantiser. So the exact bytes
# llama-quantize will write can be computed on a CPU from the BF16 weights, and
# with them the exact weight-space error.
#
# WHAT IT MEASURES, AND WHY THAT METRIC
# -------------------------------------
# rel_rmse = rmse(W, dequant(quant(W))) / rms(W), and sqnr_db, defined exactly as
# gguf_tensor_compare.py's value_metrics() defines them (lines 127-162) so the
# numbers are comparable with the rest of the suite. rel_rmse is the right
# summary here because it is scale-free, so it can be compared across tensors
# whose weight magnitudes differ by an order of magnitude.
#
# HOW A WEIGHT ERROR BECOMES A group0 DEGRADATION
# -----------------------------------------------
# It is NOT assumed. --anchor-dir points at a small sample of the model's OWN
# published shards for qtypes whose whole-model KLD is already known
# (group0/kld_results.csv). The same rel_rmse is measured on those, and the
# mapping is fitted on those points:
#
#     kld(q) = a * rel_rmse(q)^b
#
# The exponent b is not a free curve-fitting parameter chosen for looks: to
# second order a small weight perturbation dW produces a logit perturbation
# proportional to |dW| and a KL divergence proportional to its square, so b = 2
# is the prediction. The script reports the fitted b and the R^2 so that
# prediction can be checked rather than believed - if b comes out far from 2, say
# so in the recipe notes instead of shipping the number.
#
# This is a weaker instrument than benchmark_each_tensor.sh and it does not
# pretend otherwise. It is a way to get nvfp4 and mxfp4 onto the ladder at all,
# on a machine with no GPU hours to spare, with every step stated.
#
# Requires: pip install numpy pandas
# Requires: pip install "gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py"

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

# A fit log is kept next to the table it produced and is published with it, so
# every path it prints goes through the one renderer: relative inside the suite,
# a bare name outside it, never a home directory.  sglang_native.py, section 0.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
try:
    from sglang_native import publishable_path as _pub
except Exception:                                             # pragma: no cover
    def _pub(p, cwd=None, root=None):
        return (os.path.basename(p.rstrip(os.sep))
                if isinstance(p, str) and os.path.isabs(p) else p)

try:
    from gguf.gguf_reader import GGUFReader
    import gguf.quants as gguf_quants
    from gguf.constants import GGMLQuantizationType
except ImportError:
    sys.stderr.write(
        "Error: could not import gguf.\n"
        "Ensure ik_llama.cpp/gguf-py is in PYTHONPATH or install via pip.\n"
    )
    sys.exit(1)

EPS = 1e-12

# --- llama.cpp FP4 reference quantisers, transcribed -------------------------
# Every constant below is read from llama.cpp b11654 (1adda1caf). The
# transcription is verified, not asserted: --verify-against <an NVFP4 GGUF>
# re-quantises the same tensors and compares the packed bytes.
#
#   ggml-common.h:1126  kvalues_fp4 - the E2M1 codebook, stored DOUBLED
#   ggml-common.h:214   QK_MXFP4 32, block_mxfp4 = 1 E8M0 byte + 16 bytes
#   ggml-common.h:221   QK_NVFP4 64, QK_NVFP4_SUB 16,
#                       block_nvfp4 = 4 UE4M3 bytes + 32 bytes
#   ggml-quants.c:337   best_index_mxfp4 - nearest codebook entry, FIRST wins
#   ggml-quants.c:350   quantize_row_mxfp4_ref
#   ggml-quants.c:384   quantize_row_nvfp4_ref
#   ggml-impl.h:476     ggml_e8m0_to_fp32_half
#   ggml-impl.h:502     ggml_ue4m3_to_fp32 / ggml_fp32_to_ue4m3
#
# NOTE the two scale formats are very different animals, and it shows up in the
# results: MXFP4's E8M0 scale is a bare power of two (no mantissa at all), while
# NVFP4's UE4M3 has three mantissa bits. Neither block carries the per-tensor
# FP32 second-level scale that NVIDIA's reference NVFP4 layout has, so for
# typical LLM weight magnitudes (|w|max per 16 values ~ 0.03) the UE4M3 scale
# sits in its SUBNORMAL range, where it has ~7 usable levels. That is a property
# of the ggml block format, and it is measured here rather than argued about.
KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                       dtype=np.float32)


def ue4m3_to_fp32(codes):
    """ggml_ue4m3_to_fp32 (ggml-impl.h:502). Returns HALF the UE4M3 value, to
    match the doubled kvalues_fp4 convention."""
    c = np.asarray(codes, dtype=np.uint8)
    exp = ((c >> 3) & 0xF).astype(np.int32)
    man = (c & 0x7).astype(np.float32)
    raw = np.where(exp == 0,
                   np.ldexp(man, -9),
                   (1.0 + man / 8.0) * np.exp2((exp - 7).astype(np.float32)))
    out = (raw * 0.5).astype(np.float32)
    out[(c == 0) | (c == 0x7F)] = 0.0
    return out


def fp32_to_ue4m3(xin):
    """ggml_fp32_to_ue4m3 (ggml-impl.h:517), vectorised. Bit-for-bit, including
    the subnormal branch and the mantissa-carry that bumps the exponent."""
    x = np.asarray(xin, dtype=np.float32).copy()
    pos = x > 0.0
    x[~pos] = 0.0
    np.minimum(x, np.float32(448.0), out=x)
    bits = x.view(np.uint32)
    fp32_exp = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    fp32_man = ((bits >> 20) & 0x7).astype(np.int32)
    ue_exp = fp32_exp + 7
    man_sub = np.minimum((x * np.float32(512.0) + np.float32(0.5)).astype(np.int32), 7)
    out_sub = np.where(man_sub < 1, 0, man_sub).astype(np.uint8)
    round_bit = ((bits >> 19) & 1).astype(np.int32)
    ue_man = fp32_man + round_bit
    carry = ue_man > 7
    ue_man2 = np.where(carry, 0, ue_man)
    ue_exp2 = np.where(carry, ue_exp + 1, ue_exp)
    out_norm = np.where((ue_exp >= 15) | (ue_exp2 >= 15), 0x7E,
                        (ue_exp2 << 3) | ue_man2).astype(np.uint8)
    return np.where(pos, np.where(ue_exp <= 0, out_sub, out_norm), 0).astype(np.uint8)


def e8m0_half(e):
    """GGML_E8M0_TO_FP32_HALF (ggml-impl.h:476)."""
    d = np.ascontiguousarray(np.asarray(e, dtype=np.uint8).astype(np.uint32))
    bits = np.where(d < 2, np.uint32(0x00200000) << d, (d - 1) << 23).astype(np.uint32)
    return bits.view(np.float32)


def _best_index(x, d):
    """best_index_mxfp4 (ggml-quants.c:337) over a whole array at once.
    np.argmin returns the FIRST minimum, which is what the C loop's strict `<`
    does - it matters because kvalues_fp4[0] and [8] are both zero."""
    cand = d[..., None, None] * KVALUES_FP4      # (..., 1, 16)
    return np.argmin(np.abs(cand - x[..., None]), axis=-1).astype(np.uint8)


def quantize_nvfp4(w, chunk_rows=64):
    """quantize_row_nvfp4_ref. w is (rows, n_per_row) float32, n_per_row % 64 == 0."""
    rows, n = w.shape
    if n % 64:
        raise ValueError(f"nvfp4 needs n_per_row %% 64 == 0, got {n}")
    ue = np.empty((rows, n // 16), np.uint8)
    codes = np.empty((rows, n), np.uint8)
    for i in range(0, rows, chunk_rows):
        x = w[i:i + chunk_rows].reshape(-1, n // 16, 16)
        u = fp32_to_ue4m3(np.abs(x).max(axis=-1) / np.float32(6.0))
        ue[i:i + chunk_rows] = u
        codes[i:i + chunk_rows] = _best_index(x, ue4m3_to_fp32(u)).reshape(-1, n)
    return ue, codes


def dequantize_nvfp4(ue, codes):
    rows, nsub = ue.shape
    d = ue4m3_to_fp32(ue)
    return (KVALUES_FP4[codes].reshape(rows, nsub, 16) * d[..., None]).reshape(rows, nsub * 16)


def quantize_mxfp4(w, chunk_rows=64):
    """quantize_row_mxfp4_ref. w is (rows, n_per_row) float32, n_per_row % 32 == 0."""
    rows, n = w.shape
    if n % 32:
        raise ValueError(f"mxfp4 needs n_per_row %% 32 == 0, got {n}")
    e_out = np.empty((rows, n // 32), np.uint8)
    codes = np.empty((rows, n), np.uint8)
    for i in range(0, rows, chunk_rows):
        x = w[i:i + chunk_rows].reshape(-1, n // 32, 32)
        amax = np.abs(x).max(axis=-1).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.where(amax > 0.0, np.floor(np.log2(amax)) - 2 + 127, 0)
        e = np.clip(np.nan_to_num(e, nan=0.0, neginf=0.0), 0, 255).astype(np.uint8)
        e_out[i:i + chunk_rows] = e
        codes[i:i + chunk_rows] = _best_index(x, e8m0_half(e)).reshape(-1, n)
    return e_out, codes


def dequantize_mxfp4(e, codes):
    rows, nb = e.shape
    d = e8m0_half(e)
    return (KVALUES_FP4[codes].reshape(rows, nb, 32) * d[..., None]).reshape(rows, nb * 32)


FP4_QUANTISERS = {
    'nvfp4': (quantize_nvfp4, dequantize_nvfp4, 64,
              lambda ue, c: np.concatenate([ue.reshape(-1, 4),
                                            (c.reshape(-1, 4, 16)[:, :, 0:8] |
                                             (c.reshape(-1, 4, 16)[:, :, 8:16] << 4)).reshape(-1, 32)],
                                           axis=1)),
    'mxfp4': (quantize_mxfp4, dequantize_mxfp4, 32,
              lambda e, c: np.concatenate([e.reshape(-1, 1),
                                           (c.reshape(-1, 1, 32)[:, :, 0:16] |
                                            (c.reshape(-1, 1, 32)[:, :, 16:32] << 4)).reshape(-1, 16)],
                                          axis=1)),
}


# --- SGLang / ModelOpt reference quantisers, transcribed ----------------------
# The ggml quantisers above answer "what does llama-quantize write?".  These
# answer "what does a modelopt_mixed safetensors checkpoint hold?", which is a
# different question with a different answer for the SAME nominal format - most
# sharply for NVFP4, where NVIDIA's layout carries a per-tensor FP32
# `weight_scale_2` that ggml's block_nvfp4 does not; measured on the FFN
# tensors, that one difference is worth 1.27x the weight error.
#
# Sources, sglang 0.5.19.dev932+gd06f3bec8:
#   modelopt_quant.py:786-791  the NVFP4 dequant contract, verbatim:
#       eff = weight_scale.float() * weight_scale_2.float()
#       out = e2m1_values.view(rows, hidden // group_size, group_size) * eff
#     so the stored e4m3 block scale is DIVIDED by a per-tensor f32 global
#     scale, which is what recentres it inside e4m3's usable range.
#   fp8.py:210-211            "max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn
#                              (max 448)" - the 6.0 and 448.0 constants.
#   fp8_utils.py:1723         sf = ceil_to_ue8m0(x_amax / 448.0)   [MXFP8]
#   mxfp8_block_convert.py:56 sf = amax / 448.0                    [block FP8]
#   modelopt_quant.py:588-597 FP8's weight_scale is PER TENSOR, so its scale is
#                              amax over the whole tensor, not per row.
#
# All four are round-to-nearest and calibration-blind on the weights, exactly as
# the ggml FP4 pair is, so the same argument applies: the bytes are computable
# on a CPU and so is the weight error.  What they are NOT blind to is the
# ACTIVATIONS - NVFP4 here is W4A4 and FP8/MXFP8 are W8A8 - and that is measured
# nowhere in this file.  See the warning printed by --out-group0.

E4M3_MAX = np.float32(448.0)
E2M1_MAX = np.float32(6.0)


def _e4m3_step(a):
    """ULP of float8_e4m3fn at magnitude `a`: 3 mantissa bits, exponent bias 7,
    normals 2^-6 .. 1.75*2^8, subnormals in steps of 2^-9."""
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.floor(np.log2(np.where(a > 0, a, np.float32(1.0))))
    e = np.clip(np.nan_to_num(e, nan=-6.0, neginf=-6.0), -6.0, 8.0)
    return np.exp2(e - 3.0).astype(np.float32)


def round_e4m3(x):
    """Round to the nearest float8_e4m3fn value, saturating at +/-448.

    np.rint is round-half-to-even, which is what a hardware cast does.  The
    second pass exists because rounding can carry into the next binade (e.g.
    a = 1.9375 * 2^k rounds to 2^(k+1), where the ULP is twice as large).
    """
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x)
    s = _e4m3_step(a)
    q = np.rint(a / s) * s
    s2 = _e4m3_step(q)
    q = np.rint(a / s2) * s2
    q = np.minimum(q, E4M3_MAX)
    return np.copysign(q, x).astype(np.float32)


def quantize_sgl_fp8(w, chunk_rows=None):
    """quant_algo FP8: e4m3 weights, ONE f32 scale for the whole tensor."""
    amax = np.float32(np.abs(w).max())
    scale = np.float32(amax / E4M3_MAX) if amax > 0 else np.float32(1.0)
    return scale, round_e4m3(w / scale)


def dequantize_sgl_fp8(scale, q):
    return (q * scale).astype(np.float32)


def quantize_sgl_mxfp8(w, chunk_rows=None):
    """quant_algo MXFP8: e4m3 weights, UE8M0 (power-of-two) scale per [1, 32]."""
    rows, n = w.shape
    if n % 32:
        raise ValueError(f"mxfp8 needs n_per_row %% 32 == 0, got {n}")
    x = w.reshape(rows, n // 32, 32)
    amax = np.abs(x).max(axis=-1).astype(np.float32)
    r = amax / E4M3_MAX
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.ceil(np.log2(np.where(r > 0, r, np.float32(1.0))))
    e = np.clip(np.nan_to_num(e, nan=0.0, neginf=0.0), -127.0, 127.0)
    sf = np.exp2(e).astype(np.float32)
    return sf, round_e4m3(x / sf[..., None])


def dequantize_sgl_mxfp8(sf, q):
    rows, nb, blk = q.shape
    return (q * sf[..., None]).reshape(rows, nb * blk).astype(np.float32)


def quantize_sgl_fp8_pb_wo(w, chunk_rows=None):
    """quant_algo FP8_PB_WO: e4m3 weights, one f32 scale per 128x128 block."""
    rows, n = w.shape
    bn = bk = 128
    nbr, nbc = (rows + bn - 1) // bn, (n + bk - 1) // bk
    wp = np.pad(w, ((0, nbr * bn - rows), (0, nbc * bk - n)))
    blocks = wp.reshape(nbr, bn, nbc, bk)
    amax = np.abs(blocks).max(axis=(1, 3)).astype(np.float32)
    scale = np.where(amax > 0, amax / E4M3_MAX, np.float32(1.0)).astype(np.float32)
    return (scale, rows, n), round_e4m3(blocks / scale[:, None, :, None])


def dequantize_sgl_fp8_pb_wo(a, q):
    scale, rows, n = a
    nbr, bn, nbc, bk = q.shape
    out = (q * scale[:, None, :, None]).reshape(nbr * bn, nbc * bk)
    return out[:rows, :n].astype(np.float32)


# The E2M1 magnitudes, in the DOUBLED convention KVALUES_FP4 uses, and the
# parity of each code.  ModelOpt's cast to e2m1 is round-half-to-EVEN, where
# "even" is the low bit of the 3-bit code - so at an exact midpoint it picks
# codes 0, 2, 4 or 6 (magnitudes 0, 1, 2, 4 doubled).  ggml's best_index_mxfp4
# is a first-wins argmin, i.e. it always takes the SMALLER magnitude at a tie.
# The two therefore disagree on ~3.5 % of codes and nowhere else; that is worth
# 0.0064 rel_rmse against a quantisation error of 0.095, but getting it right is
# what turns "close" into a byte-exact verification against a published checkpoint.
_E2M1_MAG2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.float32)
# Midpoints between consecutive magnitudes. searchsorted(side='left') already
# gives the nearest code everywhere except exactly ON a midpoint, where it
# returns the LOWER code; RNE wants the even one, and the even neighbour is the
# upper code at every ODD-indexed midpoint (1.5 -> 2, 3.5 -> 4, 7 -> 8) and the
# lower one at every even-indexed midpoint (0.5 -> 0, 2.5 -> 2, 5 -> 4, 10 -> 8).
# So the whole tie rule is "+1 when the hit index is odd".
_E2M1_BOUNDS = np.array([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0], dtype=np.float32)


def _best_index_e2m1_rne(x, d):
    """Nearest E2M1 code for x / d, with round-half-to-even on exact midpoints."""
    safe = np.where(d > 0, d, np.float32(1.0))
    u = np.abs(x) / safe[..., None]
    i = np.searchsorted(_E2M1_BOUNDS, u, side='left')
    on_bound = (i < _E2M1_BOUNDS.size) & (u == _E2M1_BOUNDS[np.minimum(i, _E2M1_BOUNDS.size - 1)])
    code = (i + (on_bound & (i % 2 == 1))).astype(np.uint8)
    return np.where(x < 0, code | np.uint8(8), code).astype(np.uint8)


def quantize_sgl_nvfp4(w, chunk_rows=64, global_scale=None):
    """quant_algo NVFP4, ModelOpt layout - the TWO-LEVEL scale.

        weight_scale_2 = amax(tensor) / (6 * 448)                    f32, per tensor
        weight_scale   = to_e4m3( amax(block of 16) / (6 * ws2) )    e4m3, per 16
        effective      = weight_scale * weight_scale_2
        weight         = nearest E2M1 to (w / effective)

    The tensor's own maximum block therefore lands its scale on exactly 448, the
    top of e4m3, instead of at the absolute magnitude - which for typical LLM
    weights (|w|max per 16 ~ 0.03) is deep in e4m3's subnormal range.  That one
    f32 is the whole difference from ggml's block_nvfp4.

    Returned scales are HALVED to match the doubled KVALUES_FP4 codebook the
    ggml path already uses, so KVALUES_FP4 is shared and only the scale
    construction and the tie-break differ.

    `global_scale` OVERRIDES weight_scale_2, and passing it is MANDATORY for any
    tensor that is a shard of a fused module.  This is not a style preference,
    it is a correctness requirement discovered by measurement:

      * RadixArk's mlp.gate_proj and mlp.up_proj carry the SAME
        weight_scale_2 = 1.213437063e-04, which is up_proj's own
        amax / (6 * 448); gate_proj's own amax would give 7.011776879e-05.
        mlp.down_proj, which is not fused with anything, uses its own.
      * SGLang then does `weight_scale_2 = layer.weight_scale_2.max()`
        (modelopt_quant.py:1807) and builds ONE alpha for the whole fused GEMM.
        So a shard written with a SMALLER ws2 than its siblings is dequantised
        with the larger one and its weights come out scaled up by the ratio -
        a gross error, not a degradation.  The W4A16 path at least warns
        ("weight_scale_2 differs across fused parallel layers", :2188); the W4A4
        path takes the max silently.
      * Writing it the way ModelOpt does costs essentially nothing: on
        blk.30.ffn_gate it moves rel_rmse from 0.094889 to 0.094905, 0.017 %.

    With the shared scale supplied, this transcription reproduces RadixArk's
    block scales EXACTLY (100.0000 % of e4m3 bytes on every tensor tested).
    """
    rows, n = w.shape
    if n % 16:
        raise ValueError(f"sgl_nvfp4 needs n_per_row %% 16 == 0, got {n}")
    if global_scale is not None:
        ws2 = np.float32(global_scale)
    else:
        amax_t = np.float32(np.abs(w).max())
        ws2 = np.float32(amax_t / (E2M1_MAX * E4M3_MAX)) if amax_t > 0 else np.float32(1.0)
    half = np.empty((rows, n // 16), np.float32)
    codes = np.empty((rows, n), np.uint8)
    for i in range(0, rows, chunk_rows):
        x = w[i:i + chunk_rows].reshape(-1, n // 16, 16)
        bmax = np.abs(x).max(axis=-1).astype(np.float32)
        bs = round_e4m3(bmax / (E2M1_MAX * ws2))
        h = (bs * ws2 * np.float32(0.5)).astype(np.float32)
        half[i:i + chunk_rows] = h
        codes[i:i + chunk_rows] = _best_index_e2m1_rne(x, h).reshape(-1, n)
    return half, codes


def dequantize_sgl_nvfp4(half, codes):
    rows, nsub = half.shape
    return (KVALUES_FP4[codes].reshape(rows, nsub, 16) *
            half[..., None]).reshape(rows, nsub * 16).astype(np.float32)


SGLANG_QUANTISERS = {
    'sgl_nvfp4':     (quantize_sgl_nvfp4,     dequantize_sgl_nvfp4,     16, None),
    'sgl_fp8':       (quantize_sgl_fp8,       dequantize_sgl_fp8,        1, None),
    'sgl_fp8_pb_wo': (quantize_sgl_fp8_pb_wo, dequantize_sgl_fp8_pb_wo, 128, None),
    'sgl_mxfp8':     (quantize_sgl_mxfp8,     dequantize_sgl_mxfp8,     32, None),
}

# sgl_nvfp4a16 stores exactly the same bytes as sgl_nvfp4 (only its GEMM differs:
# marlin W4A16 instead of the FP4 tensor core), so its WEIGHT error is identical
# by construction and it is deliberately not measured separately.
SGLANG_ALIAS = {'sgl_nvfp4a16': 'sgl_nvfp4'}

RTN_QUANTISERS = dict(FP4_QUANTISERS)
RTN_QUANTISERS.update(SGLANG_QUANTISERS)


# --- metrics, matching gguf_tensor_compare.py:127-162 ------------------------
def value_metrics(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    diff = a - b
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    rms_signal = math.sqrt(float(np.mean(a * a)))
    var_signal = float(np.var(a))
    var_error = float(np.var(diff)) + EPS
    return {
        'rel_rmse': rmse / (rms_signal + EPS),
        'sqnr_db': 10.0 * math.log10(max(var_signal, 1e-12) / var_error),
        'mse': mse,
        'max_abs': float(np.max(np.abs(diff))),
    }


# --- GGUF reading ------------------------------------------------------------
def tensor_to_f32(t):
    """Return (weights as (rows, n_per_row) float32, n_per_row)."""
    name = t.tensor_type.name
    shp = tuple(int(x) for x in t.shape)
    n_per_row = shp[0]
    if name == 'BF16':
        u = t.data if t.data.dtype == np.uint16 else t.data.view(np.uint16)
        w = (u.ravel().astype(np.uint32) << 16).view(np.float32)
    elif name == 'F32':
        w = t.data.view(np.float32).ravel()
    elif name == 'F16':
        w = t.data.view(np.float16).ravel().astype(np.float32)
    else:
        w = gguf_quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).ravel()
    return w.reshape(-1, n_per_row), n_per_row


def index_split(directory, wanted=None):
    """tensor name -> (path, tensor_type name). Scans a SPECIAL_SPLIT directory."""
    idx = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.gguf'))):
        try:
            r = GGUFReader(path)
        except Exception as e:
            print(f"[Warning] skipping {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        for t in r.tensors:
            if wanted is None or t.name in wanted:
                idx.setdefault(t.name, (path, t.tensor_type.name))
    return idx


def read_tensor(path, name):
    for t in GGUFReader(path).tensors:
        if t.name == name:
            return tensor_to_f32(t)
    raise KeyError(name)


# --- reading a real ModelOpt safetensors checkpoint ---------------------------
# No torch: a 256-entry lookup decodes float8_e4m3fn, and safetensors headers are
# a length-prefixed JSON blob followed by raw little-endian payloads.

def _e4m3_lut():
    b = np.arange(256, dtype=np.uint8)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    man = (b & 0x7).astype(np.float32)
    val = np.where(exp == 0, man * np.float32(2.0 ** -9),
                   (1.0 + man / 8.0) * np.exp2((exp - 7).astype(np.float32)))
    val = np.where((b & 0x7F) == 0x7F, np.nan, val)
    return np.where((b >> 7) != 0, -val, val).astype(np.float32)


E4M3_LUT = _e4m3_lut()


def read_safetensors_index(model_dir):
    """{tensor name -> (path, dtype, shape, offset, nbytes, data_start)}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, '*.safetensors'))):
        with open(f, 'rb') as fh:
            hlen = int.from_bytes(fh.read(8), 'little')
            hdr = json.loads(fh.read(hlen))
        base = 8 + hlen
        for name, meta in hdr.items():
            if name == '__metadata__':
                continue
            a, b = meta['data_offsets']
            out[name] = (f, meta['dtype'], tuple(meta['shape']), a, b - a, base)
    return out


def read_safetensor(idx, name):
    f, dtype, shape, a, nb, base = idx[name]
    with open(f, 'rb') as fh:
        fh.seek(base + a)
        raw = fh.read(nb)
    if dtype == 'F32':
        arr = np.frombuffer(raw, dtype='<f4')
    elif dtype == 'BF16':
        arr = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    elif dtype == 'F16':
        arr = np.frombuffer(raw, dtype='<f2').astype(np.float32)
    elif dtype in ('U8', 'F8_E4M3'):
        arr = np.frombuffer(raw, dtype=np.uint8)
    else:
        raise ValueError(f'{name}: unhandled safetensors dtype {dtype}')
    return arr.reshape(shape) if shape else arr, dtype


def verify_modelopt(model_dir, names, quantisable, limit=8):
    """Prove the sgl_nvfp4 transcription against a published ModelOpt checkpoint.

    This is the SGLang-side equivalent of --verify-against, and it is a stronger
    check than that one: it compares the packed 4-bit CODES we would write with
    the codes NVIDIA's own toolchain wrote for the same weights, plus both
    dequantisations against the BF16 source.  A large code disagreement with a
    small dequantisation disagreement means a tie-break convention differs; a
    large disagreement in BOTH means the reference is not a quantisation of
    these weights at all (which is exactly how a third-party NVFP4 GGUF was
    caught).
    """
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import sglang_native as sgl
    idx = read_safetensors_index(model_dir)
    checked = 0
    for name in names:
        info = sgl.map_tensor(name)
        if info is None:
            continue
        prefix = info[0]
        if f'{prefix}.weight_scale_2' not in idx or f'{prefix}.weight' not in idx:
            continue
        packed, dt = read_safetensor(idx, f'{prefix}.weight')
        if dt != 'U8':
            continue                      # FP8 tensor, not an NVFP4 one
        wsc, _ = read_safetensor(idx, f'{prefix}.weight_scale')
        ws2, _ = read_safetensor(idx, f'{prefix}.weight_scale_2')
        w, n_per_row = read_tensor(quantisable[name][0], name)
        if packed.shape != (w.shape[0], w.shape[1] // 2):
            print(f"[Verify] {name}: shape {packed.shape} vs {w.shape} - skipped",
                  file=sys.stderr)
            continue
        rows, half = packed.shape
        their_codes = np.empty((rows, half * 2), np.uint8)
        their_codes[:, 0::2] = packed & 0x0F
        their_codes[:, 1::2] = packed >> 4
        ws2v = np.float32(np.asarray(ws2).reshape(-1)[0])
        own = np.float32(np.abs(w).max() / (E2M1_MAX * E4M3_MAX))
        their_half = (E4M3_LUT[wsc] * ws2v * np.float32(0.5)).astype(np.float32)
        theirs = dequantize_sgl_nvfp4(their_half, their_codes)
        # Use the REFERENCE's weight_scale_2 rather than recomputing it: for a
        # fused shard it is the shared max over the module, which no single
        # tensor can derive on its own. Whether we agree on the value of ws2 is
        # reported separately, above.
        my_half, my_codes = quantize_sgl_nvfp4(
            w, global_scale=np.float32(np.asarray(ws2).reshape(-1)[0]))
        mine = dequantize_sgl_nvfp4(my_half, my_codes)
        same_codes = 100.0 * float(np.mean(my_codes == their_codes))
        same_scales = 100.0 * float(np.mean(my_half == their_half))
        shared = '' if abs(float(ws2v) / float(own) - 1.0) < 1e-4 else \
                 f'  [ws2 SHARED across the fused module: {float(ws2v):.6g} vs own {float(own):.6g}]'
        print(f"[Verify] {name:32s} codes {same_codes:8.4f}%  scales {same_scales:8.4f}%  "
              f"rel_rmse mine {value_metrics(w, mine)['rel_rmse']:.6f}  "
              f"theirs {value_metrics(w, theirs)['rel_rmse']:.6f}  "
              f"mine-vs-theirs {value_metrics(theirs, mine)['rel_rmse']:.6f}{shared}", file=sys.stderr)
        checked += 1
        if checked >= limit:
            break
    if not checked:
        print("[Verify] no NVFP4 tensor in the reference matched a selected tensor",
              file=sys.stderr)


# --- the anchored fit --------------------------------------------------------
def fit_anchors(points):
    """points: list of (qtype, rel_rmse, known_kld). Returns (a, b, r2)."""
    x = np.log(np.array([p[1] for p in points], dtype=np.float64))
    y = np.log(np.array([p[2] for p in points], dtype=np.float64))
    b, la = np.polyfit(x, y, 1)
    pred = la + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return math.exp(la), float(b), (1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def main():
    p = argparse.ArgumentParser(
        description="Measure the weight-space error of the Blackwell FP4 GGML types "
                    "directly from the weights, and map it onto the model's group0 KLD scale.")
    p.add_argument('--split-dir', required=True,
                   help='Directory of the BF16 (or F16/F32) SPECIAL_SPLIT shards - the quantisation source.')
    p.add_argument('--qtypes', nargs='+', default=['nvfp4', 'mxfp4'],
                   help='FP4 qtypes to compute from the weights (default: nvfp4 mxfp4).')
    p.add_argument('--tensors', nargs='+', default=None,
                   help='Explicit tensor names. Default: every quantisable tensor found in --split-dir.')
    p.add_argument('--tensors-regex', default=None,
                   help='Restrict to tensor names matching this regex.')
    p.add_argument('--sample-per-class', type=int, default=0,
                   help='Instead of all tensors, take at most N per structural class '
                        '(blk.<n>.<class>), evenly spaced across layers. 0 = no sampling.')
    p.add_argument('--anchor-dir', default=None,
                   help='Directory holding <qtype>/ subdirectories of published quantised shards for '
                        'the anchor qtypes. Their rel_rmse is MEASURED (not modelled) against --split-dir.')
    p.add_argument('--anchor-qtypes', nargs='+', default=None,
                   help='Restrict the fit to these anchor qtypes. Use it to fit LOCALLY, on the anchors '
                        'nearest the FP4 error, instead of across the whole ladder - see --local-fit-k.')
    p.add_argument('--local-fit-k', type=int, default=5,
                   help='Also report a fit using only the K anchors closest in rel_rmse to the FP4 types '
                        '(default 5, 0 disables). The global fit spans four bpw octaves and several '
                        'quantiser families with different imatrix leverage, so its exponent is not the '
                        'local one; the local fit is the one to extrapolate with, and its exponent can be '
                        'checked against the theoretical 2.')
    p.add_argument('--use-local-fit', action='store_true',
                   help='Use the local fit (not the global one) for the values written by --out-group0.')
    p.add_argument('--anchor-metrics-csv', default=None,
                   help=('Reuse anchor rel_rmse values from a previously written --out-metrics CSV '
                         'instead of re-measuring them. The anchors cost a 2.11 GiB download of '
                         'published shard sets and never change, so once measured they are data; '
                         'the fit that consumes them is a second of arithmetic. Rows whose '
                         '`elements` column is empty are anchor rows.'))
    p.add_argument('--local-fit-per-qtype', action='store_true',
                   help=('Choose the local fit\'s nearest-k anchors PER MEASURED QTYPE rather than '
                         'once around their mean. Required whenever the measured set spans a wide '
                         'error range - a 4.5 bpw type and an 8 bpw type have nothing to say about '
                         'each other\'s local slope. Off by default so existing runs are unchanged.'))
    p.add_argument('--anchor-kld-csv', default=None,
                   help="The model's group0 KLD CSV (QTYPE,group0) supplying the known KLD of each anchor.")
    p.add_argument('--out-per-tensor', default=None,
                   help='Write a per-tensor CSV in kld_results.csv shape (rows = QTYPE, columns = tensor).')
    p.add_argument('--out-metrics', default=None,
                   help='Write the long-form per (tensor, qtype) metric table as CSV.')
    p.add_argument('--out-group0', default=None,
                   help='Write a group0-shaped CSV (QTYPE,group0) with the predicted FP4 rows appended '
                        'to the anchor CSV, ready for --quant-degradation-csv.')
    p.add_argument('--verify-against', default=None,
                   help='An NVFP4/MXFP4 GGUF produced by llama-quantize. Every tensor present in both it '
                        'and --split-dir is re-quantised and the packed bytes compared. Proves the '
                        'transcription rather than asserting it. Needs a gguf-py whose enum knows NVFP4.')
    p.add_argument('--verify-modelopt', default=None,
                   help=('Directory of a published ModelOpt NVFP4 safetensors checkpoint '
                         '(e.g. RadixArk/Qwen3.8-27B-NVFP4). Compares the packed 4-bit codes and '
                         'the block scales this script would write against the ones NVIDIA\'s own '
                         'toolchain wrote for the same weights. The SGLang-side counterpart of '
                         '--verify-against.'))
    p.add_argument('--progress', action='store_true', help='Report progress to stderr.')
    args = p.parse_args()

    for q in args.qtypes:
        if q.lower() not in RTN_QUANTISERS and q.lower() not in SGLANG_ALIAS:
            p.error(f"--qtypes: {q!r} is not one of {sorted(RTN_QUANTISERS) + sorted(SGLANG_ALIAS)}")

    # ---- pick the tensor set -------------------------------------------------
    t0 = time.time()
    idx = index_split(args.split_dir)
    quantisable = {n: v for n, v in idx.items() if v[1] in ('BF16', 'F16', 'F32')}
    # F32 tensors are 1-D norms/biases the recipe never quantises; drop anything
    # whose row length cannot even hold one FP4 block.
    names = sorted(quantisable)
    if args.tensors:
        names = [n for n in names if n in set(args.tensors)]
    if args.tensors_regex:
        rx = re.compile(args.tensors_regex)
        names = [n for n in names if rx.search(n)]
    if args.sample_per_class:
        by_class = {}
        for n in names:
            by_class.setdefault(re.sub(r'^blk\.\d+\.', 'blk.N.', n), []).append(n)
        picked = []
        for c, members in sorted(by_class.items()):
            members.sort(key=lambda s: (int(re.match(r'blk\.(\d+)\.', s).group(1)) if s.startswith('blk.') else -1, s))
            k = min(args.sample_per_class, len(members))
            step = len(members) / k
            picked += [members[int(i * step)] for i in range(k)]
        names = sorted(set(picked))
    if not names:
        sys.exit("No tensors selected.")
    print(f"[Info] {len(names)} tensor(s) selected from {_pub(args.split_dir)}", file=sys.stderr)

    # ---- FP4 metrics from the weights ---------------------------------------
    rows = []          # (tensor, qtype, metrics dict)
    skipped = []
    for i, name in enumerate(names, 1):
        path, _ = quantisable[name]
        try:
            w, n_per_row = read_tensor(path, name)
        except Exception as e:
            skipped.append((name, 'read', str(e)))
            continue
        for q in args.qtypes:
            qf, dqf, blk, _pack = RTN_QUANTISERS[SGLANG_ALIAS.get(q.lower(), q.lower())]
            if n_per_row % blk:
                # Not a size the type can hold. llama.cpp THROWS for nvfp4 here
                # (llama-quant.cpp:372-410), so this is a genuine exclusion.
                skipped.append((name, q, f'n_per_row {n_per_row} %% {blk} != 0'))
                continue
            a, c = qf(w)
            m = value_metrics(w, dqf(a, c))
            m['elements'] = int(w.size)
            rows.append((name, q.lower(), m))
        if args.progress and i % 25 == 0:
            print(f"[Info]   {i}/{len(names)} tensors, {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"[Info] FP4 metrics done in {time.time()-t0:.1f}s ({len(rows)} rows, {len(skipped)} skipped)",
          file=sys.stderr)
    for s in skipped[:10]:
        print(f"[Info]   skipped {s[0]} [{s[1]}]: {s[2]}", file=sys.stderr)

    # ---- optional: verify the transcription against llama.cpp's own output ---
    if args.verify_against:
        print(f"[Info] verifying transcription against {args.verify_against}", file=sys.stderr)
        ref = {t.name: t for t in GGUFReader(args.verify_against).tensors}
        checked = 0
        for name in names:
            rt = ref.get(name)
            if rt is None or rt.tensor_type.name.lower() not in FP4_QUANTISERS:
                continue
            q = rt.tensor_type.name.lower()
            qf, _dq, blk, pack = FP4_QUANTISERS[q]
            path, _ = quantisable[name]
            w, n_per_row = read_tensor(path, name)
            if n_per_row % blk:
                continue
            a, c = qf(w)
            mine = pack(a, c)
            theirs = np.asarray(rt.data).view(np.uint8).reshape(mine.shape[0], -1)
            same = 100.0 * float(np.mean(mine == theirs))
            print(f"[Verify] {name:34s} {q} bytes identical: {same:8.4f}%", file=sys.stderr)
            checked += 1
            if checked >= 12:
                break
        if not checked:
            print("[Verify] no comparable tensor found", file=sys.stderr)

    if args.verify_modelopt:
        print(f"[Info] verifying the ModelOpt NVFP4 transcription against "
              f"{args.verify_modelopt}", file=sys.stderr)
        verify_modelopt(args.verify_modelopt, names, quantisable)

    # ---- anchors -------------------------------------------------------------
    anchor_metrics = {}     # qtype -> list of (tensor, rel_rmse)
    if args.anchor_dir:
        for sub in sorted(os.listdir(args.anchor_dir)):
            d = os.path.join(args.anchor_dir, sub)
            if not os.path.isdir(d):
                continue
            aidx = index_split(d)
            for name, (apath, _tt) in sorted(aidx.items()):
                if name not in quantisable:
                    continue
                try:
                    w, _ = read_tensor(quantisable[name][0], name)
                    aw, _ = read_tensor(apath, name)
                except Exception as e:
                    print(f"[Warning] anchor {sub}/{name}: {e}", file=sys.stderr)
                    continue
                if aw.shape != w.shape:
                    print(f"[Warning] anchor {sub}/{name}: shape {aw.shape} vs {w.shape}", file=sys.stderr)
                    continue
                anchor_metrics.setdefault(sub.lower(), []).append((name, value_metrics(w, aw)['rel_rmse']))
        for q, v in sorted(anchor_metrics.items()):
            print(f"[Info] anchor {q:8s}: {len(v)} tensor(s), mean rel_rmse "
                  f"{np.mean([x[1] for x in v]):.6f}", file=sys.stderr)

    if args.anchor_metrics_csv:
        # Anchor rows are the ones with an empty `elements` column - see the
        # --out-metrics writer below. Loading them here is not a shortcut: the
        # anchors are measurements of PUBLISHED shard sets against a fixed BF16
        # source, so they are constants of this model, not of this run.
        loaded = 0
        for line in open(args.anchor_metrics_csv):
            parts = line.rstrip('\n').split(',')
            if len(parts) < 4 or parts[0] == 'tensor' or parts[2] != '':
                continue
            try:
                anchor_metrics.setdefault(parts[1].strip().lower(), []).append(
                    (parts[0].strip(), float(parts[3])))
                loaded += 1
            except ValueError:
                continue
        print(f"[Info] loaded {loaded} anchor row(s) from {args.anchor_metrics_csv}",
              file=sys.stderr)
        for q, v in sorted(anchor_metrics.items()):
            print(f"[Info] anchor {q:8s}: {len(v)} tensor(s), mean rel_rmse "
                  f"{np.mean([x[1] for x in v]):.6f}", file=sys.stderr)

    # ---- outputs -------------------------------------------------------------
    if args.out_metrics:
        with open(args.out_metrics, 'w') as f:
            f.write("tensor,qtype,elements,rel_rmse,sqnr_db,mse,max_abs\n")
            for name, q, m in rows:
                f.write(f"{name},{q},{m['elements']},{m['rel_rmse']:.9g},{m['sqnr_db']:.9g},"
                        f"{m['mse']:.9g},{m['max_abs']:.9g}\n")
            for q, v in sorted(anchor_metrics.items()):
                for name, rr in v:
                    f.write(f"{name},{q},,{rr:.9g},,,\n")
        print(f"[Info] wrote {_pub(args.out_metrics)}", file=sys.stderr)

    if args.out_per_tensor:
        # kld_results.csv shape: first column QTYPE, then one column per tensor.
        cols = sorted({r[0] for r in rows})
        table = {}
        for name, q, m in rows:
            table.setdefault(q, {})[name] = m['rel_rmse']
        with open(args.out_per_tensor, 'w') as f:
            f.write("QTYPE," + ",".join(cols) + "\n")
            for q in sorted(table):
                f.write(q + "," + ",".join(f"{table[q].get(c, 404):.9g}" for c in cols) + "\n")
        print(f"[Info] wrote {_pub(args.out_per_tensor)}", file=sys.stderr)

    if args.out_group0:
        if not ((args.anchor_dir or args.anchor_metrics_csv) and args.anchor_kld_csv):
            sys.exit("--out-group0 needs --anchor-kld-csv plus either --anchor-dir "
                     "or --anchor-metrics-csv")
        known = {}
        for line in open(args.anchor_kld_csv):
            parts = line.strip().split(',')
            if len(parts) < 2 or parts[0].upper() == 'QTYPE':
                continue
            try:
                known[parts[0].strip().lower()] = float(parts[1])
            except ValueError:
                continue
        # Only anchors measured on the SAME tensors as the FP4 side may be used,
        # otherwise the fit compares different populations.
        fp4_by_q = {}
        for name, q, m in rows:
            fp4_by_q.setdefault(q, {})[name] = m['rel_rmse']
        common = None
        for q, v in anchor_metrics.items():
            s = {n for n, _ in v}
            common = s if common is None else (common & s)
        for q in fp4_by_q:
            common = common & set(fp4_by_q[q])
        common = sorted(common or [])
        if len(common) < 3:
            sys.exit(f"Only {len(common)} tensor(s) common to the FP4 set and every anchor - "
                     "cannot fit. Widen --tensors or the anchor sample.")
        print(f"[Info] fitting on {len(common)} common tensor(s): {', '.join(common)}", file=sys.stderr)
        pts = []
        for q, v in sorted(anchor_metrics.items()):
            if q not in known or known[q] <= 0:
                print(f"[Info] anchor {q}: no positive group0 value, skipped", file=sys.stderr)
                continue
            d = dict(v)
            rr = float(np.mean([d[n] for n in common]))
            pts.append((q, rr, known[q]))
        if len(pts) < 3:
            sys.exit("Need at least 3 usable anchors.")
        if args.anchor_qtypes:
            keep = {q.lower() for q in args.anchor_qtypes}
            pts = [x for x in pts if x[0] in keep]
            if len(pts) < 3:
                sys.exit("--anchor-qtypes left fewer than 3 usable anchors.")
        a, b, r2 = fit_anchors(pts)
        print(f"[Info] GLOBAL fit  kld = {a:.6g} * rel_rmse^{b:.4f}   R^2 = {r2:.5f}   "
              f"({len(pts)} anchors; theory predicts the exponent = 2)", file=sys.stderr)
        for q, rr, k in pts:
            print(f"[Info]   anchor {q:8s} rel_rmse={rr:.6f} known_kld={k:.6f} "
                  f"fitted={a*rr**b:.6f} ratio={a*rr**b/k:.3f}", file=sys.stderr)

        # The FP4 error sits at the far end of the anchor range, so the fit that
        # matters for extrapolation is the LOCAL one, over the anchors nearest to
        # it. A global fit across four bpw octaves mixes quantiser families whose
        # imatrix leverage differs, which flattens the exponent and hides the real
        # local slope.
        fp4_rr = {q: float(np.mean([fp4_by_q[q][n] for n in common])) for q in fp4_by_q}
        la, lb, lr2 = a, b, r2
        local_by_q = {}
        if args.local_fit_k and len(pts) > args.local_fit_k:
            # The reference point for "local" is the mean of the measured qtypes'
            # errors, which is only meaningful when they cluster.  With a 4.5 bpw
            # type and an 8 bpw type in the same run they do not - their errors
            # differ by ~5x - so --local-fit-per-qtype gives each its own
            # nearest-k instead.  Off by default: a run that measures only the
            # FP4 pair gets exactly the fit it always got.
            refs = ({q: fp4_rr[q] for q in fp4_rr} if args.local_fit_per_qtype
                    else {None: float(np.mean(list(fp4_rr.values())))})
            for key, ref_rr in sorted(refs.items(), key=lambda kv: (kv[0] or '')):
                near = sorted(pts, key=lambda x: abs(math.log(x[1]) - math.log(ref_rr)))[:args.local_fit_k]
                fa, fb, fr2 = fit_anchors(near)
                local_by_q[key] = (fa, fb)
                tag = f' [{key}]' if key else ''
                print(f"[Info] LOCAL  fit{tag}  kld = {fa:.6g} * rel_rmse^{fb:.4f}   R^2 = {fr2:.5f}   "
                      f"(nearest {len(near)}: {', '.join(x[0] for x in near)})", file=sys.stderr)
                for q, rr, k in near:
                    print(f"[Info]   anchor {q:8s} rel_rmse={rr:.6f} known_kld={k:.6f} "
                          f"fitted={fa*rr**fb:.6f} ratio={fa*rr**fb/k:.3f}", file=sys.stderr)
            if None in local_by_q:
                la, lb = local_by_q[None]

        ua, ub = (la, lb) if args.use_local_fit else (a, b)
        preds = {}
        for q in sorted(fp4_by_q):
            rr = fp4_rr[q]
            if args.use_local_fit and q in local_by_q:
                ua, ub = local_by_q[q]
            elif args.use_local_fit:
                ua, ub = la, lb
            worst = max(pts, key=lambda x: x[1])
            best = min(pts, key=lambda x: x[1])
            nearest = min(pts, key=lambda x: abs(math.log(x[1]) - math.log(rr)))
            preds[q] = (rr, ua * rr ** ub)
            if rr > worst[1]:
                where = f"EXTRAPOLATION: {rr/worst[1]:.2f}x beyond the worst anchor ({worst[0]})"
            elif rr < best[1]:
                where = f"EXTRAPOLATION: {best[1]/rr:.2f}x below the best anchor ({best[0]})"
            else:
                where = f"interpolated; nearest anchor {nearest[0]} at rel_rmse {nearest[1]:.6f}"
            print(f"[Info]   PREDICT {q:8s} rel_rmse={rr:.6f} -> group0 kld={preds[q][1]:.6f}   "
                  f"[global {a*rr**b:.6f} | local {ua*rr**ub:.6f} | "
                  f"scaled-from-{nearest[0]} {nearest[2]*(rr/nearest[1])**2:.6f}]  "
                  f"{where}", file=sys.stderr)
        with open(args.out_group0, 'w') as f:
            f.write("QTYPE,group0\n")
            for line in open(args.anchor_kld_csv):
                parts = line.strip().split(',')
                if len(parts) < 2 or parts[0].upper() == 'QTYPE':
                    continue
                f.write(f"{parts[0].strip()},{parts[1].strip()}\n")
            for q in sorted(preds):
                f.write(f"{q},{preds[q][1]:.6f}\n")
        print(f"[Info] wrote {_pub(args.out_group0)}", file=sys.stderr)


if __name__ == '__main__':
    main()
