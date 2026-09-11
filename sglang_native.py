#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_native.py maps GGUF tensor names to SGLang's own   **#
#** quant types and knows which type each tensor can hold.    **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Sep-11-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.    ₗₒₛₜ ᵢₙ ₜₑₙₛₒᵣ ₜᵣₐₙₛₗₐₜᵢₒₙ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_native.py - the SGLang `modelopt_mixed` checkpoint format, expressed as a
size model this tool suite can assign over.

WHY THIS FILE EXISTS
--------------------
`quant_assign.py` is a size-targeted mixed-precision assigner whose world-view is

    damage(tensor, qtype) ~= loss(tensor) * deg(qtype)
    subject to  sum bytes(tensor, qtype) <= budget

Nothing in that is GGUF-specific.  What *is* GGUF-specific is the two things it
reads: `GGML_QUANT_SIZES` (how many bytes a qtype costs) and the
`tensors.<qtype>.map` files (per-tensor byte sizes, downloaded from the published
shard sets).  Point those at a different format family and the same optimiser
assigns over that family instead.

This module is that redirection for **SGLang's `modelopt_mixed`** format
(`sglang/srt/layers/quantization/modelopt_quant.py`, class
`ModelOptMixedPrecisionConfig`), which is a genuine per-layer recipe format:
`hf_quant_config.json` carries a `quantized_layers` map of
`{layer_prefix: {quant_algo, group_size}}` and `get_quant_method()` dispatches
per layer.  It gives us five algorithms plus "absent = BF16".

It contains four things and nothing else:

  1. THE TYPE TABLE - exact on-disk byte cost of every algo, read out of each
     method's `create_weights()` rather than assumed.  Verified to the byte
     against a real ModelOpt checkpoint (see `--verify-against`).
  2. THE LEGALITY RULES - which algo a tensor of a given shape may hold, read
     out of the same `create_weights()` (they raise, they do not fall back).
  3. THE NAME MAP - GGUF tensor name <-> HF module prefix, so a recipe line can
     become a `quantized_layers` entry, and so the fused-module groups can be
     derived rather than guessed.
  4. THE FUSED-MODULE CONSTRAINT - `modelopt_quant.py:967` raises
     "Mixed quant_algo within fused layer {prefix}" if the shards of a fused
     module disagree.  This is the one constraint the GGUF recipes do not have.
     The groups are derived from the model class' own
     `packed_modules_mapping`, so they are not folklore.

Everything here is CPU-only, offline, and reads no weights - only a
`tensors.bf16.map` (name, shape, elements) and, optionally, a reference
checkpoint's safetensors headers for the verification mode.

SOURCES, verified in the trees on this box on 2026-09-04
(sglang 0.5.19.dev932+gd06f3bec8, ~/sglang/venv-glm53-d06f3bec):
  srt/layers/quantization/modelopt_quant.py
      :796-935   ModelOptMixedPrecisionConfig / from_config
      :947-995   _resolve_quant_algo  (:967 = the fused-layer raise)
      :997-1053  get_quant_method     (the five-algo dispatch)
      :548-597   ModelOptFp8LinearMethod.create_weights   (per-TENSOR scales)
      :1715-1800 ModelOptFp4LinearMethod.create_weights   (NVFP4 W4A4)
      :2116-2185 ModelOptNvFp4A16LinearMethod.create_weights (W4A16, marlin)
      :700-760   ModelOptNvFp4EmbeddingMethod.create_weights
  srt/layers/quantization/fp8.py
      :234-352   Fp8Config (use_mxfp8 forces weight_block_size [1,32])
      :555-615   create_fp8_weight_  (block scale shape and dtype)
      :512-533   validate_block_quant_shapes
  srt/models/qwen3_5.py
      :1479-1483 packed_modules_mapping  (the fused-module truth)
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import struct
import sys
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 0. PATHS THAT REACH A PUBLISHED ARTEFACT
# =============================================================================
#
# A recipe footer, a checkpoint's `config.json` and a build manifest are all
# READ BY STRANGERS.  An absolute path from the machine that produced them says
# nothing to those readers - they do not have that directory - and says rather a
# lot about the machine: a login name, a directory layout, the name of whatever
# scratch tree the work happened in.  Neither half is wanted, so no tool in this
# suite ever writes one.
#
# The rule, in one sentence: a path INSIDE the suite (or inside the directory
# the command was run from) is recorded RELATIVE to it - which is also the form
# that reproduces the run - and a path outside it keeps only what identifies
# WHAT was read, never WHERE it sat.
#
# The one deliberate special case is a Hugging Face cache: `models--<org>--<repo>
# /snapshots/<revision>` names a PUBLIC artefact exactly, and a reader can fetch
# it, so that tail is kept and only the cache's own location is dropped.

SUITE_ROOT = os.path.dirname(os.path.abspath(__file__))

_HF_CACHE_TAIL = re.compile(r'(?:^|/)hub/(models--[^/]+/.*)$')


def _is_within(path: str, base: str) -> bool:
    """True when `path` is `base` itself or sits under it."""
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:                                        # different drives
        return False


def publishable_path(path, cwd: Optional[str] = None,
                     root: Optional[str] = None):
    """Render a filesystem path for an artefact somebody else will read.

    Relative input is returned untouched: it is already the published form, and
    it is what the user typed.  Absolute input is rewritten:

      * under `cwd`  -> relative to `cwd`   (`group0/kld_results.csv`)
      * under `root` -> relative to `cwd` when `cwd` is itself inside the suite,
        so the result still reproduces the run (`../../quant_assign.py`), and
        relative to `root` otherwise
      * a Hugging Face cache entry -> `<hf-cache>/models--org--repo/snapshots/…`
      * anything else -> `<external>/<basename>`, a name and no location.

    Never returns a home directory, a host name, or the name of a directory
    outside the suite.
    """
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        return path
    ap = os.path.normpath(path)
    cwd = os.path.abspath(cwd if cwd is not None else os.getcwd())
    root = os.path.abspath(root if root is not None else SUITE_ROOT)
    if _is_within(ap, cwd):
        return os.path.relpath(ap, cwd).replace(os.sep, '/')
    if _is_within(ap, root):
        base = cwd if _is_within(cwd, root) else root
        return os.path.relpath(ap, base).replace(os.sep, '/')
    m = _HF_CACHE_TAIL.search(ap.replace(os.sep, '/'))
    if m:
        return '<hf-cache>/' + m.group(1)
    return '<external>/' + os.path.basename(ap.rstrip('/'))


# =============================================================================
# 1. THE TYPE TABLE
# =============================================================================
#
# Byte cost is stated as (weight payload) + (block scales) + (per-tensor
# scalars).  K = n_per_row = the input/reduction dimension; N = n_rows = the
# output dimension.  GGUF `shape=(a, b)` is (K, N); safetensors `[N, K]`.  The
# two are transposes of each other and both are recorded below so nobody has to
# re-derive it at 2am.
#
# The per-tensor scalars are 4 or 8 bytes and are *kept* rather than rounded
# away, because the whole point of `--gpu-tensors-max-size <n>B` is to match
# another checkpoint to the byte, and 401 tensors x 8 B is a number the
# arithmetic either reproduces exactly or does not.

BF16 = 'sgl_bf16'

SGLANG_ALGOS: Dict[str, dict] = {
    # ---------------------------------------------------------------- BF16 --
    # Not an algo at all: it is what a layer gets when it is ABSENT from the
    # quantized_layers map (modelopt_quant.py:1027 -> UnquantizedLinearMethod).
    # It is in the pool because RadixArk itself spends 2.368 GiB on a BF16
    # embedding table, so the optimiser must be free to make the same call.
    BF16: dict(
        quant_algo=None,                 # absent from quantized_layers
        bpw=16.0,
        per_tensor_bytes=0,
        block=None,
        act='bf16',
        speed_cost=4.00,                 # cuBLAS/CUTLASS bf16 m16n8k16
        roles=('linear', 'embedding', 'other'),
        note='unquantised; the top of the pool and the only legal type for '
             'norms, biases and 1-D tensors',
    ),
    # ---------------------------------------------------------------- NVFP4 -
    # ModelOptFp4LinearMethod.create_weights (:1750-1800):
    #   weight        uint8          [N, K//2]        -> K*N/2 bytes
    #   weight_scale  float8_e4m3fn  [N, K//group]    -> K*N/16 bytes
    #   weight_scale_2 f32 scalar                     -> 4 bytes
    #   input_scale    f32 scalar                     -> 4 bytes
    # = 4 + 0.5 = 4.5 bpw, + 8 B/tensor.  W4A4: assigning this quantises the
    # ACTIVATIONS of that matmul too - see the risk note in the report.
    'sgl_nvfp4': dict(
        quant_algo='NVFP4', group_size=16,
        bpw=4.5,
        per_tensor_bytes=8,
        # Of those 8 B, 4 are the ACTIVATION scale, and an embedding has no
        # activation: `token_embd` is a gather, so ModelOpt registers
        # weight_scale_2 and no input_scale for it.  Declared rather than
        # special-cased so plan_tensor can subtract it for role='embedding'
        # without knowing which algo it is holding.
        act_scale_bytes=4,
        block=('e4m3', 16),
        act='fp4',
        speed_cost=1.00,                 # native CUTLASS FP4 GEMM on sm_120
        roles=('linear', 'embedding'),
        k_multiple=16,                   # create_weights raises otherwise
        note='W4A4, block 16, e4m3 sub-scale + per-tensor f32 global scale',
    ),
    # ------------------------------------------------------------ W4A16 FP4 -
    # ModelOptNvFp4A16LinearMethod: same bytes as NVFP4 minus input_scale (it
    # registers a placeholder purely so a fused loader can consume one, then
    # deletes it at :2185), so a checkpoint we author need not write it.
    # apply() is `apply_fp4_marlin_linear` - MARLIN, i.e. dequantise-in-register
    # then a bf16 MMA.  It is NOT the FP4 tensor core.  Same bytes as NVFP4 and
    # 4x the arithmetic cost: only ever worth it where W4A4 activation error is
    # the problem.
    'sgl_nvfp4a16': dict(
        quant_algo='W4A16_NVFP4', group_size=16,
        bpw=4.5,
        per_tensor_bytes=4,
        block=('e4m3', 16),
        act='bf16',
        speed_cost=4.00,                 # marlin W4A16, not the FP4 MMA
        roles=('linear',),
        k_multiple=16,
        note='W4A16 via marlin; same bytes as NVFP4, 4x the arithmetic cost',
    ),
    # ------------------------------------------------------------------ FP8 -
    # ModelOptFp8LinearMethod.create_weights (:588-597): weight_scale and
    # input_scale are both `_make_per_tensor_scale_parameter`, i.e. ONE f32 per
    # fused shard, not per channel.  So the honest effective bpw is 8.0 exactly
    # plus 8 bytes for the whole tensor - measured, not assumed: RadixArk's 208
    # FP8 tensors hold 7,214,202,880 weight elements in 7,214,204,544 bytes,
    # = 8.0000018 bpw, and 7,214,204,544 - 7,214,202,880 = 1,664 = 208 x 8.
    # (`process_weights_after_loading` converts to channelwise IN MEMORY via
    # `convert_to_channelwise`; that costs nothing on disk.)
    'sgl_fp8': dict(
        quant_algo='FP8',
        bpw=8.0,
        per_tensor_bytes=8,
        block=None,
        act='fp8_static',
        speed_cost=2.00,                 # dedicated sm_120 rowwise CUTLASS
        roles=('linear',),
        note='W8A8 e4m3, STATIC per-tensor weight and activation scales',
    ),
    # ------------------------------------------------------------ FP8_PB_WO -
    # Fp8Config(weight_block_size=[128,128], activation_scheme="dynamic").
    # create_fp8_weight_ (:595-609): weight_scale_inv f32
    # [ceil(N/128), ceil(K/128)], and NO input_scale (dynamic).  Costs 0.002 bpw
    # more than per-tensor FP8 and is materially more accurate for it: the
    # weight scale is per 128x128 block and the activation scale is per token
    # group at runtime rather than a single number frozen at calibration.
    'sgl_fp8_pb_wo': dict(
        quant_algo='FP8_PB_WO',
        bpw=8.0,
        per_tensor_bytes=0,
        block=('f32', (128, 128)),
        act='fp8_dynamic',
        speed_cost=2.10,                 # same tensor core, block-scale epilogue
        roles=('linear',),
        block_n=128, block_k=128,
        note='W8A8 e4m3, 128x128 block weight scales, dynamic activation scales',
    ),
    # ---------------------------------------------------------------- MXFP8 -
    # Fp8Config(use_mxfp8=True) forces weight_block_size [1,32] (fp8.py:284-288)
    # and the scale dtype to uint8 (UE8M0) (fp8.py:595).  So the scale tensor is
    # [N, ceil(K/32)] uint8 = 1 byte per 32 weights = 0.25 bpw.  No input_scale.
    'sgl_mxfp8': dict(
        quant_algo='MXFP8',
        bpw=8.25,
        per_tensor_bytes=0,
        block=('ue8m0', 32),
        act='mxfp8_dynamic',
        speed_cost=2.00,                 # native CUTLASS mm_mxfp8
        roles=('linear',),
        k_multiple=32,
        block_n=1, block_k=32,
        note='W8A8 e4m3, OCP block [1,32] UE8M0 scales, dynamic activations',
    ),
    # ------------------------------------------------------- INT4 g128 asym -
    # THE FIRST RUNG BELOW 4.5 BPW THIS BOX CAN RUN, and it is NOT a
    # modelopt_mixed algo: ModelOptMixedPrecisionConfig.get_quant_method
    # dispatches exactly five algos (modelopt_quant.py:1013-1050) and none of
    # them is an INT type.  This one lives in the compressed-tensors container
    # (`pack-quantized`) and reaches GPTQ-Marlin on sm_120.  See ALGO_CONTAINER
    # below: a recipe that names it CANNOT also name sgl_fp8*/sgl_nvfp4a16.
    #
    # Byte model read off the safetensors headers of a checkpoint this
    # writer produced, for an HF [N, K] weight:
    #   weight_packed     I32  [N, K/8]      -> N*K/2   bytes
    #   weight_scale      BF16 [N, K/128]    -> N*K/64  bytes
    #   weight_zero_point I32  [N/8, K/128]  -> N*K/256 bytes
    #   weight_shape      I64  [2]           -> 16      bytes
    # = 4 + 0.125 + 0.03125 = 4.15625 bpw + 16 B/tensor.  Verified against the
    # written file to the byte (that probe weighs 17,776,694,528 LM bytes).
    #
    # deg = 0.054767, MEASURED in-engine on 2026-09-05 - BETTER than
    # sgl_nvfp4's 0.066930 at 0.34375 FEWER bpw.  It loses only on prefill:
    # marlin dequantises into a bf16 MMA, measured k = 3.56 at chat-c1.
    'sgl_int4_g128': dict(
        quant_algo='INT4_G128_ASYM', group_size=128, num_bits=4, symmetric=False,
        bpw=4.15625,
        per_tensor_bytes=16,
        block=('bf16', 128),
        act='bf16',
        speed_cost=4.00,                 # the a-priori W16A16 tier; the MEASURED 3.556
                                         # lives in sglang_speed.MEASURED_K['chat-c1']
        roles=('linear',),
        k_multiple=128,
        marlin_n=64,
        note='compressed-tensors pack-quantized INT4, group 128, asymmetric; '
             'GPTQ-Marlin W4A16 on sm_120.  NOT expressible in modelopt_mixed.',
    ),
}

# =============================================================================
# 1b. THE CONTAINER CONSTRAINT
# =============================================================================
#
# A checkpoint has ONE quantization_config, so every algo in one recipe must be
# expressible in ONE container.  This is the hard limit on "add more quant types
# to a recipe": the ladder is not one ladder, it is two, and they overlap only
# at NVFP4 and BF16.
#
#   modelopt_mixed      NVFP4 4.500 | W4A16_NVFP4 4.500 | FP8 8.000 |
#                       FP8_PB_WO 8.002 | MXFP8 8.250 | (absent) = BF16
#                       ...and it is the ONLY container with an EMBEDDING
#                       method (ModelOptNvFp4EmbeddingMethod, :700).
#   compressed-tensors  NVFP4 4.500 | INT4 g128 4.15625 (and g64/g32) |
#                       INT8 g128 8.125 | (ignore-listed) = BF16
#                       ...and it has NO VocabParallelEmbedding branch at all,
#                       so its embedding is BF16 and that is a 2.54 GB floor.
ALGO_CONTAINER: Dict[str, Tuple[str, ...]] = {
    BF16:             ('modelopt_mixed', 'compressed-tensors'),
    'sgl_nvfp4':      ('modelopt_mixed', 'compressed-tensors'),
    'sgl_nvfp4a16':   ('modelopt_mixed',),
    'sgl_fp8':        ('modelopt_mixed',),
    'sgl_fp8_pb_wo':  ('modelopt_mixed',),
    'sgl_mxfp8':      ('modelopt_mixed',),
    'sgl_int4_g128':  ('compressed-tensors',),
}


def container_for(algos) -> Optional[str]:
    """The single container that can hold every algo in `algos`, or None.

    None is not "unknown", it is "this recipe cannot be written as one file".
    """
    common = None
    for a in algos:
        c = set(ALGO_CONTAINER.get(a, ()))
        common = c if common is None else (common & c)
    if not common:
        return None
    # modelopt_mixed first: it is the production container and the only one
    # that can quantise the embedding.
    for pref in ('modelopt_mixed', 'compressed-tensors'):
        if pref in common:
            return pref
    return sorted(common)[0]

# The pool, cheapest first.  This is the whole ladder an SGLang-native
# checkpoint can be built from, and its shape is itself a finding: there is
# NOTHING between 4.5 and 8.0 bpw.  The GGUF ladder has q5_K (5.5) and q6_K
# (6.5625) exactly there, and a quality-per-byte optimiser at this budget was
# measured to want precisely that band.  An SGLang mixture cannot
# offer it per-tensor; it can only interpolate it ACROSS tensors.
SGLANG_POOL_ORDER = ('sgl_int4_g128', 'sgl_nvfp4', 'sgl_nvfp4a16', 'sgl_fp8',
                     'sgl_fp8_pb_wo', 'sgl_mxfp8', BF16)


def algo_bytes(algo: str, k: int, n: int, count: int = 1) -> int:
    """Exact on-disk safetensors payload bytes for one [N, K] weight at `algo`.

    k = input/reduction dim (GGUF shape[0]); n = output dim (GGUF shape[1]).

    `count` is the number of INDEPENDENT modules the tensor stacks.  It is 1 for
    an ordinary weight and `n_experts` for a GGUF stacked-expert tensor such as
    `blk.N.ffn_gate_exps.weight`, whose shape is (K, N, E) and which HF/SGLang
    explode into E separate `mlp.experts.<e>.gate_proj` modules.  The distinction
    is NOT cosmetic: the per-element payload and the per-block scales scale with
    K*N*E either way, but the PER-TENSOR scalars (`weight_scale_2`, `input_scale`,
    compressed-tensors' `weight_shape`) are written once PER MODULE, so a stacked
    tensor carries E of them, not one.  On GLM-4.7 that is 90 layers x 3 x 160
    modules x 8 B = 345,600 B - small, but this model exists to be exact to the
    byte, and it is the difference between reproducing a published index and
    nearly reproducing it.
    """
    spec = SGLANG_ALGOS[algo]
    if count != 1:
        return count * algo_bytes(algo, k, n)
    if algo == BF16:
        return 2 * k * n
    if spec['quant_algo'] in ('NVFP4', 'W4A16_NVFP4'):
        gs = spec['group_size']
        return (k * n) // 2 + n * (k // gs) + spec['per_tensor_bytes']
    if spec['quant_algo'] == 'FP8':
        return k * n + spec['per_tensor_bytes']
    if spec['quant_algo'] == 'FP8_PB_WO':
        bn, bk = spec['block_n'], spec['block_k']
        return k * n + 4 * ((n + bn - 1) // bn) * ((k + bk - 1) // bk)
    if spec['quant_algo'] == 'MXFP8':
        bk = spec['block_k']
        return k * n + n * ((k + bk - 1) // bk)
    if spec['quant_algo'].startswith('INT'):
        # compressed-tensors pack-quantized, read off our written checkpoint:
        #   weight_packed I32 [N, K*bits/32] + weight_scale BF16 [N, K/g]
        #   (+ weight_zero_point I32 [N/8, K/g] when asymmetric)
        #   + weight_shape I64 [2]
        bits, g = spec['num_bits'], spec['group_size']
        b = k * n * bits // 8 + 2 * n * (k // g)
        if not spec['symmetric']:
            b += n * (k // g) * bits // 8
        return b + spec['per_tensor_bytes']
    raise KeyError(algo)


# Which algos an embedding can hold is a property of the arch, not of the format.
# `get_quant_method`'s VocabParallelEmbedding branch (:1035-1043) knows NVFP4 and
# nothing else, so NVFP4 is the ceiling - but that branch only runs when the model
# file hands the module a `quant_config`, and not every model file does.  Each
# ARCHS entry therefore declares its own `embedding_algos`; this is what an arch
# that declares none gets, i.e. exactly the rule this file shipped with.
EMBEDDING_ALGOS_DEFAULT = (BF16, 'sgl_nvfp4')


def embedding_algos(arch: Optional[dict] = None) -> Tuple[str, ...]:
    """The algos this arch's SGLang model file can load for its global embedding."""
    return tuple((arch or {}).get('embedding_algos') or EMBEDDING_ALGOS_DEFAULT)


def embedding_algos_for(name: Optional[str] = None,
                        arch: Optional[dict] = None) -> Tuple[str, ...]:
    """The algos this arch can load for the embedding tensor called `name`.

    The scope of the per-arch rule is the arch's global table alone - exactly
    what `embedding_tensors(arch)` returns.  A per-layer embedding, i.e. an MTP
    table (`blk.N.nextn.embed_tokens.weight`), belongs to the draft module
    (`glm4_moe_nextn.py`), which builds it without a `quant_config` too but
    which the engine instantiates only when speculative decoding is on.  So it
    keeps the format ceiling here - a checkpoint that never speculates loads it
    packed and serves - and it is `--nextn-optimization` that pins it to
    bf16, in `sglang_preset.preset_plan`, when the draft head has to work.  A
    caller that does not say which tensor it is asking about gets the arch rule,
    which is the safe answer; the size model and the writer always pass a name.
    """
    if arch and name is not None and name not in embedding_tensors(arch):
        return EMBEDDING_ALGOS_DEFAULT
    return embedding_algos(arch)


def algo_legal(algo: str, k: int, n: int, role: str,
               fused_shards: Optional[List[int]] = None,
               name: Optional[str] = None,
               arch: Optional[dict] = None) -> Tuple[bool, str]:
    """Is `algo` loadable for a tensor of this shape and role?

    These are RAISES in SGLang, not fallbacks - a checkpoint that violates one
    does not load slowly, it does not load.  So the assigner has to honour them
    up front, exactly as `--speed-profile blackwell` had to honour NVFP4's
    64-element GGUF row rule.
    """
    spec = SGLANG_ALGOS.get(algo)
    if spec is None:
        return False, f'unknown algo {algo!r}'
    if role not in spec['roles']:
        return False, f'{algo} is not defined for role {role!r}'
    km = spec.get('k_multiple')
    if km and (k % km) != 0:
        # NVFP4:  modelopt_quant.py:1741 "in features size is not multiple of 16"
        # MXFP8:  fp8.py block_k = 32
        return False, f'{algo} needs K %% {km} == 0, got K={k}'
    if spec['quant_algo'] == 'FP8_PB_WO':
        # validate_block_quant_shapes (fp8.py:512-533): under TP>1, or whenever
        # the linear is a MERGED (fused) one, every output partition must be
        # divisible by block_n and the input partition by block_k.
        bn, bk = spec['block_n'], spec['block_k']
        if (k % bk) != 0:
            return False, f'FP8_PB_WO needs K %% {bk} == 0, got K={k}'
        for part in (fused_shards or [n]):
            if (part % bn) != 0:
                return False, (f'FP8_PB_WO needs every fused output partition '
                               f'%% {bn} == 0, got {part}')
    mn = spec.get('marlin_n')
    if mn:
        # check_marlin_supports_shape (marlin_utils.py:178-211): the FUSED
        # output partition must be divisible by 64 and the input by 128.
        for part in ([sum(fused_shards)] if fused_shards else [n]):
            if (part % mn) != 0:
                return False, (f'{algo} needs the FUSED output partition '
                               f'% {mn} == 0, got N={part}')
        if (k % spec['group_size']) != 0:
            return False, f'{algo} needs K % {spec["group_size"]} == 0, got K={k}'
        if name and re.match(r'^output\.weight$|^lm_head', name) and not spec['symmetric']:
            # A quantised ParallelLMHead is loaded by
            # VocabParallelEmbedding.weight_loader, which asserts
            # shape[output_dim] == org_vocab_size for every parameter carrying an
            # output_dim (vocab_parallel_embedding.py:485-501).  weight_zero_point
            # has output_dim 0 and first dimension N/8, so an ASYMMETRIC WNA16 head
            # cannot load.  All three published CT checkpoints of this model leave
            # the head BF16 for exactly this reason.
            return False, f'{algo} is asymmetric and cannot be the lm_head'
    if name and re.match(r'^output\.weight$|^lm_head', name):
        # A quantised ParallelLMHead is loaded by VocabParallelEmbedding.weight_loader,
        # which asserts loaded_weight.shape[output_dim] == org_vocab_size for EVERY
        # parameter that carries an output_dim (vocab_parallel_embedding.py:485-501).
        # FP8_PB_WO's weight_scale_inv is [ceil(N/128), ceil(K/128)] -> its first
        # dimension is 1940, not the 248320-row vocabulary, so the head does not load:
        #   AssertionError: self.org_vocab_size=248320 ... loaded_weight.shape[0]=1940
        # MEASURED 2026-09-05: sgl-B2 was written with an FP8_PB_WO head and died on
        # GPU 3 with exactly that assert.  It is a RAISE at load time, not a fallback.
        # Per-tensor FP8, NVFP4, W4A16_NVFP4 and MXFP8 all keep a first dimension of N
        # and load fine.  (recipes/B shipped with FP8_PB_WO in its recipe and FP8 in
        # its BUILT config -- the earlier lane fixed the file by hand and the rule
        # never reached the tool.  This is that rule.)
        if spec['quant_algo'] == 'FP8_PB_WO':
            return False, ('FP8_PB_WO writes weight_scale_inv [ceil(N/128), ceil(K/128)]; '
                           'the lm_head loader asserts its first dim == vocab_size')
    if role == 'embedding':
        # Per arch, because the model file decides, and for the arch's global
        # table only.  get_quant_method's VocabParallelEmbedding branch
        # (:1035-1043) knows NVFP4 and nothing else; every other algo silently
        # returns None, i.e. the tensor would be read as BF16 and the file would
        # not match.  But that branch is only reached when the module was given a
        # quant_config: qwen3_5.py:1584-1591 passes one (the served sgl-F-floor
        # checkpoint carries model.language_model.embed_tokens.weight as U8
        # [248320, 2560] with weight_scale/weight_scale_2 and loads), while
        # glm4_moe.py:1029-1033 builds VocabParallelEmbedding without one, so the
        # parameter stays a plain BF16 tensor.  A packed embedding then dies in
        # vocab_parallel_embedding.py's weight_loader with "The size of tensor a
        # (5120) must match the size of tensor b (2560)" - the packed row is half
        # as wide.  Salyut1/GLM-4.7-NVFP4 keeps model.embed_tokens.weight in BF16
        # [151552, 5120] for exactly that reason, and measured 2026-09-07: a
        # checkpoint built from the 4.8505 bpw recipe, whose recipe line was
        # `^token_embd\.weight$=sgl_nvfp4`, failed to load with that message.
        #
        # A per-layer embedding is a different module and a different question.
        # glm4_moe_nextn.py:57-62 builds the MTP table without a quant_config
        # too, but that module exists only under speculative decoding, which is
        # what --nextn-optimization governs: with the flag on the preset pins
        # that table bf16 as well, and with it off nothing does, because the
        # module is never instantiated.  Measured 2026-09-07, the floor and mid
        # GLM checkpoints carry blk.92.nextn.embed_tokens in NVFP4 and serve.
        # So it keeps the format ceiling here - see embedding_algos_for().
        legal = embedding_algos_for(name, arch)
        if algo not in legal:
            if tuple(legal) == (BF16,):
                return False, (f'{algo}: this arch builds the embedding without a '
                               f'quant_config, so SGLang loads it as BF16 only')
            return False, f'{algo} has no embedding method (NVFP4 only)'
    return True, ''


# =============================================================================
# 2. THE NAME MAP AND THE FUSED-MODULE GROUPS
# =============================================================================
#
# GGUF names are fixed by gguf-py/gguf/tensor_mapping.py; HF names are what the
# safetensors index and `quantized_layers` use.  The pairs below were read from
# tensor_mapping.py (qwen3.5 entries) and cross-checked tensor-for-tensor
# against RadixArk/Qwen3.8-27B-NVFP4's own headers - every shape matches its
# transpose, all 851 GGUF tensors account for all 1,033 HF modules.
#
# `role`  : 'linear' (a GEMM weight the mixed config can name)
#           'embedding' (a gather; BF16, plus NVFP4 - for the global table, on
#                        an arch whose SGLang model file passes a quant_config;
#                        `embedding_algos_for`)
#           'other' (norms, biases, conv1d, A_log, dt_bias - always BF16)
# `fused` : the packed_modules_mapping key this tensor is a SHARD of, or None.
#           Shards of the same fused module on the same layer MUST agree
#           (modelopt_quant.py:967).

ARCH_QWEN3_5 = {
    'hf_layer_prefix': 'model.language_model.layers.{bid}.',
    # HOW A SOURCE NAMES THIS ARCHITECTURE, so `--arch` can stop being typed.
    # `hf_architectures` is config.json's own `architectures` list and is the
    # exact answer; `hf_model_types` is `model_type`, which the Moe variant of
    # this family shares, so it is only ever a fallback and the tensor names
    # decide; `gguf_arch` is `general.architecture` in a GGUF split.  Only the
    # spellings this table actually covers are listed - a family member whose
    # tensors this table does not name must not be silently claimed by it.
    'hf_architectures': ('Qwen3_5ForConditionalGeneration', 'Qwen3_5ForCausalLM'),
    'hf_model_types': ('qwen3_5', 'qwen3_5_text'),
    'gguf_arch': ('qwen35',),
    'tensors': {
        # GGUF suffix                 HF suffix                      role        fused
        'attn_qkv.weight':           ('linear_attn.in_proj_qkv',    'linear',   'in_proj_qkvz'),
        'attn_gate.weight':          ('linear_attn.in_proj_z',      'linear',   'in_proj_qkvz'),
        'ssm_out.weight':            ('linear_attn.out_proj',       'linear',   None),
        'ssm_alpha.weight':          ('linear_attn.in_proj_a',      'linear',   'in_proj_ba'),
        'ssm_beta.weight':           ('linear_attn.in_proj_b',      'linear',   'in_proj_ba'),
        'ssm_norm.weight':           ('linear_attn.norm',           'other',    None),
        'ssm_conv1d.weight':         ('linear_attn.conv1d',         'other',    None),
        'ssm_a':                     ('linear_attn.A_log',          'other',    None),
        'ssm_dt.bias':               ('linear_attn.dt_bias',        'other',    None),
        'attn_q.weight':             ('self_attn.q_proj',           'linear',   'qkv_proj'),
        'attn_k.weight':             ('self_attn.k_proj',           'linear',   'qkv_proj'),
        'attn_v.weight':             ('self_attn.v_proj',           'linear',   'qkv_proj'),
        'attn_output.weight':        ('self_attn.o_proj',           'linear',   None),
        'attn_q_norm.weight':        ('self_attn.q_norm',           'other',    None),
        'attn_k_norm.weight':        ('self_attn.k_norm',           'other',    None),
        'attn_norm.weight':          ('input_layernorm',            'other',    None),
        'post_attention_norm.weight':('post_attention_layernorm',   'other',    None),
        'ffn_gate.weight':           ('mlp.gate_proj',              'linear',   'gate_up_proj'),
        'ffn_up.weight':             ('mlp.up_proj',                'linear',   'gate_up_proj'),
        'ffn_down.weight':           ('mlp.down_proj',              'linear',   None),
    },
    'globals': {
        'token_embd.weight':  ('model.language_model.embed_tokens', 'embedding', None),
        'output.weight':      ('lm_head',                           'linear',    None),
        'output_norm.weight': ('model.language_model.norm',         'other',     None),
    },
    # The embedding can be packed here.  qwen3_5.py:1584-1591 passes
    # `quant_config=quant_config` to VocabParallelEmbedding, so
    # get_quant_method's NVFP4 branch runs: the served sgl-F-floor checkpoint
    # carries model.language_model.embed_tokens.weight as U8 [248320, 2560] with
    # weight_scale and weight_scale_2, and serves.
    'embedding_algos': (BF16, 'sgl_nvfp4'),
}

# -----------------------------------------------------------------------------
# GLM-4.7 (zai-org/GLM-4.7, `glm4_moe`, Glm4MoeForCausalLM) - and, by the same
# table, every GLM-4.5/4.6/4.7-family MoE the suite has calibration for.
#
# THREE THINGS ARE DIFFERENT FROM A DENSE MODEL and all three are expressed as
# data here rather than as branches in the algorithms:
#
#  1. STACKED EXPERTS.  `blk.N.ffn_{gate,up,down}_exps.weight` each hold ALL 160
#     routed experts in one GGUF tensor of shape (K, N, 160).  The stack count
#     is read off the shape (plan_tensor), so nothing here names 160.
#  2. ONE FusedMoE UNIT PER LAYER.  SGLang's FusedMoE owns `w13_weight`
#     (gate+up, every expert) AND `w2_weight` (down, every expert) as ONE
#     module, so all three GGUF tensors of a layer are ONE decision.  They
#     therefore share the fused key `experts` and the same HF name,
#     `...mlp.experts` - which is also the fix for the silent-first-entry trap
#     in `_resolve_quant_algo`'s prefix fallback (modelopt_quant.py:971-976):
#     with ONE key per FusedMoE there is no set of per-expert keys for it to
#     pick from arbitrarily.
#  3. NO `packed_modules_mapping` ON THE MODEL CLASS.  `Glm4MoeForCausalLM`
#     declares none (unlike glm5_next.py:1081 and qwen3_next.py:995), and
#     `model_loader/loader.py:172` reads it off the class - so the fused-shard
#     walk never fires and a `quantized_layers` map keyed on the UNFUSED HF
#     names (`q_proj`, `gate_proj`, ...) resolves to `None` for every attention
#     and dense-MLP module, silently loading them unquantised.  The HF names
#     below are therefore the SGLang FUSED names (`self_attn.qkv_proj`,
#     `mlp.gate_up_proj`), which need no engine change.
#
# The MTP/nextn layer (`blk.92.*`) is mapped for completeness; every published
# GLM-4.7 NVFP4 checkpoint drops it, and `verify_against` skips what the
# reference does not carry.
ARCH_GLM4_MOE = {
    'hf_layer_prefix': 'model.layers.{bid}.',
    # the vision variant's class name is written in two pieces: a 32-letter run
    # reads as a key to the publish gate, and a class name is not one.
    'hf_architectures': ('Glm4MoeForCausalLM', 'Glm4vMoeFor' 'ConditionalGeneration'),
    'hf_model_types': ('glm4_moe', 'glm4v_moe'),
    'gguf_arch': ('glm4moe',),
    'tensors': {
        # GGUF suffix                  HF suffix                          role         fused
        'attn_q.weight':              ('self_attn.qkv_proj',              'linear',    'qkv_proj'),
        'attn_k.weight':              ('self_attn.qkv_proj',              'linear',    'qkv_proj'),
        'attn_v.weight':              ('self_attn.qkv_proj',              'linear',    'qkv_proj'),
        'attn_q.bias':                ('self_attn.qkv_proj',              'other',     None),
        'attn_k.bias':                ('self_attn.qkv_proj',              'other',     None),
        'attn_v.bias':                ('self_attn.qkv_proj',              'other',     None),
        'attn_output.weight':         ('self_attn.o_proj',                'linear',    None),
        'attn_q_norm.weight':         ('self_attn.q_norm',                'other',     None),
        'attn_k_norm.weight':         ('self_attn.k_norm',                'other',     None),
        'attn_norm.weight':           ('input_layernorm',                 'other',     None),
        'post_attention_norm.weight': ('post_attention_layernorm',        'other',     None),
        # dense MLP - layers 0..first_k_dense_replace-1 only
        'ffn_gate.weight':            ('mlp.gate_up_proj',                'linear',    'gate_up_proj'),
        'ffn_up.weight':              ('mlp.gate_up_proj',                'linear',    'gate_up_proj'),
        'ffn_down.weight':            ('mlp.down_proj',                   'linear',    None),
        # routed experts - one FusedMoE unit per layer (see 2. above)
        'ffn_gate_exps.weight':       ('mlp.experts',                     'linear',    'experts'),
        'ffn_up_exps.weight':         ('mlp.experts',                     'linear',    'experts'),
        'ffn_down_exps.weight':       ('mlp.experts',                     'linear',    'experts'),
        # shared expert - a normal dense MLP, fused the same way
        'ffn_gate_shexp.weight':      ('mlp.shared_experts.gate_up_proj', 'linear',    'shexp_gate_up'),
        'ffn_up_shexp.weight':        ('mlp.shared_experts.gate_up_proj', 'linear',    'shexp_gate_up'),
        'ffn_down_shexp.weight':      ('mlp.shared_experts.down_proj',    'linear',    None),
        # router: tiny, and quantising it changes which experts fire
        'ffn_gate_inp.weight':        ('mlp.gate',                        'other',     None),
        'exp_probs_b.bias':           ('mlp.gate.e_score_correction_bias','other',     None),
        # MTP / nextn head (blk.92 on this model)
        'nextn.embed_tokens.weight':  ('embed_tokens',                    'embedding', None),
        'nextn.eh_proj.weight':       ('eh_proj',                         'linear',    None),
        'nextn.enorm.weight':         ('enorm',                           'other',     None),
        'nextn.hnorm.weight':         ('hnorm',                           'other',     None),
        'nextn.shared_head_norm.weight': ('shared_head.norm',             'other',     None),
        'nextn.shared_head_head.weight': ('shared_head.head',             'linear',    None),
    },
    'globals': {
        'token_embd.weight':  ('model.embed_tokens', 'embedding', None),
        'output.weight':      ('lm_head',            'linear',    None),
        'output_norm.weight': ('model.norm',         'other',     None),
    },
    # No quantised global embedding on this arch, and it is the model file that
    # says so.  glm4_moe.py:1029-1033 builds `self.embed_tokens =
    # VocabParallelEmbedding(config.vocab_size, config.hidden_size,
    # use_attn_tp_group=...)` with no quant_config, so the parameter is a plain
    # BF16 tensor and a packed NVFP4 embedding fails in
    # vocab_parallel_embedding.py's weight_loader with "The size of tensor a
    # (5120) must match the size of tensor b (2560)".  The published checkpoint
    # agrees: Salyut1/GLM-4.7-NVFP4 keeps model.embed_tokens.weight in BF16
    # [151552, 5120] while quantising everything around it.
    #
    # This declaration covers `token_embd.weight` and nothing else.  The MTP
    # table `blk.92.nextn.embed_tokens.weight` is built the same way by
    # glm4_moe_nextn.py:57-62, but only when speculative decoding is on, so it
    # is --nextn-optimization that pins it - bf16 when that flag is on, free
    # when it is off - and it stays free here: the measured floor and mid
    # checkpoints carry it in NVFP4 and serve without speculation.
    'embedding_algos': (BF16,),
    # GGUF suffixes the HF checkpoint keeps in F32 rather than BF16.  Verified
    # against Salyut1/GLM-4.7-NVFP4: exactly 89 F32 tensors larger than a
    # scalar, all of them `mlp.gate.e_score_correction_bias` [160].
    'f32_tensors': {'exp_probs_b.bias'},
    # THE SHARED-EXPERT `weight_block_size` TRAP, expressed as data.
    #
    # `glm4_moe.py:485-505`: when `shared_experts.gate_up_proj.weight.dtype ==
    # float8_e4m3fn`, the model code asserts
    #     gate_up_proj.quant_method.quant_config.weight_block_size
    #  == down_proj.quant_method.quant_config.weight_block_size
    # `ffn_down_shexp` is fused with nothing - SGLang's `down_proj` IS its own
    # module, so its `fused` key above is correctly None and `fused_groups()`,
    # which is what the WRITER enforces, must not change.  But the ASSIGNER is
    # free to put `gate_up` on FP8_PB_WO and `down` on NVFP4, and that pair
    # raises.  Exercised on CPU against the real `Glm4MoeMLP` on the meta
    # device:
    #     NVFP4 / NVFP4          -> not an fp8 dtype, assert skipped
    #     FP8 / FP8, FP8 / NVFP4 -> passes (None == None)
    #     FP8_PB_WO / NVFP4      -> AssertionError: [128,128] != None
    #     FP8_PB_WO / FP8_PB_WO  -> passes
    # So the pair must agree whenever `sgl_fp8_pb_wo` is in the pool.  The fix
    # belongs HERE rather than in a smaller pool: harmonisation is exactly the
    # machinery that makes two tensors take one qtype, and dropping
    # `sgl_fp8_pb_wo` to dodge the trap would cost every GLM recipe the 8-bit
    # block format on every tensor in the model.
    #
    # `harmonize_join` maps a GGUF suffix onto the harmonize family it must join
    # even though it is not a shard of that fused module.  It is read ONLY by
    # `harmonize_argument()`; `fused_groups()`, `map_tensor()` and the writer are
    # deliberately untouched, because this constraint is a property of the MODEL
    # CODE and not of the checkpoint format.
    'harmonize_join': {'ffn_down_shexp.weight': 'shexp_gate_up'},
    # THE DISK NAMES.  `tensors` above gives the name `quantized_layers` must be
    # keyed on - the FUSED module, because `Glm4MoeForCausalLM` declares no
    # `packed_modules_mapping`.  The safetensors NEVER carry that name.  Both the
    # BF16 source (zai-org/GLM-4.7, 44,691 tensors) and every published NVFP4
    # checkpoint (Salyut1/GLM-4.7-NVFP4, 174,465) store the UNFUSED shards, and
    # one tensor PER EXPERT.  That is the HF convention rather than a choice, so
    # the two conventions have to meet somewhere; they meet here, as data.
    #
    # Each entry is (disk suffix, SCALE GROUP key).  A scale group is the set of
    # shards SGLang collapses into ONE `weight_scale_2` and ONE `input_scale`:
    #   qkv_proj      q_proj + k_proj + v_proj            (QKVParallelLinear)
    #   gate_up_proj  gate_proj + up_proj                 (MergedColumnParallelLinear)
    #   w13           gate_proj + up_proj OF ONE EXPERT   (FusedMoE w13_*)
    #   (none)        down_proj, o_proj, ...              their own module
    # The FusedMoE split is the one that is not obvious, and it was MEASURED
    # rather than assumed: `w13_weight_scale_2` is [num_experts, 2] and
    # `w2_weight_scale_2` is [num_experts] (modelopt_quant.py:2437-2450), and in
    # Salyut1 every expert's gate_proj and up_proj carry the identical
    # weight_scale_2 and input_scale while its down_proj carries its own.  So the
    # group is PER EXPERT - one fused amax over all 160 would be wrong.
    #
    # `{eid}` is matched against whatever expert ids the checkpoint carries;
    # nothing here names 160.
    'hf_disk': {
        'attn_q.weight':         ('self_attn.q_proj',               'qkv_proj'),
        'attn_k.weight':         ('self_attn.k_proj',               'qkv_proj'),
        'attn_v.weight':         ('self_attn.v_proj',               'qkv_proj'),
        'attn_q.bias':           ('self_attn.q_proj',               None),
        'attn_k.bias':           ('self_attn.k_proj',               None),
        'attn_v.bias':           ('self_attn.v_proj',               None),
        'ffn_gate.weight':       ('mlp.gate_proj',                  'gate_up_proj'),
        'ffn_up.weight':         ('mlp.up_proj',                    'gate_up_proj'),
        'ffn_gate_exps.weight':  ('mlp.experts.{eid}.gate_proj',    'w13'),
        'ffn_up_exps.weight':    ('mlp.experts.{eid}.up_proj',      'w13'),
        'ffn_down_exps.weight':  ('mlp.experts.{eid}.down_proj',    None),
        'ffn_gate_shexp.weight': ('mlp.shared_experts.gate_proj',   'gate_up_proj'),
        'ffn_up_shexp.weight':   ('mlp.shared_experts.up_proj',     'gate_up_proj'),
    },
}

ARCHS = {'qwen3_5': ARCH_QWEN3_5, 'glm4_moe': ARCH_GLM4_MOE}

# Tensors that are LEGAL at every algo but should be pre-assigned BF16 anyway,
# with the reason.  These are advisories, NOT size-model rules: `plan_tensor`
# and the tensors.<algo>.map it writes keep pricing them at the algo they are
# asked about, which is what lets that model reproduce a published checkpoint's
# byte count exactly.  What honours a pin is what CHOOSES a type:
# `sglang_preset.py` emits them as a `--gpu-assign-tensors` argument so the
# choice is visible on the command line rather than buried in a table, and
# `sglang_write.py --split` asks `pinned_bf16()` about every tensor it is about
# to write, so the uniform repository holds them at BF16 as well - a repository
# is a pool a recipe draws single tensors out of, so a file in it the model
# code cannot consume is a trap laid for whoever writes the next recipe by
# hand.  One table, read two ways, and not a second spelling of it anywhere.
#
# in_proj_ba (GGUF ssm_alpha / ssm_beta).  Three independent reasons and they
# agree:
#   - qwen3_5.py:392-402 gates its fused AMX path on
#     `in_proj_ba._parameters["weight"].dtype == torch.bfloat16`, and :649-651
#     documents the fused-kernel contract as "FP8 in_proj_qkvz takes (fp8,
#     scale) ... the bf16 in_proj_ba consumes the unquantized bf16" - i.e. the
#     model code expects qkvz to be quantisable and ba not to be.
#   - both published checkpoints leave in_proj_a / in_proj_b in BF16.
#   - it is 96 tensors x 245,760 elements = 0.23 % of the byte budget and
#     0.09 % of the matmul FLOPs, so nothing is being given up.
PIN_BF16_ADVISORY = {
    'glm4_moe': [
        (r'^blk\.\d+\.ffn_gate_inp\.weight$',
         'router: 819,200 elements per layer (0.0002 % of bytes) and it decides '
         'WHICH experts fire - a rounding error here is a routing error, not a '
         'degradation'),
        (r'^blk\.\d+\.exp_probs_b\.bias$',
         'router bias: 160 elements; same reason as the router itself'),
        (r'^output\.weight$',
         'lm_head: measured on GLM-4.7, an NVFP4 head costs +0.5 % perplexity '
         'and +0.005 KLD in the engine at the floor size (paired against the '
         'same build with the head in BF16: t = 11 and 6) for 1.1 GB saved, '
         '0.5 % of the model; the head reads the final hidden state of a '
         '151k-token vocabulary and GLM is not forgiving there, where '
         'Qwen3.8-27B is'),
    ],
    'qwen3_5': [
        (r'^blk\.\d+\.ssm_(alpha|beta)\.weight$',
         'in_proj_ba: model code expects BF16 (qwen3_5.py:392-402, :649-651); '
         'both published checkpoints agree; 0.23 % of bytes, 0.09 % of matmul FLOPs'),
    ],
}

_PINNED = {k: [(re.compile(pat), why) for pat, why in v]
           for k, v in PIN_BF16_ADVISORY.items()}


def arch_key_of(arch) -> Optional[str]:
    """The key `ARCHS` registers an arch table under, or None for a stranger."""
    return next((k for k, v in ARCHS.items() if v is arch), None)


def pinned_bf16(name: str, arch=ARCH_QWEN3_5) -> Optional[str]:
    """Why `PIN_BF16_ADVISORY` holds this GGUF tensor at BF16, or None.

    The one place a NAME is matched against that table - `sglang_preset.py`
    hands the patterns to the command line unmatched - so a pin added there
    reaches the recipe and the uniform split in the same edit and cannot come
    to mean two different things in the two.  Nothing is pinned for an arch
    table `ARCHS` does not register: a caller that built its own table is not
    one this advisory was measured on.
    """
    for rx, why in _PINNED.get(arch_key_of(arch) or '', ()):
        if rx.match(name):
            return why
    return None


_BLK_RE = re.compile(r'^blk\.(\d+)\.(.+)$')


def _gguf_suffix(name: str) -> str:
    """`blk.42.attn_q.bias` -> `attn_q.bias`; a global name is its own suffix."""
    m = _BLK_RE.match(name)
    return m.group(2) if m else name


def map_tensor(gguf_name: str, arch: dict = ARCH_QWEN3_5):
    """GGUF tensor name -> (hf_prefix, role, fused_key, layer_id or None).

    Returns None for a name the arch does not know, which the caller must treat
    as an error rather than a BF16 default: a tensor we cannot name is a tensor
    we cannot write into `quantized_layers`, and silently leaving it BF16 would
    blow the byte budget in a run that reports success.
    """
    g = arch['globals'].get(gguf_name)
    if g is not None:
        return g[0], g[1], g[2], None
    m = _BLK_RE.match(gguf_name)
    if not m:
        return None
    bid, suffix = m.group(1), m.group(2)
    ent = arch['tensors'].get(suffix)
    if ent is None:
        return None
    hf_suffix, role, fused = ent
    prefix = arch['hf_layer_prefix'].format(bid=bid) + hf_suffix
    return prefix, role, fused, int(bid)


def fused_groups(gguf_names, arch: dict = ARCH_QWEN3_5) -> Dict[str, List[str]]:
    """{group id -> [gguf tensor names]} for every multi-shard fused module.

    THE CONSTRAINT.  `ModelOptMixedPrecisionConfig._resolve_quant_algo`
    (modelopt_quant.py:952-969) looks the layer up by its FUSED name; when that
    misses it walks `packed_modules_mapping[proj_name]`, collects the algo of
    every shard, and if `len(algos) > 1` raises

        ValueError: Mixed quant_algo within fused layer {prefix}: {algos}.
                    All shards must use the same quantization.

    So the granularity of an SGLang mixed checkpoint is the fused module, not
    the tensor.  Single-shard groups are omitted: they are unconstrained.
    """
    groups: Dict[str, List[str]] = {}
    for name in gguf_names:
        info = map_tensor(name, arch)
        if info is None:
            continue
        _, _, fused, bid = info
        if fused is None:
            continue
        groups.setdefault(f'{fused}.{bid}', []).append(name)
    return {g: sorted(v) for g, v in groups.items() if len(v) > 1}


class DiskNames:
    """The map between the MODULE names a mixed config is keyed on and the
    TENSOR names a safetensors checkpoint actually carries.

    THE PROBLEM THIS EXISTS FOR.  `arch['tensors']` gives the module name, which
    on a model class that declares no `packed_modules_mapping` must be the FUSED
    one (`self_attn.qkv_proj`, `mlp.experts`) or `_resolve_quant_algo` returns
    None and the module loads unquantised.  No checkpoint stores that name.  HF
    stores the SHARDS (`self_attn.q_proj`) and, for a FusedMoE, one tensor per
    expert (`mlp.experts.7.gate_proj`) - the BF16 source and RadixArk's NVFP4
    checkpoint agree on that, and SGLang's own weight loaders are written to
    consume exactly it.  Every read and every write therefore happens under the
    disk name; only `quantized_layers` speaks module names.

    On an arch with no `hf_disk` the two coincide, the scale group is the arch's
    own fused key, and this class is the identity - qwen3_5 goes through it
    unchanged.

    A SCALE GROUP is the set of disk tensors SGLang collapses into one
    `weight_scale_2` / `input_scale`.  `classify()` returns its id as
    `(parent, key)`, and the parent is simply the disk name minus its last
    component, which separates the dense MLP from the shared expert from expert 7
    without naming any of them.
    """

    def __init__(self, arch):
        self.arch = arch
        head, tail = arch['hf_layer_prefix'].split('{bid}')
        self._head, self._tail = head, tail
        disk = arch.get('hf_disk', {})
        self._rules = []        # (regex, hf module suffix or None, group key)
        self._leaves = {}       # group key -> [disk leaf names, in shard order]
        for gguf, (hf, role, fused) in arch['tensors'].items():
            dname, gkey = disk.get(gguf, (hf, fused))
            if role not in ('linear', 'embedding'):
                continue
            pat = (re.escape(head) + r'(\d+)' + re.escape(tail)
                   + re.escape(dname).replace(re.escape('{eid}'), r'\d+') + '$')
            self._rules.append((re.compile('^' + pat), hf, gkey))
            if gkey is not None:
                self._leaves.setdefault(gkey, [])
                leaf = dname.rsplit('.', 1)[-1]
                if leaf not in self._leaves[gkey]:
                    self._leaves[gkey].append(leaf)
        for gguf, (hf, role, fused) in arch['globals'].items():
            if role not in ('linear', 'embedding'):
                continue
            self._rules.append((re.compile('^' + re.escape(hf) + '$'), None, fused))

    def classify(self, disk_prefix):
        """(module name, group id or None) for a physical prefix, else None."""
        for rx, hf, gkey in self._rules:
            m = rx.match(disk_prefix)
            if not m:
                continue
            module = (self._head + m.group(1) + self._tail + hf) if hf else disk_prefix
            gid = (disk_prefix.rsplit('.', 1)[0], gkey) if gkey else None
            return module, gid
        return None

    def group_members(self, gid):
        """Every disk prefix a scale group is supposed to contain."""
        parent, key = gid
        return [parent + '.' + leaf for leaf in self._leaves[key]]

    def mtp_prefixes(self, names):
        """Name prefixes that belong to the MTP / NEXTN head, DERIVED.

        The head is whichever layer carries an MTP-only tensor (`nextn.*` in
        GGUF), plus - where an arch stores it outside the layer stack - anything
        under `mtp.`.  On GLM-4.7 that resolves to the whole of
        `model.layers.92`, its 160 experts included; on qwen3_5 to `mtp.`.
        Nothing is hard-coded per model, and a source without an MTP head yields
        no layer prefix at all.
        """
        out = {'mtp.'}
        disk = self.arch.get('hf_disk', {})
        for gguf, (hf, _role, _f) in self.arch['tensors'].items():
            if not gguf.startswith('nextn.'):
                continue
            dname = disk.get(gguf, (hf, None))[0]
            rx = re.compile('^' + re.escape(self._head) + r'(\d+)'
                            + re.escape(self._tail)
                            + re.escape(dname).replace(re.escape('{eid}'), r'\d+')
                            + r'\.')
            for n in names:
                m = rx.match(n)
                if m:
                    out.add(self._head + m.group(1) + self._tail)
        return tuple(sorted(out))


# The vision tower, when a checkpoint has one, is not language-model bytes
# either.  It has no ARCHS entry because nothing in this lane quantises it.
NON_LM_PREFIXES = ('model.visual.',)


# The two constants the ModelOpt NVFP4 scale construction is built from.
# fp8.py:210-211 - "max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)".
E2M1_MAX = 6.0
E4M3_MAX = 448.0


def nvfp4_global_scale(shard_amaxes) -> float:
    """weight_scale_2 for one NVFP4 module: max(amax over ALL its fused shards).

    THE SECOND FUSED-MODULE CONSTRAINT, and the one nobody writes down.  It is
    not enough for the shards of a fused module to agree on `quant_algo`; for
    NVFP4 they must also agree on `weight_scale_2`, because SGLang collapses
    them:

        modelopt_quant.py:1807   weight_scale_2 = layer.weight_scale_2.max()
        modelopt_quant.py:1829   alpha = input_scale_2 * weight_scale_2

    - ONE alpha for the whole fused GEMM.  A shard written with a smaller ws2
    than its siblings is then dequantised with the larger one, so its weights
    come out scaled up by the ratio.  That is a gross error, not a degradation.
    The W4A16 path at least warns about it (":2188 weight_scale_2 differs across
    fused parallel layers"); the W4A4 path takes the max silently.

    This was established by measurement, not by reading: RadixArk's
    `layers.30.mlp.gate_proj` and `layers.30.mlp.up_proj` carry the identical
    weight_scale_2 = 1.213437063e-04, which is up_proj's own amax / (6 * 448);
    gate_proj's own amax would give 7.011776879e-05.  `mlp.down_proj`, fused
    with nothing, uses its own.  With this rule applied,
    `fp4_rtn_metric.py --verify-modelopt` reproduces RadixArk's packed codes and
    e4m3 block scales at 100.0000 % on every tensor tested.

    It costs essentially nothing: on blk.30.ffn_gate it moves rel_rmse from
    0.094889 to 0.094905, i.e. 0.017 %.
    """
    return float(max(shard_amaxes)) / (E2M1_MAX * E4M3_MAX)


def harmonize_argument(gguf_names, arch: dict = ARCH_QWEN3_5) -> List[str]:
    """The `--harmonize-tensors` value that expresses the fused constraint.

    `quant_assign.py`'s harmonisation is EXACTLY the right machinery for this
    and it already exists: with `--use-auto-quant-assign`, `auto_quant_assign`
    (quant_assign.py:2330-2365) merges each expanded group into one virtual
    entity, intersects the members' allowed qtype lists, SUMS their sizes and
    losses, assigns ONE qtype to the group and expands it back.  That is a hard
    constraint, not a hint.  (`harmonize_row`, quant_assign.py:5590+, separately
    equalises the members' loss values with min/mean/max - that is the softer,
    older half of the feature and is orthogonal.)

    Its pairing rule is index-wise per `blk.<ID>` and it REQUIRES every pattern
    in a group to match the same number of tensors, which is why the groups are
    emitted per fused-module family rather than as one big list.

    A family is the fused module PLUS whatever `arch['harmonize_join']` attaches
    to it.  The fused key answers "which shards does the CHECKPOINT FORMAT force
    to agree"; a join answers "which tensors does the MODEL CODE force to agree",
    which on GLM is the shared expert's `down_proj` (see ARCH_GLM4_MOE).  The two
    questions have different answers, so they are two tables and only this
    function reads the second one.
    """
    fams: Dict[str, set] = {}
    join = arch.get('harmonize_join') or {}
    for name in gguf_names:
        info = map_tensor(name, arch)
        if info is None:
            continue
        _, _, fused, bid = info
        suffix = _BLK_RE.match(name).group(2) if _BLK_RE.match(name) else name
        fused = join.get(suffix, fused)
        if fused is None:
            continue
        fams.setdefault(fused, set()).add(suffix)
    out = []
    for fused, suffixes in sorted(fams.items()):
        if len(suffixes) < 2:
            continue
        pats = [r'^blk\.\d+\.' + re.escape(s) + '$' for s in sorted(suffixes)]
        out.append(','.join(pats))
    return out


# =============================================================================
# 3. THE SIZE MODEL - synthesising tensors.<algo>.map
# =============================================================================
#
# HOW THE SIZE MODEL IS DELIVERED, and why this way.
#
# `quant_assign.py` gets every byte figure from `tensors.<qtype>.map`, one line
# per tensor:
#     <shard>:<sha256>:<name>:shape=(K, N):dtype=<t>:elements=<E>:bytes=<B>
# parse_map_file() reads `bytes=` verbatim, so a map is a complete, exact size
# model with no formula in the assigner at all.  There are three ways to give it
# an SGLang size model and only one of them is right:
#
#   (a) bypass the maps and special-case SGLang inside the assigner - rejected:
#       it would put a second size model in the one place that currently has
#       none, and every engine (auto, greedy, spread) reads the maps.
#   (b) publish real SGLang shard sets and download them - impossible: there are
#       none, and building one per algo is the thing we are trying to avoid.
#   (c) SYNTHESISE the maps from `tensors.bf16.map`, which is exactly what
#       `--compute-missing-map` already does for any qtype without a published
#       shard set (it shells out to convert_map_qtype.py).  <- this one.
#
# So: one synthesised map per algo, written into the same directory the
# downloader uses, and the assigner cannot tell the difference.  The `!` prefix
# `quant_assign.py` puts on a computed qtype in the recipe keeps the provenance
# visible.
#
# TWO THINGS THE SYNTHESISER MUST DO THAT convert_map_qtype.py's GGUF PATH DOES
# NOT:
#
#   1. F32 -> BF16.  A GGUF keeps norms, biases, `ssm_a`, `ssm_dt.bias` and
#      `ssm_conv1d` at F32 (4 B/elem).  An SGLang checkpoint keeps them at BF16
#      (2 B/elem) - RadixArk has 2,645,504 such elements at 5,291,008 B, exactly
#      half the GGUF figure.  Getting this wrong is a 5.29 MB error on a
#      20.15 GB target: small, but it is 0.026 % of a budget we are matching to
#      the byte.
#   2. TRANSPOSE-AWARE LEGALITY.  GGUF's NVFP4 rule is on `ne[0] % 64`; SGLang's
#      is on the INPUT dim % 16, and GGUF `shape=(K, N)` puts K first while
#      safetensors puts N first.  Same tensor, different predicate.

def _parse_shape(text: str) -> List[int]:
    return [int(x) for x in re.findall(r'-?\d+', text or '')]


def read_bf16_map(path: str):
    """Parse a tensors.bf16.map into [(name, shape, elements, dtype)]."""
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip('\n').split(':')
            if len(parts) < 5:
                continue
            name = parts[2]
            shape, dtype, elems = None, None, None
            for p in parts[3:]:
                if p.startswith('shape='):
                    shape = _parse_shape(p.split('=', 1)[1])
                elif p.startswith('dtype='):
                    dtype = p.split('=', 1)[1]
                elif p.startswith('elements='):
                    elems = int(p.split('=', 1)[1])
            if name and shape is not None and elems is not None:
                out.append((name, shape, elems, dtype, parts[0], parts[1]))
    return out


def plan_tensor(name, shape, elems, algo, arch=ARCH_QWEN3_5,
                fused_shard_sizes=None):
    """(effective_algo, bytes, reason) for one tensor at a requested algo.

    Falls back to BF16 - and SAYS SO in the returned dtype, exactly as a GGUF
    map records a `tensor_type_fallback` - when the tensor is not a linear
    weight or the algo is not legal for its shape.
    """
    info = map_tensor(name, arch)
    if info is None:
        raise KeyError(f'{name!r} is not in the arch name map; refusing to guess')
    _prefix, role, _fused, _bid = info
    if role == 'other' or len(shape) < 2:
        # WIDTH OF A NON-MATMUL TENSOR.  BF16 is right for almost every one of
        # them, and the GGUF map is NOT the authority here: llama.cpp's
        # converter promotes norms and biases to F32 as a matter of convention,
        # while the HF checkpoint those tensors came from keeps them BF16 (GLM-
        # 4.7: `input_layernorm` and `q_proj.bias` are BF16 in
        # Salyut1/GLM-4.7-NVFP4 but F32 in the GGUF map).  So the source dtype
        # cannot be copied through, and the exceptions cannot be inferred - they
        # are a property of the upstream model, not of the tensor's shape, name
        # or role.  They are therefore DECLARED, per arch, as `f32_tensors`:
        # a set of GGUF suffixes the HF checkpoint keeps in F32.  Absent = BF16,
        # so an arch that declares nothing behaves exactly as before.
        w = 4 if _gguf_suffix(name) in arch.get('f32_tensors', ()) else 2
        return BF16, w * elems, ('non-matmul tensor: %s in every SGLang checkpoint'
                                 % ('F32' if w == 4 else 'BF16'))
    k, n = shape[0], shape[1]
    # A THIRD DIMENSION MEANS STACKED EXPERTS, in every model that has them.
    # GGUF packs a MoE layer's routed experts into one tensor of shape
    # (K, N, E) - `blk.N.ffn_gate_exps.weight` - where HF/SGLang carry E
    # separate modules.  Reading E off the shape keeps this generic: no expert
    # count, no per-model table and no tensor-name list is involved, so a model
    # the arch table has never seen still gets its per-module scalars counted
    # correctly.  (`elems` already covers K*N*E, so only the scalars change.)
    count = int(shape[2]) if len(shape) > 2 else 1
    ok, why = algo_legal(algo, k, n, role, fused_shard_sizes, name=name,
                         arch=arch)
    if not ok:
        return BF16, 2 * elems, why
    nbytes = algo_bytes(algo, k, n, count)
    if role == 'embedding':
        # A gather has no activation to quantise, so no `input_scale` is
        # registered for it.  Worth exactly 4 B on the whole checkpoint - and
        # worth getting right, because it is the difference between the
        # assigner PREDICTING a byte count and the writer REPRODUCING it.
        nbytes -= count * int(SGLANG_ALGOS[algo].get('act_scale_bytes', 0))
    return algo, nbytes, ''


def synthesise_map_lines(rows, algo: str, arch=ARCH_QWEN3_5):
    """(lines, census) for tensors.<algo>.map, from parsed bf16-map rows.

    Kept separate from synthesise_map() so convert_map_qtype.py can splice the
    lines into its own writer and keep one output-path implementation.
    """
    fused = fused_groups([r[0] for r in rows], arch)
    shard_sizes = {}
    dims = {r[0]: r[1] for r in rows}
    for members in fused.values():
        sizes = [dims[m][1] for m in members if len(dims.get(m, ())) > 1]
        for m in members:
            shard_sizes[m] = sizes
    census = {'algo': algo, 'tensors': 0, 'bytes': 0, 'elements': 0,
              'fallbacks': {}, 'by_dtype': {}}
    lines = []
    for name, shape, elems, _dtype, shard, sha in rows:
        eff, nbytes, why = plan_tensor(name, shape, elems, algo, arch,
                                       shard_sizes.get(name))
        if why and eff == BF16 and algo != BF16:
            census['fallbacks'][name] = why
        census['tensors'] += 1
        census['bytes'] += nbytes
        census['elements'] += elems
        census['by_dtype'][eff] = census['by_dtype'].get(eff, 0) + 1
        shp = '(' + ', '.join(str(d) for d in shape) + (',)' if len(shape) == 1 else ')')
        lines.append(f'{shard}:{sha}:{name}:shape={shp}:dtype={eff}:'
                     f'elements={elems}:bytes={nbytes}')
    return lines, census


def synthesise_map(bf16_map_path: str, algo: str, out_path: str,
                   arch=ARCH_QWEN3_5) -> dict:
    """Write tensors.<algo>.map.  Returns a census dict."""
    lines, census = synthesise_map_lines(read_bf16_map(bf16_map_path), algo, arch)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return census


# -----------------------------------------------------------------------------
# WHICH ARCHITECTURE IS THIS, ASKED OF EACH KIND OF SOURCE
# -----------------------------------------------------------------------------
#
# `--arch` used to be typed on every command line, defaulted to qwen3_5, and a
# wrong one produced a refusal rather than a wrong checkpoint - but only further
# in, after minutes of work.  Every source a tool in this lane is given already
# STATES its architecture, in its own dialect:
#
#   an HF snapshot   config.json `architectures` (exact) and `model_type`
#                    (a family, shared by variants this table may not cover)
#   a GGUF split     `general.architecture` in the metadata shard
#   a tensor map     the tensor names themselves - `detect_arch`, the strictest
#                    of the three, because an arch qualifies only by naming
#                    EVERY tensor
#
# The three are read here and NOWHERE else, and each returns None rather than a
# guess.  The caller combines them, prefers the strict one when it can be had,
# and refuses when none of them fits.

def arch_from_hf_config(cfg) -> Optional[str]:
    """The arch key an HF config.json names, or None.

    `architectures` decides.  `model_type` is consulted only when that misses
    AND exactly one registered arch claims it: `qwen3_5` is the model_type of
    both the dense family this suite has a table for and the MoE one it does
    not, so an ambiguous model_type must not settle anything.
    """
    if not isinstance(cfg, dict):
        return None
    names = cfg.get('architectures') or []
    if isinstance(names, str):
        names = [names]
    for key, arch in ARCHS.items():
        if any(n in arch.get('hf_architectures', ()) for n in names):
            return key
    mt = cfg.get('model_type') or (cfg.get('text_config') or {}).get('model_type')
    hits = [k for k, a in ARCHS.items() if mt in a.get('hf_model_types', ())]
    return hits[0] if len(hits) == 1 else None


def arch_from_gguf_arch(name) -> Optional[str]:
    """The arch key a GGUF's `general.architecture` names, or None."""
    for key, arch in ARCHS.items():
        if name in arch.get('gguf_arch', ()):
            return key
    return None


def detect_arch_hf(names) -> Optional[str]:
    """The arch whose module map claims an HF checkpoint's tensor names.

    The safetensors counterpart of `detect_arch`, and the same rule stated for
    the other naming convention: an arch qualifies by CLAIMING THE LAYER STACK,
    not by claiming a name or two.  Both tables declare `lm_head`, so a global
    hit means nothing and only per-layer hits are counted; the winner must be
    unique, or this returns None and the caller asks for `--arch`.
    """
    names = [n for n in names if n.endswith(('.weight', '.bias'))]
    score = {}
    for key, arch in ARCHS.items():
        dn = DiskNames(arch)
        head = arch['hf_layer_prefix'].split('{bid}')[0]
        score[key] = sum(1 for n in names if n.startswith(head)
                         and dn.classify(n.rsplit('.', 1)[0]) is not None)
    best = sorted(score, key=lambda k: -score[k])
    if not best or score[best[0]] == 0:
        return None
    if len(best) > 1 and score[best[1]] == score[best[0]]:
        return None
    return best[0]


def detect_arch(gguf_names) -> Optional[str]:
    """Pick the arch whose name map covers every tensor in the map file.

    Deliberately strict: a partial match means a tensor we cannot name, and a
    tensor we cannot name is one we would silently leave BF16 - which blows the
    byte budget in a run that reports success.
    """
    names = list(gguf_names)
    for key, arch in ARCHS.items():
        if all(map_tensor(n, arch) is not None for n in names):
            return key
    return None


# -----------------------------------------------------------------------------
# THE EMBEDDING, AND THE ONE TYPE A DRAFT MODEL CAN SHARE
# -----------------------------------------------------------------------------
#
# WHY THE EMBEDDING IS SINGLED OUT.  Speculative decoding - MTP/NEXTN and EAGLE
# alike - runs a small draft head that SHARES THE TARGET MODEL'S EMBEDDING
# TABLE, and stock SGLang hands it that table as a bare `nn.Parameter`
# (`model_runner.py` copies `embed_tokens` across to the draft worker).  A
# packed NVFP4 table is not a bare parameter: it is a uint8 code block plus
# per-block e4m3 scales plus a global `weight_scale_2`, so the draft either
# reads garbage or fails outright.  An engine can be patched to share the
# unpacked table (a fix of ours is pending upstream); a user on stock SGLang
# cannot.
#
# THE SCOPE IS EXACTLY ONE TENSOR.  `lm_head`/`output.weight` at NVFP4 runs
# NEXTN fine - MEASURED, not assumed - because the draft head resolves the
# output projection through the quant method like any other linear.  So the
# compatible recipe costs the embedding's bytes and nothing else: on
# Qwen3.8-27B, 1.70 GiB, no prefill time (the embedding is a gather, not a GEMM)
# and no decode time (one row per token, not the whole table).

def embedding_tensors(arch: dict = ARCH_QWEN3_5) -> List[str]:
    """The arch's GLOBAL embedding tensor name(s), from the role column.

    Model-agnostic on purpose: the caller never spells `token_embd`, it asks the
    arch which of its global tensors is the gather.  Per-layer embeddings (a
    dedicated MTP embedding, `blk.N.nextn.embed_tokens.weight`) are deliberately
    NOT returned - they are the draft's own table, not the one it shares.
    """
    return sorted(n for n, (_hf, role, _f) in arch['globals'].items()
                  if role == 'embedding')


def dense_embedding_algo(pool=None) -> Optional[str]:
    """The pool member that stores an embedding UNPACKED, or None.

    `algo_legal(role='embedding')` admits at most two types - BF16 and NVFP4 -
    because `get_quant_method`'s VocabParallelEmbedding branch knows only NVFP4
    (:1035-1043) and every other algo silently returns None; which of the two an
    arch actually gets is its `embedding_algos`.  Of those two only BF16 is a
    plain tensor a draft model can share, and it is the only one every arch has.

    DERIVED, not asserted: an algo qualifies by declaring role 'embedding', a
    full 16 bpw, no block/group scales and no per-tensor companions - i.e. by
    being one contiguous array of weights.  If a future pool member gains an
    unpacked embedding method it qualifies here without this function being
    edited, and `sgl_nvfp4` never does, because its `block` is ('e4m3', 16).
    """
    for algo in (pool if pool is not None else SGLANG_POOL_ORDER):
        spec = SGLANG_ALGOS.get(algo)
        if not spec or 'embedding' not in spec.get('roles', ()):
            continue
        if (float(spec.get('bpw') or 0) == 16.0
                and spec.get('block') is None
                and not spec.get('group_size')
                and not spec.get('per_tensor_bytes')):
            return algo
    return None


# MTP / NEXTN tensors by NAME, in either naming convention.  `blk.92.nextn.*`
# (GGUF, GLM-4.7), `model.layers.92.mtp.*` / `mtp.layers.0.*` (HF).  Anchored on
# a path component so `nextnorm` or a tensor merely containing the letters does
# not match.
NEXTN_NAME_RE = re.compile(r'(?:^|[.\-_/])(?:nextn|mtp)(?:[.\-_/]|\d|$)', re.I)


def nextn_tensor_names(names) -> List[str]:
    """The subset of `names` that belongs to an MTP/NEXTN draft head."""
    return [n for n in names if NEXTN_NAME_RE.search(str(n))]


# =============================================================================
# 4. THE BUDGET
# =============================================================================

def fixed_bf16_bytes(bf16_map_path: str, arch=ARCH_QWEN3_5) -> Tuple[int, int]:
    """(bytes, count) of the tensors no recipe can move: SGLang keeps them BF16.

    These are the GGUF F32 tensors.  `quant_assign.py` deducts them from the
    target itself (`f32_offset`, quant_assign.py:6641) but sizes them from the
    bf16 map, i.e. at 4 B/elem - the GGUF figure, not the SGLang one.  Running
    with `--ignore-f32` and subtracting THIS number from the byte target instead
    keeps the arithmetic exact and keeps it in one auditable place.
    """
    total = cnt = 0
    for name, shape, elems, _dtype, _s, _h in read_bf16_map(bf16_map_path):
        info = map_tensor(name, arch)
        if info is None:
            raise KeyError(f'{name!r} is not in the arch name map')
        if info[1] == 'other' or len(shape) < 2:
            total += 2 * elems
            cnt += 1
    return total, cnt


# =============================================================================
# 5. RECIPE -> CHECKPOINT
# =============================================================================

_RECIPE_RE = re.compile(r'^\^?(?P<name>.+?)\$?=(?P<q>[A-Za-z0-9_]+)\s*$')

# A recipe line's NAME is a regex.  quant_assign.py emits it as a fully-escaped
# literal (one line per tensor), but its PUBLISHED form - the shape every other
# recipe in the suite ships in - runs the recipe through `quants_regex_merger`,
# which folds a whole family of tensors into ONE alternation, e.g.
# `^blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_down\.weight$=sgl_nvfp4`.  These
# characters never appear in a real GGUF tensor name, so their presence in the
# un-escaped name is the unambiguous signal that a line is a compacted regex
# rather than a single literal - see parse_recipe().
_RECIPE_META = set('()[]|+*?')


def _recipe_lines(path: str):
    """[(pattern, qtype)] in file order; pattern keeps its ^…$ as authored."""
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = _RECIPE_RE.match(line)
            if not m:
                continue
            # LHS incl. any ^…$ anchors: a pattern never contains '=', and the
            # qtype is [A-Za-z0-9_]+, so the last '=' always splits them.
            pattern = line.rpartition('=')[0].strip()
            out.append((pattern, m.group('q').lstrip('!').strip()))
    return out


def _literal_name(pattern: str) -> str:
    """The single tensor name a fully-literal recipe pattern denotes."""
    body = pattern
    if body.startswith('^'):
        body = body[1:]
    if body.endswith('$'):
        body = body[:-1]
    return body.replace('\\.', '.').replace('\\', '')


def parse_recipe(path: str, universe=None) -> Dict[str, str]:
    """{gguf tensor name -> qtype} from a quant_assign.py recipe.

    A recipe line is `^blk\\.0\\.ffn_up\\.weight$=sgl_nvfp4`: an anchored regex
    of the (escaped) tensor name.  quant_assign.py writes ONE line per tensor,
    but the form the suite PUBLISHES - and the form every other recipe here
    already ships in - is the output of `quants_regex_merger`, which folds a
    whole family into a single alternation:
    `^blk\\.([0-9]|[1-5][0-9]|6[0-3])\\.ffn_down\\.weight$=sgl_nvfp4`.  The GGUF
    build path expands that regex NATIVELY - quant_downloader.sh matches each
    pattern against the actual tensor list with bash `[[ $name =~ $pat ]]` and
    the FIRST pattern that matches a tensor wins (its five match loops all
    `break` on first hit).  This reader is the SGLang build path's equivalent and
    must do the same, so a merged recipe rebuilds byte-for-byte the same
    checkpoint as its raw twin instead of silently collapsing to BF16.

    `universe` is the model's tensor set - the GGUF names of its
    `tensors.bf16.map` (`read_bf16_map`).  GIVEN it, every line's pattern is
    matched against that set exactly as the GGUF path matches, FIRST-match-wins,
    so a compact recipe and its per-tensor original yield identical maps.  A
    plain literal line still resolves to exactly the one name it spells (a
    literal is just a regex that matches one name), so a raw recipe is
    unchanged.  WITHOUT a universe each line is taken as that one literal (the
    historical behaviour, bit-for-bit for a raw recipe) and a COMPACTED line -
    which has no tensor set to expand over - is a hard error rather than the
    silent collapse to one bogus name it used to be.

    The leading `!` quant_assign.py puts on a qtype whose map it COMPUTED rather
    than downloaded is provenance for the reader, not part of the type name, so
    it is stripped here exactly as qa-recipe-to-ttf.sh strips it for
    llama-quantize.
    """
    lines = _recipe_lines(path)
    out: Dict[str, str] = {}

    if universe is None:
        for pattern, q in lines:
            name = _literal_name(pattern)
            if _RECIPE_META & set(name):
                raise SystemExit(
                    f'{path}: recipe line {pattern!r} is a compacted regex, not '
                    f'a single tensor name.  Pass the model tensor set '
                    f'(tensors.bf16.map) so it can be expanded '
                    f'(sglang_write.py: --bf16-map).')
            out[name] = q
        return out

    # Expansion path: FIRST-match-wins over the model's tensor set, mirroring the
    # bash `=~` (search, not fullmatch) the GGUF downloader uses.  A fully
    # anchored literal takes a direct set-membership fast path - identical result
    # to searching its escaped regex, but O(1) - which keeps a raw per-tensor
    # recipe (hundreds/thousands of literal lines) cheap.
    uni_list = list(universe)          # iteration order (the bf16-map order)
    uni_set = set(uni_list)            # O(1) membership for the literal fast path
    for pattern, q in lines:
        name = _literal_name(pattern)
        if (pattern.startswith('^') and pattern.endswith('$')
                and not (_RECIPE_META & set(name))):
            if name in uni_set and name not in out:
                out[name] = q
            continue
        rx = re.compile(pattern)
        for tname in uni_list:
            if tname not in out and rx.search(tname):
                out[tname] = q
    return out


def recipe_to_quantized_layers(recipe: Dict[str, str], arch=ARCH_QWEN3_5):
    """{hf prefix -> {quant_algo[, group_size]}} plus a list of violations.

    THE MAPPING, in four rules:
      1. `^<name>$=<qtype>`  ->  strip the regex, look the GGUF name up in the
         arch map, take the HF prefix.
      2. qtype -> `quant_algo` (and `group_size` for the FP4 pair).
      3. bf16 / f32 / anything unquantised -> OMIT the entry entirely.  Absence
         IS the encoding: get_quant_method returns UnquantizedLinearMethod for a
         prefix it cannot resolve (modelopt_quant.py:1027).  Do NOT put it in
         `exclude_modules` - that list is matched by `is_layer_skipped` and is
         for whole subtrees (RadixArk uses it only for `mtp*`).
      4. every fused group must be uniform, or SGLang raises at load
         (modelopt_quant.py:967).  Checked here rather than discovered on the
         GPU.

    AND ONE RULE THAT IS NOT ABOUT THIS FILE but about the weights written
    beside it: for NVFP4, every shard of a fused module must also share one
    `weight_scale_2` - see nvfp4_global_scale().  A config.json this function
    accepts can still produce a broken checkpoint if the writer gets that wrong,
    and the failure is silent.
    """
    layers: Dict[str, dict] = {}
    per_group: Dict[str, Dict[str, List[str]]] = {}
    problems: List[str] = []
    for name, q in sorted(recipe.items()):
        info = map_tensor(name, arch)
        if info is None:
            problems.append(f'{name}: not in the arch name map')
            continue
        prefix, role, fused, bid = info
        spec = SGLANG_ALGOS.get(q)
        if spec is None:
            if q in ('bf16', 'f32', BF16):
                spec = SGLANG_ALGOS[BF16]
            else:
                problems.append(f'{name}: {q!r} is not an SGLang-native qtype')
                continue
        if fused is not None:
            per_group.setdefault(f'{fused}.{bid}', {}).setdefault(
                spec['quant_algo'] or 'BF16', []).append(name)
        if spec['quant_algo'] is None:
            continue                      # rule 3: absence means BF16
        entry = {'quant_algo': spec['quant_algo']}
        if 'group_size' in spec:
            entry['group_size'] = spec['group_size']
        layers[prefix] = entry
    for gid, algos in sorted(per_group.items()):
        if len(algos) > 1:
            problems.append(
                f'fused module {gid}: shards disagree {sorted(algos)} - SGLang '
                f'raises "Mixed quant_algo within fused layer"')
    return layers, problems


def build_hf_quant_config(layers: Dict[str, dict],
                          kv_cache_quant_algo: str = 'FP8',
                          exclude_modules=None) -> dict:
    """The `hf_quant_config.json` an SGLang mixed checkpoint needs.

    `ModelOptMixedPrecisionConfig.from_config` (modelopt_quant.py:840-878) reads
    the nested form, requires `quant_algo == "MIXED_PRECISION"` and a NON-EMPTY
    `quantized_layers`, and takes `group_size` from the first NVFP4-ish entry it
    finds.  `override_quantization_method` (:822-827) selects this config class
    when the top-level `quant_method` is `"modelopt_mixed"` - without that key
    the loader will not pick the mixed path at all.
    """
    return {
        'producer': {'name': 'GGUF-Tool-Suite', 'version': 'blackwell-fp4'},
        'quant_method': 'modelopt_mixed',
        'quantization': {
            'quant_algo': 'MIXED_PRECISION',
            'kv_cache_quant_algo': kv_cache_quant_algo,
            'exclude_modules': list(exclude_modules or []),
            'quantized_layers': layers,
        },
    }


# =============================================================================
# 6. VERIFICATION AGAINST A REAL MODELOPT CHECKPOINT
# =============================================================================

def read_safetensors_headers(model_dir: str) -> Dict[str, Tuple[str, List[int], int]]:
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, '*.safetensors'))):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == '__metadata__':
                continue
            off = meta['data_offsets']
            out[name] = (meta['dtype'], list(meta['shape']), off[1] - off[0])
    return out


# --- TABLE-FREE role inference and byte check over ANY HF checkpoint ---------
#
# The declarative ARCHS tables above are keyed by GGUF tensor names, because
# that is the side the assigner works on.  This block is the other side: given
# nothing but a published HF checkpoint - no ARCHS entry, no bf16 map, no
# calibration - can we (a) give every tensor a role and (b) reproduce its bytes
# exactly?  If yes, the size model is genuinely architecture-independent and a
# new model is a data problem, not a code problem.  `--smoke-hf <dir>` runs it
# on any checkpoint; the selftest runs it on whatever is in the local HF cache.
#
# The rules below are HF NAMING CONVENTIONS shared across model families, not a
# per-model list: `embed_tokens` is the embedding, `lm_head` is the output head,
# a leaf ending in `_proj` is a projection, `.bias`/`.weight_scale`/
# `.input_scale` are companions of a module rather than modules, and a 1-D
# `.weight` is a norm.  Anything they do not recognise is reported as 'other'
# EXPLICITLY - never guessed into a role that would change its byte cost.

_HF_LINEAR_LEAVES = frozenset((
    'lm_head', 'dense', 'fc1', 'fc2', 'linear', 'gate', 'gate_proj', 'up_proj',
    'down_proj', 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'qkv_proj',
    'query_key_value', 'w1', 'w2', 'w3', 'out_proj', 'in_proj',
    # Megatron/ViT-style short names, used by the vision towers and the MLA
    # blocks of several families (measured on GLM-5.3-Flash, whose tower calls
    # them `qkv`/`proj` and whose MLA calls them `wq_b`/`wk`).  A `_proj`
    # SUBSTRING is checked separately, which catches `kv_a_proj_with_mqa`.
    'qkv', 'proj', 'wq', 'wk', 'wv', 'wo', 'wq_a', 'wq_b', 'wkv_a', 'wkv_b',
))
_HF_EMBEDDING_LEAVES = frozenset(('embed_tokens', 'word_embeddings', 'wte'))
# Suffixes that belong TO a module rather than being one.  They are counted
# against their parent module's byte total, never on their own.
_HF_COMPANION_SUFFIXES = frozenset((
    'bias', 'weight_scale', 'weight_scale_2', 'input_scale', 'weight_scale_inv',
    'weight_zero_point', 'k_scale', 'v_scale', 'scale',
))

# OF THOSE COMPANIONS, THE ONES `algo_bytes()` ACTUALLY MODELS.
#
# THIS DISTINCTION COST A RED SELFTEST, so it is written down.  `algo_bytes()`
# prices the QUANTISATION of a weight: the packed codes, the block scales, the
# global scale, the activation scale.  It does not price - and must not price -
# a tensor that would exist at BF16 too.  `smoke_hf_checkpoint()` used to
# compare it against the sum of EVERY tensor under the module prefix, which is
# the same number only for a checkpoint whose linears have no bias.
#
# MEASURED on Salyut1/GLM-4.7-NVFP4, the first cached checkpoint with attention
# biases: the residual was exactly -2,695,520 B and decomposed with no
# remainder into 276 BF16 `bias` tensors (2,637,824 B - GLM-4.7 sets
# `attention_bias: true`, so q/k/v each carry one per output channel), 89 F32
# `e_score_correction_bias` router biases (56,960 B) and 92+92 F32 `k_scale` /
# `v_scale` FP8 KV-CACHE scalars (736 B).  Not one of them is a ModelOpt layout
# variant; every one is a model parameter or a runtime calibration scalar that
# happens to live under a quantised module's name.
#
# So the byte check compares artefacts against artefacts, and everything else a
# module carries is reported by name and by size instead of being silently
# folded into a residual.  That also makes the check STRICTER: a 2 KB error in
# a block-scale used to be maskable by a bias of the same size.
_HF_QUANT_ARTEFACT_SUFFIXES = frozenset((
    'weight', 'weight_packed', 'weight_scale', 'weight_scale_2',
    'weight_scale_inv', 'weight_zero_point', 'weight_shape', 'input_scale',
))

# The checkpoints docs/sglang.md claims BYTE-EXACTNESS for, by HF repo id.  The
# selftest asserts on these and REPORTS on anything else in the cache: a machine
# is free to hold checkpoints this branch never chose, and one of them being
# unusual is a finding, not a broken build.  `--smoke-cache` is that sweep.
HF_EXACT_REPOS = ('RadixArk/Qwen3.8-27B-NVFP4', 'RadixArk/GLM-5.3-Flash-NVFP4')


def infer_role_hf(name: str, shape=None):
    """(role, rule) for one HF tensor, from its name and shape alone.

    role is 'linear', 'embedding' or 'other' - the same three roles the
    declarative tables use.  `rule` names the convention that fired, so a
    surprising classification can be traced instead of trusted.
    """
    parts = name.split('.')
    suffix = parts[-1]
    if suffix in _HF_COMPANION_SUFFIXES:
        return 'other', 'companion-suffix'
    if suffix != 'weight':
        # inv_freq, hc_attn_base, dt_bias, A_log, ... - buffers, not modules.
        return 'other', 'not-a-weight'
    leaf = parts[-2] if len(parts) > 1 else parts[-1]
    if leaf in _HF_EMBEDDING_LEAVES:
        return 'embedding', 'embedding-leaf'
    if shape is not None and len(shape) < 2:
        return 'other', '1d-weight-is-a-norm'
    if shape is not None and len(shape) > 3:
        # A rank-4/5 weight is a CONVOLUTION, not a GEMM - e.g. a ViT patch
        # embedder's [out, in, t, kh, kw].  Rank 3 is reserved for the stacked
        # -expert convention [E, N, K], which IS a GEMM, E of them.
        return 'other', 'nd-weight-is-a-convolution'
    if leaf.endswith('_proj') or leaf in _HF_LINEAR_LEAVES:
        return 'linear', 'projection-leaf'
    if '_proj' in leaf:
        # e.g. `kv_a_proj_with_mqa` - a projection with a descriptive tail.
        return 'linear', 'projection-substring'
    return 'other', 'unrecognised-leaf'


def _ct_qtype(weights: dict):
    """Map a compressed-tensors `weights` block onto one of OUR qtype names.

    Returns None when the published format is outside our pool - a per-CHANNEL
    FP8, say, whose byte overhead is not the per-tensor one our size model
    assumes.  Returning None is the honest answer there: modelling it with a
    near-miss qtype would produce a byte MISMATCH and blame the size model for
    a format it never claimed to cover.
    """
    if not weights:
        return None
    bits = weights.get('num_bits')
    typ = str(weights.get('type') or '').lower()
    gs = weights.get('group_size')
    strategy = str(weights.get('strategy') or '').lower()
    if typ == 'int' and bits == 4 and gs == 128:
        return 'sgl_int4_g128'
    if typ == 'float' and bits == 8 and strategy == 'tensor':
        return 'sgl_fp8'
    if typ == 'float' and bits == 4 and gs == 16:
        return 'sgl_nvfp4'
    return None


def _hf_module_algos(quant_cfg: dict, modules, roles=None):
    """Which quantisation each module carries, according to the CHECKPOINT.

    Reads the three published dialects, none of which is treated as a default:

      1. ModelOpt `hf_quant_config.json` with a per-module `quantized_layers`
         map - what our own writer emits.
      2. ModelOpt with a GLOBAL `quant_algo` plus `exclude_modules` globs - what
         RadixArk's GLM-5.3-Flash ships.
      3. compressed-tensors `config_groups` (targets are either the literal
         "Linear" or `re:` regexes) with an `ignore` list, and the plain
         `quant_method: fp8` shape with `modules_to_not_convert`.

    Returns (algos, out_of_pool) where `algos` maps module -> OUR qtype name and
    `out_of_pool` lists modules the checkpoint quantises in a format our pool
    does not express.  Keeping those two apart is what stops the smoke from
    reporting a byte MISMATCH for a format it was never asked to model.
    """
    q = quant_cfg.get('quantization', quant_cfg)
    name_of_algo = {SGLANG_ALGOS[a]['quant_algo']: a
                    for a in SGLANG_ALGOS if SGLANG_ALGOS[a]['quant_algo']}
    out, out_of_pool = {}, []

    ql = q.get('quantized_layers')
    if ql:
        for m in modules:
            ent = ql.get(m)
            if not ent:
                continue
            qn = name_of_algo.get(str(ent['quant_algo']).upper())
            (out.__setitem__(m, qn) if qn else out_of_pool.append(m))
        return out, out_of_pool

    glob_algo = q.get('quant_algo')
    if glob_algo:
        excl = list(q.get('exclude_modules') or [])
        qn = name_of_algo.get(str(glob_algo).upper())
        for m in modules:
            if any(fnmatch.fnmatchcase(m, pat) for pat in excl):
                continue
            (out.__setitem__(m, qn) if qn else out_of_pool.append(m))
        return out, out_of_pool

    groups = q.get('config_groups')
    if groups:
        ignore = set(q.get('ignore') or [])
        for _g, spec in sorted(groups.items()):
            qn = _ct_qtype(spec.get('weights') or {})
            targets = list(spec.get('targets') or [])
            pats = [t[3:] for t in targets if t.startswith('re:')]
            any_linear = any(t == 'Linear' for t in targets)
            for m in modules:
                if m in ignore or m in out or m in out_of_pool:
                    continue
                hit = any(re.search(pat, m) for pat in pats)
                if not hit and any_linear:
                    hit = (roles or {}).get(m) == 'linear'
                if not hit:
                    continue
                (out.__setitem__(m, qn) if qn else out_of_pool.append(m))
        return out, out_of_pool

    if str(q.get('quant_method') or '').lower() == 'fp8':
        # The plain `quant_method: fp8` shape does not say whether the weight
        # scale is per-TENSOR (which `sgl_fp8` models, 8 B) or per-CHANNEL
        # (which it does not).  Measured on Qwen3.8-27B-FP8: assuming
        # per-tensor is wrong by 10,880 B on a [17408, 5120] projection.  So it
        # is reported as out-of-pool rather than mismodelled.
        skip = set(q.get('modules_to_not_convert') or [])
        for m in modules:
            if m in skip or (roles or {}).get(m) != 'linear':
                continue
            out_of_pool.append(m)
        return out, out_of_pool

    return out, out_of_pool


def hf_hub_dir() -> str:
    """The local HF hub cache this box uses."""
    return os.environ.get('HF_HUB_CACHE') or os.path.join(
        os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'), 'hub')


def hf_cached_repos(hub: Optional[str] = None):
    """[(repo id, snapshot dir)] for every cached repo that has safetensors."""
    hub = hub or hf_hub_dir()
    out = []
    if not os.path.isdir(hub):
        return out
    for snap in sorted(glob.glob(os.path.join(hub, 'models--*', 'snapshots', '*'))):
        if not glob.glob(os.path.join(snap, '*.safetensors')):
            continue
        leaf = os.path.basename(os.path.dirname(os.path.dirname(snap)))
        out.append((leaf[len('models--'):].replace('--', '/'), snap))
    return out


def _hf_snapshot(repo: str, hub: Optional[str] = None) -> Optional[str]:
    """The cached snapshot of exactly this repo id, or None."""
    for r, snap in hf_cached_repos(hub):
        if r == repo:
            return snap
    return None


def smoke_cache(verbose: bool = True) -> int:
    """OPT-IN: weigh every checkpoint in the local HF cache and report.

    Not part of `--selftest`, and deliberately so.  What is in a cache is not a
    property of this code: a checkpoint somebody downloaded this morning turning
    the build red says nothing about the size model, and refusing to look at it
    says nothing either.  So this looks at all of them, prints what it found -
    including the bytes each one carries that are NOT quantisation artefacts -
    and returns non-zero only when a checkpoint cannot be CLASSIFIED, which is a
    statement about the generic rules rather than about one author.
    """
    rows = hf_cached_repos()
    if not rows:
        print(f'no checkpoints under {hf_hub_dir()}')
        return 0
    bad = 0
    print(f'{"repo":<40} {"modules":>8} {"exact":>8} {"residual B":>14} {"carried B":>13}')
    for repo, snap in rows:
        nprob, st = smoke_hf_checkpoint(snap, verbose=False)
        resid = st['modelled_bytes'] - st['disk_bytes']
        print(f'{repo:<40} {st["checked"]:>8,} {st["exact"]:>8,} {resid:>14,} '
              f'{st["carried_bytes"]:>13,}'
              + ('' if repo not in HF_EXACT_REPOS else '   <- asserted by --selftest'))
        if st['checked'] == 0:
            bad += 1
            print(f'    [!] nothing was byte-checked - the config reader found no '
                  f'dialect it knows')
        if st['carried']:
            print('    carries (not quantisation artefacts): '
                  + ', '.join(f'{k} x{st["carried_n"][k]:,} = {v:,} B'
                              for k, v in sorted(st['carried'].items(),
                                                 key=lambda kv: -kv[1])))
        if resid and verbose:
            nprob2, _st2 = smoke_hf_checkpoint(snap, verbose=True)
    print(f'\n{len(rows)} checkpoint(s); '
          + ('all classified' if not bad else f'{bad} could not be classified'))
    return 0 if bad == 0 else 1


def smoke_hf_checkpoint(model_dir: str, verbose: bool = True):
    """Classify and re-weigh a published checkpoint with NO per-model table.

    Returns (n_problems, stats).  Two assertions, both falsifiable:
      1. every tensor gets a role, and every module the checkpoint ITSELF
         quantises is classified 'linear' (the checkpoint is ground truth for
         which modules are GEMMs, so this cannot be tuned to pass);
      2. for each such module, algo_bytes() over the shape recovered from the
         PACKED weight equals the module's on-disk bytes, to the byte.
    """
    hdrs = read_safetensors_headers(model_dir)
    # TWO published places, not one.  ModelOpt-style checkpoints carry
    # hf_quant_config.json; compressed-tensors and plain FP8 builds put the
    # same information in config.json under `quantization_config`.  Reading only
    # the first made this smoke pass VACUOUSLY (0 modules checked) on four of
    # the six checkpoints in the local cache - measured 2026-09-05.
    quant_cfg = {}
    cfg_path = os.path.join(model_dir, 'hf_quant_config.json')
    if os.path.exists(cfg_path):
        quant_cfg = json.load(open(cfg_path))
    else:
        main_cfg = os.path.join(model_dir, 'config.json')
        if os.path.exists(main_cfg):
            qc = (json.load(open(main_cfg)) or {}).get('quantization_config')
            if qc:
                quant_cfg = {'quantization': qc}
    # group tensors by module (strip the companion suffix)
    modules: Dict[str, Dict[str, Tuple[str, List[int], int]]] = {}
    roles, rules = {}, {}
    for name, (dt, shape, nb) in hdrs.items():
        role, rule = infer_role_hf(name, shape)
        roles[name] = role
        rules.setdefault(rule, 0)
        rules[rule] += 1
        parts = name.split('.')
        base = '.'.join(parts[:-1])
        modules.setdefault(base, {})[parts[-1]] = (dt, shape, nb)
    module_roles = {m: infer_role_hf(m + '.weight',
                                     (ts.get('weight') or ('', None, 0))[1])[0]
                    for m, ts in modules.items()}
    algos, out_of_pool = _hf_module_algos(quant_cfg, list(modules), module_roles)
    problems, checked, exact, modelled_bytes, disk_bytes = [], 0, 0, 0, 0
    # Tensors a quantised module carries that are not quantisation artefacts,
    # by suffix.  Reported, never silently absorbed into the residual.
    carried: Dict[str, int] = {}
    carried_n: Dict[str, int] = {}
    unclassified = [n for n, r in roles.items() if r not in ('linear', 'embedding', 'other')]
    # ANTI-VACUITY.  A checkpoint whose weights are stored in a packed dtype IS
    # quantised, whatever its config says.  If the config reader attributes an
    # algo to none of them, the reader is broken and the smoke must FAIL rather
    # than report "0 modules checked, PASS".
    packed = {m for m, ts in modules.items()
              if 'weight' in ts and ts['weight'][0] in ('U8', 'I32', 'F8_E4M3', 'F8_E5M2')}
    if packed and not algos and not out_of_pool:
        problems.append(
            f'{len(packed):,} modules are stored in a packed dtype but the '
            f'config reader recognised no dialect at all - the byte check '
            f'would be vacuous')
    # Unquantised 2-D weights are modelled too (at BF16), so a plain dense
    # checkpoint is a REAL byte check rather than an empty one.
    todo = dict(algos)
    for mod, ts in modules.items():
        if mod in todo:
            continue
        w = ts.get('weight')
        if w is not None and w[0] in ('BF16', 'F16') and len(w[1]) >= 2 and len(ts) == 1:
            todo[mod] = BF16
    for mod, algo in sorted(todo.items()):
        w = modules.get(mod, {}).get('weight')
        if w is None:
            continue
        dt, shape, nb = w
        qname = algo          # already OUR qtype name (or BF16)
        if qname is None or len(shape) < 2:
            continue
        # THE DTYPE ON DISK BEATS THE CONFIG'S CLAIM.  compressed-tensors names
        # its targets with regexes, and those regexes over-match: RedHatAI's
        # `re:.*self_attn\.(q|k|v|o)_proj$` selects the MTP head, which the
        # build then leaves BF16.  A loader believes the tensor, not the
        # pattern, and so does this check.
        if qname != BF16 and dt in ('BF16', 'F16', 'F32'):
            qname = BF16
        role, _rule = infer_role_hf(mod + '.weight', shape)
        # A QUANTISED MODULE MUST BE A QUANTISABLE ROLE, and there are two of
        # them, not one: `algo_legal()` admits NVFP4 for an embedding as well as
        # for a linear.  Asked against `qname` rather than `algo` because the
        # dtype on disk has already overruled the config above - the ModelOpt
        # GLOBAL dialect (`quant_algo` + `exclude_modules`) claims its algo for
        # every module it does not exclude, so on Salyut1/GLM-4.7-NVFP4 it
        # claims NVFP4 for a `model.embed_tokens` that is plainly BF16 on disk.
        if role not in ('linear', 'embedding') and qname != BF16:
            problems.append(f'{mod}: checkpoint quantises it as {qname} but the '
                            f'generic rules called it {role!r}')
        # Recover the LOGICAL (K, N) from the PACKED weight: NVFP4 stores
        # [N, K/2] as U8, FP8 stores [N, K] as e4m3, BF16 stores [N, K].
        n_rows = shape[0]
        k = shape[-1] * (2 if dt == 'U8' else 1)
        count = 1
        if len(shape) == 3:     # stacked experts: [E, N, K'] -> E modules
            n_rows, count = shape[1], shape[0]
        elif len(shape) > 3:    # a convolution; not a GEMM, so not modelled
            continue
        want = algo_bytes(qname, k, n_rows, count)
        # ARTEFACTS AGAINST ARTEFACTS.  Everything else the module carries - a
        # bias, an FP8 KV-cache scale - is a model parameter, not part of the
        # quantisation, and is accounted separately (see
        # _HF_QUANT_ARTEFACT_SUFFIXES).
        got = 0
        for _suf, _v in modules[mod].items():
            if _suf in _HF_QUANT_ARTEFACT_SUFFIXES:
                got += _v[2]
            else:
                carried[_suf] = carried.get(_suf, 0) + _v[2]
                carried_n[_suf] = carried_n.get(_suf, 0) + 1
        checked += 1
        modelled_bytes += want
        disk_bytes += got
        if want == got:
            exact += 1
        elif len(problems) < 8:
            problems.append(f'{mod}: model says {want:,} B, on disk {got:,} B '
                            f'(dtype {dt}, shape {shape}, algo {algo})')
    if unclassified:
        problems.append(f'{len(unclassified)} tensors got no role at all')
    if checked == 0:
        problems.append('no module was byte-checked - a vacuous pass, not a pass')
    stats = dict(tensors=len(hdrs), modules=len(modules), quantised=len(algos),
                 out_of_pool=len(out_of_pool),
                 carried=carried, carried_n=carried_n,
                 carried_bytes=sum(carried.values()),
                 checked=checked, exact=exact, rules=rules,
                 modelled_bytes=modelled_bytes, disk_bytes=disk_bytes,
                 role_census={r: sum(1 for v in roles.values() if v == r)
                              for r in ('linear', 'embedding', 'other')})
    if verbose:
        print(f'  checkpoint: {model_dir}')
        print(f'  tensors {stats["tensors"]:,}   modules {stats["modules"]:,}   '
              f'quantised by its own config {stats["quantised"]:,}'
              + (f'   (+{stats["out_of_pool"]:,} in a format outside our pool, '
                 f'not modelled)' if stats['out_of_pool'] else ''))
        print(f'  roles: ' + '  '.join(f'{k}={v:,}' for k, v in stats['role_census'].items()))
        print(f'  rules fired: ' + '  '.join(f'{k}={v:,}' for k, v in sorted(rules.items())))
        print(f'  byte model checked on {checked:,} modules, exact on {exact:,}')
        print(f'  modelled {modelled_bytes:,} B vs on disk {disk_bytes:,} B  '
              f'(difference {modelled_bytes - disk_bytes:,})')
        if carried:
            print(f'  carried by those modules but NOT part of the quantisation, '
                  f'{sum(carried.values()):,} B total:')
            for _s in sorted(carried, key=lambda x: -carried[x]):
                print(f'    {_s:<26} x{carried_n[_s]:<6,} {carried[_s]:>14,} B')
        for p in problems[:10]:
            print(f'  [!] {p}')
    return len(problems), stats


def verify_against(bf16_map_path: str, ref_dir: str, arch=ARCH_QWEN3_5) -> int:
    """Rebuild the reference checkpoint's language-model byte total from the
    GGUF map alone, using this module's size model and the reference's own
    `quantized_layers` boundary.  Any difference is a bug in the size model.
    """
    ref_cfg = json.load(open(os.path.join(ref_dir, 'hf_quant_config.json')))
    ql = ref_cfg['quantization']['quantized_layers']
    algo_of = {k: v['quant_algo'].upper() for k, v in ql.items()}
    name_of_algo = {SGLANG_ALGOS[a]['quant_algo']: a
                    for a in SGLANG_ALGOS if SGLANG_ALGOS[a]['quant_algo']}
    rows = read_bf16_map(bf16_map_path)
    fused = fused_groups([r[0] for r in rows], arch)
    dims = {r[0]: r[1] for r in rows}
    shard_sizes = {}
    for members in fused.values():
        sizes = [dims[m][1] for m in members if len(dims.get(m, ())) > 1]
        for m in members:
            shard_sizes[m] = sizes
    total = 0
    per_algo: Dict[str, List[int]] = {}
    unmapped = []
    for name, shape, elems, _d, _s, _h in rows:
        info = map_tensor(name, arch)
        if info is None:
            unmapped.append(name)
            continue
        prefix, role, _f, _b = info
        want = algo_of.get(prefix)
        algo = name_of_algo.get(want, BF16) if want else BF16
        eff, nbytes, _why = plan_tensor(name, shape, elems, algo, arch,
                                        shard_sizes.get(name))
        if eff != algo:
            print(f'  [!] {name}: wanted {algo}, model says {eff}', file=sys.stderr)
        per_algo.setdefault(eff, [0, 0, 0])
        per_algo[eff][0] += 1
        per_algo[eff][1] += nbytes
        per_algo[eff][2] += elems
        total += nbytes
    hdrs = read_safetensors_headers(ref_dir)
    ref_total = 0
    for n, (_dt, _sh, nb) in hdrs.items():
        if n.startswith('model.language_model') or n.startswith('lm_head'):
            ref_total += nb
    print('  algo            count          bytes       elements    eff.bpw')
    for a in sorted(per_algo, key=lambda x: -per_algo[x][1]):
        c, b, e = per_algo[a]
        print(f'  {a:14s} {c:6d} {b:>14,} {e:>14,}   {b*8/e:9.6f}')
    print(f'  {"TOTAL":14s} {sum(v[0] for v in per_algo.values()):6d} '
          f'{total:>14,}')
    print(f'  reference (safetensors, language-model only): {ref_total:,}')
    print(f'  difference: {total - ref_total:,}'
          + ('   <- EXACT' if total == ref_total else '   <- MISMATCH'))
    if unmapped:
        print(f'  unmapped GGUF tensors: {len(unmapped)} {unmapped[:5]}')
    return 0 if (total == ref_total and not unmapped) else 1


# =============================================================================
# CLI
# =============================================================================

# =============================================================================
# 9. SELF-TEST - the invariants the size model and the name maps must hold
# =============================================================================
#
# Everything here is CPU-only, needs no GPU and no torch, and runs in ~2 s.  Two
# things are being defended:
#
#   1. THE BYTE MODEL IS EXACT.  Not "close": exact.  A recipe is a promise
#      that a checkpoint will weigh what the assigner says, and the writer has
#      to hit it to the byte.  The GLM-4.7 case below reproduces a PUBLISHED
#      200.8 GB checkpoint with a residual of 0 B once its KV-cache scalars are
#      accounted for, which is the strongest form this test can take.
#   2. THE NAME MAPS ARE TOTAL.  A tensor the arch cannot name is a tensor that
#      silently stays BF16, which blows the budget in a run that reports success.

_QWEN_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'models', 'Qwen3.8-27B', 'group0', 'tensors.bf16.map')
_GLM47_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'models', 'GLM-4.7', 'group0', 'tensors.bf16.map')


def selftest(verbose: bool = True) -> int:
    fails = []

    def ok(cond, label, extra=''):
        (print(f'  ok   {label}' + (f'  ({extra})' if extra else ''))
         if cond and verbose else None)
        if not cond:
            fails.append(label)
            print(f'  FAIL {label}' + (f'  ({extra})' if extra else ''))

    # -- 0. no published artefact carries a local path ------------------------
    _root = SUITE_ROOT
    _model = os.path.join(_root, 'models', 'Qwen3.8-27B')
    ok(publishable_path(os.path.join(_model, 'group0', 'kld_results.csv'),
                        cwd=_model, root=_root) == 'group0/kld_results.csv',
       'paths: a file under the model folder is recorded relative to it')
    ok(publishable_path(os.path.join(_root, 'quant_assign.py'),
                        cwd=_model, root=_root) == '../../quant_assign.py',
       'paths: a file elsewhere in the suite keeps the form that reproduces the run')
    ok(publishable_path('/somewhere/else/scratch/run7/degradation.csv',
                        cwd=_model, root=_root) == '<external>/degradation.csv',
       'paths: a file outside the suite keeps its name and loses its location')
    ok(publishable_path('/var/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/'
                        'snapshots/abc123', cwd=_model, root=_root)
       == '<hf-cache>/models--Qwen--Qwen3.8-27B/snapshots/abc123',
       'paths: a Hugging Face cache entry keeps the public part, drops the cache')
    ok(publishable_path('group0/kld_results.csv', cwd=_model, root=_root)
       == 'group0/kld_results.csv',
       'paths: a relative path is already the published form and is untouched')
    _home = os.path.expanduser('~')
    _rendered = [publishable_path(x, cwd=_model, root=_root) for x in
                 (os.path.join(_home, 'scratch', 'a.csv'),
                  os.path.join(_home, '.cache', 'huggingface', 'hub',
                               'models--org--repo', 'snapshots', 'r1'),
                  os.path.join(_model, 'group0', 'notes.txt'))]
    ok(not any(os.path.isabs(r) or _home in r for r in _rendered),
       'paths: nothing rendered is absolute and nothing names a home directory',
       ' '.join(_rendered))

    # -- 1. the type table ----------------------------------------------------
    ok(container_for(['sgl_fp8_pb_wo', 'sgl_int4_g128']) is None,
       'container: modelopt FP8_PB_WO and compressed-tensors INT4 cannot mix')
    ok(container_for(['sgl_nvfp4', 'sgl_int4_g128']) == 'compressed-tensors',
       'container: NVFP4 + INT4 is a compressed-tensors checkpoint')
    ok(container_for(['sgl_nvfp4', 'sgl_fp8_pb_wo']) == 'modelopt_mixed',
       'container: NVFP4 + FP8_PB_WO is modelopt_mixed')

    # -- 2. legality ----------------------------------------------------------
    bad, why = algo_legal('sgl_fp8_pb_wo', 5120, 248320, 'linear',
                          name='output.weight')
    ok(bad is False and 'weight_scale_inv' in why,
       'legality: FP8_PB_WO is refused for lm_head', why[:48])
    for q in ('sgl_fp8', 'sgl_fp8_pb_wo', 'sgl_mxfp8'):
        ok(algo_legal(q, 5120, 151552, 'embedding')[0] is False,
           f'legality: {q} is refused for role=embedding')
    ok(algo_legal('sgl_nvfp4', 5120, 151552, 'embedding')[0] is True,
       'legality: NVFP4 is the only quantised embedding SGLang can load')
    # ... and only on an arch whose model file passes a quant_config.
    ok(algo_legal('sgl_nvfp4', 5120, 151552, 'embedding',
                  name='token_embd.weight', arch=ARCH_QWEN3_5)[0] is True,
       'legality: qwen3_5 declares the packed embedding it actually serves')
    _gl, _gw = algo_legal('sgl_nvfp4', 5120, 151552, 'embedding',
                          name='token_embd.weight', arch=ARCH_GLM4_MOE)
    ok(_gl is False and 'quant_config' in _gw,
       'legality: glm4_moe refuses it, and the message says why', _gw[:64])
    ok(algo_legal(BF16, 5120, 151552, 'embedding', name='token_embd.weight',
                  arch=ARCH_GLM4_MOE)[0] is True,
       'legality: BF16 is what is left, on every arch')
    ok(plan_tensor('token_embd.weight', [5120, 151552], 5120 * 151552,
                   'sgl_nvfp4', ARCH_GLM4_MOE)[:2]
       == (BF16, 2 * 5120 * 151552),
       'bytes: so the glm4_moe size model prices the embedding at BF16')
    # The scope is the global table.  The MTP table is another module, built
    # only under speculation, so --nextn-optimization decides it, not the arch,
    # and the preset is where that pin is emitted.
    _mtp = 'blk.92.nextn.embed_tokens.weight'
    ok(map_tensor(_mtp, ARCH_GLM4_MOE)[1] == 'embedding'
       and _mtp not in embedding_tensors(ARCH_GLM4_MOE),
       'scope: the MTP table IS role=embedding and is NOT the global table')
    ok(embedding_algos_for(_mtp, ARCH_GLM4_MOE) == EMBEDDING_ALGOS_DEFAULT
       and embedding_algos_for('token_embd.weight', ARCH_GLM4_MOE) == (BF16,),
       'scope: so it keeps the format ceiling while token_embd gets the arch rule',
       ' '.join(embedding_algos_for(_mtp, ARCH_GLM4_MOE)))
    ok(algo_legal('sgl_nvfp4', 5120, 151552, 'embedding', name=_mtp,
                  arch=ARCH_GLM4_MOE)[0] is True,
       'scope: NVFP4 is legal for the MTP table on glm4_moe - the measured '
       'floor and mid checkpoints carry it and serve')
    ok(plan_tensor(_mtp, [5120, 151552], 5120 * 151552, 'sgl_nvfp4',
                   ARCH_GLM4_MOE)[1] < 2 * 5120 * 151552,
       'scope: and the size model prices it as NVFP4, not as BF16',
       f'{plan_tensor(_mtp, [5120, 151552], 5120 * 151552, "sgl_nvfp4", ARCH_GLM4_MOE)[1]:,} B')
    ok(embedding_algos_for(None, ARCH_GLM4_MOE) == (BF16,),
       'scope: a call that names no tensor gets the arch rule, the safe answer')
    ok(algo_legal('sgl_nvfp4', 5121, 4096, 'linear')[0] is False,
       'legality: NVFP4 needs K divisible by its group size')

    # -- 3. the embedding has no activation scale -----------------------------
    _lin = algo_bytes('sgl_nvfp4', 5120, 151552)
    _emb = plan_tensor('token_embd.weight', [5120, 151552], 5120 * 151552,
                       'sgl_nvfp4', ARCH_QWEN3_5)[1]
    ok(_lin - _emb == 4,
       'bytes: the NVFP4 embedding registers weight_scale_2 but no input_scale',
       f'{_lin:,} - {_emb:,} = 4 B')

    # -- 4. stacked experts ---------------------------------------------------
    one = algo_bytes('sgl_nvfp4', 5120, 1536)
    ok(algo_bytes('sgl_nvfp4', 5120, 1536, 160) == 160 * one,
       'bytes: a stacked-expert tensor costs exactly n_experts modules',
       f'160 x {one:,}')
    ok(algo_bytes('sgl_nvfp4', 5120, 1536, 160)
       - 160 * ((5120 * 1536) // 2 + 1536 * (5120 // 16)) == 160 * 8,
       'bytes: ...and carries 160 per-module scalar pairs, not one')

    # -- 4b. regex-compacted recipe expands to the SAME map as its raw twin ---
    # The published shape of every recipe in the suite is `quants_regex_merger`
    # output: one alternation per tensor family.  parse_recipe must expand it,
    # against the model's tensor set, to the byte-identical map its per-tensor
    # original parses to - or a rebuild silently drops ~99% of assignments to
    # BF16.  Proven here on a synthetic 12-block model so it can never regress,
    # with no map file and no checkpoint on disk.
    import tempfile as _tempfile
    _uni = (['token_embd.weight', 'output.weight', 'output_norm.weight']
            + [f'blk.{b}.ffn_down.weight' for b in range(12)]
            + [f'blk.{b}.ffn_up.weight' for b in range(12)]
            + [f'blk.{b}.attn_q.weight' for b in range(12)])
    _raw = (['^token_embd\\.weight$=sgl_bf16', '^output\\.weight$=sgl_bf16']
            + [f'^blk\\.{b}\\.ffn_down\\.weight$=sgl_nvfp4' for b in range(12)]
            + [f'^blk\\.{b}\\.ffn_up\\.weight$=sgl_fp8' for b in range(12)]
            # attn_q only on EVEN blocks; the odd ones are BF16 by ABSENCE and
            # must stay out of the map on both sides
            + [f'^blk\\.{b}\\.attn_q\\.weight$=sgl_nvfp4' for b in range(0, 12, 2)])
    _merged = ['^token_embd\\.weight$=sgl_bf16', '^output\\.weight$=sgl_bf16',
               '^blk\\.([0-9]|1[01])\\.ffn_down\\.weight$=sgl_nvfp4',
               '^blk\\.([0-9]|1[01])\\.ffn_up\\.weight$=sgl_fp8',
               '^blk\\.(0|2|4|6|8|10)\\.attn_q\\.weight$=sgl_nvfp4']
    with _tempfile.TemporaryDirectory() as _td:
        _rawp = os.path.join(_td, 'raw.recipe')
        _mrgp = os.path.join(_td, 'merged.recipe')
        open(_rawp, 'w').write('\n'.join(_raw) + '\n')
        open(_mrgp, 'w').write('\n'.join(_merged) + '\n')
        _rmap = parse_recipe(_rawp, _uni)
        _mmap = parse_recipe(_mrgp, _uni)
        ok(_rmap == _mmap and len(_rmap) == 32,
           'recipe: a merged recipe expands to the IDENTICAL tensor->qtype map '
           'as its raw per-tensor twin (0 differences)',
           f'{len(_rmap)} vs {len(_mmap)} entries, equal={_rmap == _mmap}')
        ok(all(_mmap[f'blk.{b}.attn_q.weight'] == 'sgl_nvfp4'
               for b in range(0, 12, 2))
           and not any(f'blk.{b}.attn_q.weight' in _mmap for b in range(1, 12, 2)),
           'recipe: BF16-by-absence survives the merge - odd-block attn_q, in '
           'neither recipe, is absent from both maps')
        # a raw recipe read the OLD way (no universe) is bit-for-bit unchanged,
        # and equals the expanded map because every literal name is in the set
        _rlit = parse_recipe(_rawp)
        ok(_rlit == _rmap,
           'recipe: a raw per-tensor recipe is unchanged with or without a '
           'tensor set (literal == regex-matching-one-name)')
        # the build path (recipe_to_quantized_layers) agrees on both forms
        _rl, _rp = recipe_to_quantized_layers(_rmap, ARCH_QWEN3_5)
        _ml, _mp = recipe_to_quantized_layers(_mmap, ARCH_QWEN3_5)
        ok(_rl == _ml and not _rp and not _mp,
           'recipe: recipe_to_quantized_layers yields the identical '
           'quantized_layers from the merged recipe as from the raw one')
        # a compacted line with NO tensor set to expand over is a LOUD error,
        # never the old silent collapse to one bogus alternation-named module
        try:
            parse_recipe(_mrgp)
            _raised = False
        except SystemExit:
            _raised = True
        ok(_raised,
           'recipe: a compacted recipe read with no tensor set is a hard error, '
           'not a silent collapse to BF16')
        # FIRST-match-wins, exactly as the GGUF downloader (its match loops all
        # break on the first pattern to hit a tensor)
        _fmw = ['^blk\\.([0-9]|1[01])\\.ffn_down\\.weight$=sgl_fp8',
                '^blk\\.0\\.ffn_down\\.weight$=sgl_nvfp4']
        _fmwp = os.path.join(_td, 'fmw.recipe')
        open(_fmwp, 'w').write('\n'.join(_fmw) + '\n')
        _fm = parse_recipe(_fmwp, _uni)
        ok(_fm['blk.0.ffn_down.weight'] == 'sgl_fp8',
           'recipe: FIRST-match-wins - the earlier broad pattern beats the later '
           'specific one, matching the GGUF build path')

    # -- 5. Qwen3.8-27B: a dense hybrid ---------------------------------------
    # -- the three ways a source states its architecture ---------------------
    for key, arch in sorted(ARCHS.items()):
        for cls in arch['hf_architectures']:
            ok(arch_from_hf_config({'architectures': [cls]}) == key,
               f'{key}: config.json architectures {cls!r} names it')
        for g in arch['gguf_arch']:
            ok(arch_from_gguf_arch(g) == key,
               f'{key}: GGUF general.architecture {g!r} names it')
    ok(arch_from_hf_config({'architectures': ['SomethingElseForCausalLM']}) is None
       and arch_from_gguf_arch('llama') is None,
       'an unregistered architecture is None, never a nearest guess')
    ok(arch_from_hf_config({'model_type': 'glm4_moe'}) == 'glm4_moe',
       'model_type settles it when exactly one arch claims that family')
    ok(detect_arch_hf(['model.language_model.layers.0.self_attn.q_proj.weight',
                       'model.language_model.layers.0.mlp.down_proj.weight',
                       'lm_head.weight']) == 'qwen3_5',
       'HF tensor names name the arch, and the shared lm_head does not vote')
    ok(detect_arch_hf(['model.layers.0.self_attn.q_proj.weight',
                       'model.layers.0.mlp.experts.7.up_proj.weight']) == 'glm4_moe',
       'HF tensor names: the stacked-expert arch is told apart from the dense one')
    ok(detect_arch_hf(['lm_head.weight']) is None,
       'HF tensor names: a global both archs declare decides nothing')

    if os.path.exists(_QWEN_MAP):
        rows = read_bf16_map(_QWEN_MAP)
        names = [r[0] for r in rows]
        ok(len(rows) == 851, 'qwen3_5: 851 tensors in the map', str(len(rows)))
        ok(all(map_tensor(n, ARCH_QWEN3_5) is not None for n in names),
           'qwen3_5: every tensor is named by the arch map')
        ok(detect_arch(names) == 'qwen3_5', 'qwen3_5: detected from the map alone')
        _, c = synthesise_map_lines(rows, 'sgl_bf16', ARCH_QWEN3_5)
        ok(c['bytes'] == 53_791_996_928,
           'qwen3_5: BF16 census is exact', f'{c["bytes"]:,}')
        _, c4 = synthesise_map_lines(rows, 'sgl_nvfp4', ARCH_QWEN3_5)
        ok(c4['bytes'] == 15_132_806_028,
           'qwen3_5: uniform-NVFP4 census is exact', f'{c4["bytes"]:,}')
        ok(len(fused_groups(names, ARCH_QWEN3_5)) == 176,
           'qwen3_5: 176 multi-shard fused groups')
        # round-trip: the map we synthesise re-reads to the bytes we wrote
        for algo in ('sgl_nvfp4', 'sgl_fp8', 'sgl_fp8_pb_wo', 'sgl_bf16'):
            lines, cen = synthesise_map_lines(rows, algo, ARCH_QWEN3_5)
            tot = sum(int(l.rsplit('bytes=', 1)[1]) for l in lines)
            ok(tot == cen['bytes'], f'qwen3_5: {algo} map round-trips its census')
    else:
        print(f'  skip qwen3_5 map cases ({_QWEN_MAP} absent)')

    # -- 6. GLM-4.7: a 160-expert MoE, and the published-index reproduction ----
    if os.path.exists(_GLM47_MAP):
        rows = read_bf16_map(_GLM47_MAP)
        names = [r[0] for r in rows]
        ok(len(rows) == 1761, 'glm4_moe: 1761 tensors in the map', str(len(rows)))
        ok(all(map_tensor(n, ARCH_GLM4_MOE) is not None for n in names),
           'glm4_moe: every tensor is named by the arch map')
        ok(detect_arch(names) == 'glm4_moe', 'glm4_moe: detected from the map alone')
        _, c = synthesise_map_lines(rows, 'sgl_bf16', ARCH_GLM4_MOE)
        # 716,675,582,592 B of weights + 28,800 B: the 90 routers'
        # `e_score_correction_bias` are F32 [160], not BF16 (see f32_tensors).
        ok(c['bytes'] == 716_675_611_392,
           'glm4_moe: BF16 census is exact, F32 router biases included',
           f'{c["bytes"]:,}')
        # one FusedMoE unit per layer: gate/up/down_exps share a group
        fg = fused_groups(names, ARCH_GLM4_MOE)
        moe = {g: v for g, v in fg.items() if g.startswith('experts.')}
        ok(len(moe) == 90 and all(len(v) == 3 for v in moe.values()),
           'glm4_moe: 90 MoE layers, each ONE fused unit of gate+up+down_exps',
           f'{len(moe)} groups')
        # THE ACCEPTANCE TEST: mirror Salyut1/GLM-4.7-NVFP4 exactly.
        tot, mods = 0, 0
        for name, shape, elems, _d, _s, _h in rows:
            if name.startswith(('blk.92.', 'nextn.')):
                continue          # every published NVFP4 drops the MTP layer
            algo = (BF16 if name in ('token_embd.weight', 'output.weight')
                    else 'sgl_nvfp4')
            eff, nb, _w = plan_tensor(name, shape, elems, algo, ARCH_GLM4_MOE)
            tot += nb
            if eff != BF16:
                mods += int(shape[2]) if len(shape) > 2 else 1
        published, kv_scales = 200_788_118_656, 184 * 4
        ok(mods == 43_364,
           'glm4_moe: 43,364 quantised modules, as published', f'{mods:,}')
        ok(tot + kv_scales == published,
           'glm4_moe: reproduces the PUBLISHED Salyut1/GLM-4.7-NVFP4 byte count '
           'exactly, once its 92 k_scale + 92 v_scale FP8 KV scalars are added',
           f'{tot:,} + {kv_scales} = {published:,}')
        ok(_gguf_suffix('blk.3.exp_probs_b.bias') in ARCH_GLM4_MOE['f32_tensors'],
           'glm4_moe: the router bias is declared F32, as the published file has it')
    else:
        print(f'  skip glm4_moe map cases ({_GLM47_MAP} absent)')

    # -- 6b. the MoE quantized_layers map the WRITER consumes -----------------
    # Metadata-level, no shards: this is what SGLang's
    # ModelOptMixedPrecisionConfig._resolve_quant_algo() looks a layer up in,
    # and three of its properties are load-bearing on a MoE.
    if os.path.exists(_GLM47_MAP):
        rows = read_bf16_map(_GLM47_MAP)
        rec = {}
        for name, shape, elems, _d, _s, _h in rows:
            info = map_tensor(name, ARCH_GLM4_MOE)
            _p, role, _f, _b = info
            if role == 'other' or len(shape) < 2:
                rec[name] = BF16
            elif name in ('token_embd.weight', 'output.weight'):
                rec[name] = BF16
            elif '_exps' in name:
                rec[name] = 'sgl_fp8'          # promote the experts, to tell them apart
            else:
                rec[name] = 'sgl_nvfp4'
        layers, problems = recipe_to_quantized_layers(rec, ARCH_GLM4_MOE)
        ok(not problems, 'glm4_moe: the recipe converts with no problems',
           str(problems[:1]))
        moe_keys = [k for k in layers if k.endswith('mlp.experts')]
        ok(len(moe_keys) == 90,
           'glm4_moe: ONE quantized_layers key per FusedMoE, one per MoE layer',
           f'{len(moe_keys)} keys')
        ok(not [k for k in layers if '.experts.' in k],
           'glm4_moe: NO per-expert keys - they would let _resolve_quant_algo\'s '
           'prefix fallback pick one arbitrarily and silently')
        ok(all(layers[k]['quant_algo'] == 'FP8' for k in moe_keys),
           'glm4_moe: the whole MoE unit carries one algo, as SGLang requires')
        ok(len([k for k in layers if k.endswith('self_attn.qkv_proj')]) == 93,
           'glm4_moe: attention is named FUSED (Glm4MoeForCausalLM declares no '
           'packed_modules_mapping, so unfused names would resolve to None)')
        ok(not [k for k in layers if k.endswith('mlp.gate')],
           'glm4_moe: the router is absent from quantized_layers - absence IS '
           'the BF16 encoding')
        ok('lm_head' not in layers and 'model.embed_tokens' not in layers,
           'glm4_moe: a BF16 head and embedding are encoded by absence too')

    # -- 6c. the shared-expert weight_block_size trap, and where it is fixed --
    # `ffn_down_shexp` must ride with `gate_up` in the ASSIGNER's harmonisation
    # (glm4_moe.py:485-505) and must NOT join it in the WRITER's fused groups
    # (SGLang's down_proj is its own module).  Two tables, two answers, and this
    # is the test that keeps them apart.
    _shexp = ['blk.%d.ffn_%s_shexp.weight' % (b, s)
              for b in (1, 2) for s in ('gate', 'up', 'down')]
    _h = harmonize_argument(_shexp, ARCH_GLM4_MOE)
    ok(len(_h) == 1 and 'ffn_down_shexp' in _h[0]
       and 'ffn_gate_shexp' in _h[0] and 'ffn_up_shexp' in _h[0],
       'glm4_moe: harmonize joins ffn_down_shexp to the shared-expert group '
       '(FP8_PB_WO gate_up + NVFP4 down is an AssertionError at load)',
       str(_h))
    ok(all(len(v) == 2 and not any('down' in m for m in v)
           for g, v in fused_groups(_shexp, ARCH_GLM4_MOE).items()),
       'glm4_moe: fused_groups does NOT join it - down_proj is its own module '
       'and the writer must keep treating it as one')
    ok(not any('shexp' in g for g in harmonize_argument(
        ['blk.1.ffn_gate.weight', 'blk.1.ffn_up.weight'], ARCH_QWEN3_5)),
       'harmonize_join is per-arch: qwen3_5 declares none and gains none')

    # -- 6d. the embedding, and the one type a draft model can share ---------
    for key, arch in sorted(ARCHS.items()):
        emb = embedding_tensors(arch)
        ok(len(emb) == 1,
           f'{key}: exactly one GLOBAL embedding, found from the role column',
           ' '.join(emb))
        ok(all(arch['globals'][n][1] == 'embedding' for n in emb),
           f'{key}: and it really is declared role=embedding')
    ok(embedding_tensors(ARCH_GLM4_MOE) == ['token_embd.weight'],
       'glm4_moe: the MTP layer\'s OWN embedding is not the shared one')
    ok(dense_embedding_algo(['sgl_nvfp4', 'sgl_fp8', 'sgl_fp8_pb_wo', BF16]) == BF16,
       'pool: sgl_bf16 is the only DENSE embedding type in the shipped pool')
    ok(dense_embedding_algo(['sgl_nvfp4', 'sgl_fp8']) is None,
       'pool: a pool without sgl_bf16 has no NEXTN-compatible embedding at all')
    for q in ('sgl_fp8', 'sgl_fp8_pb_wo', 'sgl_mxfp8', 'sgl_int4_g128'):
        ok(algo_legal(q, 5120, 151552, 'embedding')[0] is False,
           f'pool: {q} has no embedding method at all (not merely a packed one)')
    ok(algo_legal('sgl_nvfp4', 5120, 151552, 'embedding')[0] is True
       and dense_embedding_algo(['sgl_nvfp4']) is None,
       'pool: sgl_nvfp4 IS a legal embedding and is NOT a dense one - which is '
       'the whole of the NEXTN/EAGLE constraint')
    for key, arch in sorted(ARCHS.items()):
        ok(set(embedding_algos(arch)) <= set(EMBEDDING_ALGOS_DEFAULT)
           and BF16 in embedding_algos(arch),
           f'{key}: declares embedding_algos, a subset of the format ceiling '
           f'that always keeps BF16', ' '.join(embedding_algos(arch)))
    ok(embedding_algos(ARCH_GLM4_MOE) == (BF16,)
       and embedding_algos(ARCH_QWEN3_5) == (BF16, 'sgl_nvfp4'),
       'arch: glm4_moe cannot pack its embedding and qwen3_5 can - the one '
       'thing that differs between the two recipes\' token_embd lines')
    ok(embedding_algos() == EMBEDDING_ALGOS_DEFAULT,
       'arch: an arch that declares nothing keeps the old behaviour')
    for key, arch in sorted(ARCHS.items()):
        _per_layer = [n for n, (_h, role, _f) in arch['tensors'].items()
                      if role == 'embedding']
        ok(all(embedding_algos_for(f'blk.0.{n}', arch) == EMBEDDING_ALGOS_DEFAULT
               for n in _per_layer),
           f'{key}: the declaration binds the global table only - per-layer '
           f'embeddings keep the format ceiling',
           ' '.join(_per_layer) or '(none)')

    # -- 6e. MTP / NEXTN tensors by name, in either convention ---------------
    ok(nextn_tensor_names(['blk.92.nextn.eh_proj.weight', 'blk.1.attn_q.weight'])
       == ['blk.92.nextn.eh_proj.weight'],
       'nextn: the GGUF convention (blk.N.nextn.*) is recognised')
    ok(nextn_tensor_names(['model.layers.92.mtp.enorm.weight',
                           'mtp.layers.0.eh_proj.weight']) == [
           'model.layers.92.mtp.enorm.weight', 'mtp.layers.0.eh_proj.weight'],
       'nextn: both HF conventions (…mtp.… and mtp.layers.N.…) are recognised')
    ok(nextn_tensor_names(['nextnorm.weight', 'blk.1.mtparam.weight']) == [],
       'nextn: a name merely CONTAINING the letters does not match')

    # -- 7. cross-arch: the same code path, no per-model branches -------------
    for key, arch in sorted(ARCHS.items()):
        roles = {r for _h2, r, _f in list(arch['tensors'].values())
                 + list(arch['globals'].values())}
        ok(roles <= {'linear', 'embedding', 'other'},
           f'{key}: every declared role is one of linear/embedding/other',
           ' '.join(sorted(roles)))
        ok(isinstance(arch.get('f32_tensors', set()), (set, frozenset, tuple, list)),
           f'{key}: f32_tensors (if declared) is a container of GGUF suffixes')
        ok(all(v in {f for _h2, _r, f in arch['tensors'].values()}
               for v in (arch.get('harmonize_join') or {}).values()),
           f'{key}: every harmonize_join target names a real fused group')

    # ---- 6f. artefacts against artefacts, on a synthetic module -----------
    # The accounting rule the Salyut1 residual taught us, checked without
    # needing any checkpoint on disk: `algo_bytes()` prices the quantisation of
    # a weight, so a bias or a KV-cache scale must be counted separately or the
    # byte check is comparing two different things.
    ok('bias' in _HF_COMPANION_SUFFIXES and 'bias' not in _HF_QUANT_ARTEFACT_SUFFIXES,
       'smoke: a bias belongs TO a module and is NOT a quantisation artefact')
    for _s in ('k_scale', 'v_scale', 'e_score_correction_bias'):
        ok(_s not in _HF_QUANT_ARTEFACT_SUFFIXES,
           f'smoke: {_s} is a model/runtime scalar, not something algo_bytes models')
    for _s in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale'):
        ok(_s in _HF_QUANT_ARTEFACT_SUFFIXES,
           f'smoke: {_s} IS one of the artefacts algo_bytes prices')
    ok(_HF_QUANT_ARTEFACT_SUFFIXES <= (_HF_COMPANION_SUFFIXES | {'weight',
       'weight_packed', 'weight_shape'}),
       'smoke: every modelled artefact is a recognised companion (or the weight)')
    # NVFP4 on GLM-4.7's k_proj, by hand.  The packed weight is U8 [1024, 2560],
    # i.e. [N, K/2], so the LOGICAL shape is K=5120, N=1024 - and the artefacts
    # then come to exactly what the published checkpoint holds:
    #   weight       U8      [1024, 2560]  2,621,440
    #   weight_scale F8_E4M3 [1024,  320]    327,680
    #   weight_scale_2 + input_scale, F32 scalars     8
    ok(algo_bytes('sgl_nvfp4', 5120, 1024) == 2_949_128,
       'smoke: NVFP4 artefacts for a [N=1024, K=5120] weight = 2,949,128 B '
       '(Salyut1/GLM-4.7-NVFP4 k_proj, measured)',
       f'{algo_bytes("sgl_nvfp4", 5120, 1024):,}')
    ok(2_949_128 + 1024 * 2 + 4 == 2_951_180,
       'smoke: and the 2,052 B that module ALSO holds - the on-disk total this '
       'check used to fail on - is a BF16 bias [1024] plus one F32 k_scale')
    ok(algo_bytes('sgl_nvfp4', 5120, 12288) == 35_389_448
       and 35_389_448 + 12288 * 2 == 35_414_024,
       'smoke: same on q_proj, where the extra 24,576 B is the bias alone '
       '(no KV-cache scale on q)')

    # ---- TABLE-FREE, on the checkpoints the docs claim exactness FOR -------
    #
    # The two blocks above prove the size model on the two REGISTERED
    # architectures.  This one proves it with no ARCHS entry, no bf16 map and no
    # calibration - which is the only way to show that adding a model is a data
    # problem and not a code problem.
    #
    # IT ASSERTS ON A NAMED LIST, NOT ON WHATEVER IS IN THE CACHE.  A selftest
    # whose result depends on what somebody downloaded this morning is not a
    # test of this code: Salyut1/GLM-4.7-NVFP4 landing in the cache turned it red
    # for a property of Salyut1's file (attention biases), and a machine is
    # free to hold checkpoints this branch never chose.  So the assertion is on
    # HF_EXACT_REPOS - the ones docs/sglang.md quotes byte-exactness for - and
    # the rest of the cache is an OPT-IN report: `sglang_native.py --smoke-cache`.
    # A skip, not a failure, when a named repo is absent: the invariant is about
    # the code, and a machine with no checkpoints cannot exercise it.
    found = 0
    for repo in HF_EXACT_REPOS:
        snap = _hf_snapshot(repo)
        if snap is None:
            print(f'  skip table-free HF smoke for {repo} (not in the local cache)')
            continue
        found += 1
        nprob, st = smoke_hf_checkpoint(snap, verbose=False)
        ok(nprob == 0 and st['checked'] > 0
           and st['modelled_bytes'] == st['disk_bytes'],
           f'table-free: {repo} classified and re-weighed with NO arch table, '
           f'to the byte',
           f'{st["checked"]:,} modules, {st["exact"]:,} exact, '
           f'residual {st["modelled_bytes"] - st["disk_bytes"]:,} B'
           + (f', carrying {st["carried_bytes"]:,} B of non-artefact tensors'
              if st['carried_bytes'] else ''))
    if not found:
        print('  skip table-free HF smoke (none of HF_EXACT_REPOS is cached)')
    _others = [r for r, _s in hf_cached_repos() if r not in HF_EXACT_REPOS]
    if _others:
        print(f'  note: {len(_others)} other checkpoint(s) in the local cache are '
              f'NOT asserted on; run `sglang_native.py --smoke-cache` to weigh them')

    print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='SGLang modelopt_mixed size model, name map and fused-group '
                    'constraint for the GGUF Tool Suite.')
    ap.add_argument('--bf16-map', help='tensors.bf16.map of the BF16 GGUF split')
    ap.add_argument('--arch', default='qwen3_5', choices=sorted(ARCHS))
    ap.add_argument('--emit-maps', metavar='DIR',
                    help='write tensors.<algo>.map for every algo into DIR')
    ap.add_argument('--algos', nargs='*', default=list(SGLANG_POOL_ORDER))
    ap.add_argument('--budget', metavar='BYTES', type=int,
                    help='print the --gpu-tensors-max-size to use for this '
                         'total checkpoint byte target')
    ap.add_argument('--pins', action='store_true',
                    help='print the --gpu-assign-tensors value for the tensors '
                         'this arch advises pinning to BF16, and why')
    ap.add_argument('--harmonize', action='store_true',
                    help='print the --harmonize-tensors value expressing the '
                         'fused-module constraint')
    ap.add_argument('--smoke-hf', metavar='MODELDIR',
                    help=('Table-free smoke over a published HF checkpoint: give every '
                          'tensor a role and reproduce every quantised module\'s bytes '
                          'from its shape alone, with NO ARCHS entry for the model. '
                          'This is how you find out whether a new architecture needs '
                          'data or code.'))
    ap.add_argument('--verify-against', metavar='MODELDIR',
                    help='rebuild a reference ModelOpt checkpoint byte total')
    ap.add_argument('--smoke-cache', action='store_true',
                    help=('OPT-IN: run --smoke-hf over EVERY checkpoint in the '
                          'local HF cache and report what each one weighs, '
                          'including the bytes it carries that are not '
                          'quantisation artefacts (biases, KV-cache scales). '
                          'Not part of --selftest: what is in a cache is not a '
                          'property of this code.'))
    ap.add_argument('--selftest', action='store_true',
                    help='run the size-model and name-map invariants and exit')
    ap.add_argument('--recipe', help='recipe to convert to quantized_layers')
    ap.add_argument('--out-config', help='where to write hf_quant_config.json')
    a = ap.parse_args(argv)
    arch = ARCHS[a.arch]

    if a.selftest:
        return selftest()

    if a.smoke_cache:
        return smoke_cache(verbose=False)

    if a.recipe:
        # A published recipe is regex-compacted; expand it against the model's
        # tensor set (--bf16-map) exactly as the GGUF path does.  Without a map a
        # raw per-tensor recipe still converts, and a compacted one errors loudly.
        universe = ([r[0] for r in read_bf16_map(a.bf16_map)]
                    if a.bf16_map else None)
        rec = parse_recipe(a.recipe, universe)
        layers, problems = recipe_to_quantized_layers(rec, arch)
        cfg = build_hf_quant_config(layers)
        if a.out_config:
            with open(a.out_config, 'w', encoding='utf-8') as fh:
                json.dump(cfg, fh, indent=4)
            print(f'wrote {a.out_config}: {len(layers)} quantized_layers entries')
        else:
            print(json.dumps(cfg, indent=4))
        for p in problems:
            print(f'[Error] {p}', file=sys.stderr)
        return 1 if problems else 0

    if a.smoke_hf:
        # Needs no bf16 map and no arch: that is the whole point of it.
        nprob, _st = smoke_hf_checkpoint(a.smoke_hf)
        print('SMOKE ' + ('PASS' if nprob == 0 else f'FAIL ({nprob} problem(s))'))
        return 0 if nprob == 0 else 1

    if not a.bf16_map:
        ap.error('--bf16-map is required for every mode except --recipe '
                 'and --smoke-hf')

    if a.verify_against:
        return verify_against(a.bf16_map, a.verify_against, arch)

    if a.pins:
        for pat, why in PIN_BF16_ADVISORY.get(a.arch, []):
            print(f"'{pat}={BF16}'      # {why}")
        return 0

    if a.harmonize:
        for grp in harmonize_argument([r[0] for r in read_bf16_map(a.bf16_map)], arch):
            print(f"'{grp}'")
        return 0

    if a.budget is not None:
        fixed, cnt = fixed_bf16_bytes(a.bf16_map, arch)
        print(f'total checkpoint target      : {a.budget:,} B')
        print(f'fixed BF16 (norms/1-D) x{cnt:<4}: {fixed:,} B')
        print(f'assignable budget            : {a.budget - fixed:,} B')
        print(f'--gpu-tensors-max-size {a.budget - fixed}B')
        return 0

    if a.emit_maps:
        os.makedirs(a.emit_maps, exist_ok=True)
        for algo in a.algos:
            out = os.path.join(a.emit_maps, f'tensors.{algo}.map')
            c = synthesise_map(a.bf16_map, algo, out, arch)
            fb = len(c['fallbacks'])
            print(f"{algo:14s} {c['tensors']:4d} tensors  {c['bytes']:>14,} B  "
                  f"{c['bytes']*8/c['elements']:7.4f} bpw overall  "
                  f"{fb} BF16 fallback(s)")
        return 0

    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
