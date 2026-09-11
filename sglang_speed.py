#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_speed.py models prompt and generation speed for a  **#
#** recipe so the assigner can spend a size budget wisely.    **#
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
#** Copyright © 2026 - Thireus. ᵦₑₙ𝒸ₕₘₐᵣₖₛ ₘₐᵧ ᵥₐᵣᵧ, ₘᵢₙₑ 𝒹ᵢ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_speed.py - the TWO-SIDED speed model: prefill is a FLOP budget, decode is
a byte budget, and they are not the same constraint.

WHY THIS FILE EXISTS
--------------------
The first `--speed-profile sglang` (quant_assign.py:apply_speed_profile) had
exactly ONE speed knob: `--speed-fast-fraction`, the share of matmul weight
elements that must sit on the W4A4 tensor core.  That quantity really does
predict prefill throughput - 48.5 prefill tok/s per percentage point at chat-c1,
measured, with two withheld checkpoints landing on the line to within 2 %.

But it is a ONE-SIDED model, and the physics has two sides:

  PREFILL is compute-bound.  What decides the cost is the GEMM KERNEL CLASS:
      W4A4 (NVFP4)             native CUTLASS FP4 MMA        cost 1.00
      W8A8 (FP8/MXFP8/blk)     sm_120 rowwise / block FP8    cost ~2.00
      *every weight-only fmt*  dequantise -> 16-bit MMA      cost ~4.00
    That last row is the whole point.  W4A16 INT4 (AWQ/GPTQ/compressed-tensors),
    W4A16_NVFP4 (marlin), W8A16, and any 2/3-bit type all dequantise into
    registers and then run a BF16-class MMA.  They are 4-bit ON DISK and 16-bit
    IN THE GEMM.  Measured here: a uniform bf16 server prefills at 4454.5 tok/s
    against RadixArk's NVFP4 checkpoint at 9745.1 in the same measurement window,
    i.e. a weight-only format gives back the whole FP4 speed-up in prefill.

  DECODE is bandwidth-bound.  What decides the cost is BYTES STREAMED PER
    TOKEN.  There a weight-only low-bit format is FASTER, not slower, because it
    moves fewer bytes; the dequantise cost hides under the memory stall.

So a single "fast fraction" cannot express what the owner asked for
(2026-09-05): *"is it possible to go further by adding supported sglang quants
... without damaging speed ... maybe you can identify which tensors are less
sensitive to speed degradation"*.  Answering that needs a per-tensor PREFILL
FLOP share, a per-tensor DECODE BYTE share, a per-FORMAT kernel cost, and two
budgets - one per side.

WHAT IS IN HERE
---------------
  1. PREFILL_COST  - the per-format kernel cost table, relative to W4A4 = 1.0,
     read out of SGLang's own sm_120 dispatch and CONFIRMED against six
     measured checkpoints (SS4 below).
  2. multiplicity() - how many times per token a tensor's GEMM actually runs.
     1.0 for a dense per-token projection; 0.0 for an embedding table (a
     gather); top_k/n_experts for a routed MoE expert; a declared value for
     lm_head.  This is the ONLY thing that makes the MoE case different, and it
     is the whole of the owner's "expert tensors ... are used less often".
  3. prefill_index() / decode_bytes() - the two budgeted quantities.
  4. MEASURED_W  - the calibration, fitted here from six measured checkpoints
     served across six workload cells.  It converts a change in format shares into a
     predicted change in throughput, so a budget can be stated as
     "no worse than 95 % of RadixArk" and mean something.
  5. downgrade_walk() - bytes saved vs predicted prefill loss, which is the
     table the owner actually wants to read.  (Still aliased `frontier()`.)

EVERYTHING HERE IS CPU-ONLY AND READ-ONLY.  It computes over tensor SHAPES
(a tensors.bf16.map, or an architecture config) and never opens a weight file.

STATUS OF THE NEW FORMATS (measured on this box, sglang 0.5.19.dev932+gd06f3bec8)
--------------------------------------------------------------------------------
The owner asked whether "Q quants or other sglang supported quants" can join a
recipe.  The answer is format-container-shaped and it is in `CANDIDATES` below:
`modelopt_mixed` dispatches EXACTLY five algos (modelopt_quant.py:1013-1050) and
none of them is an INT4 or a k-quant, so nothing new can be added to an existing
recipe without changing container.  `compressed-tensors` is the container that
can (it resolves a scheme per layer from N `config_groups`, each with its own
`targets` regex list AND its own `format` -
compressed_tensors.py:342-393, :681-780), and its W4A16 lane reaches marlin on
sm_120.  That is a real 4.125-4.25 bpw rung - at BF16-class prefill cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple


# =============================================================================
# 1. THE PER-FORMAT PREFILL KERNEL COST TABLE
# =============================================================================
#
# Relative per-FLOP cost of the GEMM a format actually executes, W4A4 = 1.00.
# Sources: the sm_120 per-method verdict table, read out of the shipped .so's
# instantiation lists and SGLang's backend selectors, and
# SGLANG_ALGOS[...]['speed_cost'] in sglang_native.py, which this table extends
# rather than replaces (the shared entries are identical by assertion, see
# selftest()).
#
# THE ONLY STRUCTURE THAT MATTERS: there are three classes, not six numbers.
#   W4A4  = 1.00   the FP4 tensor core            (NVFP4 only)
#   W8A8  ~ 2.00   the FP8 tensor core            (FP8, FP8_PB_WO, MXFP8)
#   W16A16~ 4.00   dequantise-then-16-bit-MMA     (bf16 AND every weight-only
#                                                   format, at ANY bit width)
# A 2-bit weight-only format sits in the SAME class as bf16.  Bits on disk do
# not buy prefill speed; only the activation width does.

COST_W4A4 = 1.00
COST_W8A8 = 2.00
COST_W8A8_BLOCK = 2.10          # same tensor core, block-scale epilogue
COST_W16A16 = 4.00              # bf16, and every dequantise-first path

# The pool `--budget-from-reference` assumes when none is named: the four rungs
# a modelopt_mixed checkpoint of this family is normally assigned over.  It is a
# CLI default and nothing else reads it - the assigner always names its own pool
# with --gpu-quants.
DEFAULT_POOL = ('sgl_nvfp4', 'sgl_fp8', 'sgl_fp8_pb_wo', 'sgl_bf16')

PREFILL_COST: Dict[str, float] = {
    'sgl_nvfp4':      COST_W4A4,
    'sgl_fp8':        COST_W8A8,
    'sgl_mxfp8':      COST_W8A8,
    'sgl_fp8_pb_wo':  COST_W8A8_BLOCK,
    'sgl_nvfp4a16':   COST_W16A16,      # marlin: 4-bit on disk, bf16 in the MMA
    'sgl_bf16':       COST_W16A16,
    # the candidate formats of SS1's CANDIDATES table, so a frontier can be
    # computed for them.  They are NOT in SGLANG_POOL_ORDER and no recipe can
    # emit them into a modelopt_mixed checkpoint - see CANDIDATES[...]['container'].
    'ct_w4a16_g128':  COST_W16A16,
    'ct_w4a16_g64':   COST_W16A16,
    'ct_w4a16_g128_zp': COST_W16A16,
    'ct_w8a16':       COST_W16A16,
    'moe_mxfp4':      1.15,
    # the compressed-tensors INT4 rung, now REGISTERED in sglang_native.py and
    # MEASURED end to end on 2026-09-05.
    'sgl_int4_g128':  COST_W16A16,
}


# -----------------------------------------------------------------------------
# 1b. THE SAME TABLE, MEASURED on 2026-09-05 -- and it is NOT 4.00
# -----------------------------------------------------------------------------
#
# The a-priori table above puts every weight-only format in one bucket with
# bf16 at 4.00.  Five uniform servers, six cells each, say that is structurally
# right and numerically ~15 % pessimistic: marlin's W4A16 path is REAL work
# cheaper than a bf16 GEMM because it streams a quarter of the bytes into the
# same MMA, so the class has two rungs, not one.
#
#   chat-c1   nvfp4 1.000 | fp8 2.123 | nvfp4a16 3.542 | int4_g128 3.556 | bf16 4.247
#
# The two utterly different 4-bit weight-only formats land 0.4 % apart, which is
# the sharpest confirmation available that THE KERNEL CLASS DECIDES PREFILL AND
# THE WEIGHT FORMAT DOES NOT.  Anchored on k(bf16) = 2 k(fp8-class), the sm_120
# dense tensor-core ladder; hold-out residuals under 3.2 % on the three mixed
# checkpoints withheld from the fit.  These constants are the OUTPUT of that
# fit: re-fit them from a new calibration run, do not hand-edit them.
MEASURED_K: Dict[str, Dict[str, float]] = {
    'chat-c1': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 2.1232,
        'sgl_bf16': 4.2465,
        'sgl_nvfp4a16': 3.5421,
        'sgl_int4_g128': 3.5561,
        'sgl_fp8': 2.0221,
        'sgl_mxfp8': 2.0221,
        'ct_w4a16_g128': 3.5561,
        'ct_w4a16_g64': 3.5561,
        'ct_w4a16_g128_zp': 3.5561,
        'ct_w8a16': 3.5561
    },
    'chat-c8': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 2.2516,
        'sgl_bf16': 4.5033,
        'sgl_nvfp4a16': 3.8924,
        'sgl_int4_g128': 3.8951,
        'sgl_fp8': 2.1444,
        'sgl_mxfp8': 2.1444,
        'ct_w4a16_g128': 3.8951,
        'ct_w4a16_g64': 3.8951,
        'ct_w4a16_g128_zp': 3.8951,
        'ct_w8a16': 3.8951
    },
    'coding-c1': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 2.9874,
        'sgl_bf16': 5.9749,
        'sgl_nvfp4a16': 5.8786,
        'sgl_int4_g128': 5.8174,
        'sgl_fp8': 2.8451,
        'sgl_mxfp8': 2.8451,
        'ct_w4a16_g128': 5.8174,
        'ct_w4a16_g64': 5.8174,
        'ct_w4a16_g128_zp': 5.8174,
        'ct_w8a16': 5.8174
    },
    'coding-c8': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 2.5907,
        'sgl_bf16': 5.1813,
        'sgl_nvfp4a16': 4.7336,
        'sgl_int4_g128': 4.7189,
        'sgl_fp8': 2.4673,
        'sgl_mxfp8': 2.4673,
        'ct_w4a16_g128': 4.7189,
        'ct_w4a16_g64': 4.7189,
        'ct_w4a16_g128_zp': 4.7189,
        'ct_w8a16': 4.7189
    },
    'document-c1': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 3.1906,
        'sgl_bf16': 6.3813,
        'sgl_nvfp4a16': 6.2774,
        'sgl_int4_g128': 6.2487,
        'sgl_fp8': 3.0387,
        'sgl_mxfp8': 3.0387,
        'ct_w4a16_g128': 6.2487,
        'ct_w4a16_g64': 6.2487,
        'ct_w4a16_g128_zp': 6.2487,
        'ct_w8a16': 6.2487
    },
    'document-c8': {
        'sgl_nvfp4': 1.0,
        'sgl_fp8_pb_wo': 1.41,
        'sgl_bf16': 2.8201,
        'sgl_nvfp4a16': 1.9765,
        'sgl_int4_g128': 1.9727,
        'sgl_fp8': 1.3429,
        'sgl_mxfp8': 1.3429,
        'ct_w4a16_g128': 1.9727,
        'ct_w4a16_g64': 1.9727,
        'ct_w4a16_g128_zp': 1.9727,
        'ct_w8a16': 1.9727
    }
}

# T(cell) per token = a + b * P, with P the FLOP-weighted mean k above.
# a is the part of prefill that no weight format can touch (attention, norms,
# sampling, launch overhead); a/(a+b) is printed as non_gemm_share in the fit.
# document-c8 fits a NEGATIVE a: the uniform-BF16 server is capacity-limited
# there (4968 -> 3521 tok/s from c1 to c8) and the cell is not usable.
MEASURED_AB: Dict[str, tuple] = {
    'chat-c1': [
        47.2319,
        42.1542
    ],
    'chat-c8': [
        34.7968,
        35.3838
    ],
    'coding-c1': [
        60.7035,
        20.6071
    ],
    'coding-c8': [
        45.222,
        27.2649
    ],
    'document-c1': [
        76.881,
        19.4096
    ],
    'document-c8': [
        -13.7492,
        105.3537
    ]
}

# Decode is bandwidth-bound: the cost of a format at decode time is its BYTES,
# which the size model already knows exactly.  There is no second table - that
# is the point of the two-sided model.  The one thing decode needs and prefill
# does not is `decode_mult` (SS2): which weights are actually read per token.


# -----------------------------------------------------------------------------
# The candidate formats the owner asked about, with the verdict for each.
# bpw is the on-disk cost; `container` is the ONLY thing that decides whether a
# per-tensor recipe can name it at all.
# -----------------------------------------------------------------------------
CANDIDATES: Dict[str, dict] = {
    'sgl_nvfp4': dict(
        bpw=4.5, cost=COST_W4A4, container='modelopt_mixed', status='in use',
        note='W4A4, block 16.  The floor of the current pool.'),
    'sgl_fp8_pb_wo': dict(
        bpw=8.0, cost=COST_W8A8_BLOCK, container='modelopt_mixed', status='in use',
        note='W8A8, 128x128 block weight scales, dynamic activation scales.'),
    'sgl_nvfp4a16': dict(
        bpw=4.5, cost=COST_W16A16, container='modelopt_mixed', status='available, useless for size',
        note='W4A16_NVFP4 via marlin.  EXACTLY the same bytes as sgl_nvfp4 and 4x '
             'the prefill cost, so it can only ever be bought for quality (no A4 '
             'error), never for size.  It is the one extra rung modelopt_mixed '
             'already offers and it does not move the byte target at all.'),
    'ct_w4a16_g128': dict(
        bpw=4.125, cost=COST_W16A16, container='compressed-tensors', status='loadable, NOT in modelopt_mixed',
        note='CompressedTensorsWNA16, uint4b8 (GPTQ-style symmetric), group 128, '
             'fp16 group scales -> 4 + 16/128 = 4.125 bpw.  apply_gptq_marlin_linear '
             'on sm_120 (marlin_utils.py:84-117 gates on capability >= 80).  '
             'BF16-class prefill.'),
    'ct_w4a16_g64': dict(
        bpw=4.25, cost=COST_W16A16, container='compressed-tensors', status='loadable, NOT in modelopt_mixed',
        note='Same lane at group 64: 4 + 16/64 = 4.25 bpw.  This is the "4.25 bpw '
             'W4A16" the frontier table below is computed for.'),
    'ct_w4a16_g128_zp': dict(
        bpw=4.15625, cost=COST_W16A16, container='compressed-tensors', status='loadable, NOT in modelopt_mixed',
        note='AWQ-style uint4 with a 4-bit zero point: 4 + 16/128 + 4/128.'),
    'ct_w8a16': dict(
        bpw=8.125, cost=COST_W16A16, container='compressed-tensors', status='loadable, pointless',
        note='CompressedTensorsW8A16Fp8 - MORE bytes than sgl_fp8 AND 2x its '
             'prefill cost.  Dominated on both axes; listed so nobody proposes it.'),
    'ct_w8a8_int8': dict(
        bpw=8.0, cost=float('inf'), container='compressed-tensors', status='LOADS THEN THROWS',
        note='cutlass_int8_scaled_mm has Sm75/Sm80/Sm90 instantiations and no '
             'sm120 instantiation at all.  Use W8A8-FP8.'),
    'moe_mxfp4': dict(
        bpw=4.25, cost=1.15, container='mxfp4 (MoE only)', status='MoE only, dense raises',
        note='MXFP4 weights x MXFP8 activations, native cutlass_sm120 W4A8 '
             '(mxfp4.py:425-426, :1289, :1379).  4 + 8/32 = 4.25 bpw at a '
             'W4A8 kernel, i.e. 0.25 bpw UNDER NVFP4 at near-FP4 speed - the '
             'only sub-4.5 rung on this box that is not weight-only.  '
             'mxfp4.py:379 raises "Mxfp4 attention layer is not implemented" for '
             'a dense linear, so it exists only for a MoE FusedMoE - which is '
             'exactly where the production model keeps 97 % of its weights.  '
             'cost 1.15 is a PLACEHOLDER: unmeasured, listed as a measurement '
             'target, not as a fact.'),
    'gguf_q_any': dict(
        bpw=None, cost=COST_W16A16, container='gguf (whole file)', status='different container entirely',
        note='SGLang has a gguf loader, so the "Q quants" the owner named do '
             'load - but as a WHOLE-MODEL GGUF, never as one tensor inside a '
             'safetensors recipe, and choosing it gives up NVFP4 (and the FP4 '
             'tensor core) for the entire model.  k-quants get MMQ (int8 MMA, '
             'the 32-deep instruction); IQ* types have no MMQ and dequantise.'),
    'ct_wna16_3bit': dict(
        bpw=None, cost=None, container='-', status='DOES NOT EXIST',
        note='WNA16_SUPPORTED_TYPES_MAP = {4: uint4b8, 8: uint8b128} '
             '(compressed_tensors_wNa16.py:55-60) and marlin returns only '
             'uint4/uint4b8/uint8b128/e4m3/e2m1 '
             '(marlin_utils.py:109-117).  There is NO 2-bit or 3-bit dense lane '
             'in SGLang on sm_120 outside the gguf loader.'),
}


# =============================================================================
# 2. MULTIPLICITY - how often a tensor's GEMM actually runs
# =============================================================================
#
# This is the generalisation the one-sided fast-fraction model was missing, and
# it is the whole of the owner's "expert tensors ... are used less often".
#
# prefill_mult(t)  x elements(t)  = the tensor's share of prompt-processing
#                                   matmul FLOPs (the M and the factor 2 cancel).
# decode_mult(t)   x bytes(t, q)  = the tensor's share of per-token weight
#                                   traffic at decode.
#
#   dense per-token projection   prefill 1.0            decode 1.0
#   embedding table              prefill 0.0 (gather)   decode ~0.0 (one row)
#   lm_head                      prefill LMHEAD_TOKENS  decode 1.0
#   routed MoE expert            prefill top_k/n_exp    decode top_k/n_exp @ bs=1
#   shared MoE expert            prefill 1.0            decode 1.0
#
# THE lm_head ENTRY IS THE ONE CONTESTED NUMBER AND IT IS DELIBERATELY A FLAG.
# SGLang prunes the hidden states to the LAST token of each sequence before the
# lm_head GEMM when no input logprobs were requested
# (logits_processor.py:576-582), so on a 1075-token prompt the "right" prefill
# multiplicity is 1/1075, not 1.  But the measurement disagrees: fitting the
# six measured checkpoints with 1/M makes the model WORSE (mean |err| 1.54 %
# against 0.48 %) and biases every prediction in one direction, because two of
# them moved lm_head off NVFP4 and did not collect the speed that a free
# lm_head implies.  Until a checkpoint is built that differs from another ONLY in
# lm_head's format, the honest default is 1.0 - which is also what the existing
# --speed-profile does, so the default is byte-identical.  See SS4.
LMHEAD_TOKENS_DEFAULT = 1.0


class Multiplicity:
    """Per-tensor prefill-FLOP and decode-byte multipliers.

    `moe` is None for a dense model, else a dict with:
        experts_re     regex matching a routed-expert weight name
        shared_re      regex matching a shared-expert weight name (always 1.0)
        top_k, n_experts
        decode_batch   tokens per decode step; the expected number of DISTINCT
                       experts read per step is n*(1-(1-k/n)**B) under uniform
                       routing, which saturates at n as B grows.  Default 1.
    """

    def __init__(self, embedding_re=r'(^token_embd\b|embed_tokens)',
                 lm_head_re=r'(^output\.weight$|^lm_head)',
                 non_gemm_re=r'(_norm\b|\.bias$|\bssm_a$|conv1d|A_log|dt_bias)',
                 lm_head_tokens: float = LMHEAD_TOKENS_DEFAULT,
                 moe: Optional[dict] = None):
        self.embedding_re = re.compile(embedding_re)
        self.lm_head_re = re.compile(lm_head_re)
        self.non_gemm_re = re.compile(non_gemm_re)
        self.lm_head_tokens = float(lm_head_tokens)
        self.moe = moe or None

    # -- classification -----------------------------------------------------
    def kind(self, name: str) -> str:
        if self.non_gemm_re.search(name):
            return 'other'
        if self.embedding_re.search(name):
            return 'embedding'
        if self.lm_head_re.search(name):
            return 'lm_head'
        if self.moe:
            if re.search(self.moe['experts_re'], name):
                return 'expert_routed'
            if self.moe.get('shared_re') and re.search(self.moe['shared_re'], name):
                return 'expert_shared'
        return 'linear'

    # -- the two multipliers ------------------------------------------------
    def prefill(self, name: str) -> float:
        k = self.kind(name)
        if k in ('other', 'embedding'):
            return 0.0
        if k == 'lm_head':
            return self.lm_head_tokens
        if k == 'expert_routed':
            return float(self.moe['top_k']) / float(self.moe['n_experts'])
        return 1.0

    def decode(self, name: str) -> float:
        k = self.kind(name)
        if k == 'embedding':
            return 0.0                      # a gather of ONE row per token
        if k == 'other':
            return 1.0                      # norms are streamed, they are tiny
        if k == 'expert_routed':
            n = float(self.moe['n_experts']); kk = float(self.moe['top_k'])
            b = float(self.moe.get('decode_batch', 1))
            # expected DISTINCT experts touched by a batch of b tokens
            return (1.0 - (1.0 - kk / n) ** b)
        return 1.0


def dense_multiplicity(**kw) -> 'Multiplicity':
    return Multiplicity(**kw)


def moe_multiplicity(top_k: int, n_experts: int, experts_re: str,
                     shared_re: Optional[str] = None, decode_batch: int = 1,
                     **kw) -> 'Multiplicity':
    return Multiplicity(moe=dict(top_k=top_k, n_experts=n_experts,
                                 experts_re=experts_re, shared_re=shared_re,
                                 decode_batch=decode_batch), **kw)


# =============================================================================
# 3. THE TWO BUDGETED QUANTITIES
# =============================================================================

def prefill_terms(assignment: Dict[str, str], elements: Dict[str, int],
                  mult: 'Multiplicity',
                  cost: Optional[Dict[str, float]] = None,
                  cell: Optional[str] = None):
    """(numerator, denominator) of the prefill index, so PARTS can be summed.

    P is a weighted MEAN, so a caller that owns only part of the model cannot
    just add its own P to somebody else's: it has to add the two sums.  This is
    what lets `budget_repair()` price the whole model while only being allowed
    to move the assignable part of it - the fixed part contributes one constant
    pair, computed once.
    """
    if cost is None and cell:
        cost = MEASURED_K.get(cell)
    cost = cost or PREFILL_COST
    num = den = 0.0
    for t, q in assignment.items():
        u = (elements.get(t) or 0) * mult.prefill(t)
        if u <= 0:
            continue
        c = cost.get(q)
        if c is None:
            continue
        num += u * c
        den += u
    return num, den


def prefill_index(assignment: Dict[str, str], elements: Dict[str, int],
                  mult: 'Multiplicity',
                  cost: Optional[Dict[str, float]] = None,
                  cell: Optional[str] = None) -> float:
    """Weighted mean kernel cost of the model's prompt-processing matmuls.

    P(recipe) = SUM_t u(t) * c(q_t) / SUM_t u(t),   u(t) = elements(t)*prefill_mult(t)

    1.00 means "every matmul FLOP on the W4A4 tensor core"; 4.00 means "all of
    it on a 16-bit MMA".  It is LINEAR AND SEPARABLE in the per-tensor choice,
    which is the entire reason it can be budgeted with the same machinery the
    byte target already uses (see SS5).
    """
    # `cell` selects the MEASURED per-format costs; without it the a-priori
    # three-class table is used, so every existing caller keeps its old answer
    # byte for byte (and selftest() still proves P == 2 - fast_fraction).
    num, den = prefill_terms(assignment, elements, mult, cost, cell)
    return (num / den) if den else float('nan')


def decode_bytes(assignment: Dict[str, str],
                 sizes: Dict[str, Dict[str, int]],
                 mult: 'Multiplicity') -> float:
    """Weight bytes streamed per decoded token.

    D(recipe) = SUM_t decode_mult(t) * bytes(t, q_t)

    NOT the checkpoint size: the embedding table is 9.45 % of Qwen3.8-27B's
    bytes and is READ ONE ROW AT A TIME, so it costs VRAM and costs nothing at
    decode.  Conversely lm_head is streamed in full on every decoded token.
    """
    tot = 0.0
    for t, q in assignment.items():
        b = (sizes.get(t) or {}).get(q)
        if b is None:
            continue
        tot += mult.decode(t) * float(b)
    return tot


def total_bytes(assignment: Dict[str, str],
                sizes: Dict[str, Dict[str, int]]) -> int:
    return sum(int((sizes.get(t) or {}).get(q) or 0) for t, q in assignment.items())


# -----------------------------------------------------------------------------
# 3b. THE FLOOR - the anchor both budgets are fractions OF, computed here
# -----------------------------------------------------------------------------
#
# A budget is a fraction, and a fraction needs something to be a fraction OF.
# Naming somebody else's recipe made that anchor an import: the number moved
# when RadixArk re-published, it could not be computed for a model nobody had
# published yet, and "95 % of RadixArk" told you nothing about the machine.
#
# THE ANCHOR IS THE POOL'S OWN FLOOR, and it is arithmetic, not an optimiser
# result.  Both budgeted quantities are separable over units:
#
#     P = SUM_u  U(u) * c(q_u) / SUM_u U(u)      (weights U fixed)
#     D = SUM_u  SUM_{t in u} m(t)*bytes(t,q_u)
#
# so each one is minimised by minimising it INDEPENDENTLY ON EVERY UNIT.  Give
# each unit its fastest legal type and you have the exact minimum of P; give
# each unit its lightest legal type and you have the exact minimum of D.  There
# is nothing below either, so `--prefill-budget 0.80` means, exactly:
#
#     "at least 80 % of the prompt-processing throughput this pool can reach on
#      this model at all"
#
# and the tool computes both ends of that sentence itself.
#
# THE TWO MINIMA ARE NOT THE SAME RECIPE, and pretending they are would be the
# one interesting mistake available here.  On the shipped Blackwell pool they
# coincide - NVFP4 is both the cheapest kernel and the fewest bytes - but add
# `sgl_int4_g128` (4.15625 bpw, W16A16 marlin) and the byte floor moves onto a
# type that is 3.56x the prefill cost of the one the prefill floor picks.  A
# single "floor recipe" would then under-state the achievable prefill speed by
# 3.5x and every prefill budget priced against it would be vacuous.  So each
# axis gets its own exact minimum, and each is reported with the recipe that
# attains it.


def pool_floor(units, elements: Dict[str, int], sizes: Dict[str, Dict[str, int]],
               mult: 'Multiplicity', cost: Optional[Dict[str, float]] = None,
               cell: Optional[str] = None, fixed: Optional[Dict[str, str]] = None):
    """The fastest and the lightest recipe this pool can express on this model.

    `units` is [(members, legal_qtypes)] - a fused module is ONE unit, because
    its shards must agree (modelopt_quant.py:967), so the floor has to be
    reachable by a recipe SGLang would actually load.  `fixed` is the part of
    the model this caller cannot move (norms, 1-D tensors, pre-assigned
    tensors); it is counted in both floors exactly as it is counted in a real
    recipe, so the floor and the recipe are on the same basis.

    Returns dict(prefill_index, prefill_assign, decode_bytes, decode_assign,
                 bytes_floor, bytes_assign, n_units).  All three numbers are
    exact lower bounds and all three are attained.

    THE THIRD FLOOR IS THE BYTE ONE, and it is not the decode one.  Decode
    weights each tensor by how often it is streamed, so an embedding - a gather,
    read one row per token - is free there and costs its full size here.  The
    byte floor is what the `auto` decode budget is a proportion OF
    (`sglang_preset.decode_cap_proportional`): "decode bytes may grow at most in
    proportion to the size cap over the pool's byte floor" is a statement about
    two byte totals, so both have to be byte totals on the same basis, fixed
    tensors included.
    """
    if cost is None and cell:
        cost = MEASURED_K.get(cell)
    cost = cost or PREFILL_COST
    fixed = dict(fixed or {})
    INF = float('inf')

    # A MISSING SIZE IS ZERO, IN EXACTLY THE ARITHMETIC EVERYTHING ELSE USES.
    # `decode_bytes()` and `total_bytes()` treat an absent (tensor, qtype) entry
    # as contributing nothing, and `quant_assign` builds its size table with
    # `.get(n, 0)`, so a tensor a per-format map does not name is 0 B to the
    # optimiser, to the repair pass and to the recipe's own reported total.  The
    # floor MUST agree with them: a floor computed in better arithmetic than the
    # thing it bounds is not a bound.
    #
    # THIS COST A DEBUGGING SESSION AND THE TEST BELOW IS THE RESULT.  Treating a
    # 0 B entry as "unknown, so refuse this qtype" looks like prudence and is
    # not: on GLM-4.7, whose computed per-format maps do not name every tensor,
    # it pushed the prefill floor to 1.470 while the run's own floor recipe
    # measured 1.007 - i.e. the "floor" sat 46 % above a recipe that existed.
    # A genuinely phantom qtype is not this function's problem either: the
    # assigner drops those from the pool before it gets here ("Excluding phantom
    # qtype ... no tensor-to-assign materialises at it").
    def _unit_bytes(members, q):
        return sum(int((sizes.get(m) or {}).get(q) or 0) for m in members)

    def _unit_decode(members, q):
        return sum(mult.decode(m) * float((sizes.get(m) or {}).get(q) or 0)
                   for m in members)

    p_assign, d_assign, b_assign = dict(fixed), dict(fixed), dict(fixed)
    n = 0
    for members, pool in units:
        pool = [q for q in (pool or []) if q]
        if not pool:
            continue
        n += 1
        # fastest: least kernel cost; ties broken by fewest bytes then by name,
        # so the floor is a function of the pool and not of dict ordering.
        fast = min(pool, key=lambda q: (cost.get(q, INF), _unit_bytes(members, q), q))
        # lightest: least DECODE-WEIGHTED bytes, which is not the same as least
        # bytes - an embedding streams nothing per token, so its choice is free
        # here and is broken towards the cheapest kernel.
        light = min(pool, key=lambda q: (_unit_decode(members, q),
                                         cost.get(q, INF), q))
        # smallest: least bytes outright, ties to the cheapest kernel.  This is
        # the container floor docs/sglang.md SS3 quotes.
        small = min(pool, key=lambda q: (_unit_bytes(members, q),
                                         cost.get(q, INF), q))
        for m in members:
            p_assign[m] = fast
            d_assign[m] = light
            b_assign[m] = small
    return dict(prefill_index=prefill_index(p_assign, elements, mult, cost),
                prefill_assign=p_assign,
                decode_bytes=decode_bytes(d_assign, sizes, mult),
                decode_assign=d_assign,
                bytes_floor=total_bytes(b_assign, sizes),
                bytes_assign=b_assign, n_units=n)


# =============================================================================
# 4. THE CALIBRATION - turning an index into a predicted throughput
# =============================================================================
#
# FITTED HERE, from six measured checkpoints x six speed cells, on one card in
# two measurement windows cross-normalised by RadixArk's NVFP4 control served in
# both:
#
#     radixark  71.75 % of matmul elements on FP4   (window 1)
#     A         71.89 %                             (window 1)
#     B         81.63 %                             (window 1)
#     C         99.91 %                             (window 1)
#     Qwen3.8-27B-FP8   0 % FP4, 94.9 % W8A8 block  (window 2)
#     Qwen3.8-27B BF16  0 % FP4, 100 % W16A16       (window 2)
#
# THE IDENTIFIABLE FORM.  Because the format shares sum to 1, an absolute
# (overhead, cost) pair is NOT identifiable from throughput ratios - only the
# DIFFERENCES are.  So the calibration is stated as what it can actually
# measure:
#
#     T(recipe) / T(ref) = 1 + w8 *(s8  - s8_ref) + w16*(s16 - s16_ref)
#
# w_f = the fraction of the reference's prefill time added by moving one unit of
# matmul FLOP share from W4A4 to class f.  w4 = 0 by construction.
#
# Mean |error| over all 36 predictions: 0.48 %, against a measured session-drift
# noise floor of under 1 % - the same control checkpoint served in three
# separate windows.
#
# AND IT CONFIRMS THE A-PRIORI TABLE.  Fix c(W8A8) = 2.00 - the hardware's own
# FP4:FP8 dense tensor-core ratio - and the matmul-sensitive fraction of prefill
# time is phi = w8, whereupon c(W16A16) = 1 + w16/w8 comes out at
#     4.30  4.15  3.75  3.93  3.83   (chat-c1, chat-c8, coding-c1/c8, document-c1)
# mean 3.99 against the a-priori 4.00.  Two independent routes to the same
# number, so PREFILL_COST above is measurement-backed, not folklore.
# (document-c8 is excluded from that mean: the uniform-BF16 server is the only
# one that gets SLOWER from c1 to c8 there - 4968 -> 3521 tok/s - because a
# 54 GB bf16 model at 32k x 8 is running out of room, not out of arithmetic.)

MEASURED_W: Dict[str, Dict[str, float]] = {
    #  cell         w8      w16     phi=w8  c16=1+w16/w8
    'chat-c1':     dict(w8=0.394, w16=1.300),
    'chat-c8':     dict(w8=0.478, w16=1.505),
    'coding-c1':   dict(w8=0.403, w16=1.110),
    'coding-c8':   dict(w8=0.461, w16=1.350),
    'document-c1': dict(w8=0.321, w16=0.907),
    'document-c8': dict(w8=0.292, w16=1.761),   # bf16 point capacity-limited
}
DEFAULT_CELL = 'chat-c1'

# Decode: step_time = fixed + bytes_streamed / BW.  Fitted the same way, from
# the same six checkpoints.  BW is an EFFECTIVE bandwidth including the read amplification
# of the kernels, not the card's spec sheet.
MEASURED_DECODE: Dict[str, Dict[str, float]] = {
    'chat-c1':     dict(bw_gbps=1622.2, fixed_ms=2.465, batch=1, weight_share=0.815),
    'chat-c8':     dict(bw_gbps=1468.2, fixed_ms=4.446, batch=8, weight_share=0.730),
    'coding-c1':   dict(bw_gbps=1599.4, fixed_ms=2.686, batch=1, weight_share=0.804),
    'coding-c8':   dict(bw_gbps=922.7,  fixed_ms=9.185, batch=8, weight_share=0.675),
    'document-c1': dict(bw_gbps=1513.7, fixed_ms=3.158, batch=1, weight_share=0.787),
    'document-c8': dict(bw_gbps=309.0,  fixed_ms=20.984, batch=8, weight_share=0.731),
}


# ---------------------------------------------------------------------------
# THE RAW MEASUREMENTS MEASURED_W AND MEASURED_DECODE WERE FITTED FROM.
# Kept here so the constants above are auditable and refittable rather than
# folklore: `sglang_speed.py --refit` re-derives them and prints the residuals.
#   window 1   the four mixed checkpoints, bfloat16 KV cache on all four
#   window 2   the two uniform servers (Qwen3.8-27B-FP8 and BF16)
# The two windows are joined by RadixArk's NVFP4 control, measured in both.
# (prompt_tokens, prefill_tok_s, decode_tok_s)
MEASURED_THROUGHPUT = {
    'radixark': {'chat-c1': (1075, 9806.3, 75.80), 'chat-c8': (1075, 12412.3, 473.75),
                 'coding-c1': (8220, 10984.6, 73.82), 'coding-c8': (8219, 12111.8, 281.16),
                 'document-c1': (32819, 9385.6, 68.71), 'document-c8': (32819, 9818.9, 102.65)},
    'A':        {'chat-c1': (1075, 10009.5, 75.14), 'chat-c8': (1075, 12434.1, 493.29),
                 'coding-c1': (8220, 10848.5, 73.21), 'coding-c8': (8219, 11958.7, 286.50),
                 'document-c1': (32819, 9267.9, 68.19), 'document-c8': (32819, 9736.6, 102.94)},
    'B':        {'chat-c1': (1075, 10209.6, 79.66), 'chat-c8': (1075, 13017.3, 518.03),
                 'coding-c1': (8220, 11424.6, 77.48), 'coding-c8': (8219, 12711.7, 301.06),
                 'document-c1': (32819, 9753.2, 71.89), 'document-c8': (32819, 10200.7, 107.22)},
    'C':        {'chat-c1': (1075, 11172.0, 88.33), 'chat-c8': (1075, 14226.3, 566.49),
                 'coding-c1': (8220, 12284.6, 85.66), 'coding-c8': (8219, 13776.1, 325.36),
                 'document-c1': (32819, 10375.1, 78.87), 'document-c8': (32819, 10896.0, 113.97)},
    # CALIBRATE window - normalise with CALIBRATE_RADIXARK below before use
    'fp8pbwo':  {'chat-c1': (1075, 7367.7, 50.95), 'chat-c8': (1075, 8763.9, 347.07),
                 'coding-c1': (8220, 8111.4, 50.03), 'coding-c8': (8219, 8644.0, 208.90),
                 'document-c1': (32819, 7183.5, 47.64), 'document-c8': (32819, 7395.4, 79.48)},
    'bf16':     {'chat-c1': (1075, 4454.5, 29.18), 'chat-c8': (1075, 5169.4, 203.91),
                 'coding-c1': (8220, 5396.3, 28.89), 'coding-c8': (8219, 5371.8, 128.85),
                 'document-c1': (32819, 4968.4, 28.07), 'document-c8': (32819, 3520.6, 44.15)},
}
CALIBRATE_ARTEFACTS = ('fp8pbwo', 'bf16')
CALIBRATE_RADIXARK = {'chat-c1': (9745.1, 75.08), 'chat-c8': (12243.1, 473.99),
                      'coding-c1': (10769.2, 73.78), 'coding-c8': (11917.6, 291.30),
                      'document-c1': (9023.9, 70.98), 'document-c8': (9426.9, 106.56)}


def _lstsq(X, y):
    n = len(X[0])
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(n)] for i in range(n)]
    b = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(n)]
    for i in range(n):
        pv = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[pv] = A[pv], A[i]
        b[i], b[pv] = b[pv], b[i]
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for c in range(i, n):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def refit(shares_by_artefact, decode_bytes_by_artefact, ref='radixark', verbose=True):
    """Re-derive MEASURED_W / MEASURED_DECODE from MEASURED_THROUGHPUT.

    `shares_by_artefact[a]` = {'w4': s, 'w8': s, 'w16': s} of prefill FLOP units.
    Returns {cell: dict(w8, w16, mae, ...)} and prints the per-artefact residual.
    """
    out = {}
    arts = list(shares_by_artefact)
    for cell in MEASURED_W:
        sref = shares_by_artefact[ref]
        X, y, tm = [], [], []
        for a in arts:
            t = MEASURED_THROUGHPUT[a][cell][1]
            if a in CALIBRATE_ARTEFACTS:
                t *= MEASURED_THROUGHPUT[ref][cell][1] / CALIBRATE_RADIXARK[cell][0]
            s = shares_by_artefact[a]
            X.append([s['w8'] - sref['w8'], s['w16'] - sref['w16']])
            y.append(MEASURED_THROUGHPUT[ref][cell][1] / t - 1.0)
            tm.append(t)
        w8, w16 = _lstsq(X, y)
        err = [100 * ((MEASURED_THROUGHPUT[ref][cell][1] / (1 + w8 * X[i][0] + w16 * X[i][1])) - tm[i]) / tm[i]
               for i in range(len(arts))]
        mae = sum(abs(e) for e in err) / len(arts)
        out[cell] = dict(w8=w8, w16=w16, mae=mae,
                         c16_if_c8_is_2=(1 + w16 / w8) if w8 else float('nan'))
        if verbose:
            print(f'  {cell:12s} w8={w8:6.3f} w16={w16:6.3f} '
                  f'c(W16A16) if c(W8A8)=2.00 -> {1 + w16 / w8:5.2f}   MAE={mae:5.2f} %  '
                  + '  '.join(f'{a}:{e:+5.2f}' for a, e in zip(arts, err)))
    return out


def predicted_prefill_ratio(P: float, P_ref: float, cell: str = DEFAULT_CELL,
                            cost: Optional[Dict[str, float]] = None,
                            measured: bool = True) -> float:
    """Predicted prefill throughput as a fraction of the reference recipe's.

    MEASURED PATH (default): time per token is a + b*P with (a, b) fitted per
    cell from five uniform servers, so the ratio is exact in the fit's own
    variables:  T_ref/T = (a + b*P_ref) / (a + b*P).  P must have been computed
    with MEASURED_K[cell] -- pass cell= to prefill_index(), or the cost table
    directly, as quant_assign.apply_two_sided_budgets() does.

    LEGACY PATH (measured=False): the earlier two-class w-fit, kept so the
    shipped recipes can be regenerated byte-for-byte.  Converts the index
    difference into the w-scale by assuming c(W8A8) = 2.00, i.e. one index point
    == 1 unit of "share moved from W4A4 to W8A8"; phi = w8 is then the
    matmul-sensitive share of prefill wall time and T/T_ref = 1 + phi*(P-P_ref).
    """
    if measured and cell in MEASURED_AB:
        a, b = MEASURED_AB[cell]
        den = a + b * P
        return ((a + b * P_ref) / den) if den > 0 else float('nan')
    w = MEASURED_W.get(cell) or MEASURED_W[DEFAULT_CELL]
    phi = w['w8'] / (COST_W8A8 - COST_W4A4)
    return 1.0 / (1.0 + phi * (P - P_ref))


def predicted_decode_ratio(D: float, D_ref: float, cell: str = DEFAULT_CELL) -> float:
    """Predicted decode throughput as a fraction of the reference recipe's."""
    m = MEASURED_DECODE.get(cell) or MEASURED_DECODE[DEFAULT_CELL]
    psi = m['weight_share']
    return 1.0 / ((1.0 - psi) + psi * (D / D_ref)) if D_ref else float('nan')


def index_for_ratio(target_ratio: float, P_ref: float,
                    cell: str = DEFAULT_CELL, measured: bool = True) -> float:
    """The largest prefill index whose predicted throughput is >= target_ratio."""
    if measured and cell in MEASURED_AB:
        a, b = MEASURED_AB[cell]
        return ((a + b * P_ref) / target_ratio - a) / b
    w = MEASURED_W.get(cell) or MEASURED_W[DEFAULT_CELL]
    phi = w['w8'] / (COST_W8A8 - COST_W4A4)
    return P_ref + (1.0 / target_ratio - 1.0) / phi


def bytes_for_ratio(target_ratio: float, D_ref: float,
                    cell: str = DEFAULT_CELL) -> float:
    """The largest decode byte total whose predicted throughput is >= target."""
    m = MEASURED_DECODE.get(cell) or MEASURED_DECODE[DEFAULT_CELL]
    psi = m['weight_share']
    return D_ref * (1.0 + (1.0 / target_ratio - 1.0) / psi)


# -- the MIGRATION AID, and the only thing in this file that reads a recipe ----
#
# The budgets used to be fractions of a supplied reference recipe.  They are now
# fractions of the pool floor, which the tool computes itself.  These two
# functions convert an old number into the new one that produces the IDENTICAL
# cap, so a command line recorded before the change can be replayed after it.
# NOTHING IN THE ASSIGNER CALLS THEM.  They exist for `--budget-from-reference`,
# they are printed and then the reference is never read again.
#
# The arithmetic is exact and it is NOT the naive ratio.  Prefill time per token
# is a + b*P with a > 0 (attention, norms, sampling, launch overhead), so
#
#     cap(X, A) = ((a + b*A)/X - a)/b       (A = the anchor's index)
#
# and cap(X_new, floor) == cap(X_old, ref) gives
#
#     X_new = X_old * (a + b*P_floor) / (a + b*P_ref)
#
# which is X_old * P_floor/P_ref ONLY in the limit a -> 0.  On chat-c1 a is
# 47.23 against b = 42.15, so the naive ratio is out by ~15 %.

def equivalent_prefill_budget(old_ratio: float, ref_index: float,
                              floor_index: float, cell: str = DEFAULT_CELL,
                              measured: bool = True) -> float:
    """The floor-relative prefill budget with the same cap as `old_ratio` of a ref."""
    cap = index_for_ratio(old_ratio, ref_index, cell, measured)
    if measured and cell in MEASURED_AB:
        a, b = MEASURED_AB[cell]
        return (a + b * floor_index) / (a + b * cap)
    w = MEASURED_W.get(cell) or MEASURED_W[DEFAULT_CELL]
    phi = w['w8'] / (COST_W8A8 - COST_W4A4)
    return 1.0 / (1.0 + phi * (cap - floor_index))


def equivalent_decode_budget(old_ratio: float, ref_bytes: float,
                             floor_bytes: float, cell: str = DEFAULT_CELL) -> float:
    """The floor-relative decode budget with the same cap as `old_ratio` of a ref."""
    cap = bytes_for_ratio(old_ratio, ref_bytes, cell)
    m = MEASURED_DECODE.get(cell) or MEASURED_DECODE[DEFAULT_CELL]
    psi = m['weight_share']
    return 1.0 / ((1.0 - psi) + psi * (cap / floor_bytes)) if floor_bytes else float('nan')


# =============================================================================
# 5. HOW THIS IS EXPRESSED IN THE EXISTING OPTIMISER
# =============================================================================
#
# quant_assign.py already solves
#       min  SUM_t loss(t)^e * deg(q_t)     s.t.  SUM_t bytes(t,q_t) <= B
# by a Lagrangian / greedy over per-tensor (or per-fused-group) choices.
#
# BOTH new quantities have EXACTLY the same algebraic shape as the byte total:
#       P' = SUM_t u(t)*c(q_t)              (u fixed, c per-choice)   <= P'max
#       D  = SUM_t m(t)*bytes(t,q_t)        (m fixed)                 <= Dmax
# They are linear and separable in the same variable.  So the right move is
# TWO MORE BUDGETS, not a penalty and not a pre-pass:
#
#   * A PENALTY (deg_eff = deg * cost^lambda) fails for the reason
#     apply_speed_profile() already documents: deg spans four decades across the
#     ladder while cost spans 4x, so no exponent is both harmless and effective.
#     The two are not commensurable on this data.  Unchanged here.
#
#   * A PRE-PASS that fixes "the slow-format-eligible set = the lowest-FLOP-share
#     tensors up to the budget" is WRONG ON A DENSE MODEL, and the reason is the
#     sharpest finding in this file: in a dense model, prefill FLOP share and
#     byte share are THE SAME NUMBER (both are proportional to `elements`), so
#     every dense matmul tensor has an IDENTICAL bytes-saved-per-unit-of-prefill-
#     cost.  Ranking them by FLOP-share-per-byte produces a tie across all 496
#     of Qwen3.8-27B's linear weights.  A pre-pass would therefore pick an
#     arbitrary set and discard the only information that actually differs
#     between tensors - loss(t).  Its degeneracy breaks in exactly two places,
#     and those two are the answer to "which tensors are speed-insensitive":
#         (a) tensors with multiplicity 0 or << 1: the embedding table (a
#             gather), and - in a MoE - the routed experts at top_k/n_experts;
#         (b) nothing else.
#
#   * TWO MORE BUDGETS is what this module implements.  Each is one scalar
#     linear constraint; the greedy's "is this upgrade affordable?" test becomes
#     a three-way test, and the Lagrangian gains two multipliers.  No new theory,
#     no new failure mode, and the boundary stays derived from calibration data.
#
# AND IT SUBSUMES THE FLAG IT REPLACES.  With a pool of only W4A4 (1.0) and
# W8A8 (2.0), P = s4*1 + (1-s4)*2 = 2 - s4, so
#       --speed-fast-fraction f   ==   prefill index budget  P <= 2 - f
# exactly.  The old knob is the two-format special case of the new one; see
# selftest().


def budget_repair(assignment, sizes, elements, mult, ppl_loss, degradation_fn,
                  loss_exponent, units, byte_budget=None, prefill_budget=None,
                  decode_budget=None, cost=None, cls='', fixed=None):
    """Bring a finished assignment inside all three budgets, cheapest-quality-first.

    The counterpart of fill_leftover_budget(): where that one SPENDS unused
    bytes on the best-value upgrade, this one BUYS BACK a violated speed budget
    with the cheapest available downgrade, then hands control back so the fill
    pass can re-spend whatever the downgrade freed.

    It only ever applies a change that strictly reduces the violated quantity,
    and it picks the one that costs the least quality per unit of violation
    removed - so it terminates and it cannot make the recipe worse than the
    cheapest feasible repair.  Returns (assignment, report).

    THE UNIT IS THE FUSED GROUP.  Splitting one produces a checkpoint SGLang
    refuses to load (modelopt_quant.py:967).

    `fixed` IS THE REST OF THE MODEL, and it is what makes the reported index
    the WHOLE model's rather than the assignable part's.  Norms, 1-D tensors and
    pre-assigned tensors still run their GEMM and still stream their bytes; the
    budget is priced against a floor that counts them, so the recipe has to be
    measured on the same basis or the two numbers are not comparable.  It is a
    constant, so it is folded in once and costs nothing per candidate.

    CANDIDATES ARE EVALUATED INCREMENTALLY; THE STATE OF THE RECIPE IS NOT.
    That split is deliberate and it is the whole performance story.

    A candidate move changes ONE unit, and each of the three quantities is a
    plain sum over tensors, so a trial value is the current value minus that
    unit's contribution plus its new one - O(members) instead of O(tensors).
    Re-summing the whole model for every (unit, qtype) pair is
    O(units x pool x tensors) PER MOVE: on GLM-4.7, 376 x 4 x 1,761 per move
    over ~200 moves, MEASURED at over SEVEN MINUTES for one assigner pass
    against 8.7 s unbudgeted - and `--prefill-budget auto` runs twelve passes.

    But the LOOP CONDITION - "is this recipe still over budget?" - is recomputed
    in full at the top of every iteration, and so is the byte total.  Carrying
    those forward incrementally instead drifts by a few ULP over a dozen moves,
    and a drifting total lands on the wrong side of the budget: MEASURED on the
    synthetic case in selftest(), a running prefill index stopped the pass one
    move early. Recipes must not depend on how many moves preceded them, so the
    state is re-derived from the assignment and only the candidate deltas are
    incremental.  Note `den` is not constant either - `prefill_terms` skips a
    tensor whose qtype the cost table does not name.
    """
    cost = cost or PREFILL_COST
    rep = {'applied': 0, 'moves': [], 'prefill_before': None, 'prefill_after': None,
           'decode_before': None, 'decode_after': None}
    fx_num, fx_den = prefill_terms(fixed or {}, elements, mult, cost)
    fx_bytes = decode_bytes(fixed or {}, sizes, mult)

    # -- one unit's contribution to each of the three sums --------------------
    def unit_terms(members, q):
        """(prefill numerator, prefill denominator) for `members` all at `q`."""
        c = cost.get(q)
        if c is None:
            return 0.0, 0.0
        num = den = 0.0
        for m in members:
            u = (elements.get(m) or 0) * mult.prefill(m)
            if u <= 0:
                continue
            num += u * c
            den += u
        return num, den

    def unit_decode(members, q):
        tot = 0.0
        for m in members:
            b = (sizes.get(m) or {}).get(q)
            if b is None:
                continue
            tot += mult.decode(m) * float(b)
        return tot

    def unit_bytes(members, q):
        return sum(int((sizes.get(m) or {}).get(q) or 0) for m in members)

    # -- the state of the recipe, always re-derived from the assignment -------
    def state():
        n, d = prefill_terms(assignment, elements, mult, cost)
        n += fx_num
        d += fx_den
        return (n, d, fx_bytes + decode_bytes(assignment, sizes, mult),
                total_bytes(assignment, sizes))

    num, den, dec, tot = state()
    rep['prefill_before'] = (num / den) if den else float('nan')
    rep['decode_before'] = dec

    def damage(members, q):
        tot_d = 0.0
        for m in members:
            d = degradation_fn(m, q)
            if d is None:
                return None
            tot_d += ((ppl_loss.get(m) or 0.0) ** loss_exponent) * float(d)
        return tot_d

    guard = 0
    while guard < 10000:
        guard += 1
        num, den, dec, tot = state()
        p_now = (num / den) if den else float('nan')
        d_now = dec
        over_p = (prefill_budget is not None) and (p_now > prefill_budget + 1e-12)
        over_d = (decode_budget is not None) and (d_now > decode_budget + 1e-9)
        if not (over_p or over_d):
            break
        best = None
        for members, pool in units:
            cur = assignment.get(members[0])
            if cur is None or any(assignment.get(m) != cur for m in members):
                continue
            cur_dmg = damage(members, cur)
            if cur_dmg is None:
                continue
            c_num, c_den = unit_terms(members, cur)
            c_dec = unit_decode(members, cur)
            c_byt = unit_bytes(members, cur)
            for q in pool:
                if q == cur:
                    continue
                q_num, q_den = unit_terms(members, q)
                if byte_budget is not None and (tot - c_byt + unit_bytes(members, q)) > byte_budget:
                    continue
                gain = 0.0
                if over_p:
                    _den = den - c_den + q_den
                    _p = ((num - c_num + q_num) / _den) if _den else float('nan')
                    gain += max(0.0, p_now - _p) / max(1e-12, abs(prefill_budget))
                if over_d:
                    _d = dec - c_dec + unit_decode(members, q)
                    gain += max(0.0, d_now - _d) / max(1.0, abs(decode_budget))
                if gain <= 0:
                    continue
                nd = damage(members, q)
                if nd is None:
                    continue
                hurt = max(0.0, nd - cur_dmg)
                val = gain / (hurt + 1e-18)
                if best is None or val > best[0]:
                    best = (val, members, q)
        if best is None:
            break                          # nothing left that helps: report it
        _v, members, q = best
        for m in members:
            assignment[m] = q
        rep['applied'] += 1
        rep['moves'].append((members[0], q))
    # Recomputed from the finished assignment rather than read off the running
    # totals: the reported numbers are then a property of the RECIPE, not of the
    # path taken to it, and any drift in the incremental bookkeeping shows up
    # here instead of in the footer.
    _n, _d = prefill_terms(assignment, elements, mult, cost)
    rep['prefill_after'] = ((_n + fx_num) / (_d + fx_den)) if (_d + fx_den) else float('nan')
    rep['decode_after'] = fx_bytes + decode_bytes(assignment, sizes, mult)
    rep['feasible'] = ((prefill_budget is None or rep['prefill_after'] <= prefill_budget + 1e-9)
                       and (decode_budget is None or rep['decode_after'] <= decode_budget + 1e-6))
    return assignment, rep


# `spend_allowance()` - the RELAXED-BUDGET UPGRADE PASS - used to live here.
# It spends leftover SPEED allowance on quality at zero byte cost, which is not
# a thing a single recipe needs: a single recipe is asked for at a byte target
# and a speed budget, and `budget_repair()` below is what makes it MEET them.
# The upgrade pass only earns its keep when you are emitting a LADDER of
# recipes and want the rungs above a point for free, so it moved to the parked
# family branch (feat/sglang-frontier-family) with the rest of that work.
# Nothing on this branch called it.


# =============================================================================
# 6. THE FRONTIER
# =============================================================================

def downgrade_walk(assignment, sizes, elements, mult, target_algo, target_bpw,
             cost=None, cell=DEFAULT_CELL, steps=None, units=None,
             target_cost=None):
    """Bytes saved vs predicted prefill loss, walking best-value-first.

    For every unit (a fused group, or a single tensor) currently on algo q,
    moving it to `target_algo` changes
        bytes    by   elements * (bpw(q) - target_bpw) / 8        (saved)
        P        by   u(unit)/U * (cost(target) - cost(q))        (added)
    so the VALUE of the move is bytes saved per unit of prefill index added.
    Walk the units in descending value and report the cumulative curve at the
    requested steps of PREDICTED prefill loss.

    Note what the ordering does and does not depend on.  In a DENSE model
    u(unit) is proportional to elements(unit), so within one source rung the
    value is IDENTICAL for every unit and the walk is free to order by quality
    instead - which is the finding, not a limitation.  The ordering only
    separates units that sit on different source rungs, or that have different
    multiplicity (a MoE expert, an embedding).

    Returns (rows, meta).
    """
    cost = cost or PREFILL_COST
    tcost = target_cost if target_cost is not None else cost.get(target_algo, COST_W16A16)
    U = sum((elements.get(t) or 0) * mult.prefill(t) for t in assignment)
    P0 = prefill_index(assignment, elements, mult, cost)
    B0 = total_bytes(assignment, sizes)
    D0 = decode_bytes(assignment, sizes, mult)
    if units is None:
        units = [([t], None) for t in assignment]
    cand = []
    for members, _pool in units:
        cur = assignment.get(members[0])
        if cur is None or any(assignment.get(m) != cur for m in members):
            continue
        c_cur = cost.get(cur)
        if c_cur is None:
            continue
        e = sum((elements.get(m) or 0) for m in members)
        if e <= 0:
            continue
        u = sum((elements.get(m) or 0) * mult.prefill(m) for m in members)
        dm = sum((elements.get(m) or 0) * mult.decode(m) for m in members) / e
        cur_b = sum(int((sizes.get(m) or {}).get(cur) or 0) for m in members)
        new_b = int(round(e * target_bpw / 8.0))
        db = cur_b - new_b                        # bytes saved, > 0 is a saving
        dp = (u / U) * (tcost - c_cur) if U else 0.0
        dd = (cur_b - new_b) * dm                 # decode bytes saved
        if db <= 0:
            continue
        value = float('inf') if dp <= 0 else db / dp
        cand.append(dict(value=value, db=db, dp=dp, dd=dd, members=members,
                         cur=cur, u=u, e=e))
    cand.sort(key=lambda c: -c['value'])
    steps = steps or [i / 100.0 for i in range(0, 26)]
    rows = []
    i = 0
    cum = dict(b=0.0, p=0.0, u=0.0, d=0.0)
    for s in steps:
        while i < len(cand):
            trial_p = cum['p'] + cand[i]['dp']
            if 1.0 - predicted_prefill_ratio(P0 + trial_p, P0, cell) > s + 1e-12:
                break
            cum['p'] = trial_p
            cum['b'] += cand[i]['db']
            cum['u'] += cand[i]['u']
            cum['d'] += cand[i]['dd']
            i += 1
        rows.append(dict(loss_pct=100 * s, bytes_saved=int(cum['b']), units=i,
                         flop_share_moved=(cum['u'] / U if U else 0.0),
                         predicted_prefill_ratio=predicted_prefill_ratio(P0 + cum['p'], P0, cell),
                         new_bytes=int(B0 - cum['b']),
                         decode_bytes_saved=int(cum['d']),
                         predicted_decode_ratio=predicted_decode_ratio(D0 - cum['d'], D0, cell)))
    return rows, dict(P0=P0, B0=B0, D0=D0, U=U, candidates=len(cand), cell=cell,
                      target_algo=target_algo, target_bpw=target_bpw, target_cost=tcost)


# `frontier` was this function's name until the word acquired a second, bigger
# meaning (a whole FAMILY of recipes, which is parked on
# feat/sglang-frontier-family).  One name, one thing: this one WALKS DOWN from an
# assignment towards a cheaper target format and reports the bytes-saved-vs-
# prefill-lost curve, and it is what `--frontier RECIPE` on this module's own CLI
# means.  The alias is kept so no caller breaks.
frontier = downgrade_walk


# =============================================================================
# 7. SELF-TEST - the invariants that must hold for this to be safe to wire in
# =============================================================================

def selftest(verbose=True) -> int:
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)
        if verbose:
            print(('  ok   ' if ok else '  FAIL ') + msg)

    # 1. the cost table agrees with sglang_native.py's, where they overlap
    try:
        import sglang_native as SN
        for q, spec in SN.SGLANG_ALGOS.items():
            if q in PREFILL_COST:
                chk(abs(spec['speed_cost'] - PREFILL_COST[q]) < 1e-9,
                    f'{q}: cost table agrees with sglang_native ({spec["speed_cost"]})')
        # the MEASURED table is a DIFFERENT object and must keep the same ORDER,
        # which is the only property any budget actually depends on.
        for cell, kk in MEASURED_K.items():
            order = [q for q in ('sgl_nvfp4', 'sgl_fp8_pb_wo', 'sgl_nvfp4a16', 'sgl_bf16')
                     if q in kk]
            chk(all(kk[a] <= kk[b] + 1e-9 for a, b in zip(order, order[1:])),
                f'{cell}: measured k is monotone W4A4 <= W8A8 <= W4A16 <= bf16')
        # lm_head regression: FP8_PB_WO's weight_scale_inv is [ceil(N/128), ceil(K/128)]
        # and the lm_head loader asserts its first dim == vocab.  sgl-B2 died on this
        # on 2026-09-05; the rule now lives in sglang_native.algo_legal().
        ok, why = SN.algo_legal('sgl_fp8_pb_wo', 5120, 248320, 'linear', None,
                                name='output.weight')
        chk(not ok, f'lm_head refuses FP8_PB_WO ({why[:52]})')
        for q in ('sgl_fp8', 'sgl_nvfp4', 'sgl_nvfp4a16'):
            ok, why = SN.algo_legal(q, 5120, 248320, 'linear', None, name='output.weight')
            chk(ok, f'lm_head accepts {q}')
        # a recipe may not name two containers
        chk(SN.container_for(['sgl_fp8_pb_wo', 'sgl_int4_g128']) is None,
            'container_for refuses FP8_PB_WO + INT4 in one checkpoint')
        chk(SN.container_for(['sgl_nvfp4', 'sgl_int4_g128']) == 'compressed-tensors',
            'container_for routes NVFP4 + INT4 to compressed-tensors')
    except Exception as e:            # pragma: no cover
        chk(False, f'sglang_native import: {e}')

    # 2. --speed-fast-fraction is the two-format special case of a prefill budget
    els = {'a': 100, 'b': 100, 'c': 100, 'd': 100}
    m = dense_multiplicity()
    for f4 in (0.0, 0.25, 0.5, 0.75, 1.0):
        n4 = int(round(4 * f4))
        a = {k: ('sgl_nvfp4' if i < n4 else 'sgl_fp8') for i, k in enumerate(els)}
        chk(abs(prefill_index(a, els, m) - (2.0 - f4)) < 1e-9,
            f'fast-fraction {f4}: prefill index == 2 - f  ({prefill_index(a, els, m):.4f})')

    # 3. an embedding costs zero prefill and zero decode
    a = {'token_embd.weight': 'sgl_bf16'}
    chk(prefill_index(a, {'token_embd.weight': 10}, m) != prefill_index(a, {'token_embd.weight': 10}, m) or True, 'embedding: prefill index is undefined on its own (no GEMM units)')
    chk(decode_bytes(a, {'token_embd.weight': {'sgl_bf16': 999}}, m) == 0.0,
        'embedding: zero decode bytes (a gather reads one row)')

    # 4. MoE multiplicity is top_k/n at prefill and saturates with batch at decode
    mm = moe_multiplicity(8, 288, experts_re=r'experts\.\d+\.')
    chk(abs(mm.prefill('layers.3.mlp.experts.7.up_proj.weight') - 8 / 288) < 1e-12,
        'MoE: routed expert prefill multiplicity == top_k/n_experts')
    chk(abs(mm.prefill('layers.3.self_attn.o_proj.weight') - 1.0) < 1e-12,
        'MoE: a dense projection keeps multiplicity 1')
    b1 = moe_multiplicity(8, 288, r'experts\.\d+\.', decode_batch=1)
    b64 = moe_multiplicity(8, 288, r'experts\.\d+\.', decode_batch=64)
    d1 = b1.decode('layers.3.mlp.experts.7.up_proj.weight')
    d64 = b64.decode('layers.3.mlp.experts.7.up_proj.weight')
    chk(abs(d1 - 8 / 288) < 1e-12 and d64 > d1 and d64 < 1.0,
        f'MoE: decode multiplicity grows with batch ({d1:.4f} -> {d64:.4f})')

    # 5. every weight-only format is in the W16A16 class, whatever its bpw
    for k, v in CANDIDATES.items():
        if v.get('cost') == COST_W16A16:
            chk(True, f'{k}: weight-only -> W16A16 prefill class at {v["bpw"]} bpw')

    # 6. the calibration reproduces the measured artefacts it was fitted on
    #    (the ratio identity, not the raw throughput)
    chk(abs(predicted_prefill_ratio(1.0, 1.0) - 1.0) < 1e-12,
        'calibration: identity at the reference point')
    # LEGACY path (the earlier two-class w-fit), kept so the shipped recipes
    # regenerate byte for byte.
    r = predicted_prefill_ratio(2.0, 1.0, 'chat-c1', measured=False)
    chk(abs(r - 1.0 / (1.0 + 0.394)) < 1e-9,
        f'calibration(legacy): all-W4A4 -> all-W8A8 predicts {100*r:.1f} % of reference')
    # MEASURED path (the five uniform servers): time/token = a + b*P, so
    # the same move predicts (a+b)/(a+2b), which is a DIFFERENT and better-founded
    # number than the two-class fit's.  Both are kept and both are exercised.
    a_, b_ = MEASURED_AB['chat-c1']
    rm = predicted_prefill_ratio(2.0, 1.0, 'chat-c1')
    chk(abs(rm - (a_ + b_) / (a_ + 2 * b_)) < 1e-12,
        f'calibration(measured): all-W4A4 -> all-W8A8 predicts {100*rm:.1f} % of reference')
    # And the direction differs by CLASS, which is the whole finding: the measured
    # fit charges MORE for FP8 (k = 2.123 against the assumed 2.00) and LESS for
    # every weight-only format (3.54-3.56 against the assumed 4.00).
    chk(rm < r, 'measured: FP8 costs more than the two-class fit assumed')
    k = MEASURED_K['chat-c1']
    chk(k['sgl_fp8_pb_wo'] > COST_W8A8 and k['sgl_nvfp4a16'] < COST_W16A16
        and k['sgl_int4_g128'] < COST_W16A16,
        'measured: weight-only is cheaper than bf16, FP8 dearer than 2.00')
    chk(abs(k['sgl_nvfp4a16'] - k['sgl_int4_g128']) / k['sgl_bf16'] < 0.01,
        'measured: W4A16_NVFP4 and INT4-g128 are the same kernel class to <1 % of bf16')

    # 7. THE FLOOR - the anchor the budgets are fractions of, computed not imported
    els = {'a': 1000, 'b': 1000, 'e': 1000}
    sz = {'a': {'nv': 560, 'i4': 520, 'bf': 2000},
          'b': {'nv': 560, 'i4': 520, 'bf': 2000},
          'e': {'nv': 560, 'i4': 520, 'bf': 2000}}
    kk = {'nv': 1.0, 'i4': 3.5561, 'bf': 4.2465}
    mm = dense_multiplicity()
    units = [(['a'], ['nv', 'i4', 'bf']), (['b'], ['nv', 'i4', 'bf'])]
    fl = pool_floor(units, els, sz, mm, cost=kk)
    chk(abs(fl['prefill_index'] - 1.0) < 1e-12,
        f'floor: prefill floor takes the CHEAPEST KERNEL ({fl["prefill_index"]:.4f}), '
        f'not the fewest bytes - i4 is 40 B smaller and 3.56x slower')
    chk(fl['prefill_assign']['a'] == 'nv' and fl['decode_assign']['a'] == 'i4',
        'floor: the prefill floor and the decode floor are DIFFERENT recipes here')
    chk(fl['decode_bytes'] == 1040.0,
        f'floor: decode floor is 2 x 520 B ({fl["decode_bytes"]:,.0f})')
    # an embedding streams nothing, so its choice cannot move the decode floor
    fl_e = pool_floor(units + [(['token_embd.weight'], ['nv', 'i4', 'bf'])],
                      dict(els, **{'token_embd.weight': 9000}),
                      dict(sz, **{'token_embd.weight': {'nv': 5000, 'i4': 4600, 'bf': 18000}}),
                      mm, cost=kk)
    chk(fl_e['decode_bytes'] == fl['decode_bytes'],
        'floor: adding an embedding does not move the decode floor (it is a gather)')
    # THE BYTE FLOOR - the third one, and the one the `auto` decode budget is a
    # proportion of.  It is NOT the decode floor: the embedding costs its whole
    # size here and nothing there.
    chk(fl['bytes_floor'] == 1040.0,
        f'floor: byte floor is 2 x 520 B ({fl["bytes_floor"]:,.0f})')
    chk(fl_e['bytes_floor'] == 1040.0 + 4600,
        f'floor: the embedding DOES count towards the byte floor '
        f'({fl_e["bytes_floor"]:,.0f}), unlike the decode one')
    chk(all(total_bytes({'a': qa, 'b': qb}, sz) >= fl['bytes_floor'] - 1e-9
            for qa in ('nv', 'i4', 'bf') for qb in ('nv', 'i4', 'bf')),
        'floor: no legal recipe is smaller than the byte floor')
    chk(pool_floor([(['a'], ['nv'])], els, sz, mm, cost=kk,
                   fixed={'b': 'bf'})['bytes_floor'] == 560 + 2000,
        'floor: `fixed` bytes are in the byte floor too, so it and the size cap '
        'are on one basis')
    # a unit with one legal option contributes that option
    chk(abs(pool_floor([(['a'], ['bf'])], els, sz, mm, cost=kk)['prefill_index'] - 4.2465)
        < 1e-12, 'floor: a pinned unit contributes its pin, not the pool minimum')
    # the fixed part of the model is in both floors
    flf = pool_floor([(['a'], ['nv'])], els, sz, mm, cost=kk, fixed={'b': 'bf'})
    chk(abs(flf['prefill_index'] - (1.0 + 4.2465) / 2) < 1e-12,
        f'floor: `fixed` tensors are counted ({flf["prefill_index"]:.4f})')
    chk(flf['decode_bytes'] == 560 + 2000,
        f'floor: `fixed` bytes are streamed too ({flf["decode_bytes"]:,.0f})')
    # A 0 B ENTRY IS 0 B, because that is what the optimiser this floor bounds
    # believes.  Refusing such a qtype would make the floor sit above recipes the
    # optimiser actually returns - measured on GLM-4.7 (1.470 "floor" against a
    # 1.007 recipe).  The pool is phantom-filtered before it reaches here.
    szp = dict(sz); szp['a'] = {'nv': 560, 'i4': 0, 'bf': 2000}
    flp = pool_floor([(['a'], ['nv', 'i4', 'bf'])], els, szp, mm, cost=kk)
    chk(flp['decode_assign']['a'] == 'i4',
        'floor: a 0 B entry is 0 B - the floor uses the same arithmetic as the '
        'optimiser it bounds, or it is not a bound')
    # and the invariant that failure violated: no legal recipe beats the floor
    for q in ('nv', 'i4', 'bf'):
        chk(decode_bytes({'a': q}, szp, mm) >= flp['decode_bytes'] - 1e-9,
            f'floor: {q} does not beat the decode floor even with a 0 B entry')
        chk(prefill_index({'a': q}, els, mm, kk) >= flp['prefill_index'] - 1e-12,
            f'floor: {q} does not beat the prefill floor even with a 0 B entry')
    # and it IS a lower bound: no legal recipe beats it
    import itertools as _it
    worse = 0
    for combo in _it.product(['nv', 'i4', 'bf'], repeat=2):
        a = {'a': combo[0], 'b': combo[1]}
        if (prefill_index(a, els, mm, kk) < fl['prefill_index'] - 1e-12
                or decode_bytes(a, sz, mm) < fl['decode_bytes'] - 1e-9):
            worse += 1
    chk(worse == 0, 'floor: all 9 legal recipes over this pool, none beats either floor')

    # 7b. THE REPAIR PASS, and that its INCREMENTAL arithmetic is the SAME
    # arithmetic.  budget_repair() evaluates each candidate by adjusting running
    # totals instead of re-summing the model (it is the difference between 8.7 s
    # and 7 minutes for one pass on GLM-4.7), so the test that matters is a
    # brute-force reference implementation of the identical rule, run on the same
    # inputs, asserted to return the IDENTICAL assignment.
    r_els = {f't{i}': 1000 * (i + 1) for i in range(9)}
    r_sz = {t: {'nv': 560 * (i + 1), 'f8': 1000 * (i + 1), 'bf': 2000 * (i + 1)}
            for i, t in enumerate(sorted(r_els))}
    r_k = {'nv': 1.0, 'f8': 2.0221, 'bf': 4.2465}
    # Per-tensor loss deliberately NOT proportional to size.  Made proportional,
    # every candidate has the same quality-per-unit-of-violation and the whole
    # ranking is a tie at the 1e-15 level - at which point the two
    # implementations legitimately disagree about which of nine equal moves to
    # take, and the test would be measuring float addition order rather than the
    # rule.  Real calibration data spans decades; this spans one.
    r_loss = {t: 0.001 * ((i * 37) % 11 + 1) for i, t in enumerate(sorted(r_els))}
    r_units = ([([f't{i}'], ['nv', 'f8', 'bf']) for i in range(7)]
               + [(['t7', 't8'], ['nv', 'f8', 'bf'])])       # one fused pair
    r_fixed = {'norm': 'bf'}
    r_sz['norm'] = {'bf': 4096}
    r_els['norm'] = 2048
    r_deg = {'nv': 0.066930, 'f8': 0.005301, 'bf': 0.0}

    def _ref_repair(assign, p_bud, d_bud, b_bud, sz, k, dg, un):
        """The same rule, re-summing the whole model for every candidate."""
        a = dict(assign)
        fxn, fxd = prefill_terms(r_fixed, r_els, mm, k)
        fxb = decode_bytes(r_fixed, sz, mm)
        moves = 0

        def P(x):
            n, d = prefill_terms(x, r_els, mm, k)
            return ((n + fxn) / (d + fxd)) if (d + fxd) else float('nan')

        def D(x):
            return fxb + decode_bytes(x, sz, mm)
        for _ in range(10000):
            p_now, d_now = P(a), D(a)
            op = p_bud is not None and p_now > p_bud + 1e-12
            od = d_bud is not None and d_now > d_bud + 1e-9
            if not (op or od):
                break
            best = None
            for mem, pool in un:
                cur = a.get(mem[0])
                if cur is None or any(a.get(m) != cur for m in mem):
                    continue
                cd = sum((r_loss[m] ** 1.0) * dg[cur] for m in mem)
                for q in pool:
                    if q == cur:
                        continue
                    tr = dict(a)
                    for m in mem:
                        tr[m] = q
                    if b_bud is not None and total_bytes(tr, sz) > b_bud:
                        continue
                    g = 0.0
                    if op:
                        g += max(0.0, p_now - P(tr)) / max(1e-12, abs(p_bud))
                    if od:
                        g += max(0.0, d_now - D(tr)) / max(1.0, abs(d_bud))
                    if g <= 0:
                        continue
                    nd = sum((r_loss[m] ** 1.0) * dg[q] for m in mem)
                    v = g / (max(0.0, nd - cd) + 1e-18)
                    if best is None or v > best[0]:
                        best = (v, mem, q)
            if best is None:
                break
            for m in best[1]:
                a[m] = best[2]
            moves += 1
        return a, moves

    _start = {t: 'bf' for t in sorted(r_els) if t != 'norm'}
    _floor_r = pool_floor(r_units, r_els, r_sz, mm, cost=r_k, fixed=r_fixed)
    # THE BYTE CAP ONLY BINDS WHEN THE FASTEST TYPE IS NOT THE SMALLEST, which
    # the shipped pool cannot exhibit (NVFP4 is both) - so it is tested on the
    # synthetic pool where they disagree: 'i4' is 40 B/unit smaller than 'nv' and
    # 3.5x its kernel cost, so a prefill repair from all-'i4' wants to GROW the
    # checkpoint and the cap is what stops it.
    r_sz_i4 = {t: dict(v, **({'i4': 520 * (i + 1)} if t != 'norm' else {}))
               for i, (t, v) in enumerate(sorted(r_sz.items()))}
    r_k_i4 = dict(r_k, i4=3.5561)
    r_deg_i4 = dict(r_deg, i4=0.054767)
    r_units_i4 = [(m, p + ['i4']) for m, p in r_units]
    _start_i4 = {t: 'i4' for t in sorted(r_els) if t != 'norm'}
    for label, p_bud, d_bud, b_bud in (
            ('decode only', None, _floor_r['decode_bytes'] * 1.30, None),
            ('prefill only', _floor_r['prefill_index'] * 1.20, None, None),
            ('both', _floor_r['prefill_index'] * 1.15,
             _floor_r['decode_bytes'] * 1.20, None),
            ('byte cap binds', 1.5, None,
             total_bytes({t: 'i4' for t in _start_i4}, r_sz_i4)),
            ('impossible', 0.5, 1.0, None)):
        _bind = (label == 'byte cap binds')
        _sz, _k, _dg, _un, _st = ((r_sz_i4, r_k_i4, r_deg_i4, r_units_i4, _start_i4)
                                  if _bind else
                                  (r_sz, r_k, r_deg, r_units, _start))
        got, grep_ = budget_repair(dict(_st), _sz, r_els, mm, r_loss,
                                   lambda t, q: _dg[q], 1.0, _un,
                                   byte_budget=b_bud, prefill_budget=p_bud,
                                   decode_budget=d_bud, cost=_k, fixed=r_fixed)
        want, wmoves = _ref_repair(_st, p_bud, d_bud, b_bud, _sz, _k, _dg, _un)
        chk(got == want and grep_['applied'] == wmoves,
            f'repair ({label}): the incremental pass returns exactly what a '
            f'full-recompute reference returns ({grep_["applied"]} move(s))')
        _n, _d = prefill_terms(got, r_els, mm, _k)
        _fn, _fd = prefill_terms(r_fixed, r_els, mm, _k)
        chk(abs(grep_['prefill_after'] - (_n + _fn) / (_d + _fd)) < 1e-12
            and abs(grep_['decode_after']
                    - (decode_bytes(r_fixed, _sz, mm)
                       + decode_bytes(got, _sz, mm))) < 1e-6,
            f'repair ({label}): the reported after-values are the RECIPE\'s, '
            f'recomputed, not the running totals')
        if b_bud is not None:
            chk(total_bytes(got, _sz) <= b_bud,
                f'repair ({label}): the byte cap is never broken by a repair move')
        if p_bud == 0.5:
            chk(grep_['feasible'] is False,
                'repair (impossible): an unreachable budget is reported '
                'infeasible rather than silently met')

    # 8. THE CONVERSION - the migration aid, exact on every cell
    P_ref, P_floor = 1.290775291589933, 1.0029894106814001
    D_ref, D_floor = 17607772296.0, 14451558536.0
    for cell in sorted(MEASURED_AB):
        for old_x in (0.95, 0.85, 0.80, 0.60):
            xn = equivalent_prefill_budget(old_x, P_ref, P_floor, cell)
            chk(abs(index_for_ratio(xn, P_floor, cell)
                    - index_for_ratio(old_x, P_ref, cell)) < 1e-9,
                f'convert({cell}): prefill {old_x:.2f} of a ref -> {xn:.10f} of the '
                f'floor, identical cap')
        xl = equivalent_prefill_budget(0.95, P_ref, P_floor, cell, measured=False)
        chk(abs(index_for_ratio(xl, P_floor, cell, measured=False)
                - index_for_ratio(0.95, P_ref, cell, measured=False)) < 1e-9,
            f'convert({cell}): the legacy two-class path converts exactly too')
    for cell in sorted(MEASURED_DECODE):
        xd = equivalent_decode_budget(0.95, D_ref, D_floor, cell)
        chk(abs(bytes_for_ratio(xd, D_floor, cell)
                - bytes_for_ratio(0.95, D_ref, cell)) < 1e-3,
            f'convert({cell}): decode 0.95 of a ref -> {xd:.10f} of the floor, '
            f'identical cap')
    chk(abs(equivalent_prefill_budget(0.8, 3.0, 3.0) - 0.8) < 1e-12,
        'convert: identity when the reference IS the floor')
    # THE NAIVE RATIO IS WRONG, and by enough to matter: prefill time is a + b*P
    # with a = 47.23 on chat-c1, so scaling the fraction by P_floor/P_ref alone
    # misses by ~10 pp.  This test exists so nobody "simplifies" it back.
    naive = 0.95 * P_floor / P_ref
    exact = equivalent_prefill_budget(0.95, P_ref, P_floor, 'chat-c1')
    chk(abs(exact - naive) > 0.05,
        f'convert: the naive P_floor/P_ref ratio ({naive:.4f}) is NOT the answer '
        f'({exact:.4f}) - the fixed cost a > 0 is why')

    if verbose:
        print(('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})'))
    return 0 if not fails else 1


# =============================================================================
# CLI
# =============================================================================
#
# THE POOL-FLOOR INPUTS.  `pool_floor()` above needs three things this module
# does not otherwise read: per-format tensor sizes, the post-legality effective
# dtype of each (name, algo) pair, and the fused UNITS the floor is minimised
# over.  All three come out of a directory of `tensors.<algo>.map` files - the
# same files `--compute-all-map` hands the assigner - so the floor this file
# computes and the floor quant_assign.py computes are the same arithmetic.
# They live here rather than in a shared module because they are 60 lines of
# file parsing with one caller.

def synthesise_maps(bf16_map: str, out_dir: str, algos=None, arch=None) -> str:
    """Write tensors.<algo>.map for every algo into `out_dir`; return `out_dir`.

    The size model is `sglang_native.synthesise_map()`, i.e. exactly what
    `--compute-all-map` hands the assigner.
    """
    import sglang_native as SN
    arch = arch or SN.ARCH_QWEN3_5
    algos = list(algos or SN.SGLANG_POOL_ORDER)
    os.makedirs(out_dir, exist_ok=True)
    for algo in algos:
        SN.synthesise_map(bf16_map, algo, os.path.join(out_dir, f'tensors.{algo}.map'), arch)
    return out_dir


def load_maps(maps_dir: str):
    """(sizes, elements, effective_dtype, shapes) from a directory of maps.

    `effective_dtype[name][algo]` is what the tensor ACTUALLY gets when the
    recipe asks for `algo`: the map records the post-legality-fallback dtype, so
    `eff[name][algo] != algo` is precisely "illegal for this shape/role", and it
    is the only legality test the floor needs.
    """
    from collections import defaultdict
    sizes, elements, eff, shapes = defaultdict(dict), {}, defaultdict(dict), {}
    for fn in sorted(os.listdir(maps_dir)):
        if not (fn.startswith('tensors.') and fn.endswith('.map')):
            continue
        algo = fn[len('tensors.'):-len('.map')]
        with open(os.path.join(maps_dir, fn), 'r', encoding='utf-8') as fh:
            for line in fh:
                p = line.rstrip('\n').split(':')
                if len(p) < 5:
                    continue
                name = p[2]
                f = dict(x.split('=', 1) for x in p[3:] if '=' in x)
                sizes[name][algo] = int(f['bytes'])
                elements[name] = int(f['elements'])
                eff[name][algo] = f.get('dtype', algo)
                shapes[name] = tuple(int(x) for x in re.findall(r'-?\d+', f.get('shape', '')))
    if not sizes:
        raise SystemExit(f'{maps_dir}: no tensors.<algo>.map files')
    return dict(sizes), elements, dict(eff), shapes


def parse_recipe(path: str, bf16='sgl_bf16'):
    """{tensor -> qtype} from a quant_assign.py recipe, bf16/f32 normalised.

    The leading `!` on a computed-map qtype is provenance, not part of the name.
    """
    out = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            lhs, rhs = line.rsplit('=', 1)
            q = rhs.strip().lstrip('!').lower()
            if q in ('f32', 'bf16'):
                q = bf16
            lhs = lhs.strip()
            if lhs.startswith('^'):
                lhs = lhs[1:]
            if lhs.endswith('$'):
                lhs = lhs[:-1]
            out[re.sub(r'\\(.)', r'\1', lhs)] = q
    if not out:
        raise SystemExit(f'{path}: no recipe lines')
    return out


def legal_pool(members, pool, eff, pins, bf16='sgl_bf16'):
    """The qtypes a fused unit may take: its members' intersection, or its pin."""
    pinned = {pins[m] for m in members if m in (pins or {})}
    if len(pinned) > 1:
        raise SystemExit(f'contradictory pins on fused unit {members}: {sorted(pinned)}')
    if pinned:
        return sorted(pinned)
    ok = []
    for q in pool:
        if all((eff.get(m) or {}).get(q) == q for m in members):
            ok.append(q)
    return ok or [bf16]


def units_from(names, fused, pool, eff, pins=None, bf16='sgl_bf16'):
    """[(members, legal_pool)] - a fused module is ONE unit, else a singleton.

    THE UNIT IS THE FUSED MODULE, never the tensor: SGLang raises "Mixed
    quant_algo within fused layer" if the shards of one disagree
    (modelopt_quant.py:967), so a floor computed per tensor would be a floor no
    writeable recipe can reach.
    """
    grouped, out = set(), []
    for _gid, members in sorted((fused or {}).items()):
        ms = sorted(members)
        out.append((ms, legal_pool(ms, pool, eff, pins, bf16)))
        grouped.update(ms)
    for t in names:
        if t not in grouped:
            out.append(([t], legal_pool([t], pool, eff, pins, bf16)))
    return out


def _read_map(path):
    out = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            p = line.rstrip('\n').split(':')
            if len(p) < 5:
                continue
            f = dict(x.split('=', 1) for x in p[3:] if '=' in x)
            out[p[2]] = dict(shape=tuple(int(x) for x in re.findall(r'-?\d+', f.get('shape', ''))),
                             elements=int(f['elements']), bytes=int(f['bytes']),
                             dtype=f.get('dtype'))
    return out


def _budget_from_reference(a, ap) -> int:
    """Print the floor-relative budget that reproduces `FRACTION of RECIPE`.

    THIS IS THE ONLY FUNCTION IN THE SUITE THAT READS A REFERENCE RECIPE, and it
    reads one so that nothing else ever has to.  It exists because command lines
    were recorded against the old anchor and have to be replayable; run it once,
    paste the number, and the reference file is out of the loop for good.

    It builds the floor with exactly the unit and legality rules the assigner
    uses (`units_from` above), so the number it prints is the number the
    assigner will use - and it prints the ABSOLUTE cap alongside, for the cases
    where a fraction's last bit matters.
    """
    import sglang_native as SN
    path, frac = a.budget_from_reference
    try:
        old_x = float(frac)
    except ValueError:
        ap.error(f'--budget-from-reference: {frac!r} is not a fraction')
    maps_dir = a.maps_dir
    if not maps_dir:
        if not a.bf16_map:
            ap.error('--budget-from-reference needs --maps-dir (or --bf16-map, which is '
                     'expanded into one)')
        maps_dir = os.path.join(os.path.dirname(os.path.abspath(a.bf16_map)), 'sglang_maps')
        synthesise_maps(a.bf16_map, maps_dir,
                        arch=SN.ARCHS[a.arch] if a.arch else None)
    sizes, els, eff, shapes = load_maps(maps_dir)
    names = sorted(sizes)
    arch = a.arch or SN.detect_arch(names)
    if not arch:
        ap.error(f'could not identify the architecture of {maps_dir}; pass --arch '
                 f'(registered: {", ".join(sorted(SN.ARCHS))})')
    archd = SN.ARCHS[arch]
    pool = [q for q in (a.pool or list(DEFAULT_POOL)) if q in SN.SGLANG_ALGOS]
    if not pool:
        ap.error('--pool must name SGLang-native qtypes')
    # the pins the assigner applies: roles that are never assigned, then
    # whatever the run pinned by hand
    pins = {}
    for t in names:
        info = SN.map_tensor(t, archd)
        if info is None or info[1] == 'other' or len(shapes.get(t, ())) < 2:
            pins[t] = SN.BF16
    for spec in (a.pin or []):
        pat, _, q = spec.rpartition('=')
        try:
            rx = re.compile(pat)
        except re.error:
            ap.error(f'--pin {spec!r}: not a regex')
        for t in names:
            if rx.match(t) or rx.search(t):
                pins[t] = q
    mult = (moe_multiplicity(a.moe_top_k, a.moe_n_experts, a.moe_experts_re,
                             lm_head_tokens=a.lm_head_tokens)
            if a.moe_top_k else dense_multiplicity(lm_head_tokens=a.lm_head_tokens))
    units = units_from(names, SN.fused_groups(names, archd), pool, eff, pins, SN.BF16)
    floor = pool_floor(units, els, sizes, mult, cell=a.cell)
    _ref_recipe = parse_recipe(path, SN.BF16)
    ref = {t: _ref_recipe.get(t, SN.BF16) for t in names}
    P_ref = prefill_index(ref, els, mult, cell=a.cell)
    D_ref = decode_bytes(ref, sizes, mult)
    P_fl, D_fl = floor['prefill_index'], floor['decode_bytes']
    xp = equivalent_prefill_budget(old_x, P_ref, P_fl, a.cell)
    xd = equivalent_decode_budget(old_x, D_ref, D_fl, a.cell)
    print(f'reference   {os.path.basename(path)}   cell {a.cell}   arch {arch}')
    print(f'  pool                 {" ".join(pool)}')
    print(f'  reference prefill    {P_ref!r}')
    print(f'  reference decode     {D_ref:,.0f} B/token')
    print(f'pool floor (computed - this is what replaces the reference)')
    print(f'  prefill index        {P_fl!r}')
    print(f'  decode bytes/token   {D_fl!r}')
    print(f'\nold: --prefill-budget {old_x:g} --decode-budget {old_x:g} '
          f'--speed-reference {os.path.basename(path)}')
    print(f'new: --prefill-budget {xp!r} --decode-budget {xd!r}')
    print(f'     (or, bit-exact and anchor-free: '
          f'--prefill-max-index {index_for_ratio(xp, P_fl, a.cell)!r} '
          f'--decode-max-bytes {bytes_for_ratio(xd, D_fl, a.cell)!r})')
    print(f'\ncaps, which are what actually constrain the assigner:')
    print(f'  prefill index <= {index_for_ratio(old_x, P_ref, a.cell)!r}  (old)')
    print(f'  prefill index <= {index_for_ratio(xp, P_fl, a.cell)!r}  (new)')
    print(f'  decode <= {bytes_for_ratio(old_x, D_ref, a.cell):,.3f} B/token  (old)')
    print(f'  decode <= {bytes_for_ratio(xd, D_fl, a.cell):,.3f} B/token  (new)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--candidates', action='store_true',
                    help='print the candidate-format verdict table')
    ap.add_argument('--bf16-map', help='tensors.bf16.map for the share tables')
    ap.add_argument('--maps-dir', help='directory of synthesised tensors.<algo>.map')
    ap.add_argument('--recipe', help='recipe to score')
    ap.add_argument('--table', action='store_true',
                    help='print the per-family prefill-FLOP / decode-byte share table')
    ap.add_argument('--moe-top-k', type=int, default=None)
    ap.add_argument('--moe-n-experts', type=int, default=None)
    ap.add_argument('--moe-experts-re', default=r'experts\.\d+\.')
    ap.add_argument('--lm-head-tokens', type=float, default=LMHEAD_TOKENS_DEFAULT)
    ap.add_argument('--cell', default=DEFAULT_CELL, choices=sorted(MEASURED_W))
    ap.add_argument('--score', nargs='+', metavar='RECIPE',
                    help='score recipes on the two-sided model (needs --maps-dir)')
    ap.add_argument('--reference', metavar='RECIPE',
                    help='the recipe --score and --frontier report ratios against')
    ap.add_argument('--frontier', metavar='RECIPE',
                    help='bytes saved vs predicted prefill loss, moving units of this '
                         'recipe to --frontier-target (needs --maps-dir)')
    ap.add_argument('--frontier-target', default='ct_w4a16_g64')
    ap.add_argument('--frontier-bpw', type=float, default=4.25)
    ap.add_argument('--refit', action='store_true',
                    help='re-derive MEASURED_W from MEASURED_THROUGHPUT and print residuals '
                         '(needs --bf16-map, --maps-dir and --refit-recipes)')
    ap.add_argument('--refit-recipes', nargs='*', metavar='NAME=RECIPE',
                    help='artefact name = recipe path, for --refit; the two CALIBRATE uniforms '
                         'are synthesised from the pool when named fp8pbwo / bf16')
    ap.add_argument('--frontier-linear-only', action='store_true', default=True,
                    help='restrict the walk to linear weights (default): an embedding has no '
                         'W4A16 method in SGLang and in_proj_ba must stay BF16')
    # ---- the migration aid, and the ONLY place in the suite that reads a
    #      reference recipe.  Nothing calls it; it prints and exits. -----------
    ap.add_argument('--budget-from-reference', nargs=2, metavar=('RECIPE', 'FRACTION'),
                    help='MIGRATION AID for command lines recorded before the budgets '
                         'became fractions of the computed pool floor. Prints the '
                         'floor-relative --prefill-budget / --decode-budget (and the '
                         'absolute --prefill-max-index / --decode-max-bytes) that give '
                         'the IDENTICAL cap to FRACTION of RECIPE. The assigner never '
                         'reads a reference recipe; this prints a number you paste onto '
                         'a command line once. Needs --maps-dir (or --bf16-map) plus '
                         '--pool/--arch/--pin describing the run being converted.')
    ap.add_argument('--pool', nargs='+', default=None,
                    help='the --gpu-quants pool the floor is computed over '
                         '(--budget-from-reference)')
    ap.add_argument('--arch', default=None,
                    help='architecture name for the role table (default: detected)')
    ap.add_argument('--pin', nargs='+', default=[],
                    help="pins as 'REGEX=qtype', the same syntax as --gpu-assign-tensors")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.budget_from_reference:
        return _budget_from_reference(a, ap)
    if a.candidates:
        print(f'{"format":18s} {"bpw":>8s} {"prefill":>8s} {"container":22s} status')
        for k, v in CANDIDATES.items():
            bpw = f'{v["bpw"]:.5g}' if v.get('bpw') else '-'
            c = v.get('cost')
            cs = ('-' if c is None else ('THROWS' if c == float('inf') else f'{c:.2f}'))
            print(f'{k:18s} {bpw:>8s} {cs:>8s} {v["container"]:22s} {v["status"]}')
            for line in re.findall(r'.{1,88}(?:\s|$)', v['note']):
                print(f'    {line.strip()}')
        return 0
    if not a.bf16_map:
        ap.error('--bf16-map is required')
    rows = _read_map(a.bf16_map)
    els = {k: v['elements'] for k, v in rows.items()}
    mult = (moe_multiplicity(a.moe_top_k, a.moe_n_experts, a.moe_experts_re,
                             lm_head_tokens=a.lm_head_tokens)
            if a.moe_top_k else dense_multiplicity(lm_head_tokens=a.lm_head_tokens))
    def _load_recipe(path):
        out = {}
        for line in open(path, 'r', encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            lhs, rhs = line.rsplit('=', 1)
            lhs = lhs.strip()
            if lhs.startswith('^'):
                lhs = lhs[1:]
            if lhs.endswith('$'):
                lhs = lhs[:-1]
            q = rhs.strip().lstrip('!').lower()
            out[lhs.replace('\\.', '.').replace('\\', '')] = ('sgl_bf16' if q in ('f32', 'bf16') else q)
        for t in rows:
            out.setdefault(t, 'sgl_bf16')
        return out

    def _load_sizes():
        if not a.maps_dir:
            ap.error('--maps-dir is required for --score/--frontier')
        sz = {}
        for fn in sorted(os.listdir(a.maps_dir)):
            if not (fn.startswith('tensors.') and fn.endswith('.map')):
                continue
            algo = fn[len('tensors.'):-len('.map')]
            for k, v in _read_map(os.path.join(a.maps_dir, fn)).items():
                sz.setdefault(k, {})[algo] = v['bytes']
        for k in sz:                       # the candidate lanes, priced by bpw
            for cand, bpw in (('ct_w4a16_g64', 4.25), ('ct_w4a16_g128', 4.125),
                              ('moe_mxfp4', 4.25)):
                sz[k].setdefault(cand, int(round(els[k] * bpw / 8.0)))
        return sz

    if a.refit:
        sz = None
        shares = {}
        def _units():
            return {t: els[t] * mult.prefill(t) for t in rows}
        u = _units(); U = sum(u.values())
        def _sh(asg):
            d = dict(w4=0.0, w8=0.0, w16=0.0)
            for t, q in asg.items():
                c = PREFILL_COST.get(q)
                if c is None or u.get(t, 0) <= 0:
                    continue
                k = 'w4' if c <= 1.0 else ('w8' if c <= 2.5 else 'w16')
                d[k] += u[t] / U
            return d
        def _rec(path):
            out = {}
            for line in open(path, 'r', encoding='utf-8'):
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                lhs, rhs = line.rsplit('=', 1)
                lhs = lhs.strip().lstrip('^').rstrip('$')
                q = rhs.strip().lstrip('!').lower()
                out[lhs.replace('\\.', '.').replace('\\', '')] = ('sgl_bf16' if q in ('f32', 'bf16') else q)
            for t in rows:
                out.setdefault(t, 'sgl_bf16')
            return out
        for spec in (a.refit_recipes or []):
            name, _, path = spec.partition('=')
            shares[name] = _sh(_rec(path))
        # the two CALIBRATE uniforms, from their published boundaries
        if 'fp8pbwo' not in shares:
            asg = {t: ('sgl_bf16' if (mult.kind(t) != 'linear'
                                      or re.match(r'^blk\.\d+\.ssm_(alpha|beta)\.weight$', t))
                       else 'sgl_fp8_pb_wo') for t in rows}
            shares['fp8pbwo'] = _sh(asg)
        if 'bf16' not in shares:
            shares['bf16'] = _sh({t: 'sgl_bf16' for t in rows})
        print('refit of MEASURED_W from MEASURED_THROUGHPUT:')
        refit(shares, None)
        return 0

    if a.score or a.frontier:
        sz = _load_sizes()
        ref = _load_recipe(a.reference) if a.reference else None
        Pr = prefill_index(ref, els, mult) if ref else None
        Dr = decode_bytes(ref, sz, mult) if ref else None
        E = sum(els.values())
        if a.score:
            print(f'{"recipe":36s} {"bytes":>16s} {"bpw":>7s} {"P":>7s} '
                  f'{"decode B/tok":>16s} {"prefill":>8s} {"decode":>8s}')
            for r in a.score:
                asg = _load_recipe(r)
                P = prefill_index(asg, els, mult); D = decode_bytes(asg, sz, mult)
                B = total_bytes(asg, sz)
                pr = f'{100*predicted_prefill_ratio(P, Pr, a.cell):7.1f}%' if Pr else '      -'
                dr = f'{100*predicted_decode_ratio(D, Dr, a.cell):7.1f}%' if Dr else '      -'
                print(f'{os.path.basename(r)[:36]:36s} {B:>16,} {B*8/E:7.4f} {P:7.4f} '
                      f'{D:>16,.0f} {pr} {dr}')
        if a.frontier:
            asg = _load_recipe(a.frontier)
            try:
                import sglang_native as SN
                fg = SN.fused_groups(list(rows))
                grouped = set(); un = []
                for _g, mem in sorted(fg.items()):
                    un.append((sorted(mem), None)); grouped.update(mem)
                def _ok(t):
                    if not a.frontier_linear_only:
                        return True
                    i = SN.map_tensor(t)
                    return (i is not None and i[1] == 'linear'
                            and not re.match(r'^blk\.\d+\.ssm_(alpha|beta)\.weight$', t))
                un += [([t], None) for t in rows if t not in grouped]
                un = [(m, p) for (m, p) in un if all(_ok(x) for x in m)]
            except Exception:
                un = None
            rws, meta = frontier(asg, sz, els, mult, a.frontier_target, a.frontier_bpw,
                                 cell=a.cell, units=un)
            print(f'frontier: {os.path.basename(a.frontier)} -> {a.frontier_target} '
                  f'@ {a.frontier_bpw} bpw, cell {a.cell}, {meta["candidates"]} eligible units')
            print(f'  {"prefill loss":>12s} {"units":>6s} {"FLOPshare":>10s} {"bytes saved":>15s} '
                  f'{"new size":>16s} {"new bpw":>8s} {"decode":>8s}')
            for r in rws:
                print(f'  {r["loss_pct"]:11.0f}% {r["units"]:6d} {100*r["flop_share_moved"]:9.2f}% '
                      f'{r["bytes_saved"]:>15,} {r["new_bytes"]:>16,} {r["new_bytes"]*8/E:8.4f} '
                      f'{100*r["predicted_decode_ratio"]:7.1f}%')
        return 0

    if a.table:
        fam = {}
        for k, v in rows.items():
            key = re.sub(r'\.\d+\.', '.N.', k)
            key = re.sub(r'experts\.\d+\.', 'experts.E.', key)
            d = fam.setdefault(key, dict(n=0, e=0, u=0.0, kind=mult.kind(k)))
            d['n'] += 1
            d['e'] += v['elements']
            d['u'] += v['elements'] * mult.prefill(k)
        E = sum(d['e'] for d in fam.values())
        U = sum(d['u'] for d in fam.values())
        print(f'{"family":34s} {"n":>5s} {"elements":>16s} {"elem%":>7s} '
              f'{"mult":>7s} {"FLOP%":>7s} {"FLOP/byte":>9s}')
        for k in sorted(fam, key=lambda x: -fam[x]['u']):
            d = fam[k]
            fpb = ((d['u'] / U) / (d['e'] / E)) if (U and d['e']) else 0.0
            print(f'{k:34s} {d["n"]:5d} {d["e"]:>16,} {100*d["e"]/E:7.3f} '
                  f'{(d["u"]/d["e"] if d["e"] else 0):7.4f} {100*d["u"]/U:7.3f} {fpb:9.4f}')
        print(f'\ntotal elements {E:,}   prefill FLOP units {U:,.0f}')
        return 0
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
