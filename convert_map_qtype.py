#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** convert_map_qtype.py converts any tensors .map file to a  **#
#** different .map file qtype.                                **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Feb-11-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.    ᵢ ₖₙₒ𝓌 ₜₕᵢₛ ₛₜₑₐₖ 𝒹ₒₑₛₙ'ₜ ₑₓᵢₛₜ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
convert_map_qtype.py

Convert a .map file's shard names, hashes and tensor bytes according to a target quantization type (--qtype).

Usage:
    python convert_map_qtype.py path/to/input.map --qtype iq1_s_r4
    python convert_map_qtype.py path/to/input.map --qtype iq1_s_r4 --no-map   # prints to stdout instead of writing file

Useful tip: Run the following commands to capture the asserts of each quantization function
cat ./ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp ./ik_llama.cpp/ggml/src/ggml-quants.c | egrep 'void quantize_row_|size_t quantize_|GGML_ASSERT|assert|^\}' | sed -n -e '/^[size_t|void]/,/^\}/ p' | grep -v ' *//'
# Direct way to extract all asserts:
for q in $(cat ./ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp ./ik_llama.cpp/ggml/src/ggml-quants.c | egrep -i 'void quantize_row_|size_t quantize_|assert|^\}' | sed -n -e '/^[size_t|void]/,/^\}/ p' | grep -v ' *//' | grep -v '^            ' | grep ' quantize_' | cut -d'_' -f 3-6 | cut -d'(' -f 1 | sed 's/_ref//g' | sed 's/_impl//g' | egrep -v 'bs128|bs16|_K128|_K16|_K32|_K64|_KR8|_T' | sed 's/ //g' | sort -u); do echo $q:; cat ./ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp ./ik_llama.cpp/ggml/src/ggml-quants.c | egrep -v 'bs128|bs16|_K128|_K16|_K32|_K64|_KR8|_T' | egrep -v -i "$(exclude="" && for r in _r4 _r8 _r16; do if ! [[ "${q,,}" =~ "$r" ]]; then exclude="$exclude""$r "; fi done && echo $exclude | tr ' ' '|')" | egrep -i "void quantize_row_$q|size_t quantize_$q|assert|kBlockSize = |^\}" | sed -n -e '/^[size_t|void]/,/^\}/ p' | grep -v ' *//' | egrep -v '^            |^        |ggml_quantize_init()|missing quantization weights|must be|max_scale|static_assert|assert\(Q|GGML_ASSERT\(quant_weights\)' | egrep -i 'assert|kBlockSize = ' | sed 's/assert(k /GGML_ASSERT(n_per_row /g' | sed 's/ % /%/g' | sed 's/%/ % /g' | sed 's/GGML_ASSERT(n %/GGML_ASSERT(n_per_row %/g' | sed 's/    assert(/    GGML_ASSERT(/g' | sed 's/GGML_ASSERT(k /GGML_ASSERT(n_per_row /g' | sort -ur; done
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Tuple, Dict, List

# GGML quant sizes mapping as provided (kept same capitalization as given)
GGML_QUANT_SIZES: Dict[str, Tuple[int, int]] = {
    "F32" : (   1,    4),
    "F16" : (   1,    2),
    "Q4_0" : (  32,   18),
    "Q4_1" : (  32,   20),
    "Q5_0" : (  32,   22),
    "Q5_1" : (  32,   24),
    "Q8_0" : (  32,   34),
    "Q8_1" : (  32,   36),
    "Q2_K" : ( 256,   84),
    "Q3_K" : ( 256,  110),
    "Q4_K" : ( 256,  144),
    "Q5_K" : ( 256,  176),
    "Q6_K" : ( 256,  210),
    "Q8_K" : ( 256,  292),
    "IQ2_XXS" : ( 256,   66),
    "IQ2_XS" : ( 256,   74),
    "IQ3_XXS" : ( 256,   98),
    "IQ1_S" : ( 256,   50),
    "IQ4_NL" : (  32,   18),
    "IQ3_S" : ( 256,  110),
    "IQ2_S" : ( 256,   82),
    "IQ4_XS" : ( 256,  136),
    "I8" : (   1,    1),
    "I16" : (   1,    2),
    "I32" : (   1,    4),
    "I64" : (   1,    8),
    "F64" : (   1,    8),
    "IQ1_M" : ( 256,   56),
    "BF16" : (   1,    2),
    "MXFP4" : (  32,   17),
    "Q4_0_4_4" : (  32,   18),
    "Q4_0_4_8" : (  32,   18),
    "Q4_0_8_8" : (  32,   18),
    "I2_S" : (   1,    1),
    "Q8_0_X4" : (  32,   34),
    "Q8_1_X4" : (  32,   36),
    "Q8_2_X4" : (  32,   36),
    "Q6_0" : (  32,   26),
    "IQ1_BN" : (  64,   13),
    "IQ2_BN" : (  64,   16),
    "Q8_K64" : (  64,   68),
    "IQ2_K" : ( 256,   76),
    "IQ3_K" : ( 256,  110),
    "IQ4_K" : ( 256,  144),
    "IQ5_K" : ( 256,  176),
    "IQ6_K" : ( 256,  212),
    "IQ4_KS" : ( 256,  136),
    "IQ2_KS" : ( 256,   70),
    "IQ4_KSS" : ( 256,  128),
    "Q8_K16" : (  64,   64),
    "Q8_K32" : ( 256,  292),
    "Q8_KR8" : ( 256,  292),
    "Q8_K128" : ( 128,  140),
    "Q8_KV" : (  32,   32),
    "IQ5_KS" : ( 256,  168),
    "IQ2_KT" : ( 256,   68),
    "IQ3_KT" : ( 256,  100),
    "IQ4_KT" : ( 256,  128),
    "IQ3_KS" : ( 256,  102),
    "IQ2_KL" : ( 256,   86),
    "IQ1_KT" : ( 256,   56),
    "Q4_0_R8" : (  32,   18),
    "Q5_0_R4" : (  32,   22),
    "Q8_0_R8" : (  32,   34),
    "Q2_K_R4" : ( 256,   84),
    "Q3_K_R4" : ( 256,  110),
    "Q4_K_R4" : ( 256,  144),
    "Q5_K_R4" : ( 256,  176),
    "Q6_K_R4" : ( 256,  210),
    "IQ2_XXS_R4" : ( 256,   66),
    "IQ2_XS_R4" : ( 256,   74),
    "IQ3_XXS_R4" : ( 256,   98),
    "IQ1_S_R4" : (  32,    6),
    "IQ4_NL_R4" : (  32,   18),
    "IQ3_S_R4" : ( 256,  110),
    "IQ2_S_R4" : ( 256,   82),
    "IQ4_XS_R8" : ( 256,  136),
    "IQ1_M_R4" : (  32,    7),
    "BF16_R16" : (   1,    2),
    "Q6_0_R4" : (  32,   26),
    "IQ2_BN_R4" : (  64,   16),
    "IQ2_K_R4" : ( 256,   76),
    "IQ3_K_R4" : ( 256,  110),
    "IQ4_K_R4" : ( 256,  144),
    "IQ5_K_R4" : ( 256,  176),
    "IQ4_KS_R4" : ( 256,  136),
    "IQ5_KS_R4" : ( 256,  168),
    "Q8_KV_R8" : (  32,   32),
    "Q8_K_R8" : ( 256,  258),
}

# --- BPW lookup table for GGUF quant dtypes ---
BPW_TABLE = {
    'F32': 32,
    'F16': 16,
    'BF16': 16,
    'Q8_0_R8': 8.5,
    'Q8_0': 8.5,
    'Q8_K_R8': 8.0625,
    'Q8_KV': 8,
    'F8': 8,
    'IQ6_K': 6.625,
    'Q6_K_R4': 6.5625,
    'Q6_K': 6.5625,
    'Q6_0_R4': 6.5,
    'Q6_0': 6.5,
    'Q5_1': 6,
    'Q5_K_R4': 5.5,
    'Q5_K': 5.5,
    'Q5_0_R4': 5.5,
    'Q5_0': 5.5,
    'IQ5_K_R4': 5.5,
    'IQ5_K': 5.5,
    'IQ5_KS_R4': 5.25,
    'IQ5_KS': 5.25,
    'Q4_1': 5,
    'Q4_K_R4': 4.5,
    'Q4_K': 4.5,
    'Q4_0_R8': 4.5,
    'Q4_0': 4.5,
    'IQ4_NL_R4': 4.5,
    'IQ4_NL': 4.5,
    'IQ4_K_R4': 4.5,
    'IQ4_K': 4.5,
    'IQ4_XS_R8': 4.25,
    'IQ4_XS': 4.25,
    'IQ4_KS_R4': 4.25,
    'IQ4_KS': 4.25,
    'IQ4_KT': 4,
    'IQ4_KSS': 4,
    'IQ3_KL': 4,
    'IQ3_M': 3.66,
    'Q3_K_R4': 3.4375,
    'Q3_K': 3.4375,
    'IQ3_S_R4': 3.4375,
    'IQ3_S': 3.4375,
    'IQ3_K_R4': 3.4375,
    'IQ3_K': 3.4375,
    'IQ3_XS': 3.3,
    'IQ3_KS': 3.1875,
    'IQ3_KT': 3.125,
    'IQ3_XXS_R4': 3.0625,
    'IQ3_XXS': 3.0625,
    'IQ2_M_R4': 2.7,
    'IQ2_M': 2.7,
    'IQ2_KL': 2.6875,
    'Q2_K_R4': 2.625,
    'Q2_K': 2.625,
    'IQ2_S': 2.5625,
    'IQ2_K_R4': 2.375,
    'IQ2_K': 2.375,
    'IQ2_XS_R4': 2.3125,
    'IQ2_XS': 2.3125,
    'IQ2_KS': 2.1875,
    'IQ2_KT': 2.125,
    'IQ2_XXS_R4': 2.0625,
    'IQ2_XXS': 2.0625,
    'IQ2_BN_R4': 2,
    'IQ2_BN': 2,
    'IQ1_M_R4': 1.75,
    'IQ1_M': 1.75,
    'IQ1_KT': 1.75,
    'IQ1_BN': 1.625,
    'IQ1_S': 1.5625,
    'IQ1_S_R4': 1.5
}


class TransformFailure(Exception):
    """Raised when a tensor cannot be transformed to a requested qtype."""
    def __init__(self, tensor_name: str, requested_qtype: str, reason: str):
        super().__init__(reason)
        self.tensor_name = tensor_name
        self.requested_qtype = requested_qtype
        self.reason = reason


def failed_to_transform(tensor_name: str, requested_qtype: str, reason: str):
    """
    Print the reason for failing to transform a tensor and raise TransformFailure.

    The caller may catch TransformFailure and either abort the script or attempt a fallback
    selection. Exits only when the caller chooses not to handle the exception.
    """
    print(f"⚠️ Failed to transform tensor '{tensor_name}' to qtype '{requested_qtype}':", reason, file=sys.stderr)
    raise TransformFailure(tensor_name, requested_qtype, reason)


def tensor_size(elements: int, qtype_upper: str) -> int:
    """
    Compute new bytes for a tensor with `elements` elements when
    quantized to `qtype_upper` (uppercase key in GGML_QUANT_SIZES).

    Formula: new_bytes = elements * type_size // block_size
    We return an integer by using integer division (floor).
    """
    if qtype_upper not in GGML_QUANT_SIZES:
        raise KeyError(f"qtype '{qtype_upper}' not in GGML_QUANT_SIZES")
    block_size, type_size = GGML_QUANT_SIZES[qtype_upper]
    # use integer arithmetic
    return (elements * type_size) // block_size


def parse_kv_pairs(kv_list):
    """
    Given a list like ['shape=(2560, 151936)', 'dtype=bf16', 'elements=388956160', 'bytes=85084160']
    return list of (key, value) pairs preserving order.
    """
    pairs = []
    for item in kv_list:
        if '=' in item:
            k, v = item.split('=', 1)
            pairs.append((k, v))
        else:
            # fallback: if no '=' keep whole string as key with empty value
            pairs.append((item, ''))
    return pairs


def reconstruct_kv_string(pairs):
    """
    Given list of (k,v) pairs produce the colon-separated list of key=value strings
    """
    return ':'.join([f"{k}={v}" if v != '' else k for k, v in pairs])


def is_row_interleaved_qtype(qtype_lower: str) -> bool:
    """
    Determine if the requested quantization type is 'row-interleaved' per the user's rule:
    - ends with r4, r8 or r16
    - or contains _4_4, _4_8 or _8_8
    """
    if qtype_lower.endswith(('r4', 'r8', 'r16')):
        return True
    if any(sub in qtype_lower for sub in ('_4_4', '_4_8', '_8_8')):
        return True
    return False


def non_row_variant(qtype_upper: str) -> str:
    """
    Return the non-row-interleaved variant of a qtype if it exists.
    e.g. IQ2_K_R4 -> IQ2_K
    """
    # Remove suffixes _R4/_R8/_R16 if present
    for suf in ('_R4', '_R8', '_R16'):
        if qtype_upper.endswith(suf):
            return qtype_upper[: -len(suf)]
    # Also handle lowercase style (should be uppercase already)
    return qtype_upper


# -------------------------
# Shape/blocking constraints derived from GGML_ASSERTs captured in ik_llama.cpp
# -------------------------
# constants used in asserts
QK_K = 256
QK_IQ1BN = 64
QK_IQ2BN = 64
KBLOCKSIZE_32 = 32
QK_MXFP4 = 32  # corresponds to QK_MXFP4 used in mxfp4 assert
QK4_NL = 32

# Additional QK constants derived from the GGML_ASSERT defines referenced
# in the C asserts (all set to 32 as per the defines provided).
QK4_0 = 32
QK4_1 = 32
QK5_0 = 32
QK5_1 = 32
QK6_0 = 32
QK8_0 = 32
QK8_1 = 32

# Explicit list of quant types (non-row variants) that required `n_per_row % QK_K == 0` in the asserts
# This list was built to reflect the functions captured from ik_llama.cpp.
# It includes both GGML-style names and the non-row variant names that earlier asserts used QK_K for.
PER_ROW_QK_K_TYPES = {
    # K-block types
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    # IQ K variants
    "IQ2_S", "IQ2_K", "IQ2_KS", "IQ2_KL", "IQ3_K", "IQ3_KS", "IQ4_K", "IQ5_K", "IQ6_K",
    "IQ4_KS", "IQ5_KS", "IQ4_KSS",
    # Trellis families referenced in asserts
    "IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT",
    # small/xxs/xs families (these were validated in some asserts to require QK_K)
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ4_XS", # Note: IQ3_XS doesn't exist
    # IQ1 family (some asserts used QK_K for iq1_s/iq1_m without R4)
    "IQ1_S", "IQ1_M",
    # Q8-related that also had QK assertions in row variants
    "Q8_K", "Q8_K32", "Q8_K32",  # included generally; duplicates harmless
}

# Helper to check and enforce asserts based on qtype and the parsed shape
def check_shape_constraints(qtype_upper: str, qtype_lower: str, nrows: int or None, n_per_row: int or None, tensor_name: str, ignore_imatrix_rules: bool, imatrix: bool):
    """
    Enforce the blocking/shape constraints observed in ik_llama.cpp asserts.
    If a constraint fails, call failed_to_transform(...) to raise TransformFailure.

    Rules implemented (based on the asserts provided):
     - row-suffix rules: r4 -> nrows % 4 == 0; r8 -> nrows % 8 == 0; r16 -> nrows % 16 == 0
     - many "K" types require n_per_row % QK_K == 0 (we check non-row variant against PER_ROW_QK_K_TYPES)
     - _BN types (importance-batch normalization) require n_per_row % QK_IQ1BN == 0
     - MXFP4 requires n_per_row % QK_MXFP4 == 0
     - IQ1_S_R4 and IQ1_M_R4 require n_per_row % 32 == 0
     - special Q8_KV_R8 requires nrows%8==0 and n_per_row%16==0
     - special Q8_K_R8 / Q8_K_R16 require nrows%8/16 and n_per_row%QK_K
     - NOTE: shape dimension interpretation: for shapes like (A, B) in the .map file we treat the
             FIRST numeric value as n_per_row and the SECOND as nrows. This matches the ordering
             observed in the examples token_embd.weight:shape=(2560, 151936) where the
             per-row/blocking requirement of 256 applies to 2560.
    """
    # nothing to check if no shape info
    if nrows is None and n_per_row is None:
        return

    # Row-suffix / interleaved requirements: r4/r8/r16 (these operate on nrows)
    if qtype_lower.endswith('r4'):
        if nrows is None or (nrows % 4) != 0:
            reason = f"Quant '{qtype_lower}' requires nrows % 4 == 0, got nrows={nrows!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif qtype_lower.endswith('r8'):
        if nrows is None or (nrows % 8) != 0:
            reason = f"Quant '{qtype_lower}' requires nrows % 8 == 0, got nrows={nrows!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif qtype_lower.endswith('r16'):
        if nrows is None or (nrows % 16) != 0:
            reason = f"Quant '{qtype_lower}' requires nrows % 16 == 0, got nrows={nrows!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # Specific Q8_KV_R8: requires nrows%8==0 and n_per_row%16==0
    if qtype_upper == "Q8_KV_R8":
        if n_per_row is None or (n_per_row % 16) != 0:
            reason = f"'{qtype_upper}' requires n_per_row % 16 == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # BN / per-channel BN types require particular per-row block size
    if '_BN' in qtype_upper:
        # IQ1_BN and IQ2_BN style types use QK_IQ1BN = 64 and QK_IQ2BN = 64 in asserts
        if n_per_row is None or (n_per_row % QK_IQ1BN) != 0:
            reason = f"Quant '{qtype_upper}' requires n_per_row % {QK_IQ1BN} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # MXFP4 specific requirement
    if 'MXFP4' in qtype_upper:
        if n_per_row is None or (n_per_row % QK_MXFP4) != 0:
            reason = f"Quant '{qtype_upper}' requires n_per_row % {QK_MXFP4} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # IQ1_S_R4 / IQ1_M_R4 require n_per_row % 32 == 0 (kBlockSize in asserts)
    if qtype_upper in ("IQ1_S_R4", "IQ1_M_R4"):
        if n_per_row is None or (n_per_row % KBLOCKSIZE_32) != 0:
            reason = f"Quant '{qtype_upper}' requires n_per_row % {KBLOCKSIZE_32} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # Many K-style quantizers require n_per_row % QK_K == 0.
    # Use the non-row-interleaved variant to check whether this qtype belongs to that family.
    base = non_row_variant(qtype_upper)
    if base in PER_ROW_QK_K_TYPES:
        if n_per_row is None or (n_per_row % QK_K) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK_K} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # IQ4_NL require n_per_row % QK4_NL == 0.
    # Use the non-row-interleaved variant to check whether this qtype belongs to that family.
    base = non_row_variant(qtype_upper)
    if 'IQ4_NL' in qtype_upper:
        if n_per_row is None or (n_per_row % QK4_NL) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK4_NL} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # Additional per-base checks derived from these GGML_ASSERT defines:
    #   #define QK4_0 32
    #   #define QK4_1 32
    #   #define QK5_0 32
    #   #define QK5_1 32
    #   #define QK6_0 32
    #   #define QK8_0 32
    #   #define QK8_1 32
    # When the non-row base matches the corresponding q4_0/q4_1/... variants, enforce n_per_row % <const> == 0.
    # Also enforce q8_kv (non-row base 'Q8_KV') to require n_per_row % 32 == 0 (per captured assert).
    base = non_row_variant(qtype_upper)
    if base == "Q4_0":
        if n_per_row is None or (n_per_row % QK4_0) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK4_0} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q4_1":
        if n_per_row is None or (n_per_row % QK4_1) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK4_1} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q5_0":
        if n_per_row is None or (n_per_row % QK5_0) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK5_0} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q5_1":
        if n_per_row is None or (n_per_row % QK5_1) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK5_1} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q6_0":
        if n_per_row is None or (n_per_row % QK6_0) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK6_0} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q8_0":
        if n_per_row is None or (n_per_row % QK8_0) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK8_0} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q8_1":
        if n_per_row is None or (n_per_row % QK8_1) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % {QK8_1} == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)
    elif base == "Q8_KV":
        # Non-row (base) Q8_KV asserts require n_per_row % 32 == 0 (note: the R8 row-variant Q8_KV_R8 is handled above).
        if n_per_row is None or (n_per_row % 32) != 0:
            reason = f"Quant '{qtype_upper}' (base '{base}') requires n_per_row % 32 == 0, got n_per_row={n_per_row!r}"
            failed_to_transform(tensor_name, qtype_upper, reason)

    # If there's anything else to check in the future, add here.
# -------------------------
# End shape/blocking constraints
# -------------------------


def attempt_transform_line(parts: List[str],
                           kv_pairs: List[Tuple[str, str]],
                           kv_dict: Dict[str, str],
                           tensor_name: str,
                           fname: str,
                           first_requested_qtype_upper: str,
                           qtype_upper: str,
                           qtype_lower: str,
                           ignore_imatrix_rules: bool,
                           imatrix: bool) -> str:
    """
    Attempt to transform a single parsed data line using the specified qtype.
    Returns the transformed line string on success or raises TransformFailure on failure.
    """

    # Replace any occurrence of quant type in shard filename with qtype_lower.
    if re.search(r'-[^-]+-SPECIAL', fname, re.IGNORECASE):
        # preserve other parts of the filename, only replace occurrences of 'bf16' (case-insensitive)
        new_fname = re.sub(r'-[^-]+-SPECIAL', '-'+first_requested_qtype_upper+'-SPECIAL', fname, flags=re.IGNORECASE)
    else:
        new_fname = fname

    # Original sha from the parsed parts
    original_sha = parts[1]

    # Default behavior: set hash to 64 zeros for transformed tensors.
    # However, certain cases must keep the original shard hash:
    #  - f32 tensors are not quantized -> keep original sha
    #  - if original dtype is bf16 and requested qtype is bf16 -> keep original sha
    new_sha = '0' * 64

    # Parse elements
    if 'elements' not in kv_dict:
        # can't compute bytes without elements; treat as failure
        reason = "Missing 'elements' field; cannot compute bytes for quantized tensor."
        failed_to_transform(tensor_name, qtype_upper, reason)

    try:
        elements_val = int(kv_dict['elements'])
    except ValueError:
        reason = f"Invalid 'elements' value: {kv_dict.get('elements')!r}"
        failed_to_transform(tensor_name, qtype_upper, reason)

    # Determine dtype replacement:
    orig_dtype = kv_dict.get('dtype', '')

    # Preserve original shard hash for f32 lines (never quantized)
    # Also preserve hash when original is bf16 AND the user requested bf16 (no-op transform).
    if orig_dtype.lower() == 'f32' or (orig_dtype.lower() == 'bf16' and qtype_lower == 'bf16'):
        new_sha = original_sha

    # NEW: f32 must never be changed — keep the kv_pairs unchanged but still replace filename and sha as above.
    if orig_dtype.lower() == 'f32':
        # Preserve original dtype and bytes; only update filename and sha (sha preserved as original_sha).
        # Reconstruct kv string exactly as it was (preserving order)
        new_pairs = list(kv_pairs)
        new_rest_str = reconstruct_kv_string(new_pairs)
        new_line = ':'.join([new_fname, new_sha, tensor_name, new_rest_str])
        return new_line

    # Previously this script required orig dtype == 'bf16'. That requirement has been relaxed:
    # Accept any orig dtype (other than f32 which is preserved) and proceed to convert to the requested qtype.

    new_dtype = qtype_upper.lower()

    # ---------------------------
    # Checks added (imatrix & token embedding rules)
    # ---------------------------
    # Rule 1:
    if tensor_name == "token_embd.weight" and is_row_interleaved_qtype(qtype_lower):
        reason = (
            "The 'token_embd.weight' tensor cannot be transformed to a row-interleaved "
            f"quant ('{qtype_lower}'). Row-interleaved quantizations are incompatible "
            "with this tensor."
        )
        failed_to_transform(tensor_name, qtype_upper, reason)

    # Rule 2: adapted imatrix rules
    problematic_qtypes = {
        "IQ2_XXS", "IQ2_XXS_R4", "IQ2_XS", "IQ2_XS_R4",
        "IQ2_S", "IQ2_S_R4", "IQ1_S", "IQ1_S_R4", "IQ1_M_R4", "IQ1_M"
    }

    if (not ignore_imatrix_rules) and (not imatrix):
        if qtype_upper in problematic_qtypes and tensor_name in ("token_embd.weight", "output.weight"):
            reason = (
                "Missing importance matrix for tensor in a very low-bit quantization "
                f"('{qtype_upper}'). The result would be garbage without an importance matrix."
            )
            failed_to_transform(tensor_name, qtype_upper, reason)

        # if qtype_upper == "Q2_K" and tensor_name != "token_embd.weight":
        #     reason = (
        #         "Q2_K quantization (without an importance matrix) is unsafe for this tensor. "
        #         "Missing importance matrix for tensor in a very low-bit quantization."
        #     )
        #     failed_to_transform(tensor_name, qtype_upper, reason)

    # ---------------------------
    # End of checks
    # ---------------------------

    # --- NEW: parse shape and enforce GGML_ASSERT-derived constraints ---
    # Extract shape if available and parse into integers
    shape_val = kv_dict.get('shape')
    parsed_nrows = None
    parsed_n_per_row = None
    if shape_val:
        # Expecting strings like "(2560, 151936)" or "(2560,)" or similar
        nums = re.findall(r'-?\d+', shape_val)
        if len(nums) == 1:
            # NOTE: Interpret a single-dimension shape value as n_per_row (first numeric value),
            # and leave nrows as None. This reflects the ordering observed in .map examples
            # where shape=(per_row, nrows).
            try:
                parsed_n_per_row = int(nums[0])
            except Exception:
                parsed_n_per_row = None
            parsed_nrows = None
        elif len(nums) >= 2:
            # IMPORTANT: The .map file's shape ordering is interpreted here as (n_per_row, nrows).
            # For example: token_embd.weight:shape=(2560, 151936)
            # -> n_per_row = 2560, nrows = 151936
            try:
                parsed_n_per_row = int(nums[0])
                parsed_nrows = int(nums[1])
            except Exception:
                parsed_nrows = None
                parsed_n_per_row = None
        # otherwise leave None for missing/unknown shapes

    # Call the shape/blocking constraint checker which will call failed_to_transform on failure
    check_shape_constraints(qtype_upper=qtype_upper, qtype_lower=qtype_lower,
                            nrows=parsed_nrows, n_per_row=parsed_n_per_row,
                            tensor_name=tensor_name,
                            ignore_imatrix_rules=ignore_imatrix_rules, imatrix=imatrix)

    # Compute new bytes using tensor_size function
    try:
        new_bytes_val = tensor_size(elements_val, new_dtype.upper())
    except KeyError as e:
        reason = f"Requested qtype '{new_dtype.upper()}' not supported (needed for bytes computation)."
        failed_to_transform(tensor_name, qtype_upper, reason)

    # Reconstruct kv_pairs preserving original order, but replace dtype and bytes as required
    new_pairs = []
    for k, v in kv_pairs:
        if k == 'dtype':
            new_pairs.append((k, new_dtype))
        elif k == 'bytes':
            new_pairs.append((k, str(new_bytes_val)))
        else:
            # keep same
            new_pairs.append((k, v))

    # If there was no 'bytes' field originally, append it at the end
    if not any(k == 'bytes' for k, _ in new_pairs):
        new_pairs.append(('bytes', str(new_bytes_val)))
    # If there was no 'dtype' field originally and original dtype existed, append it
    if not any(k == 'dtype' for k, _ in new_pairs) and orig_dtype:
        new_pairs.append(('dtype', new_dtype))

    new_rest_str = reconstruct_kv_string(new_pairs)
    new_line = ':'.join([new_fname, new_sha, tensor_name, new_rest_str])
    return new_line


def build_fallback_candidates(initial_qtype_upper: str,
                              bpw_table: Dict[str, float],
                              whitelist: List[str],
                              forbidden_regexes: List[re.Pattern],
                              allowed_bn: bool) -> List[str]:
    """
    Build ordered list of fallback qtype candidates according to BPW_TABLE.
    Rules:
      - If initial was row-interleaved, try its non-row variant first (if different and available).
      - Then try qtypes with BPW >= initial_bpw in ascending order of BPW.
      - Never select any qtype containing '_BN' unless allowed_bn is True.
      - Apply whitelist (if non-empty) and forbidden regexes.
    Returns list of candidate qtypes (uppercase).
    """
    initial = initial_qtype_upper
    initial_bpw = bpw_table.get(initial)
    if initial_bpw is None:
        # If initial not in bpw table, start from smallest bpw (conservative)
        # but include initial itself at front
        initial_bpw = -1

    # Prepare whitelist set (uppercase) if provided
    whitelist_set = set([w.upper() for w in whitelist]) if whitelist else None

    # Create list of (bpw, q) sorted ascending bpw
    items = sorted(bpw_table.items(), key=lambda kv: (kv[1], kv[0]))

    # Start from equal or higher bpw
    candidates = [q for q, bpw in [(k, v) for k, v in items] if bpw >= initial_bpw]

    # Ensure there are no duplicates and filter presence in GGML_QUANT_SIZES
    filtered = []
    seen = set()

    # Step 1: try non-row variant if initial was row-interleaved
    non_row = non_row_variant(initial)
    if non_row != initial and non_row in bpw_table and non_row in GGML_QUANT_SIZES:
        # Check BN rule and whitelist/forbidden
        if (allowed_bn or '_BN' not in non_row) and (not whitelist_set or non_row in whitelist_set) and (not any(r.search(non_row) for r in forbidden_regexes)):
            filtered.append(non_row)
            seen.add(non_row)

    # Step 2: iterate through candidates (bpw >= initial_bpw) and add those allowed
    for q in candidates:
        if q == initial:
            continue
        if q in seen:
            continue
        if q not in GGML_QUANT_SIZES:
            continue
        if (not allowed_bn) and ('_BN' in q):
            continue
        if whitelist_set and (q not in whitelist_set):
            continue
        if any(r.search(q) for r in forbidden_regexes):
            continue
        filtered.append(q)
        seen.add(q)

    return filtered


def process_map_lines(lines,
                      initial_qtype_upper: str,
                      initial_qtype_lower: str,
                      ignore_imatrix_rules: bool,
                      imatrix: bool,
                      allow_fallback: bool,
                      fallback_whitelist: List[str],
                      fallback_forbidden_patterns: List[str]):
    """
    Process list of lines from input .map and return new list of output lines.
    Preserves header lines and separators '---'. Transforms data lines.

    Supports per-tensor fallback attempts when a requested qtype is not safe.
    """
    output_lines = []

    # compile forbidden regexes (case-insensitive) so that user-provided patterns are matched regardless of case
    forbidden_regexes = [re.compile(p, re.IGNORECASE) for p in fallback_forbidden_patterns] if fallback_forbidden_patterns else []

    initial_was_bn = '_BN' in initial_qtype_upper
    whitelist = [w.upper() for w in fallback_whitelist] if fallback_whitelist else []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == "" or stripped == '---' or stripped.lower().startswith('gguf_shard'):
            # keep separators and header lines unchanged
            output_lines.append(line)
            continue

        # Attempt to parse a data line of format:
        # <shard filename>:<sha256>:<tensor_name>:shape=...:dtype=...:elements=...:bytes=...
        parts = line.split(':')
        if len(parts) < 6:
            # not in expected format; keep unchanged but warn
            output_lines.append(line)
            continue

        fname = parts[0]
        sha = parts[1]
        tensor_name = parts[2]
        rest = parts[3:]  # list of key=value pieces in order
        kv_pairs = parse_kv_pairs(rest)
        kv_dict = {k: v for k, v in kv_pairs}

        # ---------------------------
        # NEW: imatrix handling per user's request
        # - If --with-imatrix (global `imatrix` param) is enabled, then we consider the presence
        #   of the 'imatrix' attribute for this line. If it's missing, issue a warning and treat
        #   this single tensor as if --with-imatrix were NOT set (local_imatrix=False).
        # - If --with-imatrix is NOT set globally, any imatrix= attributes in the line are ignored
        #   for the purpose of checks (we process with local_imatrix=False), but we DO keep the
        #   imatrix attribute in the output .map (we do not strip it).
        # This decision must happen before any conversion checks for the target qtype.
        # ---------------------------
        local_imatrix = False
        if imatrix:
            # User requested global --with-imatrix: require per-line imatrix attribute to actually enable it
            # Consider imatrix present only if the key exists and has a non-empty value.
            im_val = kv_dict.get('imatrix')
            if im_val:
                local_imatrix = True
            else:
                dtype = kv_dict.get('dtype')
                if dtype != "f32":
                    # Warn the user that this specific tensor lacks the imatrix attribute and therefore
                    # it will be processed as if --with-imatrix was NOT set for this tensor only.
                    print(
                        f"⚠️ Warning: --with-imatrix was specified but tensor '{tensor_name}' "
                        "is missing an 'imatrix=' attribute in the .map file; this tensor will be "
                        "processed as if --with-imatrix was NOT set (only this tensor's imatrix checks are disabled).",
                        file=sys.stderr
                    )
                    local_imatrix = False
                else:
                    local_imatrix = False
        else:
            # Global --with-imatrix not set: ignore any imatrix= attribute for checks (local_imatrix stays False).
            # We intentionally do NOT remove the imatrix attribute from kv_pairs; output must retain it if present.
            local_imatrix = False

        # Try transforming this line using the initially requested qtype
        try:
            new_line = attempt_transform_line(
                parts=parts,
                kv_pairs=kv_pairs,
                kv_dict=kv_dict,
                tensor_name=tensor_name,
                fname=fname,
                first_requested_qtype_upper=initial_qtype_upper,
                qtype_upper=initial_qtype_upper,
                qtype_lower=initial_qtype_lower,
                ignore_imatrix_rules=ignore_imatrix_rules,
                imatrix=local_imatrix  # pass per-tensor imatrix flag
            )
            output_lines.append(new_line)
            continue
        except TransformFailure:
            # If fallback is not allowed, abort the whole script now.
            if not allow_fallback:
                # The failure message has already been printed by failed_to_transform.
                print("Aborting due to transform failure and fallback disabled (--no-fallback).", file=sys.stderr)
                sys.exit(6)

            # Otherwise, issue a warning (failed_to_transform already printed the detailed reason)
            print(f"⚠️ Warning: attempting fallback selection for tensor '{tensor_name}' (requested {initial_qtype_upper}).", file=sys.stderr)

            # Build fallback candidate list
            candidates = build_fallback_candidates(
                initial_qtype_upper=initial_qtype_upper,
                bpw_table=BPW_TABLE,
                whitelist=whitelist,
                forbidden_regexes=forbidden_regexes,
                allowed_bn=initial_was_bn
            )

            # Try each candidate in order until one succeeds
            succeeded = False
            tried = set()
            for cand in candidates:
                if cand in tried:
                    continue
                tried.add(cand)
                cand_lower = cand.lower()
                try:
                    new_line = attempt_transform_line(
                        parts=parts,
                        kv_pairs=kv_pairs,
                        kv_dict=kv_dict,
                        tensor_name=tensor_name,
                        fname=fname,
                        first_requested_qtype_upper=initial_qtype_upper,
                        qtype_upper=cand,
                        qtype_lower=cand_lower,
                        ignore_imatrix_rules=ignore_imatrix_rules,
                        imatrix=local_imatrix  # keep per-tensor imatrix semantics for fallbacks too
                    )
                    print(f"Info: tensor '{tensor_name}' successfully transformed using fallback qtype '{cand}'.", file=sys.stderr)
                    output_lines.append(new_line)
                    succeeded = True
                    break
                except TransformFailure:
                    # try next candidate
                    continue

            if not succeeded:
                # No fallback worked; fail the script
                print(f"Error: no suitable fallback qtype could be found for tensor '{tensor_name}'. Aborting.", file=sys.stderr)
                sys.exit(6)

    return output_lines


def main():
    p = argparse.ArgumentParser(
        description=(
            "Convert a .map file to a target quantization type (--qtype).\n"
            "Preserves f32 tensors (their dtype/bytes/hash remain unchanged), sets shard hashes to zeros for\n"
            "quantized tensors, updates dtype to the requested quant (lowercased), and recomputes bytes using\n"
            "a built-in quant size table.\n\n"
            "Importance-matrix (imatrix) rules:\n"
            "  - Use --with-imatrix to tell the script that an importance matrix is available. When set,\n"
            "    each tensor must include a non-empty 'imatrix=' attribute in the .map file for that tensor\n"
            "    to be treated as having an importance matrix; missing per-line imatrix attributes will cause\n"
            "    the script to treat that tensor as if no importance matrix is present (a warning is printed).\n"
            "  - By default (no --with-imatrix), any imatrix= attributes in the file are ignored for safety checks.\n"
            "  - If you really know what you're doing, --ignore-imatrix-rules disables these imatrix-related\n"
            "    safety checks (dangerous and may produce unusable quantizations).\n\n"
            "Other features: supports per-tensor fallback quant selection, --no-map to print to stdout instead of\n"
            "writing a file, and --output to specify an alternate output directory or filename. Case-insensitive\n"
            "qtype names are accepted."
        )
    )
    p.add_argument("input_map", help="Path to the input .map file to convert")
    p.add_argument("--qtype", required=True, help="Target quantization type (e.g. iq1_s_r4). Case-insensitive.")
    p.add_argument("--no-map", action="store_true", help="Do not write a .map file. Print the resulting map contents to stdout instead.")

    # New: allow user to specify output path (either a directory or a full filename).
    # If not provided, behavior remains unchanged: output created in same directory as input_map
    # with the name tensors.<qtype_lower>.map
    p.add_argument("--output", "--out", dest="output_map", type=str, default="",
                   help=("Optional output path. Can be either a directory or a full filename (including .map). "
                         "If a directory is provided, the output filename will be tensors.<qtype_lower>.map inside that directory. "
                         "If omitted the output will be written next to the input map."))

    # Flags controlling importance-matrix rules and whether an importance matrix is present
    p.add_argument("--ignore-imatrix-rules", action="store_true",
                   help="Ignore importance-matrix related safety checks (dangerous).")
    p.add_argument("--with-imatrix", action="store_true",
                   help="Indicate that an importance matrix is available (satisfies imatrix checks).")

    # Fallback related flags
    p.add_argument("--no-fallback", action="store_true",
                   help="Do not attempt fallback quants on failure; abort on first failed_to_transform.")
    p.add_argument("--fallback-quants", type=str, default="",
                   help="Comma-separated list of qtypes to whitelist for fallback (case-insensitive). If empty, all are considered. "
                        "Example: --fallback-quants iq2_xs,IQ3_S,q8_k")
    p.add_argument("--fallback-quants-forbidden", type=str, default="",
                   help="Comma-separated list of regex patterns (case-insensitive) matching qtypes that must NOT be used as fallbacks. "
                        "Example: --fallback-quants-forbidden '^(iq1_|Q8_K$)', '.*_bn$'")

    args = p.parse_args()

    input_path = Path(args.input_map)
    if not input_path.exists() or not input_path.is_file():
        print(f"Error: input file '{input_path}' does not exist or is not a file.", file=sys.stderr)
        sys.exit(2)

    qtype_lower = args.qtype.lower()
    qtype_upper = args.qtype.upper()

    # Validate qtype exists in GGML_QUANT_SIZES
    if qtype_upper not in GGML_QUANT_SIZES:
        available = ", ".join(sorted(GGML_QUANT_SIZES.keys()))
        print(f"Error: requested qtype '{args.qtype}' (upper '{qtype_upper}') not found in supported types.", file=sys.stderr)
        print("Supported qtypes include (example subset):", file=sys.stderr)
        print(available, file=sys.stderr)
        sys.exit(2)

    # Parse fallback whitelist and forbidden patterns
    fallback_whitelist = []
    if args.fallback_quants:
        # Accept case-insensitive qtype names from the user; normalize to uppercase for internal comparisons.
        fallback_whitelist = [q.strip().upper() for q in args.fallback_quants.split(',') if q.strip()]
    fallback_forbidden_patterns = []
    if args.fallback_quants_forbidden:
        # split by commas, allow regex patterns; patterns will be compiled case-insensitively later.
        fallback_forbidden_patterns = [p.strip() for p in args.fallback_quants_forbidden.split(',') if p.strip()]

    allow_fallback = not args.no_fallback

    # Read file
    raw_lines = input_path.read_text(encoding='utf-8').splitlines(keepends=True)

    # Process lines
    new_lines = process_map_lines(
        raw_lines,
        initial_qtype_upper=qtype_upper,
        initial_qtype_lower=qtype_lower,
        ignore_imatrix_rules=args.ignore_imatrix_rules,
        imatrix=args.with_imatrix,
        allow_fallback=allow_fallback,
        fallback_whitelist=fallback_whitelist,
        fallback_forbidden_patterns=fallback_forbidden_patterns
    )

    if args.no_map:
        # print to stdout
        for l in new_lines:
            print(l)
        return

    # Determine output map path.
    # Default behavior: write to same directory as input with name tensors.<qtype_lower>.map
    # If user provided --output, it may be a directory or a full filename. Support both.
    output_path = None
    if args.output_map:
        provided = Path(args.output_map)
        if provided.exists():
            if provided.is_dir():
                # Provided path is an existing directory -> write tensors.<qtype_lower>.map inside it
                output_path = provided / f"tensors.{qtype_lower}.map"
            else:
                # Provided path is an existing file -> write to that file (overwrite)
                output_path = provided
        else:
            # Provided path does not exist. Decide based on suffix: if it looks like a filename (ends with .map) treat as file,
            # otherwise treat as directory and create it.
            if provided.suffix.lower() == '.map':
                # Treat as file path. Ensure parent exists (create if necessary).
                parent = provided.parent
                if not parent.exists():
                    try:
                        parent.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        print(f"Error: could not create parent directory '{parent}' for output file: {e}", file=sys.stderr)
                        sys.exit(5)
                output_path = provided
            else:
                # Treat as directory path; create it (and parents) and then put tensors.<qtype_lower>.map inside it.
                try:
                    provided.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"Error: could not create output directory '{provided}': {e}", file=sys.stderr)
                    sys.exit(5)
                output_path = provided / f"tensors.{qtype_lower}.map"
    else:
        # No user-provided output: use same directory as input (original behavior)
        output_name = f"tensors.{qtype_lower}.map"
        output_path = input_path.with_name(output_name)

    # Write output (overwrite if exists)
    try:
        with output_path.open('w', encoding='utf-8', newline='\n') as f:
            for l in new_lines:
                f.write(l if l.endswith('\n') else l + '\n')
    except Exception as e:
        print(f"Error: failed to write output file '{output_path}': {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Wrote converted .map to: {output_path}")


if __name__ == "__main__":
    main()
