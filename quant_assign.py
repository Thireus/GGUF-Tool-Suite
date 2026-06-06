#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_assign.py the recipe maker tool of choice! Use it   **#
#** to produce recipes that can be cooked and used by others. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jun-06-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.          Zₑᵣₒ₋ₛₕₒₜ, 𝒻ᵤₗₗ ₙₒₙₛₑₙₛₑ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install pandas numpy argparse pgpy

# Tip: You can pipe the output of this script (as long as no warning or debug logs are present) to quants_regex_merger like so: | tee /dev/tty | ./quants_regex_merger.sh
# python quant_assign.py ppl_results_guessed.csv --gpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq6_k iq5_k_r4 --cpu-tensors-max-size 230 --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528
# python quant_assign.py 'ppl_results.csv' --gpu-tensors '.*' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq5_k_r4 iq6_k --cpu-tensors-max-size 230 --gpu-tensors-max-size 90% --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528

import ast
from datetime import datetime
import math
import time
import os
import shlex
import argparse
import pandas as pd
import numpy as np
import re
import sys
import hashlib
import functools
import subprocess
import tempfile
from collections import Counter
import textwrap
from typing import Any, Callable, cast, Dict, List, Iterable, Tuple, Optional
import heapq

_cached_functions = []

def tracked_lru_cache(*args, **kwargs):
    def decorator(func):
        cached_func = functools.lru_cache(*args, **kwargs)(func)
        _cached_functions.append(cached_func)
        return cached_func
    return decorator

def clear_all_caches():
    for func in _cached_functions:
        func.cache_clear()

# Global default quants list
DEFAULT_QUANTS = ['q8_0', 'q4_0']

# ---- Adaptive greedy 2nd-pass selector defaults (hardcoded; edit here to change) ----
# These configure the greedy-winner second pass (Step 8.5). To force a specific combo
# at runtime use the --auto-force-combo CLI flag (no environment variables are used).
ADAPT_ENABLED        = True                       # run the adaptive combo selector
ADAPT_LATTICE        = ['class', 'pos', 'tier2']  # combos the selector enumerates
ADAPT_ALLOW_TIER2    = True                       # allow the tier2 win-promotion
ADAPT_ALLOW_CLASSPOS = True                       # allow the gated class+pos conservative upgrade
# Ultra-tight "starvation" degenerate regime: when EVERY combo floor-demotes into a
# tight body, the V3 starvation veto fires on all of them and the lattice / meta-score
# become uninformative (predicted_damage is INVERTED there — the lowest-damage recipe
# has the WORST PPL). Falling back to plain 'none' (under-protective greedy) is a poor
# default at sub-~1.75bpw. Instead adopt a sensitivity-PROTECTED greedy: per-tensor
# degradation scaling (loss/mean)^k, the measured win for ultra-tight MoE budgets
# (Qwen3.6-35B-A3B 1.7030bpw: 'none' 10.49 ppl -> pts 10.06 ppl). Set 0 to disable
# (faithful 'none' fallback). Only reachable when ALL combos are vetoed, which no
# previously-validated recipe hits -> those stay byte-identical.
ADAPT_STARVE_PTS     = 0.4                         # per-tensor-degradation-scaling k for the all-vetoed branch

# Default reducing factors when data not available
DEFAULT_REDUCE = {
    32: 1.000,
    16: 0.999,
     8: 0.9998,
     6: 0.9967,
     4: 0.9763,
     3: 0.918,
     2: 0.878,
     1: 0.395,
}

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

ADDITIONAL_SCALE_FACTOR_TABLE = {
    'IQ1_BN': 2,
    'IQ1_KT': 4,
    'IQ2_BN': 4,
    'IQ2_BN_R4': 4,
    'IQ2_KL': 2,
    'IQ2_KS': 2,
    'IQ2_KT': 4,
    'IQ3_KS': 2,
    'IQ3_KT': 4,
    'IQ4_KS': 4,
    'IQ4_KSS': 4,
    'IQ4_KS_R4': 4,
    'IQ4_KT': 4,
    'IQ5_KS': 4,
    'IQ5_KS_R4': 4,
    'Q8_KV': 8,
    'IQ1_S_R4': 2,
    'IQ1_M_R4': 2,
    'Q8_KV_R8': 4
}

# Cache bpw observed per qtype and per tensor name whenever a tensors map file is processed.
# Structure:
#   QTYPE_BPW_CACHE[qtype_upper][tensor_name] = actual bpw (after scale correction if needed)
#   QTYPE_BPW_BASE_CACHE[qtype_upper][tensor_name] = base bpw (without scale correction)
QTYPE_BPW_CACHE = {}
QTYPE_BPW_BASE_CACHE = {}

def _canonical_qtype_key(qtype):
    return transform_q_suffix(str(qtype)).upper()

def _product(values):
    out = 1
    for v in values:
        out *= int(v)
    return out

def _parse_shape_field(shape_text):
    """
    Parse shape=(2560, 151936) or shape=[2560, 151936] into [2560, 151936].
    """
    if not shape_text:
        return []
    s = str(shape_text).strip()
    nums = re.findall(r'-?\d+', s)
    return [int(n) for n in nums]

def _static_bpw_for_qtype(qtype_upper):
    """
    Static fallback bpw when no tensor-specific cache is available yet.
    """
    q = _canonical_qtype_key(qtype_upper)
    if q in BPW_TABLE:
        return float(BPW_TABLE[q])
    if q in GGML_QUANT_SIZES:
        block_size, type_size = GGML_QUANT_SIZES[q]
        return (type_size * 8.0) / block_size
    return float('nan')

def _register_tensor_bpw(qtype_upper, tensor_name, actual_bytes, base_bytes, elements):
    """
    Store tensor-specific bpw values for both actual and base caches.
    """
    q = _canonical_qtype_key(qtype_upper)
    QTYPE_BPW_CACHE.setdefault(q, {})[tensor_name] = (actual_bytes * 8.0 / elements) if elements else 0.0
    QTYPE_BPW_BASE_CACHE.setdefault(q, {})[tensor_name] = (base_bytes * 8.0 / elements) if elements else 0.0

def _tensor_entry_bpw(qtype, tensor_name, use_base=False):
    """
    Convenience accessor for a tensor-specific bpw entry.
    """
    q = _canonical_qtype_key(qtype)
    cache = QTYPE_BPW_BASE_CACHE if use_base else QTYPE_BPW_CACHE
    return cache.get(q, {}).get(tensor_name)

def _quant_sort_key(q):
    q_key = _canonical_qtype_key(q)
    bpw = get_bpw(q)
    scale_factor = ADDITIONAL_SCALE_FACTOR_TABLE.get(q_key, 0)
    return (bpw, scale_factor, q_key)

script_dir = os.path.dirname(os.path.realpath(__file__))

SKIP_GPG = False
ALL_GPG_SIGS_VALID = True
KEYRING_PATH = os.path.join(script_dir, "trusted-keys.asc")
if not SKIP_GPG:
    try:
        import pgpy
        import warnings
        warnings.filterwarnings("ignore", module="pgpy")
    except ImportError:
        print("[Warning] pgpy not installed; gpg signature validation skipped.", file=sys.stderr)
        SKIP_GPG = True

# Remote connection settings for tensor_downloader.sh:
# Please edit tensor_downloader.sh!
# Resolve script directory for locating tensor_downloader.sh
tensor_downloader = os.path.join(script_dir, 'tensor_downloader.sh')

if not os.path.isfile(tensor_downloader) or not os.access(tensor_downloader, os.X_OK):
    print(f"Error: tensor_downloader.sh not found or not executable at {tensor_downloader}", file=sys.stderr)
    sys.exit(1)

# Cache for fetched map files and parsed maps per quant
_fetched_maps = set()
_quant_maps = {}

# Track qtypes whose .map files were produced via convert_map_qtype.py
COMPUTED_QTYPES = set()

# Verbosity flags
DEBUG = False
INFO = False

# Flags for compute-map feature (populated from CLI)
COMPUTE_MISSING_MAP = False
COMPUTE_ALL_MAP = False
CONVERT_IGNORE_IMATRIX_RULES = False
CONVERT_WITH_IMATRIX = False
CONVERT_FALLBACK_QUANTS = ""
CONVERT_FALLBACK_QUANTS_FORBIDDEN = ""

# Constants
GIB = 1024**3 # for GiB-to-bytes conversion
STRETCH_MIN = 1.0
STRETCH_MAX = 10.0
STRETCH_STEP = 0.01

# ─── Create a unique temp‐dir at script launch ────────────────────────────────
# This will give you something like /tmp/gguf.thireus.com.ab12cd
TMP_DIR = tempfile.mkdtemp(prefix="gguf.thireus.com.", dir=tempfile.gettempdir())
if DEBUG: print(f"[Debug] Using temp directory: {TMP_DIR}", file=sys.stderr)

# Optionally, register cleanup at exit
import atexit
import shutil

def _cleanup_tempdir(path=TMP_DIR):
    try:
        shutil.rmtree(path)
        if DEBUG: print(f"[Debug] Cleaned up temp directory: {path}", file=sys.stderr)
    except Exception:
        pass

atexit.register(_cleanup_tempdir)
# ──────────────────────────────────────────────────────────────────────────────

def extract_quant_num(qtype):
    """
    Extract the first integer in a qtype string.
    """
    m = re.search(r"(\d+)", qtype)
    return int(m.group(1)) if m else float('inf')


# Cache for factors loaded via normalised_ppl.py
_factor_cache = {}


def compute_iqr_bounds(values, k):
    """
    Compute robust IQR bounds for outlier detection.
    """
    arr = np.array(list(values.values()))
    Q1, Q3 = np.percentile(arr, [25, 75])
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return lower, upper


def load_quant_degradation_values(path: str):
    """Load degradation factors from a CSV file at `path`.

    The CSV is expected to have columns: "QTYPE" and "group0" (or similar).
    Values may be percentages (e.g. "+2.12%") or absolute floats (e.g. "0.0212").

    Improvements:
    - Drop empty / missing values and values that equal 404 / '404%' (treated as missing).
      Log dropped qtypes with [Info] Dropping ...
    """
    df = pd.read_csv(path)
    # Expect columns: "QTYPE" and "group0"
    degradation_factors = {}
    dropped = []
    for _, row in df.iterrows():
        q = str(row.get("QTYPE", "")).strip()
        val = row.get("group0")
        # handle missing / NaN
        if pd.isna(val):
            dropped.append(q)
            continue
        s = str(val).strip()
        # treat 404 and variants as missing
        if s.lower() in ('404', '404.0', '404%', '404.0%'):
            dropped.append(q)
            continue
        is_percent = s.endswith('%')
        if is_percent:
            s = s[:-1].strip()
        # strip optional leading sign
        if s.startswith('+'):
            s = s[1:].strip()
        try:
            f = float(s)
        except ValueError:
            dropped.append(q)
            continue
        # if it was a percent string, convert to fraction; otherwise treat as absolute
        if is_percent:
            f = f / 100.0
        degradation_factors[q] = f

    if INFO and dropped:
        # Filter out empty q names for nicer message
        nice = [dq for dq in dropped if dq]
        print(f"[Info] Dropping quant degradation entries for missing/404 values: {nice}", file=sys.stderr)

    return degradation_factors



def _call_normalised_ppl(keys):
    """
    Call the normalised_ppl.py script for a list of keys, using edges 1 and 32.
    Returns a dict mapping each numeric key to its fetched factor (float).
    Raises RuntimeError on parse failure for a key, or subprocess errors.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'normalised_ppl.py')
    keys_list = list(keys)
    if INFO:
        print(f"[Info] Calling normalised_ppl.py for keys: {keys_list}", file=sys.stderr)
    # Compose command: include 1 and 32 as edge values
    bpw_args = ['1'] + [str(k) for k in keys_list] + ['32']
    cmd = ['python', script_path, '--bpw-list'] + bpw_args
    if DEBUG:
        print(f"[Debug] Running command: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        if DEBUG:
            print(f"[Debug] normalised_ppl.py output:\n{output}", file=sys.stderr)
    except Exception as e:
        if INFO:
            print(f"[Warning] normalised_ppl.py call failed: {e}", file=sys.stderr)
        raise

    # Parse output lines like 'KEY: VALUE'
    results = {}
    for line in output.splitlines():
        parts = line.strip().split(':')
        if len(parts) != 2:
            continue
        try:
            bpw = float(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        # Only collect requested keys
        if bpw in keys_list:
            results[bpw] = val
    # Ensure all requested keys are found
    missing = set(keys_list) - set(results.keys())
    if missing:
        raise RuntimeError(f"Keys {missing} not found in normalised_ppl output")
    return results


@tracked_lru_cache(maxsize=None)
def get_bpw(qtype, tensor_name=None, use_base=False):
    """
    Return the bpw for a given qtype.

    - If tensor_name is provided, return the tensor-specific bpw.
    - If tensor_name is omitted, return the average bpw across all cached tensors for that qtype.
    - If use_base is True, use the base cache (without scale-correction); otherwise use the actual cache.
    - If no cache is available yet, fall back to the static table.
    """
    q = _canonical_qtype_key(qtype)
    cache = QTYPE_BPW_BASE_CACHE if use_base else QTYPE_BPW_CACHE
    entries = cache.get(q, {})

    if tensor_name is not None:
        if tensor_name in entries:
            return entries[tensor_name]
        if entries:
            return float(sum(entries.values())) / len(entries)
        return _static_bpw_for_qtype(q)

    if entries:
        return float(sum(entries.values())) / len(entries)

    return _static_bpw_for_qtype(q)

@tracked_lru_cache(maxsize=None)
def get_default_factor(qtype):
    """
    Return reducing factor based on bit-width.
    Attempts to fetch a better factor using normalised_ppl.py, falling back to DEFAULT_REDUCE.
    Results are cached per bpw.
    """
    bpw = get_bpw(qtype)
    try:
        if INFO:
            print(f"[Info] bpw for qtype {qtype}: {bpw}", file=sys.stderr)
        key = bpw
    except Exception:
        if DEBUG:
            print(f"[Debug] Could not parse bpw from qtype '{qtype}', returning 1.0", file=sys.stderr)
        return 1.0

    # fallback default
    default_value = DEFAULT_REDUCE.get(int(key), 1.0)

    # return cached if available
    if bpw in _factor_cache:
        if DEBUG:
            print(f"[Debug] Returning cached factor for bpw {bpw}: {_factor_cache[bpw]}", file=sys.stderr)
        return _factor_cache[bpw]

    # try to fetch from script for this single key
    try:
        fetched = _call_normalised_ppl([bpw])
        factor = fetched.get(bpw, default_value)
    except Exception:
        factor = default_value
    else:
        if DEBUG:
            print(f"[Debug] Caching factor for bpw {bpw}: {factor}", file=sys.stderr)
        _factor_cache[bpw] = factor

    return factor


def parse_value(val):
    """
    Parse a PPL string or number, stripping '%' if present.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.endswith('%'):
        s = s[:-1]
    try:
        return float(s)
    except:
        return np.nan


def classify_tensors(columns, cpu_patterns, gpu_patterns):
    """
    Classify tensor names into CPU/GPU-friendly based on regex lists.
    """
    classes = {'cpu': [], 'gpu': []}
    for name in columns:
        assigned = False
        for pat in cpu_patterns:
            _pat = pat.split('=')[0]
            if re.fullmatch(_pat, name):
                classes['cpu'].append(name)
                assigned = True
                break
        if assigned:
            continue
        for pat in gpu_patterns:
            _pat = pat.split('=')[0]
            if re.fullmatch(_pat, name):
                classes['gpu'].append(name)
                assigned = True
                break
        if not assigned:
            classes['gpu'].append(name)
    return classes


def group_tensors(names):
    """
    Group tensor names by base name (strip leading layer indices).
    """
    groups = {}
    for name in names:
        m = re.match(r"blk\.\d+\.(.*)", name)
        base = m.group(1) if m else name
        groups.setdefault(base, []).append(name)
    return groups


def transform_q_suffix(s: str) -> str:
    """
    If s starts with 'q' (any case) and ends with 'kv' or 'k' (any case),
    return the same string but with the trailing 'k' or 'kv' uppercased.
    Otherwise return the original string.
    """
    # Try 'kv' first
    m = re.match(r'^(q.*?)(kv)$', s, re.IGNORECASE)
    if m:
        return m.group(1) + m.group(2).upper()

    # Then try single 'k'
    m = re.match(r'^(q.*?)(k)$', s, re.IGNORECASE)
    if m:
        return m.group(1) + m.group(2).upper()

    return s


def select_qtype(df, qtype_arg):
    """
    Select the row for given QTYPE or lowest quant,
    preferring QTYPEs that do NOT end with '_bn'.
    """
    if qtype_arg:
        if qtype_arg not in df['QTYPE'].values:
            print(f"Error: qtype '{qtype_arg}' not found in CSV.", file=sys.stderr)
            sys.exit(1)
        return df[df['QTYPE'] == qtype_arg].iloc[0]

    df['__quant_num__'] = df['QTYPE'].map(extract_quant_num)

    # Prefer QTYPEs that do NOT end with '_bn'
    df_non_bn = df[~df['QTYPE'].str.endswith('_bn')]

    if not df_non_bn.empty:
        sel = df_non_bn.nsmallest(1, '__quant_num__').iloc[0]
    else:
        # Fallback: all QTYPEs end with '_bn'
        sel = df.nsmallest(1, '__quant_num__').iloc[0]

    df.drop(columns='__quant_num__', inplace=True)
    return sel

# global state for fetch_map_for_qtype()
MAP_FILE_INFO     = {}   # will hold {"tensors.<qtype>.map": [qtype, sha256, last_line], …}
SIG_FILE_HASHES   = {}   # will hold {"tensors.<qtype>.map.sig": sha256, …}
# “Looks like q…k” ignoring case
_INSPECT_K_RE = re.compile(r'^q.*k$', re.IGNORECASE)
# Canonical form: lower-q, anything, upper-K
_CANONICAL_K_RE = re.compile(r'^q.*K$')
# Special case: q…KV (any case)
_INSPECT_KV_RE = re.compile(r'^q.*kv$', re.IGNORECASE)
# Canonical form: q…KV with exact case
_CANONICAL_KV_RE = re.compile(r'^q.*KV$')

def compute_map_for_qtype(qtype: str) -> bool:
    """
    Attempt to compute tensors.{qtype}.map from tensors.bf16.map using convert_map_qtype.py.
    Returns True on success, False otherwise.
    The produced map is not GPG-checked and will be recorded as computed so the recipe
    will prefix its qtypes with '!' and treat the file as trusted (no signature).
    """
    global COMPUTED_QTYPES, MAP_FILE_INFO, _fetched_maps

    if qtype == 'bf16':
        # nothing to do
        return False

    convert_script = os.path.join(script_dir, 'convert_map_qtype.py')
    if not os.path.isfile(convert_script):
        if INFO:
            print(f"[Info] convert_map_qtype.py not found in script directory ({convert_script}); cannot compute map for {qtype}.", file=sys.stderr)
        return False

    bf16_map_path = os.path.join(TMP_DIR, 'tensors.bf16.map')
    # ensure bf16 map exists (attempt to fetch it if missing)
    if not os.path.isfile(bf16_map_path):
        if INFO:
            print(f"[Info] bf16 map missing in tmpdir; attempting to fetch bf16 via tensor_downloader.", file=sys.stderr)
        try:
            # attempt fetch via existing mechanism (this will call fetch_map_for_qtype('bf16'))
            if not fetch_map_for_qtype('bf16'):
                if INFO:
                    print(f"[Info] Failed to fetch bf16 map; cannot compute map for {qtype}.", file=sys.stderr)
                return False
        except Exception as e:
            if INFO:
                print(f"[Info] Exception while fetching bf16 map: {e}", file=sys.stderr)
            return False

    # Build convert command
    cmd = ['python', convert_script, bf16_map_path, '--qtype', qtype]
    # append user-provided convert flags
    if CONVERT_IGNORE_IMATRIX_RULES:
        cmd.append('--ignore-imatrix-rules')
    if CONVERT_WITH_IMATRIX:
        cmd.append('--with-imatrix')
    if NO_FALLBACK:
        cmd.append('--no-fallback')
    if CONVERT_FALLBACK_QUANTS:
        cmd += ['--fallback-quants']
        for fallback_quant in CONVERT_FALLBACK_QUANTS:
            cmd += [fallback_quant]
    if CONVERT_FALLBACK_QUANTS_FORBIDDEN:
        cmd += ['--fallback-quants-forbidden']
        for fallback_quant_forbidden in CONVERT_FALLBACK_QUANTS_FORBIDDEN:
            cmd += [fallback_quant_forbidden]
    if INFO:
        print(f"[Info] Attempting to compute map for qtype {qtype} using convert_map_qtype.py...", file=sys.stderr)
        if DEBUG:
            print(f"[Debug] Running: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)

    try:
        # Run convert script; it should write tensors.<qtype>.map into same dir as bf16_map_path.
        # convert_map_qtype.py writes its progress to stdout, but quant_assign.py
        # uses stdout for the recipe (commonly piped into quants_regex_merger.py),
        # so redirect convert_map_qtype.py's stdout to our stderr instead.
        if DEBUG or INFO:
            subprocess.run(cmd, check=True, stdout=sys.stderr.fileno())
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        if INFO:
            print(f"[Info] convert_map_qtype.py failed for qtype {qtype}: {e}", file=sys.stderr)
        return False

    # verify output file exists
    local_map = os.path.join(TMP_DIR, f"tensors.{qtype}.map")
    if not os.path.isfile(local_map):
        if INFO:
            print(f"[Info] convert_map_qtype.py did not create expected file {local_map}", file=sys.stderr)
        return False

    # Compute and store sha256 and last_line for the newly created map file
    try:
        with open(local_map, 'rb') as f:
            sha256sum = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        sha256sum = "ERROR"

    last_line = ""
    try:
        with open(local_map, 'r') as f_text:
            lines = f_text.readlines()
            last_line = lines[-1].split(':', 1)[0] if lines else ''
            last_line = last_line.split('.gguf', 1)[0] if last_line else ''
    except Exception:
        last_line = ""

    map_key = f"tensors.{qtype}.map"
    # annotate model name / last_line to indicate it was computed
    model_name = f"{last_line} (computed)" if last_line else "(computed)"
    MAP_FILE_INFO[map_key] = [qtype, sha256sum, model_name]

    # Mark as fetched and computed to avoid gpg checks later
    _fetched_maps.add(qtype)
    COMPUTED_QTYPES.add(qtype)

    if INFO:
        print(f"[Info] Successfully computed map for qtype {qtype} -> {local_map}", file=sys.stderr)
        print(f"[Info] This computed map will NOT be gpg-checked and its qtype will be annotated in the recipe with a leading '!'.", file=sys.stderr)

    return True


def fetch_map_for_qtype(qtype: str):
    """
    Fetch and cache tensors.{qtype}.map via tensor_downloader.sh.
    """
    global ALL_GPG_SIGS_VALID, MAP_FILE_INFO, SIG_FILE_HASHES, COMPUTED_QTYPES
    if qtype in _fetched_maps:
        return True
    # If it matches q…k (any case) but not exactly q…K, warn
    if _INSPECT_K_RE.match(qtype) and not _CANONICAL_K_RE.match(qtype):
        print(
            f"[Warning] qtype={qtype!r} does not match the canonical pattern r'^q.*K$'. "
            "Q-types are case-sensitive and there are specific ones that start with lowercase 'q' "
            "and end with uppercase 'K' (e.g. 'q3_K').",
            file=sys.stderr,
        )
    # If it matches q…kv (any case) but not exactly q…KV, warn
    if _INSPECT_KV_RE.match(qtype) and not _CANONICAL_KV_RE.match(qtype):
        print(
            f"[Warning] qtype={qtype!r} does not match the canonical pattern r'^q.*KV$'. "
            "Q-types ending with 'KV' must use uppercase 'KV' (e.g. 'q8_KV').",
            file=sys.stderr,
        )
    # Warn if it's fully capitalized
    if qtype.isupper():
        print(
            f"[Warning] qtype={qtype!r} is fully capitalized. "
            "Q-types are case-sensitive and there are no known quant types that are entirely uppercase.",
            file=sys.stderr,
        )
    local_map = os.path.join(TMP_DIR, f"tensors.{qtype}.map")
    cmd = ["bash", tensor_downloader, qtype.upper(), "0", TMP_DIR, f"tensors.{qtype}.map"]
    if INFO: print(f"[Info] Fetching map for {qtype}...", file=sys.stderr)
    try:
        # If this qtype was computed previously, we consider it computed and skip gpg verification.
        if qtype in COMPUTED_QTYPES:
            if INFO:
                print(f"[Info] Map {local_map} was previously marked as computed; skipping gpg checks.", file=sys.stderr)
            return True
        elif COMPUTE_ALL_MAP and not qtype == 'bf16':
            try:
                compute_map_for_qtype(qtype)
            except Exception as e:
                if INFO:
                    print(f"[Info] Exception while computing {qtype} map: {e}", file=sys.stderr)
        else:
            # tensor_downloader.sh writes its progress messages to stdout, but
            # quant_assign.py uses stdout for the recipe itself (which is
            # commonly piped into quants_regex_merger.py). Redirect the
            # downloader's stdout to OUR stderr so its log lines join the
            # other [Info]/[Debug] log instead of polluting the recipe.
            if DEBUG or INFO:
                subprocess.run(cmd, check=True, stdout=sys.stderr.fileno())
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if INFO: print(f"[Info] Saved map to {local_map}", file=sys.stderr)

            if not SKIP_GPG:
                cmd_sig = ["bash", tensor_downloader, qtype.upper(), "-1", TMP_DIR, f"tensors.{qtype}.map.sig"]
                if INFO: print(f"[Info] Fetching map gpg signature for {qtype}...", file=sys.stderr)
                try:
                    if DEBUG or INFO:
                        subprocess.run(cmd_sig, check=True, stdout=sys.stderr.fileno())
                    else:
                        subprocess.run(cmd_sig, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if DEBUG: print(f"[Debug] Saved map gpg signature to {local_map}.sig", file=sys.stderr)
                    if not verify_detached_signature(local_map):
                        print(f"[Error] gpg signature verification of tensors.{qtype}.map failed.", file=sys.stderr)
                        ALL_GPG_SIGS_VALID = False
                        return False
                    else:
                        if INFO: print(f"[Info] gpg signature of tensors.{qtype}.map succesful.", file=sys.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"[Warning] failed to fetch tensors.map.sig: {e}", file=sys.stderr)
                    ALL_GPG_SIGS_VALID = False
                    return False
            else:
                if INFO: print(f"[Warning] gpg signature verification is disabled and won't be checked for {local_map}.sig", file=sys.stderr)

        # Record fetch and compute hashes
        _fetched_maps.add(qtype)

        # Compute and store sha256 and last_line for the map file
        import hashlib
        map_key = f"tensors.{qtype}.map"
        if map_key not in MAP_FILE_INFO:
            # Compute sha256
            with open(local_map, 'rb') as f:
                sha256sum = hashlib.sha256(f.read()).hexdigest()
            # Get last line before first ':' and '.gguf'
            with open(local_map, 'r') as f_text:
                lines = f_text.readlines()
            last_line = lines[-1].split(':', 1)[0] if lines else ''
            last_line = last_line.split('.gguf', 1)[0] if last_line else ''
            MAP_FILE_INFO[map_key] = [qtype, sha256sum, last_line]

        # Compute and store sha256 for the signature file, if applicable
        if not SKIP_GPG and not qtype in COMPUTED_QTYPES:
            sig_key = f"tensors.{qtype}.map.sig"
            sig_path = f"{local_map}.sig"
            if sig_key not in SIG_FILE_HASHES:
                with open(sig_path, 'rb') as f_sig:
                    sha256sig = hashlib.sha256(f_sig.read()).hexdigest()
                SIG_FILE_HASHES[sig_key] = sha256sig

        return True
    except subprocess.CalledProcessError as e:
        if INFO:
            print(f"[Warning] failed to fetch tensors.map: {e}", file=sys.stderr)
        # Attempt to compute map if requested
        if COMPUTE_MISSING_MAP:
            if INFO:
                print(f"[Info] Attempting to compute missing map for qtype {qtype} because --compute-missing-map is set.", file=sys.stderr)
            try:
                try:
                    compute_map_for_qtype(qtype)
                    # Record fetch and compute hashes
                    _fetched_maps.add(qtype)
                    # Compute and store sha256 and last_line for the map file
                    import hashlib
                    map_key = f"tensors.{qtype}.map"
                    if map_key not in MAP_FILE_INFO:
                        # Compute sha256
                        with open(local_map, 'rb') as f:
                            sha256sum = hashlib.sha256(f.read()).hexdigest()
                        # Get last line before first ':' and '.gguf'
                        with open(local_map, 'r') as f_text:
                            lines = f_text.readlines()
                        last_line = lines[-1].split(':', 1)[0] if lines else ''
                        last_line = last_line.split('.gguf', 1)[0] if last_line else ''
                        MAP_FILE_INFO[map_key] = [qtype, sha256sum, last_line]
                    return True
                except Exception as e:
                    if INFO:
                        print(f"[Info] compute_map_for_qtype failed for {qtype}", file=sys.stderr)
                    return False
            except Exception as ex:
                if INFO:
                    print(f"[Info] Exception while computing map for {qtype}: {ex}", file=sys.stderr)
                return False
        return False


@tracked_lru_cache(maxsize=None)
def get_map_sizes_and_elements(qtype, collect_raw = False):
    """
    Return parsed map sizes and elements for given qtype, caching results.
    """
    if qtype not in _quant_maps:
        # If caller asked for 'f32' we will probe 'bf16' and then return that result under the 'f32' key.
        probe = 'bf16' if qtype == 'f32' else qtype
        if not fetch_map_for_qtype(probe):
            print(f"Error: Fetching valid map for qtype: {probe} was unsuccessful.", file=sys.stderr)
            sys.exit(8)
        # parse_map_file now returns tuple
        _quant_maps[qtype] = parse_map_file(qtype, collect_raw)
    return _quant_maps[qtype]

def parse_map_file(qtype, collect_raw=False):
    """
    Parse local tensors.{qtype}.map into:
      - sizes: dict tensor_name -> bytes_size
      - actual_qtypes: dict tensor_name -> dtype (e.g., 'bf16', 'f32', 'q8_0', ...)
      - elements: dict tensor_name -> elements

    This version is scale-aware:
      - it captures tensor shape dims
      - it computes both base bpw and actual bpw per tensor
      - it corrects bytes for qtypes that need an additional per-row scale bump when the map file
        stored bytes still match the base equation (elements * bpw / 8)
    """
    probe = 'bf16' if qtype == 'f32' else qtype
    path = os.path.join(TMP_DIR, f"tensors.{probe}.map")
    sizes = {}
    actual_qtypes = {}
    elements = {}

    if not os.path.exists(path):
        return sizes, actual_qtypes, elements

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 5:
                continue

            tensor_name = parts[2]
            # find dtype, bytes, elements fields
            dtype = None
            size_bytes = None
            elems = None
            shape_dims = []

            for p in parts[3:]:
                if p.startswith('shape='):
                    shape_dims = _parse_shape_field(p.split('=', 1)[1])
                elif p.startswith('dtype='):
                    dtype = transform_q_suffix(p.split('=', 1)[1]).upper()
                elif p.startswith('bytes='):
                    try:
                        size_bytes = int(p.split('=', 1)[1])
                    except Exception:
                        size_bytes = None
                elif p.startswith('elements='):
                    try:
                        elems = int(p.split('=', 1)[1])
                    except Exception:
                        elems = None

            if dtype is None or size_bytes is None or elems is None:
                # skip incomplete lines
                continue

            raw_bytes = int(size_bytes)
            base_bytes = raw_bytes
            corrected_bytes = raw_bytes
            expected_base = raw_bytes

            # Detect whether this qtype needs an additional scale bump and whether the map file
            # still only contains the base equation size (elements * bpw / 8).
            bpw_hint = BPW_TABLE.get(dtype)
            if bpw_hint is None and dtype in GGML_QUANT_SIZES:
                block_size, type_size = GGML_QUANT_SIZES[dtype]
                bpw_hint = (type_size * 8.0) / block_size

            if bpw_hint is not None:
                expected_base = (elems * float(bpw_hint)) / 8.0

            if (
                bpw_hint is not None
                and expected_base is not None
                and dtype in ADDITIONAL_SCALE_FACTOR_TABLE
                and elems > 0
                and shape_dims
            ):
                if abs(raw_bytes - expected_base) < 1e-6:
                    scale_factor = ADDITIONAL_SCALE_FACTOR_TABLE[dtype]
                    tail_product = _product(shape_dims[1:]) if len(shape_dims) > 1 else 1
                    corrected_bytes = raw_bytes + (scale_factor * tail_product)

            sizes[tensor_name] = corrected_bytes
            actual_qtypes[tensor_name] = transform_q_suffix(parts[3].split('=', 1)[1]) if False else actual_qtypes.get(tensor_name, None)
            # Overwrite actual_qtypes with the parsed dtype in its original case style
            actual_qtypes[tensor_name] = transform_q_suffix(
                next((p.split('=', 1)[1] for p in parts[3:] if p.startswith('dtype=')), '')
            )

            elements[tensor_name] = elems

            # Register tensor-specific bpw values
            _register_tensor_bpw(dtype, tensor_name, corrected_bytes, expected_base, elems)

    # If NO_FALLBACK requested, synthesize faked sizes/dtypes for mismatching tensors
    if NO_FALLBACK and not collect_raw:
        for t in actual_qtypes:
            if actual_qtypes[t] != qtype:
                if INFO:
                    print(f"[Info] --no-fallback: Enforcing {qtype} qtype for {t} instead of fallback {actual_qtypes[t]} dtype present in tensors map file.", file=sys.stderr)
                actual_qtypes[t] = qtype
                sizes[t] = int(round(elements[t] * (get_bpw(qtype, tensor_name=t) / 8.0)))

    return sizes, actual_qtypes, elements

def load_sample_ppl_table(path):
    """
    Load sample PPL CSV and compute reduction factors per base name.
    """
    sample_df = pd.read_csv(path, index_col=0)
    sample_df = sample_df.replace(['404','404.0'], np.nan)
    dropped = [c for c in sample_df.columns if sample_df[c].isna().any()]
    if dropped and INFO:
        print(f"[Info] Dropping sample PPL columns with missing values: {dropped}", file=sys.stderr)
    sample_df = sample_df.drop(columns=dropped)
    max_vals = sample_df.max()
    red = sample_df.div(max_vals)
    return {col: red[col].to_dict() for col in sample_df.columns}

# --- New spread-based assignment logic ---

def compute_class_midpoint(class_values, forced_mid=None):
    """
    Compute mean PPL or use forced midpoint.
    """
    if forced_mid is not None:
        mid = forced_mid
        if DEBUG: print(f"[Debug] Forced midpoint: {mid:.4f}", file=sys.stderr)
    else:
        mid = np.mean(list(class_values.values()))
        if DEBUG: print(f"[Debug] Class midpoint (mean PPL): {mid:.4f}", file=sys.stderr)
    return mid


def compute_group_spreads(class_values, forced_mid=None):
    """
    Compute each tensor's spread in [-1,1], corrected formula for upper side.
    """
    mid = compute_class_midpoint(class_values, forced_mid)
    vals = list(class_values.values())
    min_ppl, max_ppl = min(vals), max(vals)
    down = abs(min_ppl - mid) or 1e-6
    up = abs(max_ppl - mid) or 1e-6
    spreads = {}
    for name, ppl in class_values.items():
        if ppl < mid:
            rel = (ppl - min_ppl) / down
            spread = -(1 - min(1, rel))
        else:
            rel = (max_ppl - ppl) / up  # corrected
            spread = 1 - min(1, rel)
        spreads[name] = spread
        if DEBUG: print(f"[Debug] Tensor {name}: PPL={ppl:.4f}, spread={spread:.4f}", file=sys.stderr)
    return spreads


def compute_quant_intervals(quants, stretch=1.0):
    """
    Compute normalized spread intervals from 1 to -1 per quant,
    applying stretching factor to reducing factors.
    """
    # apply stretching: new_factor = old_factor ** (1/stretch)
    widths = {}
    for q in quants:
        orig = get_default_factor(q)
        stretched = orig * (1.0 / stretch)
        #print("orig:", orig, "stretch:", stretch, "stretched:", stretched)
        widths[q] = (1 - stretched)
    total = sum(widths.values()) or 1e-6
    norm = total / 2
    intervals = []
    top = 1.0
    for q in quants:
        span = widths[q] / norm
        bottom = top - span
        intervals.append((q, top, bottom))
        if DEBUG:
            print(f"[Debug] Quant {q} @stretch={stretch:.2f}: interval ({bottom:.4f}, {top:.4f}]", file=sys.stderr)
        top = bottom
    return intervals


def assign_quants(quants, _, class_values, forced_mid=None, stretch=1.0, harmonize_groups=None):
    """
    Assign quants based on spread intervals and fetch correct tensor sizes.

    harmonize_groups: optional list-of-lists of regex strings. Matching is done
    *only* against tensors present in class_values (i.e. class_values.keys()).
    If the per-pattern match-lists (restricted to class_values) have differing
    lengths, harmonization for that group/quant is skipped with an [Info] warning.
    """
    if INFO:
        print(f"[Info] Performing spread-based quant assignment (stretch={stretch:.2f})...", file=sys.stderr)
    spreads = compute_group_spreads(class_values, forced_mid)
    intervals = compute_quant_intervals(quants, stretch)
    assignment = {}
    sizes = {}

    # Pre-compile harmonize regexes for speed (if provided)
    compiled_groups = None
    if harmonize_groups:
        if not isinstance(harmonize_groups, list):
            raise ValueError("--harmonize-quants must be a list-of-lists (or None to disable).")
        compiled_groups = []
        for g in harmonize_groups:
            if not isinstance(g, list) or len(g) < 2:
                raise ValueError("Each inner group in --harmonize-quants must be a list with >= 2 regex strings.")
            compiled_groups.append([re.compile(p) for p in g])

    # Use the names present in class_values only
    class_names = list(class_values.keys())

    # 1) Determine quant assignment (same behaviour as before)
    for name in class_names:
        spread = spreads[name]
        for q, top, bottom in intervals:
            if bottom < spread <= top:
                assignment[name] = q
                break
        else:
            assignment[name] = quants[-1]

    # 2) Compute sizes and apply index-wise harmonization when applicable
    for name in class_names:
        q_assigned = assignment[name]
        sizes_map, _, _ = get_map_sizes_and_elements(q_assigned)  # sizes_map: {tensor_name: size_in_bytes}
        base_size = sizes_map.get(name, 0)
        final_size = base_size

        if compiled_groups:
            matched_group = None
            matched_pattern_idx = -1
            matched_group_idx = -1

            # find which group & which pattern within that group matched this name
            for gi, compiled in enumerate(compiled_groups):
                for pi, cre in enumerate(compiled):
                    if cre.search(name):
                        matched_group = compiled
                        matched_pattern_idx = pi
                        matched_group_idx = gi
                        break
                if matched_group is not None:
                    break

            if matched_group is not None and matched_pattern_idx >= 0:
                # build candidate lists from class_names (NOT the full sizes_map keys)
                candidate_lists = []
                for cre in matched_group:
                    cands = [n for n in class_names if cre.search(n)]
                    candidate_lists.append(sorted(cands))

                lengths = [len(lst) for lst in candidate_lists]
                if len(set(lengths)) != 1:
                    # User split tensors across CPU/GPU (or other reason) — skip harmonization for this group
                    if INFO:
                        print(f"[Warning] skipping harmonization for group {matched_group_idx} quant {q_assigned} because pattern match counts differ (counts={lengths}); using per-tensor sizes.", file=sys.stderr)
                else:
                    group_n = lengths[0]
                    if group_n == 0:
                        # nothing to do
                        pass
                    else:
                        # find index of this name in the pattern-list it matched
                        my_list = candidate_lists[matched_pattern_idx]
                        if name not in my_list:
                            # shouldn't normally happen but be safe: skip harmonization
                            if INFO:
                                print(f"[Warning] {name!r} not found among pattern matches for harmonize group {matched_group_idx}; skipping harmonization for this tensor.", file=sys.stderr)
                        else:
                            idx_in = my_list.index(name)
                            # pair index-wise across all lists
                            matched_names = [lst[idx_in] for lst in candidate_lists]
                            # compute harmonized size using sizes_map for the assigned quant
                            group_sizes = [float(sizes_map.get(nm, 0)) for nm in matched_names]
                            if len(group_sizes) >= 2:
                                size_harmonized = float(sum(group_sizes)) / len(group_sizes)
                                final_size = size_harmonized
                                if INFO:
                                    details = ", ".join(f"{nm}={sizes_map.get(nm,0)}" for nm in matched_names)
                                    print(f"[Info] Size-harmonized group {matched_group_idx} quant {q_assigned} index {idx_in}: {details} -> harmonized={size_harmonized}", file=sys.stderr)

        sizes[name] = final_size

        if INFO:
            print(f"[Info] Assigned {assignment[name]} to {name} (spread={spreads[name]:.4f}) size={sizes[name]}", file=sys.stderr)

    return assignment, sizes

def total_size_for_quant(names, qtype):
    """
    Sum the map sizes for the given tensor names under the specified quant.
    """
    sizes_map, _, _ = get_map_sizes_and_elements(qtype)
    return sum(sizes_map.get(name, 0) for name in names)


def adjust_losses_with_synergy(
    synergistic_groups: List[List[str]],
    loss: Dict[str, float],
    tensor_sizes: Dict[str, Dict[str, int]],
    strength: float = 0.5,
    debug: bool = False
) -> Dict[str, float]:
    """
    Softly harmonize loss values across related tensors (synergistic groups).
    
    For each group, computes a weighted average loss (weights based on tensor size)
    and adjusts each tensor's loss toward that average, controlled by `strength`.

    Args:
        synergistic_groups: list of lists of tensor names belonging to the same layer
        loss: dict mapping tensor name -> measured loss value
        tensor_sizes: dict mapping tensor -> {quant_type: size_in_bytes}
        strength: float between 0.0 and 1.0, how strongly to bias losses toward group mean
        debug: print details if True

    Returns:
        Dict[str, float]: adjusted loss mapping
    """
    adjusted_loss = dict(loss)  # copy to modify

    for group in synergistic_groups:
        # find a quant type present in all tensors of the group
        common_quants = set.intersection(*(set(tensor_sizes.get(t, {}).keys()) for t in group))
        if not common_quants:
            if debug:
                print(f"[WARN] No common quant types found for group {group}, skipping.", file=sys.stderr)
            continue

        # use any one (e.g., largest quant) for weighting
        chosen_quant = sorted(list(common_quants))[0]
        sizes = {t: tensor_sizes[t][chosen_quant] for t in group if chosen_quant in tensor_sizes[t]}

        total_size = sum(sizes.values())
        if total_size <= 0:
            continue

        # compute weighted average loss
        weighted_avg_loss = sum(loss.get(t, 0.0) * sizes[t] for t in group if t in sizes) / total_size

        if debug:
            print(f"[SYNERGY] Group {group}", file=sys.stderr)
            print(f"  chosen_quant={chosen_quant}, weighted_avg_loss={weighted_avg_loss:.6f}", file=sys.stderr)

        # interpolate between original and group average
        for t in group:
            if t not in loss:
                continue
            orig_loss = loss[t]
            new_loss = (1 - strength) * orig_loss + strength * weighted_avg_loss
            adjusted_loss[t] = new_loss
            if debug:
                print(f"  {t}: {orig_loss:.6f} -> {new_loss:.6f}", file=sys.stderr)

    return adjusted_loss

def greedy_quant_assign(
    tensors: Iterable[str],
    tensor_sizes: Dict[str, Dict[str, int]],
    ppl_loss: Dict[str, float],
    degradation_fn: Callable[[str, str], float],
    tensor_quants: Optional[Dict[str, List[str]]],
    budget_bytes: int,
    *,
    preassign_missing_ppl: bool = True,
    debug: bool = False,
    harmonized_groups: Optional[List[List[str]]] = None,
    loss_exponent: float = 1.0,
    synergistic_groups: Optional[List[List[str]]] = None,
    synergy_strength: float = 0.0,
) -> Tuple[Dict[str, str], int]:
    """
    Greedy quant assignment using a min-heap (score = delta_deg / delta_size).
    - tensors: iterable of tensor names to assign
    - tensor_sizes[tensor][qtype] -> size in bytes (must exist for qtypes used)
    - ppl_loss[tensor] -> sensitivity (float). If missing and preassign_missing_ppl True, the tensor is kept at initial quant.
    - degradation_fn(tensor, qtype) -> float (bigger means worse quant)
    - tensor_quants[tensor] -> list of qtypes allowed for that tensor (if None, fallback to all qtypes available in tensor_sizes[t])
    - budget_bytes: absolute byte budget for this class (already adjusted by offsets)
    - harmonized_groups: optional list of lists of tensor-names
    - loss_exponent: exponent applied to ppl_loss values to adjust linearity of degradation accumulation
    - returns (assignment dict, total_size_bytes)
    """

    # --- Step 0: Copy/normalize inputs to avoid mutating caller data
    tensors = list(tensors)
    # ensure tensor_quants is present for lookups
    tensor_quants = tensor_quants or {}

    # --- Validation & normalize tensor_quants into allowed_map (ordered largest->smallest)
    allowed_map: Dict[str, List[str]] = {}
    for t in tensors:
        # allowed list fallback
        allowed = None
        if tensor_quants and t in tensor_quants and tensor_quants[t]:
            allowed = list(tensor_quants[t])
        else:
            # fallback: take all qtypes present in tensor_sizes[t]
            qtypes = list(tensor_sizes.get(t, {}).keys())
            if not qtypes:
                raise ValueError(f"No size map available for tensor '{t}' (cannot determine allowed quants).")
            allowed = qtypes

        # Filter allowed to qtypes that have sizes and degradation_factors (or callable)
        filtered = []
        for q in allowed:
            if q not in tensor_sizes.get(t, {}):
                # skip qtypes with no size info for this tensor
                continue
            # also ensure degradation_fn returns a value (don't filter here, just skip later)
            filtered.append(q)
        if not filtered:
            raise ValueError(f"No usable qtypes for tensor '{t}' after filtering (allowed: {allowed}).")

        # Sort descending by size (largest first) so index 0 is highest-quality (largest bytes)
        filtered.sort(key=lambda q: tensor_sizes[t][q], reverse=True)
        allowed_map[t] = filtered

    # --- Step 1a: Apply exponent scaling to all loss values
    ppl_loss_exp: Dict[str, float] = {}
    for t, v in ppl_loss.items():
        ppl_loss_exp[t] = float(v) ** float(loss_exponent)

    # --- Step 1b: Apply synergistic adjustment if requested
    if synergistic_groups and synergy_strength > 0.0:
        if debug:
            print(f"[GREEDY] applying synergistic adjustment (strength={synergy_strength}) to loss values", file=sys.stderr)
        ppl_loss_exp = adjust_losses_with_synergy(
            synergistic_groups=synergistic_groups,
            loss=ppl_loss_exp,
            tensor_sizes=tensor_sizes,
            strength=synergy_strength
        )

    # --- Step 2: Harmonization (logical grouping) - build merged view if requested
    # group_defs: mapping group_id -> [members]
    group_defs: Dict[str, List[str]] = {}
    if harmonized_groups:
        # harmonized_groups contain exact tensor names
        # We'll resolve each inner-group to concrete tensor names found in `tensors`.
        for i, group in enumerate(harmonized_groups):
            # group can be a list of patterns/strings
            members = []
            for pat in group:
                # treat as exact name match
                for t in tensors:
                    if t == pat:
                        if t not in members:
                            members.append(t)
            if members:
                gid = f"HARM_GROUP_{i}"
                group_defs[gid] = members

    # Build merged structures if group_defs not empty
    merged = False
    merged_tensor_sizes = {}
    merged_ppl_loss = {}
    merged_allowed_map = {}
    group_map = {}  # maps original tensor -> group_id (for expansion)
    if group_defs:
        merged = True
        # For each group, compute intersection of allowed quants, aggregated sizes and aggregated (exponentiated) losses.
        for gid, members in group_defs.items():
            # intersection of allowed quants across all members
            qs_sets = [set(allowed_map[m]) for m in members]
            common_qs = set.intersection(*qs_sets) if qs_sets else set()
            if not common_qs:
                # if no common quant among members, fall back to union but keep order careful:
                # union and then filter only quants that exist for all members when sizes computed
                common_qs = set().union(*qs_sets)

            # Build merged sizes only for qtypes that exist for at least one member,
            # and ensure we only include qtypes that have size listed for all members (otherwise aggregated size meaningless).
            qlist = []
            for q in sorted(common_qs, key=lambda q: next(iter(tensor_sizes[m][q] for m in members if q in tensor_sizes[m])), reverse=True):
                # but verify all members have size for q
                if all((q in tensor_sizes.get(m, {})) for m in members):
                    qlist.append(q)
            if not qlist:
                # as a final fallback, attempt to union any q present in members (but only include if sizes present for all)
                all_qs = sorted(set().union(*qs_sets))
                for q in all_qs:
                    if all((q in tensor_sizes.get(m, {})) for m in members):
                        qlist.append(q)
            if not qlist:
                raise ValueError(f"Unable to find compatible quants for harmonized group {gid} members {members}")

            # aggregate sizes per q
            merged_tensor_sizes[gid] = {q: sum(int(tensor_sizes[m][q]) for m in members) for q in qlist}

            # aggregated loss = sum of exponentiated losses (ppl_loss_exp)
            merged_ppl_loss[gid] = sum(ppl_loss_exp.get(m, 0.0) for m in members)

            # allowed map for group (sorted descending by size)
            merged_allowed_map[gid] = sorted(qlist, key=lambda q: merged_tensor_sizes[gid][q], reverse=True)

            # map members to group
            for m in members:
                group_map[m] = gid

        # Add ungrouped tensors into merged structures (they remain as-is)
        for t in tensors:
            if t not in group_map:
                merged_tensor_sizes[t] = tensor_sizes[t].copy()
                merged_ppl_loss[t] = ppl_loss_exp.get(t, 0.0)
                merged_allowed_map[t] = allowed_map[t][:]  # copy

        # Replace working views with merged views
        tensor_sizes = merged_tensor_sizes
        ppl_loss_exp = merged_ppl_loss
        allowed_map = merged_allowed_map
        tensors = list(tensor_sizes.keys())  # new set: group ids + ungrouped names

    # --- Initial assignment: highest allowed quant (largest size)
    assignment: Dict[str, str] = {}
    preassigned_due_to_missing_ppl = set()
    for t in tensors:
        # pick top quant from allowed_map
        top_q = allowed_map[t][0]
        assignment[t] = top_q
        # If ppl_loss missing and we should preassign, mark it and we will not push moves for it.
        # Note: use ppl_loss_exp for merged/unmerged keys
        if t not in ppl_loss_exp and preassign_missing_ppl:
            preassigned_due_to_missing_ppl.add(t)

    # --- initial total size
    total_size = 0
    for t in tensors:
        q = assignment[t]
        total_size += int(tensor_sizes[t][q])

    if debug:
        print(f"[GREEDY] initial total_size = {total_size / GIB:.3f} GiB; budget = {budget_bytes / GIB:.3f} GiB", file=sys.stderr)

    # --- prepare downgrade heap ---
    pq: List[Tuple[float, int, str, str, str]] = []
    counter = 0

    def push_moves(tensor: str, from_q: str):
        nonlocal counter
        # if tensor was preassigned due to missing ppl, do not push moves
        if tensor in preassigned_due_to_missing_ppl:
            return
        allowed = allowed_map[tensor]
        try:
            from_idx = allowed.index(from_q)
        except ValueError:
            # from_q not in list (shouldn't happen) -> skip
            return
        loss = ppl_loss_exp.get(tensor, None)
        if loss is None:
            return
        for to_q in allowed[from_idx + 1:]:
            size_from = int(tensor_sizes[tensor][from_q])
            size_to = int(tensor_sizes[tensor][to_q])
            delta_size = size_from - size_to
            if delta_size <= 0:
                continue
            deg_from = degradation_fn(tensor, from_q)
            deg_to = degradation_fn(tensor, to_q)
            if deg_from is None or deg_to is None:
                if debug:
                    print(f"[GREEDY] missing degradation estimate for {from_q} or {to_q}; skipping move {tensor}:{from_q}->{to_q}", file=sys.stderr)
                continue
            delta_deg = loss * (deg_to - deg_from)
            # Avoid division by zero, but delta_size>0 ensures denominator positive
            score = float(delta_deg) / float(delta_size)
            # push (score, counter) so heap is deterministic on ties
            heapq.heappush(pq, (score, counter, tensor, from_q, to_q))
            counter += 1

    # initialize moves from top quant for every tensor
    for t in tensors:
        push_moves(t, assignment[t])

    # --- main downgrade loop ---
    while total_size > budget_bytes and pq:
        score, _, tensor, from_q, to_q = heapq.heappop(pq)
        # stale-check: must still be at from_q
        if assignment.get(tensor) != from_q:
            # stale entry; ignore
            continue

        # Apply downgrade
        size_from = int(tensor_sizes[tensor][from_q])
        size_to = int(tensor_sizes[tensor][to_q])
        total_size -= (size_from - size_to)
        assignment[tensor] = to_q

        if debug:
            print(f"[GREEDY] downgraded {tensor}: {from_q} -> {to_q}; saved {(size_from-size_to)/GIB:.3f} GiB; new total {(total_size)/GIB:.3f} GiB", file=sys.stderr)

        # push next possible moves for this tensor (relative to its new quant)
        push_moves(tensor, to_q)

    if debug:
        print(f"[GREEDY] final total_size = {total_size / GIB:.3f} GiB", file=sys.stderr)

    # --- Second pass: promote tensors if we have headroom
    promote_pq: List[Tuple[float, int, str, str, str]] = []
    counter = 0

    def push_promotions(tensor: str, from_q: str):
        nonlocal counter
        # skip preassigned or tensors without ppl data
        if tensor in preassigned_due_to_missing_ppl or tensor not in ppl_loss_exp:
            return
        allowed = allowed_map[tensor]
        try:
            from_idx = allowed.index(from_q)
        except ValueError:
            return
        # explore upgrades to higher quants (i.e. lower indices)
        loss = ppl_loss_exp.get(tensor, 0.0)
        for to_q in reversed(allowed[:from_idx]):
            size_from = int(tensor_sizes[tensor][from_q])
            size_to = int(tensor_sizes[tensor][to_q])
            delta_size = size_to - size_from
            if delta_size <= 0:
                continue
            deg_from = degradation_fn(tensor, from_q)
            deg_to = degradation_fn(tensor, to_q)
            if deg_from is None or deg_to is None:
                if debug:
                    print(f"[GREEDY] missing degradation estimate for promotion {tensor}:{from_q}->{to_q}; skipping", file=sys.stderr)
                continue
            delta_deg = loss * (deg_from - deg_to)
            score = float(delta_deg) / float(delta_size)
            # push as max-heap (invert score)
            heapq.heappush(promote_pq, (-score, counter, tensor, from_q, to_q))
            counter += 1

    # initialize promotion opportunities
    for t in tensors:
        push_promotions(t, assignment[t])

    if debug:
        print(f"[GREEDY] starting promotion phase (headroom = {(budget_bytes - total_size)/GIB:.3f} GiB)", file=sys.stderr)

    while promote_pq:
        _, _, tensor, from_q, to_q = heapq.heappop(promote_pq)
        size_from = int(tensor_sizes[tensor][from_q])
        size_to = int(tensor_sizes[tensor][to_q])
        new_total = total_size + (size_to - size_from)
        if new_total > budget_bytes:
            # can't afford this promotion, skip
            continue

        # stale-check
        if assignment.get(tensor) != from_q:
            continue

        # apply promotion
        total_size = new_total
        assignment[tensor] = to_q

        if debug:
            print(f"[GREEDY] promoted {tensor}: {from_q} -> {to_q}; added {(size_to - size_from)/GIB:.3f} GiB; total = {total_size/GIB:.3f} GiB", file=sys.stderr)

        # push next possible promotion for this tensor
        push_promotions(tensor, to_q)

    if debug:
        print(f"[GREEDY] promotion phase done; final total_size = {total_size / GIB:.3f} GiB", file=sys.stderr)

    # --- If harmonized groups were used, expand group assignments back to original tensor names
    if merged and group_defs:
        expanded_assignment: Dict[str, str] = {}
        # group_defs maps gid -> members
        for gid, members in group_defs.items():
            q = assignment.get(gid)
            if q is None:
                # safety: if group not in assignment (shouldn't happen), skip
                continue
            for m in members:
                expanded_assignment[m] = q
        # add any ungrouped tensors (they kept their original names)
        for t in tensors:
            if t not in group_defs:
                # If t is actually a group id, skip; otherwise copy assigned quant
                if not t.startswith("HARM_GROUP_"):
                    val = assignment.get(t)
                    if val is None:
                        # safety: if no assignment exists for this ungrouped tensor, skip
                        continue
                    expanded_assignment[t] = val
        assignment = expanded_assignment

    return assignment, total_size


# ---- Greedy 2nd-pass combo numbering (for --auto-force-combo and footer disclosure) ----
# Canonical bitmask over the alteration toggles applied in the greedy second pass.
# Matches the default adaptive lattice (ADAPT_LATTICE=class,pos,tier2):
#   0=none 1=class 2=pos 3=class+pos 4=tier2 5=class+tier2 6=pos+tier2 7=class+pos+tier2
#   bit 0=class(1)  bit 1=pos(2)  bit 2=tier2(4)  bit 3=pareto(8)
#   0..7  = the 8 DISTINCT combos (no pareto); the adaptive selector only ever picks these.
#   8..15 = the same 8 combos + pareto, which is INERT (identical recipe), so they are
#           pareto-duplicates of 0..7 kept only for completeness.
_COMBO_TOGGLES = ('class', 'pos', 'tier2', 'pareto')   # bit order (defines the number)
_COMBO_DISPLAY = ('pareto', 'class', 'pos', 'tier2')   # name order (conventional, pareto first)
_COMBO_MAX = (1 << len(_COMBO_TOGGLES)) - 1            # 15

def _combo_num_to_set(n):
    """Bitmask int -> set of toggle names (bits outside the toggle list are ignored)."""
    return set(t for i, t in enumerate(_COMBO_TOGGLES) if (int(n) >> i) & 1)

def _combo_set_to_num(s):
    """Set of toggle names -> bitmask int, or None if it contains a non-canonical toggle."""
    if set(s) - set(_COMBO_TOGGLES):
        return None
    return sum(1 << i for i, t in enumerate(_COMBO_TOGGLES) if t in s)

def _combo_name(s):
    """Set of toggles -> canonical name ('none' / 'class' / 'pareto+class+pos' / ...)."""
    return '+'.join(t for t in _COMBO_DISPLAY if t in s) or 'none'

def _combo_num_from_name(name):
    """Combo name -> bitmask int (or None)."""
    return _combo_set_to_num(set() if name == 'none' else set(name.split('+')))


def auto_quant_assign(
    tensors: Iterable[str],
    tensor_sizes: Dict[str, Dict[str, int]],
    ppl_loss: Dict[str, float],
    degradation_fn: Callable[[str, str], float],
    tensor_quants: Optional[Dict[str, List[str]]],
    budget_bytes: int,
    *,
    preassign_missing_ppl: bool = True,
    debug: bool = False,
    harmonized_groups: Optional[List[List[str]]] = None,
    loss_exponent: float = 1.0,
    deg_exponent: float = 1.0,
    auto_sweep: bool = True,
    synergistic_groups: Optional[List[List[str]]] = None,
    synergy_strength: float = 0.0,
    zero_kld_threshold: float = 0.0,
    tolerance: float = 0.0,
    extra_outlier_qtypes: Optional[List[str]] = None,
    pareto_filter: bool = True,
    chosen_params_out: Optional[Dict[str, float]] = None,
    force_combo: Optional[int] = None,
) -> Tuple[Dict[str, str], int]:
    """
    Rank-preserving "shrinking-pool" quant assignment.

    Algorithm overview:
      1. Tensors whose sensitivity (kld) is ≤ zero_kld_threshold are treated as
         unused outliers and assigned the qtype that yields the smallest size for
         that tensor (drawn from the union of allowed qtypes and the user-provided
         extra_outlier_qtypes — typically the cpu/gpu --*-assign-qtype defaults).
      2. The remaining tensors are sorted by sensitivity (descending) and a
         rank-based mapping is applied that projects the sensitivity rank onto
         the available qtype pool (sorted by degradation ascending — best first).
         The most-sensitive tensor maps to the best qtype, the least-sensitive to
         the worst qtype, and intermediate tensors are distributed according to
         their unique-rank position.
      3. Iterative pool narrowing:
           Phase A — while size > budget, trim the best qtype from the pool
                     (forcing all tensors down a slot) and re-map.
           Phase B — while size ≤ budget, try to trim the worst qtype from the
                     pool (lifting the least-sensitive tensors up); accept the
                     change only if size stays ≤ budget, otherwise revert.
         Repeat A then B alternately until no further change can shrink the
         pool while satisfying the budget.
      4. Phase C — promote individual tensors from smallest-current-size first,
         while preserving rank monotonicity (a less-sensitive tensor never ends
         up with a strictly better qtype than a more-sensitive one).
    """
    # --- Step 0: Copy/normalize inputs to avoid mutating caller data
    #
    # Tolerance handling: --tolerance is a *symmetric acceptance band* around
    # the target (target ± tolerance), NOT a one-sided overshoot allowance.
    # The auto algorithm aims at the target center, so internally every
    # feasibility cap below uses `budget_bytes` strictly. Without this
    # override the brute-force / BA2 / greedy candidate filters honour
    # `budget_bytes * (1 + tolerance)`, which lets the meta pick a window
    # that sits at the upper edge of the band (e.g. user asks for 26.12 %
    # with --tolerance 0.01 and the recipe lands at 26.38 % instead of
    # close to 26.12 %). Strict-capping at budget_bytes makes the result
    # land NEAR target — possibly leaving headroom unused for cliff
    # models, but that trade-off is intentional.
    _orig_tolerance = float(tolerance)  # user --tolerance, before strict override
    tolerance = 0.0
    tensors = list(tensors)
    tensor_quants = tensor_quants or {}
    extra_outlier_qtypes = list(extra_outlier_qtypes or [])
    # Pristine copies of the caller inputs, captured before any of the
    # preliminary operations (loss adjustment, floor cap, outlier pinning)
    # mutate the working views. Used to re-run an UNRESTRICTED pure greedy
    # if the auto-sweep ends up selecting a greedy candidate (see the
    # greedy-winner re-run just before Phase C).
    _pristine_tensor_quants = (
        {t: list(v) for t, v in tensor_quants.items()} if tensor_quants else {}
    )
    _pristine_ppl_loss = dict(ppl_loss)
    # Captured alteration parameters (set during the preliminary passes
    # below) so the greedy second-pass can optionally re-apply a chosen
    # subset of them. Used for the ablation that finds which alterations
    # are least-damaging to apply in the greedy-selected round.
    _alt_class_factor: Dict[str, float] = {}
    _alt_tier2_w: set = set()
    _alt_bulk_allowed: Optional[set] = None

    # --- Step 1: Build per-tensor allowed-qtype map (sorted best→worst by degradation)
    allowed_map: Dict[str, List[str]] = {}
    for t in tensors:
        if tensor_quants and t in tensor_quants and tensor_quants[t]:
            allowed = list(tensor_quants[t])
        else:
            allowed = list(tensor_sizes.get(t, {}).keys())
        # Keep only qtypes that have a known size for this tensor.
        allowed = [q for q in allowed if q in tensor_sizes.get(t, {})]
        if not allowed:
            raise ValueError(f"No usable qtypes for tensor '{t}'.")
        # Sort by degradation ascending — index 0 is best (lowest degradation).
        def _deg_key(q, _t=t):
            try:
                v = degradation_fn(_t, q)
            except Exception:
                v = None
            return v if v is not None else float('inf')
        allowed.sort(key=_deg_key)
        allowed_map[t] = allowed

    # Per-tensor "outlier qtype candidates": union of allowed_map[t] and the
    # extra_outlier_qtypes the caller passed (e.g. --gpu/--cpu-assign-qtype). We
    # only use those qtypes when sizes are actually known for this tensor.
    outlier_qtypes_for_tensor: Dict[str, List[str]] = {}
    for t in tensors:
        cand = list(dict.fromkeys(allowed_map[t] + list(extra_outlier_qtypes)))
        cand = [q for q in cand if q in tensor_sizes.get(t, {})]
        outlier_qtypes_for_tensor[t] = cand

    # --- Step 2: Apply synergistic adjustment to RAW loss values
    # (we don't pre-apply the loss exponent — it's applied inside the score
    # function instead, so that the auto-sweep can vary it without re-running
    # the whole pipeline; rank-mapping and outlier detection are exponent-
    # invariant since they operate on the relative order, not the magnitude.)
    ppl_loss_raw: Dict[str, float] = {t: float(v) if v is not None else 0.0 for t, v in ppl_loss.items()}
    if synergistic_groups and synergy_strength > 0.0:
        if debug:
            print(f"[AUTO] applying synergistic adjustment (strength={synergy_strength}) to raw loss values", file=sys.stderr)
        ppl_loss_raw = adjust_losses_with_synergy(
            synergistic_groups=synergistic_groups,
            loss=ppl_loss_raw,
            tensor_sizes=tensor_sizes,
            strength=synergy_strength
        )

    # --- Step 4: Harmonization — merge tensors into virtual group entities
    group_defs: Dict[str, List[str]] = {}
    if harmonized_groups:
        for i, group in enumerate(harmonized_groups):
            members = []
            for pat in group:
                for t in tensors:
                    if t == pat and t not in members:
                        members.append(t)
            if members:
                gid = f"HARM_GROUP_{i}"
                group_defs[gid] = members

    merged = False
    group_map: Dict[str, str] = {}
    if group_defs:
        merged = True
        merged_tensor_sizes: Dict[str, Dict[str, int]] = {}
        merged_ppl_loss: Dict[str, float] = {}
        merged_allowed_map: Dict[str, List[str]] = {}
        merged_outlier_qtypes: Dict[str, List[str]] = {}

        for gid, members in group_defs.items():
            qs_sets = [set(allowed_map[m]) for m in members]
            common_qs = set.intersection(*qs_sets) if qs_sets else set()
            if not common_qs:
                common_qs = set().union(*qs_sets)
            qlist = [q for q in common_qs
                     if all(q in tensor_sizes.get(m, {}) for m in members)]
            if not qlist:
                raise ValueError(f"Unable to find compatible quants for harmonized group {gid} members {members}")

            merged_tensor_sizes[gid] = {q: sum(int(tensor_sizes[m][q]) for m in members) for q in qlist}
            merged_ppl_loss[gid] = sum(ppl_loss_raw.get(m, 0.0) for m in members)

            def _gdeg_key(q, _members=members):
                try:
                    v = degradation_fn(_members[0], q)
                except Exception:
                    v = None
                return v if v is not None else float('inf')
            qlist_sorted = sorted(qlist, key=_gdeg_key)
            merged_allowed_map[gid] = qlist_sorted

            outlier_cand = list(dict.fromkeys(qlist_sorted + list(extra_outlier_qtypes)))
            outlier_cand = [q for q in outlier_cand if q in merged_tensor_sizes[gid]]
            merged_outlier_qtypes[gid] = outlier_cand

            for m in members:
                group_map[m] = gid

        for t in tensors:
            if t not in group_map:
                merged_tensor_sizes[t] = tensor_sizes[t].copy()
                merged_ppl_loss[t] = ppl_loss_raw.get(t, 0.0)
                merged_allowed_map[t] = allowed_map[t][:]
                merged_outlier_qtypes[t] = outlier_qtypes_for_tensor[t][:]

        # Replace working views with merged views
        tensor_sizes_w = merged_tensor_sizes
        ppl_loss_w = merged_ppl_loss
        allowed_map_w = merged_allowed_map
        outlier_qtypes_w = merged_outlier_qtypes
        tensors_w = list(tensor_sizes_w.keys())
    else:
        tensor_sizes_w = tensor_sizes
        ppl_loss_w = ppl_loss_raw
        allowed_map_w = allowed_map
        outlier_qtypes_w = outlier_qtypes_for_tensor
        tensors_w = tensors

    # --- Step 5: Categorize tensors
    #     - outlier (zero / near-zero kld) -> minimum-size qtype
    #     - preassigned (missing kld) -> best qtype, exempted from later passes
    #     - sortable -> normal rank mapping
    outlier_tensors = set()
    preassigned = set()
    for t in tensors_w:
        if t not in ppl_loss_w:
            if preassign_missing_ppl:
                preassigned.add(t)
            continue
        if ppl_loss_w[t] <= zero_kld_threshold:
            outlier_tensors.add(t)

    assignment: Dict[str, str] = {}

    # Outlier (kld≈0) tensors -> qtype that yields the smallest size for that tensor,
    # picked from a candidate list that includes the user-provided assign-qtypes too.
    for t in outlier_tensors:
        candidates = outlier_qtypes_w.get(t) or allowed_map_w[t]
        smallest_q = min(candidates, key=lambda q: tensor_sizes_w[t][q])
        assignment[t] = smallest_q
        if debug:
            print(
                f"[AUTO] outlier (kld≤{zero_kld_threshold}) {t} -> {smallest_q} "
                f"(size={tensor_sizes_w[t][smallest_q]/GIB:.4f} GiB; picked from {candidates})",
                file=sys.stderr,
            )

    # Pre-assigned (no ppl data) -> best qtype in allowed list
    for t in preassigned:
        assignment[t] = allowed_map_w[t][0]
        if debug:
            print(f"[AUTO] preassigning {t} -> {allowed_map_w[t][0]} (missing ppl_loss)", file=sys.stderr)

    sortable = [t for t in tensors_w if t not in outlier_tensors and t not in preassigned]

    def compute_total(curr_assignment: Dict[str, str]) -> int:
        total = 0
        for t in tensors_w:
            q = curr_assignment.get(t)
            if q is None:
                # safety: fall back to best allowed
                q = allowed_map_w[t][0]
                curr_assignment[t] = q
            total += int(tensor_sizes_w[t][q])
        return total

    if not sortable:
        total_size = compute_total(assignment)
        # Expand harmonized groups
        if merged:
            expanded: Dict[str, str] = {}
            for gid, members in group_defs.items():
                q = assignment.get(gid)
                if q is None:
                    continue
                for m in members:
                    expanded[m] = q
            for t in tensors_w:
                if t in group_defs:
                    continue
                if t.startswith("HARM_GROUP_"):
                    continue
                expanded[t] = assignment.get(t, allowed_map_w[t][0])
            return expanded, total_size
        return assignment, total_size

    # Sort sortable tensors by sensitivity DESCENDING (most-sensitive first).
    sortable.sort(key=lambda t: -ppl_loss_w[t])

    # --- Step 6: Build global qtype pool (union of allowed qtypes across sortable)
    global_pool_set = set()
    for t in sortable:
        global_pool_set.update(allowed_map_w[t])

    # Sort pool by degradation ascending (best first). Use any tensor that
    # legitimately has the qtype in its allowed list to obtain the degradation
    # value (per-tensor degradation can differ when per-tensor scaling is in use,
    # but the *order* is what matters here).
    def _pool_sort_key(q):
        for tt in sortable:
            if q in allowed_map_w[tt]:
                try:
                    v = degradation_fn(tt, q)
                except Exception:
                    v = None
                if v is not None:
                    return v
        return float('inf')

    global_pool = sorted(global_pool_set, key=_pool_sort_key)

    # --- Pareto frontier filtering (per-tensor) ---
    # Rationale: a qtype q is "dominated" for tensor t if there exists another
    # qtype q' (also in t's allowed list) such that BOTH size_t(q') ≤ size_t(q)
    # AND deg_t(q') ≤ deg_t(q) (with at least one strict). A dominated qtype is
    # never preferable — it costs more bytes AND yields more degradation than
    # some alternative. Removing dominated qtypes from each tensor's allowed
    # list lets the rank-mapping step focus on truly meaningful tradeoffs.
    #
    # We also rebuild the global_pool after filtering: a qtype that is
    # dominated for every tensor disappears entirely. (Per-tensor dominance is
    # preserved because map-file fallbacks can make a qtype non-dominated for
    # *some* tensors while being dominated for others.)
    #
    # Related: a Lagrangian relaxation on (deg, size) — i.e. find the optimum
    # of Σ loss·deg + λ·size for some λ — would, for each tensor, only ever
    # select a qtype on this same Pareto frontier. So filtering up-front is the
    # correct preconditioner and is essentially free.
    if pareto_filter:
        if debug:
            print("[AUTO] Applying per-tensor Pareto-frontier filter to allowed qtypes...", file=sys.stderr)
        total_removed = 0
        dropped_qtype_counter: Counter = Counter()
        for t in list(allowed_map_w.keys()):
            allowed = allowed_map_w[t]
            if len(allowed) <= 1:
                continue
            sd: List[Tuple[str, int, float]] = []
            for q in allowed:
                size_t = int(tensor_sizes_w[t].get(q, 0))
                try:
                    d = degradation_fn(t, q)
                except Exception:
                    d = None
                d_val = float(d) if d is not None else float('inf')
                sd.append((q, size_t, d_val))
            # Sort by size ascending (ties broken by deg ascending — smaller is better)
            sd.sort(key=lambda x: (x[1], x[2]))
            kept: List[str] = []
            min_deg = float('inf')
            for q, _sz, d in sd:
                # A qtype is on the frontier iff its degradation is strictly
                # better than the minimum so far (among smaller-or-equal sizes).
                if d < min_deg:
                    kept.append(q)
                    min_deg = d
            # Restore the original allowed-list ordering (sorted by degradation ascending),
            # but only retain Pareto-kept qtypes.
            kept_set = set(kept)
            new_allowed = [q for q in allowed if q in kept_set]
            removed = len(allowed) - len(new_allowed)
            total_removed += removed
            for q in allowed:
                if q not in kept_set:
                    dropped_qtype_counter[q] += 1
            allowed_map_w[t] = new_allowed
            outlier_qtypes_w[t] = [q for q in outlier_qtypes_w[t] if q in tensor_sizes_w[t]]
        if debug and total_removed:
            top_dropped = ", ".join(f"{q}×{c}" for q, c in dropped_qtype_counter.most_common(10))
            print(
                f"[AUTO] Pareto filter removed {total_removed} (tensor,qtype) allowed-list entries; "
                f"most-dropped qtypes: {top_dropped}",
                file=sys.stderr,
            )

        # Rebuild global_pool from filtered allowed lists.
        global_pool_set = set()
        for t in sortable:
            global_pool_set.update(allowed_map_w[t])
        global_pool = sorted(global_pool_set, key=_pool_sort_key)
        if debug:
            print(f"[AUTO] Pareto-filtered global pool ({len(global_pool)} qtypes): {global_pool}", file=sys.stderr)

    # Build sens-rank lookup using unique sens values (ties get the same rank,
    # which means tensors with identical sensitivity map to the same qtype).
    sens_vals_sorted = sorted({ppl_loss_w[t] for t in sortable}, reverse=True)
    sens_to_rank: Dict[float, int] = {s: i for i, s in enumerate(sens_vals_sorted)}
    L = len(sens_vals_sorted)

    def rank_map(pool: List[str]) -> Dict[str, str]:
        K = len(pool)
        if K == 0:
            return {}
        out: Dict[str, str] = {}
        for t in sortable:
            rank = sens_to_rank[ppl_loss_w[t]]
            if L == 1:
                p_t = 0.0
            else:
                p_t = rank / (L - 1)  # 0 = most sens, 1 = least sens
            # Map rank position to pool index. Use round-half-down so ties prefer
            # the *better* qtype (lower index in the pool).
            ideal = p_t * (K - 1)
            idx = int(math.floor(ideal + 0.5 - 1e-9))
            if idx < 0:
                idx = 0
            elif idx > K - 1:
                idx = K - 1
            target_q = pool[idx]
            allowed_t = allowed_map_w[t]
            if target_q in allowed_t:
                out[t] = target_q
            else:
                # Closest allowed by pool-position
                best_q = allowed_t[0]
                best_diff = float('inf')
                for q in allowed_t:
                    if q in pool:
                        q_idx = pool.index(q)
                        diff = abs(q_idx - idx)
                        if diff < best_diff:
                            best_diff = diff
                            best_q = q
                out[t] = best_q
        return out

    # --- Step 7: Initial mapping with the full pool
    current_pool = list(global_pool)
    initial_mapping = rank_map(current_pool)
    assignment.update(initial_mapping)
    total_size = compute_total(assignment)

    if debug:
        print(f"[AUTO] initial pool ({len(current_pool)} qtypes): {current_pool}", file=sys.stderr)
        print(f"[AUTO] initial total_size = {total_size/GIB:.3f} GiB; budget = {budget_bytes/GIB:.3f} GiB", file=sys.stderr)

    # Helper: snapshot assignment for sortable tensors only (outliers/preassigned
    # stay put across phases).
    def apply_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
        a = dict(assignment)
        a.update(mapping)
        return a

    # Score function — parameterised by (p, q):
    #   score(assignment) = Σ_t loss_t^p · deg_t(q_t)^q
    # The loss exponent p amplifies the contribution of high-sensitivity
    # tensors so they get protected (large p ⇒ outlier preservation wins). The
    # degradation exponent q amplifies the *badness* of low-quality qtypes so
    # tensors at iq1_s/iq1_m get penalised even when their loss is small
    # (large q ⇒ the floor of the recipe is lifted, no tensor lands at a
    # disastrously high-deg qtype). The two exponents are auto-tuned by the
    # outer sweep; advanced users can override via --exponential-factor /
    # --deg-exponent.
    def _make_score_fn(p: float, q: float) -> Callable[[Dict[str, str]], float]:
        def fn(curr_assignment: Dict[str, str]) -> float:
            s = 0.0
            for t in tensors_w:
                qt = curr_assignment.get(t)
                if qt is None:
                    continue
                try:
                    d = degradation_fn(t, qt)
                except Exception:
                    d = None
                if d is None:
                    continue
                loss = ppl_loss_w.get(t, 0.0)
                try:
                    loss_p = float(loss) ** float(p)
                except Exception:
                    loss_p = float(loss) if loss is not None else 0.0
                try:
                    deg_q = float(d) ** float(q)
                except Exception:
                    deg_q = float(d) if d is not None else 0.0
                s += loss_p * deg_q
            return s
        return fn

    # Meta-score for the auto-sweep — data-adaptive
    # Σ (loss + mean_loss) · deg ** p_meta.
    #
    # Compute pool deg statistics AND the calibration data's max sensitivity.
    # The user's insight: max_loss (worst per-tensor kld in the calibration
    # data) sets the natural SCALE for what's tolerable model-wide. If a
    # qtype's degradation is much higher than max_loss, putting ANY tensor on
    # it contributes more damage than the model's worst tensor ever does at
    # its best qtype — that's catastrophic.
    #
    # We define each tensor's per-qtype cost using a NORMALISED degradation:
    #   weight(tensor, qtype) = (1 + deg / max_loss) ** p_meta
    # The (1 + ratio) base keeps the function smooth and always > 1, the
    # exponent makes the penalty grow super-linearly for ratio >> 1
    # (catastrophic qtypes) while staying gentle for ratio < 1 (safe qtypes).
    #
    # p_meta = log2(max_pool_deg / max_loss) auto-tunes the steepness using
    # only ratios derived from the data — no hardcoded thresholds. Models
    # whose worst qtype is far above max_loss (gemma: 12.83/1.65 ≈ 7.78 →
    # p_meta ≈ 3.0) get a strong penalty on the worst qtypes. Models with
    # similar ratios (Qwen3.5-4B: 2.60/0.27 ≈ 9.5 → p_meta ≈ 3.2) also do.
    # The ABSOLUTE deg values can differ wildly between models, but the
    # ratio-based scoring makes both behave consistently.
    _pool_degs: List[float] = []
    if sortable and global_pool:
        ref_t = sortable[0]
        for q in global_pool:
            try:
                _d = degradation_fn(ref_t, q)
            except Exception:
                _d = None
            if _d is not None:
                _pool_degs.append(float(_d))
    _pool_degs.sort(reverse=True)
    _pool_max_deg = _pool_degs[0] if _pool_degs else 1.0
    if _pool_max_deg <= 0:
        _pool_max_deg = 1.0
    # max_loss = the worst per-tensor kld in the calibration data. Use the
    # RAW (pre-harmonisation, pre-synergy) values so the scale is intrinsic
    # to the model, not affected by --harmonize-tensors choices.
    _raw_losses = [float(ppl_loss.get(t, 0.0)) for t in (tensors or [])]
    _raw_losses = [x for x in _raw_losses if x > 0]
    _max_loss = max(_raw_losses) if _raw_losses else 1.0
    if _max_loss <= 0:
        _max_loss = 1.0
    _mean_loss = (sum(_raw_losses) / len(_raw_losses)) if _raw_losses else 1.0
    if _mean_loss <= 0:
        _mean_loss = 1.0
    _deg_loss_ratio = _pool_max_deg / _max_loss
    # p_meta = log2(max_pool_deg / max_loss) + 1: the "+1" amplifies the
    # exponential-badness of catastrophic qtypes one tier above the bare
    # log2 ratio. Without it, the meta is too forgiving of placements like
    # "many low-loss tensors on iq1_m" (the compound damage from N tensors
    # all on the worst qtype is roughly N times the per-tensor cost — and
    # the user's stated principle is that qtype quality should win over
    # tensor sensitivity when many tensors are involved). The +1 lifts
    # the deg^p ratio between adjacent catastrophic qtypes from ~2× to
    # ~3-4×, which is enough to flip the meta from "greedy-like
    # concentration on iq1_m" to "spread across iq2_xs with minimal
    # iq1_m" at tight budgets — without breaking other budgets (smooth
    # models like Qwen3.5-4B that don't have a deg cliff still pick the
    # same greedy-like spread because the higher p simply weights
    # catastrophic qtypes more, which they don't use anyway).
    p_meta = max(1.0, math.log2(max(1.0, _deg_loss_ratio)) + 1.0)
    if debug:
        print(
            f"[AUTO] meta-score: max_loss={_max_loss:.4f}, mean_loss={_mean_loss:.4f}, "
            f"max_pool_deg={_pool_max_deg:.4f}, ratio={_deg_loss_ratio:.3f}; "
            f"p_meta={p_meta:.3f} (weight = (loss + mean_loss)·deg^p_meta); "
            f"large p_meta ⇒ rank-like, p_meta=1 ⇒ greedy-like",
            file=sys.stderr,
        )

    def _meta_score(curr_assignment: Dict[str, str]) -> Tuple[float, float, float]:
        # Primary key: Σ_t (loss(t) + mean_loss + count_qt) · deg(qt) ** p_meta
        #   where count_qt = number of tensors assigned to qtype qt.
        #
        # Three additive components, all multiplied by deg(q)^p_meta:
        #
        #   1. Loss-proportional `loss(t) · deg(q)^p_meta` — standard
        #      "expected damage" model, rewards spreading sensitive tensors
        #      onto low-deg qtypes.
        #
        #   2. Intrinsic `mean_loss · deg(q)^p_meta` — flat per-tensor cost
        #      so even zero-loss tensors get a "this qtype is bad" signal.
        #
        #   3. Compound `count_qt · deg(q)^p_meta` — the user-observed
        #      compound damage. N tensors on a single catastrophic qtype
        #      cause O(N²) total compound damage (per-tensor cost grows
        #      linearly with the count of tensors sharing that qtype, AND
        #      there are count tensors paying it). This makes the meta
        #      prefer a balanced "many at iq2_xs + few at iq1_m" over a
        #      "few at mid-tier + many at iq1_m" shape — without this term
        #      the cheap mid-tier (iq3_xxs, q2_K) lets greedy concentrate
        #      half the model on iq1_m essentially for free.
        #
        # p_meta = log2(max_pool_deg / max_loss) auto-tunes the cliff
        # steepness: gemma (ratio ≈ 7.8 → p_meta ≈ 3) gets iq1_s weight
        # 12.8^3 ≈ 2100, iq1_m ≈ 218, iq2_xs ≈ 92; Qwen3.5-4B (ratio ≈ 9.5
        # → p_meta ≈ 3.25) gets a similarly steep curve relative to its
        # smaller absolute degs.
        #
        # Secondary key: outlier loss·deg contribution (tie-break helps
        # protect outliers).
        # Tertiary key: sum of degs (tie-break favours narrower distributions).
        primary = 0.0
        outlier_ld = 0.0
        sum_d = 0.0
        # First pass: count occurrences per qtype.
        _count_per_q: Dict[str, int] = {}
        for t in tensors_w:
            qt = curr_assignment.get(t)
            if qt is None:
                continue
            _count_per_q[qt] = _count_per_q.get(qt, 0) + 1
        # Second pass: per-tensor accumulation with the compound count term.
        for t in tensors_w:
            qt = curr_assignment.get(t)
            if qt is None:
                continue
            try:
                d = degradation_fn(t, qt)
            except Exception:
                d = None
            if d is None:
                continue
            d = float(d)
            loss = float(ppl_loss_w.get(t, 0.0))
            try:
                qweight = d ** p_meta
            except Exception:
                qweight = d
            _count_q = float(_count_per_q[qt])
            primary += (loss + _mean_loss + _count_q) * qweight
            sum_d += d
            if t in high_outliers_set:
                outlier_ld += loss * d
        # PRIMARY lex key: outlier_excess = max over outliers of how much
        # the outlier's actual deg exceeds its target_deg. Zero when the
        # outlier sits at-or-below target (honors the cap), positive when
        # it overshoots. Lex-ordering by this term first means we ALWAYS
        # prefer candidates that honour the outlier target — even if a
        # competitor scores a microscopic bulk-meta win, candidates that
        # park the model's most-sensitive tensor on a worse-than-target
        # qtype lose. Falls back to the bulk meta only as a tie-break
        # (i.e. when all candidates honour the target equally, or all
        # candidates equally violate it).
        outlier_excess = 0.0
        # Per-outlier target deg = the Pareto-step-derived target qtype's
        # deg (see outlier_target_idx computation in Step 7d). Penalises
        # any candidate whose outlier qtype is WORSE (higher deg) than
        # the Pareto-walk target. Zero when at-or-below target.
        for ot in high_outliers:
            qt_o = curr_assignment.get(ot)
            if qt_o is None:
                continue
            try:
                d_o = float(degradation_fn(ot, qt_o))
            except Exception:
                d_o = None
            if d_o is None:
                continue
            try:
                target_d_o = float(outlier_allowed[ot][outlier_target_idx[ot]][1])
            except Exception:
                target_d_o = float('inf')
            excess = d_o - target_d_o
            if excess > outlier_excess:
                outlier_excess = excess
        return (outlier_excess, primary, outlier_ld, sum_d)

    # --- Step 7b: Detect *high-sensitivity outliers* — tensors whose loss is
    # so much larger than the rest that a uniform rank-mapping would under-rate
    # them (they end up wherever the window's best qtype is, which can still be
    # too low when the budget forces a narrow window). For each window we'll
    # explore, an outlier is allowed to *decouple* from the rank mapping and
    # pick any qtype in its allowed list that's at least as good as the window's
    # best qtype (preserving rank monotonicity). This matches the intuition
    # behind hand-tuned recipes that lift the embedding/output tensors above
    # the rest.
    #
    # Detection: standard IQR on the *raw* ppl_loss values (not exponent-scaled
    # — otherwise the detection becomes unstable with --exponential-factor),
    # with a conservative multiplier (k=3) AND a minimum-gap requirement (the
    # outlier's loss must be at least 1.5× the max non-outlier loss). This
    # avoids flagging mildly-above-Q3 tensors.
    high_outliers: List[str] = []
    if len(sortable) >= 4:
        # Raw loss values (before exponent / synergy adjustments) for stable
        # outlier detection independent of the score's loss exponent.
        raw_loss = {t: float(ppl_loss.get(t, 0.0)) for t in sortable}
        sens_sorted_asc = sorted(raw_loss.values())
        n_ss = len(sens_sorted_asc)
        q1_idx = max(0, n_ss // 4)
        q3_idx = min(n_ss - 1, (3 * n_ss) // 4)
        q1_v = sens_sorted_asc[q1_idx]
        q3_v = sens_sorted_asc[q3_idx]
        iqr_v = q3_v - q1_v
        upper_bound = q3_v + 3.0 * iqr_v
        candidate_outliers = [t for t in sortable if raw_loss[t] > upper_bound]
        # Require a gap: outlier's loss must be ≥ 1.5× the highest non-outlier loss.
        non_outlier_max = max(
            (raw_loss[t] for t in sortable if t not in candidate_outliers),
            default=0.0,
        )
        for t in candidate_outliers:
            if non_outlier_max <= 0 or raw_loss[t] >= 1.5 * non_outlier_max:
                high_outliers.append(t)
        # Sort outliers by raw sens descending (most-extreme first).
        high_outliers.sort(key=lambda t: -raw_loss[t])
        # Cap to a small number to keep the inner search cheap.
        if len(high_outliers) > 3:
            if debug:
                print(
                    f"[AUTO] Detected {len(high_outliers)} high-sens outliers; "
                    f"keeping top-3 for decoupled assignment: {high_outliers[:3]}",
                    file=sys.stderr,
                )
            high_outliers = high_outliers[:3]
        elif debug and high_outliers:
            print(f"[AUTO] Detected high-sens outliers: {high_outliers}", file=sys.stderr)

    # Rank-mapping should also place outliers (they're still part of sortable),
    # but at brute-force time we override their assignment with a decoupled
    # choice. Build a "non-outlier sortable" view for the score's tie-breaking.
    high_outliers_set = set(high_outliers)

    # --- Step 7b.5: Effective-loss adjustments (rank-mapping & meta input).
    #
    # Three independent data-driven adjustments to per-tensor calibration
    # loss values. Each addresses a known gap between the raw CSV signal
    # and what hand-tuned recipes do well. Applied in this order:
    #
    #   (A) Tier-2 outlier demotion. The tier-1 detector (Step 7b) flags
    #       only the most-extreme tensors (token_embd-like). Tensors that
    #       are statistically anomalous in the bulk distribution but
    #       don't satisfy the 1.5× gap requirement go undetected — and
    #       they dominate the top of the rank because the loss value
    #       drives ordering. Demoting them to the bulk Q1 (25th
    #       percentile) drops them to the bottom quartile so BA2-style
    #       splits put them at the bulk floor rather than the higher-
    #       quality Q0.
    #
    #       Detection is data-driven (no hardcoded count):
    #         threshold = max(Q3 + 1.5·IQR, X · median)
    #         cap       = max(1, ⌊N · OUTLIER_FRAC⌋)
    #       The dual threshold (IQR upper-fence AND a median-scaled
    #       multiplier) means tight distributions (e.g. Qwen3.5-4B with
    #       sub-millimetre IQR) don't over-trigger, and broad ones (e.g.
    #       gemma) catch all statistical outliers. The cap is a fraction
    #       of the bulk size, not a constant, so it scales with model
    #       size — small models cap small, big models cap big.
    #
    #   (B) Class-aware scaling. Tensors are grouped by structural class
    #       (attn_k, attn_v, attn_q, attn_output, ffn_down, ffn_up,
    #       ffn_gate, …). Each class has a median raw loss — some classes
    #       (e.g. attn_v in gemma) have a higher median than the global
    #       bulk median, indicating they are uniformly more sensitive
    #       across layers; others (e.g. attn_q, attn_k) have a lower
    #       class median, meaning their handful of high-loss outliers are
    #       anomalies, not class-representative.
    #
    #       Scale each tensor's loss by (class_median / global_median).
    #       This lifts the entire high-median class up the rank (so attn_v
    #       gets more spots in the higher-quality bin) and pushes the
    #       low-median class down (so attn_q's calibration noise doesn't
    #       grab top slots). Crucially — multiplicative scaling preserves
    #       per-tensor variance within the class, so individual sensitivity
    #       still differentiates within attn_v: a high-loss attn_v tensor
    #       still ranks above a low-loss attn_v tensor.
    #
    #   (C) Positional prior. First and last LAYER_EDGE_FRAC of layers are
    #       structurally critical (initial token embedding refinement and
    #       final logit projection paths). Even tensors with low raw loss
    #       at those layer indices contribute disproportionately to
    #       downstream quality. Multiplicative boost lifts edge-layer
    #       tensors into the mid-rank.
    #
    # All adjustments only mutate the EFFECTIVE loss used by rank-mapping
    # and meta_score's primary key. Tier-1 outlier detection (Step 7b)
    # already ran on raw losses, so its decoupling is unaffected.

    # Configurable knobs (data-driven where possible):
    OUTLIER_FRAC = 0.05            # tier-2 cap: max % of bulk tensors
    OUTLIER_IQR_K = 1.5            # IQR multiplier (Tukey-standard upper fence)
    CLASS_SCALE_ALPHA = 0.3        # class-aware scaling exponent (mild; 0 disables)
                                   # Multiplies each tensor's loss by
                                   # (class_median/global_median)^ALPHA. Strong
                                   # values (≥1.0) compress cross-class variance
                                   # and squeeze distributions at mid-budgets;
                                   # 0.3 retains per-class signal without losing
                                   # the rich rank-mapped spread at 33-50 %.
    LAYER_EDGE_FRAC = 0.15         # first/last layer fraction for positional prior
    LAYER_EDGE_BOOST = 1.2         # mild positional boost for edge layers
                                   # Higher values (≥1.5) over-cluster edge tensors
                                   # at the top of the rank and squeeze multi-qtype
                                   # rank-mapped windows; 1.2 lifts edge tensors
                                   # into mid-rank without breaking spread.

    def _tensor_class(_t: str) -> str:
        """Map a tensor name to its structural class.

        blk.N.<class>.weight → <class>
        <name>.weight (no blk) → <name>
        """
        _parts = _t.split('.')
        if _parts[0] == 'blk' and len(_parts) >= 3:
            return _parts[2]
        if _t.endswith('.weight'):
            return _t[:-len('.weight')]
        return _t

    if sortable:
        _eff_loss_in = {t: float(ppl_loss_w.get(t, 0.0)) for t in sortable}

        # Locate the DEG-CURVE INFLECTION — the first qtype (in deg-ASC
        # order) whose deg exceeds the model's worst calibration loss
        # (_max_loss). This marks the point where individual-tensor
        # damage per qtype starts exceeding the maximum sensitivity any
        # tensor showed in calibration: below the inflection, qtype
        # choices stay within the "expected" damage range; above it, the
        # tensor's loss times qtype's deg can blow past anything the
        # calibration ever measured.
        #
        # Using max_loss instead of a hardcoded deg-ratio handles BOTH
        # cliff curves (gemma — inflection lands at iq3_s, position 11
        # / 18 ≈ 61 %) and smooth curves (Qwen3.5-4B — inflection lands
        # later, around iq3_xxs at higher percentile) without false
        # triggers at the top of the curve (where small absolute degs
        # produce big ratios).
        #
        # The inflection's fractional position in the pool gives a
        # model-aware percentile that replaces the hardcoded 75th for
        # tier-2 outlier detection.
        _ref_for_curve = sortable[0]
        _pool_degs_for_curve: List[float] = []
        for _q in global_pool:
            try:
                _d = float(degradation_fn(_ref_for_curve, _q))
            except Exception:
                _d = None
            if _d is not None and _d > 0:
                _pool_degs_for_curve.append(_d)
        _pool_degs_for_curve.sort()
        _inflection_pos: Optional[int] = None
        if _max_loss > 0:
            for _i, _d in enumerate(_pool_degs_for_curve):
                if _d > _max_loss:
                    _inflection_pos = _i
                    break
        if _inflection_pos is not None and len(_pool_degs_for_curve) >= 2:
            _upper_pct = _inflection_pos / float(len(_pool_degs_for_curve))
        else:
            # No inflection found (max_loss above all degs) — fall back to Q3.
            _upper_pct = 0.75
        # Clamp to keep the IQR positive (upper > lower).
        _upper_pct = max(0.55, min(0.95, _upper_pct))
        _lower_pct = max(0.0, 1.0 - _upper_pct)

        # (A) TIER-2 OUTLIER DETECTION (uses RAW losses).
        _non_t1_losses = sorted(
            _eff_loss_in[t] for t in sortable
            if t not in high_outliers_set and _eff_loss_in[t] > 0
        )
        _tier2_outliers: List[str] = []
        _tier2_demote_to: Optional[float] = None
        _tier2_threshold: Optional[float] = None
        if len(_non_t1_losses) >= 8:
            _n_nt = len(_non_t1_losses)
            _q1_idx = max(0, min(_n_nt - 1, int(_n_nt * _lower_pct)))
            _q3_idx = max(0, min(_n_nt - 1, int(_n_nt * _upper_pct)))
            _q1_nt = _non_t1_losses[_q1_idx]
            _q3_nt = _non_t1_losses[_q3_idx]
            _iqr_nt = _q3_nt - _q1_nt
            if _iqr_nt > 0:
                _tier2_threshold = _q3_nt + OUTLIER_IQR_K * _iqr_nt
                _cap = max(1, int(_n_nt * OUTLIER_FRAC))
                _candidates_2 = [
                    t for t in sortable
                    if t not in high_outliers_set
                    and _eff_loss_in[t] > _tier2_threshold
                ]
                _candidates_2.sort(key=lambda t: -_eff_loss_in[t])
                _tier2_outliers = _candidates_2[:_cap]
                _tier2_demote_to = _q1_nt

        # (B) CLASS-AWARE SCALING (uses RAW losses, excludes tier-1 + tier-2).
        _tier2_set = set(_tier2_outliers)
        _class_of: Dict[str, str] = {t: _tensor_class(t) for t in sortable}
        _class_losses: Dict[str, List[float]] = {}
        for t in sortable:
            if t in high_outliers_set or t in _tier2_set:
                continue
            _v = _eff_loss_in[t]
            if _v > 0:
                _class_losses.setdefault(_class_of[t], []).append(_v)
        # Compute class medians and global median.
        _class_median: Dict[str, float] = {}
        _all_class_values: List[float] = []
        for _c, _vs in _class_losses.items():
            if not _vs:
                continue
            _vs_sorted = sorted(_vs)
            _class_median[_c] = _vs_sorted[len(_vs_sorted) // 2]
            _all_class_values.extend(_vs)
        if _all_class_values:
            _all_class_values.sort()
            _global_median_for_class = _all_class_values[len(_all_class_values) // 2]
        else:
            _global_median_for_class = 0.0
        # Apply scaling.
        _class_factor: Dict[str, float] = {}
        _scaled = 0
        if _global_median_for_class > 0:
            for _c, _m in _class_median.items():
                if _m > 0:
                    _class_factor[_c] = (_m / _global_median_for_class) ** CLASS_SCALE_ALPHA
            for t in sortable:
                if t in high_outliers_set or t in _tier2_set:
                    continue
                _f = _class_factor.get(_class_of[t])
                if _f is not None and _f != 1.0:
                    _eff_loss_in[t] *= _f
                    _scaled += 1

        # Tier-2 outliers: demote effective loss to ZERO so they sit
        # below ALL bulk values and naturally rank at the very bottom.
        # This pushes them to the LAST qtype of any rank-mapped window
        # (= bulk floor or near-floor), without removing them from the
        # rank pool — so the bulk distribution shape is preserved while
        # tier-2 still lands at the floor. Q1 (25th percentile) was not
        # aggressive enough — too many other tensors had losses below
        # Q1 already, so tier-2 ranked mid-bottom rather than bottom.
        for t in _tier2_outliers:
            _eff_loss_in[t] = 0.0

        # (C) POSITIONAL PRIOR.
        _layer_re = re.compile(r"^blk\.(\d+)\.")
        _layer_idx_per_t: Dict[str, Optional[int]] = {}
        _max_layer = -1
        for t in sortable:
            _m = _layer_re.match(t)
            if _m:
                _li = int(_m.group(1))
                _layer_idx_per_t[t] = _li
                if _li > _max_layer:
                    _max_layer = _li
            else:
                _layer_idx_per_t[t] = None
        _boosted = 0
        if _max_layer >= 0:
            _n_layers = _max_layer + 1
            _edge_n = max(1, int(round(_n_layers * LAYER_EDGE_FRAC)))
            _first_edge = _edge_n
            _last_edge_start = _n_layers - _edge_n
            for t in sortable:
                if t in high_outliers_set:
                    continue
                # Don't boost demoted tier-2 — defeats the demotion.
                if t in _tier2_set:
                    continue
                _li = _layer_idx_per_t.get(t)
                if _li is None:
                    continue
                if _li < _first_edge or _li >= _last_edge_start:
                    _eff_loss_in[t] *= LAYER_EDGE_BOOST
                    _boosted += 1

        # Capture alteration params for the greedy second-pass ablation.
        _alt_class_factor = dict(_class_factor)
        _alt_tier2_w = set(_tier2_outliers)

        # Write back and re-sort so BA2 / brute-force see the new order.
        for t in sortable:
            ppl_loss_w[t] = _eff_loss_in[t]
        sortable.sort(key=lambda t: -ppl_loss_w[t])
        sens_vals_sorted = sorted({ppl_loss_w[t] for t in sortable}, reverse=True)
        sens_to_rank = {s: i for i, s in enumerate(sens_vals_sorted)}
        L = len(sens_vals_sorted)

        if debug:
            _t2_str = (
                f"{_tier2_outliers[:3]}{'...' if len(_tier2_outliers) > 3 else ''}"
            )
            _thresh_str = (
                f"{_tier2_threshold:.4f}"
                if _tier2_threshold is not None
                else "N/A"
            )
            _demote_str = (
                f"{_tier2_demote_to:.4f}"
                if _tier2_demote_to is not None
                else "N/A"
            )
            _class_factor_dbg = ", ".join(
                f"{c}={f:.2f}"
                for c, f in sorted(_class_factor.items(), key=lambda kv: -kv[1])
            )
            _infl_str = (
                f"inflection at pos {_inflection_pos}/{len(_pool_degs_for_curve)} "
                f"(first deg > max_loss={_max_loss:.4f}) → upper_pct={_upper_pct:.3f}"
                if _inflection_pos is not None
                else f"no inflection (max_loss above all degs); "
                     f"upper_pct={_upper_pct:.3f} (fallback Q3)"
            )
            print(
                f"[AUTO] Loss adjustment: deg-curve {_infl_str}. "
                f"Tier-2 decoupled {len(_tier2_outliers)} ({_t2_str}) above "
                f"threshold {_thresh_str} (will be pinned at bulk floor); "
                f"scaled {_scaled} bulk tensors by class-factor "
                f"[{_class_factor_dbg}]; boosted {_boosted} first/last-"
                f"{int(LAYER_EDGE_FRAC * 100)}%-layer tensors by ×{LAYER_EDGE_BOOST}.",
                file=sys.stderr,
            )

    # --- Step 7c: Lower-bound floor cap for non-outlier tensors.
    #
    # User-observed principle: catastrophic qtypes (group0 deg ≫ max_loss)
    # should be FORBIDDEN for non-outliers whenever budget permits. The
    # natural data-derived floor is the LARGEST-bpw qtype Q such that
    # assigning ALL non-outlier tensors to Q still fits the (tolerance-
    # aware) budget. Because all non-outliers can sit at Q, the floor is
    # always feasible; using anything below Q is "wasted budget" — those
    # bytes could have been spent staying at Q rather than dropping to
    # iq1_m for the bulk while iq3_xxs holds the top-sens tail.
    #
    # Applying this filter removes catastrophic qtypes from non-outlier
    # allowed lists entirely, so the brute-force / greedy candidates can
    # never even consider e.g. parking the low-loss tail on iq1_m when
    # iq2_xxs would also fit. Outliers keep their full allowed list — the
    # outlier-pinning logic with its Pareto-target cap handles their
    # progression through the frontier separately, and monotonicity vs
    # the bulk's actual qtype usage is enforced where each candidate is
    # built.
    #
    # Worked example (gemma 26.12 %, target 7.94 GiB, tolerance 1 %):
    #   all-iq2_xs   = 8.26 GiB > 8.02 → infeasible
    #   all-iq2_xxs  = 7.37 GiB ≤ 8.02 → FEASIBLE  ← floor
    #   all-iq1_m    = 6.25 GiB ≤ 8.02 → also feasible (but smaller bpw,
    #                                    so floor stays at iq2_xxs)
    # Result: iq1_m and iq1_s are dropped from non-outlier allowed lists,
    # so the algorithm uses iq2_xs + iq2_xxs for the bulk instead of
    # iq2_xs + iq1_m.
    _floor_q_for_bulk: Optional[str] = None
    # Pre-declared so Step 8 can reference them even when Step 7c's
    # floor-relaxation branch doesn't run.
    outlier_allowed: Dict[str, List[Tuple[str, float, int]]] = {}
    outlier_target_idx: Dict[str, int] = {}
    outlier_pareto_idxs: Dict[str, List[int]] = {}
    if sortable:
        _budget_tol_floor = budget_bytes * (1.0 + max(0.0, float(tolerance)))
        # We want the LARGEST-bpw qtype Q (= most quality, most bytes
        # consumed) where all-tensors-at-Q ≤ budget. global_pool is
        # sorted by DEGRADATION ASC, not bpw DESC, so for cliff models
        # (gemma) qtypes like iq1_m (deg 6.02, bpw 1.75) come BEFORE
        # iq2_xxs (deg 6.21, bpw 2.06) — even though iq2_xxs has higher
        # bpw and is the right floor when both fit budget. Compute
        # total_at_q for every qtype and pick the one with the MAX total
        # bytes that still fits — that's the largest-bpw fitting qtype.
        #
        # Outliers are included in the sum: pinning them at a higher
        # qtype only INCREASES bytes (better outlier qtype = bigger
        # bpw), so "all-at-Q including outliers ≤ budget" is a
        # conservative bound on what's actually feasible.
        _best_fitting_total = -1
        for _q_floor_cand in global_pool:
            _total_at_q = 0
            _has_all = True
            for _t_floor in tensors_w:
                if _q_floor_cand not in tensor_sizes_w[_t_floor]:
                    _has_all = False
                    break
                _total_at_q += int(tensor_sizes_w[_t_floor][_q_floor_cand])
            if not _has_all:
                continue
            if _total_at_q <= _budget_tol_floor and _total_at_q > _best_fitting_total:
                _floor_q_for_bulk = _q_floor_cand
                _best_fitting_total = _total_at_q
        _strict_floor_q_for_bulk = _floor_q_for_bulk
        _strict_floor_total_bytes = _best_fitting_total
        if _floor_q_for_bulk is not None:
            # --- Compute outlier_allowed + outlier_target_idx EARLY (before
            # the bulk filter) so the floor-relaxation check below knows the
            # outlier's target qtype and its size. These are also reused by
            # Step 8 brute-force / BA2 / greedy pinning.
            outlier_allowed: Dict[str, List[Tuple[str, float, int]]] = {}
            for ot in high_outliers:
                _o_rows: List[Tuple[str, float, int]] = []
                for q in allowed_map_w[ot]:
                    try:
                        _d = degradation_fn(ot, q)
                    except Exception:
                        _d = None
                    _sz = int(tensor_sizes_w[ot].get(q, 0))
                    _o_rows.append((q, float(_d) if _d is not None else float('inf'), _sz))
                _o_rows.sort(key=lambda r: r[1])
                outlier_allowed[ot] = _o_rows

            K_PARETO_STEPS = 3
            outlier_target_idx: Dict[str, int] = {}
            outlier_pareto_idxs: Dict[str, List[int]] = {}
            for ot in high_outliers:
                _o_rows = outlier_allowed[ot]
                # Pareto frontier on (size, deg). Sorted ASC by deg; row i
                # is Pareto iff sz_i < min(sz_j for j < i).
                _pareto_idxs: List[int] = []
                _min_sz = float('inf')
                for _i, (_q, _d, _sz) in enumerate(_o_rows):
                    if _sz < _min_sz:
                        _pareto_idxs.append(_i)
                        _min_sz = _sz
                outlier_pareto_idxs[ot] = _pareto_idxs

                # Find floor's Pareto position by SIZE matching (stable when
                # bulk and outlier deg orderings diverge).
                _floor_pareto_pos: Optional[int] = None
                _floor_sz_outlier = (
                    int(tensor_sizes_w[ot].get(_strict_floor_q_for_bulk, 0))
                    if _strict_floor_q_for_bulk in tensor_sizes_w[ot]
                    else 0
                )
                if _floor_sz_outlier > 0:
                    _pos = None
                    for _pi, _idx in enumerate(_pareto_idxs):
                        if _o_rows[_idx][2] >= _floor_sz_outlier:
                            _pos = _pi
                        else:
                            break
                    if _pos is None:
                        _pos = 0
                    _floor_pareto_pos = _pos

                if _floor_pareto_pos is not None:
                    _target_pareto_pos = max(0, _floor_pareto_pos - K_PARETO_STEPS)
                    _target_idx = _pareto_idxs[_target_pareto_pos] if _pareto_idxs else 0
                else:
                    _max_loss_idx = None
                    for _i, (_q, _d, _sz) in enumerate(_o_rows):
                        if _d <= _max_loss:
                            _max_loss_idx = _i
                        else:
                            break
                    if _max_loss_idx is None:
                        _max_loss_idx = _pareto_idxs[0] if _pareto_idxs else 0
                    _target_idx = _max_loss_idx

                outlier_target_idx[ot] = _target_idx
                if debug:
                    _floor_label = (
                        f"bulk_floor={_strict_floor_q_for_bulk} → outlier Pareto pos {_floor_pareto_pos} "
                        f"− {K_PARETO_STEPS} steps"
                        if _floor_pareto_pos is not None
                        else f"max_loss={_max_loss:.4f} fallback"
                    )
                    print(
                        f"[AUTO] Outlier {ot} target qtype = {_o_rows[_target_idx][0]} "
                        f"(deg={_o_rows[_target_idx][1]:.4f}); rule: {_floor_label}; "
                        f"Pareto frontier: {[_o_rows[_i][0] for _i in _pareto_idxs]}.",
                        file=sys.stderr,
                    )

            # --- Floor RELAXATION for outlier-target feasibility.
            #
            # The strict floor "largest-bpw uniform that fits budget" is
            # greedy: as the budget grows, it eats the entire gain by
            # bumping the floor before the outlier sees any of it. This
            # produces NON-MONOTONIC outlier qtype across budgets (e.g.
            # gemma 33 % outlier=iq4_xs but 36 % outlier=q3_K because the
            # 36 % floor jumped from q2_K to iq3_xxs, leaving zero room
            # for outlier upgrade past q3_K).
            #
            # Fix: relax the floor so that (bulk-at-relaxed-floor +
            # outlier-at-target) fits budget_tol. The relaxed floor is the
            # LARGEST-bpw Q ≤ strict_floor_bpw where this combined total
            # fits. The outlier target is computed from the STRICT floor
            # (it stays aspirational), so relaxing the floor doesn't
            # downgrade the target.
            #
            # Safety: relaxation is bounded — it never crosses into bpw
            # territory that's lower than the strict floor by more than
            # one "tier". Concretely, the next-below qtype must have deg
            # ≤ 2.0 × strict_floor_deg. This blocks catastrophic relaxation
            # (e.g. relaxing from iq2_xxs to iq1_m at very tight budgets).
            # In practice the relaxation only fires at budget transitions
            # where the strict floor jumps a bpw tier; it lets the bulk
            # pool keep one cheaper qtype available so the outlier upgrade
            # has room.
            #
            # Worked example (gemma 36 %, budget ≈ 11.04 GiB tol):
            #   - strict floor iq3_xxs: all-at-iq3_xxs ≈ 11.0, outlier
            #     q4_K extra ≈ +0.3 → 11.3 > 11.04. INFEASIBLE.
            #   - relax 1 step → q2_K: all-at-q2_K ≈ 9.5, outlier q4_K
            #     extra ≈ +0.4 → 9.9 ≤ 11.04. FEASIBLE.
            #   - relaxed floor = q2_K; bulk pool now includes q2_K so
            #     BA2 [iq3_xxs, q2_K] candidates exist; meta picks the one
            #     with outlier at q4_K (outlier_excess = 0).
            _outlier_extras_at_target = 0
            for ot in high_outliers:
                _tq = outlier_allowed[ot][outlier_target_idx[ot]][0]
                _sz_at_target = int(tensor_sizes_w[ot].get(_tq, 0))
                _sz_at_floor = int(tensor_sizes_w[ot].get(_strict_floor_q_for_bulk, 0))
                _outlier_extras_at_target += (_sz_at_target - _sz_at_floor)

            try:
                _strict_floor_deg = float(degradation_fn(sortable[0], _strict_floor_q_for_bulk))
            except Exception:
                _strict_floor_deg = None
            _relax_deg_cap = (
                2.0 * _strict_floor_deg
                if _strict_floor_deg is not None and _strict_floor_deg > 0
                else float('inf')
            )

            # Pre-compute total_at_q for all qtypes (cached for filter too).
            _total_at_q_cache: Dict[str, int] = {}
            for q in global_pool:
                _t_q_total = 0
                _has_all_q = True
                for _t_check in tensors_w:
                    if q not in tensor_sizes_w[_t_check]:
                        _has_all_q = False
                        break
                    _t_q_total += int(tensor_sizes_w[_t_check][q])
                if _has_all_q:
                    _total_at_q_cache[q] = _t_q_total

            _floor_total_with_target = _strict_floor_total_bytes + _outlier_extras_at_target
            _relaxed_floor_q = _strict_floor_q_for_bulk
            _relaxed_floor_total_bytes = _strict_floor_total_bytes
            if _floor_total_with_target > _budget_tol_floor and _outlier_extras_at_target > 0:
                # Need to relax. Find the LARGEST-bpw Q (= largest total_at_q)
                # with total_at_q < strict_total AND total_at_q + extras
                # ≤ budget_tol AND deg(Q) ≤ relax_deg_cap.
                _best_relax_total = -1
                _best_relax_q = None
                for q, _tot in _total_at_q_cache.items():
                    if _tot >= _strict_floor_total_bytes:
                        continue  # not a relaxation (same bpw or higher)
                    try:
                        _q_deg = float(degradation_fn(sortable[0], q))
                    except Exception:
                        _q_deg = float('inf')
                    if _q_deg > _relax_deg_cap:
                        continue  # too catastrophic
                    if _tot + _outlier_extras_at_target > _budget_tol_floor:
                        continue  # doesn't fit even with relaxation
                    if _tot > _best_relax_total:
                        _best_relax_total = _tot
                        _best_relax_q = q
                if _best_relax_q is not None:
                    _relaxed_floor_q = _best_relax_q
                    _relaxed_floor_total_bytes = _best_relax_total
                    _floor_q_for_bulk = _best_relax_q  # update visible floor
                    if debug:
                        print(
                            f"[AUTO] Floor RELAXED from {_strict_floor_q_for_bulk} "
                            f"(total {_strict_floor_total_bytes/GIB:.3f} GiB + outlier "
                            f"extras {_outlier_extras_at_target/GIB:.3f} GiB = "
                            f"{_floor_total_with_target/GIB:.3f} GiB > budget tol "
                            f"{_budget_tol_floor/GIB:.3f} GiB) to {_best_relax_q} "
                            f"(total {_best_relax_total/GIB:.3f} GiB + extras = "
                            f"{(_best_relax_total + _outlier_extras_at_target)/GIB:.3f} GiB ≤ tol). "
                            f"Outlier target stays aspirational from strict floor.",
                            file=sys.stderr,
                        )
                elif debug:
                    print(
                        f"[AUTO] Floor relaxation NEEDED ({_floor_total_with_target/GIB:.3f} > "
                        f"{_budget_tol_floor/GIB:.3f}) but no non-catastrophic qtype below "
                        f"{_strict_floor_q_for_bulk} fits (deg cap = {_relax_deg_cap:.4f}). "
                        f"Outlier will downgrade in pickers.",
                        file=sys.stderr,
                    )

            # Build the set of bulk-allowed qtypes = qtypes with total_at_q
            # ≥ relaxed floor's total_at_q.
            _floor_total_bytes = _relaxed_floor_total_bytes
            _bulk_allowed_set = set()
            for q, _tot in _total_at_q_cache.items():
                if _tot >= _floor_total_bytes:
                    _bulk_allowed_set.add(q)
            _alt_bulk_allowed = set(_bulk_allowed_set)  # capture for 2nd-pass ablation
            # Filter non-outlier allowed_map_w in place.
            _filtered_count = 0
            for _t_floor in tensors_w:
                if _t_floor in high_outliers_set:
                    continue
                _orig = allowed_map_w[_t_floor]
                _new = [q for q in _orig if q in _bulk_allowed_set]
                if not _new:
                    # If a tensor would be left with no allowed qtype after
                    # the filter (e.g. its map only has below-floor qtypes),
                    # restore the original to avoid making it unassignable.
                    continue
                if len(_new) != len(_orig):
                    allowed_map_w[_t_floor] = _new
                    _filtered_count += 1
            # Trim global_pool to the bulk-allowed set (preserving the
            # deg-ASC order). The brute-force window enumeration,
            # constrained-greedy sub_pools, and the budget-aware 2-qtype
            # candidate generator all iterate over global_pool — trimming
            # it here propagates the floor to all of them. Outliers are
            # unaffected because their pinning uses outlier_allowed
            # (built from the per-tensor allowed_map, which kept its full
            # list above for outliers).
            _excluded_qtypes = [q for q in global_pool if q not in _bulk_allowed_set]
            _new_pool = [q for q in global_pool if q in _bulk_allowed_set]
            _orig_pool_len = len(global_pool)
            global_pool = _new_pool
            if debug:
                print(
                    f"[AUTO] Bulk floor = {_floor_q_for_bulk} (largest-bpw qtype where "
                    f"all-tensors-at-{_floor_q_for_bulk} fits budget). "
                    f"Forbidding bulk use of: {_excluded_qtypes}. "
                    f"Filtered {_filtered_count} non-outlier allowed lists; "
                    f"trimmed global_pool from {_orig_pool_len} to {len(global_pool)} qtypes. "
                    f"Outliers keep their full allowed lists (handled by outlier pinning).",
                    file=sys.stderr,
                )

    # --- Step 8: Brute-force search over (best_idx, worst_idx) windows
    #     We enumerate every contiguous sub-window of global_pool, build the
    #     rank-mapped assignment for that window, compute total size, and keep
    #     the one with the lowest aggregated degradation score among windows
    #     that satisfy the budget.
    #
    #     Outliers are *decoupled*: for each window, each outlier independently
    #     picks the qtype in its allowed list that minimizes its contribution
    #     to the score while still fitting the remaining budget and satisfying
    #     rank monotonicity (outlier's deg ≤ window-best's deg).
    #
    #     Complexity per (p,q): O(K^2 * (N + O*K)) — cheap. The outer auto-
    #     sweep below evaluates a handful of (p,q) pairs and picks the one
    #     whose chosen assignment minimises the L∞ meta-score.
    K_full = len(global_pool)

    # Pre-compute window-pool first qtype's degradation for the outlier ceiling.
    # We use the most "deg-permissive" tensor as the reference (first in
    # global_pool order).
    def _pool_first_deg(pool_q: str) -> float:
        for tt in sortable:
            if pool_q in allowed_map_w[tt]:
                try:
                    v = degradation_fn(tt, pool_q)
                except Exception:
                    v = None
                if v is not None:
                    return float(v)
        return float('inf')

    # outlier_allowed, outlier_target_idx, outlier_pareto_idxs were
    # pre-computed in Step 7c (inside `if _floor_q_for_bulk is not None`).
    # If that branch didn't run (no qtype fits uniformly), fall back to
    # computing them here using the max_loss target rule.
    if high_outliers and not outlier_allowed:
        for ot in high_outliers:
            _o_rows: List[Tuple[str, float, int]] = []
            for q in allowed_map_w[ot]:
                try:
                    _d = degradation_fn(ot, q)
                except Exception:
                    _d = None
                _sz = int(tensor_sizes_w[ot].get(q, 0))
                _o_rows.append((q, float(_d) if _d is not None else float('inf'), _sz))
            _o_rows.sort(key=lambda r: r[1])
            outlier_allowed[ot] = _o_rows
            _pareto_idxs: List[int] = []
            _min_sz = float('inf')
            for _i, (_q, _d, _sz) in enumerate(_o_rows):
                if _sz < _min_sz:
                    _pareto_idxs.append(_i)
                    _min_sz = _sz
            outlier_pareto_idxs[ot] = _pareto_idxs
            _max_loss_idx = None
            for _i, (_q, _d, _sz) in enumerate(_o_rows):
                if _d <= _max_loss:
                    _max_loss_idx = _i
                else:
                    break
            if _max_loss_idx is None:
                _max_loss_idx = _pareto_idxs[0] if _pareto_idxs else 0
            outlier_target_idx[ot] = _max_loss_idx

    # --- Window enumeration is independent of (p, q): rank_map, outlier
    # override, and per-tensor sizes do NOT depend on the loss/deg exponents.
    # So we compute (assignment, total) once per window and then iterate the
    # (p, q) grid over these cached results. This makes the inner sweep ~free,
    # which is what lets us use a fine (p, q) grid (steps of 0.1) without
    # blowing up the runtime.
    window_data: List[Tuple[Tuple[int, int], Dict[str, str], int]] = []
    fallback_assignment: Optional[Dict[str, str]] = None
    fallback_total: int = -1
    for best_idx in range(K_full):
        for worst_idx in range(best_idx, K_full):
            pool_subset = global_pool[best_idx:worst_idx + 1]
            mapping = rank_map(pool_subset)
            cand_assignment = apply_mapping(mapping)
            base_total = compute_total(cand_assignment)

            # Outlier decoupling — independent of (p, q).
            if high_outliers:
                window_first_deg = _pool_first_deg(pool_subset[0])
                outlier_current_size = sum(
                    int(tensor_sizes_w[ot][cand_assignment[ot]]) for ot in high_outliers
                )
                non_outlier_size = base_total - outlier_current_size
                running_total = non_outlier_size
                _feas_budget = int(budget_bytes * (1.0 + max(0.0, float(tolerance))))
                # Outlier pinning: pick the target qtype if it fits; if not,
                # fall back to WORSE-quality qtypes (smaller bpw, smaller
                # size). The meta's outlier_excess key penalises worse-than-
                # target outcomes; the brute-force window enumeration and
                # BA2 split candidates SHOULD produce at least one window
                # where target fits, and that window will win the meta.
                # Walking to BETTER-quality (lower deg, larger size) makes
                # no sense here — better quality is always more expensive,
                # so if target can't fit, better certainly can't either.
                for ot in high_outliers:
                    chosen_q: Optional[str] = None
                    _tidx = outlier_target_idx[ot]
                    _t_q, _t_d, _t_sz = outlier_allowed[ot][_tidx]
                    # Phase 1: target qtype.
                    if (_t_d <= window_first_deg
                            and running_total + _t_sz <= _feas_budget):
                        chosen_q = _t_q
                        running_total += _t_sz
                    # Phase 2: worse-quality fallback (higher idx, larger deg,
                    # smaller size).
                    if chosen_q is None:
                        for i in range(_tidx + 1, len(outlier_allowed[ot])):
                            q_, d_, sz_ = outlier_allowed[ot][i]
                            if d_ > window_first_deg:
                                break
                            if running_total + sz_ <= _feas_budget:
                                chosen_q = q_
                                running_total += sz_
                                break
                    # Phase 3: monotonicity-relaxed last resort.
                    if chosen_q is None:
                        for i in range(len(outlier_allowed[ot]) - 1, -1, -1):
                            q_, d_, sz_ = outlier_allowed[ot][i]
                            if running_total + sz_ <= _feas_budget:
                                chosen_q = q_
                                running_total += sz_
                                break
                    if chosen_q is None:
                        chosen_q = cand_assignment[ot]
                        running_total += int(tensor_sizes_w[ot][chosen_q])
                    cand_assignment[ot] = chosen_q
                cand_total = running_total
            else:
                cand_total = base_total

            # Track smallest-size candidate as fallback for impossible budgets.
            if fallback_assignment is None or cand_total < fallback_total:
                fallback_assignment = dict(cand_assignment)
                fallback_total = cand_total

            window_data.append(((best_idx, worst_idx), cand_assignment, cand_total))

    # --- Budget-aware 2-qtype rank-mapped candidates.
    #
    # Uniform rank-mapping (above) splits a 2-qtype window into N/2 tensors
    # per qtype regardless of bpw differences. At tight budgets this is
    # often suboptimal: a window like ['iq2_xs', 'iq1_m'] under-fills
    # (~7.4 GiB at a 7.94-GiB target) because half the tensors end up on
    # the bpw-cheaper qtype. Greedy fills the budget tighter but with a
    # very low "soft floor" — it dumps the entire low-loss tail on iq1_m,
    # well past what the budget actually requires.
    #
    # The fix: for every (Q0, Q1) pair from the global pool (Q0 lower deg
    # = better, Q1 higher deg = worse), compute the *budget-tight* split:
    # find the largest k such that the top-k most-sensitive non-outlier
    # tensors at Q0 + the rest at Q1 (+ outliers at their cap) fits the
    # tolerance-aware budget. This produces a non-uniform rank-monotonic
    # assignment that uses Q1 only as much as budget genuinely demands —
    # exactly the "minimum catastrophic" shape the meta would naturally
    # pick from but that uniform rank-mapping doesn't enumerate.
    #
    # ALL pairs (not just adjacent): hand-tuned recipes often combine
    # NON-ADJACENT qtypes — e.g. gemma at 26.12 % uses (iq3_xxs, iq2_xxs)
    # with iq3_xxs only for the most-sensitive ~12 % of tensors and the
    # rest at the iq2_xxs floor, skipping the intermediate q2_K and
    # iq2_xs tiers entirely. Adjacent-only BA2 misses this shape because
    # (iq3_xxs, iq2_xxs) isn't adjacent in the deg-ASC pool. Enumerating
    # all O(K²) pairs (K ≈ 13-18 after floor trim) is cheap and lets the
    # meta see these "skip-tier" candidates.
    #
    # Implementation: linear scan k from N (all at Q0) down to 0 (all at
    # Q1); pick the largest k that fits.
    if sortable and len(global_pool) >= 2:
        # Reuse the deg-sorted pool computed below by the constrained-greedy
        # block. Define it inline here so order-of-code doesn't matter.
        _ref_t_ba = sortable[0]
        def _deg_of_ba(q: str) -> float:
            try:
                v = degradation_fn(_ref_t_ba, q)
            except Exception:
                v = None
            return float(v) if v is not None else float('inf')
        # Pre-compute non-outlier list sorted by sensitivity DESC for stable
        # top-k slicing. Outliers are handled separately (their qtype is
        # decided by the outlier-pinning logic the same way the brute-force
        # window candidates do it).
        _non_outliers_sorted = [t for t in sortable if t not in high_outliers_set]
        _budget_tol = int(budget_bytes * (1.0 + max(0.0, float(tolerance))))
        # Enumerate ALL pairs (Q0 lower deg, Q1 higher deg) in deg-ASC order.
        _pool_deg_sorted_for_ba = sorted(global_pool, key=_deg_of_ba)
        _ba_pair_idx = -1
        for _i_ba in range(len(_pool_deg_sorted_for_ba) - 1):
          for _j_ba in [_i_ba + 1]:  # TEMP: adjacent only
            _ba_pair_idx += 1
            _Q0 = _pool_deg_sorted_for_ba[_i_ba]      # lower deg (better)
            _Q1 = _pool_deg_sorted_for_ba[_j_ba]      # higher deg (worse)
            # Skip if any non-outlier tensor doesn't have BOTH qtypes
            # available in its allowed list.
            if any(_Q0 not in tensor_sizes_w[t] or _Q1 not in tensor_sizes_w[t]
                   for t in _non_outliers_sorted):
                continue
            # Outlier pinning for this pair. Treat the pair as a "window"
            # whose first-deg = deg(Q0) for the monotonicity bound and
            # apply the same Pareto-target / fallback logic as the brute
            # force above.
            _ba_window_first_deg = _deg_of_ba(_Q0)
            _ba_pinned: Dict[str, str] = {}
            _ba_pinned_size = 0
            _ba_outlier_ok = True
            for ot in high_outliers:
                chosen_q_ba: Optional[str] = None
                _tidx_ba = outlier_target_idx[ot]
                _t_q_ba, _t_d_ba, _t_sz_ba = outlier_allowed[ot][_tidx_ba]
                # Phase 1: target qtype.
                if (_t_d_ba <= _ba_window_first_deg
                        and _ba_pinned_size + _t_sz_ba <= _budget_tol):
                    chosen_q_ba = _t_q_ba
                    _ba_pinned_size += _t_sz_ba
                if chosen_q_ba is None:
                    for i in range(_tidx_ba + 1, len(outlier_allowed[ot])):
                        q_, d_, sz_ = outlier_allowed[ot][i]
                        if d_ > _ba_window_first_deg:
                            break
                        if _ba_pinned_size + sz_ <= _budget_tol:
                            chosen_q_ba = q_
                            _ba_pinned_size += sz_
                            break
                if chosen_q_ba is None:
                    for i in range(len(outlier_allowed[ot]) - 1, -1, -1):
                        q_, d_, sz_ = outlier_allowed[ot][i]
                        if _ba_pinned_size + sz_ <= _budget_tol:
                            chosen_q_ba = q_
                            _ba_pinned_size += sz_
                            break
                if chosen_q_ba is None:
                    _ba_outlier_ok = False
                    break
                _ba_pinned[ot] = chosen_q_ba
            if not _ba_outlier_ok:
                continue
            # Find max k (top-k non-outliers at Q0, rest at Q1) such that
            # _ba_pinned_size + sum_at_Q0 + sum_at_Q1 ≤ budget_tol.
            # Linear scan from k=N down to 0 (cheap — adjacent pair test).
            _N_no = len(_non_outliers_sorted)
            if _N_no == 0:
                continue
            # Precompute sizes for each non-outlier at Q0 and Q1.
            _sz_Q0 = [int(tensor_sizes_w[t][_Q0]) for t in _non_outliers_sorted]
            _sz_Q1 = [int(tensor_sizes_w[t][_Q1]) for t in _non_outliers_sorted]
            # total(k) = sum(_sz_Q0[:k]) + sum(_sz_Q1[k:])
            # Start from k=0 (all at Q1) and increment k.
            _total_no = sum(_sz_Q1)
            if _ba_pinned_size + _total_no > _budget_tol:
                # Even with all non-outliers at Q1 (lowest size), can't fit
                # outliers' pinned sizes. Skip this pair.
                continue
            _best_k = 0
            for _k_ba in range(1, _N_no + 1):
                # Move tensor[k-1] from Q1 → Q0.
                _delta = _sz_Q0[_k_ba - 1] - _sz_Q1[_k_ba - 1]
                _total_no += _delta
                if _ba_pinned_size + _total_no <= _budget_tol:
                    _best_k = _k_ba
                else:
                    # Roll back and stop (any larger k will also exceed).
                    _total_no -= _delta
                    break
            if _best_k == 0:
                # Degenerate: all non-outliers at Q1. Same as 1-qtype Q1
                # window — already enumerated by brute force.
                continue
            if _best_k == _N_no:
                # All non-outliers at Q0. Same as 1-qtype Q0 window —
                # already enumerated.
                continue
            # Build the assignment.
            _ba_assignment: Dict[str, str] = {}
            for _idx, t in enumerate(_non_outliers_sorted):
                _ba_assignment[t] = _Q0 if _idx < _best_k else _Q1
            _ba_assignment.update(_ba_pinned)
            _ba_total = _ba_pinned_size + _total_no
            # Sentinel window key (-2, pair_idx) so debug/pool_label knows.
            window_data.append(((-2, _ba_pair_idx), _ba_assignment, int(_ba_total)))
            if debug:
                print(
                    f"[AUTO] Added budget-aware 2-qtype candidate "
                    f"[{_Q0}, {_Q1}] k={_best_k}/{_N_no} at {_Q0} (rest at {_Q1}); "
                    f"pinned outliers: {_ba_pinned}; total={_ba_total/GIB:.3f} GiB",
                    file=sys.stderr,
                )

    # Greedy candidate: rank-mapping is UNIFORM (same number of tensors per
    # qtype in the window), but greedy's per-byte cost heuristic produces a
    # NON-UNIFORM distribution (few tensors at top quality, many at low).
    # For models with a smooth deg curve (e.g. Qwen3.5-4B), the greedy
    # distribution beats every rank-mapped window because it can give the
    # most-sensitive tensors high-quality qtypes without wasting budget on
    # 1/K of all tensors. For models with a deg cliff (e.g. gemma), greedy
    # ends up using the catastrophic qtype for some bulk tensors, which the
    # cliff-aware meta-score (q_meta) heavily penalises. Letting the meta
    # decide gives a unified algorithm that works for both extremes.
    if sortable:
        # Order the global pool by group0 degradation ASCENDING (best deg first).
        # Use the first sortable tensor as the deg reference; this is the same
        # convention used elsewhere in the module.
        _ref_t = sortable[0]
        def _deg_of(q: str) -> float:
            try:
                v = degradation_fn(_ref_t, q)
            except Exception:
                v = None
            return float(v) if v is not None else float('inf')
        _pool_by_deg = sorted(global_pool, key=_deg_of)

        # Add constrained-greedy candidates: greedy run on TOP-k best-deg
        # qtypes for k = 1..K. The k=K case is the unconstrained greedy. For
        # smaller k, the worst-deg qtypes are *removed from greedy's choice
        # set*, forcing it to spread within the safer qtypes only.
        #
        # Why this matters: for cliff models (gemma), unconstrained greedy
        # dumps the least-sensitive tensors onto iq1_m/iq1_s because those
        # qtypes have the lowest bpw — but the meta strongly disprefers
        # those placements. Constrained greedy with the catastrophic qtypes
        # removed produces a non-uniform but safe spread (e.g. most at
        # iq3_xxs, a few promoted to better qtypes by greedy's size-cost
        # heuristic). The meta then picks the *cap* k that minimises Σ loss
        # · deg^p_meta. For smooth models (Qwen3.5-4B) the unconstrained
        # greedy usually wins because catastrophic qtypes aren't catastrophic
        # there. Data-driven, no hardcoded threshold.
        # Helper: for a given sub-pool, pin each high-sensitivity outlier to
        # the BEST qtype in its allowed list that still leaves room for the
        # non-outliers at the WORST qtype in sub-pool. This is the "sticky
        # outliers" mechanism — at tight budgets, sensitive tensors stay
        # high while the bulk descends. Without this, greedy's downgrade
        # heuristic (delta_deg / delta_size) happily downgrades huge
        # high-loss tensors like token_embd to iq1_m because the size
        # savings dominate the per-step damage. By pre-pinning outliers we
        # remove them from greedy's downgrade candidates entirely.
        def _pin_outliers_for_subpool(_sub_tensor_quants_local, _sub_pool_set, _pool_worst_q):
            """Pin outliers to the best feasible qtype with monotonicity.

            Returns (pinned, pinned_total) or (None, None) if the candidate
            is infeasible (no monotonic outlier qtype fits the budget).

            Monotonicity rule: outlier's deg must be ≤ deg of the sub-pool's
            *worst* qtype (the one greedy might use for the least-sensitive
            non-outlier tensors). This keeps the most-sensitive tensors at
            least as good as anything greedy will assign. Without this
            constraint, very tight budgets cause the outlier to be pinned
            at a *worse* qtype than non-outliers (e.g. token_embd at iq2_xs
            while non-outliers sit at iq3_xxs+q2_K), which makes no sense
            for the highest-sensitivity tensor.

            If no allowed qtype satisfies the monotonicity bound + budget,
            the candidate is rejected (returns None, None) — this prunes
            sub-pools that don't have room to honour the outlier ceiling
            and lets the meta pick a sub-pool that does.
            """
            pinned: Dict[str, str] = {}
            if not high_outliers:
                return pinned, 0
            try:
                _worst_deg = float(degradation_fn(sortable[0], _pool_worst_q))
            except Exception:
                _worst_deg = float('inf')
            # Minimum total bytes for non-outlier tensors at the smallest
            # available qtype within sub_pool.
            min_non_outlier = 0
            for t in tensors_w:
                if t in high_outliers_set:
                    continue
                sizes = [int(tensor_sizes_w[t][q]) for q in _sub_tensor_quants_local[t]]
                if not sizes:
                    return None, None  # infeasible
                min_non_outlier += min(sizes)
            # Outliers picked greedily best-first; each takes the lowest-deg
            # qtype from its full allowed list that (a) honours the
            # monotonicity bound and (b) fits the remaining budget once we
            # reserve the non-outlier minimum.
            pinned_total = 0
            # Pin each outlier to the highest-deg Pareto-optimal qtype that
            # satisfies (1) monotonicity vs sub-pool worst, (2) the target
            # cap when feasible, (3) budget. See the brute-force outlier
            # override above for the full rationale of this two-phase pick
            # (target-or-better first, monotonicity-only fallback).
            _gp_budget = int(budget_bytes * (1.0 + max(0.0, float(tolerance))))
            for ot in high_outliers:
                chosen_q = None
                _tidx_gp = outlier_target_idx[ot]
                _t_q_gp, _t_d_gp, _t_sz_gp = outlier_allowed[ot][_tidx_gp]
                # Phase 1: target qtype.
                if (_t_d_gp <= _worst_deg
                        and pinned_total + _t_sz_gp + min_non_outlier <= _gp_budget):
                    chosen_q = _t_q_gp
                    pinned_total += _t_sz_gp
                if chosen_q is None:
                    for i in range(_tidx_gp + 1, len(outlier_allowed[ot])):
                        q_, d_, sz_ = outlier_allowed[ot][i]
                        if d_ > _worst_deg:
                            break
                        if pinned_total + sz_ + min_non_outlier <= _gp_budget:
                            chosen_q = q_
                            pinned_total += sz_
                            break
                if chosen_q is None:
                    for i in range(len(outlier_allowed[ot]) - 1, -1, -1):
                        q_, d_, sz_ = outlier_allowed[ot][i]
                        if pinned_total + sz_ + min_non_outlier <= _gp_budget:
                            chosen_q = q_
                            pinned_total += sz_
                            break
                if chosen_q is None:
                    # No feasible Pareto qtype — reject this candidate.
                    return None, None
                pinned[ot] = chosen_q
            return pinned, pinned_total

        for _k in range(1, len(_pool_by_deg) + 1):
            _sub_pool = set(_pool_by_deg[:_k])
            _sub_tensor_quants = {
                t: [q for q in allowed_map_w[t] if q in _sub_pool]
                for t in tensors_w
            }
            # Skip if any tensor has no allowed qtype in this sub-pool.
            if any(not qs for qs in _sub_tensor_quants.values()):
                continue

            # Pin outliers first so they don't get crushed by greedy's
            # delta_deg/delta_size heuristic. Outliers are pinned using
            # their FULL allowed list, not sub_pool — they can go anywhere
            # (e.g. q8_0 even when sub_pool is top-3). Monotonicity is
            # enforced against the sub-pool's WORST qtype.
            _pool_worst_q = _pool_by_deg[_k - 1]
            _pinned, _pinned_size = _pin_outliers_for_subpool(_sub_tensor_quants, _sub_pool, _pool_worst_q)
            if _pinned is None:
                continue
            _non_outliers = [t for t in tensors_w if t not in _pinned]
            _remaining_budget = int(budget_bytes * (1.0 + max(0.0, float(tolerance)))) - _pinned_size
            if _remaining_budget < 0:
                continue
            _no_tensor_sizes = {t: tensor_sizes_w[t] for t in _non_outliers}
            _no_ppl_loss = {t: ppl_loss_w[t] for t in _non_outliers if t in ppl_loss_w}
            _no_tensor_quants = {t: _sub_tensor_quants[t] for t in _non_outliers}
            try:
                _g_assignment, _g_total = greedy_quant_assign(
                    tensors=_non_outliers,
                    tensor_sizes=_no_tensor_sizes,
                    ppl_loss=_no_ppl_loss,
                    degradation_fn=degradation_fn,
                    tensor_quants=_no_tensor_quants,
                    budget_bytes=int(_remaining_budget),
                    preassign_missing_ppl=preassign_missing_ppl,
                    debug=False,
                    harmonized_groups=None,
                    loss_exponent=1.0,
                )
            except Exception as _e:
                if debug:
                    print(f"[AUTO] Greedy candidate k={_k} failed: {_e}", file=sys.stderr)
                continue
            if _g_assignment is None:
                continue
            # Merge pinned outliers with greedy's assignment.
            _full_assignment = dict(_g_assignment)
            _full_assignment.update(_pinned)
            _full_total = int(_g_total) + int(_pinned_size)
            if _full_total <= int(budget_bytes * (1.0 + max(0.0, float(tolerance)))):
                # Sentinel window key (-1, _k): _k = number of qtypes in
                # greedy's allowed pool (sorted by deg, best first). _k = K
                # is unconstrained greedy.
                window_data.append(((-1, _k), _full_assignment, _full_total))
                if debug:
                    _excluded = _pool_by_deg[_k:]
                    _pin_str = ', '.join(f"{t}={q}" for t, q in _pinned.items()) if _pinned else "none"
                    print(
                        f"[AUTO] Added greedy candidate k={_k} (excluded worst-deg: {_excluded}; "
                        f"pinned outliers: {_pin_str}) total={_full_total/GIB:.3f} GiB",
                        file=sys.stderr,
                    )

    # Restrict to in-budget windows for the (p, q) sweep. If none fit, the
    # fallback assignment is used.
    #
    # Budget here is `budget_bytes * (1 + tolerance)` — strict-by-default
    # (tolerance defaults to 0) but if the user passed --tolerance >0 we
    # honour it for candidate FEASIBILITY. Without this, windows like
    # ['q2_K', 'iq2_xs'] that produce a rank-mapped total slightly over
    # the strict target get rejected, leaving only candidates that
    # crash-fit by piling onto iq1_m for the low-loss bulk. With tolerance
    # applied, the meta gets to consider the slightly-over windows and
    # often picks one that avoids catastrophic qtypes entirely.
    _feas_budget = int(budget_bytes * (1.0 + max(0.0, float(tolerance))))
    feasible_windows = [(w, a, t_) for (w, a, t_) in window_data if t_ <= _feas_budget]

    def _pick_best_for(score_fn: Callable[[Dict[str, str]], float]):
        best_a, best_t, best_s, best_w = None, -1, float('inf'), (0, K_full - 1)
        for w, a, t_ in feasible_windows:
            s = score_fn(a)
            if s < best_s:
                best_s = s
                best_t = t_
                best_a = a
                best_w = w
        return best_a, best_t, best_s, best_w

    # --- Auto-sweep over (p, q) — only if the caller didn't pin both values.
    if auto_sweep:
        # Two-stage grid sweep:
        #   1. Coarse pass at 0.1 step over the full [0.5,6.0]×[0.5,4.0] box —
        #      cheap, identifies the "winning region" (which feasible window
        #      minimises the score for each (p,q)).
        #   2. Fine refinement at 0.01 step inside each *boundary* coarse cell
        #      (a cell whose winner differs from at least one neighbour). This
        #      catches winners that only appear inside a thin (p,q) ribbon and
        #      would be missed by a 0.1-only grid, but skips the bulk of the
        #      193k full-fine-grid points where the winner doesn't change.
        #
        # Total cost is dominated by the coarse pass (≈ 2k cells) plus a few
        # thousand fine refinement cells; in practice well under a second.
        coarse_step = 0.1
        fine_step = 0.01
        p_min, p_max = 0.5, 6.0
        q_min, q_max = 0.5, 4.0
        p_coarse = np.round(np.arange(p_min, p_max + 1e-9, coarse_step), 4)
        q_coarse = np.round(np.arange(q_min, q_max + 1e-9, coarse_step), 4)

        sweep_results: List[Tuple[float, float, Dict[str, str], int, float, Tuple[float, float, float], Tuple[int, int]]] = []
        n_candidates_coarse = int(p_coarse.size * q_coarse.size)
        n_candidates_fine = 0

        if feasible_windows:
            # Build numpy arrays for vectorised scoring.
            tensor_list = list(tensors_w)
            N_t = len(tensor_list)
            L_np = np.array(
                [float(ppl_loss_w.get(t, 0.0)) for t in tensor_list],
                dtype=float,
            )

            # For each in-budget window, compute the (per-tensor) deg vector.
            window_meta: List[Tuple[Tuple[int, int], Dict[str, str], int]] = []
            d_vectors: List[np.ndarray] = []
            for win, assign_, total_ in feasible_windows:
                vec = np.zeros(N_t, dtype=float)
                for i, t in enumerate(tensor_list):
                    qt = assign_.get(t)
                    if qt is None:
                        continue
                    try:
                        d = degradation_fn(t, qt)
                    except Exception:
                        d = None
                    if d is None:
                        continue
                    vec[i] = float(d)
                d_vectors.append(vec)
                window_meta.append((win, assign_, total_))

            D_np = np.stack(d_vectors) if d_vectors else np.zeros((0, N_t))
            W_count = D_np.shape[0]

            if W_count > 0:
                safe_L = np.where(L_np > 0, L_np, 1.0)
                safe_D = np.where(D_np > 0, D_np, 1.0)
                log_L = np.log(safe_L)            # (N,)
                log_D = np.log(safe_D)            # (W, N)
                mask_L = (L_np > 0).astype(float) # (N,)
                mask_D = (D_np > 0).astype(float) # (W, N)

                def _sweep_grid(p_arr: np.ndarray, q_arr: np.ndarray):
                    """Score every (p, q) on the given grid; return:
                       - coarse_grid: 2D array (len(p_arr), len(q_arr)) of winning win_idx
                       - first_pq:    win_idx -> first (p, q) seen
                    """
                    coarse_grid = np.full((p_arr.size, q_arr.size), -1, dtype=int)
                    first_pq: Dict[int, Tuple[float, float]] = {}
                    for i, p_ in enumerate(p_arr):
                        L_pow = np.exp(p_ * log_L) * mask_L
                        for j, q_ in enumerate(q_arr):
                            D_pow = np.exp(q_ * log_D) * mask_D
                            scores = D_pow @ L_pow
                            w_ = int(np.argmin(scores))
                            coarse_grid[i, j] = w_
                            if w_ not in first_pq:
                                first_pq[w_] = (float(p_), float(q_))
                    return coarse_grid, first_pq

                # === Stage 1: coarse pass ===
                coarse_grid, winner_first_pq = _sweep_grid(p_coarse, q_coarse)

                # === Stage 2: fine refinement on boundary cells ===
                # A boundary coarse cell is one whose winner differs from any
                # of its 8 neighbours. Refine inside [p_i, p_{i+1}] × [q_j, q_{j+1}]
                # at the fine step. This catches winners that occupy a sub-cell
                # region of (p, q).
                P_n, Q_n = coarse_grid.shape
                boundary_cells: List[Tuple[int, int]] = []
                for i in range(P_n):
                    for j in range(Q_n):
                        w = coarse_grid[i, j]
                        is_boundary = False
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < P_n and 0 <= nj < Q_n and coarse_grid[ni, nj] != w:
                                    is_boundary = True
                                    break
                            if is_boundary:
                                break
                        if is_boundary:
                            boundary_cells.append((i, j))

                # Build dense fine grids inside each boundary 0.1 cell. We
                # deduplicate (p, q) so cells touching share their boundary
                # points only once.
                fine_pq_set: set = set()
                for (i, j) in boundary_cells:
                    p_lo = float(p_coarse[i])
                    p_hi = float(p_coarse[min(i + 1, P_n - 1)])
                    q_lo = float(q_coarse[j])
                    q_hi = float(q_coarse[min(j + 1, Q_n - 1)])
                    p_sub = np.round(np.arange(p_lo, p_hi + 1e-9, fine_step), 4)
                    q_sub = np.round(np.arange(q_lo, q_hi + 1e-9, fine_step), 4)
                    for p_ in p_sub:
                        for q_ in q_sub:
                            fine_pq_set.add((float(p_), float(q_)))
                # Skip points already on the coarse grid (we already scored them).
                fine_pq_list = sorted(fine_pq_set)
                fine_pq_list = [
                    (p_, q_) for (p_, q_) in fine_pq_list
                    if not (round(p_ * 10) / 10 == p_ and round(q_ * 10) / 10 == q_)
                ]
                n_candidates_fine = len(fine_pq_list)

                # Score the fine points and update winner_first_pq.
                for (p_, q_) in fine_pq_list:
                    L_pow = np.exp(p_ * log_L) * mask_L
                    D_pow = np.exp(q_ * log_D) * mask_D
                    scores = D_pow @ L_pow
                    w_ = int(np.argmin(scores))
                    if w_ not in winner_first_pq:
                        winner_first_pq[w_] = (p_, q_)

                # Compute meta for EVERY feasible window — not just the ones
                # that win under some (p, q) score. The (p, q) sweep is only a
                # heuristic for discovering interesting candidates; the meta
                # ranking is the source of truth. A window like
                # ['q3_K', 'iq3_s', 'iq3_xxs'] (no catastrophic qtypes) might
                # never win the inner score under any (p, q) yet still have
                # the lowest meta because it spreads cleanly without touching
                # iq1_m/iq1_s. Without this loop, such windows are invisible
                # to the meta and the algorithm settles for greedy (which is
                # the only catastrophic-using candidate the (p, q) sweep
                # tends to surface for cliff models).
                for win_idx in range(len(window_meta)):
                    win, assign_, total_ = window_meta[win_idx]
                    # Use a (p, q) that this window wins under if any, else
                    # fall back to (1.0, 1.0) for score reporting only — the
                    # actual ranking uses meta which doesn't depend on (p, q).
                    p_, q_ = winner_first_pq.get(win_idx, (1.0, 1.0))
                    score_fn_ = _make_score_fn(p_, q_)
                    s_ = score_fn_(assign_)
                    m_ = _meta_score(assign_)
                    sweep_results.append((p_, q_, assign_, total_, s_, m_, win))

        n_candidates = n_candidates_coarse + n_candidates_fine

        if not sweep_results:
            # No in-budget windows — fall back to the smallest-size assignment.
            if debug:
                print(
                    f"[AUTO] Warning: no window fits budget {budget_bytes/GIB:.3f} GiB. "
                    f"Falling back to smallest-size assignment ({fallback_total/GIB:.3f} GiB).",
                    file=sys.stderr,
                )
            assignment = fallback_assignment or assignment
            total_size = fallback_total if fallback_total >= 0 else 0
            current_pool = list(global_pool)
            best_p = best_q = 1.0
            best_meta = _meta_score(assignment) if assignment else (0.0, 0.0, 0.0, 0.0)
            best_score = float('inf')
            best_window = (0, K_full - 1)
            unique_results = []
        else:
            # Pick the (p, q) minimising the lexicographic meta-score.
            sweep_results.sort(key=lambda r: r[5])
            best_p, best_q, assignment, total_size, best_score, best_meta, best_window = sweep_results[0]
            if best_window[0] < 0:
                # Greedy or budget-aware-2-qtype candidate won — assignment
                # isn't tied to a contiguous global_pool sub-window. Report
                # the qtypes actually used.
                _used_qtypes = sorted(set(assignment.values()), key=lambda q: global_pool.index(q) if q in global_pool else len(global_pool))
                current_pool = _used_qtypes
            else:
                current_pool = global_pool[best_window[0]:best_window[1] + 1]
            # sweep_results already contains one entry per distinct winning
            # window (deduplicated during the sweep), so re-use it.
            unique_results = list(sweep_results)
        if chosen_params_out is not None:
            chosen_params_out['loss_exponent'] = best_p
            chosen_params_out['deg_exponent'] = best_q
        if debug and sweep_results:
            def _fmt_meta(m):
                if isinstance(m, tuple):
                    return "outlier_excess={:.4f} Σ (loss+{:.4f}+count)·deg^{:.2f}={:.4g} outlier_loss·deg={:.4f} sum_deg={:.1f}".format(m[0], _mean_loss, p_meta, m[1], m[2], m[3])
                return f"{m:.4f}"
            def _pool_label(w):
                if w[0] == -1:
                    return f"GREEDY(k={w[1]})"
                if w[0] == -2:
                    # Budget-aware 2-qtype candidate (pair_idx, _).
                    _i_lbl = w[1]
                    if 0 <= _i_lbl < len(_pool_by_deg) - 1:
                        return f"BA2[{_pool_by_deg[_i_lbl]},{_pool_by_deg[_i_lbl + 1]}]"
                    return f"BA2(pair_idx={_i_lbl})"
                return str(global_pool[w[0]:w[1] + 1])
            print(
                f"[AUTO] Auto-sweep tried {n_candidates} (p,q) candidates"
                f" ({len(unique_results)} distinct assignments). "
                f"Best: p={best_p} q={best_q}; window={_pool_label(best_window)}; "
                f"total={total_size/GIB:.3f} GiB; score={best_score:.4f}; "
                f"meta=({_fmt_meta(best_meta)})",
                file=sys.stderr,
            )
            # Show DISTINCT sweep results (sorted by meta) for diagnostics.
            for p_, q_, _a, t_, s_, m_, _w in unique_results:
                pool_str = _pool_label(_w)
                print(
                    f"[AUTO] sweep  p={p_:.2f} q={q_:.2f} meta=({_fmt_meta(m_)}) "
                    f"score={s_:.4f} total={t_/GIB:.3f}GiB pool={pool_str}",
                    file=sys.stderr,
                )
    else:
        # Caller pinned exponents — single pass over the cached window data.
        score_fn = _make_score_fn(loss_exponent, deg_exponent)
        if feasible_windows:
            a, t_, s_, win = _pick_best_for(score_fn)
            assignment, total_size, best_score, best_window = a, t_, s_, win
        else:
            if debug:
                print(
                    f"[AUTO] Warning: no window fits budget {budget_bytes/GIB:.3f} GiB.",
                    file=sys.stderr,
                )
            assignment = fallback_assignment or {}
            total_size = fallback_total if fallback_total >= 0 else 0
            best_score = float('inf')
            best_window = (0, K_full - 1)
        if best_window[0] == -1:
            _used_qtypes = sorted(set(assignment.values()), key=lambda q: global_pool.index(q) if q in global_pool else len(global_pool))
            current_pool = _used_qtypes
        else:
            current_pool = global_pool[best_window[0]:best_window[1] + 1]
        if chosen_params_out is not None:
            chosen_params_out['loss_exponent'] = float(loss_exponent)
            chosen_params_out['deg_exponent'] = float(deg_exponent)
        if debug:
            print(
                f"[AUTO] Best window (p={loss_exponent}, q={deg_exponent}): "
                f"{current_pool}; total={total_size/GIB:.3f} GiB; score={best_score:.6f}",
                file=sys.stderr,
            )

    # --- Step 8.5: Greedy-winner unrestricted re-run (meta-guarded) -------
    #
    # When the auto-sweep selects a GREEDY candidate (best_window[0] == -1),
    # the winning greedy ran on the RESTRICTED working state: the bulk-floor-
    # capped pool (below-floor qtypes removed), the class-scaled / tier-2-
    # demoted / positionally-boosted effective losses, and with tier-1/tier-2
    # outliers pre-pinned out of greedy's reach. Those restrictions protect
    # cliff models (gemma) from catastrophic-qtype overuse — but on a model
    # that genuinely benefits from the full pool (Qwen3.5-35B MoE, where the
    # floor cap forbids the efficient iq1_kt trellis quant) they handicap it
    # (~0.85 PPL worse than --use-greedy-quant-assign).
    #
    # So when greedy wins, re-run a PURE greedy on the PRISTINE inputs (full
    # pool, raw losses, no floor cap / outlier pinning) — exactly as
    # --use-greedy-quant-assign would — but ADOPT it ONLY IF it is genuinely
    # better under the meta's *primary* term (the compound size-weighted
    # damage Σ (loss + mean + count)·deg^p). greedy "winning" the sweep is NOT
    # by itself a reliable signal that unrestricted greedy is wanted — it wins
    # on gemma too, where lifting the floor cap reintroduces iq1_m (319/191/…
    # tensors). On gemma the unrestricted greedy's iq1_m usage (deg ≫
    # everything) blows up the primary term, so the constrained result is
    # kept; on Qwen35 the iq1_kt spread lowers the primary term, so it is
    # adopted. The primary term — not the full meta tuple — is the arbiter,
    # because the tuple's leading outlier_excess key reflects auto's outlier-
    # pinning policy that pure greedy intentionally does not follow, and would
    # otherwise falsely veto the legitimate Qwen35 improvement.
    if best_window and best_window[0] == -1:
        # === Greedy-winner second pass: ADAPTIVE combo selection ============
        # When greedy wins the sweep, re-run pure greedy on the PRISTINE inputs
        # and, instead of one hard-coded alteration set, ENUMERATE a small
        # lattice of alteration combos {class,pos,tier2} and auto-select the
        # predicted-best SAFE one with a verified 3-veto + regime-gated selector
        # (see to_benchmark_combinations/analysis/). Patterns learned from PPL
        # benchmarks of 38 combo recipes (GLM-4.7-Flash + Qwen3.5 0.8B/122B/397B):
        #   * predicted_damage = Σ sens·curve predicts PPL for Qwen but is
        #     INVERTED for GLM tier2 (its sensitivity data over-rates the
        #     outliers tier2 demotes; crushing them frees budget → −1.2 PPL).
        #   * the disasters share recipe signatures (token_embd crushed; a top-
        #     sensitivity tensor crushed; or floor-demotion into a TIGHT body,
        #     std_bpw/range_bpw < 0.275 = "starvation"); three vetoes catch them
        #     with ≥2× redundancy. A regime gate (HBR = (sens[token_embd]+
        #     sens[output]) / max(other sens) < 5) plus a clean-floor-pusher
        #     shape promotes the GLM-style tier2 WIN. The promoter only ever
        #     runs on veto SURVIVORS, so it can never select a disaster
        #     (verified: forcing the gate open keeps 0/6 unsafe; regret 0.048).
        #   Models that don't trigger greedy-win (e.g. gemma → normal auto
        #   window flow) never reach here; the whole Qwen family has HBR≥5 so it
        #   stays conservative; only GLM-style (greedy-win + HBR<5) is promoted.
        # Config: the ADAPT_* module constants at the top of the file (ADAPT_ENABLED,
        # ADAPT_LATTICE, ADAPT_ALLOW_TIER2, ADAPT_ALLOW_CLASSPOS). To force one specific
        # combo for troubleshooting, pass the --auto-force-combo CLI flag.
        _pure_assignment = None
        _pure_total = None
        _feas_cap = int(budget_bytes * (1.0 + max(0.0, _orig_tolerance)))

        # tier-2 (merged names) → original tensors, shared by every combo build.
        _tier2_orig = set()
        for _tw in _alt_tier2_w:
            if merged and group_defs and _tw in group_defs:
                _tier2_orig.update(group_defs[_tw])
            else:
                _tier2_orig.add(_tw)

        def _sp_build(_sp_alts):
            """Build (ppl_loss, tensor_quants) for a given alteration subset."""
            _pl = dict(_pristine_ppl_loss)
            _qz = {
                t: list(_pristine_tensor_quants.get(t) or list(tensor_sizes.get(t, {}).keys()))
                for t in tensors
            }
            if _sp_alts:
                # (pareto) per-tensor Pareto-frontier filter on (size, deg).
                if 'pareto' in _sp_alts:
                    for t in tensors:
                        _al = _qz[t]
                        if len(_al) <= 1:
                            continue
                        _sd = []
                        for q in _al:
                            try:
                                _d = float(degradation_fn(t, q))
                            except Exception:
                                _d = float('inf')
                            _sd.append((q, int(tensor_sizes.get(t, {}).get(q, 0)), _d))
                        _sd.sort(key=lambda x: (x[1], x[2]))
                        _kept, _mind = [], float('inf')
                        for q, _sz, _d in _sd:
                            if _d < _mind:
                                _kept.append(q); _mind = _d
                        _ks = set(_kept)
                        _qz[t] = [q for q in _al if q in _ks]
                # (floor) restrict bulk (non-tier-1/2) tensors to bulk-allowed.
                if 'floor' in _sp_alts and _alt_bulk_allowed:
                    for t in tensors:
                        if t in high_outliers_set or t in _tier2_orig:
                            continue
                        _new = [q for q in _qz[t] if q in _alt_bulk_allowed]
                        if _new:
                            _qz[t] = _new
                # (class) multiply bulk losses by class factor.
                if 'class' in _sp_alts and _alt_class_factor:
                    for t in tensors:
                        if t in high_outliers_set or t in _tier2_orig:
                            continue
                        _f = _alt_class_factor.get(_tensor_class(t))
                        if _f and _f != 1.0 and t in _pl:
                            _pl[t] = float(_pl[t]) * _f
                # (tier2) demote tier-2 losses to zero.
                if 'tier2' in _sp_alts:
                    for t in _tier2_orig:
                        if t in _pl:
                            _pl[t] = 0.0
                # (pos) positional boost on first/last LAYER_EDGE_FRAC layers.
                if 'pos' in _sp_alts:
                    _lre = re.compile(r"^blk\.(\d+)\.")
                    _maxl = -1
                    _lidx = {}
                    for t in tensors:
                        _m = _lre.match(t)
                        _lidx[t] = int(_m.group(1)) if _m else None
                        if _lidx[t] is not None and _lidx[t] > _maxl:
                            _maxl = _lidx[t]
                    if _maxl >= 0:
                        _nl = _maxl + 1
                        _en = max(1, int(round(_nl * LAYER_EDGE_FRAC)))
                        for t in tensors:
                            if t in high_outliers_set or t in _tier2_orig:
                                continue
                            _li = _lidx.get(t)
                            if _li is None:
                                continue
                            if (_li < _en or _li >= _nl - _en) and t in _pl:
                                _pl[t] = float(_pl[t]) * LAYER_EDGE_BOOST
            return _pl, _qz

        def _sp_greedy(_pl, _qz):
            try:
                return greedy_quant_assign(
                    tensors=tensors,
                    tensor_sizes=tensor_sizes,
                    ppl_loss=_pl,
                    degradation_fn=degradation_fn,
                    tensor_quants=_qz,
                    budget_bytes=int(budget_bytes),
                    preassign_missing_ppl=preassign_missing_ppl,
                    debug=False,
                    harmonized_groups=harmonized_groups,
                    loss_exponent=loss_exponent,
                    synergistic_groups=synergistic_groups,
                    synergy_strength=synergy_strength,
                )
            except Exception as _e:
                if debug:
                    print(f"[AUTO] greedy combo run failed ({_e}).", file=sys.stderr)
                return None, None

        def _sp_greedy_pts(_pl, _qz, _k):
            """Like _sp_greedy but with per-tensor degradation scaling (loss/mean)^k,
            replicating `--use-greedy-quant-assign --per-tensor-degradation-scaling k`
            (greedy+CSV default loss_exponent=1.0). Used ONLY in the ultra-tight
            starvation regime (all combos vetoed) where the lattice / meta-score are
            uninformative and the protected greedy is the measured win."""
            _vals = {t: float(_pristine_ppl_loss.get(t, 0.0)) for t in tensors}
            _mean = (sum(_vals.values()) / len(_vals)) if _vals else 0.0
            if _mean <= 0:
                return _sp_greedy(_pl, _qz)

            def _dfn(_t, _q):
                _b = degradation_fn(_t, _q)
                if _b is None:
                    return None
                _l = _vals.get(_t)
                return _b * ((_l / _mean) ** _k) if _l is not None else _b

            try:
                return greedy_quant_assign(
                    tensors=tensors,
                    tensor_sizes=tensor_sizes,
                    ppl_loss=_pl,
                    degradation_fn=_dfn,
                    tensor_quants=_qz,
                    budget_bytes=int(budget_bytes),
                    preassign_missing_ppl=preassign_missing_ppl,
                    debug=False,
                    harmonized_groups=harmonized_groups,
                    loss_exponent=1.0,
                    synergistic_groups=synergistic_groups,
                    synergy_strength=synergy_strength,
                )
            except Exception as _e:
                if debug:
                    print(f"[AUTO] pts greedy run failed ({_e}).", file=sys.stderr)
                return None, None

        def _sp_fold(_assign):
            """Fold an EXPANDED greedy assignment into the merged view (tensors_w)
            so it can be scored by _meta_score alongside `assignment`."""
            _m: Dict[str, str] = {}
            for _t in tensors_w:
                if merged and group_defs and _t in group_defs:
                    _q = None
                    for _g in group_defs[_t]:
                        if _g in _assign:
                            _q = _assign[_g]; break
                    if _q is None:
                        return None
                    _m[_t] = _q
                elif _t in _assign:
                    _m[_t] = _assign[_t]
                else:
                    return None
            return _m

        # --auto-force-combo CLI flag forces ONE combo & disables adaptive selection.
        _force_combo = force_combo
        _adapt = ADAPT_ENABLED and (_force_combo is None)

        if _force_combo is not None:
            # ---- user-FORCED combo (adaptive selection DISABLED; troubleshooting) ----
            _alts = _combo_num_to_set(_force_combo)
            _nm = _combo_name(_alts)
            print(f"[AUTO] --auto-force-combo {_force_combo} (={_nm}): forcing this greedy "
                  f"2nd-pass alteration set; ADAPTIVE selection DISABLED.", file=sys.stderr)
            _fpl, _fqz = _sp_build(_alts)
            _fa, _ft = _sp_greedy(_fpl, _fqz)
            if _fa is not None and _ft is not None and int(_ft) <= _feas_cap:
                if chosen_params_out is not None:
                    chosen_params_out['combo'] = int(_force_combo)
                    chosen_params_out['combo_name'] = _nm
                    chosen_params_out['combo_forced'] = True
                return _fa, int(_ft)
            elif debug:
                print(f"[AUTO] forced combo {_force_combo} infeasible (> feas cap); "
                      f"keeping constrained result.", file=sys.stderr)
        elif _adapt:
            # -------- ADAPTIVE: enumerate the combo lattice and auto-select ----
            import itertools as _itertools
            import statistics as _statistics
            _toggles = list(ADAPT_LATTICE)
            _allow_promote = ADAPT_ALLOW_TIER2
            _allow_classpos = ADAPT_ALLOW_CLASSPOS

            # selector constants (validated; see analysis/ README)
            _TE_FLOOR, _K_CRIT, _REL_THR, _ABS_THR = 0.5, 15, 1.5, 0.05
            _SR_THRESH, _FLOOR_ABS, _EPS_PD = 0.275, 2.0, 0.001
            _HBR_THR, _SKEW_GATE, _LT25_GATE = 5.0, 0.5, 0.12
            _CP_RELCRIT = 0.05   # class+pos upgrade: max rel_crit (also requires pool floor < _FLOOR_ABS)
            _sens = _pristine_ppl_loss

            def _deg(t, q):
                try:
                    v = degradation_fn(t, q)
                    return float(v) if v is not None else 0.0
                except Exception:
                    return 0.0

            def _qbpw(q):
                if q is None:
                    return None
                u = str(q).upper()
                if u in BPW_TABLE:
                    return BPW_TABLE[u]
                if u in ('F32', 'BF16', 'F16'):
                    return BPW_TABLE.get(u, 32.0)
                return BPW_TABLE.get(re.sub(r'_(R4|R8)$', '', u))

            def _bpws(a):
                return [b for b in (_qbpw(a[t]) for t in a) if b is not None]

            def _pdam(a):
                return sum(_sens.get(t, 0.0) * _deg(t, a[t]) for t in a)

            def _swbpw(a):
                _n = _d = 0.0
                for t, q in a.items():
                    b = _qbpw(q); s = abs(_sens.get(t, 0.0))
                    if b is not None:
                        _n += s * b; _d += s
                return _n / _d if _d else 0.0

            def _shape(a, floor):
                bs = _bpws(a)
                if not bs:
                    return None
                mn = min(bs); rng = max(bs) - mn
                sd = _statistics.pstdev(bs) if len(bs) > 1 else 0.0
                med = _statistics.median(bs); mean = sum(bs) / len(bs)
                return {
                    'SR': (sd / rng if rng > 0 else 1.0),
                    'SKEW': med - mean,
                    'ATFLOOR': (mn <= floor + 1e-9 and floor < _FLOOR_ABS),
                    'flt25': sum(b < 2.5 for b in bs) / len(bs),
                    'flt2': sum(b < 2.0 for b in bs) / len(bs),
                }

            # enumerate the powerset of toggles → {name: (assignment, total)}
            _combos: Dict[str, Tuple[Dict[str, str], int]] = {}
            for _r in range(len(_toggles) + 1):
                for _sub in _itertools.combinations(_toggles, _r):
                    _name = '+'.join(_sub) if _sub else 'none'
                    _pl, _qz = _sp_build(set(_sub))
                    _asg, _tot = _sp_greedy(_pl, _qz)
                    if _asg is not None and _tot is not None and int(_tot) <= _feas_cap:
                        _combos[_name] = (_asg, int(_tot))

            if 'none' in _combos:
                _base = _combos['none'][0]
                _pd_none = _pdam(_base)
                _floor = min(min(_bpws(a)) for a, _ in _combos.values())
                _topK = set(sorted(_sens, key=lambda t: abs(_sens.get(t, 0.0)),
                                   reverse=True)[:_K_CRIT])
                _te = abs(_sens.get('token_embd.weight', 0.0))
                _ou = abs(_sens.get('output.weight', 0.0))
                _others = [abs(v) for k, v in _sens.items()
                           if k not in ('token_embd.weight', 'output.weight')]
                _HBR = (_te + _ou) / max(_others) if _others else 1e9

                def _veto(a):
                    _bn = _qbpw(_base.get('token_embd.weight'))
                    _bc = _qbpw(a.get('token_embd.weight'))
                    if _bn is not None and _bc is not None and _bn - _bc > _TE_FLOOR:
                        return 'V1'                          # token_embd crush
                    _mi = 0.0
                    for t in _topK:
                        _qn = _base.get(t); _qc = a.get(t)
                        if not _qn or not _qc:
                            continue
                        _b1 = _qbpw(_qn); _b2 = _qbpw(_qc)
                        if _b1 is None or _b2 is None or _b2 >= _b1:
                            continue                         # demotions only
                        _inc = _sens.get(t, 0.0) * (_deg(t, _qc) - _deg(t, _qn))
                        if _inc > _mi:
                            _mi = _inc
                    if _pd_none > 0 and _mi / _pd_none > _REL_THR and _mi > _ABS_THR:
                        return 'V2'                          # critical-tensor damage
                    _s = _shape(a, _floor)
                    if _s and _s['ATFLOOR'] and _s['SR'] < _SR_THRESH:
                        return 'V3'                          # starvation shape
                    return None

                _vlog = {n: _veto(a) for n, (a, _t) in _combos.items()}
                _surv = {n: a for n, (a, _t) in _combos.items() if not _vlog[n]}
                if not _surv:
                    # DEGENERATE ULTRA-TIGHT "starvation" regime: every combo floor-
                    # demotes into a tight body so V3 vetoes them ALL, leaving the
                    # lattice with no signal. predicted_damage is INVERTED here (the
                    # lowest-damage recipe has the WORST PPL), so the conservative
                    # selector + meta-guard would just pick the under-protective
                    # 'none' greedy. Instead adopt a sensitivity-PROTECTED greedy
                    # (per-tensor degradation scaling), the measured win for ultra-
                    # tight MoE budgets (Qwen3.6-35B-A3B 1.7030: 10.49 -> 10.06).
                    # Reachable ONLY when ALL combos are vetoed — no previously-
                    # validated recipe hits this, so they stay byte-identical.
                    if ADAPT_STARVE_PTS and ADAPT_STARVE_PTS > 0:
                        _pl0, _qz0 = _sp_build(set())
                        _spa, _spt = _sp_greedy_pts(_pl0, _qz0, ADAPT_STARVE_PTS)
                        if (_spa is not None and _spt is not None
                                and int(_spt) <= _feas_cap):
                            if debug:
                                print(
                                    f"[AUTO] adaptive selector: all {len(_combos)} "
                                    f"combos starvation-vetoed (ultra-tight budget); "
                                    f"adopting sensitivity-protected greedy "
                                    f"(pts={ADAPT_STARVE_PTS:g}; "
                                    f"pred_damage={_pdam(_spa):.4g}; "
                                    f"{_spt/GIB:.3f} GiB).",
                                    file=sys.stderr,
                                )
                            if chosen_params_out is not None:
                                chosen_params_out['combo'] = _combo_num_from_name('none')
                                chosen_params_out['combo_name'] = (
                                    f'none+pts{ADAPT_STARVE_PTS:g}')
                                chosen_params_out['combo_forced'] = False
                            return _spa, int(_spt)
                    _surv = {'none': _base}

                _pick = None; _mode = 'conservative'
                # Tier 2A — regime-gated tier2 WIN promotion (GLM-style free win).
                # Prefer the GENTLEST winshape survivor: fewest toggles (→ plain
                # 'tier2', the canonical validated win) then fewest sub-2.0
                # tensors. The most-aggressive variants (pos/class+tier2) can, at
                # tight budgets, push into iq1_m — recipes we have NO PPL for and
                # that don't load on every build — so they must not be preferred.
                if _allow_promote and _HBR < _HBR_THR:
                    _wc = []
                    for n, a in _surv.items():
                        _s = _shape(a, _floor)
                        if _s and _s['ATFLOOR'] and _s['SKEW'] > _SKEW_GATE and _s['flt25'] < _LT25_GATE:
                            _ntog = 1 + n.count('+') if n != 'none' else 0
                            _wc.append((_ntog, _s['flt2'], -_s['SKEW'], n))
                    if _wc:
                        _pick = sorted(_wc)[0][3]; _mode = 'win-promote'
                # Tier 2A.5 — opportunistic gated class+pos upgrade. class+pos is the
                # best conservative combo on several models (GLM-2.5107, Qwen122B,
                # Qwen3.6-27B at tight budgets) but a disaster on a few (Qwen0.8B,
                # DeepSeek) where it survives the vetoes. pred_damage can't tell them
                # apart (≈identical), but two per-class features cleanly do:
                #   * rel_crit — does it crush a critical tensor? (excludes Qwen0.8B)
                #   * the quant-pool floor (_floor) — class+pos only helps in the tight
                #     regime that actually uses sub-2-bit quants; high-bpw-pool models
                #     like DeepSeek have floor >= _FLOOR_ABS, so they are excluded.
                # Both are computed per class, so this holds for split (CPU/GPU) recipes
                # too. Prefer class+pos only when both gates pass; else fall through.
                # (Verified live: +4 net "beats greedy", 0 regression on Qwen0.8B/DeepSeek.)
                if (_pick is None and _allow_classpos and 'class+pos' in _surv
                        and _floor < _FLOOR_ABS):
                    _cp = _surv['class+pos']
                    _mi = 0.0
                    for t in _topK:
                        _qn = _base.get(t); _qc = _cp.get(t)
                        if not _qn or not _qc:
                            continue
                        _b1 = _qbpw(_qn); _b2 = _qbpw(_qc)
                        if _b1 is None or _b2 is None or _b2 >= _b1:
                            continue
                        _mi = max(_mi, _sens.get(t, 0.0) * (_deg(t, _qc) - _deg(t, _qn)))
                    _relc = (_mi / _pd_none) if _pd_none > 0 else 0.0
                    if _relc < _CP_RELCRIT:
                        _pick = 'class+pos'; _mode = 'conservative(class+pos)'
                # Tier 2B — conservative selector (default)
                if _pick is None:
                    _pds = {n: _pdam(a) for n, a in _surv.items()}
                    _mn = min(_pds.values())
                    _win = [n for n in _surv if _pds[n] <= _mn + _EPS_PD]
                    _pick = sorted(_win, key=lambda n: (-_swbpw(_surv[n]),
                                                        0 if n == 'none' else 1, n))[0]

                _pure_assignment, _pure_total = _combos[_pick]
                if debug:
                    print(
                        f"[AUTO] adaptive selector: HBR={_HBR:.3f} "
                        f"combos={sorted(_combos)} "
                        f"vetoed={ {n: v for n, v in _vlog.items() if v} } "
                        f"→ '{_pick}' ({_mode}; pred_damage={_pdam(_pure_assignment):.4g}; "
                        f"{_pure_total/GIB:.3f} GiB).",
                        file=sys.stderr,
                    )

                def _record_combo():
                    if chosen_params_out is not None:
                        chosen_params_out['combo'] = _combo_num_from_name(_pick)
                        chosen_params_out['combo_name'] = _pick
                        chosen_params_out['combo_forced'] = False
                if _mode == 'win-promote':
                    # GLM-style free win: adopt directly (bypass the meta-guard,
                    # as SP_FORCE did for validation). Safe: reachable only on
                    # greedy-win + HBR<5 + a veto-surviving clean floor-pusher.
                    _record_combo()
                    return _pure_assignment, int(_pure_total)
                # Conservative pick: keep the meta-guard so any cliff model that
                # DOES reach here (pure greedy worse than the constrained
                # candidate) is still protected — adopt only if it beats it.
                _pure_merged = _sp_fold(_pure_assignment)
                if _pure_merged is not None:
                    if _meta_score(_pure_merged)[1] < _meta_score(assignment)[1]:
                        _record_combo()
                        return _pure_assignment, int(_pure_total)
                    elif debug:
                        print(
                            "[AUTO] adaptive conservative pick scored WORSE than "
                            "constrained (primary); keeping constrained result.",
                            file=sys.stderr,
                        )
            # 'none' infeasible (or merged-fold failed) → fall through (keep constrained).

        else:
            # -------- ADAPT_ENABLED is False: faithful pure-greedy 'none' fallback ----
            # No adaptive selection — re-run pure greedy and adopt it only if it beats
            # the constrained candidate on the meta primary (the pre-adaptive behaviour).
            _sp_ppl, _sp_quants = _sp_build(set())
            _pure_assignment, _pure_total = _sp_greedy(_sp_ppl, _sp_quants)
            if (_pure_assignment is not None and _pure_total is not None
                    and int(_pure_total) <= _feas_cap):
                _pure_merged = _sp_fold(_pure_assignment)
                if (_pure_merged is not None
                        and _meta_score(_pure_merged)[1] < _meta_score(assignment)[1]):
                    if debug:
                        print(f"[AUTO] adaptive disabled → pure greedy adopted. "
                              f"total {total_size/GIB:.3f} → {_pure_total/GIB:.3f} GiB.",
                              file=sys.stderr)
                    return _pure_assignment, int(_pure_total)

    # --- Step 9: Phase C — promote individual sortable tensors with headroom,
    #     respecting rank monotonicity. Smallest current-size tensors go first
    #     because each promotion costs less, so we get more quality per byte.
    # Build (and continually update) the per-tensor "ceiling" — the minimum
    # degradation observed among MORE-sensitive tensors. A promotion that would
    # make this tensor strictly better than a more-sensitive tensor is rejected.
    sortable_sens_desc = sorted(sortable, key=lambda t: -ppl_loss_w[t])
    sens_pos: Dict[str, int] = {t: i for i, t in enumerate(sortable_sens_desc)}

    def ceiling_for(t: str) -> float:
        """Minimum degradation among tensors strictly more sensitive than t."""
        i = sens_pos[t]
        if i == 0:
            return float('-inf')  # no upper bound for the most sensitive tensor
        best = float('inf')
        for k in range(0, i):
            tt = sortable_sens_desc[k]
            try:
                d = degradation_fn(tt, assignment[tt])
            except Exception:
                d = None
            if d is not None and d < best:
                best = d
        return best

    # Budget for phase C — strict: never exceed budget_bytes. (`tolerance` is
    # only a hint that small over/undershoot is acceptable for OTHER methods;
    # the auto method already finds the best window within budget via the
    # brute-force search, so Phase C should only fill the leftover headroom
    # rather than spend any of the user's tolerance allowance.)
    headroom_budget = budget_bytes

    if debug:
        print(
            f"[AUTO] Phase C: starting promotions "
            f"(headroom = {(headroom_budget - total_size)/GIB:.3f} GiB up to "
            f"{headroom_budget/GIB:.3f} GiB)",
            file=sys.stderr,
        )

    # Sort sortable by current size ascending (smallest first).
    promotions = 0
    while True:
        # Re-sort each pass; a tensor's "current size" changes after promotion.
        order = sorted(sortable, key=lambda t: tensor_sizes_w[t][assignment[t]])
        any_promoted = False
        for t in order:
            # Skip outliers — they've been pinned to a specific qtype by
            # the outlier-pinning logic (target cap + monotonicity +
            # Pareto). Letting Phase C "promote" them (even via free
            # Pareto upgrades with delta ≤ 0) would silently undo the cap
            # and pull the outlier above its intended qtype.
            if t in high_outliers_set:
                continue
            current_q = assignment[t]
            allowed_t = allowed_map_w[t]
            try:
                cur_idx = allowed_t.index(current_q)
            except ValueError:
                continue
            if cur_idx == 0:
                continue  # already at best allowed qtype
            ceil_d = ceiling_for(t)
            # Try the SMALLEST upgrade first (one index up). Repeated passes will
            # naturally explore larger upgrades when the budget permits.
            new_q = allowed_t[cur_idx - 1]
            try:
                new_deg = degradation_fn(t, new_q)
            except Exception:
                new_deg = None
            if new_deg is not None and ceil_d != float('-inf') and new_deg < ceil_d:
                # would strictly outrank a more-sensitive tensor — reject
                continue
            delta = int(tensor_sizes_w[t][new_q]) - int(tensor_sizes_w[t][current_q])
            if delta <= 0:
                # No size gain — apply if it doesn't break rank; harmless
                assignment[t] = new_q
                total_size += delta
                any_promoted = True
                promotions += 1
                continue
            if total_size + delta > headroom_budget:
                continue
            assignment[t] = new_q
            total_size += delta
            any_promoted = True
            promotions += 1
            if debug:
                print(
                    f"[AUTO] Phase C: promoted {t} {current_q} -> {new_q} "
                    f"(+{delta/GIB:.4f} GiB; total={total_size/GIB:.3f} GiB)",
                    file=sys.stderr,
                )
        if not any_promoted:
            break

    if debug:
        print(
            f"[AUTO] Phase C: {promotions} promotions performed. "
            f"Final total_size = {total_size/GIB:.3f} GiB",
            file=sys.stderr,
        )

    # --- Step 10: Expand harmonized groups back to original tensor names
    if merged and group_defs:
        expanded: Dict[str, str] = {}
        for gid, members in group_defs.items():
            q = assignment.get(gid)
            if q is None:
                continue
            for m in members:
                expanded[m] = q
        for t in tensors_w:
            if t in group_defs:
                continue
            if t.startswith("HARM_GROUP_"):
                continue
            val = assignment.get(t)
            if val is None:
                continue
            expanded[t] = val
        assignment = expanded

    return assignment, total_size



def optimize_midpoint_and_assign(quants, _, class_values,
                                 max_bytes, tolerance=0.05, exp_factor=1.0, harmonize_groups=None):
    """
    Loop over stretch factors and perform midpoint optimization using class mean with dichotomy.
    exp_factor controls exponent in stretch calculation: higher = more aggressive extremes.
    """
    if INFO:
        print(f"[Info] Starting optimization for target size {max_bytes} bytes ±{tolerance*100}% with exp_factor={exp_factor:.2f}...", file=sys.stderr)
    best_assign, best_size = {}, float('inf')
    # compute initial midpoint as class mean
    class_mid = compute_class_midpoint(class_values)
    # outer loop: stretch factor sweep
    stretch = STRETCH_MIN
    while stretch <= STRETCH_MAX:
        if INFO and stretch > STRETCH_MIN:
            print(f"[Info] Trying stretch factor {stretch:.2f}...", file=sys.stderr)
        # reset bisection bounds for each stretch
        low_val, high_val = min(class_values.values()), max(class_values.values())
        # compute exponential boundary modifier
        exponential_factor = (STRETCH_MAX/stretch) ** exp_factor
        low_val *= exponential_factor
        high_val *= exponential_factor
        # start midpoint clamped to [low_val, high_val]
        mid = max(low_val, min(high_val, class_mid))
        prev_mid = None
        change = None
        change_min_threshold = 0.0001
        mid_min_threshold = 0.00001
        if INFO:
            print(f"[Info] Progress: {stretch/STRETCH_MAX*100:.2f}%", file=sys.stderr)
        # inner loop: dichotomy until converged
        while (prev_mid == None or prev_mid > mid_min_threshold) and (change == None or change >= change_min_threshold):
            if INFO:
                print(f"[Info] Evaluating midpoint={mid:.6f}, stretch={stretch:.2f}...", file=sys.stderr)
            assignment, sizes = assign_quants(quants, None,
                                             class_values,
                                             forced_mid=mid, stretch=stretch, harmonize_groups=harmonize_groups)
            size = sum(sizes.values())
            # tolerance check
            if abs(size - max_bytes) / max_bytes <= tolerance:
                if INFO:
                    print(f"[Info] Found acceptable size {size} at midpoint={mid:.6f}, stretch={stretch:.2f}.", file=sys.stderr)
                return assignment, size
            # check midpoint change
            if prev_mid is not None:
                change = abs(mid - prev_mid) / prev_mid
                if change < change_min_threshold:  # less than 0.01%
                    if INFO:
                        print(f"[Info] Midpoint change {change*100:.4f}% below threshold; breaking inner loop.", file=sys.stderr)
                    break
            prev_mid = mid
            # decide direction and update bounds
            if size < max_bytes:
                high_val = mid
            else:
                low_val = mid
            if INFO:
                reason = 'too small' if size < max_bytes else 'too large'
                direction = 'down' if size < max_bytes else 'up'
                print(f"[Info] Size {size} is {reason}; moving midpoint {direction}.", file=sys.stderr)
            # compute next midpoint by dichotomy
            mid = (low_val + high_val) / 2
            # track best
            if abs(size - max_bytes) < abs(best_size - max_bytes):
                best_size, best_assign = size, assignment.copy()
        # increment stretch factor
        stretch = round(stretch + STRETCH_STEP, 2)
    if INFO:
        print("[Warning] Optimization finished; using best found assignment.", file=sys.stderr)
    return best_assign, best_size

def scale_for_size(assignment, sizes, quants, max_size_bytes):
    """
    Fallback simple scaling if optimized assignment not used.
    """
    total = sum(sizes.values())
    if INFO: print(f"[Info] Starting fallback scaling: current total {total}, target {max_size_bytes}", file=sys.stderr)
    if total <= max_size_bytes:
        return assignment, total
    items = list(assignment.items())
    while total > max_size_bytes:
        made_change = False
        for name, q in items:
            idx = quants.index(q)
            if idx + 1 < len(quants):
                new_q = quants[idx+1]
                assignment[name] = new_q
                sizes[name], _, _ = get_map_sizes_and_elements(new_q)
                sizes[name] = sizes[name].get(name, 0)
                made_change = True
                total = sum(sizes.values())
                if INFO: print(f"[Info] Scaling {name} from {q} to {new_q}, new total {total}", file=sys.stderr)
                if total <= max_size_bytes:
                    return assignment, total
        if not made_change:
            if INFO: print("[Warning] Cannot reduce size further via fallback scaling.", file=sys.stderr)
            break
    return assignment, total

def _convert_value(v):
    """
    Convert a CSV cell value v to float, handling percentage strings.
    """
    if isinstance(v, str) and v.endswith('%'):
        try:
            return float(v.rstrip('%')) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

def assign_qtype(default_qtype, regex_assign_list, quants, names):
    """
    Build a dict mapping each tensor in `names` to a QTYPE.
    - If regex_assign_list is non-empty, scan in order, first match wins.
    - Otherwise fall back to default_qtype (or highest-bpw if default_qtype is None).
    """
    def _bpw_for_tensor(q, tensor_name):
        try:
            bpw = get_bpw(q, tensor_name=tensor_name)
            if bpw is None or not math.isfinite(float(bpw)):
                raise ValueError
            return float(bpw)
        except Exception:
            try:
                bpw = get_bpw(q)
                return float(bpw)
            except Exception:
                return float('-inf')

    out = {}
    for name in names:
        if default_qtype:
            base_q = default_qtype
        else:
            # Since we know the tensor name here, pick the best qtype for this specific tensor.
            base_q = max(quants, key=lambda q: (_bpw_for_tensor(q, name), q))

        assigned = None
        # Try regex overrides first
        for pat, qt in regex_assign_list:
            if pat.fullmatch(name):
                assigned = qt
                break
        if assigned is None:
            assigned = base_q
        out[name] = assigned
    return out

# Module-level cache for public keys
PUB_KEYS = []
def load_public_keys():
    """
    Load and cache all PGP public keys from the ASCII-armored keyring.
    Raises if keyring is missing or invalid.
    """
    global PUB_KEYS
    if PUB_KEYS:
        return PUB_KEYS

    if not os.path.isfile(KEYRING_PATH):
        raise FileNotFoundError(f"Keyring not found: {KEYRING_PATH!r}")

    blob = open(KEYRING_PATH, 'r').read()
    # Find all ASCII-armored public key blocks
    pattern = re.compile(
        r"-----BEGIN PGP PUBLIC KEY BLOCK-----(?:.|\n)*?-----END PGP PUBLIC KEY BLOCK-----",
        re.DOTALL
    )
    matches = pattern.findall(blob)

    if not matches:
        raise ValueError(f"[Error] No PGP public keys found in {KEYRING_PATH!r}")

    for block in matches:
        try:
            key = pgpy.PGPKey.from_blob(block) # type: ignore
            # pgpy.PGPKey.from_blob may return key or (key, _), handle both
            if isinstance(key, tuple):
                key = key[0]
            PUB_KEYS.append(key)
        except Exception as e:
            # skip invalid blocks
            continue

    if not PUB_KEYS:
        raise ValueError(f"[Error] Failed to parse any public keys from {KEYRING_PATH!r}")

    return PUB_KEYS


def verify_detached_signature(file_path):
    """
    Verify that file_path + ".sig" is a valid detached signature
    by one of the loaded public keys.
    Returns True on success, False on failure.
    """
    if not PUB_KEYS:
        load_public_keys()

    sig_path = file_path + ".sig"
    if not os.path.isfile(sig_path):
        raise FileNotFoundError(f"[Error] Signature file not found: {sig_path!r}")

    # Read signature in binary mode to handle both ASCII and binary sigs
    with open(sig_path, 'rb') as f:
        sig_blob = f.read()
    try:
        signature = pgpy.PGPSignature.from_blob(sig_blob) # type: ignore
    except Exception as e:
        print(f"[Error] Error parsing signature: {e}", file=sys.stderr)
        return False

    # Read the signed data as binary
    with open(file_path, 'rb') as f:
        data = f.read()

    for key in PUB_KEYS:
        try:
            if key.verify(data, signature):
                return True
        except Exception:
            continue

    return False

def harmonize_row(row: pd.Series, cols: list, harmonize_groups: list, technique: int = 1) -> pd.Series:
    """
    Harmonize values inside `row` according to harmonize_groups.
    Pairing rules:
      - For each inner group, collect matches for each regex.
      - All match-lists must have the same length (or ValueError).
      - Extract layer id via 'blk.<ID>' (preferred), fallback to first numeric token.
        * If every matched name across all lists has an ID -> sort each list by ID and pair index-wise.
        * If none have IDs -> sort by name deterministically and pair index-wise.
        * If mixed (some lists/entries have IDs and some don't) -> raise ValueError.
      - For each paired tuple, ensure IDs match (or None in no-ID case) then harmonize the numeric values
        (technique 1=max, 2=mean, 3=min) and write the number back into `row` for all paired names.
    """
    if not harmonize_groups:
        return row

    if not isinstance(harmonize_groups, list):
        raise ValueError("--harmonize-tensors must be a list-of-lists (or '[]' to disable).")

    # validate shape
    for g in harmonize_groups:
        if not isinstance(g, list) or len(g) < 2:
            raise ValueError("Each inner group in --harmonize-tensors must be a list containing at least 2 regex strings.")
        for p in g:
            if not isinstance(p, str):
                raise ValueError("Each pattern in --harmonize-tensors must be a string (regex).")

    for gi, group in enumerate(harmonize_groups):
        compiled = [re.compile(p) for p in group]

        # collect matches for each pattern
        matches_per_pattern = [[name for name in cols if compiled[i].search(name)] for i in range(len(compiled))]

        # ensure same length across patterns
        lengths = [len(l) for l in matches_per_pattern]
        if len(set(lengths)) != 1:
            raise ValueError(f"Harmonize group {gi}: matched lists have different lengths: {lengths}. All patterns must match the same number of tensors.")

        n = lengths[0]
        if n == 0:
            # nothing to do for this group
            continue

        # build (name, id_or_none) lists
        def extract_id(name: str):
            m = re.search(r"blk\.(\d+)", name)
            if m:
                return int(m.group(1))
            m2 = re.search(r"(\d+)", name)
            if m2:
                return int(m2.group(1))
            return None

        lists_with_ids = []
        for lst in matches_per_pattern:
            lists_with_ids.append([(name, extract_id(name)) for name in lst])

        # decide sorting strategy:
        # - if all entries across all lists have non-None id -> sort by id
        # - elif all entries across all lists have None id -> sort by name
        # - else -> ambiguous -> raise
        all_ids = [id for lst in lists_with_ids for (_, id) in lst]
        any_id = any(id is not None for id in all_ids)
        all_have_id = all(id is not None for id in all_ids)

        if all_have_id:
            # sort each list by id (convert None to -1 for type-safety; here all_have_id so None shouldn't appear)
            for i in range(len(lists_with_ids)):
                lists_with_ids[i].sort(key=lambda t: (-1 if t[1] is None else t[1]))
        elif not any_id:
            # no ids anywhere -> deterministic sort by name
            for i in range(len(lists_with_ids)):
                lists_with_ids[i].sort(key=lambda t: t[0])
        else:
            raise ValueError(f"Harmonize group {gi}: inconsistent layer id presence across matched tensors (some have IDs, some don't).")

        # now pair index-wise and verify IDs match
        for idx in range(n):
            tuple_names = []
            tuple_ids = []
            for lst in lists_with_ids:
                name, lid = lst[idx]
                tuple_names.append(name)
                tuple_ids.append(lid)

            # check ids: either all equal (and not None), or all None
            if all(x is None for x in tuple_ids):
                # ok (no ids case)
                pass
            else:
                # require all equal and not None
                if not all((x == tuple_ids[0]) and (x is not None) for x in tuple_ids):
                    raise ValueError(f"Harmonize group {gi} index {idx}: mismatched layer ids for pair {tuple_names} -> ids {tuple_ids}")

            # collect numeric values for each name
            numeric_vals = []
            for nm in tuple_names:
                raw = row.get(nm, np.nan)
                try:
                    val = _convert_value(raw)
                except Exception:
                    try:
                        val = float(raw)
                    except Exception:
                        val = float("nan")
                numeric_vals.append(val)

            valid_vals = [v for v in numeric_vals if not np.isnan(v)]
            if not valid_vals:
                # nothing numeric to harmonize
                continue

            if technique == 1:
                new_val = max(valid_vals)
            elif technique == 2:
                new_val = float(sum(valid_vals)) / len(valid_vals)
            elif technique == 3:
                new_val = min(valid_vals)
            else:
                raise ValueError("--harmonization-technique must be 1, 2 or 3")

            # write back
            for nm in tuple_names:
                row.at[nm] = new_val

            if INFO:
                lid_str = tuple_ids[0] if tuple_ids and tuple_ids[0] is not None else "no-id"
                print(f"[Info] Harmonized group {gi} layer {lid_str}: {tuple_names} -> {new_val}", file=sys.stderr)

    return row


def expand_harmonize_groups(harmonize_groups: List[List[str]], tensors: List[str]) -> List[List[str]]:
    """
    Expand list-of-regex-groups into concrete per-layer lists of tensor names.

    - harmonize_groups: e.g. [["blk\\..*\\.ffn_up_exps.*","blk\\..*\\.ffn_gate_exps.*"]]
    - tensors: list of available tensor names to match against (class-specific)

    Returns a flattened list-of-lists where each inner list contains the concrete
    tensor names paired index-wise per layer. If a group cannot be safely
    expanded (mismatched counts, inconsistent IDs), the group is skipped with
    an info message and not included in the returned list.
    """
    out: List[List[str]] = []
    if not harmonize_groups:
        return out

    for gi, group in enumerate(harmonize_groups):
        # compile patterns safely
        try:
            compiled = [re.compile(p) for p in group]
        except Exception:
            if INFO:
                print(f"[Info] Skipping harmonize group {gi}: invalid regex in {group}", file=sys.stderr)
            continue

        # collect matches for each pattern (use re.search semantics)
        matches_per_pattern = []
        for cre in compiled:
            matched = [t for t in tensors if cre.search(t)]
            # remove duplicates while preserving order
            matched = list(dict.fromkeys(matched))
            matches_per_pattern.append(matched)

        lengths = [len(l) for l in matches_per_pattern]
        if len(set(lengths)) != 1:
            if INFO:
                print(f"[Info] Skipping harmonize group {gi}: pattern match counts differ {lengths}", file=sys.stderr)
            continue

        n = lengths[0]
        if n == 0:
            # nothing matched for this group
            continue

        # extract numeric id helper
        def extract_id(name: str):
            m = re.search(r"blk\.(\d+)", name)
            if m:
                return int(m.group(1))
            m2 = re.search(r"(\d+)", name)
            return int(m2.group(1)) if m2 else None

        lists_with_ids = [[(name, extract_id(name)) for name in lst] for lst in matches_per_pattern]

        all_ids = [iid for lst in lists_with_ids for (_, iid) in lst]
        any_id = any(i is not None for i in all_ids)
        all_have_id = all(i is not None for i in all_ids)

        if all_have_id:
            for l in lists_with_ids:
                # convert None to -1 if present; primarily for type-safety (shouldn't be None when all_have_id)
                l.sort(key=lambda x: (-1 if x[1] is None else x[1]))
        elif not any_id:
            for l in lists_with_ids:
                l.sort(key=lambda x: x[0])
        else:
            if INFO:
                print(f"[Info] Skipping harmonize group {gi}: inconsistent id presence across matches", file=sys.stderr)
            continue

        # pair index-wise and append concrete tuples
        for i in range(n):
            pair = [lists_with_ids[j][i][0] for j in range(len(lists_with_ids))]
            out.append(pair)

    return out

def parse_group_argument(arg_value, arg_name: str, parser, info_flag=False):
    """
    Normalize an argument like --harmonize-tensors or --synergistic-tensors
    into a list-of-lists of strings.

    Supports:
      - Python literal strings (e.g. "[['p1','p2'],['p3','p4']]")
      - List of comma-separated strings (via nargs='+')
      - List-of-lists directly
      - Single-element list containing a literal string

    Returns:
        list[list[str]] (empty list if disabled)
    """
    groups = []

    if arg_value and arg_value == ['']:
        if info_flag:
            print(f"[Info] {arg_name} disabled by the user", file=sys.stderr)
        return groups

    # Case 1: direct Python literal string
    if isinstance(arg_value, str):
        try:
            parsed = ast.literal_eval(arg_value)
            if not isinstance(parsed, list):
                raise ValueError("not a list")
            groups = parsed
        except Exception:
            parser.error(
                f"Invalid {arg_name}: must be a Python literal list-of-lists, e.g. [['pat1','pat2'], ['p3','p4']]."
            )

    # Case 2: list form (from nargs='+')
    elif isinstance(arg_value, list):
        # Single-element list containing a literal string
        if len(arg_value) == 1 and isinstance(arg_value[0], str) and arg_value[0].strip().startswith('['):
            try:
                parsed = ast.literal_eval(arg_value[0])
                if not isinstance(parsed, list):
                    raise ValueError("not a list")
                groups = parsed
            except Exception:
                parser.error(
                    f"Invalid {arg_name}: must be a Python literal list-of-lists, e.g. [['pat1','pat2'], ['p3','p4']]."
                )

        # List form directly, possibly list-of-lists
        elif all(isinstance(elem, list) for elem in arg_value):
            groups = arg_value

        # List of comma-separated strings (e.g. 'pat1,pat2')
        else:
            for elem in arg_value:
                if isinstance(elem, str):
                    parts = [p for p in re.split(r'\s*,\s*', elem.strip()) if p != '']
                    if parts:
                        groups.append(parts)
                else:
                    parser.error(
                        f"Invalid {arg_name} element: expected string or list"
                    )
    else:
        parser.error(f"Invalid {arg_name}: expected string or list")

    return groups


def build_recipe_bpw_stats(tensor_index):
    """
    Aggregate actual/base bytes and bpw per qtype from the recipe tensors actually produced.
    Returns:
      {
        qtype: {
          'count': int,
          'bytes': int,
          'base_bytes': int,
          'elements': int,
          'bpw_actual': float,
          'bpw_base': float,
          'dynamic': bool,
        }
      }
    """
    stats = {}

    for cls in ('cpu', 'gpu'):
        for e in tensor_index.get(cls, []):
            q = e.get('final_q')
            if not q:
                continue

            q = str(q)
            s = stats.setdefault(q, {
                'count': 0,
                'bytes': 0,
                'base_bytes': 0,
                'elements': 0,
                'dynamic': False,
            })

            s['count'] += 1
            s['bytes'] += int(e.get('size', 0) or 0)
            s['base_bytes'] += int(e.get('base_size', e.get('size', 0)) or 0)
            s['elements'] += int(e.get('elements', 0) or 0)

            if abs(float(e.get('bpw_actual', 0.0)) - float(e.get('bpw_base', 0.0))) > 1e-9:
                s['dynamic'] = True

    for q, s in stats.items():
        elems = s['elements']
        if elems > 0:
            s['bpw_actual'] = (s['bytes'] * 8.0) / elems
            s['bpw_base'] = (s['base_bytes'] * 8.0) / elems
        else:
            s['bpw_actual'] = 0.0
            s['bpw_base'] = 0.0

    return stats


def record_tensor_index_entry(tensor_index, cls, name, orig_q, final_q, size, elements, kind):
    """
    Record a tensor entry with both actual and base bpw/size values.
    """
    q_for_bpw = final_q if final_q else orig_q
    if not q_for_bpw:
        q_for_bpw = 'f32'

    actual_bpw = get_bpw(q_for_bpw, tensor_name=name)
    base_bpw = get_bpw(q_for_bpw, tensor_name=name, use_base=True)

    if actual_bpw is None or not math.isfinite(float(actual_bpw)):
        actual_bpw = get_bpw(q_for_bpw)
    if base_bpw is None or not math.isfinite(float(base_bpw)):
        base_bpw = get_bpw(q_for_bpw, use_base=True)

    base_size = int(round((elements * float(base_bpw)) / 8.0)) if elements else int(size)

    tensor_index[cls].append({
        'name': name,
        'orig_q': orig_q,
        'final_q': final_q,
        'size': int(size),
        'base_size': int(base_size),
        'elements': int(elements),
        'bpw_actual': float(actual_bpw),
        'bpw_base': float(base_bpw),
        'kind': kind,
    })


def main():
    global DEBUG, INFO, SKIP_GPG, ALL_GPG_SIGS_VALID, NO_FALLBACK
    global COMPUTE_MISSING_MAP, COMPUTE_ALL_MAP
    global CONVERT_IGNORE_IMATRIX_RULES, CONVERT_WITH_IMATRIX, CONVERT_FALLBACK_QUANTS, CONVERT_FALLBACK_QUANTS_FORBIDDEN

    parser = argparse.ArgumentParser(description="Assign optimal quants per tensor based on the calibration data CSV file.")
    parser.add_argument('--debug', action='store_true', help='Show debug logs')
    parser.add_argument('--info', action='store_true', help='Show info logs')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Relative GiB tolerance for size optimization. NOTE: ignored when --use-auto-quant-assign is set — the auto method always aims for the exact target size (with --info, a note is printed to stderr if --tolerance is passed alongside --use-auto-quant-assign).')
    parser.add_argument('--cpu-irq-k', type=float, default=1.5,
                        help='IQR multiplier k for CPU-friendly outlier detection')
    parser.add_argument('--gpu-irq-k', type=float, default=1.5,
                        help='IQR multiplier k for GPU-friendly outlier detection')
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('--qtype', help='Case-sensitive qtype (e.g. q3_K) to analyze from the calibration data CSV file (default: lowest quant, preferring QTYPEs that do NOT end with "_bn")')
    parser.add_argument('--cpu-assign-qtype', help='Case-sensitive qtype (e.g. q6_K) to assign to non-measured CPU-friendly tensors or tensors missing from csv (default: highest quant)')
    parser.add_argument('--gpu-assign-qtype', help='Case-sensitive qtype (e.g. q3_K) to assign to non-measured GPU-friendly tensors or tensors missing from csv (default: highest quant)')
    parser.add_argument('--cpu-assign-tensors', nargs='+', default=[], help="List of regex=qtype (case-sensitive, e.g. q6_K) patterns for CPU-friendly tensors to force-assign")
    parser.add_argument('--gpu-assign-tensors', nargs='+', default=[], help="List of regex=qtype (case-sensitive, e.g. q3_K) patterns for GPU-friendly tensors to force-assign")
    #parser.add_argument('--sample-ppl', help='CSV sample PPL file path', required=True)
    parser.add_argument('--cpu-tensors', nargs='+', default=[], help='Regex patterns for CPU-friendly tensors')
    parser.add_argument('--gpu-tensors', nargs='+', default=[], help='Regex patterns for GPU-friendly tensors')
    parser.add_argument('--cpu-quants', nargs='+', help='Ordered list of CPU-friendly case-sensitive quants (e.g. q6_K)')
    parser.add_argument('--gpu-quants', nargs='+', help='Ordered list of GPU-friendly case-sensitive quants (e.g. q3_K)')
    parser.add_argument('--cpu-tensors-max-size', type=str, help='Max CPU-friendly tensors size in GiB or percent (e.g., 80%%)')
    parser.add_argument('--gpu-tensors-max-size', type=str, help='Max GPU-friendly tensors size in GiB or percent (e.g., 80%%)')
    parser.add_argument('--exponential-factor', type=float, default=None,
                        help=('Exponent controlling midpoint adjustment aggressiveness during stretch sweeps for default quant assignment method. '
                              'Higher values push quantization toward extremes. When not using --use-greedy-quant-assign the default value is 8. When using --use-greedy-quant-assign with --quant-degradation-csv (using the model\'s group0/kld_results.csv for example), the script will use a default exponential factor value of 1 as the curvature of the degradation data is expected to be an exact match with the model\'s quant optimum distribution. When using --use-greedy-quant-assign alone, the script will compute an approximate value using the equation: y = 0.5 * ln(x), '
                              'which aims to re-shape the provided (or default) quant degradation data to approximate a theoretical degradation data (doesn\'t always work well and requires to be manually tweaked) - x is the bf16 total tensor size in GiB, and use that as the exponential-factor. If computation fails, it will fallback to 3.0 as default value. '
                              'If you do provide --exponential-factor it overrides the automatic calculation. Recommended manual range for greedy quant assign is 0.3 to 5.0 when using KLD metrics with the default value seen in recipe hidden parameters section when --exponential-factor isn\'t used being a good starting point.'))
    parser.add_argument('--ignore-f32', action='store_true', help='Ignore f32 tensors (default: not ignored)')
    parser.add_argument('--tensors-from-csv', action='store_true', help='Obtains list of tensors from csv file only (default: tensors are obtained from map file)')
    parser.add_argument('--skip-gpg', action='store_true',
                        help='Skip gpg signature validation')
    parser.add_argument('--harmonize-tensors', nargs='+', default=[["blk\\..*\\.ffn_up_exps.*","blk\\..*\\.ffn_gate_exps.*"]],
                        help=('A Python literal list-of-lists of regex patterns. Each inner list declares a group of regexes whose matching tensors will be qtype harmonized **per layer**. ' 
                            "Example: --harmonize-tensors blk\\..\\*\\.ffn_up_exps.\\*,blk\\..\\*\\.ffn_gate_exps.\\* ... 'another_pat1,another_pat2'. "
                            "Use --harmonize-tensors \"\" to disable harmonization. Or use --harmonization-technique 0. "
                            "Note: harmonizing tensors to allow for fused ffn_up_exps and ffn_gate_exps can improve PP and TG speed, at the cost of slim restrictive dynamic quantization flexibility. "
                            "It is highly recommended to leave this parameter value default when using ik_llama.cpp for significant speed improvements (can be as high as +20%% speed gain) with MoE models - when using ik_llama.cpp with fmoe these tensors are fused, which can only happen if they are of the same qtype. " 
                            "Future versions of ik_llama.cpp may also take advantage of fused ffn_up_shexp and ffn_gate_shexp tensors."))
    parser.add_argument('--harmonization-technique', type=int, default=3, choices=[0,1,2,3],
                        help=('Harmonization technique to use when --harmonize-tensors is set: 0=disabled, 1=max, 2=mean, 3=min (default). ' 
                            'Values are applied element-wise per layer across the matched tensors.'
                            'Max ensures calibration data measurement is not negatively degraded. Min will degrade calibration data accuracy but appears to give the best results. Mean is a compromise in-between. Disabled means harmonization is disabled.'))
    parser.add_argument('--use-greedy-quant-assign', action='store_true', help='Use greedy priority-queue quant assignment instead of default spread/midpoint method. The method tries to minimize overall degradation by prioritizing quant downgrades that yield the least degradation per byte saved. This method requires per-tensor degradation data (e.g. KLD) to be present in the csv_file - perplexity data only works suboptimally. It also requires per quant type degradation estimates which can be supplied either via --quant-degradation-csv - if not present hardcoded Qwen3-4B-Thinking-2507 degradation values are used. override with --quant-degradation-csv. It is recommended to use --exponential-factor between 1.0 and 5.0 when using this method to try to map per-tensor degradation values into a more linear space.')
    parser.add_argument('--use-auto-quant-assign', action='store_true',
                        help=('Use the data-adaptive "auto" quant assignment instead of the greedy or default spread/midpoint methods. '
                              'Everything is tuned from the data — no need to hand-pick --exponential-factor, --per-tensor-degradation-scaling, or a window of qtypes. The auto method: '
                              '(1) detects zero-kld outliers and pins them at the smallest qtype; '
                              '(2) detects high-sensitivity outliers (e.g. token_embd) via IQR and pins each to a budget-progressive Pareto target derived from max_loss and the budget-natural qtype, so the most-sensitive tensors walk smoothly through the Pareto frontier as the budget grows; '
                              '(3) enumerates every contiguous sub-window of the qtype pool and rank-maps tensors uniformly across each window; '
                              '(4) adds constrained-greedy candidates that exclude the worst-deg qtypes one tier at a time, letting the meta pick the optimal "cap k" from the data; '
                              '(5) auto-tunes the score exponents (p, q) over a fine grid; '
                              '(6) ranks all candidates with a data-tuned meta Σ (loss + mean_loss) · deg^p_meta where p_meta = log2(max_pool_deg / max_loss) auto-adjusts cliff-aware penalty steepness; '
                              '(7) runs a final promotion pass (Phase C) on non-outlier tensors to consume leftover headroom. '
                              'Requires --quant-degradation-csv (strongly recommended) or falls back to hardcoded defaults.'))
    parser.add_argument('--auto-no-pareto-filter', action='store_true',
                        help=('Only valid with --use-auto-quant-assign. By default the auto method drops per-tensor allowed qtypes that are Pareto-dominated on the (size, degradation) plane '
                              '— a qtype that is BOTH larger AND more-degrading than another available qtype is never preferable, so filtering it eliminates wasted budget. '
                              'Pass this flag to disable Pareto filtering (e.g. if you deliberately want to use qtypes that look "objectively worse" by group0 stats but behave better at inference time on your specific hardware).'))
    parser.add_argument('--auto-force-combo', type=int, default=None, metavar='N',
                        help=('Only valid with --use-auto-quant-assign. Troubleshooting only. When the auto method '
                              'selects greedy, it runs a second pass that enumerates alteration combos and ADAPTIVELY '
                              'picks the best safe one. Pass --auto-force-combo N to instead FORCE a specific combo and '
                              'DISABLE adaptive selection. N is a bitmask over the toggles {class=1, pos=2, tier2=4, '
                              'pareto=8}. The combinations are: '
                              '0=none (pure greedy), 1=class, 2=pos, 3=class+pos, 4=tier2, 5=class+tier2, '
                              '6=pos+tier2, 7=class+pos+tier2, '
                              '8=pareto, 9=pareto+class, 10=pareto+pos, 11=pareto+class+pos, 12=pareto+tier2, '
                              '13=pareto+class+tier2, 14=pareto+pos+tier2, 15=pareto+class+pos+tier2. '
                              'NOTE: pareto is INERT (it never changes the recipe), so combos 8-15 are identical to '
                              '0-7 respectively — in practice use 0-7. '
                              'Toggle meanings: "class" scales bulk per-tensor losses by their per-class factor; '
                              '"pos" boosts first/last edge-layer tensors; "tier2" demotes tier-2 outlier tensors to '
                              'the floor (the GLM-style "free win" when the calibration data over-rates them); '
                              '"pareto" prunes Pareto-dominated qtypes. '
                              'Only affects the greedy second pass (i.e. when greedy wins the auto sweep). '
                              'When NOT set, the adaptive selector picks automatically and discloses the chosen combo '
                              'number per class in the recipe\'s hidden-parameters footer.'))
    parser.add_argument('--auto-deg-exponent', type=float, default=None,
                        help=('Only valid with --use-auto-quant-assign. Pins the degradation exponent q in the auto method\'s candidate-selection score Σ loss^p · deg^q. '
                              'q > 1 amplifies the badness of high-degradation qtypes (e.g. iq1_s) and pushes the recipe to avoid using them; q = 1 is the linear regime. '
                              'When neither --exponential-factor (p) nor --auto-deg-exponent (q) are provided, the auto method sweeps a fine (p, q) grid internally and picks the pair whose chosen assignment minimises the meta-score — and reports the selection in the recipe\'s hidden-parameters footer. '
                              'Override only if you know what you\'re doing.'))
    parser.add_argument('--quant-degradation-csv', type=str, help='Path to CSV file containing quant degradation values for use by greedy quant assign method (optional). If not provided, hardcoded Qwen3-4B-Thinking-2507 degradation values are used, and you will need to tweak --exponential-factor. ')
    parser.add_argument('--quant-degradation-equation', type=str,
                        help=('Deprecated option; no longer used in this script. Use group0_enricher.py instead to fill the gaps of missing group0/kld_results_partial.csv degradation data.'))
    parser.add_argument('--per-tensor-degradation-scaling', type=float, default=None,
                        help='Exponent for scaling group degradation values per tensor based on its loss relative to the mean. Only valid when greedy quant assign method is used. '
                            '0 = disabled. Higher values protect highly sensitive tensors more strongly. '
                            'Recommended range 0.0 - 0.5. Default (disabled when parameter isn\'t set): 0.0')
    parser.add_argument('--synergistic-tensors', nargs='+', default=[["blk\\..*\\.ffn_up_exps.*","blk\\..*\\.ffn_gate_exps.*","blk\\..*\\.ffn_down_exps.*"]],
                        help=('A Python literal list-of-lists of regex patterns. Each inner list defines tensors that '
                            'exhibit synergistic effects and should have their loss adjusted together. '
                            'Example: --synergistic-tensors blk\\..\\*\\.ffn_up_exps.\\*,blk\\..\\*\\.ffn_gate_exps.\\*,blk\\..\\*\\.ffn_down_exps.\\* '
                            "'another_pat1,another_pat2'. "
                            'Use --synergistic-tensors "" to disable synergy adjustment. '
                            'Note: synergy encourages similar quantization within each layer, '
                            'typically improving quality without strictly enforcing identical qtypes.'))
    parser.add_argument('--synergy-strength',type=float, default=0.0,
                        help='Strength of synergy-based loss adjustment (0 = disabled, 1 = fully averaged losses). Default: 0')
    parser.add_argument('--no-fallback', action='store_true',
                        help=('Disable automatic fallback checks: do NOT attempt to inspect map files to detect per-tensor dtype mismatches. '
                              'When set, the script will act as if the quantized tensors of the map files were pure and any tensor mismatching the quant type will have its size "guessed" as if it had been quantized to that qtype. (Also forwarded to convert_map_qtype.py when used)'))
    parser.add_argument('--compute-missing-map', action='store_true',
                        help=('When set, if a tensors.<qtype>.map file is missing the script will attempt to compute it from tensors.bf16.map using convert_map_qtype.py. '
                              'Computed maps are not gpg-checked and their qtypes will be annotated in the produced recipe with a leading "!"'))
    parser.add_argument('--compute-all-map', action='store_true',
                        help=('When set instead of --compute-missing-map (mutually exclusive), produce all non-bf16 map files via convert_map_qtype.py. '
                              'Computed maps are not gpg-checked and their qtypes will be annotated in the produced recipe with a leading "!"'))
    parser.add_argument('--ignore-imatrix-rules', action='store_true',
                        help='(forwarded to convert_map_qtype.py) Ignore importance-matrix related checks.')
    parser.add_argument('--with-imatrix', action='store_true',
                        help='(forwarded to convert_map_qtype.py) Indicate that an importance matrix is available (satisfies imatrix checks).')
    parser.add_argument('--fallback-quants', nargs='+', default=[],
                        help=('(forwarded to convert_map_qtype.py) List of qtypes (space-separated) to whitelist for fallback (case-insensitive). '
                              'Example: --fallback-quants iq2_xs IQ3_S q8_k.'))
    parser.add_argument('--fallback-quants-forbidden', nargs='+', default=[],
                        help=("(forwarded to convert_map_qtype.py) List of regex patterns (space-separated) matching qtypes that must NOT be used as fallbacks. "
                              "Example: --fallback-quants-forbidden '^(iq1_|Q8_K$)' '.*_bn$'."))

    args = parser.parse_args()

    # Validate compute flags mutual exclusion
    if args.compute_missing_map and args.compute_all_map:
        parser.error("--compute-missing-map and --compute-all-map are mutually exclusive")

    # --use-greedy-quant-assign and --use-auto-quant-assign are mutually exclusive
    if args.use_greedy_quant_assign and args.use_auto_quant_assign:
        parser.error("--use-greedy-quant-assign and --use-auto-quant-assign are mutually exclusive")

    # --auto-force-combo must be a valid combo bitmask (0..15) and only applies to the auto method.
    if args.auto_force_combo is not None:
        if not (0 <= args.auto_force_combo <= _COMBO_MAX):
            parser.error(f"--auto-force-combo must be between 0 and {_COMBO_MAX} "
                         f"(bitmask of class=1, pos=2, tier2=4, pareto=8); got {args.auto_force_combo}")
        if not args.use_auto_quant_assign:
            parser.error("--auto-force-combo only applies to --use-auto-quant-assign (it forces the "
                         "greedy second-pass combo).")

    # --tolerance is ignored under --use-auto-quant-assign: the auto method
    # aims at the exact target size (it strict-caps at budget_bytes
    # internally — see auto_quant_assign Step 0). This is expected behaviour,
    # so only note it under --info/--debug if the user passed --tolerance
    # explicitly (the INFO global isn't set yet here, so check args directly).
    if (args.info or args.debug) and args.use_auto_quant_assign and any(
        a == '--tolerance' or a.startswith('--tolerance=') for a in sys.argv[1:]
    ):
        print(
            f"[Info] --tolerance {args.tolerance} is ignored when "
            f"--use-auto-quant-assign is set: the auto method always aims "
            f"for the exact target size.",
            file=sys.stderr,
        )

    # Convenience: a single flag indicating whether ANY degradation-aware
    # assignment method is in use. The same auxiliary options (--quant-degradation-csv,
    # --synergistic-tensors, --per-tensor-degradation-scaling) apply to both.
    using_degradation_method = bool(args.use_greedy_quant_assign or args.use_auto_quant_assign)

    # Enforce: --quant-degradation-csv only valid with greedy/auto methods
    if args.quant_degradation_csv and not using_degradation_method:
        parser.error("--quant-degradation-csv may only be used with --use-greedy-quant-assign or --use-auto-quant-assign")

    # Enforce: --synergistic-tensors only valid with greedy/auto methods
    if args.synergistic_tensors and not using_degradation_method:
        parser.error("--synergistic-tensors may only be used with --use-greedy-quant-assign or --use-auto-quant-assign")

    # Enforce: --synergy-strength is only valid when using --synergistic-tensors
    if args.synergy_strength and args.synergy_strength > 0 and not args.synergistic_tensors:
        parser.error("--synergy-strength may only be used with --synergistic-tensors")

    # Enforce: --per-tensor-degradation-scaling only valid with greedy/auto methods
    if args.per_tensor_degradation_scaling and args.per_tensor_degradation_scaling > 0 and not using_degradation_method:
        parser.error("--per-tensor-degradation-scaling may only be used with --use-greedy-quant-assign or --use-auto-quant-assign")

    # Enforce: --auto-no-pareto-filter only valid with --use-auto-quant-assign
    if args.auto_no_pareto_filter and not args.use_auto_quant_assign:
        parser.error("--auto-no-pareto-filter may only be used with --use-auto-quant-assign")

    # Enforce: --auto-deg-exponent only valid with --use-auto-quant-assign
    if args.auto_deg_exponent is not None and not args.use_auto_quant_assign:
        parser.error("--auto-deg-exponent may only be used with --use-auto-quant-assign")
    
    if not args.per_tensor_degradation_scaling:
        per_tensor_degradation_scaling_final = 0.0 # Default value
    else:
        per_tensor_degradation_scaling_final = args.per_tensor_degradation_scaling

    # ---- BEGIN pgpy-based “trusted-keys.asc” check ----
    if not SKIP_GPG:
        SKIP_GPG = args.skip_gpg
    if not SKIP_GPG:
        # Validate and load public keys
        try:
            load_public_keys()
        except FileNotFoundError:
            print("[Error] trusted-keys.asc not found in script directory.", file=sys.stderr)
            print("[Hint] Provide trusted-keys.asc or use --skip-gpg.", file=sys.stderr)
            sys.exit(6)
        except ValueError as ve:
            print(f"[Error] {ve}", file=sys.stderr)
            print("[Hint] Add at least one valid public key or use --skip-gpg.", file=sys.stderr)
            sys.exit(7)
    # ---- END pgpy-based check ----

    def parse_regex_assign_list(raw_list):
        parsed = []
        for item in raw_list:
            try:
                pat, qt = item.split('=', 1)
            except ValueError:
                parser.error(f"Invalid regex-assign spec: {item}. Must be PATTERN=QTYPE")
            parsed.append((re.compile(pat), qt))
        return parsed

    cpu_regex_assign = parse_regex_assign_list(args.cpu_assign_tensors)
    gpu_regex_assign = parse_regex_assign_list(args.gpu_assign_tensors)

    DEBUG = args.debug
    INFO = args.info or DEBUG

    # Populate compute-map flags
    COMPUTE_MISSING_MAP = args.compute_missing_map
    COMPUTE_ALL_MAP = args.compute_all_map
    CONVERT_IGNORE_IMATRIX_RULES = args.ignore_imatrix_rules
    CONVERT_WITH_IMATRIX = args.with_imatrix
    CONVERT_FALLBACK_QUANTS = args.fallback_quants or ""
    CONVERT_FALLBACK_QUANTS_FORBIDDEN = args.fallback_quants_forbidden or ""

    # ---------------------------
    # Quant degradation handling
    # ---------------------------

    def uppercase_quant_degradation_keys(values: Dict[str, float]) -> Dict[str, float]:
        return {k.upper(): v for k, v in values.items()}

    # User provided CSV?
    quant_degradation_values: Dict[str, float] = {}
    if args.quant_degradation_csv:
        try:
            quant_degradation_values = load_quant_degradation_values(args.quant_degradation_csv)
            quant_degradation_values = uppercase_quant_degradation_keys(quant_degradation_values)
        except Exception as e:
            print(f"[Error] Failed to load quant degradation CSV {args.quant_degradation_csv}: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        # No CSV provided: use Qwen3-4B-Thinking-2507's degradation values from models/Qwen3-4B-Thinking-2507/group0/kld_results.csv
        quant_degradation_values = {'bf16': 0.0, 'iq1_bn': 14.758228, 'iq1_kt': 2.692801, 'iq1_m': 4.684445, 'iq1_m_r4': 4.617296, 'iq1_s': 4.562480, 'iq1_s_r4': 5.124850, 'iq2_bn': 15.467749, 'iq2_bn_r4': 15.445743, 'iq2_k': 0.883945, 'iq2_k_r4': 0.883945, 'iq2_kl': 0.584754, 'iq2_ks': 1.347207, 'iq2_kt': 1.214565, 'iq2_s': 0.465971, 'iq2_xs': 0.633596, 'iq2_xs_r4': 0.636844, 'iq2_xxs': 1.202639, 'iq2_xxs_r4': 1.208328, 'iq3_k': 0.164337, 'iq3_k_r4': 0.164337, 'iq3_ks': 0.211158, 'iq3_kt': 0.214378, 'iq3_s': 0.210123, 'iq3_s_r4': 0.213001, 'iq3_xxs': 0.348842, 'iq3_xxs_r4': 0.351102, 'iq4_k': 0.034494, 'iq4_k_r4': 0.034494, 'iq4_ks': 0.047722, 'iq4_ks_r4': 0.047722, 'iq4_kss': 0.073993, 'iq4_kt': 0.071823, 'iq4_nl': 0.052065, 'iq4_nl_r4': 0.051893, 'iq4_xs': 0.052575, 'iq4_xs_r8': 0.055797, 'iq5_k': 0.009814, 'iq5_k_r4': 0.009814, 'iq5_ks': 0.012268, 'iq5_ks_r4': 0.012268, 'iq6_k': 0.003411, 'q2_K': 0.895361, 'q2_k_r4': 0.896636, 'q3_K': 0.226457, 'q3_k_r4': 0.228156, 'q4_0': 0.070737, 'q4_0_r8': 0.070601, 'q4_1': 0.050200, 'q4_K': 0.046677, 'q4_k_r4': 0.046609, 'q5_0': 0.018810, 'q5_0_r4': 0.018876, 'q5_1': 0.014465, 'q5_K': 0.015590, 'q5_k_r4': 0.015766, 'q6_0': 0.005317, 'q6_0_r4': 0.005244, 'q6_K': 0.004040, 'q6_k_r4': 0.006687, 'q8_0': 0.001449, 'q8_0_r8': 0.001515, 'q8_k_r8': 0.004102, 'q8_KV': 0.038383}
        quant_degradation_values = uppercase_quant_degradation_keys(quant_degradation_values)

    if INFO:
        if args.quant_degradation_csv:
            print(f"[Info] Loaded degradation values for {len(quant_degradation_values)} quant types from CSV" + f": {args.quant_degradation_csv}", file=sys.stderr)
        else:
            print(f"[Info] Loaded Qwen3-4B-Thinking-2507 default degradation values for {len(quant_degradation_values)} quant types. You must tweak --exponential-factor for optimum quality if your recipe model isn't Qwen3-4B-Thinking-2507. Consider using --quant-degradation-csv group0/kld_results.csv of your model instead.", file=sys.stderr)

    # helper lookup that greedy_quant_assign will use (returns float or None)
    warned_missing_qtypes = set()
    @tracked_lru_cache(maxsize=None)
    def quant_deg_lookup(qtype: str):
        # Prefer CSV value

        # 1. Exact match
        if qtype.upper() in quant_degradation_values:
            return quant_degradation_values[qtype.upper()]

        # The degradation data for iq1_s != iq1_s_r4, this is the only exception
        if qtype != "iq1_s" and qtype != "iq1_s_r4":
            # 2. If qtype ends with _r4 or _r8 → try base
            base_qtype = re.sub(r"_r[48]$", "", qtype)
            _base_qtype = base_qtype.upper()
            if _base_qtype != qtype.upper() and _base_qtype in quant_degradation_values:
                return quant_degradation_values[_base_qtype]

            # 3. If base not found → try adding _r4 and _r8
            for suffix in ("_r4", "_r8"):
                candidate = base_qtype + suffix
                _candidate = candidate.upper()
                if _candidate in quant_degradation_values:
                    return quant_degradation_values[_candidate]
            
        # If absent, request user to run group0_enricher.py
        if args.quant_degradation_csv:
            suggested_cmd = (
                'quant_degradation_equation_target=$(cd models/__TARGET_MODEL__/group0 && ../../../model_tensor_bpw_metric.py --results-csv kld_results.csv --c-free --exclude-qtypes \'.*_bn.*$\' --transforms "identity" --ignore-outliers 50 --allow-impure-map --p-grid-max 15 --p-grid-steps 100 --d-from-lowest 1 --penalize-above 15 --resemblance-metric r2 --equation-only 2> /dev/null) && '
                f'./group0_enricher.py --target-csv {shlex.quote(args.quant_degradation_csv)} --output group0_enriched.csv --target-mean-equation "$quant_degradation_equation_target"'
            )
            print(f"[Error] Quant degradation value for qtype '{qtype}' missing from custom CSV. "
                f"You can enrich your {shlex.quote(args.quant_degradation_csv)} using group0_enricher.py with this command line (replace __TARGET_MODEL__ with your model name):", file=sys.stderr)
            print(f"[Error]   {suggested_cmd}", file=sys.stderr)
            sys.exit(2)
        else:
            print(f"[Error] Quant degradation value for qtype '{qtype}' missing, please provide a group0 degradation csv via --quant-degradation-csv. ", file=sys.stderr)
            sys.exit(2)

    # make --no-fallback visible to top-level helpers
    NO_FALLBACK = bool(args.no_fallback)

    if args.cpu_tensors and not args.cpu_quants:
        parser.error("--cpu-quants is required when --cpu-tensors is used")
    if args.gpu_tensors and not args.gpu_quants:
        parser.error("--gpu-quants is required when --gpu-tensors is used")

    cpu_quants = args.cpu_quants
    # if not cpu_quants and args.gpu_quants:
    #     cpu_quants = DEFAULT_QUANTS
    # Reorder cpu_quants from highest to lowest bpw
    try:
        cpu_quants = sorted(cpu_quants, key=_quant_sort_key, reverse=True)
        if INFO: print(f"[Info] CPU-friendly quants reordered by bpw: {cpu_quants}", file=sys.stderr)
    except Exception:
        pass

    gpu_quants = args.gpu_quants
    # By default we assume the user wants everything on the GPU
    if not cpu_quants and not gpu_quants:
        if INFO: print(f"[Info] No quants selected, reverting to GPU default selection: {DEFAULT_QUANTS}", file=sys.stderr)
        gpu_quants = DEFAULT_QUANTS
    # if not gpu_quants and args.cpu_quants:
    #     gpu_quants = DEFAULT_QUANTS
    # Reorder gpu_quants from highest to lowest bpw
    try:
        gpu_quants = sorted(gpu_quants, key=_quant_sort_key, reverse=True)
        if INFO: print(f"[Info] GPU-friendly quants reordered by bpw: {gpu_quants}", file=sys.stderr)
    except Exception:
        pass

    if INFO: print(f"[Info] Loading CSV: {args.csv_file}", file=sys.stderr)
    df = pd.read_csv(args.csv_file)
    if 'QTYPE' not in df.columns:
        print("Error: CSV must have 'QTYPE' as first column.", file=sys.stderr)
        sys.exit(1)

    #reduction_factors = load_sample_ppl_table(args.sample_ppl)
    row = select_qtype(df, args.qtype)
    qtype = row['QTYPE']
    if INFO: print(f"[Info] Selected QTYPE: {qtype}", file=sys.stderr)

    # ---- NEW: Parse synergistic tensor groupps into per layer groups----
    synergistic_groups = parse_group_argument(args.synergistic_tensors, "--synergistic-tensors", parser, info_flag=INFO)

    #print(row.to_string(max_rows=None))
    # ---- NEW: Harmonize matching tensor rows ----
    # Convert nargs='+' form (list of comma-separated strings) into list-of-lists
    harmonize_groups = []
    if args.harmonization_technique != 0:
        harmonize_groups = parse_group_argument(args.harmonize_tensors, "--harmonize-tensors", parser, info_flag=INFO)

    # harmonize_groups is now a list-of-lists of regex strings (or empty list to disable)

    # --- Disable harmonization here when using greedy or rank quant assign — both methods handle harmonization internally ---
    if not (args.use_greedy_quant_assign or args.use_auto_quant_assign):
        try:
            # Provide df columns (excluding QTYPE) so the helper can match against available tensor names
            harmonize_row(row, [c for c in df.columns if c != 'QTYPE'], harmonize_groups, args.harmonization_technique)
        except ValueError as ve:
            parser.error(str(ve))

        # Which columns we actually want to update
        cols_to_update = [c for c in row.index if c != "QTYPE" and c in df.columns]

        # Determine the df index to update
        if hasattr(row, "name") and row.name is not None and row.name in df.index:
            idx = row.name
        else:
            mask = (df["QTYPE"] == row["QTYPE"])
            matches = df.index[mask].tolist()
            if len(matches) == 0:
                raise ValueError(f"Could not find any row in df with QTYPE == {row['QTYPE']!r} to update.")
            if len(matches) > 1:
                # warning; choose first. Adjust if you prefer to update all matches.
                print(f"[Warning] Multiple rows with QTYPE == {row['QTYPE']!r}; updating the first match.", file=sys.stderr)
            idx = matches[0]

        # Update columns one-by-one (avoids type-checker issues and is explicit)
        for col in cols_to_update:
            # row[col] might be a numpy scalar or python scalar — both are fine
            df.at[cast(Any, idx), col] = row[col]

    # ---- END harmonization ----
    #print(row.to_string(max_rows=None))

    # Pre-fetch maps
    if not fetch_map_for_qtype(qtype):
        print(f"Error: Fetching valid map for qtype: {qtype} was unsuccessful.", file=sys.stderr)
        sys.exit(8)
    _, items, _ = get_map_sizes_and_elements(qtype, True)
    _, items_f32, _ = get_map_sizes_and_elements('f32', True)

    _items = items_f32
    if not _items:
        _items = items

    # -------------------------
    # Compute bf16 total size and determine exponential-factor default (if needed)
    # -------------------------
    # We'll compute a final effective exponential-factor (exp_factor_final) which will be used
    # throughout the assignment flow instead of directly using args.exponential_factor, so we
    # can substitute the automatic value when --use-greedy-quant-assign is used and the user
    # did not explicitly pass --exponential-factor.
    try:
        bf16_sizes, _, _ = get_map_sizes_and_elements('bf16', True)
    except Exception:
        bf16_sizes = {}
    bf16_total_bytes = sum(bf16_sizes.values()) if bf16_sizes else 0
    bf16_total_gib = bf16_total_bytes / GIB if bf16_total_bytes else 0.0

    # Determine effective exponential-factor
    if args.exponential_factor is not None:
        exp_factor_final = float(args.exponential_factor)
        if INFO:
            print(f"[Info] Exponential-factor value set by the user: {exp_factor_final}", file=sys.stderr)
    else:
        # Not specified by user
        if args.use_auto_quant_assign:
            # Rank method scores assignments by Σ loss^p · deg^q. When the user
            # leaves --exponential-factor unset, we run an internal auto-sweep
            # over a small (p, q) grid inside auto_quant_assign and pick the
            # pair whose chosen assignment minimises the worst per-tensor
            # loss·degradation product. exp_factor_final is left at 1.0 here
            # purely as a fallback / display value — the real chosen p will
            # come back via chosen_params_out.
            if INFO:
                print(f"[Info] --exponential-factor not specified for --use-auto-quant-assign: enabling internal (p, q) auto-sweep.", file=sys.stderr)
            exp_factor_final = 1.0
        elif args.use_greedy_quant_assign and args.quant_degradation_csv:
            if INFO:
                print(f"[Info] Using exponential-factor value of 1.0 because quant degradation csv provided", file=sys.stderr)
            exp_factor_final = 1.0
        elif args.use_greedy_quant_assign:
            # Use the requested equation: y = 0.5 * ln(x), where x is the bf16 total tensor size in GiB.
            # If bf16_total_gib is <= 0 or ln is non-positive result, fallback to 8.0
            try:
                if bf16_total_gib > 0:
                    y = 0.5 * math.log(bf16_total_gib)
                    if y <= 0 or not math.isfinite(y):
                        # If the computed y is not positive (or not finite), fall back conservatively
                        if INFO:
                            print(f"[Info] Computed y = 0.5*ln(x) with x={bf16_total_gib:.6f} GiB produced non-positive/invalid value {y}; falling back to 3.0", file=sys.stderr)
                        exp_factor_final = 3.0
                    else:
                        exp_factor_final = float(y)
                else:
                    if INFO:
                        print(f"[Info] BF16 total size x is {bf16_total_gib:.6f} GiB (<=0); cannot compute ln(x). Falling back to exponential-factor=3.0", file=sys.stderr)
                    exp_factor_final = 3.0
            except Exception as e:
                if INFO:
                    print(f"[Info] Failed to compute automatic exponential-factor from bf16 size: {e}; falling back to 3.0", file=sys.stderr)
                exp_factor_final = 3.0

            if INFO:
                print(f"[Info] Automatic exponential-factor equation: y = 0.5 * ln(x) (x = bf16 total size in GiB).", file=sys.stderr)
                print(f"[Info] bf16 total size x = {bf16_total_gib:.6f} GiB -> y = {exp_factor_final}", file=sys.stderr)
        else:
            if INFO:
                print(f"[Info] Greedy not used, exponential-factor value set to 8.0", file=sys.stderr)
            # Not using greedy and user did not specify => default 8.0
            exp_factor_final = 8.0

    # -------------------------

    # Collect tensor names (either from csv or from map file)
    if INFO: print(f"[Info] Get all tensor names", file=sys.stderr)
    if args.tensors_from_csv:
        tensor_names = [c for c in df.columns if c != 'QTYPE']
    else:
        tensor_names = [n for n,d in _items.items()]

    # Identify all f32 tensors once
    if INFO: print(f"[Info] Get f32 tensor names", file=sys.stderr)
    # get_map_sizes_and_elements returns (sizes, actual_qtypes, elements)
    f32_names = [n for n,d in _items.items() if d == 'f32']

    # Classify tensors
    classes = classify_tensors(tensor_names, args.cpu_tensors + args.cpu_assign_tensors, args.gpu_tensors + args.gpu_assign_tensors)

    subclasses_to_assign = {'cpu': [], 'gpu': []}
    subclasses_assigned = {'cpu': [], 'gpu': []}

    # Build values dict, converting strings (e.g. '0.0653%') properly and pre-assign tensors that haven't been measured
    values = {}
    pre_assignments = {}
    pre_assignments_offset = {'cpu': 0, 'gpu': 0}
    for cls in ['cpu', 'gpu']:
        if cls == 'cpu' and not cpu_quants:
            if INFO: print(f"[Info] CPU-friendly quants skipped because not being user-specified.", file=sys.stderr)
            continue # Skip loop if empty quants
        if cls == 'gpu' and not gpu_quants:
            if INFO: print(f"[Info] GPU-friendly quants skipped because not being user-specified.", file=sys.stderr)
            continue # Skip loop if empty quants
        quants = cpu_quants if cls == 'cpu' else gpu_quants
        pat_assign_tensors = [pat.split('=')[0] for pat in args.cpu_assign_tensors] if cls == 'cpu' else [pat.split('=')[0] for pat in args.gpu_assign_tensors]
        names = classes.get(cls, [])
        if cls == 'cpu':
            _assign_qtype = assign_qtype(args.cpu_assign_qtype, cpu_regex_assign, quants, names)
        else:
            _assign_qtype = assign_qtype(args.gpu_assign_qtype, gpu_regex_assign, quants, names)

        # skip if nothing for this cls
        if not names:
            continue

        for name in names:
            # when specifically mentioned in --cpu/gpu-assign-tensors param → pre-assign
            if any(re.fullmatch(pattern, name) for pattern in pat_assign_tensors):
                pre_assignments[name] = _assign_qtype[name]

                subclasses_assigned[cls].append(name)
                if INFO: print(f"[Info] Assigning {name!r} → {pre_assignments[name]!r} (in --cpu/gpu-assign-tensors parameter)", file=sys.stderr)

                # jump to next tensor
                continue
            # missing measurement → pre-assign
            elif name not in row or pd.isna(row.at[name]):
                if name in f32_names:
                    # This is a f32 tensor which we must skip
                    continue
                pre_assignments[name] = _assign_qtype[name]

                subclasses_assigned[cls].append(name)
                if INFO: print(f"[Info] Assigning {name!r} → {pre_assignments[name]!r} (missing metrics)", file=sys.stderr)

                # jump to next tensor
                continue

            # got a raw value → convert and store
            raw = row[name]
            conv = _convert_value(raw)
            if np.isnan(conv):
                print(f"Error: could not parse numeric value for tensor {name!r}: {raw!r}", file=sys.stderr)
                sys.exit(1)

            values[name] = conv
            subclasses_to_assign[cls].append(name)

        # 1. Get all unique q-types
        _assign_qtype_qtypes = set(_assign_qtype.values())

        # 2. Loop over each q-type
        for _qtype in _assign_qtype_qtypes:
            # 2a. Collect all tensor names that were assigned this qtype
            _tensor_subgroup_names = [
                name
                for name, assigned_q in _assign_qtype.items()
                if assigned_q == _qtype and name in pre_assignments
            ]

            # 2b. Compute the total size for this group
            size = total_size_for_quant(_tensor_subgroup_names, _qtype)

            # 2c. Add it into your pre_assignments_offset for whatever class 'cls' is
            #     (you’ll need to define or look up `cls` in your context)
            pre_assignments_offset[cls] += size

    totals = {}
    # Create separate assignment storage per class to avoid mixing identical qnames
    assignments = {'cpu': {}, 'gpu': {}}

    # prepare per-class f32 offsets (skip if user manually included 'f32' as a quant)
    f32_offset = {'cpu': 0, 'gpu': 0}
    f32_classes = {}
    add_f32 = not args.ignore_f32
    if add_f32:
        f32_classes = classify_tensors(f32_names, args.cpu_tensors + args.cpu_assign_tensors, args.gpu_tensors + args.gpu_assign_tensors)
        for cls in ['gpu', 'cpu']:
            if cls == 'cpu' and not cpu_quants:
                continue # Skip loop if empty quants
            if cls == 'gpu' and not gpu_quants:
                continue # Skip loop if empty quants
            # if user did *not* list 'f32' in 'cls'_quants, add to cls offset
            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
                f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), 'f32')
                if f32_offset[cls] == 0:
                    f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), qtype)
    
    # Track how many fallback corrections happened
    fallback_corrections = 0

    # Track precomputed extremes per class
    extremes = {}

    # ---- Ensure single registry for all tensors (persist across classes) ----
    # tensor_index[cls] is a list of dicts:
    # {'name': str, 'orig_q': str|None, 'final_q': str|None, 'size': int, 'kind': 'f32'|'pre'|'measured'}
    tensor_index = {'cpu': [], 'gpu': []}

    # Process GPU and CPU classes
    for cls in ['gpu', 'cpu']:
        if cls == 'cpu' and not cpu_quants:
            continue # Skip loop if empty quants
        if cls == 'gpu' and not gpu_quants:
            continue # Skip loop if empty quants
        quants = cpu_quants if cls == 'cpu' else gpu_quants
        names = classes.get(cls, [])
        names_to_assign = subclasses_to_assign.get(cls, [])
        names_assigned = subclasses_assigned.get(cls, [])
        if not names:
            continue
   
        print(f"\n## {'CPU' if cls=='cpu' else 'GPU'}-loaded tensors")
        class_vals = {n: values[n] for n in names_to_assign}

        # Determine bounds and outliers
        k_val = args.cpu_irq_k if cls=='cpu' else args.gpu_irq_k
        lower, upper = compute_iqr_bounds(class_vals, k_val)
        if INFO: print(f"[Info] {cls.upper()} outlier bounds: lower={lower:.4f}, upper={upper:.4f}", file=sys.stderr)
        out_low = [n for n,v in class_vals.items() if v < lower]
        out_high = [n for n,v in class_vals.items() if v > upper]
        if DEBUG: print(f"[Debug] {cls.upper()} low outliers: {out_low}", file=sys.stderr)
        if DEBUG: print(f"[Debug] {cls.upper()} high outliers: {out_high}", file=sys.stderr)

        # Assign extremes and compute outlier size deduction
        outlier_bytes = 0

        # Parse/compile harmonize groups (safe: if args.harmonize_quants is invalid, treat as disabled)
        try:
            _harm_groups = ast.literal_eval(args.harmonize_quants)
        except Exception:
            _harm_groups = []
        compiled_groups = []
        if _harm_groups:
            for g in _harm_groups:
                try:
                    compiled_groups.append([re.compile(p) for p in g])
                except Exception:
                    # If a user provided an invalid regex, just skip harmonization for safety
                    compiled_groups = []
                    break

        # Only consider names from class_vals for harmonization (index-wise matching)
        class_names = sorted(list(class_vals.keys()))

        processed_outliers = set()

        def _find_harmony_partners_and_mean(name, extreme_q, group_idx_hint=None):
            """
            Try to find index-wise partners for 'name' within compiled_groups,
            restricted to class_names. If successful returns (matched_names_list, mean_size, group_idx).
            Otherwise returns None.
            """
            if not compiled_groups:
                return None

            for gi, compiled in enumerate(compiled_groups):
                # see if any pattern in this group matches `name`
                matched_pi = None
                for pi, cre in enumerate(compiled):
                    if cre.search(name):
                        matched_pi = pi
                        break
                if matched_pi is None:
                    continue  # this group doesn't include 'name'

                # build candidate lists from class_names (NOT sizes_map/full map)
                candidate_lists = []
                for cre in compiled:
                    lst = [n for n in class_names if cre.search(n)]
                    candidate_lists.append(sorted(lst))

                lengths = [len(l) for l in candidate_lists]
                if len(set(lengths)) != 1:
                    # counts differ => user likely split tensors across CPU/GPU or similar.
                    if INFO:
                        print(f"[Info] Warning: skipping harmonization for group {gi} quant {extreme_q} because pattern match counts differ (counts={lengths}); using per-tensor sizes.", file=sys.stderr)
                    return None

                if lengths[0] == 0:
                    return None

                # find index of name in its pattern list
                my_list = candidate_lists[matched_pi]
                if name not in my_list:
                    # safety: unexpected
                    if INFO:
                        print(f"[Info] Warning: {name!r} not present in pattern list for harmonization group {gi}; skipping harmonization for this tensor.", file=sys.stderr)
                    return None
                idx_in = my_list.index(name)

                # partner names = same index from all candidate_lists
                matched_names = [lst[idx_in] for lst in candidate_lists]

                # compute per-tensor sizes for the assigned extreme quant and take mean
                partner_sizes = [total_size_for_quant([nm], extreme_q) for nm in matched_names]
                mean_size = float(sum(partner_sizes)) / len(partner_sizes)

                return matched_names, mean_size, gi

            return None

        # helper to process a list of outliers (either out_low or out_high)
        def _process_outliers_list(out_list, assigned_q, desc):
            nonlocal outlier_bytes, processed_outliers
            for n in out_list:
                if n in processed_outliers:
                    continue

                # try to find harmonized partners (only among class_names)
                found = _find_harmony_partners_and_mean(n, assigned_q)
                if not found:
                    # no harmonization applies -> normal single-tensor assignment
                    assignments[cls][n] = assigned_q
                    size = total_size_for_quant([n], assigned_q)
                    outlier_bytes += size
                    processed_outliers.add(n)
                    if INFO:
                        print(f"[Info] Assigned {desc} quant {assigned_q} to outlier {n}, size={size/GIB:.3f} GiB", file=sys.stderr)
                else:
                    matched_names, size_harmonized, group_idx = found
                    # assign same extreme quant and harmonized size to each matched partner
                    for nm in matched_names:
                        if nm in processed_outliers:
                            continue
                        assignments[cls][nm] = assigned_q
                        outlier_bytes += size_harmonized
                        processed_outliers.add(nm)
                        if INFO:
                            print(f"[Info] Assigned {desc} quant {assigned_q} to outlier {nm}, size={size_harmonized/GIB:.3f} GiB (harmonized group {group_idx})", file=sys.stderr)

        if not (args.use_greedy_quant_assign or args.use_auto_quant_assign):
            # process low and high outliers (lowest quant = quants[-1], highest quant = quants[0])
            _process_outliers_list(out_low, quants[-1], "lowest")
            _process_outliers_list(out_high, quants[0], "highest")

            # remove processed outliers from class_vals so they are not considered in normal assignment
            for n in list(processed_outliers):
                class_vals.pop(n, None)

        # Normal assignment on remaining
        
        # Determine max-size argument, allowing percent
        raw_max = args.cpu_tensors_max_size if cls == 'cpu' else args.gpu_tensors_max_size
        max_arg_bytes = None
        # Precompute extremes once
        highest_q = max(quants, key=_quant_sort_key)
        lowest_q = min(quants, key=_quant_sort_key)
        max_ref = total_size_for_quant(names_to_assign, highest_q) + f32_offset[cls] + pre_assignments_offset[cls]
        min_ref = total_size_for_quant(names_to_assign, lowest_q) + f32_offset[cls] + pre_assignments_offset[cls]
        extremes[cls] = {
            'highest_q': highest_q, 'lowest_q': lowest_q,
            'max_ref': max_ref, 'min_ref': min_ref
        }

        _max_arg_bytes = 0
        if raw_max:
            if isinstance(raw_max, str) and raw_max.endswith('%'):
                pct = float(raw_max.rstrip('%')) / 100.0
                _max_arg_bytes = pct * max_ref
                if INFO: print(f"[Info] {cls.upper()} max-size set to {raw_max} of {highest_q} total ({max_ref/GIB:.3f} GiB) = {_max_arg_bytes/GIB:.3f} GiB", file=sys.stderr)
            else:
                _max_arg_bytes = float(raw_max) * GIB
            max_arg_bytes = _max_arg_bytes
            max_arg_bytes -= outlier_bytes # deduct outliers
            max_arg_bytes -= f32_offset.get(cls, 0) # deduct f32 offset
            max_arg_bytes -= pre_assignments_offset.get(cls, 0) # deduct pre-assigned offset
            if INFO: print(f"[Info] Deducted outliers and f32 total {outlier_bytes/GIB:.3f} GiB from target, adjusted max={max_arg_bytes/GIB:.3f} GiB", file=sys.stderr)

        if _max_arg_bytes >= (max_ref - max_ref*0.0001):
            # Assign highest quant to all (except extremes)
            if INFO: print(f"[Info] Reasonably assigning highest quant to all tensors...", file=sys.stderr)
            assignment, sizes = assign_quants(
                [highest_q], None, class_vals)
            total_bytes = sum(sizes.values())
        elif _max_arg_bytes == 0:
            # Assign lowest quant to all (except extremes)
            if INFO: print(f"[Info] Reasonably assigning lowest quant to all tensors...", file=sys.stderr)
            assignment, sizes = assign_quants(
                [lowest_q], None, class_vals)
            total_bytes = sum(sizes.values())
        elif max_arg_bytes:
            if args.use_greedy_quant_assign or args.use_auto_quant_assign:
                # ---- Exclude PHANTOM qtypes ----
                # A requested qtype is "phantom" for this class when NONE of the
                # tensors-to-assign actually materialise at it: every tensor in
                # that qtype's tensors.<q>.map falls back to a different stored
                # dtype. Example: iq2_s on gemma — no tensor can be quantised to
                # iq2_s, so the map stores every tensor as iq2_xs. The phantom
                # then carries iq2_xs's *sizes* but iq2_s's own (better) tabulated
                # degradation, so on the (size, deg) Pareto frontier it spuriously
                # dominates the real qtype it aliases (here both iq2_xs and q2_K),
                # corrupting the bulk assignment AND the outlier-target rung count
                # (observed: token_embd jumping q3_K → iq4_xs at 26.12% purely
                # because iq2_s collapsed two frontier rungs). Drop such qtypes
                # up-front so they never enter the pool, the allowed lists, or the
                # size table.
                _class_quants_raw = gpu_quants if cls == 'gpu' else cpu_quants
                _class_quants_eff = []
                for _q in _class_quants_raw:
                    try:
                        _, _actual_q, _ = get_map_sizes_and_elements(_q)
                    except Exception:
                        _class_quants_eff.append(_q)  # can't determine → keep
                        continue
                    _q_norm = transform_q_suffix(str(_q)).lower()
                    _materialised = any(
                        transform_q_suffix(str(_actual_q.get(n, ''))).lower() == _q_norm
                        for n in names_to_assign
                    )
                    if _materialised:
                        _class_quants_eff.append(_q)
                    elif DEBUG or INFO:
                        print(
                            f"[Info] Excluding phantom qtype {_q!r} for {cls.upper()}: "
                            f"no tensor-to-assign materialises at it (every tensor in its "
                            f"map falls back to another dtype).",
                            file=sys.stderr,
                        )
                # Safety: never drop everything (pathological all-phantom case).
                if not _class_quants_eff:
                    _class_quants_eff = list(_class_quants_raw)

                # Build tensor_quants mapping (default to cls-specific quants,
                # minus any phantom qtypes excluded above).
                tensor_quants_local = {n: list(_class_quants_eff) for n in names_to_assign}

                # ---- Expand CLI regex groups into concrete per-layer lists restricted to this class' tensors ----
                # Harmonization groups
                try:
                    expanded_harmonize_groups = expand_harmonize_groups(harmonize_groups, names_to_assign) if harmonize_groups else []
                except Exception:
                    expanded_harmonize_groups = []
                if INFO and harmonize_groups and not expanded_harmonize_groups:
                    print(f"[Info] No harmonize groups expanded for class {cls}.", file=sys.stderr)

                # Synergistic groups
                try:
                    expanded_synergistic_groups = expand_harmonize_groups(synergistic_groups, names_to_assign) if synergistic_groups else []
                except Exception:
                    expanded_synergistic_groups = []
                if INFO and synergistic_groups and not expanded_synergistic_groups:
                    print(f"[Info] No synergistic groups expanded for class {cls}.", file=sys.stderr)

                # Build per-tensor scaled degradation function
                scaling_exponent = per_tensor_degradation_scaling_final
                if scaling_exponent > 0 and class_vals:
                    mean_loss = sum(class_vals.values()) / len(class_vals)
                    def make_degradation_fn(base_lookup, cls_vals, mean_loss, exponent):
                        def fn(tensor, qtype):
                            base = base_lookup(qtype)      # may exit if qtype unknown
                            if base is None:
                                return None
                            loss = cls_vals.get(tensor)
                            if loss is not None and mean_loss > 0:
                                scale = (loss / mean_loss) ** exponent
                                return base * scale
                            return base
                        return fn
                    degradation_fn = make_degradation_fn(quant_deg_lookup, class_vals, mean_loss, scaling_exponent)
                else:
                    # No scaling – just wrap the old lookup as a two-argument callable
                    degradation_fn = lambda tensor, qtype: quant_deg_lookup(qtype)

                # Build the per-tensor sizes mapping used by both methods.
                # Use the phantom-filtered quant list so excluded qtypes never
                # appear in the size table / pool.
                _quants_for_class = _class_quants_eff
                tensor_sizes_local = {
                    n: {q: get_map_sizes_and_elements(q)[0].get(n, 0)
                        for q in _quants_for_class}
                    for n in names_to_assign
                }

                if args.use_auto_quant_assign:
                    # The auto method also benefits from being told about the
                    # class-default assign-qtype so that zero-kld outliers can be
                    # assigned to the smallest-size qtype across the full set the
                    # user has allowed (gpu/cpu quants + the class assign-qtype).
                    _extra_outlier_qtypes: List[str] = []
                    cls_assign_qtype = args.cpu_assign_qtype if cls == 'cpu' else args.gpu_assign_qtype
                    if cls_assign_qtype:
                        _extra_outlier_qtypes.append(cls_assign_qtype)

                    # Make sure outlier-candidate qtypes have known sizes for each
                    # tensor (lazily fetch maps just in case the user passed a
                    # qtype outside of gpu/cpu-quants).
                    for _q in list(_extra_outlier_qtypes):
                        try:
                            fetched, _, _ = get_map_sizes_and_elements(_q)
                            for n in names_to_assign:
                                if _q not in tensor_sizes_local[n]:
                                    tensor_sizes_local[n][_q] = int(fetched.get(n, 0))
                        except Exception:
                            pass

                    # The auto method uses *label* degradation (the tabulated
                    # deg for the qtype the recipe will list), not the actual
                    # fallback-storage deg. Reasons:
                    #   - The user reads recipes by label; "no iq1_s entries"
                    #     should mean exactly that.
                    #   - Label deg keeps Pareto filtering predictable — a label
                    #     is dominated iff its tabulated (size, deg) is, without
                    #     two different labels collapsing into one storage point.
                    #   - For tensors whose label falls back to higher-quality
                    #     storage, sizing is already accurate (tensor_sizes uses
                    #     actual fallback bytes), so budget is correct even
                    #     though the labeled deg is a slight pessimism.

                    # Auto-sweep (p, q) inside the auto method ONLY when the
                    # user hasn't pinned both exponents. If --exponential-factor
                    # is set, the user has expressed a preference for p; if
                    # --auto-deg-exponent is also set, both are pinned and we
                    # skip the sweep. Otherwise we sweep and report the chosen
                    # (p, q) in the recipe's hidden-parameters footer.
                    _auto_sweep = (args.exponential_factor is None)
                    _user_q = args.auto_deg_exponent if args.auto_deg_exponent is not None else 1.0
                    if args.auto_deg_exponent is not None:
                        # User pinned q — still skip sweep only when p is also
                        # pinned (avoid surprising behaviour).
                        _auto_sweep = _auto_sweep and (args.exponential_factor is None and False)
                    chosen_auto_params: Dict[str, float] = {}
                    assignment, total_bytes = auto_quant_assign(
                        tensors=names_to_assign,
                        tensor_sizes=tensor_sizes_local,
                        ppl_loss=class_vals,
                        degradation_fn=degradation_fn,
                        tensor_quants=tensor_quants_local,
                        budget_bytes=int(max_arg_bytes),
                        debug=DEBUG,
                        harmonized_groups=expanded_harmonize_groups,
                        loss_exponent=exp_factor_final,
                        deg_exponent=_user_q,
                        auto_sweep=_auto_sweep,
                        synergistic_groups=expanded_synergistic_groups,
                        synergy_strength=args.synergy_strength,
                        tolerance=args.tolerance,
                        extra_outlier_qtypes=_extra_outlier_qtypes,
                        pareto_filter=not args.auto_no_pareto_filter,
                        chosen_params_out=chosen_auto_params,
                        force_combo=args.auto_force_combo,
                    )
                    # Stash the chosen (p, q) onto args so the hidden-parameter
                    # footer can include them in the recipe.
                    if chosen_auto_params:
                        args.__dict__.setdefault('_auto_chosen_params', {})[cls] = chosen_auto_params
                else:
                    # ---- Call greedy quant assignment with all parameters ----
                    assignment, total_bytes = greedy_quant_assign(
                        tensors=names_to_assign,
                        tensor_sizes=tensor_sizes_local,
                        ppl_loss=class_vals,
                        degradation_fn=degradation_fn,
                        tensor_quants=tensor_quants_local,
                        budget_bytes=int(max_arg_bytes),
                        debug=DEBUG,
                        harmonized_groups=expanded_harmonize_groups,
                        loss_exponent=exp_factor_final,
                        synergistic_groups=expanded_synergistic_groups,
                        synergy_strength=args.synergy_strength
                    )
            else:
                assignment, total_bytes = optimize_midpoint_and_assign(
                    quants, None, class_vals,
                    max_arg_bytes, args.tolerance, exp_factor_final, harmonize_groups=harmonize_groups)
            #print(f"# Optimized sub-total {cls.upper()} size excluding outliers and f32: {total_bytes/GIB:.3f} GiB")
        else:
            assignment, sizes = assign_quants(
                quants, None, class_vals, harmonize_groups=harmonize_groups)
            total_bytes = sum(sizes.values())

        assignments[cls].update(assignment)  # Store per-class assignments
        totals[cls] = total_bytes + outlier_bytes # add outliers back
        totals[cls] += f32_offset.get(cls, 0) # add f32 offset to the grand total
        totals[cls] += pre_assignments_offset.get(cls, 0) # add pre-assigned offset to the grand total
        print(f"# Total {cls.upper()} size: {totals[cls]/GIB:.3f} GiB")
        print(f"# Outlier tensors total size: {outlier_bytes/GIB:.3f} GiB")
        print(f"# f32 tensors total size: {f32_offset.get(cls, 0)/GIB:.3f} GiB")
        print(f"# Pre-assigned tensors total size: {pre_assignments_offset.get(cls, 0)/GIB:.3f} GiB")
        if max_arg_bytes:
            print(f"# Optimized sub-total {cls.upper()} size excluding outliers and f32: {total_bytes/GIB:.3f} GiB")

        # ----------------- Verbose fallback-inspection & printing block (enhanced) -----------------
        # Purpose: when we print assignments like ^regex$=QTYPE, verify the map actually
        # contains that tensor with that QTYPE. If not, substitute the qtype found in the map.
        # This block records the final qtype back into pre_assignments / assignments[cls]
        # so later summary and size/bpw calculations use the corrected map-reported values.

        def _regex_override_for(name, cls_local):
            """
            Check compiled regex override lists (cpu_regex_assign / gpu_regex_assign).
            Returns the forced qtype (string) if any override applies, else None.
            """
            try:
                if cls_local == 'cpu':
                    for cre, qt in cpu_regex_assign:
                        if cre.fullmatch(name):
                            if DEBUG:
                                print(f"[Debug] Regex override (cpu) matched {name} -> {qt}", file=sys.stderr)
                            return qt
                else:
                    for cre, qt in gpu_regex_assign:
                        if cre.fullmatch(name):
                            if DEBUG:
                                print(f"[Debug] Regex override (gpu) matched {name} -> {qt}", file=sys.stderr)
                            return qt
            except Exception as e:
                if DEBUG:
                    print(f"[Debug] Regex override check error for {name}: {e}", file=sys.stderr)
            return None

        def _infer_assigned_q(name, cls_local):
            """
            Infer what qtype the script/user intended for `name` if the direct original_q
            is missing. Priority:
              1) pre_assignments (explicit user assign / missing-csv pre-assign)
              2) regex overrides (--*-assign-tensors via parsed cpu_regex_assign/gpu_regex_assign)
              3) assignments[cls] (what script computed earlier)
              4) explicit --cpu-assign-qtype / --gpu-assign-qtype (default assigned qtype)
              5) None (no assigned qtype)
            """
            # 1) pre_assignments (explicit user-provided or missing-from-csv pre-assign)
            if name in pre_assignments:
                q = pre_assignments.get(name)
                if DEBUG:
                    print(f"[Debug] Inferred from pre_assignments: {name} -> {q}", file=sys.stderr)
                return q
            # 2) regex override lists (user-provided patterns)
            overridden = _regex_override_for(name, cls_local)
            if overridden:
                return overridden
            # 3) already computed assignments (post-assignment by this tool)
            if name in assignments.get(cls_local, {}):
                q = assignments[cls_local].get(name)
                if DEBUG:
                    print(f"[Debug] Inferred from assignments[{cls_local}]: {name} -> {q}", file=sys.stderr)
                return q
            # 4) the explicit class-level default qtype flags
            if cls_local == 'cpu' and args.cpu_assign_qtype:
                if DEBUG:
                    print(f"[Debug] Inferred class default cpu_assign_qtype for {name} -> {args.cpu_assign_qtype}", file=sys.stderr)
                return args.cpu_assign_qtype
            if cls_local == 'gpu' and args.gpu_assign_qtype:
                if DEBUG:
                    print(f"[Debug] Inferred class default gpu_assign_qtype for {name} -> {args.gpu_assign_qtype}", file=sys.stderr)
                return args.gpu_assign_qtype
            if DEBUG:
                print(f"[Debug] No inferred assignment for {name}", file=sys.stderr)
            return None

        def _basename_of_tensor(name: str) -> str:
            m = re.match(r'^blk\.(\d+)\.(.*)$', name)
            if m:
                return m.group(2)
            return name

        def _format_type_list(names):
            from collections import Counter
            basenames = [_basename_of_tensor(n) for n in names]
            cnt = Counter(basenames)
            items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
            parts = []
            for name, c in items:
                parts.append(f"{c}x {name}" if c != 1 else f"{name}")
            return ", ".join(parts)

        # ---- Process auto-included f32 tensors ----
        if add_f32 and f32_offset.get(cls, 0) > 0:
            print(f"# Auto-included f32 tensors for {cls.upper()}:")
            f32_entries = []
            for name in sorted(f32_names):
                orig_q = 'f32'
                # compute size under final_q and record full entry
                sizes_map, actual_qtypes_map, elements_map = get_map_sizes_and_elements(orig_q)
                size = int(sizes_map.get(name, 0))
                final_q = actual_qtypes_map.get(name, 0)
                elements = int(elements_map.get(name, 0))
                f32_entries.append((name, orig_q, final_q))
                # write back the final qtype into assignments / pre_assignments so later steps use it
                if name in pre_assignments:
                    pre_assignments[name] = final_q
                else:
                    assignments.setdefault(cls, {})[name] = final_q
                record_tensor_index_entry(
                    tensor_index=tensor_index,
                    cls=cls,
                    name=name,
                    orig_q=orig_q,
                    final_q=final_q,
                    size=size,
                    elements=elements,
                    kind='f32',
                )
            # group mismatches and print warnings (same behaviour as before)
            from collections import defaultdict
            pair_to_names = defaultdict(list)
            for name, o, f in f32_entries:
                if o and o != f:
                    pair_to_names[(o, f)].append(name)

            phase_mismatch_count = sum(len(v) for v in pair_to_names.values())
            if INFO:
                print(f"[Info] f32 phase mismatches: {phase_mismatch_count}", file=sys.stderr)

            for (o, f), names_list in pair_to_names.items():
                type_list_str = _format_type_list(names_list)
                print(f"# WARNING - {type_list_str} qtype fallback to {f} from {o}")
            for name, o, f in f32_entries:
                print(f"^{re.escape(name)}$={f}")

        # ---- Process grouped tensors (pre-assigned and measured) ----
        groups = group_tensors(names)
        for base, full in groups.items():
            # pre-assigned entries
            pre_list = sorted((n for n in full if n in pre_assignments), key=lambda n: pre_assignments[n], reverse=True)
            pre_entries = []
            for name in pre_list:
                orig_q = pre_assignments.get(name, '')
                # compute size under final_q and record full entry (kind = 'pre')
                sizes_map, actual_qtypes_map, elements_map = get_map_sizes_and_elements(orig_q)
                size = int(sizes_map.get(name, 0))
                final_q = actual_qtypes_map.get(name, 0)
                elements = int(elements_map.get(name, 0))
                pre_entries.append((name, orig_q, final_q))
                # writeback
                pre_assignments[name] = final_q
                record_tensor_index_entry(
                    tensor_index=tensor_index,
                    cls=cls,
                    name=name,
                    orig_q=orig_q,
                    final_q=final_q,
                    size=size,
                    elements=elements,
                    kind='pre',
                )

            # measured entries
            val_list = sorted((n for n in full if n in values), key=lambda n: values[n], reverse=True)
            val_entries = []
            for name in val_list:
                orig_q = assignments.get(cls, {}).get(name, '')
                if not orig_q:
                    inferred = _infer_assigned_q(name, cls)
                    if inferred:
                        orig_q = inferred
                # compute size under final_q and record full entry (kind = 'measured')
                sizes_map, actual_qtypes_map, elements_map = get_map_sizes_and_elements(orig_q)
                size = int(sizes_map.get(name, 0))
                final_q = actual_qtypes_map.get(name, 0)
                elements = int(elements_map.get(name, 0))
                val_entries.append((name, orig_q, final_q))
                # writeback
                assignments.setdefault(cls, {})[name] = final_q
                record_tensor_index_entry(
                    tensor_index=tensor_index,
                    cls=cls,
                    name=name,
                    orig_q=orig_q,
                    final_q=final_q,
                    size=size,
                    elements=elements,
                    kind='measured',
                )

            # grouped warnings for pre / val phase (unchanged behavior)
            from collections import defaultdict
            pre_pair_to_names = defaultdict(list)
            for name, o, f in pre_entries:
                if o and o != f:
                    pre_pair_to_names[(o, f)].append(name)
            val_pair_to_names = defaultdict(list)
            for name, o, f in val_entries:
                if o and o != f:
                    val_pair_to_names[(o, f)].append(name)

            pre_mismatch = sum(len(v) for v in pre_pair_to_names.values())
            val_mismatch = sum(len(v) for v in val_pair_to_names.values())
            if INFO:
                print(f"[Info] Group '{base}' pre_mismatch={pre_mismatch} val_mismatch={val_mismatch}", file=sys.stderr)

            if pre_entries or val_entries:
                printed_group_header = False
                for entries, pair_to_names in ((pre_entries, pre_pair_to_names), (val_entries, val_pair_to_names)):
                    if entries and not printed_group_header:
                        print(f"# Group: {re.escape(base)}")
                        printed_group_header = True
                    for (o, f), names_list in pair_to_names.items():
                        type_list_str = _format_type_list(names_list)
                        print(f"# WARNING - {type_list_str} qtype fallback to {f} from {o}")
                    for name, o, f in entries:
                        print(f"^{re.escape(name)}$={f if f is not None else ''}")

        if INFO:
            print(f"[Info] Completed verbose assignment inspection for class '{cls}'.", file=sys.stderr)

    # Recompute fallback_corrections from the single canonical registry (avoid double counting)
    fallback_corrections = sum(
        1 for cls in ('cpu', 'gpu') for e in tensor_index.get(cls, [])
        if e.get('orig_q') and e.get('final_q') and e['orig_q'] != e['final_q']
    )

    if DEBUG:
        print(f"[Debug] tensor_index - ", tensor_index, file=sys.stderr)

    # ----------------- SUMMARY: build everything from tensor_index (single source-of-truth) -----------------
    print("\n## Summary of tensor sizes per class")

    # Recompute totals from the registry
    recomputed_totals = {}
    for cls in ['gpu', 'cpu']:
        if cls == 'cpu' and not cpu_quants:
            continue
        if cls == 'gpu' and not gpu_quants:
            continue
        total_bytes = sum(e['size'] for e in tensor_index.get(cls, []))
        recomputed_totals[cls] = total_bytes

    # Print recomputed totals (use existing extremes map for max/min context)
    _tb = 0
    _pct = 0
    for cls, tb in recomputed_totals.items():
        ext = extremes.get(cls, {})
        highest_q = ext.get('highest_q')
        lowest_q = ext.get('lowest_q')
        max_size = ext.get('max_ref', 0) / GIB
        min_size = ext.get('min_ref', 0) / GIB
        # Percentage of max q-size
        pct = (tb / (max_size * GIB)) * 100 if max_size > 0 else 0
        _tb += tb
        _pct += pct
        print(f"#{cls.upper():>4} Total: {tb/GIB:.2f} GiB ({pct:.2f}%) | {max_size:.2f} GiB max, if all were {highest_q} | {min_size:.2f} GiB min, if all were {lowest_q}")

    if cpu_quants and gpu_quants:
        print(f"# GPU+CPU Total: {_tb/GIB:.2f} GiB ({_pct/2:.2f}%)")

    # Summary tensor counts and bits-per-weight per qtype
    print("\n## Summary of tensor counts and bpw per qtype")

    # Build master qtype list: user quants + discovered final_q types + pre_assign orig qtypes
    all_qtypes = []
    if cpu_quants:
        all_qtypes.extend(cpu_quants)
    if gpu_quants:
        all_qtypes.extend(gpu_quants)

    # include final qtypes observed in registry
    for cls in ('cpu', 'gpu'):
        for e in tensor_index.get(cls, []):
            fq = e.get('final_q')
            if fq and fq not in all_qtypes:
                all_qtypes.append(fq)
            oq = e.get('orig_q')
            if oq and oq not in all_qtypes:
                all_qtypes.append(oq)

    # preserve order unique
    seen = set()
    ordered_qtypes = []
    for qt in all_qtypes:
        if qt not in seen:
            seen.add(qt)
            ordered_qtypes.append(qt)

    # Build per-qtype bpw stats from the actual recipe tensors
    recipe_bpw_stats = build_recipe_bpw_stats(tensor_index)
    any_dynamic_bpw = any(v.get('dynamic') for v in recipe_bpw_stats.values())

    for cls in ['gpu', 'cpu']:
        if cls == 'cpu' and not cpu_quants:
            continue # Skip loop if empty quants
        if cls == 'gpu' and not gpu_quants:
            continue # Skip loop if empty quants

        quants_list = gpu_quants if cls == 'gpu' else cpu_quants
        _quants_list = quants_list
        if add_f32:
            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
                quants_list = ['f32'] + _quants_list

        # Section header per class
        if cls == 'cpu':
            print(f"#\n# {cls.upper()}-friendly quants:")
        else:
            print(f"#\n# {cls.upper()}-loaded quants:")
        print(f"# QTYPE\t\tCount\tBPW\tAssigned GiB\t% Assigned\tMax GiB (all)")

        # Prepare the regex-derived assign map for '+' lines (same as before)

        # candidate list: union of user quants and discovered qtypes
        if DEBUG:
            print(f"[Debug] quants_list - ", quants_list, file=sys.stderr)
        if DEBUG:
            print(f"[Debug] _quants_list - ", _quants_list, file=sys.stderr)
        if DEBUG:
            print(f"[Debug] ordered_qtypes - ", ordered_qtypes, file=sys.stderr)

        candidate_quants = list(dict.fromkeys((quants_list or []) + list(ordered_qtypes)))

        # Use the actual recipe bpw when available; fall back to the current get_bpw lookup otherwise.
        def _sort_bpw_for_summary(q):
            stats = recipe_bpw_stats.get(q, {})
            bpw = stats.get('bpw_actual', None)
            if bpw is None or not math.isfinite(float(bpw)):
                try:
                    bpw = get_bpw(q)
                except Exception:
                    bpw = 0
            return float(bpw)

        sorted_quants = sorted(candidate_quants, key=_sort_bpw_for_summary, reverse=True)

        if DEBUG:
            print(f"[Debug] sorted_quants - ", sorted_quants, file=sys.stderr)

        # build quick index for this class by final_q
        by_final = {}
        for e in tensor_index.get(cls, []):
            by_final.setdefault(e['final_q'], []).append(e)

        # '+' section: show user pre-assigned or f32 grouped by qt
        # '*' section: show user fallback grouped by qt
        # ':' section: show user qt with dynamic bpw
        # '!' section: show user qt computed map files
        for qt in sorted_quants:
            stats = recipe_bpw_stats.get(qt, {})
            dynamic_bpw = bool(stats.get('dynamic', False))

            # bpw for this qtype (actual recipe bpw if available, otherwise safe fallback)
            bpw_val = stats.get('bpw_actual', None)
            if bpw_val is None or not math.isfinite(float(bpw_val)):
                try:
                    bpw_val = get_bpw(qt)
                except Exception:
                    bpw_val = 0
            bpw_str = f"{float(bpw_val):.4f}".rstrip('0').rstrip('.')

            # display version of qt
            # keep the existing computed-map marker, and mark dynamic-bpw qtypes with :
            display_qt = qt
            if dynamic_bpw:
                display_qt = f":{display_qt}"
            if qt in COMPUTED_QTYPES:
                display_qt = f"!{display_qt}"

            # all entries in the canonical registry for this class that end up with final_q == qt
            group_entries = [e for e in tensor_index.get(cls, []) if e.get('final_q') == qt]

            # partition by kind
            f32_entries = [e for e in group_entries if e.get('kind') == 'f32']
            pre_entries = [e for e in group_entries if e.get('kind') == 'pre']
            measured_entries = [e for e in group_entries if e.get('kind') == 'measured']

            # helper: split fallback vs non-fallback (orig_q present and different => fallback)
            def split_fb(lst):
                fb = [e for e in lst if e.get('orig_q') and e.get('orig_q') != e.get('final_q')]
                nfb = [e for e in lst if not (e.get('orig_q') and e.get('orig_q') != e.get('final_q'))]
                return nfb, fb

            # split each kind
            pre_nfb, pre_fb = split_fb(pre_entries)
            meas_nfb, meas_fb = split_fb(measured_entries)

            # Compute max_gib context once
            max_gib = total_size_for_quant(subclasses_to_assign.get(cls, []), qt) / GIB

            # 1) f32 entries (displayed with '+' prefix). Always emit line (possibly zero).
            cnt_f32 = len(f32_entries)
            if cnt_f32 > 0:
                bytes_f32 = sum(e['size'] for e in f32_entries)
                gib_f32 = bytes_f32 / GIB
                print(f"# +{display_qt:<10}\t{cnt_f32:<3}\t{bpw_str:<6}\t{gib_f32:>6.2f} GiB\t-\t\t-")

            # 2) pre-assigned non-fallback (+)
            cnt_pre = len(pre_nfb)
            if cnt_pre > 0:
                bytes_pre = sum(e['size'] for e in pre_nfb)
                gib_pre = bytes_pre / GIB
                print(f"# +{display_qt:<10}\t{cnt_pre:<3}\t{bpw_str:<6}\t{gib_pre:>6.2f} GiB\t-\t\t-")

            # 3) pre-assigned fallback (*+)
            cnt_pre_fb = len(pre_fb)
            if cnt_pre_fb > 0:
                bytes_pre_fb = sum(e['size'] for e in pre_fb)
                gib_pre_fb = bytes_pre_fb / GIB
                #pct_pre_fb = (bytes_pre_fb / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# *+{display_qt:<8}\t{cnt_pre_fb:<3}\t{bpw_str:<6}\t{gib_pre_fb:>6.2f} GiB\t-\t\t-")

            # 4) measured fallback (*)
            cnt_meas_fb = len(meas_fb)
            if cnt_meas_fb > 0:
                bytes_meas_fb = sum(e['size'] for e in meas_fb)
                gib_meas_fb = bytes_meas_fb / GIB
                pct_meas_fb = (bytes_meas_fb / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# *{display_qt:<9}\t{cnt_meas_fb:<3}\t{bpw_str:<6}\t{gib_meas_fb:>6.2f} GiB\t{pct_meas_fb:>5.2f}%\t\t{max_gib:.2f}")

            # 5) measured non-fallback (regular, no prefix). Always emit line (possibly zero).
            cnt_meas = len(meas_nfb)
            #if cnt_meas > 0:
            if cnt_meas > 0 or ((cnt_meas == 0 and qt != "f32") and qt in quants_list):
                bytes_meas = sum(e['size'] for e in meas_nfb)
                gib_meas = bytes_meas / GIB
                pct_meas = (bytes_meas / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# {display_qt:<10}\t{cnt_meas:<3}\t{bpw_str:<6}\t{gib_meas:>6.2f} GiB\t{pct_meas:>5.2f}%\t\t{max_gib:.2f}")

    _bytes = sum(e.get('size', 0) for lst in tensor_index.values() for e in lst)
    _elements = sum(e.get('elements', 0) for lst in tensor_index.values() for e in lst)

    print(f"#\n# -Average BPW: {_bytes * 8 / _elements:.4f}")

    total_fallbacks = sum(
        1 for cls in ('cpu','gpu') for e in tensor_index.get(cls, []) if e.get('orig_q') and e.get('final_q') and e['orig_q'] != e['final_q']
    )

    print(f"#\n# -Notes:")
    print("# - '+' means user-defined pre-assigned tensors, or tensor missing from csv data or f32 tensors")
    if total_fallbacks > 0:
        print("# - '*' means fallback tensors: these tensors were present in the map(s) with a different dtype than the originally-intended qtype;")
        print("#   They have been grouped and displayed as '*<qtype>' above to show the final (map-observed) qtype and sizes separately.")
    if any_dynamic_bpw:
        print("# - ':' means this qtype has a tensor-shape-dependent bpw in this recipe due to a discovered additional per-row scale overhead;")
        print("#   For more information: https://github.com/Thireus/GGUF-Tool-Suite/discussions/53")
    if len(COMPUTED_QTYPES) > 0:
        print("# - '!' means qtypes for which the tensors map file was computed instead of downloaded;")
        print("#   This means the tensors assigned to these '!<qtype>' will likely need to be quantized locally as download links may not be available. ")
        print("#   This also means there is alaways a chance that these tensors can't be quantized to their assigned '!<qtype>'.")
    # Conditionally warn user that automatic fallbacks may have changed assignments
    if fallback_corrections > 0:
        print(f"# - WARNING: {fallback_corrections} tensor assignments were substituted to the dtype actually present in their tensor map files. ")
        print("#   This may change the final size relative to the expected thresholds and chosen quants. ")
        print("#   To disable automatic map-based fallbacks and preserve the script's original assigned qtypes exactly, re-run with --no-fallback.")

    now = datetime.now().astimezone()  # Gets local time with tzinfo if available
    current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"# - Recipe produced on the {current_time} using Thireus' GGUF tools (https://gguf.thireus.com/)")
    # Compute SHA-256 of the current script (if readable)
    script_path = sys.argv[0]
    if os.path.isfile(script_path):
        try:
            with open(script_path, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            sha256 = "ERROR"
    else:
        sha256 = "N/A"
    print(f"# - Script SHA-256: {sha256}")
    # Reconstruct a safely quoted command‐line
    quoted_args = [shlex.quote(arg) for arg in sys.argv]
    command_line = ' '.join(quoted_args)
    # Compute SHA-256 of the *_results.csv file (if readable)
    if os.path.isfile(args.csv_file):
        try:
            with open(args.csv_file, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            sha256 = "ERROR"
    else:
        sha256 = "N/A"
    print(f"# - Calibration dataset '{args.csv_file}' SHA-256: {sha256}")

    # Compute SHA-256 for --quant-degradation-csv if provided (improvement #1)
    if args.quant_degradation_csv:
        if os.path.isfile(args.quant_degradation_csv):
            try:
                with open(args.quant_degradation_csv, 'rb') as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                sha256 = "ERROR"
        else:
            sha256 = "N/A"
        print(f"# - Degradation dataset '{args.quant_degradation_csv}' SHA-256: {sha256}")

    def print_maps_sorted_by_bpw(MAP_FILE_INFO: Dict[str, Tuple[str, str, str]]) -> None:
        """
        Print entries from MAP_FILE_INFO sorted by get_bpw(qtype) (desc).
        Each printed block shows BPW (if available), SHA-256 and model name.
        """
        def _map_sort_key(item):
            map_filename, (qtype, _, _) = item
            q_key = _canonical_qtype_key(qtype)
            try:
                bpw = float(get_bpw(qtype))
            except Exception:
                bpw = float("-inf")
            scale_factor = ADDITIONAL_SCALE_FACTOR_TABLE.get(q_key, 0)
            return (bpw, scale_factor, map_filename)

        # Cache bpw for each qtype to avoid repeated calls
        bpw_cache: Dict[str, float] = {}
        for _, (qtype, _, _) in MAP_FILE_INFO.items():
            if qtype in bpw_cache:
                continue
            try:
                # Expect get_bpw to be defined externally
                bpw = get_bpw(qtype)
                # Ensure the returned value is a float or can be converted
                bpw_cache[qtype] = float(bpw) if bpw is not None else float("-inf")
            except Exception:
                # If get_bpw fails for any qtype, place it at the bottom
                bpw_cache[qtype] = float("-inf")

        # Sort items by bpw descending; tie-break by higher additional scale factor first, then filename
        sorted_items = sorted(MAP_FILE_INFO.items(), key=_map_sort_key, reverse=True)

        # Print lines in the requested format, including BPW
        for map_filename, (qtype, sha256sum, model_name) in sorted_items:
            bpw = bpw_cache.get(qtype, float("-inf"))
            bpw_str = f"{bpw:.6g}" if math.isfinite(bpw) else "N/A"
            # If this qtype map was computed, annotate it in output (we already set model_name to include '(computed)')
            print(f"# - {map_filename} SHA-256: {sha256sum}")
            print(f"# - {map_filename} model name: {model_name}")

    print_maps_sorted_by_bpw(MAP_FILE_INFO)

    if not SKIP_GPG:
        if ALL_GPG_SIGS_VALID:
            print(f"# - GPG signatures: PASSED")
        else:
            print(f"# - GPG signatures: FAILED")
    else:
        print(f"# - GPG signatures: DISABLED")

    # List some important parameters that would otherwise be hard to guess (because dynamic or changing over versions or arg-dependant and complex)
    auto_chosen = args.__dict__.get('_auto_chosen_params', {}) or {}
    # Was a greedy 2nd-pass combo auto-selected (not forced by --auto-force-combo)? If so, disclose
    # its number per class so it can be reproduced/overridden via --auto-force-combo.
    _auto_combo = any(v.get('combo') is not None and not v.get('combo_forced')
                      for v in auto_chosen.values())
    if not args.exponential_factor or not args.per_tensor_degradation_scaling or auto_chosen:
        print(f"# - Hidden parameters (not passed as CLI args):")
        if _auto_combo:
            for _cls, _params in auto_chosen.items():
                _cnum = _params.get('combo')
                if _cnum is not None and not _params.get('combo_forced'):
                    _cnm = _params.get('combo_name', '?')
                    print(f"#   Greedy 2nd-pass combo [{_cls.upper()}]: {_cnum} ({_cnm}) "
                          f"— adaptively selected; reproduce/override with --auto-force-combo {_cnum}")
        if not args.exponential_factor:
            if args.use_auto_quant_assign and auto_chosen:
                # The auto method auto-sweeps (p, q). Report the chosen values
                # per class (loss_exponent = p, deg_exponent = q).
                for _cls, _params in auto_chosen.items():
                    p_v = _params.get('loss_exponent')
                    q_v = _params.get('deg_exponent')
                    if p_v is not None:
                        s_p = (f"{p_v:.8f}".rstrip('0').rstrip('.'))
                        print(f"#   Loss exponent (p) [{_cls.upper()}]: {s_p}")
                    if q_v is not None:
                        s_q = (f"{q_v:.8f}".rstrip('0').rstrip('.'))
                        print(f"#   Degradation exponent (q) [{_cls.upper()}]: {s_q}")
            else:
                print(f"#   Exponential factor: {exp_factor_final:.8f}".rstrip('0').rstrip('.'))
        if not args.per_tensor_degradation_scaling and per_tensor_degradation_scaling_final != 0.0:
            print(f"#   Per tensor degradation scaling exponent: {per_tensor_degradation_scaling_final:.8f}".rstrip('0').rstrip('.'))

    # Wrap the command into lines starting with "# "
    wrapped_lines = textwrap.wrap(
        command_line,
        width=115,  # 80 - len("# ") - len(" \\")
        break_long_words=False,
        break_on_hyphens=False
    )
    # Add "# " prefix and " \\" suffix to each line, except the last one
    formatted_lines = [
        f"# {line} \\" if i < len(wrapped_lines) - 1 else f"# {line}"
        for i, line in enumerate(wrapped_lines)
    ]
    print(f"# - Command used:")
    print('\n'.join(formatted_lines))

    if all(tb == 0 for tb in totals.values()):
        print("\n[Warning] All tensor sizes are zero—did you fetch the map files correctly?", file=sys.stderr)

if __name__ == '__main__':
    main()
