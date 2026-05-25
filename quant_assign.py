#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_assign.py the recipe maker tool of choice! Use it   **#
#** to produce recipes that can be cooked and used by others. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: May-25-2026 -------------------- **#
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
            print(f"Error: qtype '{qtype_arg}' not found in CSV.")
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
            "and end with uppercase 'K' (e.g. 'q3_K')."
        )
    # If it matches q…kv (any case) but not exactly q…KV, warn
    if _INSPECT_KV_RE.match(qtype) and not _CANONICAL_KV_RE.match(qtype):
        print(
            f"[Warning] qtype={qtype!r} does not match the canonical pattern r'^q.*KV$'. "
            "Q-types ending with 'KV' must use uppercase 'KV' (e.g. 'q8_KV')."
        )
    # Warn if it's fully capitalized
    if qtype.isupper():
        print(
            f"[Warning] qtype={qtype!r} is fully capitalized. "
            "Q-types are case-sensitive and there are no known quant types that are entirely uppercase."
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
                    print(f"[Warning] failed to fetch tensors.map.sig: {e}")
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
            print(f"Error: Fetching valid map for qtype: {probe} was unsuccessful.")
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


def rank_quant_assign(
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
    tensors = list(tensors)
    tensor_quants = tensor_quants or {}
    extra_outlier_qtypes = list(extra_outlier_qtypes or [])

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
            print(f"[RANK] applying synergistic adjustment (strength={synergy_strength}) to raw loss values", file=sys.stderr)
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
                f"[RANK] outlier (kld≤{zero_kld_threshold}) {t} -> {smallest_q} "
                f"(size={tensor_sizes_w[t][smallest_q]/GIB:.4f} GiB; picked from {candidates})",
                file=sys.stderr,
            )

    # Pre-assigned (no ppl data) -> best qtype in allowed list
    for t in preassigned:
        assignment[t] = allowed_map_w[t][0]
        if debug:
            print(f"[RANK] preassigning {t} -> {allowed_map_w[t][0]} (missing ppl_loss)", file=sys.stderr)

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
            print("[RANK] Applying per-tensor Pareto-frontier filter to allowed qtypes...", file=sys.stderr)
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
                f"[RANK] Pareto filter removed {total_removed} (tensor,qtype) allowed-list entries; "
                f"most-dropped qtypes: {top_dropped}",
                file=sys.stderr,
            )

        # Rebuild global_pool from filtered allowed lists.
        global_pool_set = set()
        for t in sortable:
            global_pool_set.update(allowed_map_w[t])
        global_pool = sorted(global_pool_set, key=_pool_sort_key)
        if debug:
            print(f"[RANK] Pareto-filtered global pool ({len(global_pool)} qtypes): {global_pool}", file=sys.stderr)

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
        print(f"[RANK] initial pool ({len(current_pool)} qtypes): {current_pool}", file=sys.stderr)
        print(f"[RANK] initial total_size = {total_size/GIB:.3f} GiB; budget = {budget_bytes/GIB:.3f} GiB", file=sys.stderr)

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
    p_meta = max(1.0, math.log2(max(1.0, _deg_loss_ratio)))
    if debug:
        print(
            f"[RANK] meta-score: max_loss={_max_loss:.4f}, mean_loss={_mean_loss:.4f}, "
            f"max_pool_deg={_pool_max_deg:.4f}, ratio={_deg_loss_ratio:.3f}; "
            f"p_meta={p_meta:.3f} (weight = (loss + mean_loss)·deg^p_meta); "
            f"large p_meta ⇒ rank-like, p_meta=1 ⇒ greedy-like",
            file=sys.stderr,
        )

    def _meta_score(curr_assignment: Dict[str, str]) -> Tuple[float, float, float]:
        # Primary key: Σ (loss + mean_loss) · deg ** p_meta.
        #
        # The contribution of a tensor t at qtype q has two parts, both
        # weighted by deg(q)^p_meta:
        #
        #   1. A loss-proportional term `loss(t) · deg(q)^p_meta` — this is
        #      the standard "expected damage" model. It rewards spreading
        #      sensitive tensors onto low-deg qtypes and insensitive tensors
        #      onto higher-deg qtypes, because the savings on the sensitive
        #      tensors (high loss × big deg reduction) outweigh the costs
        #      on the insensitive ones (low loss × small deg increase).
        #
        #   2. A flat per-tensor "intrinsic" term `mean_loss · deg(q)^p_meta`
        #      — every tensor pays this just for being on q. This term is
        #      what makes the algorithm avoid catastrophic qtypes even for
        #      low-loss tensors. A bottom-of-pack tensor (loss ≈ 0.005)
        #      contributes essentially nothing under the loss-proportional
        #      term alone, so placing it on iq1_m vs iq2_xs is nearly free —
        #      and greedy happily dumps 199 tensors on iq1_m. Adding the
        #      intrinsic term `mean_loss · deg(iq1_m)^p ≈ 0.15 × 216 ≈ 32`
        #      per tensor means 199 such tensors pay ~6300 extra, which
        #      dwarfs the savings greedy gains by promoting the top tensors.
        #
        # Choice of α = mean_loss: this is the natural data-derived scale
        # for "what a typical tensor's loss looks like." It makes the
        # intrinsic term comparable to the loss-proportional term for a
        # median tensor (so neither dominates), while ensuring that even
        # zero-loss tensors get a meaningful "this qtype is bad" signal.
        # No hardcoded threshold; both terms shrink/grow together with the
        # model's loss distribution.
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
            primary += (loss + _mean_loss) * qweight
            sum_d += d
            if t in high_outliers_set:
                outlier_ld += loss * d
        return (primary, outlier_ld, sum_d)

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
                    f"[RANK] Detected {len(high_outliers)} high-sens outliers; "
                    f"keeping top-3 for decoupled assignment: {high_outliers[:3]}",
                    file=sys.stderr,
                )
            high_outliers = high_outliers[:3]
        elif debug and high_outliers:
            print(f"[RANK] Detected high-sens outliers: {high_outliers}", file=sys.stderr)

    # Rank-mapping should also place outliers (they're still part of sortable),
    # but at brute-force time we override their assignment with a decoupled
    # choice. Build a "non-outlier sortable" view for the score's tie-breaking.
    high_outliers_set = set(high_outliers)

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

    # Pre-compute outlier allowed lists sorted by actual degradation (best→worst).
    outlier_allowed: Dict[str, List[Tuple[str, float, int]]] = {}
    for ot in high_outliers:
        rows: List[Tuple[str, float, int]] = []
        for q in allowed_map_w[ot]:
            try:
                d = degradation_fn(ot, q)
            except Exception:
                d = None
            sz = int(tensor_sizes_w[ot].get(q, 0))
            rows.append((q, float(d) if d is not None else float('inf'), sz))
        rows.sort(key=lambda r: r[1])
        outlier_allowed[ot] = rows

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
                for ot in high_outliers:
                    chosen_q: Optional[str] = None
                    for q_, d_, sz_ in outlier_allowed[ot]:
                        if d_ > window_first_deg:
                            continue
                        if running_total + sz_ <= budget_bytes:
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
        for _k in range(1, len(_pool_by_deg) + 1):
            _sub_pool = set(_pool_by_deg[:_k])
            _sub_tensor_quants = {
                t: [q for q in allowed_map_w[t] if q in _sub_pool]
                for t in tensors_w
            }
            # Skip if any tensor has no allowed qtype in this sub-pool.
            if any(not qs for qs in _sub_tensor_quants.values()):
                continue
            try:
                _g_assignment, _g_total = greedy_quant_assign(
                    tensors=list(tensors_w),
                    tensor_sizes=tensor_sizes_w,
                    ppl_loss=ppl_loss_w,
                    degradation_fn=degradation_fn,
                    tensor_quants=_sub_tensor_quants,
                    budget_bytes=int(budget_bytes),
                    preassign_missing_ppl=preassign_missing_ppl,
                    debug=False,
                    harmonized_groups=None,
                    loss_exponent=1.0,
                )
            except Exception as _e:
                if debug:
                    print(f"[RANK] Greedy candidate k={_k} failed: {_e}", file=sys.stderr)
                continue
            if _g_assignment and _g_total <= budget_bytes:
                # Sentinel window key (-1, _k): _k = number of qtypes in
                # greedy's allowed pool (sorted by deg, best first). _k = K
                # is unconstrained greedy.
                window_data.append(((-1, _k), dict(_g_assignment), int(_g_total)))
                if debug:
                    _excluded = _pool_by_deg[_k:]
                    print(
                        f"[RANK] Added greedy candidate k={_k} (excluded worst-deg: {_excluded}) "
                        f"total={_g_total/GIB:.3f} GiB",
                        file=sys.stderr,
                    )

    # Restrict to in-budget windows for the (p, q) sweep. If none fit, the
    # fallback assignment is used.
    feasible_windows = [(w, a, t_) for (w, a, t_) in window_data if t_ <= budget_bytes]

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
                    f"[RANK] Warning: no window fits budget {budget_bytes/GIB:.3f} GiB. "
                    f"Falling back to smallest-size assignment ({fallback_total/GIB:.3f} GiB).",
                    file=sys.stderr,
                )
            assignment = fallback_assignment or assignment
            total_size = fallback_total if fallback_total >= 0 else 0
            current_pool = list(global_pool)
            best_p = best_q = 1.0
            best_meta = _meta_score(assignment) if assignment else (0.0, 0.0, 0.0)
            best_score = float('inf')
            best_window = (0, K_full - 1)
            unique_results = []
        else:
            # Pick the (p, q) minimising the lexicographic meta-score.
            sweep_results.sort(key=lambda r: r[5])
            best_p, best_q, assignment, total_size, best_score, best_meta, best_window = sweep_results[0]
            if best_window[0] == -1:
                # Greedy candidate won — assignment isn't tied to a contiguous
                # global_pool sub-window. Report the qtypes actually used.
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
                    return "Σ (loss+{:.4f})·deg^{:.2f}={:.4f} outlier_loss·deg={:.4f} sum_deg={:.1f}".format(_mean_loss, p_meta, *m)
                return f"{m:.4f}"
            def _pool_label(w):
                if w[0] == -1:
                    return f"GREEDY(k={w[1]})"
                return str(global_pool[w[0]:w[1] + 1])
            print(
                f"[RANK] Auto-sweep tried {n_candidates} (p,q) candidates"
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
                    f"[RANK] sweep  p={p_:.2f} q={q_:.2f} meta=({_fmt_meta(m_)}) "
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
                    f"[RANK] Warning: no window fits budget {budget_bytes/GIB:.3f} GiB.",
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
                f"[RANK] Best window (p={loss_exponent}, q={deg_exponent}): "
                f"{current_pool}; total={total_size/GIB:.3f} GiB; score={best_score:.6f}",
                file=sys.stderr,
            )

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
    # the rank method already finds the best window within budget via the
    # brute-force search, so Phase C should only fill the leftover headroom
    # rather than spend any of the user's tolerance allowance.)
    headroom_budget = budget_bytes

    if debug:
        print(
            f"[RANK] Phase C: starting promotions "
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
                    f"[RANK] Phase C: promoted {t} {current_q} -> {new_q} "
                    f"(+{delta/GIB:.4f} GiB; total={total_size/GIB:.3f} GiB)",
                    file=sys.stderr,
                )
        if not any_promoted:
            break

    if debug:
        print(
            f"[RANK] Phase C: {promotions} promotions performed. "
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
                        help='Relative GiB tolerance for size optimization')
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
    parser.add_argument('--use-rank-quant-assign', action='store_true',
                        help=('Use the rank-preserving "exhaustive-window" quant assignment instead of the greedy or default spread/midpoint methods. '
                              'The method strictly preserves the sensitivity rank order of tensors (the most-sensitive tensor never gets a worse qtype than a less-sensitive one), '
                              'identifies zero-kld outliers (tensors unused by the model) and assigns them the smallest-size qtype, '
                              'then projects the sensitivity rank onto every contiguous sub-window of the qtype pool (sorted by degradation best→worst) and picks the window that minimizes total Σ loss·deg while fitting the size budget. '
                              'A final promotion pass lifts the smallest tensors first to consume any remaining headroom while still preserving rank monotonicity. '
                              'Designed to work well out-of-the-box without tweaking --exponential-factor / --per-tensor-degradation-scaling / --harmonize-tensors / --synergistic-tensors — those parameters remain available for advanced users. '
                              'Requires --quant-degradation-csv (recommended) or falls back to hardcoded defaults.'))
    parser.add_argument('--rank-no-pareto-filter', action='store_true',
                        help=('Only valid with --use-rank-quant-assign. By default the rank method drops per-tensor allowed qtypes that are Pareto-dominated on the (size, degradation) plane '
                              '— a qtype that is BOTH larger AND more-degrading than another available qtype is never preferable, so filtering it eliminates wasted budget. '
                              'Pass this flag to disable Pareto filtering (e.g. if you deliberately want to use qtypes that look "objectively worse" by group0 stats but behave better at inference time on your specific hardware).'))
    parser.add_argument('--rank-deg-exponent', type=float, default=None,
                        help=('Only valid with --use-rank-quant-assign. Degradation exponent q in the rank method\'s score function Σ loss^p · deg^q. '
                              'q > 1 amplifies the badness of high-degradation qtypes (e.g. iq1_s) and pushes the recipe to avoid using them; q = 1 is the linear regime. '
                              'When neither --exponential-factor (p) nor --rank-deg-exponent (q) are provided, the rank method auto-sweeps a small grid of (p, q) and selects the pair whose chosen assignment minimises the worst per-tensor loss·degradation product — and reports the selection in the recipe\'s hidden-parameters footer. '
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

    # --use-greedy-quant-assign and --use-rank-quant-assign are mutually exclusive
    if args.use_greedy_quant_assign and args.use_rank_quant_assign:
        parser.error("--use-greedy-quant-assign and --use-rank-quant-assign are mutually exclusive")

    # Convenience: a single flag indicating whether ANY degradation-aware
    # assignment method is in use. The same auxiliary options (--quant-degradation-csv,
    # --synergistic-tensors, --per-tensor-degradation-scaling) apply to both.
    using_degradation_method = bool(args.use_greedy_quant_assign or args.use_rank_quant_assign)

    # Enforce: --quant-degradation-csv only valid with greedy/rank methods
    if args.quant_degradation_csv and not using_degradation_method:
        parser.error("--quant-degradation-csv may only be used with --use-greedy-quant-assign or --use-rank-quant-assign")

    # Enforce: --synergistic-tensors only valid with greedy/rank methods
    if args.synergistic_tensors and not using_degradation_method:
        parser.error("--synergistic-tensors may only be used with --use-greedy-quant-assign or --use-rank-quant-assign")

    # Enforce: --synergy-strength is only valid when using --synergistic-tensors
    if args.synergy_strength and args.synergy_strength > 0 and not args.synergistic_tensors:
        parser.error("--synergy-strength may only be used with --synergistic-tensors")

    # Enforce: --per-tensor-degradation-scaling only valid with greedy/rank methods
    if args.per_tensor_degradation_scaling and args.per_tensor_degradation_scaling > 0 and not using_degradation_method:
        parser.error("--per-tensor-degradation-scaling may only be used with --use-greedy-quant-assign or --use-rank-quant-assign")

    # Enforce: --rank-no-pareto-filter only valid with --use-rank-quant-assign
    if args.rank_no_pareto_filter and not args.use_rank_quant_assign:
        parser.error("--rank-no-pareto-filter may only be used with --use-rank-quant-assign")

    # Enforce: --rank-deg-exponent only valid with --use-rank-quant-assign
    if args.rank_deg_exponent is not None and not args.use_rank_quant_assign:
        parser.error("--rank-deg-exponent may only be used with --use-rank-quant-assign")
    
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
        print("Error: CSV must have 'QTYPE' as first column.")
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
    if not (args.use_greedy_quant_assign or args.use_rank_quant_assign):
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
                print(f"[Warning] Multiple rows with QTYPE == {row['QTYPE']!r}; updating the first match.")
            idx = matches[0]

        # Update columns one-by-one (avoids type-checker issues and is explicit)
        for col in cols_to_update:
            # row[col] might be a numpy scalar or python scalar — both are fine
            df.at[cast(Any, idx), col] = row[col]

    # ---- END harmonization ----
    #print(row.to_string(max_rows=None))

    # Pre-fetch maps
    if not fetch_map_for_qtype(qtype):
        print(f"Error: Fetching valid map for qtype: {qtype} was unsuccessful.")
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
        if args.use_rank_quant_assign:
            # Rank method scores assignments by Σ loss^p · deg^q. When the user
            # leaves --exponential-factor unset, we run an internal auto-sweep
            # over a small (p, q) grid inside rank_quant_assign and pick the
            # pair whose chosen assignment minimises the worst per-tensor
            # loss·degradation product. exp_factor_final is left at 1.0 here
            # purely as a fallback / display value — the real chosen p will
            # come back via chosen_params_out.
            if INFO:
                print(f"[Info] --exponential-factor not specified for --use-rank-quant-assign: enabling internal (p, q) auto-sweep.", file=sys.stderr)
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
                print(f"Error: could not parse numeric value for tensor {name!r}: {raw!r}")
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

        if not (args.use_greedy_quant_assign or args.use_rank_quant_assign):
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
            if args.use_greedy_quant_assign or args.use_rank_quant_assign:
                # Build tensor_quants mapping (default to cls-specific quants)
                tensor_quants_local = {n: (gpu_quants if cls == 'gpu' else cpu_quants) for n in names_to_assign}

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

                # Build the per-tensor sizes mapping used by both methods
                _quants_for_class = gpu_quants if cls == 'gpu' else cpu_quants
                tensor_sizes_local = {
                    n: {q: get_map_sizes_and_elements(q)[0].get(n, 0)
                        for q in _quants_for_class}
                    for n in names_to_assign
                }

                if args.use_rank_quant_assign:
                    # The rank method also benefits from being told about the
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

                    # The rank method uses *label* degradation (the tabulated
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

                    # Auto-sweep (p, q) inside the rank method ONLY when the
                    # user hasn't pinned both exponents. If --exponential-factor
                    # is set, the user has expressed a preference for p; if
                    # --rank-deg-exponent is also set, both are pinned and we
                    # skip the sweep. Otherwise we sweep and report the chosen
                    # (p, q) in the recipe's hidden-parameters footer.
                    _auto_sweep = (args.exponential_factor is None)
                    _user_q = args.rank_deg_exponent if args.rank_deg_exponent is not None else 1.0
                    if args.rank_deg_exponent is not None:
                        # User pinned q — still skip sweep only when p is also
                        # pinned (avoid surprising behaviour).
                        _auto_sweep = _auto_sweep and (args.exponential_factor is None and False)
                    chosen_rank_params: Dict[str, float] = {}
                    assignment, total_bytes = rank_quant_assign(
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
                        pareto_filter=not args.rank_no_pareto_filter,
                        chosen_params_out=chosen_rank_params,
                    )
                    # Stash the chosen (p, q) onto args so the hidden-parameter
                    # footer can include them in the recipe.
                    if chosen_rank_params:
                        args.__dict__.setdefault('_rank_chosen_params', {})[cls] = chosen_rank_params
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
        print(f"#{cls.upper():>4} Total: {tb/GIB:.2f} GiB ({pct:.1f}%) | {max_size:.2f} GiB max, if all were {highest_q} | {min_size:.2f} GiB min, if all were {lowest_q}")

    if cpu_quants and gpu_quants:
        print(f"# GPU+CPU Total: {_tb/GIB:.2f} GiB ({_pct/2:.1f}%)")

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
                print(f"# *{display_qt:<9}\t{cnt_meas_fb:<3}\t{bpw_str:<6}\t{gib_meas_fb:>6.2f} GiB\t{pct_meas_fb:>3.1f}%\t\t{max_gib:.2f}")

            # 5) measured non-fallback (regular, no prefix). Always emit line (possibly zero).
            cnt_meas = len(meas_nfb)
            #if cnt_meas > 0:
            if cnt_meas > 0 or ((cnt_meas == 0 and qt != "f32") and qt in quants_list):
                bytes_meas = sum(e['size'] for e in meas_nfb)
                gib_meas = bytes_meas / GIB
                pct_meas = (bytes_meas / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# {display_qt:<10}\t{cnt_meas:<3}\t{bpw_str:<6}\t{gib_meas:>6.2f} GiB\t{pct_meas:>3.1f}%\t\t{max_gib:.2f}")

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
    rank_chosen = args.__dict__.get('_rank_chosen_params', {}) or {}
    if not args.exponential_factor or not args.per_tensor_degradation_scaling or rank_chosen:
        print(f"# - Hidden parameters (not passed as CLI args):")
        if not args.exponential_factor:
            if args.use_rank_quant_assign and rank_chosen:
                # The rank method auto-sweeps (p, q). Report the chosen values
                # per class (loss_exponent = p, deg_exponent = q).
                for _cls, _params in rank_chosen.items():
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
        print("\n[Warning] All tensor sizes are zero—did you fetch the map files correctly?")

if __name__ == '__main__':
    main()
