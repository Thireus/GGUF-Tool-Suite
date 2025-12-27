#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_assign.py the recipe maker tool of choice! Use it   **#
#** to produce recipes that can be cooked and used by others. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Dec-27-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.          Zₑᵣₒ₋ₛₕₒₜ, 𝒻ᵤₗₗ ₙₒₙₛₑₙₛₑ **#
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
from typing import Any, cast, Dict, List, Iterable, Tuple, Optional
import heapq

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

# Cache bpw observed per qtype whenever a tensors map file is processed
QTYPE_BPW_CACHE = {}

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

# Verbosity flags
DEBUG = False
INFO = False

# Constants
GIB = 1024**3 # for GiB-to-bytes conversion
STRETCH_MIN = 1.0
STRETCH_MAX = 10.0
STRETCH_STEP = 0.01

# ─── Create a unique temp‐dir at script launch ────────────────────────────────
# This will give you something like /tmp/gguf.thireus.com.ab12cd
TMP_DIR = tempfile.mkdtemp(prefix="gguf.thireus.com.", dir=tempfile.gettempdir())
if DEBUG: print(f"[Debug] Using temp directory: {TMP_DIR}")

# Optionally, register cleanup at exit
import atexit
import shutil

def _cleanup_tempdir(path=TMP_DIR):
    try:
        shutil.rmtree(path)
        if DEBUG: print(f"[Debug] Cleaned up temp directory: {path}")
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
    """
    df = pd.read_csv(path)
    # Expect columns: "QTYPE" and "group0"
    degradation_factors = {}
    for _, row in df.iterrows():
        q = str(row["QTYPE"]).strip()
        val = row["group0"]
        # e.g. " +2.12%" or "0%" or absolute values like "0.0212"
        if pd.isna(val):
            continue
        s = str(val).strip()
        # detect percent form
        is_percent = s.endswith('%')
        if is_percent:
            s = s[:-1].strip()
        # strip optional leading sign
        if s.startswith('+'):
            s = s[1:].strip()
        try:
            f = float(s)
        except ValueError:
            continue
        # if it was a percent string, convert to fraction; otherwise treat as absolute
        if is_percent:
            f = f / 100.0
        degradation_factors[q] = f
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
        print(f"[Info] Calling normalised_ppl.py for keys: {keys_list}")
    # Compose command: include 1 and 32 as edge values
    bpw_args = ['1'] + [str(k) for k in keys_list] + ['32']
    cmd = ['python', script_path, '--bpw-list'] + bpw_args
    if DEBUG:
        print(f"[Debug] Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        if DEBUG:
            print(f"[Debug] normalised_ppl.py output:\n{output}")
    except Exception as e:
        if INFO:
            print(f"[Warning] normalised_ppl.py call failed: {e}")
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


@functools.lru_cache(maxsize=None)
def get_bpw(qtype):
    """
    Return the bpw for a given qtype.
    """
    # infer bits-per-weight from data instead of hardcoding
    if qtype not in QTYPE_BPW_CACHE:
        _, _, _ = get_map_sizes_and_elements(qtype)
    return QTYPE_BPW_CACHE[qtype]

@functools.lru_cache(maxsize=None)
def get_default_factor(qtype):
    """
    Return reducing factor based on bit-width.
    Attempts to fetch a better factor using normalised_ppl.py, falling back to DEFAULT_REDUCE.
    Results are cached per bpw.
    """
    bpw = get_bpw(qtype)
    try:
        if INFO:
            print(f"[Info] bpw for qtype {qtype}: {bpw}")
        key = bpw
    except Exception:
        if DEBUG:
            print(f"[Debug] Could not parse bpw from qtype '{qtype}', returning 1.0")
        return 1.0

    # fallback default
    default_value = DEFAULT_REDUCE.get(int(key), 1.0)

    # return cached if available
    if bpw in _factor_cache:
        if DEBUG:
            print(f"[Debug] Returning cached factor for bpw {bpw}: {_factor_cache[bpw]}")
        return _factor_cache[bpw]

    # try to fetch from script for this single key
    try:
        fetched = _call_normalised_ppl([bpw])
        factor = fetched.get(bpw, default_value)
    except Exception:
        factor = default_value
    else:
        if DEBUG:
            print(f"[Debug] Caching factor for bpw {bpw}: {factor}")
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
    Select the row for given QTYPE or lowest quant.
    """
    if qtype_arg:
        if qtype_arg not in df['QTYPE'].values:
            print(f"Error: qtype '{qtype_arg}' not found in CSV.")
            sys.exit(1)
        return df[df['QTYPE'] == qtype_arg].iloc[0]
    df['__quant_num__'] = df['QTYPE'].map(extract_quant_num)
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
def fetch_map_for_qtype(qtype: str):
    """
    Fetch and cache tensors.{qtype}.map via tensor_downloader.sh.
    """
    global ALL_GPG_SIGS_VALID, MAP_FILE_INFO, SIG_FILE_HASHES
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
    if INFO: print(f"[Info] Fetching map for {qtype}...")
    try:
        if DEBUG or INFO:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if INFO: print(f"[Info] Saved map to {local_map}")
        if not SKIP_GPG:
            cmd_sig = ["bash", tensor_downloader, qtype.upper(), "-1", TMP_DIR, f"tensors.{qtype}.map.sig"]
            if INFO: print(f"[Info] Fetching map gpg signature for {qtype}...")
            try:
                if DEBUG or INFO:
                    subprocess.run(cmd_sig, check=True)
                else:
                    subprocess.run(cmd_sig, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if DEBUG: print(f"[Debug] Saved map gpg signature to {local_map}.sig")
                if not verify_detached_signature(local_map):
                    print(f"[Error] gpg signature verification of tensors.{qtype}.map failed.", file=sys.stderr)
                    ALL_GPG_SIGS_VALID = False
                    return False
                else:
                    if INFO: print(f"[Info] gpg signature of tensors.{qtype}.map succesful.")
            except subprocess.CalledProcessError as e:
                print(f"[Warning] failed to fetch tensors.map.sig: {e}")
                ALL_GPG_SIGS_VALID = False
                return False
        else:
            if INFO: print(f"[Warning] gpg signature verification is disabled and won't be checked for {local_map}.sig")

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
        if not SKIP_GPG:
            sig_key = f"tensors.{qtype}.map.sig"
            sig_path = f"{local_map}.sig"
            if sig_key not in SIG_FILE_HASHES:
                with open(sig_path, 'rb') as f_sig:
                    sha256sig = hashlib.sha256(f_sig.read()).hexdigest()
                SIG_FILE_HASHES[sig_key] = sha256sig

        return True
    except subprocess.CalledProcessError as e:
        print(f"[Warning] failed to fetch tensors.map: {e}")
        return False


@functools.lru_cache(maxsize=None)
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

def parse_map_file(qtype, collect_raw = False):
    """
    Parse local tensors.{qtype}.map into:
      - sizes: dict tensor_name -> bytes_size
      - actual_qtypes: dict tensor_name -> dtype (e.g., 'bf16', 'f32', 'q8_0', ...)
      - elements: dict tensor_name -> elements
    """
    probe = 'bf16' if qtype == 'f32' else qtype
    path = os.path.join(TMP_DIR, f"tensors.{probe}.map")
    sizes = {}
    actual_qtypes = {}
    elements = {}
    if not os.path.exists(path):
        return sizes, actual_qtypes, elements

    with open(path) as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 5:
                continue
            # parts example:
            # [file, checksum, tensor_name, shape=..., dtype=f32, elements=..., bytes=...]
            tensor_name = parts[2]
            # find dtype, bytes, elements fields
            dtype = None
            size_bytes = None
            elems = None
            for p in parts:
                if p.startswith('dtype='):
                    dtype = transform_q_suffix(p.split('=', 1)[1]) # Ensures q..K and q..KV
                elif p.startswith('bytes='):
                    size_bytes = int(p.split('=', 1)[1])
                elif p.startswith('elements='):
                    elems = int(p.split('=', 1)[1])
                if dtype and size_bytes and elems and dtype not in QTYPE_BPW_CACHE:
                    QTYPE_BPW_CACHE[dtype] = size_bytes * 8 / elems
            if dtype is None or size_bytes is None or elems is None:
                # skip incomplete lines
                continue

            sizes[tensor_name] = size_bytes
            actual_qtypes[tensor_name] = dtype
            elements[tensor_name] = elems

    # If NO_FALLBACK requested, synthesize faked sizes/dtypes for mismatching tensors
    if NO_FALLBACK and not collect_raw:
        for t in actual_qtypes:
            if actual_qtypes[t] != qtype:
                if INFO:
                    print(f"[Info] --no-fallback: Enforcing {qtype} qtype for {t} instead of fallback {actual_qtypes[t]} dtype present in tensors map file.")
                actual_qtypes[t] = qtype
                sizes[t] = int(round(elements[t] * (QTYPE_BPW_CACHE[qtype] / 8)))

    return sizes, actual_qtypes, elements

def load_sample_ppl_table(path):
    """
    Load sample PPL CSV and compute reduction factors per base name.
    """
    sample_df = pd.read_csv(path, index_col=0)
    sample_df = sample_df.replace(['404','404.0'], np.nan)
    dropped = [c for c in sample_df.columns if sample_df[c].isna().any()]
    if dropped and INFO:
        print(f"[Info] Dropping sample PPL columns with missing values: {dropped}")
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
        if DEBUG: print(f"[Debug] Forced midpoint: {mid:.4f}")
    else:
        mid = np.mean(list(class_values.values()))
        if DEBUG: print(f"[Debug] Class midpoint (mean PPL): {mid:.4f}")
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
        if DEBUG: print(f"[Debug] Tensor {name}: PPL={ppl:.4f}, spread={spread:.4f}")
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
            print(f"[Debug] Quant {q} @stretch={stretch:.2f}: interval ({bottom:.4f}, {top:.4f}]")
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
        print(f"[Info] Performing spread-based quant assignment (stretch={stretch:.2f})...")
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
                        print(f"[Warning] skipping harmonization for group {matched_group_idx} quant {q_assigned} because pattern match counts differ (counts={lengths}); using per-tensor sizes.")
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
                                print(f"[Warning] {name!r} not found among pattern matches for harmonize group {matched_group_idx}; skipping harmonization for this tensor.")
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
                                    print(f"[Info] Size-harmonized group {matched_group_idx} quant {q_assigned} index {idx_in}: {details} -> harmonized={size_harmonized}")

        sizes[name] = final_size

        if INFO:
            print(f"[Info] Assigned {assignment[name]} to {name} (spread={spreads[name]:.4f}) size={sizes[name]}")

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
                print(f"[WARN] No common quant types found for group {group}, skipping.")
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
            print(f"[SYNERGY] Group {group}")
            print(f"  chosen_quant={chosen_quant}, weighted_avg_loss={weighted_avg_loss:.6f}")

        # interpolate between original and group average
        for t in group:
            if t not in loss:
                continue
            orig_loss = loss[t]
            new_loss = (1 - strength) * orig_loss + strength * weighted_avg_loss
            adjusted_loss[t] = new_loss
            if debug:
                print(f"  {t}: {orig_loss:.6f} -> {new_loss:.6f}")

    return adjusted_loss

def greedy_quant_assign(
    tensors: Iterable[str],
    tensor_sizes: Dict[str, Dict[str, int]],
    ppl_loss: Dict[str, float],
    degradation_factors: Dict[str, float],
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
    - degradation_factors[qtype] -> numeric factor (bigger means worse quant)
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

        # Filter allowed to qtypes that have sizes and degradation_factors
        filtered = []
        for q in allowed:
            if q not in degradation_factors:
                # skip qtypes with no degradation factor (can't evaluate)
                continue
            if q not in tensor_sizes.get(t, {}):
                # skip qtypes with no size info for this tensor
                continue
            filtered.append(q)
        if not filtered:
            raise ValueError(f"No usable qtypes for tensor '{t}' after filtering (allowed: {allowed}).")

        # Sort descending by size (largest first) so index 0 is highest-quality (largest bytes)
        filtered.sort(key=lambda q: tensor_sizes[t][q], reverse=True)
        allowed_map[t] = filtered

    # --- Step 1a: Apply exponent scaling to all loss values (do not mutate original)
    ppl_loss_exp: Dict[str, float] = {}
    for t, v in ppl_loss.items():
        try:
            ppl_loss_exp[t] = float(v) ** float(loss_exponent)
        except Exception:
            # fallback to original float value
            ppl_loss_exp[t] = float(v)

    # --- Step 1b: Apply synergistic adjustment if requested ---
    if synergistic_groups and synergy_strength > 0.0:
        if debug:
            print(f"[GREEDY] applying synergistic adjustment (strength={synergy_strength}) to loss values")
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
        print(f"[GREEDY] initial total_size = {total_size / GIB:.3f} GiB; budget = {budget_bytes / GIB:.3f} GiB")

    # --- prepare heap with (score, counter, tensor, from_q, to_q)
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
        # consider all smaller-quality quants (larger index)
        for to_q in allowed[from_idx + 1:]:
            size_from = int(tensor_sizes[tensor][from_q])
            size_to = int(tensor_sizes[tensor][to_q])
            delta_size = size_from - size_to
            if delta_size <= 0:
                continue
            # use exponentiated/adjusted loss
            loss = ppl_loss_exp.get(tensor, None)
            if loss is None:
                # If no ppl data and we didn't preassign, skip (safety)
                continue
            delta_deg = loss * (degradation_factors[to_q] - degradation_factors[from_q])
            # Avoid division by zero, but delta_size>0 ensures denominator positive
            score = float(delta_deg) / float(delta_size)
            # push (score, counter) so heap is deterministic on ties
            heapq.heappush(pq, (score, counter, tensor, from_q, to_q))
            counter += 1

    # initialize moves from top quant for every tensor
    for t in tensors:
        push_moves(t, assignment[t])

    # --- main loop (downgrade until within budget)
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
            print(f"[GREEDY] downgraded {tensor}: {from_q} -> {to_q}; saved {(size_from-size_to)/GIB:.3f} GiB; new total {(total_size)/GIB:.3f} GiB")

        # push next possible moves for this tensor (relative to its new quant)
        push_moves(tensor, to_q)

    if debug:
        print(f"[GREEDY] final total_size = {total_size / GIB:.3f} GiB")

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
        for to_q in reversed(allowed[:from_idx]):
            size_from = int(tensor_sizes[tensor][from_q])
            size_to = int(tensor_sizes[tensor][to_q])
            delta_size = size_to - size_from
            if delta_size <= 0:
                continue
            loss = ppl_loss_exp.get(tensor, 0.0)
            delta_deg = loss * (degradation_factors[from_q] - degradation_factors[to_q])
            score = float(delta_deg) / float(delta_size)
            # push as max-heap (invert score)
            heapq.heappush(promote_pq, (-score, counter, tensor, from_q, to_q))
            counter += 1

    # initialize promotion opportunities
    for t in tensors:
        push_promotions(t, assignment[t])

    if debug:
        print(f"[GREEDY] starting promotion phase (headroom = {(budget_bytes - total_size)/GIB:.3f} GiB)")

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
            print(f"[GREEDY] promoted {tensor}: {from_q} -> {to_q}; added {(size_to - size_from)/GIB:.3f} GiB; total = {total_size/GIB:.3f} GiB")

        # push next possible promotion for this tensor
        push_promotions(tensor, to_q)

    if debug:
        print(f"[GREEDY] promotion phase done; final total_size = {total_size / GIB:.3f} GiB")

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



def optimize_midpoint_and_assign(quants, _, class_values,
                                 max_bytes, tolerance=0.05, exp_factor=1.0, harmonize_groups=None):
    """
    Loop over stretch factors and perform midpoint optimization using class mean with dichotomy.
    exp_factor controls exponent in stretch calculation: higher = more aggressive extremes.
    """
    if INFO:
        print(f"[Info] Starting optimization for target size {max_bytes} bytes ±{tolerance*100}% with exp_factor={exp_factor:.2f}...")
    best_assign, best_size = {}, float('inf')
    # compute initial midpoint as class mean
    class_mid = compute_class_midpoint(class_values)
    # outer loop: stretch factor sweep
    stretch = STRETCH_MIN
    while stretch <= STRETCH_MAX:
        if INFO and stretch > STRETCH_MIN:
            print(f"[Info] Trying stretch factor {stretch:.2f}...")
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
            print(f"[Info] Progress: {stretch/STRETCH_MAX*100:.2f}%")
        # inner loop: dichotomy until converged
        while (prev_mid == None or prev_mid > mid_min_threshold) and (change == None or change >= change_min_threshold):
            if INFO:
                print(f"[Info] Evaluating midpoint={mid:.6f}, stretch={stretch:.2f}...")
            assignment, sizes = assign_quants(quants, None,
                                             class_values,
                                             forced_mid=mid, stretch=stretch, harmonize_groups=harmonize_groups)
            size = sum(sizes.values())
            # tolerance check
            if abs(size - max_bytes) / max_bytes <= tolerance:
                if INFO:
                    print(f"[Info] Found acceptable size {size} at midpoint={mid:.6f}, stretch={stretch:.2f}.")
                return assignment, size
            # check midpoint change
            if prev_mid is not None:
                change = abs(mid - prev_mid) / prev_mid
                if change < change_min_threshold:  # less than 0.01%
                    if INFO:
                        print(f"[Info] Midpoint change {change*100:.4f}% below threshold; breaking inner loop.")
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
                print(f"[Info] Size {size} is {reason}; moving midpoint {direction}.")
            # compute next midpoint by dichotomy
            mid = (low_val + high_val) / 2
            # track best
            if abs(size - max_bytes) < abs(best_size - max_bytes):
                best_size, best_assign = size, assignment.copy()
        # increment stretch factor
        stretch = round(stretch + STRETCH_STEP, 2)
    if INFO:
        print("[Warning] Optimization finished; using best found assignment.")
    return best_assign, best_size

def scale_for_size(assignment, sizes, quants, max_size_bytes):
    """
    Fallback simple scaling if optimized assignment not used.
    """
    total = sum(sizes.values())
    if INFO: print(f"[Info] Starting fallback scaling: current total {total}, target {max_size_bytes}")
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
                if INFO: print(f"[Info] Scaling {name} from {q} to {new_q}, new total {total}")
                if total <= max_size_bytes:
                    return assignment, total
        if not made_change:
            if INFO: print("[Warning] Cannot reduce size further via fallback scaling.")
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
    # Resolve ultimate default
    if default_qtype:
        base_q = default_qtype
    else:
        base_q = max(quants, key=get_bpw)

    out = {}
    for name in names:
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
                print(f"[Info] Harmonized group {gi} layer {lid_str}: {tuple_names} -> {new_val}")

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
                print(f"[Info] Skipping harmonize group {gi}: invalid regex in {group}")
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
                print(f"[Info] Skipping harmonize group {gi}: pattern match counts differ {lengths}")
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
                print(f"[Info] Skipping harmonize group {gi}: inconsistent id presence across matches")
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
            print(f"[Info] {arg_name} disabled by the user")
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

        # List-of-lists directly
        elif all(isinstance(elem, list) for elem in arg_value):
            groups = arg_value

        # List of comma-separated strings
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


def main():
    global DEBUG, INFO, SKIP_GPG, ALL_GPG_SIGS_VALID, NO_FALLBACK
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
    parser.add_argument('--qtype', help='Case-sensitive qtype (e.g. q3_K) to analyze from the calibration data CSV file (default: lowest quant)')
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
    parser.add_argument('--exponential-factor', type=float, default=1.0,
                        help='Exponent controlling midpoint adjustment aggressiveness during stretch sweeps for default quant assignment method. Higher values push quantization toward extremes; default is 1.0. When using --use-greedy-quant-assign, the exponent is used to try to map per-tensor degradation values into a additively linear space. Recommended range for greedy quant assign is 1.0 to 5.0 when using KLD metrics with 3.0 being a good starting point.')
    parser.add_argument('--quant-degradation-csv', type=str, help='Path to CSV file containing quant degradation values for use by greed quant assing method (overrides bundled quants_graphs/kld_results.csv)')
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
                            "Future versions of ik_llama.cpp may also take advantage of fused ffn_up_shexp and ffn_gate_shexp tensors. " ) )
    parser.add_argument('--harmonization-technique', type=int, default=3, choices=[0,1,2,3],
                        help=('Harmonization technique to use when --harmonize-tensors is set: 0=disabled, 1=max, 2=mean, 3=min (default). ' 
                            'Values are applied element-wise per layer across the matched tensors.'
                            'Max ensures calibration data measurement is not negatively degraded. Min will degrade calibration data accuracy but appears to give the best results. Mean is a compromise in-between. Disabled means harmonization is disabled.'))
    parser.add_argument('--use-greedy-quant-assign', action='store_true', help='Use greedy priority-queue quant assignment instead of default spread/midpoint method. The method tries to minimize overall degradation by prioritizing quant downgrades that yield the least degradation per byte saved. This method requires per-tensor degradation data (e.g. KLD) to be present in the CSV file - perplexity data only works suboptimally. It also requires per quant type degradation estimates; per default KLD values are used from a qwen 3 4b benchmark. override with --quant-degradation-csv. It is recommended to use --exponential-factor between 1.0 and 5.0 when using this method to try to map per-tensor degradation values into a more linear space.')
    parser.add_argument(
        '--synergistic-tensors',
        nargs='+',
        default=[["blk\\..*\\.ffn_up_exps.*","blk\\..*\\.ffn_gate_exps.*","blk\\..*\\.ffn_down_exps.*"]],
        help=(
            'A Python literal list-of-lists of regex patterns. Each inner list defines tensors that '
            'exhibit synergistic effects and should have their loss adjusted together. '
            'Example: --synergistic-tensors blk\\..\\*\\.ffn_up_exps.\\*,blk\\..\\*\\.ffn_gate_exps.\\*,blk\\..\\*\\.ffn_down_exps.\\* '
            "'another_pat1,another_pat2'. "
            'Use --synergistic-tensors "" to disable synergy adjustment. '
            'Note: synergy encourages similar quantization within each layer, '
            'typically improving quality without strictly enforcing identical qtypes.'
        )
    )
    parser.add_argument('--synergy-strength',type=float,default=0.0,help='Strength of synergy-based loss adjustment (0 = disabled, 1 = fully averaged losses). Default: 0')
    parser.add_argument('--no-fallback', action='store_true',
                        help=('Disable automatic fallback checks: do NOT attempt to inspect map files to detect per-tensor dtype mismatches. '
                              'When set, the script will act as if the quantized tensors of the map files were pure and any tensor mismatching the quant type will have its size "guessed" as if it had been quantized to that qtype.'))
    args = parser.parse_args()

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

    here = os.path.dirname(os.path.abspath(__file__))
    default_quant_degradation_path = os.path.join(here, "quants_graphs", "kld_results.csv")
    quant_degradation_path = args.degradation_csv if getattr(args, 'quant_degradation_csv', None) else default_quant_degradation_path
    quant_degradation_values = load_quant_degradation_values(quant_degradation_path)

    if INFO:
        print(f"[Info] Loaded degradation values for ({len(quant_degradation_values)} quant types)")

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
        cpu_quants = sorted(cpu_quants, key=get_bpw, reverse=True)
        if INFO: print(f"[Info] CPU-friendly quants reordered by bpw: {cpu_quants}")
    except Exception:
        pass

    gpu_quants = args.gpu_quants
    # By default we assume the user wants everything on the GPU
    if not cpu_quants and not gpu_quants:
        if INFO: print(f"[Info] No quants selected, reverting to GPU default selection: {DEFAULT_QUANTS}")
        gpu_quants = DEFAULT_QUANTS
    # if not gpu_quants and args.cpu_quants:
    #     gpu_quants = DEFAULT_QUANTS
    # Reorder gpu_quants from highest to lowest bpw
    try:
        gpu_quants = sorted(gpu_quants, key=get_bpw, reverse=True)
        if INFO: print(f"[Info] GPU-friendly quants reordered by bpw: {gpu_quants}")
    except Exception:
        pass

    if INFO: print(f"[Info] Loading CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    if 'QTYPE' not in df.columns:
        print("Error: CSV must have 'QTYPE' as first column.")
        sys.exit(1)

    #reduction_factors = load_sample_ppl_table(args.sample_ppl)
    row = select_qtype(df, args.qtype)
    qtype = row['QTYPE']
    if INFO: print(f"[Info] Selected QTYPE: {qtype}")

    # ---- NEW: Parse synergistic tensor groupps into per layer groups----
    synergistic_groups = parse_group_argument(args.synergistic_tensors, "--synergistic-tensors", parser, info_flag=INFO)

    #print(row.to_string(max_rows=None))
    # ---- NEW: Harmonize matching tensor rows ----
    # Convert nargs='+' form (list of comma-separated strings) into list-of-lists
    harmonize_groups = []
    if args.harmonization_technique != 0:
        harmonize_groups = parse_group_argument(args.harmonize_tensors, "--harmonize-tensors", parser, info_flag=INFO)

    # harmonize_groups is now a list-of-lists of regex strings (or empty list to disable)

    # --- Disable harmonization here when using greedy quant assign. for greedy quant assign the harmonization is handled inside greedy_quant_assign itself ---
    if not args.use_greedy_quant_assign:
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

    # Collect tensor names (either from csv or from map file)
    if INFO: print(f"[Info] Get all tensor names")
    if args.tensors_from_csv:
        tensor_names = [c for c in df.columns if c != 'QTYPE']
    else:
        tensor_names = [n for n,d in _items.items()]

    # Identify all f32 tensors once
    if INFO: print(f"[Info] Get f32 tensor names")
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
            if INFO: print(f"[Info] CPU-friendly quants skipped because not being user-specified.")
            continue # Skip loop if empty quants
        if cls == 'gpu' and not gpu_quants:
            if INFO: print(f"[Info] GPU-friendly quants skipped because not being user-specified.")
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
                if INFO: print(f"[Info] Assigning {name!r} → {pre_assignments[name]!r} (in --cpu/gpu-assign-tensors parameter)")

                # jump to next tensor
                continue
            # missing measurement → pre-assign
            elif name not in row or pd.isna(row.at[name]):
                if name in f32_names:
                    # This is a f32 tensor which we must skip
                    continue
                pre_assignments[name] = _assign_qtype[name]

                subclasses_assigned[cls].append(name)
                if INFO: print(f"[Info] Assigning {name!r} → {pre_assignments[name]!r} (missing metrics)")

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
        if INFO: print(f"[Info] {cls.upper()} outlier bounds: lower={lower:.4f}, upper={upper:.4f}")
        out_low = [n for n,v in class_vals.items() if v < lower]
        out_high = [n for n,v in class_vals.items() if v > upper]
        if DEBUG: print(f"[Debug] {cls.upper()} low outliers: {out_low}")
        if DEBUG: print(f"[Debug] {cls.upper()} high outliers: {out_high}")

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
                        print(f"[Info] Warning: skipping harmonization for group {gi} quant {extreme_q} because pattern match counts differ (counts={lengths}); using per-tensor sizes.")
                    return None

                if lengths[0] == 0:
                    return None

                # find index of name in its pattern list
                my_list = candidate_lists[matched_pi]
                if name not in my_list:
                    # safety: unexpected
                    if INFO:
                        print(f"[Info] Warning: {name!r} not present in pattern list for harmonization group {gi}; skipping harmonization for this tensor.")
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
                        print(f"[Info] Assigned {desc} quant {assigned_q} to outlier {n}, size={size/GIB:.3f} GiB")
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
                            print(f"[Info] Assigned {desc} quant {assigned_q} to outlier {nm}, size={size_harmonized/GIB:.3f} GiB (harmonized group {group_idx})")

        if not args.use_greedy_quant_assign:
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
        highest_q = max(quants, key=get_bpw)
        lowest_q = min(quants, key=get_bpw)
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
                if INFO: print(f"[Info] {cls.upper()} max-size set to {raw_max} of {highest_q} total ({max_ref/GIB:.3f} GiB) = {_max_arg_bytes/GIB:.3f} GiB")
            else:
                _max_arg_bytes = float(raw_max) * GIB
            max_arg_bytes = _max_arg_bytes
            max_arg_bytes -= outlier_bytes # deduct outliers
            max_arg_bytes -= f32_offset.get(cls, 0) # deduct f32 offset
            max_arg_bytes -= pre_assignments_offset.get(cls, 0) # deduct pre-assigned offset
            if INFO: print(f"[Info] Deducted outliers and f32 total {outlier_bytes/GIB:.3f} GiB from target, adjusted max={max_arg_bytes/GIB:.3f} GiB")

        if _max_arg_bytes >= (max_ref - max_ref*0.0001):
            # Assign highest quant to all (except extremes)
            if INFO: print(f"[Info] Reasonably assigning highest quant to all tensors...")
            assignment, sizes = assign_quants(
                [highest_q], None, class_vals)
            total_bytes = sum(sizes.values())
        elif _max_arg_bytes == 0:
            # Assign lowest quant to all (except extremes)
            if INFO: print(f"[Info] Reasonably assigning lowest quant to all tensors...")
            assignment, sizes = assign_quants(
                [lowest_q], None, class_vals)
            total_bytes = sum(sizes.values())
        elif max_arg_bytes:
            if args.use_greedy_quant_assign:
                # Build tensor_quants mapping (default to cls-specific quants)
                tensor_quants_local = {n: (gpu_quants if cls == 'gpu' else cpu_quants) for n in names_to_assign}

                # ---- Expand CLI regex groups into concrete per-layer lists restricted to this class' tensors ----
                # Harmonization groups
                try:
                    expanded_harmonize_groups = expand_harmonize_groups(harmonize_groups, names_to_assign) if harmonize_groups else []
                except Exception:
                    expanded_harmonize_groups = []
                if INFO and harmonize_groups and not expanded_harmonize_groups:
                    print(f"[Info] No harmonize groups expanded for class {cls}.")

                # Synergistic groups
                try:
                    expanded_synergistic_groups = expand_harmonize_groups(synergistic_groups, names_to_assign) if synergistic_groups else []
                except Exception:
                    expanded_synergistic_groups = []
                if INFO and synergistic_groups and not expanded_synergistic_groups:
                    print(f"[Info] No synergistic groups expanded for class {cls}.")

                # ---- Call greedy quant assignment with all parameters ----
                # Use get_map_sizes_and_elements to build the tensor_sizes mapping
                assignment, total_bytes = greedy_quant_assign(
                    tensors=names_to_assign,
                    tensor_sizes={
                        n: {q: get_map_sizes_and_elements(q)[0].get(n, 0)
                            for q in (gpu_quants if cls == 'gpu' else cpu_quants)}
                        for n in names_to_assign
                    },
                    ppl_loss=class_vals,
                    degradation_factors=quant_degradation_values,
                    tensor_quants=tensor_quants_local,
                    budget_bytes=int(max_arg_bytes),
                    debug=DEBUG,
                    harmonized_groups=expanded_harmonize_groups,
                    loss_exponent=args.exponential_factor,
                    synergistic_groups=expanded_synergistic_groups,
                    synergy_strength=args.synergy_strength
                )
            else:
                assignment, total_bytes = optimize_midpoint_and_assign(
                    quants, None, class_vals,
                    max_arg_bytes, args.tolerance, args.exponential_factor, harmonize_groups=harmonize_groups)
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
                                print(f"[Debug] Regex override (cpu) matched {name} -> {qt}")
                            return qt
                else:
                    for cre, qt in gpu_regex_assign:
                        if cre.fullmatch(name):
                            if DEBUG:
                                print(f"[Debug] Regex override (gpu) matched {name} -> {qt}")
                            return qt
            except Exception as e:
                if DEBUG:
                    print(f"[Debug] Regex override check error for {name}: {e}")
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
                    print(f"[Debug] Inferred from pre_assignments: {name} -> {q}")
                return q
            # 2) regex override lists (user-provided patterns)
            overridden = _regex_override_for(name, cls_local)
            if overridden:
                return overridden
            # 3) already computed assignments (post-assignment by this tool)
            if name in assignments.get(cls_local, {}):
                q = assignments[cls_local].get(name)
                if DEBUG:
                    print(f"[Debug] Inferred from assignments[{cls_local}]: {name} -> {q}")
                return q
            # 4) the explicit class-level default qtype flags
            if cls_local == 'cpu' and args.cpu_assign_qtype:
                if DEBUG:
                    print(f"[Debug] Inferred class default cpu_assign_qtype for {name} -> {args.cpu_assign_qtype}")
                return args.cpu_assign_qtype
            if cls_local == 'gpu' and args.gpu_assign_qtype:
                if DEBUG:
                    print(f"[Debug] Inferred class default gpu_assign_qtype for {name} -> {args.gpu_assign_qtype}")
                return args.gpu_assign_qtype
            if DEBUG:
                print(f"[Debug] No inferred assignment for {name}")
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
                tensor_index[cls].append({
                    'name': name, 'orig_q': orig_q, 'final_q': final_q,
                    'size': size, 'elements': elements, 'kind': 'f32'
                })

            # group mismatches and print warnings (same behaviour as before)
            from collections import defaultdict
            pair_to_names = defaultdict(list)
            for name, o, f in f32_entries:
                if o and o != f:
                    pair_to_names[(o, f)].append(name)

            phase_mismatch_count = sum(len(v) for v in pair_to_names.values())
            if INFO:
                print(f"[Info] f32 phase mismatches: {phase_mismatch_count}")

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
                tensor_index[cls].append({
                    'name': name, 'orig_q': orig_q, 'final_q': final_q,
                    'size': size, 'elements': elements, 'kind': 'pre'
                })

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
                tensor_index[cls].append({
                    'name': name, 'orig_q': orig_q, 'final_q': final_q,
                    'size': size, 'elements': elements, 'kind': 'measured'
                })

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
                print(f"[Info] Group '{base}' pre_mismatch={pre_mismatch} val_mismatch={val_mismatch}")

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
            print(f"[Info] Completed verbose assignment inspection for class '{cls}'.")

    # Recompute fallback_corrections from the single canonical registry (avoid double counting)
    fallback_corrections = sum(
        1 for cls in ('cpu', 'gpu') for e in tensor_index.get(cls, [])
        if e.get('orig_q') and e.get('final_q') and e['orig_q'] != e['final_q']
    )

    if DEBUG:
        print(f"[Debug] tensor_index - ", tensor_index)

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
            print(f"[Debug] quants_list - ", quants_list)
        if DEBUG:
            print(f"[Debug] _quants_list - ", _quants_list)
        if DEBUG:
            print(f"[Debug] ordered_qtypes - ", ordered_qtypes)
        candidate_quants = list(dict.fromkeys((quants_list or []) + list(ordered_qtypes)))
        sorted_quants = sorted(candidate_quants, key=lambda q: get_bpw(q) or 0, reverse=True)
        if DEBUG:
            print(f"[Debug] sorted_quants - ", sorted_quants)

        # build quick index for this class by final_q
        by_final = {}
        for e in tensor_index.get(cls, []):
            by_final.setdefault(e['final_q'], []).append(e)

        # '+' section: show user pre-assigned or f32 grouped by qt
        # '*' section: show user fallback grouped by qt
        for qt in sorted_quants:
            # bpw for this qtype (safe fallback 0)
            try:
                bpw_val = get_bpw(qt)
            except Exception:
                bpw_val = 0
            bpw_str = f"{bpw_val:.4f}".rstrip('0').rstrip('.')

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
                print(f"# +{qt:<10}\t{cnt_f32:<3}\t{bpw_str:<6}\t{gib_f32:>6.2f} GiB\t-\t\t-")

            # 2) pre-assigned non-fallback (+)
            cnt_pre = len(pre_nfb)
            if cnt_pre > 0:
                bytes_pre = sum(e['size'] for e in pre_nfb)
                gib_pre = bytes_pre / GIB
                print(f"# +{qt:<10}\t{cnt_pre:<3}\t{bpw_str:<6}\t{gib_pre:>6.2f} GiB\t-\t\t-")

            # 3) pre-assigned fallback (*+)
            cnt_pre_fb = len(pre_fb)
            if cnt_pre_fb > 0:
                bytes_pre_fb = sum(e['size'] for e in pre_fb)
                gib_pre_fb = bytes_pre_fb / GIB
                #pct_pre_fb = (bytes_pre_fb / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# *+{qt:<8}\t{cnt_pre_fb:<3}\t{bpw_str:<6}\t{gib_pre_fb:>6.2f} GiB\t-\t\t-")

            # 4) measured fallback (*)
            cnt_meas_fb = len(meas_fb)
            if cnt_meas_fb > 0:
                bytes_meas_fb = sum(e['size'] for e in meas_fb)
                gib_meas_fb = bytes_meas_fb / GIB
                pct_meas_fb = (bytes_meas_fb / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# *{qt:<9}\t{cnt_meas_fb:<3}\t{bpw_str:<6}\t{gib_meas_fb:>6.2f} GiB\t{pct_meas_fb:>3.1f}%\t\t{max_gib:.2f}")

            # 5) measured non-fallback (regular, no prefix). Always emit line (possibly zero).
            cnt_meas = len(meas_nfb)
            #if cnt_meas > 0:
            if cnt_meas > 0 or ((cnt_meas == 0 and qt != "f32") and qt in quants_list):
                bytes_meas = sum(e['size'] for e in meas_nfb)
                gib_meas = bytes_meas / GIB
                pct_meas = (bytes_meas / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# {qt:<10}\t{cnt_meas:<3}\t{bpw_str:<6}\t{gib_meas:>6.2f} GiB\t{pct_meas:>3.1f}%\t\t{max_gib:.2f}")

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
        print("#   they have been grouped and displayed as '*<qtype>' above to show the final (map-observed) qtype and sizes separately.")
    # Conditionally warn user that automatic fallbacks may have changed assignments
    if fallback_corrections > 0:
        print(f"# - WARNING: {fallback_corrections} tensor assignments were substituted to the dtype actually present in their tensor map files. "
              "\n#   This may change the final size relative to the expected thresholds and chosen quants. "
              "\n#   To disable automatic map-based fallbacks and preserve the script's original assigned qtypes exactly, re-run with --no-fallback.")

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

    def print_maps_sorted_by_bpw(MAP_FILE_INFO: Dict[str, Tuple[str, str, str]]) -> None:
        """
        Print entries from MAP_FILE_INFO sorted by get_bpw(qtype) (desc).
        Each printed block shows BPW (if available), SHA-256 and model name.
        """
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

        # Sort items by bpw descending; tie-break by filename (stable deterministic)
        sorted_items = sorted(
            MAP_FILE_INFO.items(),
            key=lambda kv: (bpw_cache.get(kv[1][0], float("-inf")), kv[0]),
            reverse=True
        )

        # Print lines in the requested format, including BPW
        for map_filename, (qtype, sha256sum, model_name) in sorted_items:
            bpw = bpw_cache.get(qtype, float("-inf"))
            bpw_str = f"{bpw:.6g}" if math.isfinite(bpw) else "N/A"
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
