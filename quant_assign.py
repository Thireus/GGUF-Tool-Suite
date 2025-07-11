#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_assign.py the recipe maker tool of choice! Use it   **#
#** to produce recipes that can be cooked and used by others. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-11-2025 -------------------- **#
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

# Requires: pip install pandas numpy argparse

# Tip: You can pipe the output of this script (as long as no warning or debug logs are present) to quants_regex_merger like so: | tee /dev/tty | ./quants_regex_merger.sh
# python quant_assign.py ppl_results_guessed.csv --gpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq6_k iq5_k_r4 --cpu-tensors-max-size 230 --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528
# python quant_assign.py 'ppl_results.csv' --gpu-tensors '.*' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq5_k_r4 iq6_k --cpu-tensors-max-size 230 --gpu-tensors-max-size 90% --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528

from datetime import datetime
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

# Global default quants list
DEFAULT_QUANTS = ['q8', 'q4']

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

# Remote connection settings for tensor_downloader.sh:
# Please edit tensor_downloader.sh!
# Resolve script directory for locating tensor_downloader.sh
script_dir = os.path.dirname(os.path.realpath(__file__))
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


def obtain_quants_bpw(qtype):
    """
    Infer bits-per-weight (bpw) for each tensor of a quantized map.
    Compares sizes in the bf16 base map to the qtype map.
    Returns: dict tensor_name -> bpw (float)
    """
    # load base sizes and types from bf16 map
    base_sizes, base_actual_qtypes = get_map_sizes('bf16')
    # load quantized sizes (map returns tuple even if actual_qtypes unused)
    if qtype == 'f32':
        _qtype = 'bf16'
    else:
        _qtype = qtype
    quant_sizes, quant_actual_qtypes = get_map_sizes(_qtype)
    bpw_map = {}
    for name, Sq in quant_sizes.items():
        Sbase = base_sizes.get(name)
        if Sbase is None or Sbase == 0:
            if DEBUG:
                print(f"[Debug] No base size for tensor {name}, skipping")
            continue
        dtype_base = base_actual_qtypes.get(name, 'bf16')
        # bits per weight in base format
        if dtype_base in ('bf16', 'fp16'):
            bbase = 16
        else:
            bbase = 32
        dtype_quant = {quant_actual_qtypes.get(name, qtype)}
        bpw = bbase * (Sq / Sbase)
        if quant_actual_qtypes.get(name, qtype) == qtype:
            bpw_map[name] = bpw
        else:
            if DEBUG:
                print(f"[Debug] Skipping tensor {name} because dtype_quant={dtype_quant} mismatch with exepcted qtype: {qtype}")
        if DEBUG:
            print(f"[Debug] Tensor {name}: base dtype={dtype_base}, Sbase={Sbase}, dtype_quant={dtype_quant}, Sq={Sq}, bpw={bpw:.3f}")
    return bpw_map


@functools.lru_cache(maxsize=None)
def get_bpw(qtype):
    """
    Return the bpw for a given qtype, caching results for performance.
    """
    # infer bits-per-weight from data instead of hardcoding
    bpw_map = obtain_quants_bpw(qtype)
    # compute average bpw across all tensors, fallback if empty
    if bpw_map:
        return sum(bpw_map.values()) / len(bpw_map)
    else:
        if DEBUG:
            print(f"[Debug] Could not infer bpw for qtype {qtype}, using extract_quant_num fallback")
        return extract_quant_num(qtype)

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
    Classify tensor names into CPU/GPU based on regex lists.
    """
    classes = {'cpu': [], 'gpu': []}
    for name in columns:
        assigned = False
        for pat in cpu_patterns:
            if re.fullmatch(pat, name):
                classes['cpu'].append(name)
                assigned = True
                break
        if assigned:
            continue
        for pat in gpu_patterns:
            if re.fullmatch(pat, name):
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


def fetch_map_for_qtype(qtype):
    """
    Fetch and cache tensors.{qtype}.map via tensor_downloader.sh.
    """
    if qtype in _fetched_maps:
        return True
    tmpdir = tempfile.gettempdir()
    local_map = os.path.join(tmpdir, f"tensors.{qtype}.map")
    cmd = [tensor_downloader, qtype.upper(), "0", tmpdir, f"tensors.{qtype}.map"]
    if INFO: print(f"[Info] Fetching map for {qtype}...")
    try:
        if DEBUG or INFO:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if INFO: print(f"[Info] Saved map to {local_map}")
        _fetched_maps.add(qtype)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Warning] failed to fetch tensors.map: {e}")
        return False


def get_map_sizes(qtype):
    """
    Return parsed map sizes for given qtype, caching results.
    """
    if qtype not in _quant_maps:
        fetch_map_for_qtype(qtype)
        # parse_map_file now returns tuple
        _quant_maps[qtype] = parse_map_file(qtype)
    return _quant_maps[qtype]


def parse_map_file(qtype):
    """
    Parse local tensors.{qtype}.map into:
      - sizes: dict tensor_name -> bytes_size
      - actual_qtypes: dict tensor_name -> dtype (e.g., 'bf16', 'f32')
    """
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"tensors.{qtype}.map")
    sizes = {}
    actual_qtypes = {}
    if not os.path.exists(path):
        return sizes, actual_qtypes
    with open(path) as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 5:
                continue
            # parts example:
            # [file, checksum, tensor_name, shape=..., dtype=f32, elements=..., bytes=...]
            tensor_name = parts[2]
            # find dtype and bytes fields
            dtype = None
            size_bytes = None
            for p in parts:
                if p.startswith('dtype='):
                    dtype = p.split('=')[1]
                elif p.startswith('bytes='):
                    size_bytes = int(p.split('=')[1])
            if dtype is None or size_bytes is None:
                continue
            sizes[tensor_name] = size_bytes
            actual_qtypes[tensor_name] = dtype
    return sizes, actual_qtypes


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


def assign_quants(quants, _, class_values, forced_mid=None, stretch=1.0):
    """
    Assign quants based on spread intervals and fetch correct tensor sizes.
    """
    if INFO:
        print(f"[Info] Performing spread-based quant assignment (stretch={stretch:.2f})...")
    spreads = compute_group_spreads(class_values, forced_mid)
    intervals = compute_quant_intervals(quants, stretch)
    assignment = {}
    sizes = {}
    for name in class_values:
        spread = spreads[name]
        for q, top, bottom in intervals:
            if bottom < spread <= top:
                assignment[name] = q
                break
        else:
            assignment[name] = quants[-1]
        sizes[name], _ = get_map_sizes(assignment[name])
        sizes[name] = sizes[name].get(name, 0)
        if INFO:
            print(f"[Info] Assigned {assignment[name]} to {name} (spread={spread:.4f}) size={sizes[name]}")
    return assignment, sizes


def total_size_for_quant(names, qtype):
    """
    Sum the map sizes for the given tensor names under the specified quant.
    """
    sizes_map, _ = get_map_sizes(qtype)
    return sum(sizes_map.get(name, 0) for name in names)


def optimize_midpoint_and_assign(quants, _, class_values,
                                 max_bytes, tolerance=0.05, exp_factor=1.0):
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
                                             forced_mid=mid, stretch=stretch)
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
                sizes[name], _ = get_map_sizes(new_q)
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
    - Otherwise fall back to default_qtype (or highest‑bpw if default_qtype is None).
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


def main():
    global DEBUG, INFO
    parser = argparse.ArgumentParser(description="Assign optimal quants per tensor based on PPL CSV.")
    parser.add_argument('--debug', action='store_true', help='Show debug logs')
    parser.add_argument('--info', action='store_true', help='Show info logs')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Relative GiB tolerance for size optimization')
    parser.add_argument('--cpu-irq-k', type=float, default=1.5,
                        help='IQR multiplier k for CPU outlier detection')
    parser.add_argument('--gpu-irq-k', type=float, default=1.5,
                        help='IQR multiplier k for GPU outlier detection')
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('--qtype', help='QTYPE to analyze (default: lowest quant)')
    parser.add_argument('--cpu-assign-qtype', help='QTYPE to assign to non-measured CPU tensors or tensors missing from csv (default: highest quant)')
    parser.add_argument('--gpu-assign-qtype', help='QTYPE to assign to non-measured GPU tensors or tensors missing from csv (default: highest quant)')
    parser.add_argument('--cpu-assign-tensors', nargs='+', default=[], help="List of regex=QTYPE patterns for CPU tensors to force-assign")
    parser.add_argument('--gpu-assign-tensors', nargs='+', default=[], help="List of regex=QTYPE patterns for GPU tensors to force-assign")
    #parser.add_argument('--sample-ppl', help='CSV sample PPL file path', required=True)
    parser.add_argument('--cpu-tensors', nargs='+', default=[], help='Regex patterns for CPU tensors')
    parser.add_argument('--gpu-tensors', nargs='+', default=[], help='Regex patterns for GPU tensors')
    parser.add_argument('--cpu-quants', nargs='+', help='Ordered list of CPU quants')
    parser.add_argument('--gpu-quants', nargs='+', help='Ordered list of GPU quants')
    parser.add_argument('--cpu-tensors-max-size', type=str, help='Max CPU tensors size in GiB or percent (e.g., 80%)')
    parser.add_argument('--gpu-tensors-max-size', type=str, help='Max GPU tensors size in GiB or percent (e.g., 80%)')
    parser.add_argument('--exponential-factor', type=float, default=1.0,
                        help='Exponent controlling midpoint adjustment aggressiveness during stretch sweeps. '
                             'Higher values push quantization toward extremes; default is 1.0.')
    parser.add_argument('--ignore-f32', action='store_true', help='Ignore f32 tensors (default: not ignored)')
    parser.add_argument('--tensors-from-csv', action='store_true', help='Obtains list of tensors from csv file only (default: tensors are obtained from map file)')
    args = parser.parse_args()

    def parse_regex_assign_list(raw_list):
        parsed = []
        for item in raw_list:
            try:
                pat, qt = item.split('=', 1)
            except ValueError:
                parser.error(f"Invalid regex‑assign spec: {item}. Must be PATTERN=QTYPE")
            parsed.append((re.compile(pat), qt))
        return parsed

    cpu_regex_assign = parse_regex_assign_list(args.cpu_assign_tensors)
    gpu_regex_assign = parse_regex_assign_list(args.gpu_assign_tensors)

    DEBUG = args.debug
    INFO = args.info or DEBUG

    if args.cpu_tensors and not args.cpu_quants:
        parser.error("--cpu-quants is required when --cpu-tensors is used")
    if args.gpu_tensors and not args.gpu_quants:
        parser.error("--gpu-quants is required when --gpu-tensors is used")

    cpu_quants = args.cpu_quants or DEFAULT_QUANTS
    # Reorder cpu_quants from highest to lowest bpw
    try:
        cpu_quants = sorted(cpu_quants, key=get_bpw, reverse=True)
        if INFO: print(f"[Info] CPU quants reordered by bpw: {cpu_quants}")
    except Exception:
        pass

    gpu_quants = args.gpu_quants or DEFAULT_QUANTS
    # Reorder gpu_quants from highest to lowest bpw
    try:
        gpu_quants = sorted(gpu_quants, key=get_bpw, reverse=True)
        if INFO: print(f"[Info] GPU quants reordered by bpw: {gpu_quants}")
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

    # Pre-fetch maps
    fetch_map_for_qtype(qtype)
    _, items = get_map_sizes(qtype)
    _, items_bf16 = get_map_sizes('bf16')
    _items = items_bf16
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
    # get_map_sizes returns (sizes, actual_qtypes)
    f32_names = [n for n,d in _items.items() if d == 'f32']

    # Classify tensors
    classes = classify_tensors(tensor_names, args.cpu_tensors, args.gpu_tensors)

    subclasses_to_assign = {'cpu': [], 'gpu': []}
    subclasses_assigned = {'cpu': [], 'gpu': []}

    # Build values dict, converting strings (e.g. '0.0653%') properly and pre-assign tensors that haven't been measured
    values = {}
    pre_assignments = {}
    pre_assignments_offset = {'cpu': 0, 'gpu': 0}
    for cls in ['cpu', 'gpu']:
        quants = cpu_quants if cls == 'cpu' else gpu_quants
        names = classes.get(cls, [])
        if cls == 'cpu':
            _assign_qtype = assign_qtype(args.cpu_assign_qtype, cpu_regex_assign, quants, names)
        else:
            _assign_qtype = assign_qtype(args.gpu_assign_qtype, gpu_regex_assign, quants, names)

        # skip if nothing for this cls
        if not names:
            continue

        for name in names:
            # missing measurement → pre‑assign
            if name not in row or pd.isna(row.at[name]):
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

        # 1. Get all unique q‑types
        _assign_qtype_qtypes = set(_assign_qtype.values())

        # 2. Loop over each q‑type
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
        f32_classes = classify_tensors(f32_names, args.cpu_tensors, args.gpu_tensors)
        for cls in ['gpu', 'cpu']:
            # if user did *not* list 'f32' in 'cls'_quants, add to cls offset
            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
                f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), 'bf16')
                if f32_offset[cls] == 0:
                    f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), qtype)

    # Track precomputed extremes per class
    extremes = {}

    # Process GPU and CPU classes
    for cls in ['gpu', 'cpu']:
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
        for n in out_low:
            assignments[cls][n] = quants[-1]
            size = total_size_for_quant([n], quants[-1])
            outlier_bytes += size
            if INFO: print(f"[Info] Assigned lowest quant {quants[-1]} to low outlier {n}, size={size/GIB:.3f} GiB")
        for n in out_high:
            assignments[cls][n] = quants[0]
            size = total_size_for_quant([n], quants[0])
            outlier_bytes += size
            if INFO: print(f"[Info] Assigned highest quant {quants[0]} to high outlier {n}, size={size/GIB:.3f} GiB")
        for n in out_low + out_high:
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
            assignment, total_bytes = optimize_midpoint_and_assign(
                quants, None, class_vals,
                max_arg_bytes, args.tolerance, args.exponential_factor)
            #print(f"# Optimized sub-total {cls.upper()} size excluding outliers and f32: {total_bytes/GIB:.3f} GiB")
        else:
            assignment, sizes = assign_quants(
                quants, None, class_vals)
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

        # List any auto‑added f32 tensors in the output
        if add_f32 and f32_offset.get(cls,0) > 0:
            print(f"# Auto‑included f32 tensors for {cls.upper()}:")
            for n in sorted(f32_names):
                # _size = total_size_for_quant([n], 'bf16')
                # if _size == 0:
                #     _size = total_size_for_quant([n], qtype)
                # size = _size / GIB
                print(f"{re.escape(n)}=f32")

        groups = group_tensors(names)
        for base, full in groups.items():
            displayed_already = False
            for name in sorted((n for n in full if n in pre_assignments), key=lambda n: pre_assignments[n], reverse=True):
                if not displayed_already:
                    print(f"# Group: {re.escape(base)}")
                    displayed_already = True
                print(f"{re.escape(name)}={pre_assignments.get(name,'')}")
            for name in sorted((n for n in full if n in values), key=lambda n: values[n], reverse=True):
                if not displayed_already:
                    print(f"# Group: {re.escape(base)}")
                    displayed_already = True
                print(f"{re.escape(name)}={assignments[cls].get(name,'')}")

    # Summary of tensor sizes per class
    print("\n## Summary of tensor sizes per class")
    _tb = 0
    _pct = 0
    for cls, tb in totals.items():
        # Retrieve extremes
        ext = extremes.get(cls, {})
        highest_q = ext.get('highest_q')
        lowest_q = ext.get('lowest_q')
        max_size = ext.get('max_ref', 0) / GIB
        min_size = ext.get('min_ref', 0) / GIB
        # Percentage of max q-size
        pct = (tb / (max_size * GIB)) * 100 if max_size > 0 else 0
        _tb += tb
        _pct += pct
        print(f"#{cls.upper():>4} Total: {tb/GIB:.3f} GiB ({pct:.1f}%) | {max_size:.2f} GiB max, if all were {highest_q} | {min_size:.2f} GiB min, if all were {lowest_q}")
    
    print(f"# GPU+CPU Total: {_tb/GIB:.3f} GiB ({_pct/2:.1f}%)")

    # Summary tensor counts and bits-per-weight per qtype
    print("\n## Summary of tensor counts and bpw per qtype")

    # Build a combined list of all qtypes, maintaining order
    all_qtypes = []
    if cpu_quants:
        all_qtypes.extend(cpu_quants)
    if gpu_quants:
        all_qtypes.extend(gpu_quants)
    seen = set()
    ordered_qtypes = []
    for qt in all_qtypes:
        if qt not in seen:
            seen.add(qt)
            ordered_qtypes.append(qt)

    # Use separate assignment maps per class (already handled earlier)
    quant_counts_by_class = {
        'cpu': Counter(assignments.get('cpu', {}).values()),
        'gpu': Counter(assignments.get('gpu', {}).values())
    }

    _bytes = 0
    _bpw = 0
    _w = 0
    for cls in ['gpu', 'cpu']:
        quants_list = gpu_quants if cls == 'gpu' else cpu_quants
        _quants_list = quants_list
        if add_f32:
            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
                quants_list = ['f32'] + _quants_list
        names = classes.get(cls, [])
        names_assigned = subclasses_assigned.get(cls, [])
        names_post_assigned = subclasses_to_assign.get(cls, [])
        if not quants_list:
            continue
        # Section header per class
        print(f"#\n# {cls.upper()}-loaded quants:")
        print(f"# QTYPE\t\tCount\tBPW\tAssigned GiB\t% Assigned\tMax GiB (all)")
        if cls == 'cpu':
            _assign_qtype = assign_qtype(args.cpu_assign_qtype, cpu_regex_assign, _quants_list, names_assigned)
        else:
            _assign_qtype = assign_qtype(args.gpu_assign_qtype, gpu_regex_assign, _quants_list, names_assigned)
        if _assign_qtype not in (cpu_quants if cls=='cpu' else gpu_quants):
            unique_qtypes = set(_assign_qtype.values())
            quants_list = list(set(quants_list).union(unique_qtypes))
        # Sort quants by bits-per-weight descending
        sorted_quants = sorted(quants_list, key=lambda q: get_bpw(q) or 0, reverse=True)
        for qt in sorted_quants:
            try:
                bpw_val = get_bpw(qt)
            except:
                bpw_val = 0
            if qt == 'f32':
                sizes_map, _ = get_map_sizes('bf16')
            else:
                sizes_map, _ = get_map_sizes(qt)

            # Pre-assigned tensors
            if qt in _assign_qtype.values() or qt == 'f32':
                # If we’re in the f32‐override case, swap in the f32 list; otherwise keep the original.
                if add_f32 and qt == 'f32':
                    # Overwrite names_assigned list
                    names_assigned = f32_classes.get(cls, [])
                    cnt = len(names_assigned)
                    if cnt > 0:
                        assigned_bytes = sum(sizes_map.get(n, 0) for n in names_assigned)
                        assigned_gib = assigned_bytes / GIB
                        _bytes += assigned_bytes
                        _bpw += bpw_val * assigned_bytes
                        _w += assigned_bytes / bpw_val
                        print(f"# +{qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t-\t\t-")
                else:
                    # for each qt in whatever loop you have:
                    # collect the names whose assigned q‐type is qt
                    names_assigned = [name 
                                    for name, q in _assign_qtype.items() 
                                    if q == qt]
                    # now cnt and assigned_bytes only refer to that subset:
                    cnt = len(names_assigned)
                    assigned_bytes = sum(sizes_map.get(n, 0) for n in names_assigned)
                    assigned_gib = assigned_bytes / GIB
                    _bytes += assigned_bytes
                    _bpw   += bpw_val * assigned_bytes
                    _w += assigned_bytes / bpw_val
                    print(f"# +{qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t-\t\t-")
            
            # Post-assigned tensors
            cnt = Counter(assignments[cls].values()).get(qt, 0)
            if (cnt > 0 or qt in _quants_list) and qt != 'f32':
                # Assigned size
                assigned = [n for n, q in assignments[cls].items() if q == qt]
                assigned_bytes = sum(sizes_map.get(n, 0) for n in assigned)
                assigned_gib = assigned_bytes / GIB
                _bytes += assigned_bytes
                _bpw += bpw_val * assigned_bytes
                _w += assigned_bytes / bpw_val
                # Max size if all
                max_gib = total_size_for_quant(names_post_assigned, qt) / GIB
                pct = (assigned_bytes / (max_gib * GIB) * 100) if max_gib > 0 else 0
                print(f"# {qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t{pct:>3.1f}%\t\t{max_gib:.2f}")
    
    print(f"#\n# -Average BPW: {_bytes/_w:.4f}")

    print(f"#\n# -Notes:\n# - '+' means user-defined pre-assigned tensors and f32 tensors")

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
