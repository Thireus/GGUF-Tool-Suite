#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** plot_llama_sweep.py plots llama-sweep-bench logs stored   **#
#** as files into graphs and finds the best -u/-ub combo.     **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-10-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.           ₙₑ𝓌 𝓌ₑₑₖ, ₙₑ𝓌 ₘₒ𝒹ₑₗ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# type: ignore 

"""
Parse grep-style lines like:
GLM-4.5-Air_1024_512.txt:|   512 |    128 |   8192 |    0.675 |   758.07 |    3.754 |    34.10 |

Group rows by filename (one series per file).
Plot two windows:
  1) N_KV vs S_TG t/s
  2) N_KV vs S_PP t/s

Gracefully handle Ctrl+C to close windows and exit.

Features:
- Read input from stdin (pipe), a single infile, or a directory of files.
- If directory is provided (via --dir or --infile pointing to a directory),
  each matching file (glob pattern) is scanned for lines that start with '|'
  and those lines are fed into the parser with "basename:| ..." format
  (matches the output of `grep '^|' some/dir/*.txt`).
- Legends are ordered best->worst per-figure and titled "Series (best to worst)".
- Series annotated with bold labels and arrow to last point.
- Interpolates metrics on a common grid and computes means for ranking.
- Choose BEST candidate within per-metric margins using combined dimension tie-breaker.

Example of script that produces the llama-sweep-bench log files:

#!/usr/bin/env bash

BATCH_SIZES=(512 1024 2048 4096 8192 16384)

CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0,1,2
MODEL="GLM-4.5-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01762.gguf"
BENCH_PATH="$HOME/ik_llama-main-b4065-a09bed8-bin-win-cuda-12.8-x64-avx512/llama-sweep-bench"

for i in "${BATCH_SIZES[@]}"; do
    for j in "${BATCH_SIZES[@]}"; do
        # Skip cases where i < j
        if (( i < j )); then
            continue
        fi

        OUTPUT_FILE="GLM-4.5_${i}_${j}.txt"

        "$BENCH_PATH" \
            -m "$MODEL" \
            -fa \
            -amb 1024 \
            -fmoe \
            -ctk f16 \
            -c 65536 \
            -ngl 99 \
            -ot "blk\.([0-9]|[1-3][0-9]|4[0-2])\.ffn_.*=CUDA0" \
            -ot "blk\.(4[3-9]|[5-7][0-9])\.ffn_.*=CUDA1" \
            -ot "blk\.(8[0-9]|90|91|92)\.ffn_.*=CUDA2" \
            -b "$i" \
            -ub "$j" \
            --warmup-batch \
            --no-mmap \
            --threads 36 \
            --main-gpu 0 \
            > "$OUTPUT_FILE" 2>&1
    done
done
"""

import sys
import argparse
import re
from collections import defaultdict
import signal
import os
import math
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

DEFAULT_HEADERS = ['PP', 'TG', 'N_KV', 'T_PP s', 'S_PP t/s', 'T_TG s', 'S_TG t/s']

METRIC_TABLE = [
    ("PP", "Number of prompt tokens processed before generation"),
    ("TG", "Number of tokens generated in generation phase"),
    ("N_KV", "Size of the KV-cache in tokens (lookback window)"),
    ("T_PP s", "Time taken for prompt processing (in seconds)"),
    ("S_PP t/s", "Throughput for prompt processing (tokens per second)"),
    ("T_TG s", "Time taken for generation phase (in seconds)"),
    ("S_TG t/s", "Throughput for token generation (tokens per second)"),
]

GRID_POINTS = 300  # number of interpolation points between 0 and cutoff

# simple markers (no letters/numbers)
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', '+', '*', 'x']

def print_metric_help():
    print("\nMetric meanings:")
    col1 = max(len(r[0]) for r in METRIC_TABLE) + 2
    for metric, meaning in METRIC_TABLE:
        print(f"  {metric.ljust(col1)}{meaning}")
    print()

def parse_line(line):
    """Return (filename, tokens_list) or (None, None)"""
    if ':' not in line:
        return None, None
    filename, rest = line.split(':', 1)
    if '|' not in rest:
        return filename.strip(), []
    parts = [p.strip() for p in rest.split('|')]
    tokens = [p for p in parts if p != '']
    return filename.strip(), tokens

def try_number(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == '':
        return None
    s = s.replace(',', '')
    try:
        if '.' in s or 'e' in s.lower():
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def make_label_from_filename(fn):
    m = re.search(r'_(\d+)[^_]*_(\d+)\.txt$', fn)
    if m:
        a, b = m.group(1), m.group(2)
        return f"-b {a} -ub {b}"
    nums = re.findall(r'(\d+)', fn)
    if len(nums) >= 2:
        return f"-b {nums[-2]} -ub {nums[-1]}"
    return os.path.basename(fn)

def extract_b_ub_from_filename(fn):
    bn = os.path.basename(fn)
    m = re.search(r'_(\d+)[^_]*_(\d+)\.txt$', bn)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None, None
    nums = re.findall(r'(\d+)', bn)
    if len(nums) >= 2:
        try:
            return int(nums[-2]), int(nums[-1])
        except Exception:
            return None, None
    return None, None

def extract_model_from_filename(fn):
    bn = os.path.basename(fn)
    m = re.match(r'(?P<model>.+?)_(\d+)[^_]*_(\d+)\.txt$', bn)
    if m:
        return m.group('model')
    return os.path.splitext(bn)[0]

def find_col_indices(header_tokens):
    h = [t.strip().lower().replace(' ', '').replace('/', '') for t in header_tokens]
    idx = {}
    for i, tok in enumerate(h):
        if 'n_kv' in tok or tok == 'nkv' or tok.startswith('n_kv') or tok == 'n_k':
            idx['nkv'] = i
        if 't_pp' in tok or tok.startswith('tpp') or 'tpp' in tok:
            idx['tpp'] = i
        if 's_pp' in tok or tok == 'spp':
            idx['spp'] = i
        if 't_tg' in tok or 'ttg' in tok or tok.startswith('t_tg'):
            idx['ttg'] = i
        if 's_tg' in tok or tok == 'stg':
            idx['stg'] = i
    return idx.get('nkv'), idx.get('tpp'), idx.get('spp'), idx.get('ttg'), idx.get('stg')

def build_series(lines):
    headers_by_file = {}
    rows_by_file = defaultdict(list)

    for L in lines:
        if not L.strip():
            continue
        fn, toks = parse_line(L)
        if fn is None:
            continue
        if not toks:
            continue
        up = [t.upper() for t in toks]
        if any('N_KV' in t or 'S_TG' in t or 'S_PP' in t or 'T_PP' in t or 'T_TG' in t for t in up):
            headers_by_file[fn] = toks
            continue
        if all(set(t) <= set('- ') for t in toks):
            continue
        rows_by_file[fn].append(toks)

    for fn in rows_by_file.keys():
        if fn not in headers_by_file:
            headers_by_file[fn] = DEFAULT_HEADERS

    series_data = {}
    for fn, rows in rows_by_file.items():
        hdr = headers_by_file.get(fn, DEFAULT_HEADERS)
        idx_nkv, idx_tpp, idx_spp, idx_ttg, idx_stg = find_col_indices(hdr)
        if idx_nkv is None and len(hdr) >= 3:
            idx_nkv = 2
        if idx_tpp is None and len(hdr) >= 4:
            idx_tpp = 3
        if idx_spp is None and len(hdr) >= 5:
            idx_spp = 4
        if idx_ttg is None and len(hdr) >= 6:
            idx_ttg = 5
        if idx_stg is None:
            idx_stg = len(hdr) - 1

        nkv_list = []
        s_tg_list = []
        s_pp_list = []
        t_pp_list = []
        t_tg_list = []
        for r in rows:
            if (idx_nkv is None or idx_nkv >= len(r) or
                idx_tpp is None or idx_tpp >= len(r) or
                idx_spp is None or idx_spp >= len(r) or
                idx_ttg is None or idx_ttg >= len(r) or
                idx_stg is None or idx_stg >= len(r)):
                numeric_tokens = [try_number(tok) for tok in r]
                nums = [p for p in numeric_tokens if p is not None]
                if len(nums) >= 3:
                    nkv_val = nums[2]
                    tpp_val = nums[3] if len(nums) >= 4 else None
                    spp_val = nums[4] if len(nums) >= 5 else None
                    ttg_val = nums[5] if len(nums) >= 6 else None
                    stg_val = nums[-1] if len(nums) >= 4 else None
                else:
                    nkv_val = numeric_tokens[2] if len(numeric_tokens) > 2 else None
                    stg_val = numeric_tokens[-1] if numeric_tokens else None
                    tpp_val = numeric_tokens[3] if len(numeric_tokens) > 3 else None
                    spp_val = numeric_tokens[4] if len(numeric_tokens) > 4 else None
                    ttg_val = numeric_tokens[5] if len(numeric_tokens) > 5 else None
            else:
                nkv_val = try_number(r[idx_nkv]) if idx_nkv is not None and idx_nkv < len(r) else None
                tpp_val = try_number(r[idx_tpp]) if idx_tpp is not None and idx_tpp < len(r) else None
                spp_val = try_number(r[idx_spp]) if idx_spp is not None and idx_spp < len(r) else None
                ttg_val = try_number(r[idx_ttg]) if idx_ttg is not None and idx_ttg < len(r) else None
                stg_val = try_number(r[idx_stg]) if idx_stg is not None and idx_stg < len(r) else None

                if nkv_val is None or stg_val is None:
                    numeric_tokens = [try_number(tok) for tok in r]
                    nums = [p for p in numeric_tokens if p is not None]
                    if len(nums) >= 3:
                        nkv_val = nums[2]
                        stg_val = nums[-1]
                        if len(nums) >= 4:
                            tpp_val = nums[3]
                        if len(nums) >= 5:
                            spp_val = nums[4]
                        if len(nums) >= 6:
                            ttg_val = nums[5]

            if nkv_val is None or stg_val is None:
                continue
            try:
                nkv_int = int(nkv_val)
            except Exception:
                nkv_int = int(round(float(nkv_val)))

            nkv_list.append(nkv_int)
            s_tg_list.append(float(stg_val))
            s_pp_list.append(float(spp_val) if spp_val is not None else float('nan'))
            t_pp_list.append(float(tpp_val) if tpp_val is not None else float('nan'))
            t_tg_list.append(float(ttg_val) if ttg_val is not None else float('nan'))

        if nkv_list:
            combined = list(zip(nkv_list, s_tg_list, s_pp_list, t_pp_list, t_tg_list))
            combined.sort(key=lambda z: z[0])
            nkv_s, stg_s, spp_s, tpp_s, ttg_s = map(list, zip(*combined))
            series_data[fn] = {
                'nkv': nkv_s,
                's_tg': stg_s,
                's_pp': spp_s,
                't_pp': tpp_s,
                't_tg': ttg_s,
            }

    return series_data

def interpolate_on_grid(nkvs, vals, grid_x):
    if nkvs is None or vals is None:
        return None
    nkvs = np.array(nkvs, dtype=float)
    vals = np.array(vals, dtype=float)
    mask = ~np.isnan(vals)
    if not np.any(mask):
        return None
    nkvs_clean = nkvs[mask]
    vals_clean = vals[mask]
    if nkvs_clean.size == 0:
        return None
    if nkvs_clean.size == 1:
        return np.full_like(grid_x, vals_clean[0], dtype=float)
    interp_vals = np.interp(grid_x, nkvs_clean, vals_clean, left=vals_clean[0], right=vals_clean[-1])
    return interp_vals

def compute_means_for_eligible(series_data, min_nkv):
    eligible = []
    for fn, d in series_data.items():
        max_nkv = max(d.get('nkv', [0]))
        if max_nkv >= min_nkv:
            eligible.append((fn, max_nkv))
    if not eligible:
        return None, {}

    cutoff = int(min(max_nkv for (_, max_nkv) in eligible))
    grid_x = np.linspace(0.0, float(cutoff), GRID_POINTS)

    results = {}
    for fn, _ in eligible:
        d = series_data[fn]
        nkvs = np.array(d['nkv'], dtype=float)
        s_pp_interp = interpolate_on_grid(nkvs, d.get('s_pp', []), grid_x)
        s_tg_interp = interpolate_on_grid(nkvs, d.get('s_tg', []), grid_x)
        if s_pp_interp is None and s_tg_interp is None:
            continue
        s_pp_mean = float(np.nanmean(s_pp_interp)) if s_pp_interp is not None else None
        s_tg_mean = float(np.nanmean(s_tg_interp)) if s_tg_interp is not None else None
        b, ub = extract_b_ub_from_filename(fn)
        results[fn] = {'s_pp_mean': s_pp_mean, 's_tg_mean': s_tg_mean, 'n_pts': GRID_POINTS, 'b': b, 'ub': ub}
    return cutoff, results

def choose_best_by_margin_and_smallest_dim(means_dict, margin_pp_frac, margin_tg_frac):
    s_pp_items = [(fn, v['s_pp_mean'], v.get('b'), v.get('ub')) for fn, v in means_dict.items() if v.get('s_pp_mean') is not None]
    s_tg_items = [(fn, v['s_tg_mean'], v.get('b'), v.get('ub')) for fn, v in means_dict.items() if v.get('s_tg_mean') is not None]

    def pick_best(items, margin_frac):
        if not items:
            return None
        items_sorted = sorted(items, key=lambda z: z[1], reverse=True)
        top_mean = items_sorted[0][1]
        thresh = top_mean * (1.0 - margin_frac)
        candidates = [it for it in items_sorted if it[1] >= thresh]
        parsed = [it for it in candidates if it[2] is not None and it[3] is not None]
        if parsed:
            parsed_sorted = sorted(parsed, key=lambda z: (z[2] + z[3], -z[1], z[2]*z[3]))
            return parsed_sorted[0][0]
        else:
            return items_sorted[0][0]

    best_pp = pick_best(s_pp_items, margin_pp_frac)
    best_tg = pick_best(s_tg_items, margin_tg_frac)
    return best_pp, best_tg

def print_means_summary_with_best(means_dict, cutoff, min_nkv, margin_pp_percent, margin_tg_percent):
    if means_dict == {}:
        print(f"No eligible series (max(N_KV) >= {min_nkv}) — cannot compute means.\n")
        return
    margin_pp_frac = margin_pp_percent / 100.0
    margin_tg_frac = margin_tg_percent / 100.0

    print(f"Cutoff for means (common range used): 0 .. {int(cutoff)} (shortest max(N_KV) among eligible series)")
    print(f"Interpolation grid: {GRID_POINTS} points over 0..{int(cutoff)}")
    print(f"Only series with max(N_KV) >= {min_nkv} were considered.")
    print(f"Selection margins: S_PP margin = {margin_pp_percent}% ; S_TG margin = {margin_tg_percent}%")
    print()

    rows = []
    for fn, v in means_dict.items():
        label = make_label_from_filename(fn)
        model = extract_model_from_filename(fn)
        s_tg = v.get('s_tg_mean')
        s_pp = v.get('s_pp_mean')
        b = v.get('b')
        ub = v.get('ub')
        combined = (b + ub) if (b is not None and ub is not None) else None
        rows.append((fn, label, model, s_tg, s_pp, v.get('n_pts'), b, ub, combined))

    rows_s_tg = sorted([r for r in rows if r[3] is not None], key=lambda z: z[3], reverse=True)
    rows_s_pp = sorted([r for r in rows if r[4] is not None], key=lambda z: z[4], reverse=True)

    best_pp_fn, best_tg_fn = choose_best_by_margin_and_smallest_dim(means_dict, margin_pp_frac, margin_tg_frac)

    print("S_TG t/s means (best -> worst):")
    if not rows_s_tg:
        print("  No S_TG data available for eligible series.")
    else:
        for fn, label, model, s_tg, s_pp, npts, b, ub, combined in rows_s_tg:
            best_marker = " <-- BEST" if fn == best_tg_fn else ""
            dims = f"(b={b}, ub={ub})" if (b is not None and ub is not None) else ""
            print(f"  {label} (model: {model}) {dims} -> S_TG mean = {s_tg:.6g} over {npts} grid points{best_marker}")
    print()

    print("S_PP t/s means (best -> worst):")
    if not rows_s_pp:
        print("  No S_PP data available for eligible series.")
    else:
        for fn, label, model, s_tg, s_pp, npts, b, ub, combined in rows_s_pp:
            best_marker = " <-- BEST" if fn == best_pp_fn else ""
            dims = f"(b={b}, ub={ub})" if (b is not None and ub is not None) else ""
            print(f"  {label} (model: {model}) {dims} -> S_PP mean = {s_pp:.6g} over {npts} grid points{best_marker}")
    print()

def make_color_list(n):
    """
    Build a visually diverse color list of length >= n.
    Try to combine tab20/tab20b/tab20c (qualitative) first; otherwise sample a continuous cmap.
    """
    colors = []
    try:
        c1 = plt.get_cmap('tab20')
        if hasattr(c1, 'colors'):
            colors.extend(list(c1.colors))
        else:
            colors.extend([c1(i) for i in range(c1.N)])
    except Exception:
        pass
    # try tab20b and tab20c (may not exist on older matplotlib)
    try:
        c2 = plt.get_cmap('tab20b')
        if hasattr(c2, 'colors'):
            colors.extend(list(c2.colors))
        else:
            colors.extend([c2(i) for i in range(c2.N)])
    except Exception:
        pass
    try:
        c3 = plt.get_cmap('tab20c')
        if hasattr(c3, 'colors'):
            colors.extend(list(c3.colors))
        else:
            colors.extend([c3(i) for i in range(c3.N)])
    except Exception:
        pass

    if len(colors) >= n:
        return colors[:n]
    # fallback: sample a continuous but colorful cmap
    cmap = cm.get_cmap('nipy_spectral')
    sampled = [cmap(i) for i in np.linspace(0, 1, n)]
    return sampled

def annotate_series(ax, xs, ys, label, color, square_box=True, alpha=0.75):
    """
    Annotate series on axis `ax` near the last non-NaN point.
    Annotation uses bold font and a square bbox if requested.
    Now semi-transparent according to `alpha` (0..1).
    """
    if not xs or not ys:
        return
    for i in range(len(xs)-1, -1, -1):
        y = ys[i]
        if y is None:
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        x = xs[i]
        try:
            # use an RGBA white fill so bbox is semi-transparent
            bbox_fc = (1.0, 1.0, 1.0, alpha)
            bbox = dict(boxstyle='square,pad=0.25') if square_box else dict(boxstyle='round,pad=0.2')
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, 2),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color=color,
                alpha=alpha,  # makes the text semi-transparent too
                bbox=dict(boxstyle='square,pad=0.25', fc=bbox_fc, ec=color, lw=0.6),
                arrowprops=dict(arrowstyle='->', lw=0.5, color=color, shrinkA=2, shrinkB=2, alpha=alpha)
            )
        except Exception:
            pass
        break

def plot_two_figures(series_data, means_dict, title_stg=None, title_spp=None, save_prefix=None, min_nkv=60000):
    """
    Plot two figures (S_TG and S_PP). Legend placed on the right (outside).
    Only annotate series that do NOT reach min_nkv.
    """
    nseries = len(series_data)
    colors = make_color_list(max(nseries, 1))
    markers = MARKERS

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # N_KV vs S_TG
    fig2, ax2 = plt.subplots(figsize=(9, 6))  # N_KV vs S_PP

    # keep handles so we can reorder legend later
    handles_by_fn = {}

    # prepare ordered label lists for the two metrics (best->worst) from means_dict
    rows = []
    for fn, v in means_dict.items():
        label = make_label_from_filename(fn)
        model = extract_model_from_filename(fn)
        s_tg = v.get('s_tg_mean')
        s_pp = v.get('s_pp_mean')
        rows.append((fn, label, model, s_tg, s_pp, v.get('n_pts')))

    rows_s_tg = sorted([r for r in rows if r[3] is not None], key=lambda z: z[3], reverse=True)
    rows_s_pp = sorted([r for r in rows if r[4] is not None], key=lambda z: z[4], reverse=True)

    # plot and collect handles
    for i, fn in enumerate(sorted(series_data.keys())):
        d = series_data[fn]
        xs = d['nkv']
        ys_stg = d['s_tg']
        ys_spp = d['s_pp']
        label = make_label_from_filename(fn)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        l1, = ax1.plot(xs, ys_stg, marker=marker, markersize=3, markeredgewidth=0.35,
                       linestyle='-', label=label, color=color, linewidth=0.8, alpha=0.95)
        l2, = ax2.plot(xs, ys_spp, marker=markers[(i+1) % len(markers)], markersize=3, markeredgewidth=0.35,
                       linestyle='--', label=label, color=color, linewidth=0.8, alpha=0.95)
        handles_by_fn[fn] = (l1, l2)

        # annotate only if this series does NOT reach min_nkv
        max_nkv = max(d.get('nkv', [0]))
        if max_nkv < min_nkv:
            annotate_series(ax1, xs, ys_stg, label, color, square_box=True)
            annotate_series(ax2, xs, ys_spp, label, color, square_box=True)

    # build ordered legend for ax1 (S_TG): eligible best->worst first, then others
    ordered_fns_tg = [r[0] for r in rows_s_tg]
    remaining_fns_tg = [fn for fn in sorted(series_data.keys()) if fn not in ordered_fns_tg]
    legend_order_tg = ordered_fns_tg + remaining_fns_tg
    handles_tg = [handles_by_fn[fn][0] for fn in legend_order_tg if fn in handles_by_fn]
    labels_tg = [make_label_from_filename(fn) for fn in legend_order_tg if fn in handles_by_fn]
    if handles_tg:
        # place legend outside to the right
        ax1.legend(handles=handles_tg, labels=labels_tg, title="Series (best to worst)",
                   loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', frameon=True)

    # build ordered legend for ax2 (S_PP)
    ordered_fns_pp = [r[0] for r in rows_s_pp]
    remaining_fns_pp = [fn for fn in sorted(series_data.keys()) if fn not in ordered_fns_pp]
    legend_order_pp = ordered_fns_pp + remaining_fns_pp
    handles_pp = [handles_by_fn[fn][1] for fn in legend_order_pp if fn in handles_by_fn]
    labels_pp = [make_label_from_filename(fn) for fn in legend_order_pp if fn in handles_by_fn]
    if handles_pp:
        ax2.legend(handles=handles_pp, labels=labels_pp, title="Series (best to worst)",
                   loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', frameon=True)

    # make room on the right for the legends
    fig1.subplots_adjust(right=0.72)
    fig2.subplots_adjust(right=0.72)

    ax1.set_xlabel("N_KV")
    ax1.set_ylabel("S_TG t/s")
    ax1.set_title(title_stg or "N_KV vs S_TG (t/s)")
    ax1.grid(True, linestyle=':', linewidth=0.4)

    ax2.set_xlabel("N_KV")
    ax2.set_ylabel("S_PP t/s")
    ax2.set_title(title_spp or "N_KV vs S_PP (t/s)")
    ax2.grid(True, linestyle=':', linewidth=0.4)

    plt.tight_layout()
    plt.show(block=False)

    if save_prefix:
        fig1.savefig(f"{save_prefix}_stg.png", dpi=200, bbox_inches='tight')
        fig2.savefig(f"{save_prefix}_spp.png", dpi=200, bbox_inches='tight')
        print(f"Saved {save_prefix}_stg.png and {save_prefix}_spp.png")

def collect_lines_from_directory(directory, pattern):
    """
    Scan `directory` for files matching `pattern` (glob), read each file,
    and for every line that starts with '|' (ignoring leading whitespace) return a
    synthetic line 'basename:| ...' to match the parser expectation.
    """
    lines = []
    glob_pattern = os.path.join(directory, pattern)
    files = sorted(glob.glob(glob_pattern))
    for path in files:
        if not os.path.isfile(path):
            continue
        basename = os.path.basename(path)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                for raw in fh:
                    if raw.lstrip().startswith('|'):
                        # avoid backslash in f-string expression: use concatenation
                        lines.append(basename + ':' + raw.rstrip('\n'))
        except Exception:
            # skip files we can't read
            continue
    return lines

def main():
    ap = argparse.ArgumentParser(description="Plot N_KV vs S_TG and N_KV vs S_PP from grep output")
    ap.add_argument("--infile", "-i", help="Input file with grep output or a directory (see --dir). If omitted reads stdin.")
    ap.add_argument("--dir", "-d", help="Directory to scan; when provided, script will scan files matching --pattern and extract lines starting with '|'")
    ap.add_argument("--pattern", "-p", default="*.txt", help="Glob pattern for files inside directory (default: '*.txt')")
    ap.add_argument("--outprefix", "-o", help="Save plots with this prefix (two files added: _stg.png and _spp.png). If omitted, do not save.")
    ap.add_argument("--title", "-t", default="N_KV vs S_TG / S_PP t/s", help="Plot title (short base title; model prefix will be prepended automatically)")
    ap.add_argument("--min-nkv", type=int, default=60000, help="Minimum max(N_KV) required for a series to be eligible when choosing the 'best' series (default: 60000)")
    ap.add_argument("--margin-pp", type=float, default=2.5, help="Margin percent for S_PP top-mean closeness (default: 2.5)")
    ap.add_argument("--margin-tg", type=float, default=0.5, help="Margin percent for S_TG top-mean closeness (default: 0.5)")
    args = ap.parse_args()

    print_metric_help()

    lines = []
    # Priority: explicit --dir > infile-is-directory > infile-file > stdin
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"--dir provided but is not a directory: {args.dir}")
            sys.exit(1)
        lines = collect_lines_from_directory(args.dir, args.pattern)
    elif args.infile and os.path.isdir(args.infile):
        lines = collect_lines_from_directory(args.infile, args.pattern)
    elif args.infile:
        try:
            with open(args.infile, 'r', encoding='utf-8') as f:
                lines = [L.rstrip("\n") for L in f]
        except Exception as e:
            print(f"Failed to open infile {args.infile}: {e}")
            sys.exit(1)
    else:
        if sys.stdin.isatty():
            print("Reading from stdin but stdin is a TTY and no input provided. Use --dir or --infile or pipe data in.")
            sys.exit(1)
        lines = [L.rstrip("\n") for L in sys.stdin]

    series_data = build_series(lines)
    if not series_data:
        print("No series found. Ensure input lines are like 'filename:| ... |' or provide a directory with files containing lines that start with '|'.")
        sys.exit(1)

    models = sorted({extract_model_from_filename(fn) for fn in series_data.keys()})
    if len(models) == 1:
        model_prefix = f"{models[0]} - "
    else:
        model_prefix = f"models: {', '.join(models)} - "

    title_stg = f"{model_prefix}N_KV vs S_TG (t/s)"
    title_spp = f"{model_prefix}N_KV vs S_PP (t/s)"

    cutoff, means_dict = compute_means_for_eligible(series_data, args.min_nkv)
    print_means_summary_with_best(means_dict, cutoff, args.min_nkv, args.margin_pp, args.margin_tg)

    def sigint_handler(signum, frame):
        print("\nReceived Ctrl+C — closing plots and exiting.")
        try:
            plt.close('all')
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # pass means_dict into plotting so we can order the default legend
    plot_two_figures(series_data, means_dict, title_stg=title_stg, title_spp=title_spp, save_prefix=args.outprefix, min_nkv=args.min_nkv)

    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: closing plots.")
        plt.close('all')
        sys.exit(0)

if __name__ == '__main__':
    main()
