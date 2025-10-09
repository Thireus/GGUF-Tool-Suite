#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** compare_results.py is a tool that helps idenfity the      **#
#** qtypes that provide the best quality/speed/compression.   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Oct-09-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.            Dₒₙ'ₜ ₜᵣᵤₛₜ, ᵥₑᵣᵢ𝒻ᵧ! **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Example:
# ../compare_results.py --metric-results metric_results.csv --pp-results pp_results_cpu-Intel-7980XE.csv,pp_results_cuda-RTX-6000-Pro.csv --tg-results tg_results_cpu-Intel-7980XE.csv,tg_results_cuda-RTX-6000-Pro.csv --group-tensors '.*' --show --export-csv quant_score --export-dir .

"""
compare_results.py

Compare METRIC / PP / TG CSVs, compute weight reductions from tensors.*.map (using bytes= field),
compute combined scores and plot.

Usage example:
  python compare_results.py \
    --metric-results metric_results.csv \
    --pp-results pp_results_cpu.csv,pp_results_cuda.csv \
    --tg-results tg_results_cpu.csv,tg_results_cuda.csv \
    --group-tensors '.*' \
    --export-dir ./plots \
    --export-csv combined_scores \
    --metric-name "ppl" \
    --show
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
import signal
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# ============== USER CONFIGURATION ==============
USER_REGEX = [
  r'^blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_gate\.weight$',
  r'^blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_down\.weight$',
  r'^blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_up\.weight$',
]
# =========== End USER CONFIGURATION ============


def warning(msg: str, exit_code: int = 1) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)

def error(msg: str, exit_code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def parse_csv_file(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not os.path.isfile(path):
        error(f"CSV file not found: {path}")
    rows: Dict[str, Dict[str, str]] = {}
    columns: List[str] = []
    # initialize hdr to satisfy static analyzers that it is always defined
    hdr: List[str] = []

    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        try:
            hdr = next(reader)
        except StopIteration:
            error(f"CSV file {path} is empty")
        # sanity check header
        if len(hdr) < 2 or hdr[0].strip() != 'QTYPE':
            error(f"CSV '{path}' must have 'QTYPE' as first header column. Found: {hdr}")
        columns = [h.strip() for h in hdr[1:]]
        for i, line in enumerate(reader, start=2):
            if len(line) == 0:
                continue
            if len(line[1:]) != len(columns):
                error(f"Row for qtype in {path} at line {i} has {len(line[1:])} columns but header has {len(columns)} columns.")
            qtype = line[0].strip()
            values = [v.strip() for v in line[1:]]
            if qtype in rows:
                error(f"Duplicate QTYPE '{qtype}' found in {path}")
            rows[qtype] = {col: val for col, val in zip(columns, values)}
    return columns, rows


def parse_pct_value(s: str) -> float:
    if s is None:
        raise ValueError("missing")
    s = s.strip()
    if s == '':
        raise ValueError("empty")
    if not s.endswith('%'):
        raise ValueError("missing-percent")
    s2 = s[:-1].strip()
    try:
        return float(s2.replace('+', ''))
    except Exception as e:
        raise ValueError(f"not-a-number:{s2}") from e


def read_map_file(qtype: str) -> List[Tuple[str, str, int]]:
    """
    Read tensors.<qtype>.map and return list of (fname, tensor_name, bytes_int).
    Expects lines like:
      file.gguf:<hash>:tensor.name:...:elements=...:bytes=777912320
    If bytes= is missing or cannot be parsed, the line is skipped (with a warning).
    """
    fn = f"tensors.{qtype}.map"
    if not os.path.isfile(fn):
        error(f"Expected map file not found: {fn}")
    result: List[Tuple[str, str, int]] = []
    with open(fn, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if ln == '':
                continue
            parts = ln.split(':')
            if len(parts) < 3:
                print(f"[WARN] Unexpected map line format in {fn}: {ln}", file=sys.stderr)
                continue
            fname = parts[0]
            tensor_name = parts[2]
            # try to find bytes=NNN in the remainder of the line
            bytes_val = None
            tail = ':'.join(parts[3:]) if len(parts) > 3 else ''
            m = re.search(r'bytes=([0-9]+)', tail)
            if m:
                try:
                    bytes_val = int(m.group(1))
                except ValueError:
                    bytes_val = None
            if bytes_val is None:
                print(f"[WARN] Could not find bytes= in map line for tensor '{tensor_name}' in {fn}. Line: {ln}", file=sys.stderr)
                continue
            result.append((fname, tensor_name, bytes_val))
    return result


def select_tensors_from_map(qtype: str, user_regexes: List[re.Pattern]) -> Dict[str, Dict[str, object]]:
    """
    Returns map tensor_name -> {'fname': fname, 'bytes': bytes} for tensors in tensors.<qtype>.map that match any user_regex.
    """
    entries = read_map_file(qtype)
    out: Dict[str, Dict[str, object]] = {}
    for fname, tname, b in entries:
        matched = any(rx.search(tname) for rx in user_regexes)
        if matched:
            out[tname] = {'fname': fname, 'bytes': b}
    return out


def build_group_members(qtype: str, group_specs: List[List[re.Pattern]], tensor_names_in_map: List[str]) -> Dict[int, List[str]]:
    gm: Dict[int, List[str]] = {}
    for i, regs in enumerate(group_specs):
        members = []
        for t in tensor_names_in_map:
            if any(r.search(t) for r in regs):
                members.append(t)
        gm[i] = sorted(set(members))
    return gm


def compute_total_size_for_set(qtype: str, tensor_set: List[str], tensor_meta_map: Dict[str, Dict[str, object]]) -> int:
    """
    Sum the 'bytes' values from the tensor_meta_map for the given tensor_set.
    tensor_meta_map: mapping tensor_name -> {'fname':..., 'bytes': <int>}
    Errors out if a tensor in tensor_set is not found in the meta map.
    """
    total = 0
    for t in tensor_set:
        if t not in tensor_meta_map:
            error(f"Tensor '{t}' not found in map for qtype '{qtype}'. Cannot compute size.")
        meta = tensor_meta_map[t]
        b = meta.get('bytes')
        if b is None:
            error(f"Tensor '{t}' in map for qtype '{qtype}' does not contain bytes information.")
        total += int(b) # type: ignore
    return total


def parse_group_specs(raw_list: List[str]) -> List[List[re.Pattern]]:
    out: List[List[re.Pattern]] = []
    for spec in raw_list:
        parts = [p.strip() for p in spec.split(',') if p.strip() != '']
        if len(parts) == 0:
            continue
        compiled = []
        for p in parts:
            try:
                compiled.append(re.compile(p))
            except re.error as e:
                error(f"Invalid regex in group spec '{p}': {e}")
        out.append(compiled)
    return out


def extract_suffix_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    name = base.rsplit('.', 1)[0]
    if '_' in name:
        return name.split('_')[-1]
    return name


def format_pct(v: Optional[float]) -> str:
    if v is None:
        return ""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare METRIC/PP/TG CSVs, compute weight reductions and combined scores, and plot."
    )
    parser.add_argument('--metric-results', required=True, help='METRIC results CSV (required)')
    parser.add_argument('--pp-results', default='', help='Comma-separated list of PP results CSV files (optional)')
    parser.add_argument('--tg-results', default='', help='Comma-separated list of TG results CSV files (optional)')
    parser.add_argument('--group-tensors', nargs='*', default=[], help="One or more group specs (each is a comma-separated list of regexes).")
    parser.add_argument('--metric-factor', type=float, default=1.0, help='Weight for metric in score (default 1.0)')
    parser.add_argument('--tg-factor', type=float, default=1.0, help='Weight for TG in score (default 1.0)')
    parser.add_argument('--pp-factor', type=float, default=1.0, help='Weight for PP in score (default 1.0)')
    parser.add_argument('--score-cutoff', type=float, default=-100.0, help='Minimum score cutoff; scores below this are clamped (default -100.0)')
    parser.add_argument('--export-dir', default='', help='Directory to save plots (optional)')
    parser.add_argument('--export-csv', default='', help='Base filename to export per-suffix CSVs (optional). Example: --export-csv combined_scores')
    parser.add_argument('--metric-name', default='', help='Name of the metric which will appear in the csv and plot files (default obtains it from filename if possible, otherwise uses "ppl")')
    parser.add_argument('--show', action='store_true', help='Display plots interactively (all at once, blocking)')
    parser.add_argument('--verbose', action='store_true', help='Verbose debug output')

    args = parser.parse_args(argv)

    metric_csv = args.metric_results
    pp_csvs = [s.strip() for s in args.pp_results.split(',') if s.strip()] if args.pp_results else []
    tg_csvs = [s.strip() for s in args.tg_results.split(',') if s.strip()] if args.tg_results else []
    group_specs_raw = args.group_tensors or []
    metric_factor, tg_factor, pp_factor = args.metric_factor, args.tg_factor, args.pp_factor
    score_cutoff = args.score_cutoff
    export_dir = args.export_dir or None
    export_csv_base = args.export_csv or None
    metric_name = args.metric_name
    show = args.show
    verbose = args.verbose

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    if verbose:
        print(f"[INFO] Loading metric CSV: {metric_csv}", file=sys.stderr)
    metric_cols, metric_rows = parse_csv_file(metric_csv)

    pp_colsets = {}
    for fn in pp_csvs:
        if verbose:
            print(f"[INFO] Loading PP CSV: {fn}", file=sys.stderr)
        cols, rows = parse_csv_file(fn)
        pp_colsets[fn] = (cols, rows)

    tg_colsets = {}
    for fn in tg_csvs:
        if verbose:
            print(f"[INFO] Loading TG CSV: {fn}", file=sys.stderr)
        cols, rows = parse_csv_file(fn)
        tg_colsets[fn] = (cols, rows)

    all_csvs = [(metric_csv, metric_cols, metric_rows)]
    all_csvs += [(fn, *pp_colsets[fn]) for fn in pp_colsets]
    all_csvs += [(fn, *tg_colsets[fn]) for fn in tg_colsets]

    # column equality checks
    first_cols = all_csvs[0][1]
    for fn, cols, rows in all_csvs:
        if cols != first_cols:
            error(f"Column mismatch: CSV '{fn}' columns differ from '{all_csvs[0][0]}'.\n  {all_csvs[0][0]} columns: {first_cols}\n  {fn} columns: {cols}")

    # QTYPE set equality
    qtype_sets = {fn: set(rows.keys()) for fn, cols, rows in all_csvs}
    base_qtypes = qtype_sets[metric_csv]
    for fn, s in qtype_sets.items():
        if s != base_qtypes:
            missing_in_fn = base_qtypes - s
            missing_in_base = s - base_qtypes
            msg = []
            if missing_in_fn:
                msg.append(f"QTYPEs {sorted(missing_in_fn)} present in {metric_csv} but missing in {fn}")
            if missing_in_base:
                msg.append(f"QTYPEs {sorted(missing_in_base)} present in {fn} but missing in {metric_csv}")
            warning("QTYPE mismatch between CSVs: " + "; ".join(msg))
            # Keep only the intersection of valid QTYPEs across all files
            valid_qtypes = base_qtypes & s
            qtype_sets[fn] = valid_qtypes

    # Update base_qtypes to reflect only those present in *all* CSVs
    base_qtypes = set.intersection(*qtype_sets.values())
    QTYPES = sorted(base_qtypes)
    if verbose:
        print(f"[INFO] QTYPEs: {QTYPES}", file=sys.stderr)

    # Validate cell formats
    for fn, cols, rows in all_csvs:
        for q in QTYPES:
            for col in cols:
                val = rows[q][col]
                if val == '' or val is None:
                    error(f"Empty value found in {fn} for QTYPE='{q}', column='{col}'")
                if not val.endswith('%'):
                    error(f"Non-percent value found in {fn} for QTYPE='{q}', column='{col}': '{val}'")

    # Obtain metric name
    if not metric_name:
        # Try to extract metric name from filename pattern like "metricname_results.csv"
        filename = os.path.basename(metric_csv)
        match = re.match(r"([a-zA-Z0-9_-]+)_results", filename)
        if match:
            metric_name = match.group(1)
        else:
            metric_name = "ppl"

    # build suffix maps
    suffixes: Set[str] = set()
    pp_by_suffix: Dict[str, Tuple[List[str], Dict[str, Dict[str, str]]]] = {}
    tg_by_suffix: Dict[str, Tuple[List[str], Dict[str, Dict[str, str]]]] = {}

    for fn, (cols, rows) in pp_colsets.items():
        suff = extract_suffix_from_filename(fn)
        pp_by_suffix[suff] = (cols, rows)
        suffixes.add(suff)
    for fn, (cols, rows) in tg_colsets.items():
        suff = extract_suffix_from_filename(fn)
        tg_by_suffix[suff] = (cols, rows)
        suffixes.add(suff)
    if len(suffixes) == 0:
        suffixes = {f'{metric_name.lower()}-only'}

    # compile user regexes
    compiled_user_regexes = []
    for pat in USER_REGEX:
        try:
            compiled_user_regexes.append(re.compile(pat))
        except re.error as e:
            error(f"Invalid USER_REGEX pattern '{pat}': {e}")

    group_specs = parse_group_specs(group_specs_raw)

    if 'bf16' not in QTYPES:
        error("bf16 QTYPE must be present in CSVs for comparing size reductions (bf16 baseline missing)")

    # Build per-qtype tensor metadata (using bytes= field)
    per_q_tensor_meta: Dict[str, Dict[str, Dict[str, object]]] = {}
    per_q_tensor_names: Dict[str, List[str]] = {}
    for q in QTYPES:
        selected = select_tensors_from_map(q, compiled_user_regexes)
        per_q_tensor_meta[q] = selected
        per_q_tensor_names[q] = sorted(selected.keys())
        if verbose:
            print(f"[DEBUG] qtype={q} selected {len(selected)} tensors", file=sys.stderr)

    # group members
    per_q_group_members: Dict[str, Dict[int, List[str]]] = {}
    if group_specs:
        for q in QTYPES:
            per_q_group_members[q] = build_group_members(q, group_specs, per_q_tensor_names[q])

    # decide header columns (from first csv)
    header_columns = first_cols

    # build column -> tensor set resolver
    col_to_tensorset_func = {}
    for col in header_columns:
        if col.startswith('group') and group_specs:
            m = re.match(r'group(\d+)$', col)
            if m is None:
                error(f"Column named '{col}' looks like a group column but doesn't match 'groupN' format")
            # help static checkers: m is definitely not None here
            assert m is not None
            gidx = int(m.group(1))
            if gidx < 0 or gidx >= len(group_specs):
                error(f"Column '{col}' references group index {gidx} but only {len(group_specs)} groups were provided")
            def make_func(gindex):
                def func(q: str):
                    return per_q_group_members.get(q, {}).get(gindex, [])
                return func
            col_to_tensorset_func[col] = make_func(gidx)
        else:
            def make_single(tname):
                def func(q: str):
                    return [tname] if tname in per_q_tensor_meta.get(q, {}) else []
                return func
            col_to_tensorset_func[col] = make_single(col)

    # compute sizes and reductions using bytes field
    sizes: Dict[str, Dict[str, int]] = defaultdict(dict)
    reductions: Dict[str, Dict[str, float]] = defaultdict(dict)

    for q in QTYPES:
        for col in header_columns:
            tensor_set = col_to_tensorset_func[col](q)
            if len(tensor_set) == 0:
                sizes[q][col] = 0
                reductions[q][col] = 0.0
                continue
            size_q = compute_total_size_for_set(q, tensor_set, per_q_tensor_meta[q])
            size_bf16 = compute_total_size_for_set('bf16', tensor_set, per_q_tensor_meta['bf16'])
            sizes[q][col] = size_q
            if size_bf16 == 0:
                reductions[q][col] = 0.0
            else:
                reductions[q][col] = (size_bf16 - size_q) / size_bf16 * 100.0

    # compute scores per suffix, produce plots (collect figures to display all at once if --show)
    results_by_suffix: Dict[str, Dict[str, float]] = {}
    figures: List[Figure] = []
    axes: List[Axes] = []

    def sigint_handler(signum, frame):
        # Close all figures and exit when Ctrl+C pressed while figures are open.
        try:
            print("\n[INFO] SIGINT received. Closing all plots and exiting.", file=sys.stderr)
        except Exception:
            pass
        plt.close('all')
        sys.exit(130)

    try:
        for suff in sorted(suffixes):
            pp_rows_for_suffix = pp_by_suffix.get(suff, (None, None))[1]
            tg_rows_for_suffix = tg_by_suffix.get(suff, (None, None))[1]

            metric_pct_map: Dict[str, float] = {}
            pp_pct_map: Dict[str, float] = {}
            tg_pct_map: Dict[str, float] = {}

            for q in QTYPES:
                # pre-declare to satisfy static analyzers
                metric_val: float = 0.0
                pp_val: Optional[float] = None
                tg_val: Optional[float] = None

                try:
                    metric_val = parse_pct_value(metric_rows[q][header_columns[0]])
                except Exception as e:
                    error(f"{metric_name.upper()} PARSE ERROR FOR QTYPE={q}: {e}")
                metric_pct_map[q] = metric_val

                if pp_rows_for_suffix:
                    try:
                        pp_val = parse_pct_value(pp_rows_for_suffix[q][header_columns[0]])
                        pp_pct_map[q] = pp_val
                    except Exception as e:
                        error(f"PP parse error for suffix={suff}, QTYPE={q}: {e}")

                if tg_rows_for_suffix:
                    try:
                        tg_val = parse_pct_value(tg_rows_for_suffix[q][header_columns[0]])
                        tg_pct_map[q] = tg_val
                    except Exception as e:
                        error(f"TG parse error for suffix={suff}, QTYPE={q}: {e}")

            scores: Dict[str, float] = {}
            for q in QTYPES:
                metric = metric_pct_map.get(q, 0.0)
                ppv = pp_pct_map.get(q, None)
                tgv = tg_pct_map.get(q, None)
                speed_sum = 0.0
                speed_count = 0
                if tgv is not None:
                    speed_sum += tg_factor * tgv
                    speed_count += 1
                if ppv is not None:
                    speed_sum += pp_factor * ppv
                    speed_count += 1
                avg_speed = speed_sum / speed_count if speed_count > 0 else 0.0
                raw_score = (-metric_factor * metric + avg_speed) / 2.0
                # clamp to cutoff
                score = raw_score if raw_score >= score_cutoff else score_cutoff
                scores[q] = score

            results_by_suffix[suff] = scores

            avg_reduction_per_q: Dict[str, float] = {}
            for q in QTYPES:
                vals = [reductions[q][col] for col in header_columns]
                avg_reduction_per_q[q] = sum(vals) / len(vals) if vals else 0.0

            sorted_qtypes = sorted(QTYPES, key=lambda qq: avg_reduction_per_q[qq], reverse=True)

            # Print table to stdout — note order now: metric_factor, pp_factor, tg_factor, score_cutoff
            print(f"\n=== Scores for suffix '{suff}' (metric_factor={metric_factor}, pp_factor={pp_factor}, tg_factor={tg_factor}, score_cutoff={score_cutoff}) ===")
            print(f"{'QTYPE':20s} {'avg_reduction%':>14s} {'score':>12s} {f'{metric_name.lower()}%':>10s} {'pp%':>10s} {'tg%':>10s}")
            for q in sorted_qtypes:
                sc = scores[q]
                red = avg_reduction_per_q[q]
                metricv = metric_pct_map.get(q, float('nan'))
                ppv = pp_pct_map.get(q, float('nan')) if pp_pct_map else float('nan')
                tgv = tg_pct_map.get(q, float('nan')) if tg_pct_map else float('nan')
                print(f"{q:20s} {red:14.2f}% {sc:12.4f} {metricv:10.2f}% {ppv:10.2f}% {tgv:10.2f}%")

            # CSV export per suffix (if requested)
            if export_csv_base:
                if len(suffixes) > 1 or (len(suffixes) == 1 and next(iter(suffixes)) != f'{metric_name.lower()}-only'):
                    out_csv_name = f"{export_csv_base.rstrip('.csv')}_{suff}.csv"
                else:
                    out_csv_name = f"{export_csv_base if export_csv_base.endswith('.csv') else export_csv_base + '.csv'}"
                if export_dir:
                    out_csv_path = os.path.join(export_dir, out_csv_name)
                else:
                    out_csv_path = out_csv_name
                with open(out_csv_path, 'w', newline='', encoding='utf-8') as fh:
                    writer = csv.writer(fh)
                    hdr = ['QTYPE', 'avg_reduction%', 'score', f'{metric_name.lower()}%', 'pp%', 'tg%']
                    writer.writerow(hdr)
                    for q in sorted_qtypes:
                        writer.writerow([
                            q,
                            f"{avg_reduction_per_q[q]:.2f}",
                            f"{scores[q]:.4f}",
                            format_pct(metric_pct_map.get(q)),
                            format_pct(pp_pct_map.get(q)) if pp_pct_map else "",
                            format_pct(tg_pct_map.get(q)) if tg_pct_map else ""
                        ])
                print(f"[INFO] Saved CSV for suffix '{suff}' to {out_csv_path}")

            # Build plot (store figure to display later if requested)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            xs = [avg_reduction_per_q[q] for q in sorted_qtypes]
            ys = [scores[q] for q in sorted_qtypes]
            labels = sorted_qtypes
            # custom scatter plot with different markers/colors
            for xi, yi, lab in zip(xs, ys, labels):
                if re.match(r"q\d+_\d+", lab):          # q*_INT, e.g. q8_0, q4_1
                    marker, color = '+', 'C0'
                elif lab.startswith("iq"):              # iq**, e.g. iq4_xxs, iq2_ks
                    marker, color = 'x', 'C1'
                elif lab.endswith("_kt"):               # *_kt, e.g. q2_kt
                    marker, color = '.', 'C2'
                else:                                   # all others
                    marker, color = '*', 'C3'

                ax.plot(xi, yi, marker=marker, color=color,
                        markersize=3, linestyle='', alpha=0.75)
                ax.annotate(lab, (xi, yi),
                            textcoords="offset points", xytext=(5, 3),
                            ha='left', fontsize=7)

            ax.set_xlabel("Average Weight Reduction (%) vs bf16")
            ax.set_ylabel("Score")
            title = f"Score vs Weight Reduction ({suff})"
            ax.set_title(title)
            ax.grid(True)

            # show formula and factors as textbox inside plot
            formula_lines = []
            formula_lines.append(f"score = (-{metric_factor} * {metric_name.lower()} + avg_speed) / 2")
            formula_lines.append(f"avg_speed = sum(factor*metric)/count_metrics")
            # show pp_factor before tg_factor as requested, and include cutoff
            formula_lines.append(f"metric_factor={metric_factor}, pp_factor={pp_factor}, tg_factor={tg_factor}, score_cutoff={score_cutoff}")
            textbox = "\n".join(formula_lines)
            ax.text(0.01, 0.99, textbox, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

            plt.tight_layout()

            if export_dir:
                outfn = os.path.join(export_dir, f"score_vs_reduction_{suff}.png")
                fig.savefig(outfn)
                print(f"[INFO] Saved plot for suffix '{suff}' to {outfn}")

            figures.append(fig)
            axes.append(ax)

        # If export only and not showing, close figures after saving
        if export_dir and not show:
            for f in figures:
                plt.close(f)
            print("[INFO] Saved all plots to export-dir and closed figures.")
            print("\n[INFO] Done.")
            return

        # If show requested: register SIGINT handler that closes all figures and exits,
        # then call plt.show() once so all figures are displayed together.
        if show:
            # Draw all figures and enter a small polling loop using plt.pause().
            # plt.pause() processes GUI events and allows KeyboardInterrupt (Ctrl+C)
            # to be delivered reliably across backends.
            try:
                print("[INFO] Showing plots. Press Ctrl+C to close all plots and exit.", file=sys.stderr)
                # make sure canvases drawn at least once
                for fig in figures:
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass

                # poll until all figures are closed or Ctrl+C is pressed
                while plt.get_fignums():
                    # short pause to process GUI events; KeyboardInterrupt will be raised here on Ctrl+C
                    try:
                        plt.pause(0.1)
                    except KeyboardInterrupt:
                        # user pressed Ctrl+C
                        print("\n[INFO] KeyboardInterrupt received. Closing all plots.", file=sys.stderr)
                        break
                    except Exception:
                        # some backends may raise other exceptions occasionally; ignore and continue
                        pass
            finally:
                plt.close('all')
                print("\n[INFO] Done.")
                return

        # neither export nor show: close figures and inform user (same behaviour as before)
        for f in figures:
            plt.close(f)
        print("[INFO] No export dir given and --show not used; plots not saved (displayed only in interactive env).")
        print("\n[INFO] Done.")

    except KeyboardInterrupt:
        # catch during processing (before plotting) - close any figures
        plt.close('all')
        print("\n[INFO] KeyboardInterrupt received during processing. Exiting.", file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()
