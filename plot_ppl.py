#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** plot_ppl.py a useful ppl/kld/topP plot utility designed   **#
#** to visualise tensor quantisation sensitiveness patterns.  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jan-03-2026 -------------------- **#
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
#** Copyright © 2025 - Thireus.                  Bᵢₐₛ ᵦₐₖₑ𝒹 ᵢₙ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install pandas numpy matplotlib

# Example:
# python3 plot_ppl.py ppl_results.csv --tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight'

import os
import argparse, re, sys, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import CheckButtons, Button
import tkinter as tk
from tkinter import simpledialog

DEBUG = False

def parse_args():
    p = argparse.ArgumentParser(
        description='Interactive METRIC % bar chart with dynamic QTYPE and tensor filtering.'
    )
    p.add_argument('csv_file', help='Path to the primary CSV file (OUTPUT_CSV)')
    p.add_argument('--interp_csv', help='Path to the second CSV file with interpolated %% results', default=None)
    p.add_argument('--qtypes', nargs='+', default=None,
                   help='List of QTYPEs to pre-select (default: all)')
    p.add_argument('--tensors', nargs='+', default=None,
                   help='List of regex patterns for tensor pre-filter')
    p.add_argument('--metric-name', default='', help='Name of the metric which will appear on the graph (default obtains it from filename if possible, otherwise uses "PPL")')
    return p.parse_args()


def load_and_parse(path):
    df = pd.read_csv(path)
    if 'QTYPE' not in df.columns:
        sys.exit("CSV must contain 'QTYPE' header")
    df = df.set_index('QTYPE')

    # detect if any cell contains a '%' sign
    raw_strings = df.astype(str)
    has_pct = raw_strings.stack().str.contains('%').any()

    def to_pct(x):
        s = str(x).strip()
        if pd.isna(x) or s in ('', '404', '404%'):
            return np.nan
        v = float(s.rstrip('%'))
        return np.nan if v == 404 else v

    # convert all values to floats, stripping '%' if present
    num_df = pd.DataFrame({c: df[c].apply(to_pct) for c in df.columns}, index=df.index)

    # if no '%' in data, compute baseline and adjust to percent relative to baseline
    if not has_pct:
        vals = num_df.values.flatten()
        valid = [v for v in vals if not pd.isna(v) and v != 0]
        if valid:
            baseline = min(valid)
            num_df = (num_df - baseline) / baseline * 100

    # drop any QTYPEs with no data
    num_df = num_df.dropna(how='all')
    return num_df, raw_strings


def load_interp(path):
    """Load interpolated CSV: values already in % (no normalization)."""
    df = pd.read_csv(path)
    if 'QTYPE' not in df.columns:
        sys.exit("Interpolated CSV must contain 'QTYPE' header")
    df = df.set_index('QTYPE')

    def to_pct(x):
        s = str(x).strip()
        if pd.isna(x) or s in ('', '404', '404%'):
            return np.nan
        v = float(s.rstrip('%'))
        return np.nan if v == 404 else v

    interp_df = pd.DataFrame({c: df[c].apply(to_pct) for c in df.columns}, index=df.index)
    return interp_df.dropna(how='all')


def extract_layer(name):
    m = re.match(r'blk\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def draw_chart(metric_name, pct_df, interp_df, qtypes, patterns, ax):
    # filter primary data by QTYPEs and tensor patterns
    sub = pct_df.loc[qtypes]
    cols = list(sub.columns)
    if patterns:
        keep = []
        for pat in patterns:
            rx = re.compile(pat)
            keep += [c for c in cols if rx.search(c)]
        cols = [c for c in cols if c in dict.fromkeys(keep)]
    sub = sub[cols]

    # long-form for primary
    long1 = sub.reset_index().melt('QTYPE', var_name='tensor', value_name='pct')
    long1['layer'] = long1['tensor'].apply(extract_layer)

    # long-form for interpolated (if provided)
    long2 = None
    if interp_df is not None:
        sub2 = interp_df.reindex(index=qtypes, columns=cols)
        long2 = sub2.reset_index().melt('QTYPE', var_name='tensor', value_name='pct')
        long2['layer'] = long2['tensor'].apply(extract_layer)

    ax.clear()
    # combine values to determine y-limits
    vals = long1['pct'].dropna()
    if long2 is not None:
        vals = pd.concat([vals, long2['pct'].dropna()])
    if vals.empty:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
        return

    ymin, ymax = vals.min(), vals.max()
    order = cols

    # bar positioning
    n, m = len(order), len(qtypes)
    x = np.arange(n)
    width = 0.8 / m  # total span 0.8 across qtypes

    # pivot for faster lookup
    primary_pivot = sub
    interp_pivot = interp_df.reindex(index=qtypes, columns=cols) if interp_df is not None else None

    # plot bars: blue for primary when present, red for interpolated only where primary missing
    for i, qt in enumerate(qtypes):
        heights = []
        bar_colors = []
        for t in order:
            # Try to fetch the primary value
            try:
                primary_val = primary_pivot.at[qt, t]
            except KeyError:
                primary_val = np.nan

            has_primary = pd.notna(primary_val)

            if has_primary:
                heights.append(primary_val)
                bar_colors.append('blue')
                if DEBUG: print(f"[DEBUG] QTYPE: {qt}, Tensor: {t} — primary = {primary_val:.2f} → color = blue")
            else:
                # Try interpolated fallback
                if interp_pivot is not None:
                    try:
                        interp_val = interp_pivot.at[qt, t]
                    except KeyError:
                        interp_val = np.nan
                else:
                    interp_val = np.nan

                has_interp = pd.notna(interp_val)
                heights.append(interp_val if has_interp else np.nan)
                bar_colors.append('red' if has_interp else 'gray')  # Optional gray for fully missing
                interp_str = f"{interp_val:.2f}" if has_interp else "NaN"
                if DEBUG: print(f"[DEBUG] QTYPE: {qt}, Tensor: {t} — primary = NaN, interpolated = {interp_str} → color = {'red' if has_interp else 'gray'}")

        ax.bar(x + i * width, heights, width=width, color=bar_colors, label=f"{qt}")


    ax.set_xticks(x + 0.8 / 2 - width / 2)
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel(f"{metric_name} %")
    ax.legend(title='QTYPE')
    ax.set_title(f"Interactive {metric_name} % by QTYPE and Tensor")


def print_summary(pct_df, raw_strings, patterns=None):
    # Overall diagnostics
    total_cells = pct_df.size
    missing_count = pct_df.isna().sum().sum()
    missing_pct = missing_count / total_cells * 100 if total_cells else 0
    mask_404 = raw_strings.isin(['404', '404%'])
    count_404 = int(mask_404.values.sum())
    print("Processed data summary:")
    print(f"  Total cells: {total_cells}")
    print(f"  Missing entries (NaN): {missing_count} ({missing_pct:.2f}%)")
    print(f"  404 entries (failed benchmarks): {count_404}")
    if count_404:
        s404 = mask_404.stack()
        details = [idx for idx, flag in zip(s404.index, s404.values) if flag]
        for qtype, tensor in details:
            print(f"    - {qtype}: {tensor}")
    print(f"  Tensors: {pct_df.shape[1]}, QTYPEs: {pct_df.shape[0]}")
    print()

    if patterns:
        print("Filter summaries:")
        for pat in patterns:
            rx = re.compile(pat)
            matched = [c for c in pct_df.columns if rx.search(c)]
            cnt = len(matched)
            total_entries = cnt * pct_df.shape[0]
            missing = pct_df[matched].isna().sum().sum() if cnt else 0
            missing_pct = missing / total_entries * 100 if total_entries else 0
            print(f"  Pattern '{pat}': {cnt} tensors, {missing} missing of {total_entries} entries ({missing_pct:.2f}% missing)")
        print()

    # Global value metrics (excluding NaN/404)
    all_vals = pct_df.values.flatten()
    valid_vals = all_vals[~np.isnan(all_vals)]
    if valid_vals.size:
        print(f"Global stats (raw values): mean={valid_vals.mean():.4f}, median={np.median(valid_vals):.4f}, std={valid_vals.std():.4f}, range=({valid_vals.min():.4f} to {valid_vals.max():.4f})")
    print()

    # Per-QTYPE detailed metrics
    print("Per-QTYPE detailed stats (excluding NaN/404):")
    for qt in pct_df.index:
        series = pct_df.loc[qt].dropna()
        if series.empty:
            continue
        arr = series.values
        mean = arr.mean()
        median = np.median(arr)
        std = arr.std()
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        min_val, max_val = arr.min(), arr.max()
        min_t = series[series == min_val].index.tolist()
        max_t = series[series == max_val].index.tolist()
        print(f"  {qt}: count={len(arr)}, mean={mean:.4f}, median={median:.4f}, std={std:.4f}, IQR={iqr:.4f}")
        print(f"       min={min_val:.4f} at {min_t}; max={max_val:.4f} at {max_t}")
    print()

    # Per-QTYPE missing rates
    print("Per-QTYPE missing rates:")
    for qt in pct_df.index:
        m = pct_df.loc[qt].isna().sum()
        total = pct_df.shape[1]
        print(f"  {qt}: {m}/{total} missing ({m/total*100:.2f}% missing)")
    print()


def main():
    args = parse_args()
    csv_path = args.csv_file
    interp_df = load_interp(args.interp_csv) if args.interp_csv else None

    pct_df, raw_strings = load_and_parse(csv_path)
    selected = args.qtypes.copy() if args.qtypes else sorted(pct_df.index).copy()
    patterns = args.tensors.copy() if args.tensors else []
    base_presets = patterns.copy()
    toggle_state = True  # True => all checked; button reads “Unselect All”

    metric_name = args.metric_name

    # Obtain metric name
    if not metric_name:
        # Try to extract metric name from filename pattern like "metricname_results.csv"
        filename = os.path.basename(csv_path)
        match = re.match(r"([a-zA-Z0-9_-]+)_results", filename)
        if match:
            metric_name = match.group(1).upper()
        else:
            metric_name = "PPL"

    # Initial diagnostics & summary
    print_summary(pct_df, raw_strings, patterns)

    # Setup plot environment
    plt.rcParams.update({'font.size': 8})
    root = tk.Tk()
    root.withdraw()
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.217, right=0.995, top=0.971, bottom=0.217)
    draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)

    # QTYPE selector
    all_q = sorted(pct_df.index)
    q_ax = plt.axes((0.05, 0.60, 0.10, 0.30))
    q_checks = CheckButtons(q_ax, all_q, [q in selected for q in all_q])
    def on_q(label):
        if label in selected: selected.remove(label)
        else: selected.append(label)
        draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)
        fig.canvas.draw_idle()
    q_checks.on_clicked(on_q)

    # Reload CSV button
    r_ax = plt.axes((0.05, 0.53, 0.10, 0.05))
    btn_reload = Button(r_ax, 'Reload CSV')
    def do_reload(event):
        nonlocal pct_df, raw_strings, interp_df
        pct_df, raw_strings = load_and_parse(csv_path)
        if args.interp_csv:
            interp_df = load_interp(args.interp_csv)
        print_summary(pct_df, raw_strings, patterns)
        draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)
        fig.canvas.draw_idle()
    btn_reload.on_clicked(do_reload)

    # Edit regex (asks a string; Cancel returns None and does nothing)
    e_ax = plt.axes((0.05, 0.46, 0.10, 0.05))
    btn_edit = Button(e_ax, 'Edit tensor filters')
    def do_edit(event):
        nonlocal patterns, base_presets, toggle_state
        txt = simpledialog.askstring(
            'Tensor filter regex',
            'Enter comma-separated regex patterns:',
            initialvalue=','.join(patterns)
        )
        if txt is None:
            return    # Cancel: just close the dialog
        patterns[:] = [t.strip() for t in txt.split(',') if t.strip()]
        base_presets = patterns.copy()
        toggle_state = True
        update_presets()
        draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)
        fig.canvas.draw_idle()
    btn_edit.on_clicked(do_edit)

    # Presets panel & toggle button
    p_ax = plt.axes((0.05, 0.16, 0.10, 0.28))
    t_ax = plt.axes((0.05, 0.10, 0.10, 0.05))
    btn_toggle = Button(t_ax, 'Unselect All')

    # We'll store the CheckButtons here so it stays alive
    preset_check = None

    def update_presets():
        nonlocal preset_check, base_presets, toggle_state, patterns

        p_ax.clear()
        if not base_presets:
            p_ax.set_visible(False)
            btn_toggle.ax.set_visible(False)
            return

        p_ax.set_visible(True)
        btn_toggle.ax.set_visible(True)

        labels = ['\n'.join(textwrap.wrap(p,25)) for p in base_presets]
        states = [toggle_state]*len(labels)

        # recreate the widget and keep a reference
        preset_check = CheckButtons(p_ax, labels, states)

        def on_preset(label):
            nonlocal patterns, toggle_state
            idx = labels.index(label)
            pat = base_presets[idx]
            status = preset_check.get_status()[idx] # type: ignore
            if status and pat not in patterns:
                patterns.append(pat)
            elif not status and pat in patterns:
                patterns.remove(pat)

            # adjust the toggle button label
            toggle_state = (len(patterns) == len(base_presets))
            btn_toggle.label.set_text(
                'Unselect All' if toggle_state else 'Select All'
            )
            draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)
            fig.canvas.draw_idle()

        preset_check.on_clicked(on_preset)

    def do_toggle(event):
        nonlocal patterns, toggle_state
        if toggle_state:
            patterns.clear()
            toggle_state = False
            btn_toggle.label.set_text('Select All')
        else:
            patterns[:] = base_presets
            toggle_state = True
            btn_toggle.label.set_text('Unselect All')
        update_presets()
        draw_chart(metric_name, pct_df, interp_df, selected, patterns, ax)
        fig.canvas.draw_idle()

    btn_toggle.on_clicked(do_toggle)

    # draw them once (or hide if empty)
    update_presets()

    plt.show()


if __name__ == '__main__':
    main()
