#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** plot_ppl.py a useful tensor ppl visualisation utility to  **#
#** identify tensor quantisation sensitiveness patterns.      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-03-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.         Fₐ𝒸ₜ₋𝒸ₕₑ𝒸ₖₛ? Wₕₐₜ’ₛ ₜₕₐₜ? **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install matplotlib

import os
import re
import argparse
import matplotlib

def parse_filename(filename):
    # Extract model name, bpw, ppl from recipe filename
    base = os.path.basename(filename)
    model_name = base.split('.', 1)[0]
    m_bpw = re.search(r"([0-9]+\.?[0-9]*)bpw", base)
    m_ppl = re.search(r"([0-9]+\.?[0-9]*)ppl", base)
    bpw = float(m_bpw.group(1)) if m_bpw else None
    ppl = float(m_ppl.group(1)) if m_ppl else None
    return model_name, bpw, ppl

def collect_recipe_data(recipe_dir):
    data = {}
    for fname in os.listdir(recipe_dir):
        if not fname.endswith('.recipe'):
            continue
        model, bpw, ppl = parse_filename(fname)
        if None in (bpw, ppl):
            continue
        data.setdefault(model, []).append((bpw, ppl))
    return data

def collect_imported_data(import_file):
    # Parse additional series from pipe-delimited DB
    # Format: model_name|author|recipe_name|bpw|ppl|model_link
    imported = {}
    markers = []
    with open(import_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 5:
                continue
            model_name, author = parts[0], parts[1]
            bpw_str, ppl_str = parts[3], parts[4]
            try:
                bpw = float(bpw_str)
                ppl = float(ppl_str)
            except ValueError:
                continue
            # Group by model and author
            imported.setdefault(model_name, {}).setdefault(author, []).append((bpw, ppl))
    return imported

def plot_data(recipe_data, recipe_dir, imported_data=None, export=False, out_dir=None):
    # Use non-interactive backend if exporting
    if export:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dir_name = os.path.basename(os.path.normpath(recipe_dir))
    main_label = f"Thireus' GGUF Tool Suite {dir_name}"
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '<', '>']

    if not export:
        plt.ion()

    for model, vals in recipe_data.items():
        xs_main, ys_main = zip(*sorted(vals))
        fig = plt.figure()
        plt.plot(xs_main, ys_main, marker='x', linestyle='', label=main_label)
        # Plot any imported series for this model
        if imported_data and model in imported_data:
            for idx, (author, series_vals) in enumerate(imported_data[model].items()):
                xs_imp, ys_imp = zip(*sorted(series_vals))
                marker = markers[idx % len(markers)]
                plt.plot(xs_imp, ys_imp, marker=marker, linestyle='', label=author)

        plt.xlabel('Bits per word (bpw)')
        plt.ylabel('Perplexity (ppl)')
        plt.title(f'{model}: ppl vs bpw')
        plt.legend()

        if export:
            filename = f"{model}.svg"
            out_path = os.path.join(out_dir or '.', filename)
            plt.savefig(out_path, format='svg')
            plt.close(fig)
        else:
            fig.show()

    if not export:
        print("Plots are displayed in separate windows. Close them manually when done.")
        input("Press Enter to exit and close all plots...")
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(
        description='Plot ppl vs bpw from recipe filenames, with optional imported series.')
    parser.add_argument('recipe_dir', help='Directory containing .recipe files')
    parser.add_argument('--import', dest='import_file', help='Import additional series from DB file')
    parser.add_argument('--export', action='store_true', help='Export plots as SVG without rendering')
    parser.add_argument('--out-dir', default='.', help='Output directory for exported SVG files')
    args = parser.parse_args()

    recipe_data = collect_recipe_data(args.recipe_dir)
    imported_data = None
    if args.import_file:
        imported_data = collect_imported_data(args.import_file)

    plot_data(
        recipe_data,
        args.recipe_dir,
        imported_data=imported_data,
        export=args.export,
        out_dir=args.out_dir
    )

if __name__ == '__main__':
    main()
