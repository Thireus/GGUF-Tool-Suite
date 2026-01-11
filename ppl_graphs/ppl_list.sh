#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** ppl_list.sh is a helper tool that collects the estimated  **#
#** ppl equations of models using model_tensor_bpw_metric.py  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jan-11-2026 -------------------- **#
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
#** Copyright © 2025 - Thireus.          ᵣₑₚᵣₒₘₚₜ ᵤₙₜᵢₗ ₛₐₜᵢₛ𝒻ᵢₑ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# collect_ppl.sh
# Finds all .csv in current dir, runs the model script on those with >5 data rows,
# extracts the "y = ..." equation from stdout and writes model_name:equation to ppl_list.txt

set -u

OUTFILE="ppl_list.txt"
CMD="../model_tensor_bpw_metric.py"

# sanity check: script exists
if [ ! -f "$CMD" ]; then
  echo "Error: command not found at $CMD" >&2
  exit 1
fi

# gather CSV files (handles spaces in names)
shopt -s nullglob
files=( *.csv )
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
  echo "No CSV files found in current directory."
  exit 0
fi

# truncate/overwrite output file
: > "$OUTFILE"

for f in "${files[@]}"; do
  # count data rows (exclude header)
  rows=$(awk 'NR>1{c++} END{print c+0}' "$f" 2>/dev/null)
  # ensure rows is a number (fallback to 0 if awk failed)
  if ! [[ $rows =~ ^[0-9]+$ ]]; then
    rows=0
  fi

  if [ "$rows" -le 5 ]; then
    echo "Skipping '$f' (only $rows data rows)"
    continue
  fi

  echo "Processing '$f' ($rows rows)..."

  # run command, capture stdout (suppress stderr). continue on non-zero exit.
  output=$("$CMD" --recipe-results-csv "$f" --metric-name "perplexity" --d-from-lowest 1 --c-free --exclude-qtypes '.*_bn.*$' --transforms "identity" --ignore-outliers 50 --p-grid-max 15 --p-grid-steps 100 --penalize-above 15 --drift-below 5 --equation-only 2>/dev/null) || true

  # extract first line that looks like "y = ..."
  equation=$(printf '%s\n' "$output" | grep -m1 -E '^\s*y\s*=' | sed 's/^[[:space:]]*//')

  if [ -n "$equation" ]; then
    model_name="${f%.*}"
    printf '%s:%s\n' "$model_name" "$equation" >> "$OUTFILE"
    echo "Saved: $model_name"
  else
    echo "No equation found for '$f' — skipping"
  fi
done

echo "Done. Results written to $OUTFILE"
