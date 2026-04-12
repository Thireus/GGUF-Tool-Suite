#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** ppl_list.sh is a helper tool that collects the predicted  **#
#** ppl equations of models using model_tensor_bpw_metric.py  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Apr-13-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.          ᵣₑₚᵣₒₘₚₜ ᵤₙₜᵢₗ ₛₐₜᵢₛ𝒻ᵢₑ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

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

# gather CSV files (handles spaces in names), separate _no-others.csv to prioritize
shopt -s nullglob
files_no_others=( *_no-others.csv )
files_all=( *.csv )
shopt -u nullglob

# combine, no_others first, avoiding duplicates
files=()
for f in "${files_no_others[@]}"; do
  files+=( "$f" )
done
for f in "${files_all[@]}"; do
  if [[ "$f" == *_no-others.csv ]]; then
    continue
  fi
  files+=( "$f" )
done

if [ ${#files[@]} -eq 0 ]; then
  echo "No CSV files found in current directory."
  exit 0
fi

# truncate/overwrite output file
: > "$OUTFILE"

# Keep track of models we have already successfully processed
declare -A processed_models

for f in "${files[@]}"; do
  # Determine model name and fallback file
  if [[ "$f" == *_no-others.csv ]]; then
    model_name="${f%_no-others.csv}"
    fallback_f="${model_name}.csv"
  else
    model_name="${f%.*}"
    fallback_f=""
  fi

  # If we already have an equation for this model, skip
  if [[ -n "${processed_models[$model_name]:-}" ]]; then
    continue
  fi

  cur_f="$f"

  while true; do
    # count data rows (exclude header)
    # read header line (safe for files with spaces in name)
    if ! IFS= read -r header_line < "$cur_f"; then
      rows=0
    else
      # count data rows (tail + wc; no awk)
      rows=$(tail -n +2 -- "$cur_f" | wc -l | tr -d '[:space:]')
      # fallback if wc output is weird
      if ! [[ $rows =~ ^[0-9]+$ ]]; then
        rows=0
      fi
    fi

    # ensure at least 6 data rows
    if [ "$rows" -le 5 ]; then
      echo "Skipping '$cur_f' (only $rows data rows)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    # find bpw column index (0-based) from header (no awk)
    IFS=',' read -r -a header_cols <<< "$header_line"
    bpw_idx=-1
    for i in "${!header_cols[@]}"; do
      col=$(printf '%s' "${header_cols[$i]}" | tr -d '[:space:]' | tr -d '"')
      if [ "$col" = "bpw" ]; then
        bpw_idx=$i
        break
      fi
    done

    if [ "$bpw_idx" -lt 0 ]; then
      echo "Skipping '$cur_f' (no bpw column in header)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    # check that at least one bpw value (in data rows) is > 4 (use bc for float compare)
    has_gt4=0
    # read data rows and test the bpw column
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      IFS=',' read -r -a fields <<< "$line"
      val=${fields[$bpw_idx]:-}
      # strip quotes and whitespace
      val=$(printf '%s' "$val" | tr -d '"' | tr -d '[:space:]')
      [ -z "$val" ] && continue
      # use bc for floating-point comparison; invalid values yield 0
      cmp=$(echo "$val > 4" | bc -l 2>/dev/null || echo 0)
      if [ "$cmp" = "1" ]; then
        has_gt4=1
        break
      fi
    done < <(tail -n +2 -- "$cur_f")

    if [ "$has_gt4" -ne 1 ]; then
      echo "Skipping '$cur_f' (no bpw > 4 in data rows)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    echo "Processing '$cur_f' ($rows rows)..."

    # run command, capture stdout (suppress stderr). continue on non-zero exit.
    output=$("$CMD" --recipe-results-csv "$cur_f" --metric-name "perplexity" --d-from-lowest 1 --c-free --transforms "identity" --ignore-outliers 25 --p-grid-max 15 --p-grid-steps 100 --penalize-above 15 --drift-below 15 --resemblance-metric "abs_mean" --ignore-bpw-below 1.8 --equation-only 2>/dev/null) || true

    # extract first line that looks like "y = ..."
    equation=$(printf '%s\n' "$output" | grep -m1 -E '^\s*y\s*=' | sed 's/^[[:space:]]*//')

    if [ -n "$equation" ]; then
      printf '%s:%s\n' "$model_name" "$equation" >> "$OUTFILE"
      echo "Saved: $model_name"
      processed_models[$model_name]=1
      break
    else
      echo "No equation found for '$cur_f'"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      echo "Skipping '$model_name' completely"
      break
    fi
  done
done

echo "Done. Results written to $OUTFILE"
