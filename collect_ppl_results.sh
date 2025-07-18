#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** collect_ppl_results.sh is a helper tool that collects the **#
#** benchmark ppl results of benchmark_each_tensor.sh.        **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-10-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.     ₙₒ𝓌 𝓌ᵢₜₕ ₑₓₜᵣₐ ₕₐₗₗᵤ𝒸ᵢₙₐₜᵢₒₙₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# Usage message
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --chunks CHUNKS                      Number of PPL chunks to process (integer)
  --baseline-ppl PPL                   Baseline PPL value for percent-delta computation (float)
  --inject-baseline-ppl-qtype QTYPE    Baseline PPL value will be injected in results for matching qtype (float)
  --hide-empty                         Don't include empty benchmark results to the output csv
  --output-csv FILE                    Path to output CSV file (default: $OUTPUT_CSV)
  --qtypes Q1,Q2,...                   Comma-separated list of qtypes to use (overrides auto-discovery)
  -h, --help                           Show this help message and exit
EOF
}

# ============== USER CONFIGURATION ==============

# List of tensor-name regex patterns (Bash regex) to include in the CSV.
# Adjust these as needed.
USER_REGEX=(
  # Token embedding and output tensors (GPU)
  # note token_embd cannot be repacked quant type
  '^output\.weight'
  '^token_embd\.weight'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight'
  '^blk\.[0-2]\.ffn_up\.weight'
  '^blk\.[0-2]\.ffn_gate\.weight'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight'

  ## CPU-loaded ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight'
)

# Default output CSV filename (can be overridden via --output-csv)
OUTPUT_CSV="ppl_results.csv"

# =========== End USER CONFIGURATION ============

# Initialize variables
PPL_CHUNKS=""
BASELINE_PPL=""
BASELINE_PPL_QTYPE=""
HIDE_EMPTY=false
qtypes=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks)
      shift
      PPL_CHUNKS="$1"
      if ! [[ $PPL_CHUNKS =~ ^[0-9]+$ ]]; then
        echo "Error: --chunks value must be an integer (got '$PPL_CHUNKS')" >&2
        exit 1
      fi
      ;;
    --baseline-ppl)
      shift
      BASELINE_PPL="$1"
      if ! [[ $BASELINE_PPL =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --baseline-ppl value must be a number (got '$BASELINE_PPL')" >&2
        exit 1
      fi
      ;;
    --inject-baseline-ppl-qtype)
      shift
    #   if [[ -n $BASELINE_PPL ]]; then
    #     echo "Error: --baseline-ppl value must be specified when using --inject-baseline-ppl-qtype" >&2
    #     exit 1
    #   fi
      BASELINE_PPL_QTYPE="$1"
      ;;
    --hide-empty)
      HIDE_EMPTY=true
      ;;
    --output-csv)
      shift
      OUTPUT_CSV="$1"
      ;;
    --qtypes)
      shift
      qtypes="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

# Echo chosen settings
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting collection of PPL results."
[[ -n "$PPL_CHUNKS" ]] && echo "Using PPL chunks: $PPL_CHUNKS"
[[ -n "$BASELINE_PPL" ]] && echo "Using baseline PPL: $BASELINE_PPL"
[[ -n "$BASELINE_PPL_QTYPE" ]] && echo "Injecting baseline PPL for this qtype: $BASELINE_PPL_QTYPE"
[[ "$HIDE_EMPTY" == true ]] && echo "Hide empty qtype bench results from the csv: $HIDE_EMPTY"
[[ -n "$qtypes" ]] && echo "Overriding qtypes with: $qtypes"
echo "Output CSV: $OUTPUT_CSV"

# 1. Discover qtypes by finding tensors.{qtype}.map files in current directory
declare -a QTYPES=()

# Override discovered qtypes if user provided --qtypes
if [[ -n "$qtypes" ]]; then
    IFS=',' read -r -a QTYPES <<< "$qtypes"
else
    for f in tensors.*.map; do
        [[ -f $f ]] || continue
        qtype="${f#tensors.}"
        qtype="${qtype%.map}"
        QTYPES+=("$qtype")
    done
fi

if [[ ${#QTYPES[@]} -eq 0 ]]; then
    echo "Warning: No tensors.*.map files found in current directory${qtypes:+ and no valid --qtypes provided}. Exiting." >&2
    exit 1
fi

# Sort qtypes lexically and remove duplicates
IFS=$'\n' sorted_qtypes=($(printf '%s\n' "${QTYPES[@]}" | sort -u))
unset IFS
QTYPES=("${sorted_qtypes[@]}")

echo "Found qtypes: ${QTYPES[*]}"

declare -A PPL_VALUES    # key: "qtype|tensor_name" => PPL value (string)
declare -A TENSOR_SET    # key: tensor_name => 1

# 2. For each qtype, parse tensors.{qtype}.map
all_bench_result_files=$(find . -maxdepth 1 -type f -printf "%f\n" 2>/dev/null | grep -E "^bench_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)
for qtype in "${QTYPES[@]}"; do
    mapfile="tensors.${qtype}.map"
    if [[ ! -f "$mapfile" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Expected map file '$mapfile' not found. Skipping qtype='$qtype'." >&2
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing map file: $mapfile"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        IFS=':' read -r fname file_hash tensor_name _ <<< "$line"
        matched=false
        for pat in "${USER_REGEX[@]}"; do
            if [[ $tensor_name =~ $pat ]]; then
                matched=true
                break
            fi
        done
        [[ "$matched" == true ]] || continue
        
        # bench_result file regex pattern for grep (not glob)
        regex="^bench_result\.${tensor_name}\..*\.${PPL_CHUNKS}\.txt$"
        # Use find + grep to match files in current directory only
        bench_result_matches=$(printf '%s\n' "$all_bench_result_files" | grep -m1 -E "$regex" 2>/dev/null || true)
        [[ "$HIDE_EMPTY" == false ]] && TENSOR_SET["$tensor_name"]=1 # Don't remove empty columns in the csv
        [[ -n "$bench_result_matches" ]] || continue

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found bench results for tensor_name: $tensor_name"

        [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$tensor_name"]=1 # Remove empty qtype result columns in the csv

        ppl_value=""
        if [[ "$BASELINE_PPL_QTYPE" == "$qtype" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using baseline PPL for qtype: $qtype"
            if [[ -n "$BASELINE_PPL" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using user-provided PPL value: $BASELINE_PPL"
                ppl_value="$BASELINE_PPL"
            else
                result_file="bench_result.baseline.${qtype}.${PPL_CHUNKS}.txt"
            fi
        else
            result_file="bench_result.${tensor_name}.${qtype}.${PPL_CHUNKS}.txt"
        fi

        if [[ -f "$result_file" && "$ppl_value" == "" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found result file: $result_file"
            ppl_line=$(grep 'Final estimate: PPL' "$result_file" || true)
            if [[ -n "$ppl_line" ]]; then
                extracted=$(awk '/Final estimate: PPL/ {
                    for(i=1;i<=NF;i++){
                        if($i=="=" && i+1<=NF){
                            print $(i+1)
                            exit
                        }
                    }
                }' <<< "$ppl_line")
                if [[ -n "$extracted" ]]; then
                    ppl_value="$extracted"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracted PPL: $ppl_value"
                else
                    ppl_value="404"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not extract numeric PPL from line: '$ppl_line'. Using PPL=404."
                fi
            else
                ppl_value="404"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: No 'Final estimate: PPL' line found in $result_file. Using PPL=404."
            fi
        fi

        PPL_VALUES["${qtype}|${tensor_name}"]="$ppl_value"
    done < "$mapfile"
done

# 3. Build sorted list of all tensor names
tensor_list=("${!TENSOR_SET[@]}")
if [[ ${#tensor_list[@]} -eq 0 ]]; then
    echo "Warning: No tensor names matched USER_REGEX in any map files. Exiting." >&2
    exit 1
fi
IFS=$'\n' sorted_tensors=($(printf '%s\n' "${tensor_list[@]}" | sort -Vu))
unset IFS

# 4. Write CSV
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing CSV to $OUTPUT_CSV"

echo "[DEBUG] Writing CSV..."

{
    printf 'QTYPE'
    for t in "${sorted_tensors[@]}"; do
        echo "[DEBUG] Header tensor: $t" >&2
        printf ',%s' "$t"
    done
    printf '\n'

    for qtype in "${QTYPES[@]}"; do
        echo "[DEBUG] Writing row for QTYPE: $qtype" >&2
        printf '%s' "$qtype"
        for t in "${sorted_tensors[@]}"; do
            key="${qtype}|${t}"
            val="${PPL_VALUES[$key]:-}"
            echo "[DEBUG] Raw value for [$key] = '$val'" >&2

            if [[ -n "$BASELINE_PPL" && -n "$val" ]]; then
                if [[ "$val" == "404" ]]; then
                    val="404%"
                else
                    pct=$(awk -v b="$BASELINE_PPL" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
                    val="$pct"
                fi
                echo "[DEBUG] Final value for [$key] = '$val'" >&2
            fi

            printf ',%s' "$val"
        done
        printf '\n'
    done
} > "$OUTPUT_CSV"

echo "[DEBUG] Finished writing CSV."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. CSV available at: $OUTPUT_CSV"
