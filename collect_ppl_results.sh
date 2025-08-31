#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** collect_ppl_results.sh is a helper tool that collects the **#
#** benchmark ppl results of benchmark_each_tensor.sh.        **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-31-2025 -------------------- **#
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

# Note: convert old bench_results to new bench_ppl_results txt filename:
# for f in $(ls | grep bench_result\.); do mv $f $(echo $f | sed 's/bench_result\./bench_ppl_result./g'); done

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
  --auto-baseline QTYPE                Automatically read bench_ppl_result.baseline.QTYPE.<CHUNKS>.txt and use it
  --group-tensors REG1[,REG2] [REG3,..] Specify one or more group specifications (same syntax as benchmark_each_tensor.sh).
                                         Each argument is a group: comma-separated regexes. If omitted, grouping disabled.
  --expand-groups                      When present, expand groups into individual tensor columns (default: disabled)
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
  '^output\.weight$'
  '^token_embd\.weight$'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight$'
  '^blk\.[0-2]\.ffn_up\.weight$'
  '^blk\.[0-2]\.ffn_gate\.weight$'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight$'

  ## CPU-loaded ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight$'
)

# Default output CSV filename (can be overridden via --output-csv)
OUTPUT_CSV="ppl_results.csv"

# =========== End USER CONFIGURATION ============

# Initialize variables
PPL_CHUNKS=""
BASELINE_PPL=""             # global baseline
BASELINE_PPL_QTYPE=""       # qtype that should get injected baseline (if set)
AUTO_BASELINE_QTYPE=""      # qtype to auto-read baseline file for
HIDE_EMPTY=false
qtypes=""
GROUP_TENSORS_RAW=()
GROUP_TENSORS_DISABLED=true
EXPAND_GROUPS=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --chunks requires an argument" >&2; usage; exit 1
      fi
      PPL_CHUNKS="$2"
      if ! [[ $PPL_CHUNKS =~ ^[0-9]+$ ]]; then
        echo "Error: --chunks value must be an integer (got '$PPL_CHUNKS')" >&2
        exit 1
      fi
      shift 2
      ;;
    --baseline-ppl)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --baseline-ppl requires an argument" >&2; usage; exit 1
      fi
      BASELINE_PPL="$2"
      if ! [[ $BASELINE_PPL =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --baseline-ppl value must be a number (got '$BASELINE_PPL')" >&2
        exit 1
      fi
      shift 2
      ;;
    --inject-baseline-ppl-qtype)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --inject-baseline-ppl-qtype requires an argument" >&2; usage; exit 1
      fi
      BASELINE_PPL_QTYPE="$2"
      shift 2
      ;;
    --auto-baseline)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --auto-baseline requires a qtype argument" >&2; usage; exit 1
      fi
      AUTO_BASELINE_QTYPE="$2"
      shift 2
      ;;
    --hide-empty)
      HIDE_EMPTY=true
      shift
      ;;
    --output-csv)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --output-csv requires a filename argument" >&2; usage; exit 1
      fi
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --qtypes)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --qtypes requires an argument (comma-separated list)" >&2; usage; exit 1
      fi
      qtypes="$2"
      shift 2
      ;;
    --group-tensors)
      # collect one or more group specs (nargs '+')
      shift
      GROUP_TENSORS_RAW=()
      if [[ $# -eq 0 || "${1:0:2}" == "--" ]]; then
        echo "Error: --group-tensors requires at least one group specification" >&2; usage; exit 1
      fi
      while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
        GROUP_TENSORS_RAW+=("$1")
        shift
      done
      ;;
    --expand-groups)
      EXPAND_GROUPS=true
      shift
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Echo chosen settings
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting collection of PPL results."
[[ -n "$PPL_CHUNKS" ]] && echo "Using PPL chunks: $PPL_CHUNKS"
[[ -n "$BASELINE_PPL" ]] && echo "Using baseline PPL: $BASELINE_PPL"
[[ -n "$BASELINE_PPL_QTYPE" ]] && echo "Injecting baseline PPL for this qtype: $BASELINE_PPL_QTYPE"
[[ -n "$AUTO_BASELINE_QTYPE" ]] && echo "Auto-baseline will attempt to read bench_ppl_result.baseline.${AUTO_BASELINE_QTYPE}.${PPL_CHUNKS}.txt"
[[ "$HIDE_EMPTY" == true ]] && echo "Hide empty qtype bench results from the csv: $HIDE_EMPTY"
[[ -n "$qtypes" ]] && echo "Overriding qtypes with: $qtypes"
echo "Output CSV: $OUTPUT_CSV"
if [[ "$GROUP_TENSORS_DISABLED" != "true" ]]; then
  echo "Group tensors: ENABLED; groups:"
  for g in "${GROUP_TENSORS_RAW[@]}"; do echo "  - $g"; done
  if [[ "$EXPAND_GROUPS" == "true" ]]; then
    echo "Group expansion: ENABLED (show all member tensors)"
  else
    echo "Group expansion: DISABLED (show one column per group)"
  fi
fi

# If the single token '[]' is passed, grouping disabled (mirror benchmark_each_tensor behaviour)
if (( ${#GROUP_TENSORS_RAW[@]} == 0 )) || ( (( ${#GROUP_TENSORS_RAW[@]} == 1 )) && [[ "${GROUP_TENSORS_RAW[0]}" == "[]" ]] ); then
  GROUP_TENSORS_DISABLED=true
else
  GROUP_TENSORS_DISABLED=false
fi

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
declare -A TENSOR_SET    # key: tensor_name or groupN => 1
declare -A PROCESSED_GROUP_QTYPE  # key: "qtype|groupidx" => 1/MISSING

# gather list of ppl result files in current dir matching chunks (includes group and baseline files)
all_bench_ppl_result_files=$(find . -maxdepth 1 -type f -printf "%f\n" 2>/dev/null | grep -E "^bench_ppl_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)

# Helper: find_group_index_for_tensor <tensor> -> prints group index (0-based) or -1
find_group_index_for_tensor() {
  local tensor="$1"
  if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
    printf '%s\n' -1
    return
  fi
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    local group_raw="${GROUP_TENSORS_RAW[$idx]}"
    IFS=',' read -r -a regs <<< "$group_raw"
    for reg in "${regs[@]}"; do
      # trim spaces
      reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
      [[ -z "$reg" ]] && continue
      if [[ $tensor =~ $reg ]]; then
        printf '%s\n' "$idx"
        return
      fi
    done
  done
  printf '%s\n' -1
}

# Helper to extract Final estimate PPL from a result file (returns numeric PPL or empty)
extract_ppl_from_file() {
  local file="$1"
  awk '
    # remove leading/trailing whitespace helper
    function trim(s) { sub(/^[ \t\r\n]+/, "", s); sub(/[ \t\r\n]+$/, "", s); return s }

    /Final estimate: PPL/ {
      # try to match "PPL = 123.456" (with optional spaces)
      if (match($0, /PPL[ \t]*=[ \t]*([0-9]+(\.[0-9]+)?)/, m)) {
        print m[1]; exit
      }
      # fallback: any "=" followed by a number
      if (match($0, /=[ \t]*([0-9]+(\.[0-9]+)?)/, m)) {
        print m[1]; exit
      }
    }
  ' "$file" 2>/dev/null || true
}

# If grouping is enabled and groups are NOT expanded, and the user does not hide empty columns,
# create column placeholders for each group (group0, group1, ...) so they appear in CSV headers by default.
if [[ "$GROUP_TENSORS_DISABLED" != "true" && "$EXPAND_GROUPS" == "false" && "$HIDE_EMPTY" == "false" ]]; then
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    TENSOR_SET["group${idx}"]=1
  done
fi

# If auto-baseline requested, attempt to read bench_ppl_result.baseline.<qtype>.<CHUNKS>.txt
if [[ -n "$AUTO_BASELINE_QTYPE" ]]; then
  baseline_fname="bench_ppl_result.baseline.${AUTO_BASELINE_QTYPE}.${PPL_CHUNKS}.txt"
  if printf '%s\n' "$all_bench_ppl_result_files" | grep -qF -- "$baseline_fname"; then
    base_val=$(extract_ppl_from_file "./${baseline_fname}" || true)
    if [[ -n "$base_val" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: extracted for qtype=${AUTO_BASELINE_QTYPE}: PPL=${base_val}"
      # If user already provided BASELINE_PPL, respect it and don't override
      if [[ -n "$BASELINE_PPL" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_PPL already user-defined, not replaced!"
      else
        BASELINE_PPL="$base_val"
        BASELINE_PPL_QTYPE="${AUTO_BASELINE_QTYPE}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_PPL='$BASELINE_PPL' and BASELINE_PPL_QTYPE='$BASELINE_PPL_QTYPE' have now been set"
      fi
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: baseline file exists but no 'Final estimate: PPL' line found in $baseline_fname"
    fi
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: baseline file $baseline_fname not found for qtype=${AUTO_BASELINE_QTYPE}"
  fi
fi

# 2. For each qtype, parse tensors.{qtype}.map and collect results (with grouping support)
for qtype in "${QTYPES[@]}"; do
    mapfile="tensors.${qtype}.map"
    if [[ ! -f "$mapfile" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: expected map file '$mapfile' not found. Skipping qtype='$qtype'." >&2
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing map file: $mapfile"

    # read all lines of mapfile into array for flexible scanning/group collection
    mapfile -t MAP_LINES < "$mapfile"

    # Build quick list of tensor names available in this qtype's map
    declare -a TENS_IN_MAP=()
    for line in "${MAP_LINES[@]}"; do
        [[ -z "$line" ]] && continue
        IFS=':' read -r _fname _hash tensor_name _ <<< "$line"
        TENS_IN_MAP+=("$tensor_name")
    done

    # iterate through entries in MAP_LINES
    for line in "${MAP_LINES[@]}"; do
        [[ -z "$line" ]] && continue
        IFS=':' read -r fname file_hash tensor_name _ <<< "$line"

        # match tensor_name against USER_REGEX
        matched=false
        for pat in "${USER_REGEX[@]}"; do
          if [[ $tensor_name =~ $pat ]]; then matched=true; break; fi
        done
        [[ "$matched" == true ]] || continue

        # Determine group membership (if any) for this tensor
        gid=$(find_group_index_for_tensor "$tensor_name")

        # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
        if (( gid >= 0 )); then
          if [[ "$EXPAND_GROUPS" == "true" ]]; then
            # user wants member columns: include the individual tensor as a column unless hide-empty==true
            [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
          else
            # user wants group columns: ensure group column is present (unless hide-empty==true)
            [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["group${gid}"]=1
          fi
        else
          # not in a group -> individual tensor column
          [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
        fi

        # If grouping is enabled and this tensor belongs to a group, attempt to process the group
        if (( gid >= 0 )); then
            proc_key="${qtype}|${gid}"
            # If this group for this qtype has already been processed (value '1'), skip individual handling.
            # We do NOT skip when the marker is 'MISSING' — that allows falling back to per-tensor files.
            if [[ "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" == "1" ]]; then
              continue
            fi

            # collect all group members present in this qtype's map
            group_raw="${GROUP_TENSORS_RAW[$gid]}"
            IFS=',' read -r -a regs <<< "$group_raw"
            declare -a group_members=()
            for reg in "${regs[@]}"; do
              reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
              [[ -z "$reg" ]] && continue
              for t in "${TENS_IN_MAP[@]}"; do
                if [[ $t =~ $reg ]]; then
                  if [[ ! " ${group_members[*]} " =~ " $t " ]]; then
                    group_members+=("$t")
                  fi
                fi
              done
            done

            if (( ${#group_members[@]} == 0 )); then
              echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: no group members found in map for group #${gid} (qtype=${qtype}). Skipping group." >&2
              PROCESSED_GROUP_QTYPE["$proc_key"]=1
              continue
            fi

            # Look for group result file: bench_ppl_result.group{gid}.{qtype}.{PPL_CHUNKS}.txt
            group_result_filename="bench_ppl_result.group${gid}.${qtype}.${PPL_CHUNKS}.txt"
            if ! printf '%s\n' "$all_bench_ppl_result_files" | grep -qF -- "$group_result_filename"; then
                # Only log missing once per (qtype,group)
                if [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No group PPL result file found for group #${gid}, qtype=${qtype}: expected '$group_result_filename'. Will fall back to individual tensor files."
                    PROCESSED_GROUP_QTYPE["$proc_key"]="MISSING"
                fi
                # fall back to per-tensor
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found group PPL result file: $group_result_filename -> applying to ${#group_members[@]} member(s)."
                result_file="./${group_result_filename}"
                val=""
                if [[ -f "$result_file" ]]; then
                    val=$(extract_ppl_from_file "$result_file" || true)
                fi
                if [[ -z "$val" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not extract PPL from $result_file. Marking 404 for group."
                    val="404"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracted group #${gid} (qtype=${qtype}): PPL=$val"
                fi

                if [[ "$EXPAND_GROUPS" == "true" ]]; then
                  for gm in "${group_members[@]}"; do
                    PPL_VALUES["${qtype}|${gm}"]="$val"
                    [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$gm"]=1
                  done
                else
                  PPL_VALUES["${qtype}|group${gid}"]="$val"
                  [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["group${gid}"]=1
                fi

                PROCESSED_GROUP_QTYPE["$proc_key"]=1
                continue
            fi
        fi

        # Fallback: look for individual per-tensor result
        result_file="bench_ppl_result.${tensor_name}.${qtype}.${PPL_CHUNKS}.txt"
        if ! printf '%s\n' "$all_bench_ppl_result_files" | grep -qF -- "$result_file"; then
            # no individual file: leave empty
            continue
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found bench results for tensor_name: $tensor_name (qtype=${qtype})"
        # ensure included if hide-empty true
        [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$tensor_name"]=1

        # If this qtype is the injected-baseline qtype, handle specially
        if [[ "$BASELINE_PPL_QTYPE" == "$qtype" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using baseline PPL for qtype: $qtype"
            if [[ -n "$BASELINE_PPL" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using user-provided PPL value: $BASELINE_PPL"
                PPL_VALUES["${qtype}|${tensor_name}"]="$BASELINE_PPL"
                continue
            else
                # try to read bench_ppl_result.baseline.<qtype>.<chunks>.txt
                baseline_fname="bench_ppl_result.baseline.${qtype}.${PPL_CHUNKS}.txt"
                if [[ -f "$baseline_fname" ]]; then
                    bval=$(extract_ppl_from_file "./${baseline_fname}" || true)
                    if [[ -n "$bval" ]]; then
                        PPL_VALUES["${qtype}|${tensor_name}"]="$bval"
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Read baseline PPL=$bval from $baseline_fname"
                        continue
                    else
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: baseline file exists but could not extract PPL. Falling back to individual result file."
                    fi
                fi
                # else fallthrough to read individual result file
            fi
        fi

        if [[ -f "$result_file" ]]; then
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
                    PPL_VALUES["${qtype}|${tensor_name}"]="$extracted"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracted PPL: $extracted for ${tensor_name}.${qtype}"
                else
                    PPL_VALUES["${qtype}|${tensor_name}"]="404"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not extract numeric PPL from line: '$ppl_line'. Using PPL=404."
                fi
            else
                PPL_VALUES["${qtype}|${tensor_name}"]="404"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: No 'Final estimate: PPL' line found in $result_file. Using PPL=404."
            fi
        fi
    done
done

# 3. Build sorted list of all tensor names (or groups) for header
tensor_list=("${!TENSOR_SET[@]}")
if [[ ${#tensor_list[@]} -eq 0 ]]; then
    echo "Warning: No tensor names matched USER_REGEX in any map files (or no results found). Exiting." >&2
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

            # If a global baseline exists, compute percent-delta across all qtypes
            if [[ -n "$BASELINE_PPL" && -n "$val" ]]; then
                if [[ "$val" == "404" ]]; then
                    val="404%"
                else
                    pct=$(awk -v b="$BASELINE_PPL" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
                    val="$pct"
                fi
                echo "[DEBUG] Final value for [$key] = '$val'" >&2
                elif [[ -n $BASELINE_PPL && "$BASELINE_PPL_QTYPE" == "$qtype" ]]; then
                val="0%"
                echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
            fi

            printf ',%s' "$val"
        done
        printf '\n'
    done
} > "$OUTPUT_CSV"

echo "[DEBUG] Finished writing CSV."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. CSV available at: $OUTPUT_CSV"
