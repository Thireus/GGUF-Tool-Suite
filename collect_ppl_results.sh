#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** collect_ppl_results.sh is a helper tool that collects the **#
#** benchmark PPL and KLD results of benchmark_each_tensor.sh **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-26-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.     ₙₒ𝓌 𝓌ᵢₜₕ ₑₓₜᵣₐ ₕₐₗₗᵤ𝒸ᵢₙₐₜᵢₒₙₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Note: convert old bench_results to new bench_ppl_results txt filename:
# for f in $(ls | grep bench_result\.); do mv $f $(echo $f | sed 's/bench_result\./bench_ppl_result./g'); done

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Usage message
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --chunks CHUNKS                       Number of PPL chunks to process, default is 250 if not specified (integer)
  --baseline-ppl PPL_VALUE              Baseline PPL value for percent-delta computation (float)
  --baseline-regex REGEX_VALUE          Baseline value for metrics obtained via REGEX for percent-delta computation (float), see --regex parameter
  --inject-baseline-qtype QTYPE         Baseline PPL/REGEX value will be injected in results for this specific qtype (e.g. bf16). The value is resolved from --baseline-ppl/--baseline-regex, else from --auto-baseline, else from bench_ppl(_kld)_result.baseline.QTYPE.<CHUNKS>.txt in the current directory; the script errors out if none of those yields a value.
  --auto-baseline QTYPE                 Automatically read bench_ppl(_kld)_result.baseline.QTYPE.<CHUNKS>.txt and use it
  --group-tensors REG1[,REG2] [REG3,..] Specify one or more group specifications (same syntax as benchmark_each_tensor.sh).
                                          Each argument is a group: comma-separated regexes. If omitted, grouping disabled.
  --group-tensors-map FILE              Path to a group mapping file (each line "groupN:regex[,regex2]" or "regex[,regex2]").
                                          This replicates --group-tensors but reads groups from a file. Mutually exclusive
                                          with --group-tensors.
  --groups-only.                        When present, will only collect the group metrics (default: disabled)
  --expand-groups                       When present, expand groups into individual tensor columns (default: disabled)
  --hide-empty                          Don't include empty benchmark results to the output csv
  --output-ppl-csv FILE                 Path to output PPL CSV file (default: $OUTPUT_PPL_CSV)
  --output-kld-csv FILE                 Path to output KLD CSV file (default: $OUTPUT_KLD_CSV)
  --output-regex-csv FILE               Path to output REGEX CSV file (default: $OUTPUT_REGEX_CSV)
  --regex REGEX                         String pattern to match any desired metric from bench log file, e.g. '.*Mean[[:space:]]*Δp[[:space:]]*:[[:space:]]*(-?[0-9]+(\.[0-9]+)?).*'
  --qtypes Q1,Q2,...                    Comma-separated list of qtypes to use (overrides auto-discovery)
  --no-kld                              Disable kld collection
  --no-percentage                       Disable percent-delta computation (emit raw values; baseline values will be injected as-is)
  -h, --help                            Show this help message and exit

Note: bench_*_result.*.txt.unused quarantine files (tensors ignored by the model, produced by benchmark_each_tensor.sh --hotswap) are accepted by default when the normal .txt result is absent, and collected as baseline-equivalent results (KLD=0, PPL=baseline) - showing that lowering the quant of these tensors has no effect.

Note: q*_K and q*_KV quants must be used with a capital "K" and "KV" letters at the end of their name (e.g. q2_K, q6_K, q8_KV). All other quants are lowercase (e.g. bf16, iq3_kt, iq4_k, q8_0, q4_k_r4, q8_k_r8). Map and result filenames are matched case-sensitively, so a mis-cased qtype finds no files.
EOF
}

# ============== USER CONFIGURATION ==============

# List of tensor-name regex patterns (Bash regex) to include in the CSV.
# Adjust these as needed.
USER_REGEX=(
  ## Model head & embeddings
  '^token_embd\.weight$'
  '^output\.weight$'
  '^output_norm\.weight$'

  ## Multi-headed attention parameters
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_v\.bias$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_v\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_q\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_output\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_k\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_q\.bias$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_q_norm\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_k_norm\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_k\.bias$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.attn_norm\.weight$'

  ## Core FFN weights
  '^blk\.[0-2]\.ffn_down\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_gate_inp\.weight$'
  '^blk\.[0-2]\.ffn_gate\.weight$'
  '^blk\.[0-2]\.ffn_up\.weight$'

  ## Other tensors
  '^blk\.92\.nextn\.shared_head_norm\.weight$'
  '^blk\.92\.nextn\.enorm\.weight$'
  '^blk\.([0-9]|[1-8][0-9]|9[0-2])\.post_attention_norm\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.exp_probs_b\.bias$'
  '^blk\.92\.nextn\.eh_proj\.weight$'
  '^blk\.92\.nextn\.hnorm\.weight$'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_down_shexp\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_up_shexp\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_gate_shexp\.weight$'

  ## CPU-friendly ffn_*_exps
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_down_exps\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_up_exps\.weight$'
  '^blk\.([3-9]|[1-8][0-9]|9[0-2])\.ffn_gate_exps\.weight$'
)

# Default output CSV filename (can be overridden via --output-ppl-csv, --output-kld-csv and --output-regex-csv)
OUTPUT_PPL_CSV="ppl_results.csv"
OUTPUT_KLD_CSV="kld_results.csv"
OUTPUT_REGEX_CSV="regex_results.csv"

# =========== End USER CONFIGURATION ============

# Initialize variables
PPL_CHUNKS="250"                  # Default value
BASELINE_PPL_VALUE=""             # global baseline for PPL
BASELINE_REGEX_VALUE=""           # global baseline for REGEX
BASELINE_QTYPE=""                 # qtype that should get injected baseline (if set)
AUTO_BASELINE_QTYPE=""            # qtype to auto-read baseline file for
HIDE_EMPTY=false
REGEX=""
qtypes=""
GROUP_TENSORS_RAW=()
GROUP_TENSORS_DISABLED=true
GROUPS_ONLY=false
EXPAND_GROUPS=false
NO_KLD=false
GROUP_TENSORS_MAP_FILE=""
NO_PERCENTAGE=false

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
      BASELINE_PPL_VALUE="$2"
      if ! [[ $BASELINE_PPL_VALUE =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --baseline-ppl value must be a number (got '$BASELINE_PPL_VALUE')" >&2
        exit 1
      fi
      shift 2
      ;;
    --baseline-regex)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --baseline-regex requires an argument" >&2; usage; exit 1
      fi
      BASELINE_REGEX_VALUE="$2"
      if ! [[ $BASELINE_REGEX_VALUE =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: --baseline-regex value must be a number (got '$BASELINE_REGEX_VALUE')" >&2
        exit 1
      fi
      shift 2
      ;;
    --inject-baseline-qtype)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --inject-baseline-qtype requires an argument" >&2; usage; exit 1
      fi
      BASELINE_QTYPE="$2"
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
    --output-ppl-csv)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --output-ppl-csv requires a filename argument" >&2; usage; exit 1
      fi
      OUTPUT_PPL_CSV="$2"
      shift 2
      ;;
    --output-kld-csv)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --output-kld-csv requires a filename argument" >&2; usage; exit 1
      fi
      OUTPUT_KLD_CSV="$2"
      shift 2
      ;;
    --output-regex-csv)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --output-regex-csv requires a filename argument" >&2; usage; exit 1
      fi
      OUTPUT_REGEX_CSV="$2"
      shift 2
      ;;
    --regex)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --regex requires a regex string argument" >&2; usage; exit 1
      fi
      REGEX="$2"
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
    --group-tensors-map)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --group-tensors-map requires a filename argument" >&2; usage; exit 1
      fi
      GROUP_TENSORS_MAP_FILE="$2"
      shift 2
      ;;
    --groups-only)
      GROUPS_ONLY=true
      shift
      ;;
    --expand-groups)
      EXPAND_GROUPS=true
      shift
      ;;
    --no-kld)
      NO_KLD=true
      shift
      ;;
    --no-percentage)
      NO_PERCENTAGE=true
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

# Qtype naming rule: q*_K and q*_KV quants must be spelled with a capital "K"
# and "KV" at the end of their name (q2_K, q6_K, q8_KV). All other quants are
# lowercase (bf16, iq3_kt, iq4_k, q8_0, q4_k_r4, q8_k_r8). Map and result files
# are matched case-sensitively, so a mis-cased qtype silently finds nothing.
canonical_qtype() {
  local q="${1,,}"
  if [[ "$q" =~ ^q[^_]+_kv$ ]]; then
    printf '%s' "${q%_kv}_KV"
  elif [[ "$q" =~ ^q[^_]+_k$ ]]; then
    printf '%s' "${q%_k}_K"
  else
    printf '%s' "$q"
  fi
}

warn_if_bad_qtype_casing() {
  local origin="$1" q="$2" canon
  [[ -z "$q" ]] && return 0
  canon="$(canonical_qtype "$q")"
  [[ "$q" == "$canon" ]] && return 0
  echo "⚠️  Warning! ${origin} qtype '$q' does not follow the qtype naming rule - q*_K and q*_KV quants must be used with a capital \"K\" and \"KV\" letters at the end of their name, all other quants are lowercase. Did you mean '$canon'? Map and result filenames are matched case-sensitively, so '$q' will most likely find no files." >&2
}

warn_if_bad_qtype_casing "--inject-baseline-qtype" "$BASELINE_QTYPE"
warn_if_bad_qtype_casing "--auto-baseline" "$AUTO_BASELINE_QTYPE"
if [[ -n "${qtypes:-}" ]]; then
  IFS=',' read -r -a __qtypes_to_check <<< "$qtypes"
  for __q in "${__qtypes_to_check[@]}"; do
    warn_if_bad_qtype_casing "--qtypes" "$__q"
  done
  unset __qtypes_to_check __q
fi

# Ensure user didn't supply both --group-tensors and --group-tensors-map
if [[ -n "${GROUP_TENSORS_MAP_FILE:-}" && ${#GROUP_TENSORS_RAW[@]} -gt 0 ]]; then
  echo "Error: --group-tensors and --group-tensors-map are mutually exclusive. Please provide only one of them." >&2
  exit 1
fi

# If a group mapping file was provided, read it and populate GROUP_TENSORS_RAW.
# File lines can be:
#   group0:^blk\.0\.attn_(k|v)\.weight$
#   group1:^another_regex1$,^another_regex2$
#   or simply:
#   ^blk\.0\.attn_(k|v)\.weight$
# blank lines and lines starting with '#' are ignored.
if [[ -n "${GROUP_TENSORS_MAP_FILE:-}" ]]; then
  if [[ ! -f "$GROUP_TENSORS_MAP_FILE" ]]; then
    echo "Error: group mapping file '$GROUP_TENSORS_MAP_FILE' not found." >&2
    exit 1
  fi

  # Read file, collect named group indices and/or unnamed groups preserving order.
  declare -A __GTMP_idx_map=()
  declare -a __GTMP_idx_list=()
  declare -a __GTMP_ordered_unnamed=()

  while IFS= read -r __line || [[ -n "$__line" ]]; do
    # Trim whitespace
    __line="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$__line")"
    # skip empty or comment lines
    [[ -z "$__line" ]] && continue
    [[ "${__line:0:1}" == "#" ]] && continue

    if [[ "$__line" == *:* ]]; then
      __prefix="${__line%%:*}"
      __rest="${__line#*:}"
      __rest="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$__rest")"
      if [[ "$__prefix" =~ ^group([0-9]+)$ ]]; then
        __idx="${BASH_REMATCH[1]}"
        __GTMP_idx_map["$__idx"]="$__rest"
        __GTMP_idx_list+=("$__idx")
      else
        # no groupN prefix, treat entire line after first colon as a single group spec
        __GTMP_ordered_unnamed+=("$__rest")
      fi
    else
      # whole line is a group regex list
      __GTMP_ordered_unnamed+=("$__line")
    fi
  done < "$GROUP_TENSORS_MAP_FILE"

  # Sort numeric indices ascending and build GROUP_TENSORS_RAW
  if [[ ${#__GTMP_idx_list[@]} -gt 0 ]]; then
    # remove duplicates and sort numeric
    IFS=$'\n' __sorted_idx=($(printf '%s\n' "${__GTMP_idx_list[@]}" | sort -n -u))
    unset IFS
    for __i in "${__sorted_idx[@]}"; do
      GROUP_TENSORS_RAW+=("${__GTMP_idx_map[$__i]}")
    done
  fi
  # Append unnamed groups preserving file order
  for __u in "${__GTMP_ordered_unnamed[@]}"; do
    GROUP_TENSORS_RAW+=("$__u")
  done

  # cleanup temp variables
  unset __GTMP_idx_map __GTMP_idx_list __GTMP_ordered_unnamed __sorted_idx __i __u __prefix __rest __line __idx

  if [[ ${#GROUP_TENSORS_RAW[@]} -eq 0 ]]; then
    echo "Warning: group mapping file '$GROUP_TENSORS_MAP_FILE' parsed but no groups found." >&2
  else
    echo "[$(timestamp)] Loaded ${#GROUP_TENSORS_RAW[@]} group(s) from mapping file: $GROUP_TENSORS_MAP_FILE"
  fi
fi

# Validate that if user asked to collect groups only, they passed --group-tensors or --group-tensors-map
if [[ "$GROUPS_ONLY" == "true" ]] && [[ ! (-n "${GROUP_TENSORS_MAP_FILE:-}" || ${#GROUP_TENSORS_RAW[@]} -gt 0) ]]; then
  echo "Error: --groups-only requires --group-tensors or --group-tensors-map to be set." >&2
  exit 1
fi

# If the single token '[]' is passed, grouping disabled (mirror benchmark_each_tensor behaviour)
if (( ${#GROUP_TENSORS_RAW[@]} == 0 )) || ( (( ${#GROUP_TENSORS_RAW[@]} == 1 )) && [[ "${GROUP_TENSORS_RAW[0]}" == "[]" ]] ); then
  GROUP_TENSORS_DISABLED=true
else
  GROUP_TENSORS_DISABLED=false
fi

# Echo chosen settings
echo "[$(timestamp)] Starting collection of PPL results."
[[ -n "$PPL_CHUNKS" ]] && echo "Using PPL chunks: $PPL_CHUNKS"
[[ -n "$BASELINE_PPL_VALUE" ]] && echo "Using baseline value for PPL: $BASELINE_PPL_VALUE"
[[ -n "$BASELINE_REGEX_VALUE" ]] && echo "Using baseline value for REGEX: $BASELINE_REGEX_VALUE"
[[ -n "$BASELINE_QTYPE" ]] && echo "Injecting baseline PPL/REGEX for this qtype: $BASELINE_QTYPE"
[[ -n "$AUTO_BASELINE_QTYPE" ]] && echo "Auto-baseline will attempt to read bench_ppl(_kld)_result.baseline.${AUTO_BASELINE_QTYPE}.${PPL_CHUNKS}.txt"
[[ "$HIDE_EMPTY" == true ]] && echo "Hide empty qtype bench results from the csv: $HIDE_EMPTY"
[[ -n "${qtypes:-}" ]] && echo "Overriding qtypes with: $qtypes"
echo "Output PPL CSV: $OUTPUT_PPL_CSV"
[[ "$NO_KLD" == "false" ]] && echo "Output KLD CSV: $OUTPUT_KLD_CSV"
[[ "$REGEX" != "" ]] && echo "Output REGEX CSV: $OUTPUT_REGEX_CSV"
[[ "$REGEX" != "" ]] && echo "REGEX: $REGEX"
ALL_GROUP_IDS=()
# Split and whitespace-trim every group's regex list ONCE, into a flat table indexed by
# group. This used to be redone - with a sed(1) fork per regex - on every single call to
# find_group_indexes_for_tensor and on every group-member rebuild, i.e. tens of thousands
# of forks per run. Values are identical, this is purely hoisting invariant work.
declare -a GROUP_REGEX_FLAT=()   # all trimmed regexes, concatenated
declare -a GROUP_REGEX_START=()  # group idx -> first offset into GROUP_REGEX_FLAT
declare -a GROUP_REGEX_COUNT=()  # group idx -> number of regexes
for _gr_idx in "${!GROUP_TENSORS_RAW[@]}"; do
  IFS=',' read -r -a _gr_regs <<< "${GROUP_TENSORS_RAW[$_gr_idx]}"
  GROUP_REGEX_START[$_gr_idx]=${#GROUP_REGEX_FLAT[@]}
  _gr_cnt=0
  for _gr_r in ${_gr_regs[@]+"${_gr_regs[@]}"}; do
    _gr_r="${_gr_r#"${_gr_r%%[![:space:]]*}"}"   # ltrim
    _gr_r="${_gr_r%"${_gr_r##*[![:space:]]}"}"   # rtrim
    [[ -z "$_gr_r" ]] && continue
    GROUP_REGEX_FLAT+=("$_gr_r")
    _gr_cnt=$((_gr_cnt + 1))
  done
  GROUP_REGEX_COUNT[$_gr_idx]=$_gr_cnt
done
unset _gr_idx _gr_regs _gr_cnt _gr_r

if [[ "$GROUP_TENSORS_DISABLED" != "true" ]]; then
  echo "Group tensors: ENABLED; groups:"
  gid=0
  for g in "${GROUP_TENSORS_RAW[@]}"; do echo "  - group$gid: $g"; ALL_GROUP_IDS+=("$gid"); gid=$((gid + 1)); done
  if [[ "$GROUPS_ONLY" == "true" ]] && (( ${#ALL_GROUP_IDS[@]} == 0 )); then
    echo "[$(timestamp)] ❌ Error: --groups-only is set but there are no groups set!" >&2
  fi

  if [[ "$EXPAND_GROUPS" == "true" ]]; then
    echo "Group expansion: ENABLED (show all member tensors)"
    if [[ "$GROUPS_ONLY" == "true" ]]; then
      echo "⚠️  Warning! If a group contains tensor(s) presents in other groups the metrics for that tensor will be overwritten by the latest group processed."
    fi
  else
    echo "Group expansion: DISABLED (show one column per group)"
  fi
  if [[ "$GROUPS_ONLY" == "true" ]]; then
    echo "Collecting group metrics only: ENABLED"
  else
    echo "Collecting group metrics only: DISABLED"
  fi
fi
[[ "$NO_KLD" == "false" ]] && echo "KLD collection disabled: $NO_KLD"
[[ "$NO_PERCENTAGE" == "true" ]] && echo "Percentage computation: DISABLED ( --no-percentage )"

# 1. Discover qtypes by finding tensors.{qtype}.map files in current directory
declare -a QTYPES=()

# Override discovered qtypes if user provided --qtypes
if [[ -n "${qtypes:-}" ]]; then
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

# The casing warning above is only a warning - it does not stop the run. So that a
# mis-cased --inject-baseline-qtype does not silently fail to match any discovered
# qtype (which leaves the baseline uninjected and sends every single tensor down
# the slow per-tensor baseline path below), resolve it case-insensitively against
# the qtypes we actually found and adopt the spelling that exists.
if [[ -n "$BASELINE_QTYPE" ]] && ! printf '%s\n' "${QTYPES[@]}" | grep -qxF -- "$BASELINE_QTYPE"; then
  _fixed_qtype="$(printf '%s\n' "${QTYPES[@]}" | grep -ixF -- "$BASELINE_QTYPE" | head -n1 || true)"
  if [[ -n "$_fixed_qtype" ]]; then
    echo "[$(timestamp)] ⚠️  Corrected --inject-baseline-qtype casing: '$BASELINE_QTYPE' -> '$_fixed_qtype' (matched a discovered qtype)"
    BASELINE_QTYPE="$_fixed_qtype"
  fi
  unset _fixed_qtype
fi

declare -A PPL_VALUES    # key: "qtype|tensor_name" => PPL value (string)
declare -A KLD_VALUES    # key: "qtype|tensor_name" => KLD value (string)
declare -A REGEX_VALUES  # key: "qtype|tensor_name" => REGEX value (string)
declare -A TENSOR_SET    # key: tensor_name or groupN => 1
declare -A PROCESSED_GROUP_QTYPE  # key: "qtype|groupidx" => 1/MISSING

# gather list of ppl (and ppl_kld) result files in current dir matching chunks (includes group and baseline files)
bench_files_list=$(
  for f in ./bench_ppl_*.txt; do
    [ -e "$f" ] || continue    # skip non-matching globs
    [ -f "$f" ] && printf '%s\n' "${f##*/}"
  done 2>/dev/null
)
if [[ "$NO_KLD" == "true" ]]; then
  # Try to find bench_ppl_result files first
  _kld=''
  all_bench_ppl_result_files=$(printf '%s\n' "$bench_files_list" | grep -E "^bench_ppl${_kld}_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)

  # If none found, fall back to bench_ppl_kld_result files
  if [[ -z "$all_bench_ppl_result_files" ]]; then
    echo "Warning: No bench_ppl${_kld}_result.*.txt found in current directory - PPL will be collected from the PPL+KLD bench result files."
    _kld='_kld'
    all_bench_ppl_result_files=$(printf '%s\n' "$bench_files_list" | grep -E "^bench_ppl${_kld}_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)
  fi
else
  # First try bench_ppl_kld_result
  _kld='_kld'
  all_bench_ppl_result_files=$(printf '%s\n' "$bench_files_list" | grep -E "^bench_ppl${_kld}_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)

  # Fallback to bench_ppl_result if none found, but also disable KLD collection because these files won't contain KLD
  if [[ -z "$all_bench_ppl_result_files" ]]; then
    echo "Warning: No bench_ppl${_kld}_result.*.txt found in current directory - KLD collection is now disabled!"
    _kld=''
    all_bench_ppl_result_files=$(printf '%s\n' "$bench_files_list" | grep -E "^bench_ppl${_kld}_result\..*\.${PPL_CHUNKS}\.txt$" 2>/dev/null || true)
    NO_KLD=true
  fi
fi

# Also gather .txt.unused quarantine files produced by benchmark_each_tensor.sh
# --hotswap for tensors the model ignores at load time ("model has unused
# tensor", e.g. NextN/MTP layers). These are accepted by default: lowering the
# quant of such tensors has no effect on the model, so their metrics equal the
# baseline's (KLD=0, PPL=baseline) - which is a valid and useful data point.
bench_unused_files_list=$(
  for f in ./bench_ppl_*.txt.unused; do
    [ -e "$f" ] || continue    # skip non-matching globs
    [ -f "$f" ] && printf '%s\n' "${f##*/}"
  done 2>/dev/null
)
all_bench_ppl_unused_result_files=$(printf '%s\n' "$bench_unused_files_list" | grep -E "^bench_ppl${_kld}_result\..*\.${PPL_CHUNKS}\.txt\.unused$" 2>/dev/null || true)
if [[ -n "$all_bench_ppl_unused_result_files" ]]; then
  echo "[$(timestamp)] Found $(printf '%s\n' "$all_bench_ppl_unused_result_files" | wc -l | tr -d ' ') .txt.unused result file(s) (tensors ignored by the model); they will be collected as baseline-equivalent results."
fi

# find_group_indexes_for_tensor <tensor> -> sets GROUP_IDXS_RESULT to zero-or-more group
# indices. It used to print them and be read back through a process substitution, which
# cost one fork per tensor per qtype (tens of thousands of forks per run) on top of the
# sed forks; the group indices produced are identical.
declare -a GROUP_IDXS_RESULT=()
find_group_indexes_for_tensor() {
  GROUP_IDXS_RESULT=()
  if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
    # leave GROUP_IDXS_RESULT empty -> caller receives an empty array
    return 0
  fi

  local tensor="$1" idx start count i
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    start="${GROUP_REGEX_START[$idx]}"
    count="${GROUP_REGEX_COUNT[$idx]}"
    for (( i = start; i < start + count; i++ )); do
      if [[ $tensor =~ ${GROUP_REGEX_FLAT[$i]} ]]; then
        GROUP_IDXS_RESULT+=("$idx")
        # one matching regex per group is enough -> move to next group
        break
      fi
    done
  done
  return 0
}

# Helper to extract PPL from a result file (returns numeric PPL or empty)
extract_ppl_from_file() {
  local file="$1"
  local val

  # Try "PPL = 123.456"
  val=$(grep 'PPL' "$file" | grep 'Final' | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -nE 's/.*PPL[[:space:]]*=[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: "Mean PPL(Q) : 39.632743 ± ..."
  val=$(grep 'PPL(Q)' "$file" | grep 'Mean' | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -nE 's/.*Mean[[:space:]]PPL\(Q\)[[:space:]]*:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: any "PPL=" followed by a number
  val=$(grep 'PPL= *[0-9]' "$file" | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -nE 's/.*PPL=[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: any "PPL:" followed by a number
  val=$(grep 'PPL: *[0-9]' "$file" | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -nE 's/.*PPL:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: any " +/-" after a number
  val=$(grep '[0-9] *+/-' "$file" | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -nE 's/.*=[[:space:]]*([0-9]+(\.[0-9]+)?)[[:space:]]*\+\/\-.*$/\1/p')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi
}

# Helper to extract KLD from a result file (returns numeric KLD or empty)
extract_kld_from_file() {
  local file="$1"
  local val

  # Try "Mean    KLD:   1.249031 ± ..." (tabs or spaces)
  val=$(grep 'KLD' "$file" | grep 'Mean' | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -E 's/.*Mean[[:space:]]*KLD[[:space:]]*:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: any "KLD:" followed by a number (tabs or spaces)
  val=$(grep 'KLD:' "$file" | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -E 's/.*KLD:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi

  # Fallback: any "KL Divergence:" followed by a number (tabs or spaces)
  val=$(grep 'KL Divergence:' "$file" | head -n 1)
  if [[ -n "$val" ]]; then
    val=$(echo "$val" | sed -E 's/.*KL Divergence:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/')
    [[ -n "$val" ]] && { echo "$val"; return; }
  fi
}

# Helper to extract REGEX from a result file (returns numeric REGEX or empty)
extract_regex_from_file() {
  local file="$1"
  local line val

  while IFS= read -r line; do
    if [[ $line =~ $REGEX ]]; then
      val="${BASH_REMATCH[1]}"
      if [[ -n $val ]]; then
        echo "$val"
        return
      fi
    fi
  done < "$file"
}

# If grouping is enabled and groups are NOT expanded, and the user does not hide empty columns,
# create column placeholders for each group (group0, group1, ...) so they appear in CSV headers by default.
if [[ "$GROUP_TENSORS_DISABLED" != "true" && "$EXPAND_GROUPS" == "false" && "$HIDE_EMPTY" == "false" ]]; then
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    TENSOR_SET["group${idx}"]=1
  done
fi

# If auto-baseline requested, attempt to read bench_ppl(_kld)_result.baseline.<qtype>.<CHUNKS>.txt
if [[ -n "$AUTO_BASELINE_QTYPE" ]]; then
  baseline_fname="bench_ppl${_kld}_result.baseline.${AUTO_BASELINE_QTYPE}.${PPL_CHUNKS}.txt"
  # Filenames are matched case-sensitively, so a mis-cased qtype finds nothing and
  # leaves BASELINE_PPL_VALUE empty - which is precisely what makes the injected
  # baseline qtype re-read its baseline file once per tensor further down. The
  # user was already warned about the casing; retry case-insensitively here and
  # adopt the spelling that actually exists on disk.
  if ! grep -qF -- "$baseline_fname" <<< "$all_bench_ppl_result_files"; then
    _ci_fname="$(grep -ixF -- "$baseline_fname" <<< "$all_bench_ppl_result_files" | head -n1 || true)"
    if [[ -n "$_ci_fname" ]]; then
      _fixed_qtype="${_ci_fname#bench_ppl${_kld}_result.baseline.}"
      _fixed_qtype="${_fixed_qtype%.${PPL_CHUNKS}.txt}"
      echo "[$(timestamp)] ⚠️  Auto-baseline: corrected qtype casing '${AUTO_BASELINE_QTYPE}' -> '${_fixed_qtype}' to match the on-disk file '${_ci_fname}'"
      AUTO_BASELINE_QTYPE="$_fixed_qtype"
      baseline_fname="$_ci_fname"
    fi
    unset _ci_fname _fixed_qtype
  fi
  if grep -qF -- "$baseline_fname" <<< "$all_bench_ppl_result_files"; then
    base_ppl_val=$(extract_ppl_from_file "./${baseline_fname}" || true)
    [[ "$REGEX" != "" ]] && base_regex_val=$(extract_regex_from_file "./${baseline_fname}" || true)
    if [[ -n "${base_ppl_val:-}" ]]; then
      echo "[$(timestamp)] Auto-baseline: extracted for qtype=${AUTO_BASELINE_QTYPE}: PPL=${base_ppl_val}"
      # If user already provided BASELINE_PPL_VALUE, respect it and don't override
      if [[ -n "${BASELINE_PPL_VALUE:-}" ]]; then
        echo "[$(timestamp)] BASELINE_PPL_VALUE already user-defined, not replaced!"
      else
        BASELINE_PPL_VALUE="$base_ppl_val"
        BASELINE_QTYPE="${AUTO_BASELINE_QTYPE}"
        echo "[$(timestamp)] BASELINE_PPL_VALUE='$BASELINE_PPL_VALUE' and BASELINE_QTYPE='$BASELINE_QTYPE' have now been set"
      fi
    else
      echo "[$(timestamp)] Auto-baseline: baseline file exists but no matching PPL line found in $baseline_fname"
    fi
    if [[ -n "${base_regex_val:-}" ]]; then
      echo "[$(timestamp)] Auto-baseline: extracted for qtype=${AUTO_BASELINE_QTYPE}: REGEX=${base_regex_val}"
      # If user already provided BASELINE_REGEX_VALUE, respect it and don't override
      if [[ -n "${BASELINE_REGEX_VALUE:-}" ]]; then
        echo "[$(timestamp)] BASELINE_REGEX_VALUE already user-defined, not replaced!"
      else
        BASELINE_REGEX_VALUE="$base_regex_val"
        BASELINE_QTYPE="${AUTO_BASELINE_QTYPE}"
        echo "[$(timestamp)] BASELINE_REGEX_VALUE='$BASELINE_REGEX_VALUE' and BASELINE_QTYPE='$BASELINE_QTYPE' have now been set"
      fi
    else
      echo "[$(timestamp)] Auto-baseline: baseline file exists but no matching REGEX line found in $baseline_fname"
    fi
  else
    echo "[$(timestamp)] Auto-baseline: baseline file $baseline_fname not found for qtype=${AUTO_BASELINE_QTYPE}"
  fi
fi

# --inject-baseline-qtype only makes sense if there is an actual baseline value to
# inject. It can come from three places:
#   1. --baseline-ppl / --baseline-regex                                (explicit)
#   2. --auto-baseline QTYPE                                            (resolved just above)
#   3. bench_ppl*_result.baseline.<qtype>.<chunks>.txt in this directory
# Source 3 used to be resolved per-tensor much further down, which meant it never
# reached the group columns (they came out as 404) and behaved unlike sources 1 and 2.
# Resolve it here instead so all three are equivalent, and hard-fail when none of them
# yields a value rather than quietly emitting 404 for the entire baseline qtype.
if [[ -n "$BASELINE_QTYPE" && -z "${BASELINE_PPL_VALUE:-}" && -z "${BASELINE_REGEX_VALUE:-}" ]]; then
  _ibq_file="bench_ppl${_kld}_result.baseline.${BASELINE_QTYPE}.${PPL_CHUNKS}.txt"
  if [[ -f "$_ibq_file" ]]; then
    BASELINE_PPL_VALUE="$(extract_ppl_from_file "./${_ibq_file}" || true)"
    if [[ "$REGEX" != "" ]]; then
      BASELINE_REGEX_VALUE="$(extract_regex_from_file "./${_ibq_file}" || true)"
    fi
    if [[ -n "${BASELINE_PPL_VALUE:-}" ]]; then
      echo "[$(timestamp)] --inject-baseline-qtype: resolved PPL=${BASELINE_PPL_VALUE} for qtype='${BASELINE_QTYPE}' from ${_ibq_file}"
    fi
    if [[ -n "${BASELINE_REGEX_VALUE:-}" ]]; then
      echo "[$(timestamp)] --inject-baseline-qtype: resolved REGEX=${BASELINE_REGEX_VALUE} for qtype='${BASELINE_QTYPE}' from ${_ibq_file}"
    fi
  fi
  if [[ -z "${BASELINE_PPL_VALUE:-}" && -z "${BASELINE_REGEX_VALUE:-}" ]]; then
    echo "[$(timestamp)] ❌ Error: --inject-baseline-qtype '${BASELINE_QTYPE}' was given but no baseline value could be resolved for it. Supply one of: --baseline-ppl VALUE (and/or --baseline-regex VALUE), or --auto-baseline ${BASELINE_QTYPE}, or place a readable '${_ibq_file}' in the current directory." >&2
    exit 1
  fi
  unset _ibq_file
fi

# write result into an output array passed by name
remove_items_from_list_lines_inplace() {
  local -n _list="$1"
  local -n _remove="$2"
  local -n _out="$3"

  _out=()
  for item in "${_list[@]}"; do
    local skip=false
    for rm in "${_remove[@]}"; do
      [[ "$item" == "$rm" ]] && { skip=true; break; }
    done
    $skip || _out+=("$item")
  done
}

# 2. For each qtype, parse tensors.{qtype}.map and collect results (with grouping support)
for qtype in "${QTYPES[@]}"; do
  mapfile="tensors.${qtype}.map"
  if [[ ! -f "$mapfile" ]]; then
    echo "[$(timestamp)] Warning: expected map file '$mapfile' not found. Skipping qtype='$qtype'." >&2
    continue
  fi
  echo "[$(timestamp)] Processing map file: $mapfile"

  # Track unprocessed group ids for this qtype
  QTYPE_REMAINING_GROUP_IDS=("${ALL_GROUP_IDS[@]}")

  # read all lines of mapfile into array for flexible scanning/group collection
  mapfile -t MAP_LINES < "$mapfile"

  # Build quick list of tensor names available in this qtype's map
  declare -a TENS_IN_MAP=()
  for line in "${MAP_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r _fname _hash tensor_name _ <<< "$line"
    TENS_IN_MAP+=("$tensor_name")
  done

  # Precompute each group's member list for THIS qtype's map, once. It depends only on
  # the map and the group regexes, so it is invariant across the map-line loop below.
  # It used to be rebuilt for every map line, scanning all tensors and doing a linear
  # " ${group_members[*]} " substring test per candidate - O(n^2) with a big string
  # rebuild each time. On a 1524-tensor map with a '.*' group that alone took minutes.
  declare -a QT_GROUP_MEMBERS_FLAT=()
  declare -a QT_GROUP_MEMBERS_START=()
  declare -a QT_GROUP_MEMBERS_COUNT=()
  for _qgm_idx in "${!GROUP_TENSORS_RAW[@]}"; do
    QT_GROUP_MEMBERS_START[$_qgm_idx]=${#QT_GROUP_MEMBERS_FLAT[@]}
    _qgm_cnt=0
    declare -A _qgm_seen=()
    _qgm_s="${GROUP_REGEX_START[$_qgm_idx]}"
    _qgm_c="${GROUP_REGEX_COUNT[$_qgm_idx]}"
    for (( _qgm_i = _qgm_s; _qgm_i < _qgm_s + _qgm_c; _qgm_i++ )); do
      _qgm_reg="${GROUP_REGEX_FLAT[$_qgm_i]}"
      for _qgm_t in ${TENS_IN_MAP[@]+"${TENS_IN_MAP[@]}"}; do
        if [[ $_qgm_t =~ $_qgm_reg ]] && [[ -z "${_qgm_seen[$_qgm_t]:-}" ]]; then
          _qgm_seen["$_qgm_t"]=1
          QT_GROUP_MEMBERS_FLAT+=("$_qgm_t")
          _qgm_cnt=$((_qgm_cnt + 1))
        fi
      done
    done
    QT_GROUP_MEMBERS_COUNT[$_qgm_idx]=$_qgm_cnt
    unset _qgm_seen
  done
  unset _qgm_idx _qgm_cnt _qgm_s _qgm_c _qgm_i _qgm_reg _qgm_t
  _gm_cur_key=""   # which (qtype|group) group_members currently holds

  if [[ "$BASELINE_QTYPE" == "$qtype" ]]; then
    echo "[$(timestamp)] Using baseline PPL or REGEX for qtype: $qtype"
    [[ -n "${BASELINE_PPL_VALUE:-}" ]] && echo "[$(timestamp)] Using baseline PPL value: $BASELINE_PPL_VALUE"
    [[ -n "${BASELINE_REGEX_VALUE:-}" ]] && echo "[$(timestamp)] Using baseline REGEX value: $BASELINE_REGEX_VALUE"
  fi

  PROCESSED_GROUP_IDS=()

  # iterate through entries in MAP_LINES
  for line in "${MAP_LINES[@]}"; do
    # If we are in --groups-only mode and there are no more groups to process, then we break this loop
    if [[ "$GROUPS_ONLY" == "true" ]] && (( ${#QTYPE_REMAINING_GROUP_IDS[@]} == 0 )); then
      echo "[$(timestamp)] There are no more groups to process. Moving to next qtype..."
      break
    fi

    [[ -z "$line" ]] && continue
    IFS=':' read -r fname file_hash tensor_name _ <<< "$line"

    # match tensor_name against USER_REGEX
    matched=false
    for pat in "${USER_REGEX[@]}"; do
      if [[ $tensor_name =~ $pat ]]; then matched=true; break; fi
    done
    [[ "$matched" == true ]] || continue

    # Determine all group indices for this tensor (could be zero..N)
    find_group_indexes_for_tensor "$tensor_name"
    group_idxs_for_tensor=( ${GROUP_IDXS_RESULT[@]+"${GROUP_IDXS_RESULT[@]}"} )

    # If grouping is enabled and this tensor belongs to one or more group, attempt to process the groups
    if (( ${#group_idxs_for_tensor[@]} > 0 )); then

      # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
      if [[ "$EXPAND_GROUPS" == "true" ]]; then
        # user wants member columns: include the individual tensor as a column unless hide-empty==true
        [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
      fi

      # iterate over all groups this tensor belongs to and handle each group separately
      for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
        proc_key="${qtype}|${group_idx_for_tensor}"
        # If this group for this qtype has already been processed (value '1'), skip individual handling.
        # We do NOT skip when the marker is 'MISSING' — that allows falling back to per-tensor files.
        if [[ "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" == "1" ]]; then
          continue
        fi

        # collect all group members present in this qtype's map
        # IMPORTANT: If there is more than one group, this array will be overwritten, which is fine, just make sure to inform the user!
        # Taken from the per-qtype table precomputed above; only re-sliced when the
        # (qtype, group) actually changes, so the common single-group case copies once.
        _gm_key="${qtype}|${group_idx_for_tensor}"
        if [[ "$_gm_cur_key" != "$_gm_key" ]]; then
          _gm_s="${QT_GROUP_MEMBERS_START[$group_idx_for_tensor]}"
          _gm_c="${QT_GROUP_MEMBERS_COUNT[$group_idx_for_tensor]}"
          if (( _gm_c > 0 )); then
            group_members=( "${QT_GROUP_MEMBERS_FLAT[@]:_gm_s:_gm_c}" )
          else
            group_members=()
          fi
          _gm_cur_key="$_gm_key"
        fi

        if (( ${#group_members[@]} == 0 )); then
          echo "[$(timestamp)] Warning: no group members found in map for group #${group_idx_for_tensor} (qtype=${qtype}). Skipping group." >&2
          PROCESSED_GROUP_QTYPE["$proc_key"]=1
          PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
          _tmp_qtype_remaining=()
          remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
          QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
          unset _tmp_qtype_remaining
          continue
        fi
      done
    # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
    else
      # not in a group -> individual tensor column
      ([[ "$HIDE_EMPTY" == "false" ]] && [[ "$GROUPS_ONLY" != "true" ]]) && TENSOR_SET["$tensor_name"]=1
    fi

    # If this qtype is the injected-baseline qtype, handle specially
    if [[ "$BASELINE_QTYPE" == "$qtype" ]]; then

      # Process BASELINE_PPL_VALUE if set
      if [[ -n "${BASELINE_PPL_VALUE:-}" ]]; then
        PPL_VALUES["${qtype}|${tensor_name}"]="$BASELINE_PPL_VALUE"
        if [[ "$NO_KLD" == "false" ]]; then 
          # If grouping is enabled and this tensor belongs to a group
          if (( ${#group_idxs_for_tensor[@]} > 0 )); then
            # iterate over all groups this tensor belongs to and handle each group separately
            for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
              if [[ "$EXPAND_GROUPS" == "true" ]]; then
                for gm in "${group_members[@]}"; do
                  KLD_VALUES["${qtype}|${gm}"]=0
                done
              else
                KLD_VALUES["${qtype}|group${group_idx_for_tensor}"]=0
              fi
            done
          else
            KLD_VALUES["${qtype}|${tensor_name}"]=0
          fi
        fi
      fi

      # Process BASELINE_REGEX_VALUE if set
      if [[ -n "${BASELINE_REGEX_VALUE:-}" ]]; then
        REGEX_VALUES["${qtype}|${tensor_name}"]="$BASELINE_REGEX_VALUE"
        if [[ "$NO_KLD" == "false" ]]; then 
          # If grouping is enabled and this tensor belongs to a group
          if (( ${#group_idxs_for_tensor[@]} > 0 )) && [[ -z "${BASELINE_PPL_VALUE:-}" ]]; then
            # iterate over all groups this tensor belongs to and handle each group separately
            for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
              if [[ "$EXPAND_GROUPS" == "true" ]]; then
                for gm in "${group_members[@]}"; do
                  KLD_VALUES["${qtype}|${gm}"]=0
                done
              else
                KLD_VALUES["${qtype}|group${group_idx_for_tensor}"]=0
              fi
            done
          else
            KLD_VALUES["${qtype}|${tensor_name}"]=0
          fi
        fi
      fi

      # Nothing left to do for the injected-baseline qtype: BASELINE_PPL_VALUE and/or
      # BASELINE_REGEX_VALUE is guaranteed to be set here (see the --inject-baseline-qtype
      # resolution/validation above, which exits when neither can be resolved), so the
      # values were already assigned by the two blocks above.
      continue

    fi

    # If grouping is enabled and this tensor belongs to a group, attempt to process the group (continuation)
    if (( ${#group_idxs_for_tensor[@]} > 0 )); then
      # iterate over all groups this tensor belongs to and handle each group separately
      for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
        proc_key="${qtype}|${group_idx_for_tensor}"
        # Look for group result file: bench_ppl${_kld}_result.group{group_idx_for_tensor}.{qtype}.{PPL_CHUNKS}.txt
        group_result_filename="bench_ppl${_kld}_result.group${group_idx_for_tensor}.${qtype}.${PPL_CHUNKS}.txt"
        # Accept the .txt.unused quarantine variant when the normal result is
        # absent (group members ignored by the model: metrics are the
        # baseline's by definition)
        group_result_is_unused=false
        if ! grep -qF -- "$group_result_filename" <<< "$all_bench_ppl_result_files" \
           && grep -qF -- "${group_result_filename}.unused" <<< "$all_bench_ppl_unused_result_files"; then
          group_result_filename="${group_result_filename}.unused"
          group_result_is_unused=true
        fi
        # confirm it exists in directory listing
        if [[ "$group_result_is_unused" == "false" ]] && ! grep -qF -- "$group_result_filename" <<< "$all_bench_ppl_result_files"; then
          # Only log the "missing group file" message once per (qtype, group).
          if [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
            echo "[$(timestamp)] No group PPL result file found for group #${group_idx_for_tensor}, qtype=${qtype}: expected '$group_result_filename'. Will fall back to individual tensor files (unless --groups-only is enabled)."
            # Mark as 'missing' so we don't re-print this for other members of the same group/qtype.
            PROCESSED_GROUP_QTYPE["$proc_key"]="MISSING"
            PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
            _tmp_qtype_remaining=()
            remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
            QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
            unset _tmp_qtype_remaining
          fi
          # fall back to per-tensor handling
        elif [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
          echo "[$(timestamp)] Found group PPL result file: $group_result_filename -> applying to ${#group_members[@]} member(s)."
          result_file="./${group_result_filename}"

          # Extract PPL
          val_ppl=""
          if [[ -f "$result_file" ]]; then
            val_ppl=$(extract_ppl_from_file "$result_file" || true)
          fi
          if [[ -z "${val_ppl:-}" ]]; then
            if [[ "$group_result_is_unused" == "true" && -n "${BASELINE_PPL_VALUE:-}" ]]; then
              val_ppl="$BASELINE_PPL_VALUE"
              echo "[$(timestamp)] Group #${group_idx_for_tensor} (qtype=${qtype}) is unused by the model; injecting baseline PPL=$val_ppl"
            else
              echo "[$(timestamp)] Warning: Could not extract PPL from $result_file. Marking 404 for group."
              val_ppl="404"
            fi
          else
            echo "[$(timestamp)] Extracted group #${group_idx_for_tensor} (qtype=${qtype}): PPL=$val_ppl"
          fi

          # Extract KLD
          if [[ "$NO_KLD" == "false" ]]; then
            val_kld=""
            if [[ -f "$result_file" ]]; then
              val_kld=$(extract_kld_from_file "$result_file" || true)
            fi
            if [[ -z "${val_kld:-}" ]]; then
              if [[ "$group_result_is_unused" == "true" ]]; then
                val_kld="0"
                echo "[$(timestamp)] Group #${group_idx_for_tensor} (qtype=${qtype}) is unused by the model; injecting KLD=0 (identical to baseline by definition)"
              else
                echo "[$(timestamp)] Warning: Could not extract KLD from $result_file. Marking 404 for group."
                val_kld="404"
              fi
            else
              echo "[$(timestamp)] Extracted group #${group_idx_for_tensor} (qtype=${qtype}): KLD=$val_kld"
            fi
          fi

          # Extract REGEX
          if [[ "$REGEX" != "" ]]; then
            val_regex=""
            if [[ -f "$result_file" ]]; then
              val_regex=$(extract_regex_from_file "$result_file" || true)
            fi
            if [[ -z "${val_regex:-}" ]]; then
              if [[ "$group_result_is_unused" == "true" && -n "${BASELINE_REGEX_VALUE:-}" ]]; then
                val_regex="$BASELINE_REGEX_VALUE"
                echo "[$(timestamp)] Group #${group_idx_for_tensor} (qtype=${qtype}) is unused by the model; injecting baseline REGEX=$val_regex"
              else
                echo "[$(timestamp)] Warning: Could not extract REGEX from $result_file. Marking 404 for group."
                val_regex="404"
              fi
            else
              echo "[$(timestamp)] Extracted group #${group_idx_for_tensor} (qtype=${qtype}): REGEX=$val_regex"
            fi
          fi

          # assign values either to group column (default) or to each member (when expanded)
          if [[ "$EXPAND_GROUPS" == "true" ]]; then
            # assign per-member values
            for gm in "${group_members[@]}"; do
              PPL_VALUES["${qtype}|${gm}"]="$val_ppl"
              [[ "$NO_KLD" == "false" ]] && KLD_VALUES["${qtype}|${gm}"]="$val_kld"
              [[ "$REGEX" != "" ]] && REGEX_VALUES["${qtype}|${gm}"]="$val_regex"
              # ensure tensor column present when hide-empty==true and a result exists
              [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$gm"]=1
            done
          else
            # assign to group column key, not individual members
            PPL_VALUES["${qtype}|group${group_idx_for_tensor}"]="$val_ppl"
            [[ "$NO_KLD" == "false" ]] && KLD_VALUES["${qtype}|group${group_idx_for_tensor}"]="$val_kld"
            [[ "$REGEX" != "" ]] && REGEX_VALUES["${qtype}|group${group_idx_for_tensor}"]="$val_regex"
            # when hide-empty==true and we found a result, ensure the group column is present
            [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["group${group_idx_for_tensor}"]=1
          fi

          PROCESSED_GROUP_QTYPE["$proc_key"]=1
          PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
          _tmp_qtype_remaining=()
          remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
          QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
          unset _tmp_qtype_remaining
          continue
        fi
      done
    fi

    # Skip individual tensors fallback if groups only is used.
    [[ "$GROUPS_ONLY" == "true" ]] && continue

    # Fallback: look for individual per-tensor result
    # If we reach here: either grouping disabled, tensor not in group, OR group file not present -> handle per-tensor file

    result_file="bench_ppl${_kld}_result.${tensor_name}.${qtype}.${PPL_CHUNKS}.txt"
    result_is_unused=false
    if ! grep -qF -- "$result_file" <<< "$all_bench_ppl_result_files"; then
      # Accept the .txt.unused quarantine variant (tensor ignored by the
      # model: metrics are the baseline's by definition)
      if grep -qF -- "${result_file}.unused" <<< "$all_bench_ppl_unused_result_files"; then
        result_file="${result_file}.unused"
        result_is_unused=true
      else
        # no individual file: leave empty
        continue
      fi
    fi

    echo "[$(timestamp)] Found bench results for tensor_name: $tensor_name (qtype=${qtype})"
    [[ "$result_is_unused" == "true" ]] && echo "[$(timestamp)] Note: '$tensor_name' is unused by the model (collected from ${result_file}); its metrics equal the baseline's."
    # ensure included if hide-empty true
    [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$tensor_name"]=1

    if [[ -f "$result_file" ]]; then
      val_ppl=$(extract_ppl_from_file "$result_file" || true)
      if [[ -n "${val_ppl:-}" ]]; then
        PPL_VALUES["${qtype}|${tensor_name}"]="$val_ppl"
        echo "[$(timestamp)] Extracted PPL: $val_ppl for ${tensor_name}.${qtype}"
      elif [[ "$result_is_unused" == "true" && -n "${BASELINE_PPL_VALUE:-}" ]]; then
        PPL_VALUES["${qtype}|${tensor_name}"]="$BASELINE_PPL_VALUE"
        echo "[$(timestamp)] Injected baseline PPL=$BASELINE_PPL_VALUE for unused tensor ${tensor_name}.${qtype}"
      else
        PPL_VALUES["${qtype}|${tensor_name}"]="404"
        echo "[$(timestamp)] Warning: No matching PPL line found in $result_file. Using PPL=404."
      fi
      if [[ "$NO_KLD" == "false" ]]; then
        val_kld=$(extract_kld_from_file "$result_file" || true)
        if [[ -n "${val_kld:-}" ]]; then
          KLD_VALUES["${qtype}|${tensor_name}"]="$val_kld"
          echo "[$(timestamp)] Extracted KLD: $val_kld for ${tensor_name}.${qtype}"
        elif [[ "$result_is_unused" == "true" ]]; then
          KLD_VALUES["${qtype}|${tensor_name}"]="0"
          echo "[$(timestamp)] Injected KLD=0 for unused tensor ${tensor_name}.${qtype} (identical to baseline by definition)"
        else
          KLD_VALUES["${qtype}|${tensor_name}"]="404"
          echo "[$(timestamp)] Warning: No matching KLD line found in $result_file. Using KLD=404."
        fi
      fi
      if [[ "$REGEX" != "" ]]; then
        val_regex=$(extract_regex_from_file "$result_file" || true)
        if [[ -n "${val_regex:-}" ]]; then
          REGEX_VALUES["${qtype}|${tensor_name}"]="$val_regex"
          echo "[$(timestamp)] Extracted REGEX: $val_regex for ${tensor_name}.${qtype}"
        elif [[ "$result_is_unused" == "true" && -n "${BASELINE_REGEX_VALUE:-}" ]]; then
          REGEX_VALUES["${qtype}|${tensor_name}"]="$BASELINE_REGEX_VALUE"
          echo "[$(timestamp)] Injected baseline REGEX=$BASELINE_REGEX_VALUE for unused tensor ${tensor_name}.${qtype}"
        else
          REGEX_VALUES["${qtype}|${tensor_name}"]="404"
          echo "[$(timestamp)] Warning: No matching REGEX line found in $result_file. Using REGEX=404."
        fi
      fi
    fi
  done # end iterating MAP_LINES
done # end for qtype

# 3. Build sorted list of all tensor names (or groups) for header
tensor_list=("${!TENSOR_SET[@]}")
if [[ ${#tensor_list[@]} -eq 0 ]]; then
  echo "Warning: No tensor names matched USER_REGEX in any map files (or no results found). Exiting." >&2
  exit 1
fi
IFS=$'\n' sorted_tensors=($(printf '%s\n' "${tensor_list[@]}" | sort -Vu))
unset IFS

# 4. Write PPL CSV
echo "[$(timestamp)] Writing PPL CSV to $OUTPUT_PPL_CSV"

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
      if [[ -n "$val" ]]; then
        echo "[DEBUG] Raw value for [$key] = '$val'" >&2
      elif [[ "$BASELINE_QTYPE" != "$qtype" || -z "${BASELINE_PPL_VALUE:-}" ]]; then
        echo "[DEBUG] Empty value for [$key] = '$val', will use \"404\" instead" >&2
        val="404"
      else
        echo "[DEBUG] Empty value for [$key] = '$val' (expected for baseline), will use baseline value \"${BASELINE_PPL_VALUE:-}\" instead" >&2
        val="$BASELINE_PPL_VALUE"
      fi

      # Percentage computation: only when baseline present and --no-percentage not set
      if [[ "$NO_PERCENTAGE" != "true" && -n "${BASELINE_PPL_VALUE:-}" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          # detect division by zero for baseline (exactly zero numeric)
          if awk -v b="$BASELINE_PPL_VALUE" 'BEGIN{ if ((b+0)==0) exit 0; exit 1 }'; then
            # baseline is numeric zero -> avoid division by zero, mark as 404%
            val="404%"
            echo "[DEBUG] Baseline value is zero, avoiding division by zero for [$key]. Using '404%'" >&2
          else
            pct=$(awk -v b="$BASELINE_PPL_VALUE" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
            val="$pct"
            echo "[DEBUG] Final value for [$key] = '$val'" >&2
          fi
        fi
      elif [[ "$NO_PERCENTAGE" == "true" ]]; then
        # when no-percentage requested: do not compute %, just output raw value.
        # baseline numeric values were already injected earlier (if applicable).
        # make sure baseline qtype row is not forced to "0%"; leave raw value.
        :
      else
        # If baseline present but value empty? handled above; else, if baseline present and this qtype equals baseline qtype,
        # the script previously forced "0%". Keep that behavior only if percentages are enabled.
        if [[ -n "${BASELINE_PPL_VALUE:-}" && "$BASELINE_QTYPE" == "$qtype" && "$NO_PERCENTAGE" != "true" ]]; then
          val="0%"
          echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
        fi
      fi

      printf ',%s' "$val"
    done
    printf '\n'
  done
} > "$OUTPUT_PPL_CSV"

echo "[DEBUG] Finished writing PPL CSV."

# 5. Write KLD CSV if enabled
if [[ "$NO_KLD" == "false" ]]; then
  echo "[$(timestamp)] Writing KLD CSV to $OUTPUT_KLD_CSV"

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
        val="${KLD_VALUES[$key]:-}"
        if [[ -n "$val" ]]; then
          echo "[DEBUG] Raw value for [$key] = '$val'" >&2
        else
          echo "[DEBUG] Empty value for [$key] = '$val', will use \"404\" instead" >&2
          val="404"
        fi

        printf ',%s' "$val"
      done
      printf '\n'
    done
  } > "$OUTPUT_KLD_CSV"

  echo "[DEBUG] Finished writing KLD CSV."
fi

# 6. Write REGEX CSV if enabled
if [[ "$REGEX" != "" ]]; then
  echo "[$(timestamp)] Writing REGEX CSV to $OUTPUT_REGEX_CSV"

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
        val="${REGEX_VALUES[$key]:-}"
        if [[ -n "$val" ]]; then
          echo "[DEBUG] Raw value for [$key] = '$val'" >&2
        elif [[ "$BASELINE_QTYPE" != "$qtype" || -z "${BASELINE_REGEX_VALUE:-}" ]]; then
          echo "[DEBUG] Empty value for [$key] = '$val', will use \"404\" instead" >&2
          val="404"
        else
          echo "[DEBUG] Empty value for [$key] = '$val' (expected for baseline), will use baseline value \"${BASELINE_REGEX_VALUE:-}\" instead" >&2
          val="$BASELINE_REGEX_VALUE"
        fi

      # If a global baseline exists, compute percent-delta across all qtypes (unless disabled)
      if [[ "$NO_PERCENTAGE" != "true" && -n "${BASELINE_REGEX_VALUE:-}" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          # detect division by zero for baseline (exactly zero numeric)
          if awk -v b="$BASELINE_REGEX_VALUE" 'BEGIN{ if ((b+0)==0) exit 0; exit 1 }'; then
            val="404%"
            echo "[DEBUG] Baseline REGEX value is zero, avoiding division by zero for [$key]. Using '404%'" >&2
          else
            pct=$(awk -v b="$BASELINE_REGEX_VALUE" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
            val="$pct"
            echo "[DEBUG] Final value for [$key] = '$val'" >&2
          fi
        fi
      elif [[ "$NO_PERCENTAGE" == "true" ]]; then
        # no-percentage: keep raw value (baseline values injected earlier)
        :
      else
        # when baseline qtype equals current qtype and percentages enabled, set 0%
        if [[ -n "${BASELINE_REGEX_VALUE:-}" && "$BASELINE_QTYPE" == "$qtype" && "$NO_PERCENTAGE" != "true" ]]; then
          val="0%"
          echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
        fi
      fi

        printf ',%s' "$val"
      done
      printf '\n'
    done
  } > "$OUTPUT_REGEX_CSV"

  echo "[DEBUG] Finished writing REGEX CSV."
fi

# 7. The end!

echo "[$(timestamp)] All Done."
echo "[$(timestamp)] PPL CSV available at: $OUTPUT_PPL_CSV"
[[ "$NO_KLD" == "false" ]] && echo "[$(timestamp)] KLD CSV available at: $OUTPUT_KLD_CSV"
[[ "$REGEX" != "" ]] && echo "[$(timestamp)] REGEX CSV available at: $OUTPUT_REGEX_CSV"
exit 0 # do not leak the status of the optional echo above as the script's exit code
