#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** estimate_gguf_size.sh is a script that computes total     **#
#** tensor sizes for matched regex tensors.                   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-31-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.           ₙₒ ₚᵣₒₘₚₜ, ₛₜᵢₗₗ ₜₐₗₖᵢₙ𝓰 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail
trap 'echo; echo "[$(date "+%Y-%m-%d %H:%M:%S")] Interrupted. Exiting." >&2; exit 1' SIGINT SIGTERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GGUF size estimation..."

# ============== USER CONFIGURATION ==============
# Remote connection settings for tensor_downloader.sh:
# Please edit tensor_downloader.sh!
# Resolve script directory for locating tensor_downloader.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"

if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
    echo "Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
    exit 1
fi

run_downloader() {
  set +e
  "$TENSOR_DOWNLOADER" "$@"
  local ret=$?
  set -e
  return $ret
}

# Lookup tables for estimating tensor size adjustments.
# BPW is bytes-per-weight; the additional scale-factor table identifies qtypes
# whose tensors may need an extra per-row scale bump depending on the parsed shape.
declare -A BPW_TABLE=(
  [F32]=32
  [F16]=16
  [BF16]=16
  [Q8_0_R8]=8.5
  [Q8_0]=8.5
  [Q8_K_R8]=8.0625
  [Q8_KV]=8
  [F8]=8
  [IQ6_K]=6.625
  [Q6_K_R4]=6.5625
  [Q6_K]=6.5625
  [Q6_0_R4]=6.5
  [Q6_0]=6.5
  [Q5_1]=6
  [Q5_K_R4]=5.5
  [Q5_K]=5.5
  [Q5_0_R4]=5.5
  [Q5_0]=5.5
  [IQ5_K_R4]=5.5
  [IQ5_K]=5.5
  [IQ5_KS_R4]=5.25
  [IQ5_KS]=5.25
  [Q4_1]=5
  [Q4_K_R4]=4.5
  [Q4_K]=4.5
  [Q4_0_R8]=4.5
  [Q4_0]=4.5
  [IQ4_NL_R4]=4.5
  [IQ4_NL]=4.5
  [IQ4_K_R4]=4.5
  [IQ4_K]=4.5
  [IQ4_XS_R8]=4.25
  [IQ4_XS]=4.25
  [IQ4_KS_R4]=4.25
  [IQ4_KS]=4.25
  [IQ4_KT]=4
  [IQ4_KSS]=4
  [IQ3_KL]=4
  [IQ3_M]=3.66
  [Q3_K_R4]=3.4375
  [Q3_K]=3.4375
  [IQ3_S_R4]=3.4375
  [IQ3_S]=3.4375
  [IQ3_K_R4]=3.4375
  [IQ3_K]=3.4375
  [IQ3_XS]=3.3
  [IQ3_KS]=3.1875
  [IQ3_KT]=3.125
  [IQ3_XXS_R4]=3.0625
  [IQ3_XXS]=3.0625
  [IQ2_M_R4]=2.7
  [IQ2_M]=2.7
  [IQ2_KL]=2.6875
  [Q2_K_R4]=2.625
  [Q2_K]=2.625
  [IQ2_S]=2.5625
  [IQ2_K_R4]=2.375
  [IQ2_K]=2.375
  [IQ2_XS_R4]=2.3125
  [IQ2_XS]=2.3125
  [IQ2_KS]=2.1875
  [IQ2_KT]=2.125
  [IQ2_XXS_R4]=2.0625
  [IQ2_XXS]=2.0625
  [IQ2_BN_R4]=2
  [IQ2_BN]=2
  [IQ1_M_R4]=1.75
  [IQ1_M]=1.75
  [IQ1_KT]=1.75
  [IQ1_BN]=1.625
  [IQ1_S]=1.5625
  [IQ1_S_R4]=1.5
)

declare -A ADDITIONAL_SCALE_FACTOR_TABLE=(
  [IQ1_BN]=2
  [IQ1_KT]=4
  [IQ2_BN]=4
  [IQ2_BN_R4]=4
  [IQ2_KL]=2
  [IQ2_KS]=2
  [IQ2_KT]=4
  [IQ3_KS]=2
  [IQ3_KT]=4
  [IQ4_KS]=4
  [IQ4_KSS]=4
  [IQ4_KS_R4]=4
  [IQ4_KT]=4
  [IQ5_KS]=4
  [IQ5_KS_R4]=4
  [Q8_KV]=8
  [IQ1_S_R4]=2
  [IQ1_M_R4]=2
  [Q8_KV_R8]=4
)

shape_tail_product() {
  local shape="$1"
  local -a dims=()
  local prod=1
  local i d

  # Accept strings such as "shape=(2560, 151936)" or "(4096,)".
  shape="${shape#shape=}"
  shape="${shape//[\(\)\[\]]/}"

  IFS=',' read -r -a dims <<< "$shape"
  for ((i=1; i<${#dims[@]}; i++)); do
    d="${dims[$i]//[[:space:]]/}"
    [[ -z "$d" ]] && continue
    prod=$((prod * d))
  done

  echo "$prod"
}

# Compute target bytes from the source tensor bytes and source dtype bpw.
# For qtypes present in BPW_TABLE, this applies:
#   target_bytes = source_bytes * target_bpw / source_bpw
# and then adds the optional per-row bump for qtypes in ADDITIONAL_SCALE_FACTOR_TABLE.
adjust_tensor_bytes() {
  local target_qtype="$1"
  local source_dtype="$2"
  local source_bytes="$3"
  local shape="$4"

  local target_qtype_upper="${target_qtype^^}"
  local source_dtype_upper="${source_dtype^^}"

  local target_bpw="${BPW_TABLE[$target_qtype_upper]:-}"
  local source_bpw="${BPW_TABLE[$source_dtype_upper]:-}"
  local scale_factor="${ADDITIONAL_SCALE_FACTOR_TABLE[$target_qtype_upper]:-}"

  # If we do not know this target dtype or source dtype, keep the original bytes value.
  if [[ -z "$target_bpw" || -z "$source_bpw" ]]; then
    echo "$source_bytes"
    return 0
  fi

  local target_bytes
  target_bytes="$(awk -v sb="$source_bytes" -v t="$target_bpw" -v s="$source_bpw" 'BEGIN{printf "%.0f", (sb * t) / s}')"

  if [[ -n "$scale_factor" ]]; then
    local tail_product bump adjusted
    tail_product="$(shape_tail_product "$shape")"
    bump=$(( scale_factor * tail_product ))
    adjusted=$(( target_bytes + bump ))
    echo "$adjusted"
    return 0
  fi

  echo "$target_bytes"
}

format_bpw_summary() {
  local bpw="$1"
  awk -v v="$bpw" 'BEGIN{
    if (v == "" || v < 0) {
      print "n/a"
      exit
    }
    if (v == int(v)) {
      printf "%.1f", v
      exit
    }
    s = sprintf("%.4f", v)
    sub(/0+$/, "", s)
    sub(/\.$/, "", s)
    print s
  }'
}

format_gib_precision() {
  local bytes="$1"
  local precision="$2"
  awk -v b="$bytes" -v p="$precision" 'BEGIN{printf "%.*f", p, b/1024/1024/1024}'
}

format_gb_precision() {
  local bytes="$1"
  local precision="$2"
  awk -v b="$bytes" -v p="$precision" 'BEGIN{printf "%.*f", p, b/1000000000}'
}

actual_bpw_from_bytes_and_elements() {
  local bytes="$1"
  local elements="$2"
  awk -v b="$bytes" -v e="$elements" 'BEGIN{
    if (e <= 0) {
      print ""
      exit
    }
    printf "%.10f", (b*8.0)/e
  }'
}

# Default map (used if not piped via stdin)
DEFAULT_MAP=$(cat <<'EOF'
# Low - Resistant to quant
^blk\.([3-9]|1[0-6])\.ffn_down_exps\.weight$=iq2_k
^blk\.([3-9]|1[0-6])\.ffn_gate_exps\.weight$=iq1_m_r4
^blk\.([3-9]|1[0-6])\.ffn_up_exps\.weight$=iq1_m_r4
# Medium - Resistant to quant
^blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_down_exps\.weight$=iq3_k
^blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_gate_exps\.weight$=iq2_k
^blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_up_exps\.weight$=iq2_k
# High - Sensitive to quant
^blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_down_exps\.weight$=iq4_ks
^blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_gate_exps\.weight$=iq3_k
^blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_up_exps\.weight$=iq3_k
# Medium-High - Sensitive to quant
^blk\.(5[7-9]|60)\.ffn_down_exps\.weight$=iq4_ks
^blk\.(5[7-9]|60)\.ffn_gate_exps\.weight$=iq3_k
^blk\.(5[7-9]|60)\.ffn_up_exps\.weight$=iq3_k
EOF
)

SKIP_GPG=false # If true, skip the gpg signature verification of the signed files
# =================================================

# Verify gpg readiness
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SKIP_GPG" != "true" ]]; then
  if [ ! -f "$SCRIPT_DIR/trusted-keys.asc" ]; then
    echo "Error: trusted-keys.asc not found in the script directory."
    echo "Hint: Provide trusted-keys.asc in the same directory as this script or use the --skip-gpg option to disable gpg signature verification."
    exit 6
  fi
  if command -v gpg >/dev/null 2>&1; then
    # Create a temporary GNUPGHOME
    GNUPG_TMPDIR=$(mktemp -d)
    if [ -z "$GNUPG_TMPDIR" ]; then
      echo "Error: Failed to create temporary GPG home directory." >&2
      exit 8
    fi
    # Try importing the keys (silently) to check validity
    if ! gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --import "$SCRIPT_DIR/trusted-keys.asc" > /dev/null 2>&1; then
      echo "Error: trusted-keys.asc contains missing or invalid GPG public keys."
      echo "Hint: Add valid public keys to this file or re-run with the --skip-gpg option to bypass signature verification."
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 7
    fi
  else
    echo "Warning: 'gpg' command not found. Valid GPG public keys verification skipped." >&2
  fi
fi

# Determine if stdin provided
RAW_MAP=""
if [ -t 0 ]; then
  echo "No stdin detected; using default USER_MAP."
  RAW_MAP="$DEFAULT_MAP"
else
  echo "Reading USER_MAP from stdin..."
  RAW_MAP=$(cat)
fi

# Parse RAW_MAP into USER_MAP array
declare -A USER_MAP=()
while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line//[[:space:]]/}" ]] && continue
  regex="${line%%=*}"
  tag="${line#*=}"
  USER_MAP["$regex"]="$tag"
done <<< "$RAW_MAP"
echo "Loaded ${#USER_MAP[@]} USER_MAP entries."

# Build unique list of quant types
declare -A SUMMARY_SEEN=()
declare -A DOWNLOAD_SEEN=()
declare -a SUMMARY_QTYPES=()
declare -a DOWNLOAD_QTYPES=()
for q in "${USER_MAP[@]}"; do
  q_lc="${q,,}"
  q_uc="${q_lc^^}"

  if [[ -z "${SUMMARY_SEEN[$q_lc]:-}" ]]; then
    SUMMARY_SEEN["$q_lc"]=1
    SUMMARY_QTYPES+=("$q_lc")
  fi

  # Only qtypes that are not covered by the BPW table need their own map download.
  if [[ -z "${BPW_TABLE[$q_uc]:-}" ]]; then
    if [[ -z "${DOWNLOAD_SEEN[$q_lc]:-}" ]]; then
      DOWNLOAD_SEEN["$q_lc"]=1
      DOWNLOAD_QTYPES+=("$q_lc")
    fi
  fi
done

# Fetch bf16 first, always.
TMPDIR=$(mktemp -d)
echo "Using temp dir: $TMPDIR"

BF16_AVAILABLE=false
BF16_MAP_FILE="$TMPDIR/tensors.bf16.map"

echo "Fetching tensors.bf16.map..."
if run_downloader "BF16" "0" "$TMPDIR" "tensors.bf16.map"; then
  echo "  -> saved to $BF16_MAP_FILE"
  BF16_AVAILABLE=true
  # Download the signature
  if [[ "$SKIP_GPG" != "true" ]]; then
    if ! run_downloader "BF16" -1 "$TMPDIR" "tensors.bf16.map.sig"; then
      echo "  [Error] failed to fetch map gpg signature for BF16" >&2
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      [ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
      exit 2
    else
      if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$BF16_MAP_FILE.sig" "$BF16_MAP_FILE" > /dev/null 2>&1; then
        echo "  ✓ GPG signature verification successful."
      else
        echo "  [Error] GPG signature verification failed for '$BF16_MAP_FILE.sig'."
        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
        [ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
        exit 3
      fi
    fi
  fi
else
  echo "  [Warning] failed to fetch bf16 tensors.map" >&2
  rm -f "$BF16_MAP_FILE"
fi

# If bf16 is available, we can derive all qtypes covered by the BPW table from it.
# Only qtypes that are not covered by the BPW table need their own map download.
if [[ ${#DOWNLOAD_QTYPES[@]} -gt 0 ]]; then
  echo "QTYPEs to fetch: ${DOWNLOAD_QTYPES[*]}"
else
  echo "QTYPEs to fetch: (none)"
fi

for q in "${DOWNLOAD_QTYPES[@]}"; do
  local_map="$TMPDIR/tensors.${q}.map"
  echo "Fetching tensors.${q}.map..."
  if run_downloader "${q^^}" "0" "$TMPDIR" "tensors.${q}.map"; then
    echo "  -> saved to $local_map"
    # Download the signature
    if [[ "$SKIP_GPG" != "true" ]]; then
      if ! run_downloader "${q^^}" -1 "$TMPDIR" "tensors.${q}.map.sig"; then
          echo "  [Error] failed to fetch map gpg signature for ${q^^}" >&2
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          [ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
          exit 2
      else
        if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$local_map.sig" "$local_map" > /dev/null 2>&1; then
            echo "  ✓ GPG signature verification successful."
        else
            echo "  [Error] GPG signature verification failed for '$local_map.sig'."
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            [ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
            exit 3
        fi
      fi
    fi
  else
    echo "  [Warning] failed to fetch tensors.map" >&2
    rm -f "$local_map"
  fi
done

#
# 5) Scan all USER_MAP entries via regex matching and accumulate adjusted bytes
#
total_bytes=0
total_elements=0
total_nonf32_bytes=0

declare -A QTYPE_COUNT=()
declare -A QTYPE_BYTES=()
declare -A QTYPE_ELEMENTS=()

for regex in "${!USER_MAP[@]}"; do
  tag=${USER_MAP[$regex]}
  tag_lc="${tag,,}"
  tag_uc="${tag_lc^^}"

  if [[ -n "${BPW_TABLE[$tag_uc]:-}" ]]; then
    source_tag="bf16"
    target_known=true
  else
    source_tag="$tag_lc"
    target_known=false
  fi

  map_file="$TMPDIR/tensors.${source_tag}.map"
  if [[ ! -f "$map_file" ]]; then
    echo "  [Warning] map file missing for tag '$source_tag': $map_file" >&2
    continue
  fi

  echo "Scanning for tensor regex='$regex' with dtype='$tag' in '$source_tag' map file…"

  matched_lines_num=0
  sum_bytes=0
  sum_elements=0

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    # Parse the colon-separated line into fields; the tensor name is the third field.
    IFS=':' read -r -a parts <<< "$line"
    [[ ${#parts[@]} -lt 4 ]] && continue

    tensor_name="${parts[2]}"
    if ! [[ "$tensor_name" =~ $regex ]]; then
      continue
    fi

    dtype=""
    elements=""
    bytes=""
    shape=""

    for token in "${parts[@]:3}"; do
      case "$token" in
        dtype=*) dtype="${token#dtype=}" ;;
        elements=*) elements="${token#elements=}" ;;
        bytes=*) bytes="${token#bytes=}" ;;
        shape=*) shape="${token#shape=}" ;;
      esac
    done

    dtype="${dtype//[[:space:]]/}"
    elements="${elements//[[:space:]]/}"
    bytes="${bytes//[[:space:]]/}"

    [[ -z "$dtype" || -z "$bytes" ]] && continue

    [[ -z "$elements" ]] && elements=0

    if [[ "$target_known" == "true" ]]; then
      adjusted_bytes="$(adjust_tensor_bytes "$tag_lc" "$dtype" "$bytes" "$shape")"
    else
      adjusted_bytes="$bytes"
    fi

    if [[ "$adjusted_bytes" != "$bytes" ]]; then
      bump=$(( adjusted_bytes - bytes ))
      printf "  → %s: bytes=%s -> %s (%+d)\n" "$tensor_name" "$bytes" "$adjusted_bytes" "$bump"
    fi

    sum_bytes=$(( sum_bytes + adjusted_bytes ))
    sum_elements=$(( sum_elements + elements ))
    matched_lines_num=$(( matched_lines_num + 1 ))
  done < "$map_file"

  echo "  → tensors for this pattern: $matched_lines_num → bytes for this pattern: $sum_bytes"
  total_bytes=$(( total_bytes + sum_bytes ))
  total_elements=$(( total_elements + sum_elements ))

  QTYPE_COUNT["$tag_lc"]=$(( ${QTYPE_COUNT["$tag_lc"]:-0} + matched_lines_num ))
  QTYPE_BYTES["$tag_lc"]=$(( ${QTYPE_BYTES["$tag_lc"]:-0} + sum_bytes ))
  QTYPE_ELEMENTS["$tag_lc"]=$(( ${QTYPE_ELEMENTS["$tag_lc"]:-0} + sum_elements ))

  if [[ "$tag_lc" != "f32" ]]; then
    total_nonf32_bytes=$(( total_nonf32_bytes + sum_bytes ))
  fi
done

#
# 6) Convert to GB/GiB and report
#
echo
total_gb="$(format_gb_precision "$total_bytes" 3)"
total_gib="$(format_gib_precision "$total_bytes" 3)"

if [[ "$total_elements" -gt 0 ]]; then
  avg_bpw="$(awk -v b="$total_bytes" -v e="$total_elements" 'BEGIN{printf "%.4f", (b*8.0)/e}')"
else
  avg_bpw="n/a"
fi

echo "Total bytes matched: $total_bytes"
echo "≈ $total_gb GB"
echo "≈ $total_gib GiB"
echo "Total elements matched: $total_elements"
echo "Average bpw: $avg_bpw"

echo "Cleaning up..."
[ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
[ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
echo "Done."

#
# 7) Final summary
#
echo
echo "## Summary of tensor sizes"

declare -a SUMMARY_LINES=()
for q in "${SUMMARY_QTYPES[@]}"; do
  count="${QTYPE_COUNT[$q]:-0}"
  bytes="${QTYPE_BYTES[$q]:-0}"
  elements="${QTYPE_ELEMENTS[$q]:-0}"

  if [[ "$count" -gt 0 && "$elements" -gt 0 ]]; then
    bpw_actual="$(actual_bpw_from_bytes_and_elements "$bytes" "$elements")"
    sort_key="$bpw_actual"
  else
    bpw_actual=""
    sort_key="-1"
  fi

  SUMMARY_LINES+=("$sort_key"$'\t'"$q"$'\t'"$count"$'\t'"$bytes"$'\t'"$elements")
done

mapfile -t SORTED_SUMMARY_LINES < <(printf '%s\n' "${SUMMARY_LINES[@]}" | sort -t $'\t' -k1,1nr -k2,2)

highest_qtype=""
lowest_qtype=""
highest_bpw=""
lowest_bpw=""

for line in "${SORTED_SUMMARY_LINES[@]}"; do
  IFS=$'\t' read -r sort_key q count bytes elements <<< "$line"
  if [[ "$q" != "f32" && "$count" -gt 0 && "$elements" -gt 0 && "$sort_key" != "-1" ]]; then
    if [[ -z "$highest_qtype" ]]; then
      highest_qtype="$q"
      highest_bpw="$sort_key"
    fi
    lowest_qtype="$q"
    lowest_bpw="$sort_key"
  fi
done

if [[ -n "$highest_qtype" && -n "$lowest_qtype" && "$total_elements" -gt 0 ]]; then
  total_max_gib="$(awk -v e="$total_elements" -v b="$highest_bpw" 'BEGIN{printf "%.2f", (e*b/8.0)/1024/1024/1024}')"
  total_min_gib="$(awk -v e="$total_elements" -v b="$lowest_bpw" 'BEGIN{printf "%.2f", (e*b/8.0)/1024/1024/1024}')"
  total_pct_of_max="$(awk -v tb="$total_bytes" -v e="$total_elements" -v b="$highest_bpw" 'BEGIN{
    ref = (e*b/8.0)
    if (ref > 0) printf "%.1f", (tb/ref)*100
    else printf "n/a"
  }')"
  echo "# Total: ${total_gib} GiB (${total_pct_of_max}%) | ${total_max_gib} GiB max, if all were ${highest_qtype} | ${total_min_gib} GiB min, if all were ${lowest_qtype}"
else
  echo "# Total: ${total_gib} GiB"
fi

echo
echo "## Summary of tensor counts and bpw per qtype"
echo "#"
echo "# QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)"

WARN_NA=false
for line in "${SORTED_SUMMARY_LINES[@]}"; do
  IFS=$'\t' read -r sort_key q count bytes elements <<< "$line"

  if [[ "$count" -gt 0 && "$elements" -gt 0 && "$sort_key" != "-1" ]]; then
    bpw_display="$(format_bpw_summary "$sort_key")"
  else
    bpw_display="n/a"
    WARN_NA=true
  fi

  assigned_gib="$(format_gib_precision "$bytes" 2)"

  if [[ "$q" == "f32" || "$total_nonf32_bytes" -le 0 ]]; then
    pct_assigned="-"
  else
    pct_assigned="$(awk -v b="$bytes" -v t="$total_nonf32_bytes" 'BEGIN{printf "%.1f%%", (b/t)*100}')"
  fi

  if [[ "$q" != "f32" && "$count" -gt 0 && "$elements" -gt 0 && "$total_elements" -gt 0 && "$sort_key" != "-1" ]]; then
    max_gib_all="$(awk -v e="$total_elements" -v b="$sort_key" 'BEGIN{printf "%.2f", (e*b/8.0)/1024/1024/1024}')"
  else
    max_gib_all="-"
  fi

  printf "# %-12s\t%5d\t%-6s\t%7s GiB\t%-8s\t%7s\n" \
    "$q" "$count" "$bpw_display" "$assigned_gib" "$pct_assigned" "$max_gib_all"
done

echo "#"
echo "# -Average BPW: $avg_bpw"

if [[ "$WARN_NA" == "true" ]]; then
  echo
  echo "Warning: n/a was produced in the summary."
  echo "This may mean you are not downloading the right artifacts for the model of the recipe."
  echo "Make sure the download.conf file present in your working directory corresponds to the model download.conf." >&2
fi
