#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** estimate_gguf_size.sh is a script that computes total     **#
#** tensor sizes for matched regex tensors.                   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-29-2026 -------------------- **#
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

adjust_tensor_bytes() {
  local dtype="$1"
  local elements="$2"
  local bytes="$3"
  local shape="$4"
  local dtype_upper="${dtype^^}"
  local bpw="${BPW_TABLE[$dtype_upper]:-}"
  local scale_factor="${ADDITIONAL_SCALE_FACTOR_TABLE[$dtype_upper]:-}"

  # If we do not know this dtype, keep the original bytes value.
  if [[ -z "$bpw" ]]; then
    echo "$bytes"
    return 0
  fi

  local base_bytes
  base_bytes="$(awk -v e="$elements" -v b="$bpw" 'BEGIN{printf "%.10f", (e*b)/8.0}')"

  # If the bytes match the base equation and this dtype has an extra scale factor,
  # add the per-row scale bump using dim2*dim3*dim4... (tail product).
  if [[ -n "$scale_factor" ]] && awk -v a="$bytes" -v b="$base_bytes" 'BEGIN{d=a-b; if (d<0) d=-d; exit(d <= 0.0001 ? 0 : 1)}'; then
    local tail_product bump adjusted
    tail_product="$(shape_tail_product "$shape")"
    bump=$(( scale_factor * tail_product ))
    adjusted=$(( bytes + bump ))
    echo "$adjusted"
    return 0
  fi

  echo "$bytes"
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
declare -A SEEN=()
declare -a QTYPES=()
for q in "${USER_MAP[@]}"; do
  if [[ -z "${SEEN[$q]:-}" ]]; then
    if [[ "$q" == "f32" ]]; then
      continue
    fi
    SEEN[$q]=1
    QTYPES+=("$q")
  fi
done

# Display qtypes to fetch
echo "QTYPEs to fetch: ${QTYPES[*]}"

# Fetch map files into temp dir
TMPDIR=$(mktemp -d)
echo "Using temp dir: $TMPDIR"
# Add bf16 if we are processing f32 only
if [ ${#QTYPES[@]} -eq 0 ]; then
  _QTYPES=("bf16")
else
  _QTYPES=("${QTYPES[@]}")
fi
for q in "${_QTYPES[@]}"; do
  local_map="$TMPDIR/tensors.${q}.map"
  echo "Fetching tensors.${q}.map..."
  if run_downloader "${q^^}" "0" "${TMPDIR}" "tensors.${q}.map"; then
    echo "  -> saved to $local_map"
    # Download the signature
    if [[ "$SKIP_GPG" != "true" ]]; then
      if ! run_downloader "${q^^}" -1 "${TMPDIR}" "tensors.${q}.map.sig"; then
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
for regex in "${!USER_MAP[@]}"; do
  tag=${USER_MAP[$regex]}
  if [ "$tag" == "f32" ]; then
    _tag="${_QTYPES[0]}"
  else
    _tag=$tag
  fi

  map_file="$TMPDIR/tensors.${_tag}.map"
  if [[ ! -f "$map_file" ]]; then
    echo "  [Warning] map file missing for tag '$_tag': $map_file" >&2
    continue
  fi

  echo "Scanning for tensor regex='$regex' with dtype='$tag' in '$_tag' map file…"

  matched_lines_num=0
  sum_bytes=0

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
    [[ "${dtype,,}" != "${tag,,}" ]] && continue

    adjusted_bytes="$(adjust_tensor_bytes "$dtype" "$elements" "$bytes" "$shape")"

    if [[ "$adjusted_bytes" != "$bytes" ]]; then
      bump=$(( adjusted_bytes - bytes ))
      echo "  → ${tensor_name}: bytes=${bytes} -> ${adjusted_bytes} (+${bump})"
    fi

    sum_bytes=$(( sum_bytes + adjusted_bytes ))
    matched_lines_num=$(( matched_lines_num + 1 ))
  done < "$map_file"

  echo "  → tensors for this pattern: $matched_lines_num → bytes for this pattern: $sum_bytes"
  total_bytes=$(( total_bytes + sum_bytes ))
done

#
# 6) Convert to GiB and report
#
echo
total_gib=$(awk -v b="$total_bytes" 'BEGIN{printf "%.2f", b/1024/1024/1024}')
echo "Total bytes matched: $total_bytes"
echo "≈ $total_gib GiB"

echo "Cleaning up..."
[ -n "$TMPDIR" ] && rm -rf "$TMPDIR"
[ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
echo "Done."
