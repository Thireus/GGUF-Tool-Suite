#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** estimate_gguf_size.sh is a script that computes total     **#
#** tensor sizes for matched regex tensors.                   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-23-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.           ₙₒ ₚᵣₒₘₚₜ, ₛₜᵢₗₗ ₜₐₗₖᵢₙ𝓰 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail
trap 'echo; echo "[$(date "+%Y-%m-%d %H:%M:%S")] Interrupted. Exiting."; exit 1' SIGINT SIGTERM

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

# Default map (used if not piped via stdin)
DEFAULT_MAP=$(cat <<'EOF'
# Low - Resistant to quant
blk\.([3-9]|1[0-6])\.ffn_down_exps\.weight=iq2_k
blk\.([3-9]|1[0-6])\.ffn_gate_exps\.weight=iq1_m_r4
blk\.([3-9]|1[0-6])\.ffn_up_exps\.weight=iq1_m_r4
# Medium - Resistant to quant
blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_down_exps\.weight=iq3_k
blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_gate_exps\.weight=iq2_k
blk\.(1[7-9]|2[0-9]|3[0-2])\.ffn_up_exps\.weight=iq2_k
# High - Sensitive to quant
blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_down_exps\.weight=iq4_ks
blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_gate_exps\.weight=iq3_k
blk\.(3[3-9]|4[0-9]|5[0-6])\.ffn_up_exps\.weight=iq3_k
# Medium-High - Sensitive to quant
blk\.(5[7-9]|60)\.ffn_down_exps\.weight=iq4_ks
blk\.(5[7-9]|60)\.ffn_gate_exps\.weight=iq3_k
blk\.(5[7-9]|60)\.ffn_up_exps\.weight=iq3_k
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
# 5) Scan all USER_MAP entries via grep+awk and accumulate
#
total_bytes=0
for regex in "${!USER_MAP[@]}"; do
  tag=${USER_MAP[$regex]}
  if [ "$tag" == "f32" ]; then
    _tag="${_QTYPES[0]}"
  else
    _tag=$tag
  fi
  echo "Scanning for tensor regex='$regex' with dtype='$tag' in '$_tag' map file…"

  # get matching lines (or empty), never exit
  matched_lines=$(grep -E "$regex" "$TMPDIR"/tensors.$_tag.map 2>/dev/null || true)

  # get number of matching lines
  matched_lines_num=$(echo "$matched_lines"| grep "^.*$" -c)

  # now filter & sum: catch any pipeline errors and default to 0
  sum_bytes=$(
    { printf "%s\n" "$matched_lines" \
      | grep "dtype=$tag" \
      | grep -o 'bytes=[0-9]\+' \
      | cut -d= -f2 \
      | awk '{s+=$1} END{print s+0}'; } || echo 0
  )

  if ! [[ "$sum_bytes" =~ ^[0-9]+$ ]]; then
    sum_bytes=0
  fi

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
