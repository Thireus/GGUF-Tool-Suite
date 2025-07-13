#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** tensor_downloader.sh is a powerful downloader tool that   **#
#** downloads pre-quantised tensors/shards to cook recipes.   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-13-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus. ₚₒ𝓌ₑᵣₑ𝒹 ᵦᵧ Gₚₜ₋ₒ₃.₁ᵢ₋ᵤₗₜᵣₐ₋ₘᵢₙᵢ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -u

# -----------------------------------------------------------------------------
# tensor_downloader.sh
#
# Usage:
#   ./tensor_downloader.sh QUANT FileID [DestinationDir] [Filename]
#
# Params:
#   QUANT           (mandatory) quantization tag, e.g. "BF16"
#   FileID          (mandatory) integer chunk ID; 0 => "tensors.map"
#   DestinationDir  (optional)  default: "."
#   Filename        (optional)  default: same as downloaded file
#

# -----------------------------------------------------------------------------
# Default configuration (used if not overridden by download.conf)
MODEL_NAME="DeepSeek-R1-0528" # Name of the LLM model
MAINTAINER="THIREUS" # Name of the GGUF maintainer which appears next to the model name
CHUNK_FIRST=2 # First chunk found in tensors.map
CHUNKS_TOTAL=1148 # Total number of chunks of the model

# Default download sources:
# RSYNC_SERVERS: rsync endpoints (user:host:port:base_path)
RSYNC_SERVERS=(
  #"thireus:65.108.205.124:22:~/AI/DeepSeek-R1-0528-BF16-GGUF/SPECIAL/"
)
# CURL_ORGS: Hugging Face org/user and branch (org_or_user:branch)
CURL_ORGS=(
  "Thireus:main"
)

# -----------------------------------------------------------------------------
# Load user config if present (must be in same directory)
# Any variable or array defined in download.conf will override the above defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/download.conf"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi

# -----------------------------------------------------------------------------
# Parse args & display help if missing
show_help() {
  cat <<EOF
Usage: $0 QUANT FileID [DestinationDir] [Filename]

QUANT           (mandatory) quantization tag, e.g. "BF16"
FileID          (mandatory) integer chunk ID; 0 => "tensors.map"
DestinationDir  (optional)  default: "."
Filename        (optional)  default: same as downloaded file
EOF
}

if [ $# -lt 2 ]; then
  echo "Error: QUANT and FileID are mandatory."
  show_help
  exit 2
fi

QUANT="$1"
FileID="$(expr $2 + 0)"
DEST="${3:-.}"
CUSTOM_FILENAME="${4:-}"
QUANT_U="${QUANT^^}"
REPOSITORY_NAME="${MODEL_NAME}-${MAINTAINER}-${QUANT_U}-SPECIAL_SPLIT"
CHUNKS=$(printf "%05d" "$CHUNKS_TOTAL") # Total number of chunks of the model

# -----------------------------------------------------------------------------
# Logging helper
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# -----------------------------------------------------------------------------
# Verification functions
verify_chunk() {
  local f=$1
  if ! head -c 4 "$f" | grep -q '^GGUF'; then
    return 1
  fi
  return 0
}

# For tensors.map: ensure exactly one line for each "-of-<CHUNKS>.gguf" entry
verify_map() {
  local f=$1
  # strip leading zeros to get numeric count
  local num
  num=$(echo "$CHUNKS" | sed 's/^0*//')
  # avoid empty
  [ -z "$num" ] && num=0

  pattern="^${MODEL_NAME}-${MAINTAINER}-${QUANT_U}-SPECIAL_TENSOR-.*-of-${CHUNKS}.gguf:"
  count=$(grep -c "$pattern" "$f")
  if [ "$count" -ne $((CHUNKS_TOTAL-CHUNK_FIRST+1)) ]; then
    log "  ✗ Missing or duplicate entry for chunk(s) (found $count chunks)"
    return 1
  fi
  return 0
}

# Wrapper: choose which verify to run
verify_download() {
  local file="$1"
  if [ "$FileID" -eq 0 ]; then
    verify_map "$file"
  else
    verify_chunk "$file"
  fi
}

# -----------------------------------------------------------------------------
# Build filename and prepare download
if [ "$FileID" -eq 0 ]; then
  FILENAME="tensors.map"
else
  IDX=$(printf "%05d" "$FileID")
  FILENAME="${MODEL_NAME}-${MAINTAINER}-${QUANT_U}-SPECIAL_TENSOR-${IDX}-of-${CHUNKS}.gguf"
fi
if [[ "${CUSTOM_FILENAME}" == "" ]]; then
  CUSTOM_FILENAME="${FILENAME}"
fi

log "Starting download of ${FILENAME} into ${DEST}"
mkdir -p "${DEST}"

# -----------------------------------------------------------------------------
# Rsync attempts
for srv in "${RSYNC_SERVERS[@]}"; do
  IFS=":" read -r RUSER RHOST RPORT RPATH <<< "$srv"
  SRC="${RUSER}@${RHOST}:${RPATH}/${REPOSITORY_NAME}/${FILENAME}"
  DST="${DEST}/${CUSTOM_FILENAME}"

  if [ -f "${DST}" ]; then
    log "File already exists, verifying…"
    if verify_download "${DST}"; then
      log "✓ Verified; no need to download it again - ${DST} (${QUANT_U})"
      exit 0
    else
      log "✗ Verification failed; removing and downloading it"
      rm -f "${DST}"
    fi
  fi

  log "Trying rsync from ${SRC} (port ${RPORT})"
  if [ "${SHOW_PROGRESS:-false}" = true ]; then
    RSYNC_OPTS="-vP"
  else
    RSYNC_OPTS="-q"
  fi
  rsync ${RSYNC_OPTS} --inplace -t -c -e "ssh -p ${RPORT}" \
    "${SRC}" "${DST}"

  if [ $? -eq 0 ]; then
    log "Download complete, verifying…"
    if verify_download "${DST}"; then
      log "✓ Verified and saved via rsync (${RHOST}) - ${DST} (${QUANT_U})"
      chmod 444 "${DST}" # Apply special permission
      exit 0
    else
      log "✗ Verification failed; removing and trying next"
      rm -f "${DST}"
    fi
  else
    log "✗ Rsync failed; trying next"
  fi
done

# -----------------------------------------------------------------------------
# Curl attempts
for crl in "${CURL_ORGS[@]}"; do
  IFS=":" read -r ORG BRANCH <<< "$crl"
  URL="https://huggingface.co/${ORG}/${REPOSITORY_NAME}/resolve/${BRANCH}/${FILENAME}?download=true"
  DST="${DEST}/${CUSTOM_FILENAME}"

  if [ -f "${DST}" ]; then
    log "File already exists, verifying…"
    if verify_download "${DST}"; then
      log "✓ Verified; no need to download it again - ${DST} (${QUANT_U})"
      chmod 444 "${DST}" # Apply special permission
      exit 0
    else
      log "✗ Verification failed; removing and downloading it"
      rm -f "${DST}"
    fi
  fi

  log "Trying curl from ${URL}"
  if [ "${SHOW_PROGRESS:-false}" = true ]; then
    CURL_OPTS="--progress-bar"
  else
    CURL_OPTS="--silent"
  fi
  curl --fail -L --retry 3 -C - ${CURL_OPTS} -R \
    "${URL}" -o "${DST}"

  if [ $? -eq 0 ]; then
    log "Download complete, verifying…"
    if verify_download "${DST}"; then
      log "✓ Verified and saved via curl (org: ${ORG}, banch: ${BRANCH}) - ${DST} (${QUANT_U})"
      chmod 444 "${DST}" # Apply special permission
      exit 0
    else
      log "✗ Verification failed; removing and trying next"
      rm -f "${DST}"
    fi
  else
    log "✗ Curl failed; trying next"
  fi
done

# -----------------------------------------------------------------------------
# All methods failed
log "ERROR: All download methods failed for ${FILENAME}"
exit 1
