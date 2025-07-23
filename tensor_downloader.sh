#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** tensor_downloader.sh is a powerful downloader tool that   **#
#** downloads pre-quantised tensors/shards to cook recipes.   **#
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
#   FileID          (mandatory) integer chunk ID; 0 => "tensors.map"; -1 => "tensors.map.sig"; -2 => "*-00001-of-*.gguf.sig"
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

# Local fallbacks:
# COPY_FOLDERS: local directories to copy from
COPY_FOLDERS=(
  #"/mnt/local_storage/DeepSeek"
)
# SYMLINK_FOLDERS: local directories to symlink from
SYMLINK_FOLDERS=(
  #"/mnt/local_storage/DeepSeek"
)

# Default download order if DOWNLOAD_ORDER is unset or empty:
#   RSYNC first, then CURL, then COPY, then SYMLINK
DEFAULT_ORDER=(SYMLINK COPY RSYNC CURL)

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
# Compute effective DOWNLOAD_ORDER
# - If user did not set DOWNLOAD_ORDER or it’s empty, use DEFAULT_ORDER.
# - Otherwise, take only valid entries (CURL or RSYNC for example) in the user-provided list.
USER_ORDER=( "${DOWNLOAD_ORDER[@]:-}" )
ORDER=()
if [ ${#USER_ORDER[@]} -eq 0 ]; then
  ORDER=( "${DEFAULT_ORDER[@]}" )
else
  ORDER=()
  for m in "${USER_ORDER[@]}"; do
    m_up="${m^^}"
    if [[ " ${DEFAULT_ORDER[*]} " == *" $m_up "* ]]; then
      ORDER+=( "$m_up" )
    fi
  done
fi

# # Append any missing defaults
# for m in "${DEFAULT_ORDER[@]}"; do
#   skip=
#   for u in "${ORDER[@]}"; do
#     [[ "$u" == "$m" ]] && { skip=1; break; }
#   done
#   [[ -n "$skip" ]] || ORDER+=( "$m" )
# done

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

verify_asc() {
  local f=$1
  local begin_count end_count first_line last_line

  # 1) File must exist and be readable
  if [[ ! -r "$f" ]]; then
    echo "Error: '$f' not found or not readable." >&2
    return 1
  fi

  # 2) Count BEGIN / END blocks
  begin_count=$(grep -c '^-----BEGIN PGP PUBLIC KEY BLOCK-----$' "$f")
  end_count  =$(grep -c '^-----END PGP PUBLIC KEY BLOCK-----$'  "$f")

  # 3) There must be at least one block, and counts must match
  if (( begin_count == 0 )); then
    echo "Error: no BEGIN PGP PUBLIC KEY BLOCK found." >&2
    return 1
  fi
  if (( begin_count != end_count )); then
    echo "Error: $begin_count BEGIN but $end_count END markers." >&2
    return 1
  fi

  # 4) First non-empty line must be BEGIN
  first_line=$(grep -v '^[[:space:]]*$' "$f" | head -n1)
  if [[ "$first_line" != '-----BEGIN PGP PUBLIC KEY BLOCK-----' ]]; then
    echo "Error: first non-empty line is not BEGIN marker." >&2
    return 1
  fi

  # 5) Last non-empty line must be END
  last_line=$(grep -v '^[[:space:]]*$' "$f" | tail -n1)
  if [[ "$last_line" != '-----END PGP PUBLIC KEY BLOCK-----' ]]; then
    echo "Error: last non-empty line is not END marker." >&2
    return 1
  fi

  # 6) (Optional) Ensure each block contains only valid base64 or header lines
  #    Here we check that no lines between BEGIN/END contain illegal chars:
  if grep -v -E '^(-----BEGIN PGP PUBLIC KEY BLOCK-----|Version:|Comment:|[A-Za-z0-9+/=]+|-----END PGP PUBLIC KEY BLOCK-----)$' "$f" >/dev/null; then
    echo "Warning: found lines that aren't headers or valid Base64 characters." >&2
    # not fatal, you can choose to return 1 here if you want to enforce stricter checks
  fi

  return 0
}

verify_sig() {
  local sigfile=$1
  local size

  # 1) exists & readable
  if [[ ! -r "$sigfile" ]]; then
    echo "Error: '$sigfile' not found or not readable." >&2
    return 1
  fi

  # 2) non-empty
  if [[ ! -s "$sigfile" ]]; then
    echo "Error: '$sigfile' is empty." >&2
    return 1
  fi

  # 3) too small (less than 20 bytes?)
  size=$(wc -c < "$sigfile")
  if (( size < 20 )); then
    echo "Error: '$sigfile' is too small ($size bytes) to be a valid signature." >&2
    return 1
  fi

  return 0
}

# Wrapper: choose which verify to run
verify_download() {
  local file="$1"
  if [ "$FileID" -lt 0 ]; then
    verify_sig "$file"
  elif [ "$FileID" -eq 0 ]; then
    verify_map "$file"
  else
    verify_chunk "$file"
  fi
}

# -----------------------------------------------------------------------------
# Build filename and prepare download
if [ "$FileID" -eq 0 ]; then
  FILENAME="tensors.map"
elif [ "$FileID" -eq -1 ]; then
  FILENAME="tensors.map.sig"
elif [ "$FileID" -eq -2 ]; then
  FILENAME="${MODEL_NAME}-${MAINTAINER}-${QUANT_U}-SPECIAL_TENSOR-00001-of-${CHUNKS}.gguf.sig"
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
do_rsync() {
  for srv in "${RSYNC_SERVERS[@]}"; do
    IFS=":" read -r RUSER RHOST RPORT RPATH <<< "$srv"
    SRC="${RUSER}@${RHOST}:${RPATH}/${REPOSITORY_NAME}/${FILENAME}"
    DST="${DEST}/${CUSTOM_FILENAME}"

    if [ -f "${DST}" ]; then
      log "File already exists, verifying…"
      if verify_download "${DST}"; then
        log "✓ Verified; no need to download it again - ${DST} (${QUANT_U})"
        chmod 444 "${DST}"
        return 0
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
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Verification failed; removing and trying next"
        rm -f "${DST}"
      fi
    else
      log "✗ Rsync failed; trying next"
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Curl attempts
do_curl() {
  for crl in "${CURL_ORGS[@]}"; do
    IFS=":" read -r ORG BRANCH <<< "$crl"
    URL="https://huggingface.co/${ORG}/${REPOSITORY_NAME}/resolve/${BRANCH}/${FILENAME}?download=true"
    DST="${DEST}/${CUSTOM_FILENAME}"

    if [ -f "${DST}" ]; then
      log "File already exists, verifying…"
      if verify_download "${DST}"; then
        log "✓ Verified; no need to download it again - ${DST} (${QUANT_U})"
        chmod 444 "${DST}"
        return 0
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
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Verification failed; removing and trying next"
        rm -f "${DST}"
      fi
    else
      log "✗ Curl failed; trying next"
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Copy attempts
do_copy() {
  for folder in "${COPY_FOLDERS[@]}"; do
    SRC="${folder}/${REPOSITORY_NAME}/${FILENAME}"
    DST="${DEST}/${CUSTOM_FILENAME}"

    if [ ! -f "${SRC}" ]; then
      log "✗ Source not found for copy: ${SRC}; trying next"
      continue
    fi

    if [ -f "${DST}" ]; then
      log "File already exists, verifying…"
      if verify_download "${DST}"; then
        log "✓ Verified; no need to copy it again - ${DST} (${QUANT_U})"
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Verification failed; removing and copying it"
        rm -f "${DST}"
      fi
    fi

    log "Trying copy from ${SRC}"
    cp --preserve=mode,timestamps "${SRC}" "${DST}"
    if [ $? -eq 0 ]; then
      log "Copy complete, verifying…"
      if verify_download "${DST}"; then
        log "✓ Verified and saved via copy - ${DST} (${QUANT_U})"
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Verification failed after copy; removing and trying next"
        rm -f "${DST}"
      fi
    else
      log "✗ Copy failed; trying next"
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Symlink attempts
do_symlink() {
  for folder in "${SYMLINK_FOLDERS[@]}"; do
    SRC="${folder}/${REPOSITORY_NAME}/${FILENAME}"
    DST="${DEST}/${CUSTOM_FILENAME}"

    if [ ! -f "${SRC}" ]; then
      log "✗ Source not found for symlink: ${SRC}; trying next"
      continue
    fi

    if [ -e "${DST}" ]; then
      if [ -L "${DST}" ] && [ "$(readlink "${DST}")" = "${SRC}" ] && verify_download "${DST}"; then
        log "✓ Verified existing symlink - ${DST} → ${SRC}"
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Existing file/symlink invalid; removing and recreating symlink"
        rm -f "${DST}"
      fi
    fi

    log "Trying symlink from ${SRC}"
    ln -s "${SRC}" "${DST}"
    if [ $? -eq 0 ]; then
      log "Symlink created, verifying…"
      if verify_download "${DST}"; then
        log "✓ Verified and linked - ${DST} → ${SRC}"
        chmod 444 "${DST}"
        return 0
      else
        log "✗ Verification failed after symlink; removing and trying next"
        rm -f "${DST}"
      fi
    else
      log "✗ Symlink creation failed; trying next"
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Execute in the requested order
for method in "${ORDER[@]}"; do
  case "$method" in
    RSYNC)
      do_rsync && exit 0
      ;;
    CURL)
      do_curl && exit 0
      ;;
    COPY)
      do_copy && exit 0
      ;;
    SYMLINK)
      do_symlink && exit 0
      ;;
  esac
done

# -----------------------------------------------------------------------------
# All methods failed
log "ERROR: All download methods failed for ${FILENAME}"
exit 1
