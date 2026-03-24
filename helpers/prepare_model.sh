#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** prepare_model.sh is used to produce GPG signatures of the **#
#** first model GGUF shard and tensors.map file.              **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-24-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.  ₗₗₘ ₘₐₗ𝓌ₐᵣₑ ₗₒᵥₑ ₚₕᵢₛₕ₋ₐₙ𝒹₋𝒸ₕᵢₚₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail
IFS=$'\n\t'

# Script to prepare a model directory with GPG signing,
# Modes:
#   --metadata-only       Only sign/verify the 00001 gguf (metadata) file; ignore tensors.map
#   --tensors-map-only    Only sign/verify tensors.map; ignore 00001 gguf

SRC_PATH=""
METADATA_ONLY=0
TENSORS_MAP_ONLY=0

usage() {
  cat <<EOF
Usage: $0 -p <path-to-model-folder> [--metadata-only | --tensors-map-only]

  -p PATH                 Path to local folder you want to upload (when required)
  --metadata-only         Ignore tensors.map; wait for and sign the *-00001-of-*.gguf metadata file only
  --tensors-map-only      Ignore the 00001 gguf file; wait for and sign tensors.map only
  -h, --help              Show this help and exit

Notes:
  - --metadata-only and --tensors-map-only are mutually exclusive.
  - In default mode (no extra flags) the script requires tensors.map and will:
      * count tensors from tensors.map to compute total shards,
      * wait for tensors.map to be complete (last line references final shard),
      * wait for the *-00001-of-<PADDED_SHARDS>.gguf file to exist,
      * then create and verify detached signatures for tensors.map and the 00001 gguf file.
EOF
  exit 1
}

# Parse command-line arguments (support long options)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) SRC_PATH=$2; shift 2 ;;
    --metadata-only) METADATA_ONLY=1; shift ;;
    --tensors-map-only) TENSORS_MAP_ONLY=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# Validate flags
if [[ $METADATA_ONLY -eq 1 && $TENSORS_MAP_ONLY -eq 1 ]]; then
  echo "ERROR: --metadata-only and --tensors-map-only are mutually exclusive." >&2
  usage
fi

# Path is required for all modes
if [[ -z $SRC_PATH ]]; then
  echo "ERROR: -p <path-to-model-folder> is required." >&2
  usage
fi

# Resolve absolute path
SRC_PATH=$(realpath "$SRC_PATH")

# Safety: ensure the path is a directory
if [[ ! -d $SRC_PATH ]]; then
  echo "ERROR: Path '$SRC_PATH' is not a directory or doesn't exist." >&2
  exit 2
fi

# Helper: run mode loops inside the source directory
cd "$SRC_PATH"

# Enable globbing that yields empty array (so we can check matches)
shopt -s nullglob

# MODE: metadata-only
if [[ $METADATA_ONLY -eq 1 ]]; then
  echo "Running in --metadata-only mode: ignoring tensors.map and signing the *-00001-of-*.gguf metadata file only."

  # Wait until a file matching *-00001-of-*.gguf exists, then sign & verify it
  until
    matches=( *-00001-of-*.gguf ) && [[ ${#matches[@]} -gt 0 ]] \
    && { 
         # prefer the first match; if multiple exist warn but proceed with the first
         if [[ ${#matches[@]} -gt 1 ]]; then
           echo "Warning: multiple *-00001-of-*.gguf files found — using '${matches[0]}'." >&2
         fi
         gguf_basename="${matches[0]}"
         # create signature and verify it (interactive style)
         gpg --pinentry-mode loopback --detach-sign "$gguf_basename" \
         && gpg --verify "${gguf_basename}.sig"
       }
  do
    # If until condition failed, retry after a pause
    echo "metadata gguf file not ready or GPG failed — retrying in 5 seconds..."
    sleep 5
  done

  echo "metadata-only: signed and verified '$gguf_basename' successfully."
  exit 0
fi

# MODE: tensors-map-only
if [[ $TENSORS_MAP_ONLY -eq 1 ]]; then
  echo "Running in --tensors-map-only mode: ignoring 00001 gguf file and signing tensors.map only."

  # Wait until tensors.map exists, then sign & verify it
  until
    [[ -f tensors.map ]] \
    && gpg --pinentry-mode loopback --detach-sign tensors.map \
    && gpg --verify tensors.map.sig
  do
    echo "tensors.map not ready or GPG failed — retrying in 5 seconds..."
    sleep 5
  done

  echo "tensors-map-only: signed and verified 'tensors.map' successfully."
  exit 0
fi

# DEFAULT MODE (both tensors.map and metadata gguf)
MAP_FILE="tensors.map"
SRC_PATH="${SRC_PATH:-.}"   # keep earlier behaviour if you use SRC_PATH elsewhere

# Loop until everything is ready and successfully signed/verified.
# All checks & calculations are re-run each iteration.
until
  # Recompute checks every iteration (do NOT use a subshell so variables remain visible)
  if [[ ! -f "$MAP_FILE" ]]; then
    echo "ERROR: $MAP_FILE not found in '$SRC_PATH'." >&2
    false
  else
    # Recompute counts and padded values
    TENSORS=$(wc -l < "$MAP_FILE" | tr -d '[:space:]')
    TOTAL_SHARDS=$((TENSORS + 1))

    # Format with leading zeros (width=5)
    PADDED_TENSORS=$(printf "%05d" "$TENSORS")
    PADDED_SHARDS=$(printf "%05d" "$TOTAL_SHARDS")

    echo "Detected $TENSORS TENSORS in $MAP_FILE, expecting last shard #$PADDED_SHARDS."

    # Pattern for the metadata file we expect for the first shard
    TARGET_GGUF_PATTERN="*-00001-of-${PADDED_SHARDS}.gguf"

    # Expand glob safely: enable nullglob so no-match -> empty array (not literal pattern)
    shopt -s nullglob
    matches=( $TARGET_GGUF_PATTERN )
    shopt -u nullglob

    # Check last-line presence of the final-shard reference, row count sanity, presence of metadata gguf,
    # then perform GPG sign & verify operations (map file first, then target gguf)
    if tail -n1 "$MAP_FILE" | grep -q -- "-${PADDED_SHARDS}-of-${PADDED_SHARDS}\.gguf" \
       && [[ $(wc -l < "$MAP_FILE" | tr -d '[:space:]') -eq $TENSORS ]] \
       && [[ ${#matches[@]} -gt 0 ]] \
       && gpg --pinentry-mode loopback --detach-sign "$MAP_FILE" \
       && gpg --verify "${MAP_FILE}.sig" "$MAP_FILE" \
    ; then
      # choose first match, warn if multiple
      if [[ ${#matches[@]} -gt 1 ]]; then
        echo "Warning: multiple files matching '$TARGET_GGUF_PATTERN' — using '${matches[0]}'." >&2
      fi
      target_gguf="${matches[0]}"

      # sign & verify the target gguf; whole if returns true only if these succeed
      gpg --pinentry-mode loopback --detach-sign "$target_gguf" \
      && gpg --verify "${target_gguf}.sig" "$target_gguf"
    else
      false
    fi
  fi
do
  echo "tensors.map not ready, metadata gguf missing, or GPG failed — retrying in 5 seconds..." >&2
  sleep 5
done

echo "Signed and verified tensors.map and metadata gguf ('$target_gguf') successfully."
exit 0
