#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** create_map_file.sh is tool that creates tensors.map files **#
#** for your gguf models.                                     **#
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
#** Copyright © 2026 - Thireus.        ₁₋ᵦᵢₜ ᵦᵣₐᵢₙ, ₃₂₋ᵦᵢₜ ₑ𝓰ₒ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# Trap SIGINT/SIGTERM for clean exit
trap 'echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Exiting." >&2; exit 0' SIGINT SIGTERM

#
# USER CONFIGURATION
#
# (Leave empty or add any bash vars you need here)

#
# End of user configuration
#

# --------------- DETECT & DEFINE SHA256 HELPER ---------------
# Try to find a suitable sha256 utility
if command -v sha256sum >/dev/null 2>&1; then
  # GNU coreutils on Linux
  sha256tool=(sha256sum)
  args=()
elif command -v gsha256sum >/dev/null 2>&1; then
  # GNU coreutils on macOS (via Homebrew)
  sha256tool=(gsha256sum)
  args=()
elif command -v shasum >/dev/null 2>&1; then
  # macOS built-in (Perl script)
  sha256tool=(shasum)
  args=(-a 256)
elif command -v openssl >/dev/null 2>&1; then
  # OpenSSL fallback
  sha256tool=(openssl)
  args=(dgst -sha256)
else
  # No hashing tool found; leave sha256tool empty to indicate failure
  sha256tool=()
  args=()
fi

# Only define _sha256sum if we actually found a tool
if (( ${#sha256tool[@]} > 0 )); then
  # _sha256sum reads either from file (if you pass an arg) or from stdin
  _sha256sum() {
    if (( $# > 0 )); then
      # file-mode: pass filename as $1
      "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
    else
      # stdin-mode: read data from pipe
      "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
    fi
  }
fi

command -v _sha256sum &>/dev/null || echo "⚠️  Warning: _sha256sum command missing - hash cannot be computed!" >&2

# Check that _sha256sum is available
if ! command -v _sha256sum &>/dev/null; then
    echo "Warning: '_sha256sum' not found. Entries will omit hash." >&2
    USE_SHA256=false
else
    USE_SHA256=true
fi

# Require at least one .gguf file argument
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 file1.gguf [file2.gguf ...]"
    exit 1
fi

for gguf_file in "$@"; do
    # Verify file exists and is .gguf
    if [[ ! -f "$gguf_file" || "${gguf_file##*.}" != "gguf" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping '$gguf_file': not a .gguf file or not found." >&2
        continue
    fi

    dir="$(dirname "$gguf_file")"
    fname="$(basename "$gguf_file")"
    map_file="$dir/tensors.map"

    # Create map file if it doesn't exist
    if [[ ! -f "$map_file" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating map file: $map_file"
        : > "$map_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Appending to existing map: $map_file"
    fi

    # Compute SHA256 if possible
    if [[ "$USE_SHA256" == "true" ]]; then
        file_hash="$(_sha256sum "$gguf_file" | awk '{print $1}')"
    else
        file_hash=""
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running gguf_info.py on '$fname'..."
    if gguf_info.py "$gguf_file" | \
       while IFS= read -r line; do
           # Skip headers or empty lines
           [[ "$line" =~ ^=== ]] && continue
           [[ -z "$line" ]] && continue

           # Parse tab‑delimited fields
           IFS=$'\t' read -r tensor_name field_shape field_dtype field_elements field_bytes <<< "$line"
           if [[ -z "$tensor_name" || -z "$field_shape" ]]; then
               echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Malformed line: '$line'" >&2
               continue
           fi

           # Append entry to map:
           # filename:sha256?:tensor_name:shape=…:dtype=…:elements=…:bytes=…
           if [[ -n "$file_hash" ]]; then
               echo "${fname}:${file_hash}:${tensor_name}:${field_shape}:${field_dtype}:${field_elements}:${field_bytes}" \
                   >> "$map_file"
           else
               echo "${fname}:${file_hash}:${tensor_name}:${field_shape}:${field_dtype}:${field_elements}:${field_bytes}" \
                   >> "$map_file"
           fi
       done
    then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished processing '$fname'."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: gguf_info.py failed on '$fname'." >&2
    fi

done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done."
