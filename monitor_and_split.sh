#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** monitor_and_split.sh is a script that auto splits GGUF    **#
#** models.                                                   **#
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
#** Copyright © 2025 - Thireus.         Wᵣₒₜ 𝒹ᵢₛ ᵢₙ ₁₋ᵦᵢₜ ₘₒₒ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# GLOBAL OPTION: delete original .gguf after successful split?
# Default is "false". Override by exporting DELETE_ORIGINAL=true or editing here.
DELETE_ORIGINAL="${DELETE_ORIGINAL:-false}"

# Trap SIGINT/SIGTERM to allow a clean exit message
trap 'echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Exiting."; exit 0' SIGINT SIGTERM

# Directory to monitor: first argument, default to current directory if not provided
WATCH_DIR="${1:-.}"

# Verify that the directory exists and is indeed a directory
if [[ ! -d "$WATCH_DIR" ]]; then
    echo "Error: '$WATCH_DIR' is not a directory or does not exist."
    exit 1
fi

# Check that llama-gguf-split is available
if ! command -v llama-gguf-split &>/dev/null; then
    echo "Warning: 'llama-gguf-split' not found in PATH. Ensure it is installed and in PATH."
    # We do not exit here; we'll let the split fail per-file if it's missing.
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting monitoring of directory: $WATCH_DIR"
echo "Polling every 60 seconds. Looking for '*-SPECIAL.gguf' older than 10 minutes."

# Infinite loop
while true; do
    # Find files matching "*-SPECIAL.gguf" in WATCH_DIR (non-recursive by default; adjust -maxdepth as needed)
    # whose modification time is more than 10 minutes ago.
    # Using -maxdepth 1 to limit to the top-level of WATCH_DIR. If you want recursive search, remove -maxdepth 1.
    while IFS= read -r -d '' file; do
        # Get the base filename (no directory)
        filename="$(basename "$file")"
        # Strip the .gguf extension
        base="${filename%.gguf}"
        # Define the split directory name
        split_dir="$WATCH_DIR/${base}_SPLIT"
        
        # If the split directory does not exist, create it and run the split
        if [[ ! -d "$split_dir" ]]; then
            timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
            echo "[$timestamp] Detected file older than 10 minutes: '$file'"
            echo "[$timestamp] Creating directory: '$split_dir'"
            if mkdir -p "$split_dir"; then
                echo "[$timestamp] Directory created successfully."
            else
                echo "[$timestamp] ERROR: Failed to create directory '$split_dir'. Skipping this file."
                continue
            fi

            # Build the output split filename
            out_gguf="$split_dir/${base}_TENSOR"
            echo "[$timestamp] Running split: llama-gguf-split --no-tensor-first-split --split-max-tensors 1 '$file' '$out_gguf'"
            
            if llama-gguf-split --no-tensor-first-split --split-max-tensors 1 "$file" "$out_gguf"; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Split succeeded for '$file'. Output in '$split_dir/'."
                # If enabled, delete the original gguf
                if [[ "$DELETE_ORIGINAL" == "true" ]]; then
                    if rm -f "$file"; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deleted original file: '$file'."
                    else
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Failed to delete original file: '$file'."
                    fi
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping deletion of original (DELETE_ORIGINAL=false)."
                fi
                chmod 444 "$split_dir"/*.gguf # Lock files
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Split failed for '$file'. You may want to check llama-gguf-split availability or file integrity."
                # Optionally, could remove the created directory or leave for inspection.
            fi
        else
            # Directory already exists: skip
            :
            # If you want verbose logging, uncomment the next line:
            # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Split directory already exists for '$file'; skipping."
        fi
    done < <(find "$WATCH_DIR" -maxdepth 1 -type f -name "*-SPECIAL.gguf" -mmin +10 -print0)

    # Sleep for 60 seconds before next check
    sleep 60
done
