#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_downloader.sh is a tool that downloads GGUF shards  **#
#** from a recipe file containing tensor regexe entries.      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Feb-04-2026 -------------------- **#
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
#** Copyright © 2025 - Thireus.        𝒻ₐᵢₗₑ𝒹 ₜₒ ₐₗₗₒ𝒸ₐₜₑ ᵦᵤ𝒻𝒻ₑᵣ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ----- Debugging: verbose ERR trap -----
# Capture the exact command line used to start the script (script name + all args).
# We build this safely so it works on Bash 3.2 and preserves arguments with spaces.
ORIGINAL_INVOCATION="$0"
for __arg in "$@"; do
  # protect any dollar signs so later printing doesn't unexpectedly expand things
  # keep it simple and safe for older bash versions
  escaped="${__arg//\$/\\$}"
  ORIGINAL_INVOCATION="${ORIGINAL_INVOCATION} ${escaped}"
done
# Also save the raw args array if you want to inspect individual params later
ORIGINAL_ARGS=("$@")

error_reporting() {
  # save exit status immediately
  local _err="$?"
  # save the command that caused the trap
  local _cmd="${BASH_COMMAND:-}"
  # save shell flags and pid info
  local _shellflags="$-"
  local _pid=$$
  local _ppid=$PPID
  # avoid set -e in trap doing weird things; make trap body robust
  set +e

  {
    printf '\n===== ERR-TRAP - PLEASE REPORT THE ISSUE HERE: https://github.com/Thireus/GGUF-Tool-Suite/issues =====\n' >&2
    printf 'Timestamp: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >&2
    printf 'Exit status: %s\n' "$_err" >&2
    printf 'Failed command: %s\n' "$_cmd" >&2

    # Print the exact invocation that started this script (script + parameters)
    printf 'Script invocation (user command): %s\n' "${ORIGINAL_INVOCATION:-unknown}" >&2
    # show the raw args as an indexed list (useful if args contain spaces)
    if [[ ${#ORIGINAL_ARGS[@]} -gt 0 ]]; then
      printf 'Script args (one per line):\n' >&2
      local __i=0
      for __a in "${ORIGINAL_ARGS[@]}"; do
        printf '  [%d] %s\n' "$__i" "$__a" >&2
        __i=$((__i + 1))
      done
    else
      printf 'Script args: (none)\n' >&2
    fi

    # try to show source file and line number where the error happened
    local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]:-unknown}}"
    local lineno="${BASH_LINENO[0]:-unknown}"
    printf 'Location: %s:%s\n' "$src" "$lineno" >&2

    # Print function/stack trace (if any). Compatible with bash 3.2.
    printf 'Stack trace (most recent call first):\n' >&2
    local i
    for i in "${!FUNCNAME[@]}"; do
      # FUNCNAME[0] is the current function (error_reporting); skip it if desired
      printf '  %d: %s()  at %s:%s\n' "$i" "${FUNCNAME[$i]:-MAIN}" "${BASH_SOURCE[$i]:-?}" "${BASH_LINENO[$i-1]:-?}" >&2
    done

    printf 'Shell flags: %s\n' "$_shellflags" >&2
    printf 'PID: %s  PPID: %s\n' "$_pid" "$_ppid" >&2

    # show active background jobs (if any)
    printf 'Background jobs (jobs -rp): %s\n' "$(jobs -rp 2>/dev/null || true)" >&2

    # dump some useful shell variables for debugging (avoid unbound vars)
    printf 'BASH_COMMAND (raw): %s\n' "$BASH_COMMAND" >&2
    printf 'Last pipeline status: %s\n' "${PIPESTATUS[*]:-unknown}" >&2

    printf '===== END ERR-TRAP =====\n\n' >&2
  } || true

  # restore errexit behaviour for the rest of the script
  set -e
}

# Install the trap. Use single quotes so expansion occurs at trap time, not now.
trap 'error_reporting' ERR
# ----------------------------------------------------------------

# DEBUG function
if [[ -n "${DEBUG:-}" ]]; then
  DEBUG() { printf "DEBUG: %s\n" "$*" >&2; }
else
  DEBUG() { :; }
fi

# ---------------- SIGNAL HANDLING ----------------
# Graceful shutdown on Ctrl+C / SIGTERM
INT_TIMESTAMP=0

shutdown_on_signal() {
  local rc=130  # conventional exit code for SIGINT
  DEBUG "shutdown_on_signal: received signal, initiating graceful shutdown (exit $rc)" >&2

  # Get running child PIDs started by this shell (jobs -rp)
  local pids
  pids="$(jobs -rp 2>/dev/null || true)"

  if [[ -n "${pids:-}" ]]; then
    DEBUG "shutdown_on_signal: killing child PIDs: $pids" >&2

    # ask nicely first
    kill $pids 2>/dev/null || true
    # give them a second to exit gracefully
    sleep 1

    # force-kill any remaining
    pids="$(jobs -rp 2>/dev/null || true)"
    if [[ -n "${pids:-}" ]]; then
      DEBUG "shutdown_on_signal: force-killing remaining PIDs: $pids" >&2
      kill -KILL $pids 2>/dev/null || true
    fi

    # Reap all children (wait without args waits for all children)
    # Use set +e to avoid aborting if wait returns non-zero.
    set +e
    wait
    set -e
  else
    DEBUG "shutdown_on_signal: no running child jobs found" >&2
  fi

  # Optionally print partial state if you maintain WRAPPER_STATUS/WRAPPER_RAW
  if declare -p WRAPPER_STATUS >/dev/null 2>&1; then
    DEBUG "shutdown_on_signal: current WRAPPER_STATUS snapshot:" >&2
    for k in "${!WRAPPER_STATUS[@]:-}"; do
      printf "DEBUG: idx=%s status=%s raw=%s\n" "$k" "${WRAPPER_STATUS[$k]:-}" "${WRAPPER_RAW[$k]:-}" >&2
    done
  fi

  # If you have any cleanup traps for FIFOs etc, they will run because we exit now.
  if [[ $INT_TIMESTAMP -eq 0 ]]; then
    echo "💀 Termination signal received — forwarding SIGINT to the entire process group and starting graceful shutdown (exit ${rc:-?})." >&2
    echo "   Allowing up to 10 seconds for subprocesses to exit cleanly if possible." >&2
    INT_TIMESTAMP=$(date +%s)
  elif [[ $(( $(date +%s) - INT_TIMESTAMP )) -gt 10 ]]; then
    echo "💀 10 seconds elapsed — forcing termination now: sending SIGKILL to the process group (-$$)." >&2
    kill -s KILL -- -$$ 2>/dev/null || true
  fi
  sleep 1 # Slow down the shutdown_on_signal loop
  kill -s INT -- -$$ 2>/dev/null || true
  exit "$rc"
}

# install the trap for INT and TERM
trap 'shutdown_on_signal' INT TERM
# ------------------------------------------------------------

# Record main script PID so subshells/background workers can cause the entire script to exit.
# This is used by exit_from_subprocess() below to ensure that any fatal condition (previously returning 666)
# terminates the whole script rather than just the function or subshell.
SCRIPT_MAIN_PID=$$

# Fatal exit helper: ensures whole script terminates from functions or subshells.
exit_from_subprocess() {
  local _code="${1:-1}"
  kill -TERM "${SCRIPT_MAIN_PID}" 2>/dev/null || true
  # Exit this process (if parent still running, above kills should stop it).
  exit "$_code"
}

# ----------------- DEFAULTS & INITIALIZATION -----------------
MAX_JOBS=8                   # Default concurrency level
NEW_MAP=true                 # Whether to try obtain the latest map files
FORCE_REDOWNLOAD=false       # Whether to redownload all files (maps, shards, first shard)
VERIFY=false                 # If true, only verify hashes and report errors
VERIFY_READONLY=false        # If true, verify in read-only mode: do not create files in the target dir; use temporary workspace
BASE_DIR="."                 # Base directory for model and download dirs
QTYPE="BF16"                 # Default quantization type used for the first shard and filenames - also used for F32 tensors
QTYPE_SPECIFIED=false        # Whether the user explicitly passed --qtype (new flag)
SKIP_GPG=false               # If true, skip the gpg signature verification of the signed files
SKIP_HASH=false              # If true, skip sha256 hash computations and treat them as valid
SPECIAL_NODE_ID=""           # Optional: number of nodes (see --special-node-id). If non-empty, only shards assigned to the deterministic node are downloaded.
TOTAL_NODES=""               # Optional: Total number of nodes (see --total-nodes).
RM_SKIPPED_SHARDS=false      # Optional: Remove shards that are skipped for this node (not performed in --verify mode).
MODEL_NAME=""                # Will be read from $SCRIPT_DIR/download.conf when --special-node-id is used.
RETRY_ATTEMPTS=3             # Retry config for pathological symlink/network cases (only for read ops)
RETRY_DELAY=10               # Retry config for pathological symlink/network cases (only for read ops)
ARCHIVE_COMPRESS=false       # -z / --z-compress : compress .gguf -> .gguf.zbst after verification (and move compressed files)
ARCHIVE_DECOMPRESS=false     # --z-decompress (aka -zd) : accept .gguf.zbst files and decompress them into .gguf (removing .zbst)
ARCHIVE_COMPRESS_OPT=""      # Deprecated single-string fallback; new preferred format: per-tool options via --z-compress-opt 'tool:opts'
ARCHIVE_DECOMPRESS_OPT=""    # Deprecated single-string fallback; new preferred format: per-tool options via --z-decompress-opt 'tool:opts'
ARCHIVE_NOAUTO=false         # --z-noauto: prevent automatic enabling of -z or -zd based on files present

# Enforce that any files produced by the downloader that are .gguf or .gguf.zbst must remain symlinks
SYMLINK_ONLY=false

# Only process individual tensors list (numbers, comma separated)
INDIVIDUAL_TENSORS_ENABLED=false
INDIVIDUAL_TENSORS_RAW=""
declare -A IND_TENSOR_SET=()   # filled after reading maps, keys are decimal integers (1-based chunk ids)

# Default tools and their magic hex (user can override with --z-custom-tools)
# Example magic hex values: zstd -> 28B52FFD, lbzip2 -> 425A68
CUSTOM_TOOLS=("zstd:28B52FFD" "lbzip2:425A68")  # default custom tools if user doesn't override
# arrays filled after parsing: CUSTOM_TOOL_NAMES[], CUSTOM_TOOL_MAGICS[]
CUSTOM_TOOL_NAMES=()
CUSTOM_TOOL_MAGICS=()

# Raw per-tool compress/decompress option specifications (multiple allowed)
# Each entry expected like 'zstd:-19 -D mydict' or 'lbzip2:-9 -u'
COMPRESS_OPTS_RAW=()     # holds 'tool:opts' strings from --z-compress-opt (or defaults)
DECOMPRESS_OPTS_RAW=()   # holds 'tool:opts' strings from --z-decompress-opt (default empty unless user supplies)

# parsed per-tool opts: COMP_OP_TOOL_NAMES[], COMP_OP_TOOL_VALUES[], same for DECOMP
COMP_OP_TOOL_NAMES=()
COMP_OP_TOOL_VALUES=()
DECOMP_OP_TOOL_NAMES=()
DECOMP_OP_TOOL_VALUES=()
# -------------------------------------------------------------------------

# --------------------- USAGE & ARG PARSING -------------------
usage() {
  echo "Usage: $0 [options] <recipe-file>" >&2
  echo "       --no-new-map                      Prevent the script from downloading new map files" >&2
  echo "       --force-redownload                Force redownload of all shards and maps, ignoring existing files" >&2
  echo "       --verify                          Only verify existing shard hashes; report mismatches; skip downloads" >&2
  echo "       --verify-readonly                 Same as --verify but do not create files in the target directory (use a temporary workspace)." >&2
  echo "       --qtype QUANT                     Set quantization type for the first shard and filenames (default: BF16); use highest qtype of the model!" >&2
  echo "                                         NOTE: When --qtype is explicitly provided and the corresponding" >&2
  echo "                                         tensors.<qtype,,>.map and tensors.<qtype,,>.map.sig files are present (<qtype,,> is automatically lowercased)," >&2
  echo "                                         the script will create these helpful symlinks in the model dir:" >&2
  echo "                                           tensors.map -> tensors.<qtype,,>.map" >&2
  echo "                                           tensors.map.sig -> tensors.<qtype,,>.map.sig" >&2
  echo "                                         This makes it easier to use quantized model repositories locally." >&2
  echo "       --skip-gpg                        Do not verify the gpg signature of the downloaded files" >&2
  echo "       --skip-hash                       Do not compute or verify SHA256 hashes; treat all hashes as valid (useful when quantizing BF16 shards with a different imatrix for example)" >&2
  echo "       --special-node-id N               Only process shards assigned to this node for this model." >&2
  echo "       --total-nodes N                   Total number of nodes." >&2
  echo "       --rm-skipped-shards               Remove shards not assigned to this node if present." >&2
  echo "       --z-noauto                        Disable automatic enabling of -z or -zd based on files present in the model dir." >&2
  echo "       --z-custom-tools TOOL:MAGIC_HEX   Single comma-separated string of tool:magic pairs (example: 'zstd:28B52FFD,lbzip2:425A68,brotli:'), with one," >&2
  echo "                                         allowed not to have any magic which must be set at the end (tool selected upon first magic match in list order). Defaults to 'zstd:28B52FFD,lbzip2:425A68'" >&2
  echo "       --symlink-only                    When present, newly downloaded .gguf/.gguf.zbst files that are symlinks will remain symlinks; the script will fail," >&2
  echo "                                         instead of replacing or creating regular files." >&2
  echo "       --individual-tensors LIST         Comma-separated list of tensor numbers to operate on (example: --individual-tensors 1,2,6)." >&2
  echo "                                         When provided the script will SKIP downloading and verifying any shard not listed." >&2
  echo "                                         Note: numbers must be positive integers within the model's shard count and unique." >&2
  echo "  -j,  --max-jobs N                      Set maximum concurrent downloads (default: $MAX_JOBS)" >&2
  echo "  -d,  --dest DIR                        Base path for model and download dirs (default: .)" >&2
  echo "  -z,  --z-compress                      Compress verified .gguf files to .gguf.zbst (using the best compression produced by --z-custom-tools tools) before," >&2
  echo "                                         moving out of the download dir" >&2
  echo "       --z-compress-opt OPTS             Single comma-separated string of per-tool compress opts: 'zstd:-19,lbzip2:-9 -u,brotli:-Z -n'. Defaults to ," >&2
  echo "                                         'zstd:-19,lbzip2:-9 -u'." >&2
  echo "  -zd, --z-decompress                    Accept .gguf.zbst files: they will be decompressed to .gguf and .gguf.zbst removed." >&2
  echo "       --z-decompress-opt OPTS           Single comma-separated string of per-tool decompress opts (default empty for all tools)." >&2
  echo "  -h,  --help                            Show this help and exit" >&2
  echo "  <recipe-file>: path to recipe containing USER_REGEX lines (one per tensor; must have .recipe extension)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  # Download all model GGUF shards for this DeepSeek-R1-0528 recipe:" >&2
  echo "    ./quant_downloader.sh DeepSeek-R1-0528.THIREUS-1.9413bpw-4.3624ppl.151GB-GGUF_11GB-GPU_140GB-CPU.569b7f6_bb4f3c8.recipe" >&2
  echo "  # Will download GGUF shards and compress them for archival or storage optimisation purpose - make sure to install the mentioned compression tools 'apt-get install lbzip2 brotli zstd'" >&2
  echo "    ./quant_downloader.sh -z --qtype Q8_0_R8 --z-custom-tools 'zstd:28B52FFD,lbzip2:425A68,brotli:' --z-compress-opt 'zstd:-19,lbzip2:-9 -u,brotli:-Z' -j 18 q8_0_r8.recipe" >&2
  echo "  # Automatically decompresses .zbst GGUF shards - must ensure the list of custom tools matches all the tools that may have been used to create the .zbst present on the host repositories" >&2
  echo "    ./quant_downloader.sh -zd --z-custom-tools 'zstd:28B52FFD,lbzip2:425A68,brotli:' my_custom.recipe" >&2
  echo "  # Verify only individual tensors 2,3,1094:" >&2
  echo "    ./quant_downloader.sh --individual-tensors 2,3,1094 --verify recipe.recipe" >&2
  exit 1
}

# Support user passing the two-letter short "-zd" (convert to long opt) to be compatible
# with the user's requested syntax. This translation happens before getopt so getopt
# doesn't try to interpret '-zd' as '-z -d'.
_preprocess_args=()
for __a in "$@"; do
  if [[ "$__a" == "-zd" ]]; then
    _preprocess_args+=(--z-decompress)
  else
    _preprocess_args+=("$__a")
  fi
done
set -- "${_preprocess_args[@]:-}"

# Parse arguments (supports GNU long options)
PARSED_OPTS=$(getopt -n "$0" -o j:d:hz -l max-jobs:,no-new-map,force-redownload,verify,verify-readonly,qtype:,skip-gpg,skip-hash,dest:,destination:,special-node-id:,total-nodes:,rm-skipped-shards,help,z-compress,z-decompress,z-noauto,z-compress-opt:,z-decompress-opt:,z-custom-tools:,symlink-only,individual-tensors: -- "$@") || usage
eval set -- "$PARSED_OPTS"
while true; do
  case "$1" in
    -j|--max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --no-new-map)
      NEW_MAP=false
      shift
      ;;
    --force-redownload)
      FORCE_REDOWNLOAD=true
      shift
      ;;
    --verify)
      VERIFY=true
      shift
      ;;
    --verify-readonly)
      VERIFY_READONLY=true
      VERIFY=true   # treat readonly as a variant of verify mode
      shift
      ;;
    --qtype)
      QTYPE="${2^^}"
      QTYPE_SPECIFIED=true
      shift 2
      ;;
    --skip-gpg)
      SKIP_GPG=true
      shift
      ;;
    --skip-hash)
      SKIP_HASH=true
      shift
      ;;
    --special-node-id)
      SPECIAL_NODE_ID="$2"
      shift 2
      ;;
    --total-nodes)
      TOTAL_NODES="$2"
      shift 2
      ;;
    --rm-skipped-shards)
      RM_SKIPPED_SHARDS=true
      shift
      ;;
    -d|--dest|--destination)
      BASE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    -z|--z-compress)
      ARCHIVE_COMPRESS=true
      shift
      ;;
    --z-decompress)
      ARCHIVE_DECOMPRESS=true
      shift
      ;;
    --z-noauto)
      ARCHIVE_NOAUTO=true
      shift
      ;;
    --z-compress-opt)
      # Expect a single comma-separated string in $2: 'zstd:-19,lbzip2:-9 -u'
      IFS=',' read -r -a __comp_parts <<< "$2"
      for __c in "${__comp_parts[@]}"; do
        __c="${__c#"${__c%%[![:space:]]*}"}"
        __c="${__c%"${__c##*[![:space:]]}"}"
        if [[ -n "$__c" ]]; then
          COMPRESS_OPTS_RAW+=("$__c")
        fi
      done
      shift 2
      ;;
    --z-decompress-opt)
      # Expect a single comma-separated string in $2
      IFS=',' read -r -a __decomp_parts <<< "$2"
      for __c in "${__decomp_parts[@]}"; do
        __c="${__c#"${__c%%[![:space:]]*}"}"
        __c="${__c%"${__c##*[![:space:]]}"}"
        if [[ -n "$__c" ]]; then
          DECOMPRESS_OPTS_RAW+=("$__c")
        fi
      done
      shift 2
      ;;
    --z-custom-tools)
      # Accept only a single comma-separated string of tool:magic entries.
      IFS=',' read -r -a __tools_parts <<< "$2"
      for __t in "${__tools_parts[@]}"; do
        # trim whitespace
        __t="${__t#"${__t%%[![:space:]]*}"}"
        __t="${__t%"${__t##*[![:space:]]}"}"
        if [[ -n "$__t" ]]; then
          CUSTOM_TOOLS+=("$__t")
        fi
      done
      shift 2
      ;;
    --symlink-only)
      SYMLINK_ONLY=true
      shift
      ;;
    --individual-tensors)
      INDIVIDUAL_TENSORS_RAW="$2"
      INDIVIDUAL_TENSORS_ENABLED=true
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      ;;
  esac
done

if [[ "$ARCHIVE_COMPRESS" == true && "$ARCHIVE_DECOMPRESS" == true ]]; then
  echo "❌ Error: --z-compress and --z-decompress cannot bed used at the same time." >&2
  exit 3
fi

# ------------------------------------------------------------------------
# Build per-tool compress/decompress option structures.
# If the user didn't supply per-tool options, set sensible defaults.
# ------------------------------------------------------------------------
# If user did not supply any COMPRESS_OPTS_RAW, use default per-tool compress opts:
if [[ ${#COMPRESS_OPTS_RAW[@]} -eq 0 ]]; then
  COMPRESS_OPTS_RAW=("zstd:-19" "lbzip2:-9 -u")
fi
# DECOMP defaults are empty (user may set)
# Now parse CUSTOM_TOOLS list into names+magics
CUSTOM_TOOL_NAMES=()
CUSTOM_TOOL_MAGICS=()
for entry in "${CUSTOM_TOOLS[@]}"; do
  # allow both "tool:hex" or "tool:hex:extra" (we only use first two fields)
  tool_name="${entry%%:*}"
  rest="${entry#*:}"
  magic="${rest%%:*}"
  # normalize
  tool_name="${tool_name,,}" # lowercase for command lookup convenience

  # deduplicate tool_name entries: keep the first and skip duplicates
  skip=false
  for _existing in "${CUSTOM_TOOL_NAMES[@]}"; do
    if [[ "$_existing" == "$tool_name" ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == true ]]; then
    DEBUG "Skipping duplicate custom tool entry for '$tool_name' from CUSTOM_TOOLS" >&2
    continue
  fi

  CUSTOM_TOOL_NAMES+=("$tool_name")
  CUSTOM_TOOL_MAGICS+=("${magic^^}")
done

# parse COMPRESS_OPTS_RAW into COMP_OP_TOOL_NAMES/VALUES
COMP_OP_TOOL_NAMES=()
COMP_OP_TOOL_VALUES=()
for e in "${COMPRESS_OPTS_RAW[@]}"; do
  tool="${e%%:*}"
  opts="${e#*:}"
  tool="${tool,,}"
  # deduplicate entries: only add if tool name not already present in COMP_OP_TOOL_NAMES
  already=false
  for _n in "${COMP_OP_TOOL_NAMES[@]}"; do
    if [[ "$_n" == "$tool" ]]; then
      already=true
      break
    fi
  done
  if [[ "$already" == false ]]; then
    COMP_OP_TOOL_NAMES+=("$tool")
    COMP_OP_TOOL_VALUES+=("$opts")
  else
    # if duplicate, prefer the first occurrence; skip subsequent duplicates quietly
    DEBUG "Skipping duplicate compress opts for tool: $tool" >&2
  fi
done
# parse DECOMP_OPs
DECOMP_OP_TOOL_NAMES=()
DECOMP_OP_TOOL_VALUES=()
for e in "${DECOMPRESS_OPTS_RAW[@]}"; do
  tool="${e%%:*}"
  opts="${e#*:}"
  tool="${tool,,}"
  already=false
  for _n in "${DECOMP_OP_TOOL_NAMES[@]}"; do
    if [[ "$_n" == "$tool" ]]; then
      already=true
      break
    fi
  done
  if [[ "$already" == false ]]; then
    DECOMP_OP_TOOL_NAMES+=("$tool")
    DECOMP_OP_TOOL_VALUES+=("$opts")
  else
    DEBUG "Skipping duplicate decompress opts for tool: $tool" >&2
  fi
done

# helpers to fetch per-tool options (string)
get_compress_opts_for_tool() {
  local t="$1"
  for i in "${!COMP_OP_TOOL_NAMES[@]}"; do
    if [[ "${COMP_OP_TOOL_NAMES[$i]}" == "$t" ]]; then
      echo "${COMP_OP_TOOL_VALUES[$i]}"
      return 0
    fi
  done
  # not found -> empty
  echo ""
  return 0
}
get_decompress_opts_for_tool() {
  local t="$1"
  for i in "${!DECOMP_OP_TOOL_NAMES[@]}"; do
    if [[ "${DECOMP_OP_TOOL_NAMES[$i]}" == "$t" ]]; then
      echo "${DECOMP_OP_TOOL_VALUES[$i]}"
      return 0
    fi
  done
  # not found -> empty
  echo ""
  return 0
}

# ------------------------------------------------------------------------
# If the compression options include a dictionary (-D <path> or -D/path or -D=path)
# on a given tool, and the user did NOT supply -D in that tool's --z-decompress-opt,
# automatically add the same -D <path> to that tool's decompression options so
# decompression used for GPG verification (or stream-hash) uses the same dictionary.
#
# Warn the user when we add it automatically so they can override with
# --z-decompress-opt if they prefer a different decompression configuration.
# ------------------------------------------------------------------------
for i in "${!COMP_OP_TOOL_NAMES[@]}"; do
  tool="${COMP_OP_TOOL_NAMES[$i]}"
  opts="${COMP_OP_TOOL_VALUES[$i]}"

  # check for -D patterns in opts; support forms: -D <arg>, -D=arg, -Darg
  dict_path=""

  # Split opts into an array of tokens without touching positional params
  # Use read -r -a for bash array splitting on IFS (whitespace)
  IFS=$' \t\n' read -r -a tokens <<< "$opts"

  # Walk tokens looking for -D forms
  for (( j=0; j < ${#tokens[@]}; j++ )); do
    token="${tokens[j]}"
    if [[ "$token" == "-D" ]]; then
      # next token is the argument (if present)
      dict_path="${tokens[j+1]:-}"
      break
    elif [[ "$token" == -D=* ]]; then
      dict_path="${token#-D=}"
      break
    elif [[ "${token:0:2}" == "-D" && "${#token}" -gt 2 ]]; then
      dict_path="${token:2}"
      break
    fi
  done

  if [[ -n "$dict_path" ]]; then
    # See if decompress opts for this tool already contain any -D*
    existing="$(get_decompress_opts_for_tool "$tool")"
    hasD=false

    # Split existing into array as well
    IFS=$' \t\n' read -r -a ex_tokens <<< "$existing"
    for ex in "${ex_tokens[@]}"; do
      case "$ex" in
        -D* ) hasD=true; break ;;
      esac
    done

    if [[ "$hasD" == false ]]; then
      echo "⚠️ Warning: compression options for tool '$tool' include a dictionary (-D ${dict_path}) but --z-decompress-opt did not include -D for this tool. Adding same -D option to decompression settings automatically so verification/decompression will use the same dictionary." >&2
      # append to DECOMP_OP_TOOL_NAMES/VALUES arrays
      DECOMP_OP_TOOL_NAMES+=("$tool")
      # preserve any prior decompress opts for tool as empty + add -D
      DECOMP_OP_TOOL_VALUES+=("-D ${dict_path}")
    fi
  fi
done
# ------------------------------------------------------------------------

# ----------------------- DIRECTORIES -------------------------
# If verify-readonly mode is requested, we must not create files in BASE_DIR.
# Prepare a temporary workspace for maps/signatures and fail marker.
VERIFY_TMPDIR=""
MAP_DIR="."   # default map location is current working directory
if [[ "$VERIFY_READONLY" == true ]]; then
  VERIFY_TMPDIR=$(mktemp -d) || { echo "❌ Error: failed to create temporary workspace for --verify-readonly"; exit 8; }
  MAP_DIR="$VERIFY_TMPDIR"
  FAIL_MARKER="$VERIFY_TMPDIR/.verify_failed"
else
  # Ensure base directory exists (only when not verify-readonly)
  mkdir -p "$BASE_DIR"
  FAIL_MARKER="$BASE_DIR/.verify_failed"
fi

LOCAL_DOWNLOAD_DIR="$BASE_DIR/downloaded_shards"
LOCAL_MODEL_DIR="$BASE_DIR"
# Only create the download dir when not in verify-readonly mode (avoid creating dirs in read-only target)
if [[ "$VERIFY_READONLY" != true ]]; then
  mkdir -p "$LOCAL_DOWNLOAD_DIR"
fi

echo "[INFO] Using base directory: $BASE_DIR"
echo "[INFO] Download dir: $LOCAL_DOWNLOAD_DIR"
echo "[INFO] Model dir: $LOCAL_MODEL_DIR"
echo "[INFO] Max jobs: $MAX_JOBS, Obtain new map: $NEW_MAP, Force redownload: $FORCE_REDOWNLOAD, Verify only: $VERIFY, Verify-readonly: $VERIFY_READONLY, Skip signature verification: $SKIP_GPG, Skip hash verification: $SKIP_HASH"
# Automatic selection of z mode based on files present
#
# 1) If neither -z nor -zd specified and neither was auto-disabled (--z-noauto),
#    and there are more .gguf.zbst files than .gguf files in the model dir, auto-enable -z.
# 2) If neither -z nor -zd specified and neither was auto-disabled, and there are more
#    .gguf files than .gguf.zbst files in the model dir, auto-enable -zd.
#    HOWEVER: when running with --verify/--verify-only, prefer enabling -z instead.
# 3) Users can disable this automatic behavior with --z-noauto.
if [[ "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true && "$ARCHIVE_NOAUTO" != true ]]; then
  # Count files robustly using nullglob to avoid find portability differences
  shopt -s nullglob 2>/dev/null || true
  gguf_files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
  zbst_files=(  "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
  shopt -u nullglob 2>/dev/null || true

  # Get array lengths (do NOT use ${#array[@]:-0} — some shells reject that)
  count_gguf=${#gguf_files[@]}
  count_zbst=${#zbst_files[@]}

  # Ensure numeric 0 if arrays are unsupported for some reason
  : ${count_gguf:=0}
  : ${count_zbst:=0}

  if (( count_gguf < count_zbst )); then
    ARCHIVE_COMPRESS=true
    echo "⚠️ Warning: Auto-enabled -z (compression) because there are more .gguf.zbst (${count_zbst}) than .gguf (${count_gguf}) files in '$LOCAL_MODEL_DIR'!" >&2
    echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
  elif (( count_zbst > 0 )) && (( count_zbst < count_gguf )); then
    if [[ "$VERIFY" == true ]]; then
      # In verify mode, prefer verifying compressed streams; choose -z
      ARCHIVE_COMPRESS=true
      echo "⚠️ Warning: Auto-enabled -z (verify .gguf.zbst) because --verify mode detected more .gguf (${count_gguf}) than .gguf.zbst (${count_zbst}) files in '$LOCAL_MODEL_DIR'!" >&2
      echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
    else
      ARCHIVE_DECOMPRESS=true
      echo "⚠️ Warning: Auto-enabled -zd (decompress) because there are more .gguf (${count_gguf}) than .gguf.zbst (${count_zbst}) files in '$LOCAL_MODEL_DIR'!" >&2
      echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
    fi
  fi
fi

if [[ "$ARCHIVE_COMPRESS" == true ]]; then
  echo "[INFO] compression enabled: verified .gguf files will be (or will remain) compressed to .gguf.zbst"
  # print per-tool compress opts for visibility
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    opts="$(get_compress_opts_for_tool "$t")"
    if [[ -n "$opts" ]]; then
      echo "[INFO] compress opts for $t: ${opts}"
    fi
  done
elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  echo "[INFO] z-decompress mode enabled: will accept .gguf.zbst files and decompress them into .gguf (removing .zbst)"
fi
if [[ "$ARCHIVE_COMPRESS" == true ]] || [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  # Echo tools on a single line (comma-separated). Use a subshell so we don't change the caller's IFS.
  if [[ ${#CUSTOM_TOOL_NAMES[@]} -gt 0 ]]; then
    ( IFS=','; printf '[INFO] compression tools: %s\n' "${CUSTOM_TOOL_NAMES[*]}" )
  else
    echo "[INFO] compression tools: (none)"
  fi
  # Show per-tool decompress opts (if any)
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    opts="$(get_decompress_opts_for_tool "$t")"
    if [[ -n "$opts" ]]; then
      echo "[INFO] decompress opts for $t: ${opts}"
    fi
  done
fi

# If either compression-related option is used (either user-provided or auto-enabled), ensure compression tools are available
if [[ "$ARCHIVE_COMPRESS" == true || "$ARCHIVE_DECOMPRESS" == true ]]; then
  missing_tools=()
  
  # Check each tool
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    if ! command -v "$t" >/dev/null 2>&1; then
      missing_tools+=("$t")
    fi
  done
  
  # If any tools are missing, show detailed error and exit
  if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "❌ Error: None of these configured compression tools are available: ${missing_tools[*]}" >&2
    exit 2
  fi
fi

# --------------- Incompatibility checks -------------
# -zd (z-decompress) must NOT be used with --verify (or --verify-readonly), because verify may alter files.
if [[ "$ARCHIVE_DECOMPRESS" == true && "$VERIFY" == true ]]; then
  echo "❌ Error: --z-decompress (-zd) is incompatible with --verify/--verify-readonly. Use --z-compress (-z) instead. Usage of --z-decompress-opt is permitted." >&2
  exit 1
fi
# --------------------------------------------------

# -----------------------------------------------------------------
# Filter out accidental empty-string positional arguments (e.g. when caller passes "")
# This preserves existing behavior for legitimate non-empty args but ignores blank placeholders.
NEW_POS_ARGS=()
for _arg in "$@"; do
  if [[ -n "$_arg" ]]; then
    NEW_POS_ARGS+=("$_arg")
  fi
done
# Reset positional parameters to the filtered list (works even if empty)
set -- "${NEW_POS_ARGS[@]:-}"
# -----------------------------------------------------------------

# Check recipe-file argument
if [[ $# -ne 1 ]]; then
  usage
fi
RECIPE_FILE="$1"
# Enforce .recipe extension
if [[ ! "$RECIPE_FILE" == *.recipe && ! "$RECIPE_FILE" == *.recipe.txt ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' must have a .recipe or .recipe.txt extension." >&2
  exit 1
fi
if [[ ! -f "$RECIPE_FILE" ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' not found." >&2
  exit 1
fi

# ----------------------- DIRECTORIES -------------------------
# If verify-readonly mode is requested, we must not create files in BASE_DIR.
# Prepare a temporary workspace for maps/signatures and fail marker.
VERIFY_TMPDIR=""
MAP_DIR="."   # default map location is current working directory
if [[ "$VERIFY_READONLY" == true ]]; then
  VERIFY_TMPDIR=$(mktemp -d) || { echo "❌ Error: failed to create temporary workspace for --verify-readonly"; exit 8; }
  MAP_DIR="$VERIFY_TMPDIR"
  FAIL_MARKER="$VERIFY_TMPDIR/.verify_failed"
else
  # Ensure base directory exists (only when not verify-readonly)
  mkdir -p "$BASE_DIR"
  FAIL_MARKER="$BASE_DIR/.verify_failed"
fi

LOCAL_DOWNLOAD_DIR="$BASE_DIR/downloaded_shards"
LOCAL_MODEL_DIR="$BASE_DIR"
# Only create the download dir when not in verify-readonly mode (avoid creating dirs in read-only target)
if [[ "$VERIFY_READONLY" != true ]]; then
  mkdir -p "$LOCAL_DOWNLOAD_DIR"
fi

echo "[INFO] Using base directory: $BASE_DIR"
echo "[INFO] Download dir: $LOCAL_DOWNLOAD_DIR"
echo "[INFO] Model dir: $LOCAL_MODEL_DIR"
echo "[INFO] Max jobs: $MAX_JOBS, Obtain new map: $NEW_MAP, Force redownload: $FORCE_REDOWNLOAD, Verify only: $VERIFY, Verify-readonly: $VERIFY_READONLY, Skip signature verification: $SKIP_GPG, Skip hash verification: $SKIP_HASH"

if [[ -n "$SPECIAL_NODE_ID" ]]; then
  if [[ -n "$TOTAL_NODES" ]]; then
    echo "[INFO] special-node-id/(total-nodes - 1) set to: $SPECIAL_NODE_ID/$((TOTAL_NODES-1)) (only shards assigned to this model/node pair will be downloaded)"
  else
    echo "❌ Error: --total-nodes N must be specified when using the --special-node-id option!" >&2
    exit 1
  fi
elif [[ -n "$TOTAL_NODES" ]]; then
  echo "❌ Error: --special-node-id N must be specified when using the --total-nodes option!" >&2
  exit 1
fi

# Ensure we clean up the temporary verify/read-only workspace on exit if created.
cleanup_verify_readonly() {
  # remove tempdir if we created it
  if [[ -n "$VERIFY_TMPDIR" && -d "$VERIFY_TMPDIR" ]]; then
    rm -rf "$VERIFY_TMPDIR" || true
  fi
  # also remove GNUPG_TMPDIR if left behind (best-effort)
  if [[ -n "${GNUPG_TMPDIR:-}" && -d "$GNUPG_TMPDIR" ]]; then
    rm -rf "$GNUPG_TMPDIR" || true
  fi
}
# Install EXIT trap to always attempt cleanup (runs after other EXIT handlers; it's ok)
trap 'cleanup_verify_readonly' EXIT

# ----------------------- SYMLINK TRACKING -------------------------
# Track previous symlink targets for downloaded files so we can detect repeated identical symlink sources.
declare -A PREV_SYMLINK_SOURCE=()

# Helper: get absolute path of symlink target if possible (readlink -f), fall back to readlink raw value.
_resolve_symlink_target() {
  local path="$1"
  local resolved=""
  if command -v readlink >/dev/null 2>&1; then
    # Try readlink -f for canonical absolute path (works on GNU)
    if readlink -f "$path" >/dev/null 2>&1; then
      resolved="$(readlink -f "$path" 2>/dev/null || true)"
    else
      # fallback to raw link content and convert relative to absolute based on symlink location
      local target
      target="$(readlink "$path" 2>/dev/null || true)"
      if [[ -z "$target" ]]; then
        resolved=""
      elif [[ "$target" = /* ]]; then
        resolved="$target"
      else
        # relative symlink -> resolve relative to symlink dir
        resolved="$(cd "$(dirname "$path")" >/dev/null 2>&1 && printf '%s/%s' "$(pwd)" "$target" 2>/dev/null || printf '%s/%s' "$(dirname "$path")" "$target")"
      fi
    fi
  else
    # no readlink? degrade gracefully
    resolved="$(readlink "$path" 2>/dev/null || true)"
  fi
  printf '%s' "$resolved"
}

# -------------------- TIMESTAMP FUNCTION ---------------------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# --------------- DETECT & DEFINE SHA256 HELPER ---------------
# If SKIP_HASH is set, don't require any sha256 utility and provide a harmless stub that
# returns an empty string (the calling logic will treat skip mode as "passed").
if [[ "$SKIP_HASH" == true ]]; then
  # Define a stub _sha256sum that always succeeds but returns empty (caller will substitute expected hash where needed).
  _sha256sum() {
    # Do nothing, return success with empty output.
    printf ''
    return 0
  }
else
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
    # fallback stub: always errors out (we don't have hashing capability)
    sha256tool=()
    args=()
  fi

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

# ----------------------- SYMLINK/RETRY HELPERS (minimal) -----------------------
# retry_exec: run a command repeatedly (exec mode). Usage: retry_exec cmd arg1 arg2 ...
retry_exec() {
  local attempt=1
  while :; do
    if "$@"; then
      return 0
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: command failed (attempt $attempt/${RETRY_ATTEMPTS}): $*" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

# retry_capture: run a command and capture stdout with retries. Usage: out="$(retry_capture cmd arg1 ...)"
retry_capture() {
  local attempt=1
  local out
  while :; do
    if out="$("$@" 2>/dev/null)"; then
      printf '%s' "$out"
      return 0
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: command failed (attempt $attempt/${RETRY_ATTEMPTS}): $*" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

# ensure_path_available: when a path is a symlink pointing to a network mount
# we retry waiting for the *target* to become available. Returns 0 if path (or
# symlink target) becomes available within RETRY_ATTEMPTS; non-zero otherwise.
ensure_path_available() {
  local path="$1"
  local attempt=1
  while :; do
    if [[ -e "$path" ]]; then
      return 0
    fi
    # If it's a symlink, but target not present, wait and retry
    if [[ -L "$path" ]]; then
      # if target becomes available, -e "$path" above will be true; otherwise we retry
      :
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: path not available (attempt $attempt/${RETRY_ATTEMPTS}): $path" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

is_symlink() { [[ -L "$1" ]]; }

# safe_file_exists: when checking file existence, if path is a symlink attempt retries
safe_file_exists() {
  local path="$1"
  if is_symlink "$path"; then
    ensure_path_available "$path" || return 1
  fi
  [[ -f "$path" ]]
}

# safe_sha256sum: ensure path available if symlink, then run _sha256sum (with retries if symlink)
safe_sha256sum() {
  local path="$1"
  if [[ "$SKIP_HASH" == true ]]; then
    # Skip computing hash; caller will treat result as valid (return empty string).
    printf ''
    return 0
  fi
  if is_symlink "$path"; then
    if ! ensure_path_available "$path"; then
      return 1
    fi
    retry_capture _sha256sum "$path"
  else
    _sha256sum "$path"
  fi
}

# safe_stream_sha256_from_z: compute sha256 of the decompressed stream of a .gguf.zbst file without creating a .gguf file.
# Robustly capture tool exit status even when piping (treat decode errors as failures).
safe_stream_sha256_from_z() {
  local z="$1"
  if [[ "$SKIP_HASH" == true ]]; then
    # Skip computing stream hash; return empty so callers can decide behaviour (we typically set got=exp when skip enabled)
    printf ''
    return 0
  fi
  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local out
  local rc=404
  # Temporarily disable errexit so we can capture the pipeline's exit status cleanly.
  set +e

  # Calculate the maximum number of bytes needed
  local max_magic_bytes=0
  for magic in "${CUSTOM_TOOL_MAGICS[@]}"; do
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -gt "$max_magic_bytes" ]; then
      max_magic_bytes="$magic_byte_count"
    fi
  done

  # Read the first N bytes of the file only once
  local file_header=""
  if [ "$max_magic_bytes" -gt 0 ]; then
    file_header=$(xxd -p -l "$max_magic_bytes" "$1" | tr '[:upper:]' '[:lower:]')
  fi

  # Run magic-matching tool to produce the output file.
  local tool=''
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    # Check if file starts with magic
    local magic="${CUSTOM_TOOL_MAGICS[$i]}"
    local magic_byte_count=$(( ${#magic} / 2 ))
    
    # Compare with the pre-read header
    if [ "$magic_byte_count" -le "$max_magic_bytes" ] && [ "${file_header:0:${#magic}}" = "$(echo "$magic" | tr '[:upper:]' '[:lower:]')" ]; then
      tool="${CUSTOM_TOOL_NAMES[$i]}"
      local opts="$(get_decompress_opts_for_tool "$tool")"

      # Run pipeline in a subshell with pipefail so any failure (including tool decoding) sets non-zero exit code.
      echo "[$(timestamp)] z-decompress (stream): ${tool} ${opts} -k -d -c -- \"$z\"" >&2
      out=$(
        (
          set -o pipefail
          ${tool} ${opts} -k -d -c -- "$z" 2>/dev/null | _sha256sum
        )
      )
      
      rc=$?
      # Special condition for lbzip2 exit code 4
      [[ "${tool}" == "lbzip2" && $rc -eq 4 ]] && rc=0

      break # Don't proceed further
    fi
  done

  set -e

  if [[ $rc -ne 0 ]]; then
    # Treat tool decode errors (and any other pipeline failures) as corruption.
    return $rc
  fi

  printf '%s' "$out"
  return 0
}

# decompress_archive_to_file: write decompressed .gguf from a .gguf.zbst to a target file
# Now checks tool exit code and removes partial output on failure.
# Additionally: If the produced file is size 0 it is considered a decompression failure.
decompress_archive_to_file() {
  local z="$1"
  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local out="$2"

  # If symlink-only policy active: decompressing a symlinked .gguf.zbst would produce a regular file in this location;
  # refuse because we must not replace or create a regular file when the source is a symlink.
  if [[ "$SYMLINK_ONLY" == true && -L "$z" && "$3" != "skip_symlink_force" ]]; then
    echo "❌ Error: Refusing to decompress '$z' because it is a symlink and --symlink-only is enabled. Decompression would create a regular .gguf file. Remove --symlink-only or provide a non-symlink input." >&2
    exit_from_subprocess 15
  fi

  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local rc=404
  # Temporarily disable errexit to capture lbzip2 exit code.
  set +e
  rm -f "$out" || true

  # Calculate the maximum number of bytes needed
  local max_magic_bytes=0
  for magic in "${CUSTOM_TOOL_MAGICS[@]}"; do
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -gt "$max_magic_bytes" ]; then
      max_magic_bytes="$magic_byte_count"
    fi
  done

  # Read the first N bytes of the file only once
  local file_header=""
  if [ "$max_magic_bytes" -gt 0 ]; then
    file_header=$(xxd -p -l "$max_magic_bytes" "$1" | tr '[:upper:]' '[:lower:]')
  fi

  # Run magic-matching tool to produce the output file.
  local tool=''
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    # Check if file starts with magic
    local magic="${CUSTOM_TOOL_MAGICS[$i]}"
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -le "$max_magic_bytes" ] && [ "${file_header:0:${#magic}}" = "$(echo "$magic" | tr '[:upper:]' '[:lower:]')" ]; then
      tool="${CUSTOM_TOOL_NAMES[$i]}"
      local opts="$(get_decompress_opts_for_tool "$tool")"
      echo "[$(timestamp)] z-decompress: ${tool} ${opts} -k -d -c -- \"$z\" > \"$out\"" >&2
      "${tool}" ${opts} -k -d -c -- "$z" > "$out" 2>/dev/null
      rc=$?
      # Special condition for lbzip2 exit code 4
      [[ "${tool}" == "lbzip2" && $rc -eq 4 ]] && rc=0
      break # Don't proceed further
    fi
  done

  # If tool returned non-zero, treat as failure and remove any partial output.
  if [[ $rc -ne 0 ]]; then
    rm -f "$out" || true
    set -e
    return $rc
  fi

  # tool reported success — but ensure the produced file is not empty.
  if [[ ! -s "$out" ]]; then
    # zero-length file -> treat as decompression failure
    rm -f "$out" || true
    set -e
    return 1
  fi

  # Set restrictive permissions on the valid output.
  chmod 444 "$out" || true
  set -e
  return 0
}

# compress_gguf_to_archive: compress a .gguf into multiple compression formats using get_compress_opts_for_tool, only keep the smallest compressed file and name it .gguf.zbst and remove original on success
compress_gguf_to_archive() {
  local gguf="$1"

  # If symlink-only policy is active and the input file is a symlink we must refuse because compression will create a regular .zbst file.
  if [[ "$SYMLINK_ONLY" == true && -L "$gguf" ]]; then
    echo "❌ Error: Refusing to compress '$gguf' because it is a symlink and --symlink-only is enabled. Compression would create a regular .gguf.zbst file and could alter the symlink source directory. Remove --symlink-only or provide a non-symlink input." >&2
    exit_from_subprocess 14
  fi

  local z="${gguf}.zbst"
  # Use the system lbzip2 (we already checked earlier it's installed when needed)
  rm -f "$z" || true
  local prev_z_tool=''
  local final_z_tool=''

  # Temporary directory to record per-tool exit statuses when running in parallel
  local tmpdir
  tmpdir="$(mktemp -d)" || return 1
  trap 'rm -rf "${tmpdir:-}"' RETURN

  # Start all compression tools in the background so they operate simultaneously.
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local z_tool="${gguf}.${tool}"
    local opts
    opts="$(get_compress_opts_for_tool "$tool")"

    rm -f "$z_tool" || true

    (
      echo "[$(timestamp)] z-compress: ${tool} ${opts} -k -c -- \"$gguf\" > \"$z_tool\"" >&2

      # Some tools don't support symlinks
      if [[ -L "$gguf" ]]; then
        local resolved_gguf=$(readlink -f "$gguf")
      else
        local resolved_gguf="$gguf"
      fi

      if "${tool}" ${opts} -k -c -- "$resolved_gguf" > "$z_tool"; then
        local _rc=0
        chmod 444 "$z_tool" || true
      else
        local _rc=$?
        rm -f "$z_tool" || true
      fi

      printf "%s" "${_rc}" > "${tmpdir}/status_${i}"
    ) &
  done

  wait

  # Check statuses
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local rc
    if [[ -f "${tmpdir}/status_${i}" ]]; then
      rc="$(cat "${tmpdir}/status_${i}")"
    else
      rc=127
    fi

    if [[ "${tool}" == "lbzip2" ]]; then
      if [[ "${rc}" -ne 0 && "${rc}" -ne 4 ]]; then
        echo "❌ Error: ${tool} compression failed for '$gguf' (rc=${rc}), please verify the compression parameters!" >&2
        for j in "${!CUSTOM_TOOL_NAMES[@]}"; do
          rm -f "${gguf}.${CUSTOM_TOOL_NAMES[$j]}" || true
        done
        exit_from_subprocess 18
        return "${rc}"
      fi
    else
      if [[ "${rc}" -ne 0 ]]; then
        echo "❌ Error: ${tool} compression failed for '$gguf' (rc=${rc}), please verify the compression parameters!" >&2
        for j in "${!CUSTOM_TOOL_NAMES[@]}"; do
          rm -f "${gguf}.${CUSTOM_TOOL_NAMES[$j]}" || true
        done
        exit_from_subprocess 19
        return "${rc}"
      fi
    fi
  done

  # Find smallest compressed file
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local z_tool="${gguf}.${tool}"
    if [[ -f "$z_tool" ]]; then
      if [[ -n "$prev_z_tool" && -f "$prev_z_tool" ]]; then
        size_prev_z_tool=$(ls -l "$prev_z_tool" | awk '{print $5}')
        size_z_tool=$(ls -l "$z_tool" | awk '{print $5}')
        if (( size_prev_z_tool > size_z_tool )); then
          rm -f "$prev_z_tool" || true
          final_z_tool="$z_tool"
        else
          rm -f "$z_tool" || true
          final_z_tool="$prev_z_tool"
        fi
      else
        final_z_tool="$z_tool"
      fi
      prev_z_tool="$final_z_tool"
    fi
  done

  if [[ -z "$final_z_tool" || ! -f "$final_z_tool" ]]; then
    echo "❌ Error: No compressed files were produced for '$gguf'." >&2
    return 404
  fi

  mv "$final_z_tool" "$z"
  rm -f "$gguf"
  return 0
}

# safe_gpg_verify: run gpg --verify with retry semantics when either file is a symlink
safe_gpg_verify() {
  local sigfile="$1"
  local datafile="$2"
  if is_symlink "$sigfile" || is_symlink "$datafile"; then
    if is_symlink "$sigfile"; then
      ensure_path_available "$sigfile" || return 1
    fi
    if is_symlink "$datafile"; then
      ensure_path_available "$datafile" || return 1
    fi
    retry_exec gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$sigfile" "$datafile"
  else
    gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$sigfile" "$datafile"
  fi
}
# ----------------------------------------------------------------

# ------------------ READ USER_REGEX PATTERNS -----------------
declare -a USER_REGEX
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line##*( )}"
  line="${line%%*( )}"
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  USER_REGEX+=("$line")
done < "$RECIPE_FILE"

if [[ ${#USER_REGEX[@]} -eq 0 ]]; then
  echo "❌ Error: No valid USER_REGEX entries found in '$RECIPE_FILE'." >&2
  exit 1
fi

# ------------------ LOCATE DOWNLOADER ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"
if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
  echo "❌ Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
  exit 1
fi

# Wrap run_downloader to implement symlink-aware behavior and the new --symlink-only policy.
run_downloader() {
  # Original behavior: call the downloader and return its exit status.
  set +e
  "$TENSOR_DOWNLOADER" "$@" & CHILD_PID=$!; wait $CHILD_PID; ret=$?; set -e

  # After downloader returns, perform symlink handling for .gguf / .gguf.zbst files.
  # $@ typically: <qtype> <chunk_id> <destdir> <filename>
  local _args=("$@")
  local _n=${#_args[@]}
  if (( _n >= 1 )); then
    local filename="${_args[$((_n-1))]}"
    local destdir=""
    if (( _n >= 2 )); then
      destdir="${_args[$((_n-2))]}"
    fi
    # Only consider files that are .gguf or .gguf.zbst
    if [[ "$filename" == *.gguf || "$filename" == *.gguf.zbst ]]; then
      # Build full path (if destdir empty, assume current)
      local fullpath
      if [[ -n "$destdir" ]]; then
        fullpath="$destdir/$filename"
      else
        fullpath="./$filename"
      fi

      # If the downloader succeeded, but file is a symlink, process policies
      if [[ -L "$fullpath" ]]; then
        # Resolve symlink target (absolute preferred)
        local symlink_target
        symlink_target="$(_resolve_symlink_target "$fullpath")"

        # Detect repeated identical symlink source across subsequent downloader invocations.
        local prev="${PREV_SYMLINK_SOURCE["$fullpath"]:-}"
        if [[ -n "$prev" && -n "$symlink_target" && "$prev" == "$symlink_target" ]]; then
          echo "❌ Error: Re-downloaded '$fullpath' but it is still a symlink pointing to the same source ('$symlink_target'). This indicates the symlink source is missing or corrupted (or cannot be decompressed if enabled); aborting to avoid infinite retries. Investigate the symlink source directory." >&2
          exit_from_subprocess 10
        fi
        # Record this symlink target for future detection.
        PREV_SYMLINK_SOURCE["$fullpath"]="$symlink_target"

        # Enforce symlink-only policy conditions:
        if [[ "$SYMLINK_ONLY" == true ]]; then
          if [[ "$filename" == *.gguf && "$ARCHIVE_COMPRESS" == true ]]; then
            echo "❌ Error: Download created a .gguf symlink at '$fullpath'. --z (compress) is enabled and --symlink-only is set, so we cannot compress this symlink into a regular .gguf.zbst file. Remove --symlink-only or disable -z to proceed." >&2
            exit_from_subprocess 11
          fi
          if [[ "$filename" == *.gguf.zbst && "$ARCHIVE_DECOMPRESS" == true ]]; then
            echo "❌ Error: Download created a .gguf.zbst symlink at '$fullpath'. -zd (decompress) is enabled and --symlink-only is set, so we cannot decompress this symlink into a regular .gguf file in the working dir. Remove --symlink-only or disable -zd to proceed." >&2
            exit_from_subprocess 12
          fi
        fi
      fi

      # If downloader produced a .gguf.zbst but user has not enabled -z or -zd, that is unexpected:
      if [[ -f "$destdir/$filename" || -L "$destdir/$filename" ]]; then
        if [[ "$filename" == *.gguf.zbst && "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true ]]; then
          echo "❌ Error: The downloader produced a compressed file '$destdir/$filename' (.gguf.zbst) but you did not enable -z or -zd. Please rerun the script with either --z-compress (-z) or --z-decompress (-zd) to work with compressed files." >&2
          exit_from_subprocess 13
        fi
      fi
    fi
  fi

  return $ret
}

# ------------------------------------------------------------------------
# Track per-qtype whether the remote repository serves .gguf.zbst files.
# If we detect a successful fetch of a .gguf.zbst for a given qtype, we remember it
# so future downloads for that qtype request zbst directly.
# ------------------------------------------------------------------------
declare -A QTYPE_USES_ZBST=()

# Cleanup any stale qtype marker files from previous runs (startup)
if [[ -d "$LOCAL_MODEL_DIR" ]] && [[ "$VERIFY_READONLY" != true ]]; then
  # Use a safe glob delete (nullglob not required here but keep it robust)
  rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || true
fi

qtype_flagfile() { printf '%s/.qtype_zbst_%s' "$LOCAL_DOWNLOAD_DIR" "$1"; }

mark_qtype_zbst() {
  local q="${1^^}"
  QTYPE_USES_ZBST["$q"]=1   # best-effort in current shell
  # also write a marker visible to other processes
  [[ "$VERIFY_READONLY" != true ]] && touch "$(qtype_flagfile "$q")"
}

qtype_uses_zbst() {
  local q="${1^^}"
  if [[ -n "${QTYPE_USES_ZBST[$q]:-}" ]]; then
    return 0
  fi
  if [[ -f "$(qtype_flagfile "$q")" ]]; then
    # populate in-memory copy for speed
    QTYPE_USES_ZBST["$q"]=1
    return 0
  fi
  return 1
}

# Helper: transform a normal (chunk, filename) request to its .zbst equivalent.
# Accepts chunk and filename, returns transformed_chunk and transformed_filename.
_transform_to_zbst_request() {
  local chunk="$1"
  local filename="$2"

  local out_chunk="$chunk"
  local out_filename="$filename"

  # If chunk already starts with + (zbst positive), leave chunk as-is.
  if [[ "${chunk}" == +* ]]; then
    out_chunk="${chunk}"
  else
    # Numeric positive chunk -> prefix with + and zero-pad to 5 digits as tensor_downloader expects.
    if [[ "$chunk" =~ ^[0-9]+$ ]] && (( 10#$chunk >= 1 )); then
      out_chunk="+$(printf "%05d" "$chunk")"
    fi
  fi

  # Transform the filename suffixes
  if [[ "$filename" == *.gguf ]]; then
    out_filename="${filename%.gguf}.gguf.zbst"
  elif [[ "$filename" == *.gguf.zbst ]]; then
    out_filename="$filename"
  fi

  echo "$out_chunk" "$out_filename"
  return 0
}

# run_downloader_shard: wrapper for run_downloader that attempts a .gguf request first (normal behavior).
# If the requested file isn't produced and we are operating in compression/decompression modes,
# it will automatically attempt the corresponding .gguf.zbst request and, on success, remember that
# this qtype serves zbst files so future calls for that qtype request zbst directly.
run_downloader_shard() {
  local qtype="$1"
  local chunk="$2"
  local destdir="$3"
  local filename="$4"

  local _qtype="${qtype^^}"

  # Determine whether to attempt zbst-first (if we've recorded zbst for this qtype)
  if qtype_uses_zbst "$_qtype"; then
    # Try zbst-first for this qtype
    read -r zbst_chunk zbst_filename < <(_transform_to_zbst_request "$chunk" "$filename")
    # If transform didn't change anything (e.g. filename not .gguf), fall back to original attempt below
    if [[ "$zbst_chunk" != "$chunk" || "$zbst_filename" != "$filename" ]]; then
      set +e
      run_downloader "$qtype" "$zbst_chunk" "$destdir" "$zbst_filename"; ret=$?
      set -e
      if safe_file_exists "$destdir/$zbst_filename"; then
        # record again (idempotent) and return success
        mark_qtype_zbst "$_qtype"
        return 0
      fi
      # if zbst-first failed, fallthrough to try original normal request
    fi
  fi

  # 1) Try the normal/original request first (preserves existing behavior).
  set +e
  run_downloader "$qtype" "$chunk" "$destdir" "$filename"; ret=$?
  set -e

  if safe_file_exists "$destdir/$filename"; then
    return 0
  fi

  # 2) If we are not operating in any z mode, there's nothing to try.
  if [[ "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true ]]; then
    return $ret
  fi

  # 3) Try the zbst variant (only meaningful when original request was for shard files / signatures)
  read -r zbst_chunk zbst_filename < <(_transform_to_zbst_request "$chunk" "$filename")

  # If transformation didn't change anything, nothing more to try.
  if [[ "$zbst_chunk" == "$chunk" && "$zbst_filename" == "$filename" ]]; then
    return $ret
  fi

  # Attempt the .gguf.zbst request
  set +e
  run_downloader "$qtype" "$zbst_chunk" "$destdir" "$zbst_filename"; ret2=$?
  set -e

  if safe_file_exists "$destdir/$zbst_filename"; then
    # record that this qtype serves zbst so future requests use zbst-first
    echo "[$(timestamp)] Note: detected that the '${qtype^^}' repository serves .gguf.zbst files; switching to zbst-mode for subsequent downloads of this qtype." >&2
    mark_qtype_zbst "$_qtype"
  fi

  # Neither attempt produced the expected file.
  return $ret2
}
# ------------------------------------------------------------------------

# ------------------ VERIFY GPG READINESS ----------------------

if [[ "$SKIP_GPG" != true ]]; then
  if [ ! -f "$SCRIPT_DIR/trusted-keys.asc" ]; then
    echo "❌ Error: trusted-keys.asc not found in the script directory."
    echo "Hint: Provide trusted-keys.asc in the same directory as this script or use the --skip-gpg option to disable gpg signature verification."
    exit 6
  fi
  if command -v gpg >/dev/null 2>&1; then
    # Create a temporary GNUPGHOME
    GNUPG_TMPDIR=$(mktemp -d)
    if [ -z "$GNUPG_TMPDIR" ]; then
      echo "❌ Error: Failed to create temporary GPG home directory." >&2
      exit 8
    fi
    # Try importing the keys (silently) to check validity
    if ! gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --import "$SCRIPT_DIR/trusted-keys.asc" > /dev/null 2>&1; then
      echo "❌ Error: trusted-keys.asc contains missing or invalid GPG public keys."
      echo "Hint: Add valid public keys to this file or re-run with the --skip-gpg option to bypass signature verification."
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 7
    fi
  else
    echo "⚠️ Warning: 'gpg' command not found. Valid GPG public keys verification skipped." >&2
  fi
fi

# -------------------- HASH & SHARD STORAGE -------------------
declare -A T_HASHES SHARD_ID
set_t_hash() { local key="${1,,}::${2,,}"; T_HASHES["$key"]="$3"; DEBUG "set_t_hash T_HASHES["$key"]=${T_HASHES["$key"]}"; }
get_t_hash() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_HASHES["$key"]}"; DEBUG "get_t_hash T_HASHES["$key"]=${T_HASHES["$key"]}"; }
set_shard_id() { SHARD_ID["${1,,}"]="$2"; }
get_shard_id() { echo "${SHARD_ID["${1,,}"]}"; }

# -------- PREPARE QTYPES & PATTERNS --------
declare -a PATTERNS PATTERN_QTYPES
for entry in "${USER_REGEX[@]}"; do
  IFS='=' read -r pat qtype _ <<< "$entry"
  PATTERNS+=("$pat")
  PATTERN_QTYPES+=("$qtype")
done
readarray -t UNIQUE_QTYPES < <(printf "%s\n" "${PATTERN_QTYPES[@]}" | sort -u)

# Ensure QTYPE is present and is the first element (case-insensitive)
if [[ " ${UNIQUE_QTYPES[*]^^} " != *" ${QTYPE^^} "* ]]; then
  # not present -> prepend
  UNIQUE_QTYPES=("${QTYPE}" "${UNIQUE_QTYPES[@]}")
else
  # present, ensure it's first (case-insensitive match; preserve original casing)
  for i in "${!UNIQUE_QTYPES[@]}"; do
    if [[ "${UNIQUE_QTYPES[$i]^^}" == "${QTYPE^^}" ]]; then
      if [[ $i -ne 0 ]]; then
        val="${UNIQUE_QTYPES[$i]}"
        unset 'UNIQUE_QTYPES[i]'
        UNIQUE_QTYPES=("$val" "${UNIQUE_QTYPES[@]}")
      fi
      break
    fi
  done
fi

# --------------- FETCH MAPS & COLLECT ----------------
declare -a TENSORS_TO_FETCH SHARD_FILENAMES
# Keep track of which mapfiles we've already processed (case-insensitive)
declare -A PROCESSED_MAPFILES=()
for _q in "${UNIQUE_QTYPES[@]}"; do
  qtype=${_q^^}
  _qtype=$qtype
  [[ "$qtype" == "F32" ]] && _qtype="${QTYPE}"
  echo "[$(timestamp)] Fetching ${_qtype} tensor map (and gpg signature if enabled) for ${qtype} quants"

  # canonical mapfile name (lowercase) used as key to prevent duplicate processing
  mapfile="tensors.${_qtype,,}.map"
  mapkey="${mapfile,,}"    # normalized key (all-lowercase)
  mappath="$MAP_DIR/$mapfile"
  mapsigpath="$MAP_DIR/$mapfile.sig"

  # If we've already processed this mapfile, skip the rest of the loop.
  if [[ -n "${PROCESSED_MAPFILES[$mapkey]:-}" ]]; then
    echo "[$(timestamp)] Skipping already-processed mapfile: $mapfile"
    continue
  fi

  # Mark it as being processed now (prevents re-entrance if UNIQUE_QTYPES had duplicates)
  PROCESSED_MAPFILES["$mapkey"]=1

  if [[ "$FORCE_REDOWNLOAD" == true ]]; then
    echo "[$(timestamp)] Force redownload: removing existing map $mapfile and $mapfile.sig"
    rm -f "$mappath"
    rm -f "$mapsigpath"
    sync || true
  fi
  if [[ "$NEW_MAP" == true ]]; then
      if [[ -f "$mappath" ]]; then
          mv -f "$mappath" "$MAP_DIR/${mapfile}.bak"
          if [[ -f "$mapsigpath" ]]; then
            mv -f "$mapsigpath" "$MAP_DIR/${mapfile}.sig.bak"
          else
            rm -f "$MAP_DIR/${mapfile}.sig.bak" # Delete the backup because now the backup may not correspond to the $mapfile.bak
          fi
          if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
              echo "⚠️ Warning: failed to fetch map for $_qtype. Using existing map file." >&2
              mv -f "$MAP_DIR/${mapfile}.bak" "$mappath"
              if [[ -f "$MAP_DIR/${mapfile}.sig.bak" ]]; then
                mv -f "$MAP_DIR/${mapfile}.sig.bak" "$mapsigpath"
              fi
          else
              # Success: optionally remove backup or keep it
              rm -f "$MAP_DIR/${mapfile}.bak"
              # Download the signature
              if [[ "$SKIP_GPG" != true ]]; then
                if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
                    if [[ -f "$MAP_DIR/${mapfile}.sig.bak" ]]; then
                        echo "⚠️ Warning: failed to fetch map gpg signature for $_qtype. Using existing map gpg signature file." >&2
                        mv -f "$MAP_DIR/${mapfile}.sig.bak" "$mapsigpath"
                    else
                        echo "❌ Error: failed to fetch map gpg signature for $_qtype and no existing map gpg signature file present!" >&2
                        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                        exit 3
                    fi
                else
                    # Success: optionally remove backup or keep it
                    rm -f "$MAP_DIR/${mapfile}.sig.bak"
                fi
              fi
          fi
      else
          # $mapfile does not exist; just try downloading
          if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
              echo "❌ Error: failed to fetch map for $_qtype" >&2
              exit 1
          else
              # Download the signature
              if [[ "$SKIP_GPG" != true ]]; then
                # Do NOT remove existing signatures in the target model dir when in verify-readonly mode.
                if [[ "$VERIFY_READONLY" != true ]]; then
                  rm -f "$LOCAL_MODEL_DIR/$mapfile.sig"
                  sync || true
                else
                  DEBUG "verify-readonly: skipping removal of existing signature in LOCAL_MODEL_DIR for $mapfile.sig"
                fi
                if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
                    echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
                    [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                    exit 3
                fi
              fi
          fi
      fi
  else
      # NEW_MAP is false; just download normally (which will only happen if the file doesn't already exist)
      if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
          echo "❌ Error: failed to fetch map for $_qtype" >&2
          exit 1
      fi
      # Download the signature
      if [[ "$SKIP_GPG" != true ]]; then
        if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
            echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 3
        fi
      fi
  fi

  if [[ "$SKIP_GPG" != true ]]; then
    if command -v gpg >/dev/null 2>&1; then
      if [ ! -f "$mappath" ]; then
          echo "❌ Error: Map file '$mapfile' not found."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 5
      fi
      if [ ! -f "$mapsigpath" ]; then
          echo "❌ Error: Signature file '$mapfile.sig' is missing."
          echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 5
      fi
      if safe_gpg_verify "$mapsigpath" "$mappath" > /dev/null 2>&1; then
          echo "[$(timestamp)] GPG signature verification successful."
      else
          echo "❌ Error: GPG signature verification failed for '$mapfile.sig'."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 4
      fi
    else
      echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
    fi
  fi

  while IFS=: read -r fname hash tname _; do
    if [[ $fname =~ -([0-9]{5})-of-[0-9]{5}\.gguf$ ]]; then
      shard_id=$((10#${BASH_REMATCH[1]}))
      set_shard_id "$tname" "$shard_id"
      set_t_hash "$qtype" "$tname" "$hash"
      # Filling these lists should only happen once now
      if [[ "$qtype" == "${QTYPE}" ]]; then
        SHARD_FILENAMES+=("$fname")
        TENSORS_TO_FETCH+=("$tname")
      fi
    else
      echo "⚠️ Warning: skipping invalid filename '$fname'" >&2
    fi
  done < "$mappath"
done

# If the user explicitly requested a --qtype, create helpful symlinks in the model dir
# so callers that expect tensors.map / tensors.map.sig (generic names) can work.
# Only create these symlinks when not in --verify-readonly mode (we shouldn't write into
# the model dir when the user asked for readonly verification).
if [[ "$QTYPE_SPECIFIED" == true && "$VERIFY_READONLY" != true ]]; then
  src_map="$MAP_DIR/tensors.${QTYPE,,}.map"
  src_sig="$MAP_DIR/tensors.${QTYPE,,}.map.sig"
  if [[ -f "$src_map" && -f "$src_sig" ]]; then
    echo "[$(timestamp)] Ensure tensors.map and tensors.map.sig symlinks in model dir pointing to tensors.${QTYPE,,}.map and tensors.${QTYPE,,}.map.sig respectively"
    ln -sfn "$src_map" "$LOCAL_MODEL_DIR/tensors.map" || true
    ln -sfn "$src_sig" "$LOCAL_MODEL_DIR/tensors.map.sig" || true
  fi
fi

# ------------------ IMPLEMENT INDIVIDUAL TENSORS VALIDATION ----------------
# If the user specified --individual-tensors, parse and validate the list now that
# SHARD_FILENAMES (hence shard count) is known.
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
  num_shards=${#SHARD_FILENAMES[@]}
  num_shards=$((num_shards + 1))
  if (( num_shards == 1 )); then
    echo "❌ Error: cannot validate --individual-tensors because no shards were discovered in the maps." >&2
    exit 1
  fi

  # Parse comma-separated list into tokens
  IFS=',' read -r -a __ind_parts <<< "$INDIVIDUAL_TENSORS_RAW"

  if [[ ${#__ind_parts[@]} -eq 0 ]]; then
    echo "❌ Error: --individual-tensors provided but empty." >&2
    exit 1
  fi

  for tok in "${__ind_parts[@]}"; do
    # trim whitespace
    tok="${tok#"${tok%%[![:space:]]*}"}"
    tok="${tok%"${tok##*[![:space:]]}"}"
    if [[ -z "$tok" ]]; then
      echo "❌ Error: empty token in --individual-tensors list." >&2
      exit 1
    fi
    if ! [[ "$tok" =~ ^[0-9]+$ ]]; then
      echo "❌ Error: invalid non-numeric token in --individual-tensors: '$tok'." >&2
      exit 1
    fi
    # convert decimal safely (avoid leading zero octal)
    val=$((10#$tok))
    if (( val < 1 || val > num_shards )); then
      echo "❌ Error: --individual-tensors value $val out of range (must be between 1 and $num_shards)." >&2
      exit 1
    fi
    if [[ -n "${IND_TENSOR_SET[$val]:-}" ]]; then
      echo "❌ Error: duplicate tensor number in --individual-tensors: $val" >&2
      exit 1
    fi
    IND_TENSOR_SET[$val]=1
  done

  # For logging, prepare a sorted, comma-separated presentation
  sorted_list="$(printf "%s\n" "${!IND_TENSOR_SET[@]}" | sort -n | paste -sd, -)"
  echo "[$(timestamp)] --individual-tensors enabled: will only process tensors: $sorted_list"
fi
# --------------------------------------------------------------------------

# ------------------ SPECIAL NODE ASSIGNMENT (if requested) ----------------
# New approach: compute an xxh3-based digest over MODEL_NAME+chunkid and use that
# to decide if *this* runner should download that chunk. This avoids a fixed
# assigned_node and uses per-chunk hashing. The rule implemented:
#   should_process_chunk(chunk_id) -> true if (dec_from_xxhsum(MODEL_NAME+chunkid) % SPECIAL_NODE_ID) == 0
#
# The function below returns success (0) when the chunk should be downloaded by this node.
if [[ -n "$SPECIAL_NODE_ID" ]]; then
  # validate SPECIAL_NODE_ID
  if ! [[ "$SPECIAL_NODE_ID" =~ ^[0-9]+$ ]] || (( SPECIAL_NODE_ID < 0 )); then
    echo "❌ Error: --special-node-id must be a positive integer." >&2
    exit 1
  fi

  # try to read MODEL_NAME from $SCRIPT_DIR/download.conf
  CONFIG_FILE="$SCRIPT_DIR/download.conf"
  MODEL_NAME=""
  MAINTAINER=""
  if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
  fi

  if [[ -z "$MODEL_NAME" ]]; then
    echo "❌ Error: --special-node-id provided but MODEL_NAME could not be read from $DOWNLOAD_CONF" >&2
    exit 1
  fi

  # Ensure xxhsum is available
  if ! command -v xxhsum >/dev/null 2>&1; then
    echo "❌ Error: --special-node-id requires 'xxhsum' to be installed and on PATH." >&2
    echo "Hint: Install xxhash (xxhsum) or omit --special-node-id." >&2
    exit 1
  fi

  # should_process_chunk: returns 0 if this node should process the provided chunk_id
  # Uses: printf '%s' "${MODEL_NAME}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT${chunk_id}" | xxhsum -H3
  should_process_chunk() {
    local chunk_id="$1"
    chunk_id=$((10#$chunk_id))

    # Concatenate model name + chunk_id (no separator; deterministic)
    local input="${MODEL_NAME}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT${chunk_id}"

    # Example: echo abc | xxhsum -H3
    local out=$(printf '%s' "$input" | xxhsum -H3 2>/dev/null) || true

    # Extract a hex-like token from the output robustly.
    # Require at least 12 hex chars, then take the first 12.
    local hex=$(printf '%s' "$out" | grep -oE '[0-9a-fA-F]{12,}' | head -n1 || true)

    if [[ -z "$hex" ]]; then
      echo "[$(timestamp)] ⚠️ Warning: failed to parse xxhsum output for chunk_id=$chunk_id; output='$out'." >&2
      return 1
    fi

    # Use only the first 12 hex chars to avoid integer overflow in bash arithmetic
    hex="${hex:0:12}"

    # Convert to decimal and modulo
    if ! [[ "$hex" =~ ^[0-9a-fA-F]+$ ]]; then
      echo "[$(timestamp)] ⚠️ Warning: invalid hex digest ('$hex') for chunk_id=$chunk_id." >&2
      return 1
    fi

    local dec=$((16#$hex))
    local mod=$(( dec % TOTAL_NODES ))

    # We choose the rule: this node handles chunk if mod == SPECIAL_NODE_ID
    if (( mod == SPECIAL_NODE_ID )); then
      return 0
    else
      return 1
    fi
  }

  echo "[$(timestamp)] SPECIAL NODE MODE: MODEL_NAME='$MODEL_NAME' using xxhsum. Only chunks where xxhsum(MODEL_NAME+chunkid) % $TOTAL_NODES == $SPECIAL_NODE_ID will be downloaded/verified by this runner."
fi

# --------------- CONCURRENCY HELPERS ----------------
wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.5; done
}

# ------------- SHARD DOWNLOAD/VERIFY LOGIC --------------
download_shard() {
  local idx="$1"
  local tensor="${TENSORS_TO_FETCH[$idx]}"

  local chunk_id=$(get_shard_id "$tensor")

  local shard_file="${SHARD_FILENAMES[$idx]}"
  local local_gguf="$LOCAL_MODEL_DIR/$shard_file"
  local dl_gguf="$LOCAL_DOWNLOAD_DIR/$shard_file"
  local local_z="${local_gguf}.zbst"
  local dl_z="${dl_gguf}.zbst"

  # If individual-tensors list is enabled, skip shards not in set.
  if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
    if [[ -z "${IND_TENSOR_SET[$chunk_id]:-}" ]]; then
      echo "[$(timestamp)] Skipping tensor='$tensor' chunk_id=$chunk_id because it's not in --individual-tensors list."
      return 0
    fi
  fi

  # If special node assignment is enabled, skip shards not assigned to this node
  if [[ -n "$SPECIAL_NODE_ID" ]]; then
    if ! should_process_chunk "$chunk_id"; then
      _del=""
      [[ "$RM_SKIPPED_SHARDS" == true ]] && _del=" and deleting (if present)" && rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
      echo "[$(timestamp)] Skipping$_del tensor='$tensor' chunk_id=$chunk_id not assigned to this node (xxhsum-based selection)."
      return 0
    else
      echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id assigned to this node (xxhsum-based selection) — proceeding to HASH verification"
    fi
  else
    echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id HASH verification — proceeding"
  fi

  for i in "${!PATTERNS[@]}"; do
    local pat="${PATTERNS[$i]}"
    if [[ "$tensor" =~ $pat ]]; then
      local qtype="${PATTERN_QTYPES[$i]^^]}"
      local dl_type="$qtype"
      [[ "${qtype^^}" == "F32" ]] && dl_type="${QTYPE}"

      local shard_id=$(echo "$shard_file" | sed -E 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf/\1/')

      local got=""
      local need_download=false
      local failed_hash=0
      local skip_mv=true
      if [[ "$FORCE_REDOWNLOAD" == true ]]; then
          echo "[$(timestamp)] Force redownload: removing existing shard $shard_file"
          rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
          sync || true
          skip_mv=false
      fi
      local _path=""
      # If ARCHIVE_DECOMPRESS true: if a .zbst exists, decompress to .gguf (overwrite) so script works on .gguf
      if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
        if safe_file_exists "$local_z"; then
          echo "[$(timestamp)] z-decompress: found $local_z -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $local_z — treating as corrupted and will redownload." >&2
          fi
          rm -f "$local_z" || true
        elif safe_file_exists "$dl_z"; then
          echo "[$(timestamp)] z-decompress: found $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $dl_z in download dir — treating as corrupted and will redownload." >&2
          fi
          rm -f "$dl_z" || true
        fi
      fi

      while [[ "$need_download" == false ]] && [[ "$got" == "" ]]; do
        # 1) If ARCHIVE_COMPRESS or ARCHIVE_DECOMPRESS is enabled, prefer .gguf.zbst when present and validate from stream
        if safe_file_exists "$local_z" || safe_file_exists "$dl_z"; then
          if [[ "$ARCHIVE_COMPRESS" == true ]]; then
            if safe_file_exists "$local_z"; then
              _path="$local_z"
              skip_mv=true
            else
              _path="$dl_z"
              skip_mv=false
            fi

            if [[ "$SKIP_HASH" == true ]]; then
              # In skip-hash mode, pretend the stream hash matches expected
              got="$(get_t_hash "$qtype" "$tensor")"
            else
              if got="$(safe_stream_sha256_from_z "$_path")"; then
                got="${got%%[^0-9a-fA-F]*}"
              else
                got=""
              fi
            fi

            local exp=$(get_t_hash "$qtype" "$tensor")
            if [[ "$got" != "$exp" ]]; then
              if (( failed_hash > 9 )); then
                echo "[$(timestamp)] ❌ Too many hash verification attempt failures (count: $failed_hash) for '${_path}' (stream) - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                exit_from_subprocess 16
              fi
              failed_hash=$((failed_hash + 1))
              echo "[$(timestamp)] Will redownload due to hash mismatch (count: $failed_hash) for '${_path}' (stream) - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
              rm -f "$dl_z" "$local_z" || true
              sync || true
              need_download=true
            else
              echo "[$(timestamp)] Stream-hash OK for '${_path}' - tensor '$tensor' of qtype: '$qtype'"
              # If the valid zbst was in download dir, move to model dir
              if [[ "$skip_mv" == false ]]; then
                mv -f "$dl_z" "$LOCAL_MODEL_DIR/"
                echo "[$(timestamp)] Saved file id '$shard_id' (zbst) - tensor '$tensor' of qtype: '$qtype'"
              fi
              # Ensure we remove any corresponding .gguf if it exists (we keep only .gguf.zbst in compress mode)
              if [[ -f "$LOCAL_MODEL_DIR/$shard_file" ]]; then
                rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                echo "[$(timestamp)] Removed corresponding .gguf for '${shard_file}' because .gguf.zbst is present and valid"
              fi
              # nothing more to do for this shard
            fi
          elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
            if safe_file_exists "$local_z"; then
              echo "[$(timestamp)] z-decompress: found $local_z -> decompressing to $local_gguf (overwrite)"
              if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
                echo "[$(timestamp)] ⚠️ decompression failed for $local_z — treating as corrupted and will redownload." >&2
              fi
              rm -f "$local_z" || true
            elif safe_file_exists "$dl_z"; then
              echo "[$(timestamp)] z-decompress: found $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
              if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
                echo "[$(timestamp)] ⚠️ decompression failed for $dl_z in download dir — treating as corrupted and will redownload." >&2
              fi
              rm -f "$dl_z" || true
            fi
            # Proceed with .gguf present in $local_gguf, assuming decompression was successful
            skip_mv=true
            got=""
          else
            # No zbst found; fallthrough to check .gguf
            :
          fi
        fi

        # If we didn't find a valid zbst (or compress mode not enabled) or if decompress mode was enabled, check .gguf files
        if [[ -z "$got" ]]; then
          if safe_file_exists "$local_gguf" || safe_file_exists "$dl_gguf"; then
              if safe_file_exists "$local_gguf"; then
                  _path=$local_gguf
              else
                  _path=$dl_gguf
                  skip_mv=false
              fi
              if command -v _sha256sum &>/dev/null || [[ "$SKIP_HASH" == true ]]; then
                  [ "$failed_hash" -gt 0 ] && sync || true
                  # use safe_sha256sum which will retry if symlink
                  if [[ "$SKIP_HASH" == true ]]; then
                    got=$(get_t_hash "$qtype" "$tensor")
                    echo "[$(timestamp)] SKIP-HASH: treating ${_path} as valid (skipping sha256 computation)."
                  elif got="$(safe_sha256sum "$_path" 2>/dev/null)"; then
                    got="${got%%[^0-9a-fA-F]*}"
                  else
                    got=""
                  fi
                  local exp=$(get_t_hash "$qtype" "$tensor")
                  if [[ "$got" != "$exp" ]]; then
                      if (( failed_hash > 9 )); then
                        echo "[$(timestamp)] ❌ Too many hash verification attempt failures (count: $failed_hash) for '$shard_file' - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                        exit_from_subprocess 17
                      fi
                      failed_hash=$((failed_hash + 1))
                      echo "[$(timestamp)] Will redownload due to hash mismatch (count: $failed_hash) for '$shard_file' - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                      rm -f "$dl_gguf" "$local_gguf" || true
                      sync || true
                      need_download=true
                  else
                      if [[ "$SKIP_HASH" == true ]]; then
                        echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' processed!"
                      else
                        echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' hash is valid!"
                      fi
                      # If compression mode is active, compress and remove the .gguf (only keep .gguf.zbst)
                      if [[ "$ARCHIVE_COMPRESS" == true ]]; then
                        # compress the validated file in-place (in whichever location it currently resides)
                        if [[ "$skip_mv" == true ]]; then
                          echo "[$(timestamp)] z-compress validated ${_path} -> ${_path}.zbst"
                          compress_gguf_to_archive "$_path"
                          # After compressing, ensure any lingering .gguf (could be symlinked elsewhere) is removed
                          rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                        else
                          # file is in download dir; compress there and move .zbst to model dir
                          echo "[$(timestamp)] z-compress validated ${_path} in download dir"
                          compress_gguf_to_archive "$_path"
                          mv -f "${_path}.zbst" "$LOCAL_MODEL_DIR/"
                          echo "[$(timestamp)] Saved file id '$shard_id' (zbst) - tensor '$tensor' of qtype: '$qtype'"
                          # remove any .gguf counterpart in model dir (just in case)
                          rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                        fi
                      else
                        if [[ "$skip_mv" == false ]]; then
                          mv -f "$dl_gguf" "$LOCAL_MODEL_DIR/"
                          echo "[$(timestamp)] Saved file id '$shard_id' - tensor '$tensor' of qtype: '$qtype'"
                        fi
                      fi
                  fi
              else
                  echo "⚠️ Warning: _sha256sum command missing - hash cannot be verified!"
              fi
          else
              need_download=true
          fi
        fi

        if [[ "$need_download" == true ]]; then
            echo "[$(timestamp)] Downloading file id '$shard_id' - tensor '$tensor' of qtype: '$qtype' (chunk_id=$chunk_id)"
            until run_downloader_shard "$dl_type" "$chunk_id" "$LOCAL_DOWNLOAD_DIR" "$shard_file"; do
                echo "[$(timestamp)] Download failed; retrying in 10s..."
                sleep 10
            done
            need_download=false
            skip_mv=false
            got=""
        else
          # If no download was needed and we have already handled saving/compressing above, nothing to do.
          :
        fi
      done

      break
    fi
  done
}

# Prepare failure marker
# (FAIL_MARKER already set earlier depending on verify-readonly / normal mode)
rm -f "$FAIL_MARKER" 2>/dev/null || true

# ------------------ VERIFY-ONLY MODE -------------------
if [[ "$VERIFY" == true ]]; then
  echo "[$(timestamp)] VERIFY: verifying existing shards (readonly mode: $VERIFY_READONLY)"

  # helper to cap concurrent jobs
  wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.1; done
  }

  SCRIPT_PID=$$

  # Warn about mismatched file types depending on flags:
  # - If -z is used with --verify: verify only .gguf.zbst; warn if any .gguf present (they will be ignored)
  # - If neither -z nor -zd used with --verify: verify only .gguf; warn if any .gguf.zbst present (they will be ignored)
  if [[ "$ARCHIVE_COMPRESS" == true ]]; then
    # Verify only .gguf.zbst
    if comp=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*-*-of-*.gguf" -print -quit 2>/dev/null || true); then
      if [[ -n "$comp" ]]; then
        echo "⚠️ Warning: found .gguf files in model dir while --verify + -z is used. These will be ignored; verifying only .gguf.zbst files." >&2
      fi
    fi
  else
    # Verify only .gguf
    if comp=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*-*-of-*.gguf.zbst" -print -quit 2>/dev/null || true); then
      if [[ -n "$comp" ]]; then
        echo "⚠️ Warning: found .gguf.zbst files in model dir while --verify without -z. These will be ignored; verifying only .gguf files." >&2
      fi
    fi
  fi

  # 1) check first shard explicitly, in background
  wait_for_slot
  (
    # Build files list robustly using shell globbing (nullglob) so we don't depend on find -printf portability.
    shopt -s nullglob 2>/dev/null || true
    files=()
    if [[ "$ARCHIVE_COMPRESS" == true ]]; then
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    else
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    fi
    shopt -u nullglob 2>/dev/null || true

    IFS= read -r first <<< "$(printf '%s\n' "${files[@]:-}" | head -n1 || true)"
    if [[ -n "$first" && "$first" =~ -([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
      total="${BASH_REMATCH[2]}"
    else
      total=""
    fi

    gguf_first=""
    # Determine whether we should perform first-shard GPG download/verification under special-node-mode (does it by default for BF16 models)
    should_verify_first=true
    if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
      if ! should_process_chunk 1; then
        should_verify_first=false
      fi
    fi
    # If individual-tensors is enabled and chunk 1 is not in the list, skip first-shard verification
    if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
      if [[ -z "${IND_TENSOR_SET[1]:-}" ]]; then
        should_verify_first=false
      fi
    fi

    if [[ "$should_verify_first" == true ]]; then
      if [[ -n "$first" ]]; then
        gguf_first=$(basename "$(printf '%s\n' "$first" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
        local_first="$LOCAL_MODEL_DIR/$gguf_first"
        local_first_z="${local_first}.zbst"

        # Enforce expectations: when --verify + -z, first shard MUST be .gguf.zbst (no auto-convert/compress)
        if [[ "$ARCHIVE_COMPRESS" == true ]]; then
          if [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" && ! -L "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]; then
            echo "[$(timestamp)] ❌ VERIFY: Expected first shard '${gguf_first}.zbst' not found; when using --verify with -z the first shard must be .gguf.zbst." >&2
            kill -s TERM "$SCRIPT_PID"  # Kill parent
            exit 5
          fi
          # Verify signature using a temporary decompressed file (gpg requires a file)
          if [[ "$SKIP_GPG" != true ]]; then
            if command -v gpg >/dev/null 2>&1; then
              # create temp file in writable temp workspace (respect verify-readonly)
              if [[ "$VERIFY_READONLY" == true ]]; then
                tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
              else
                tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
              fi
              if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
                echo "[$(timestamp)] ❌ VERIFY: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption). Maybe you need to specify --z-decompress-opt? Treating as verification failure." >&2
                rm -f "$tmpf"
                kill -s TERM "$SCRIPT_PID"
                touch "$FAIL_MARKER"
                exit 1
              fi
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ❌ VERIFY: Error - Signature file '$gguf_first.sig' is missing." >&2
                kill -s TERM "$SCRIPT_PID"
                rm -f "$tmpf"
                exit 5
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
                echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful (via temp decompressed file)."
              else
                echo "[$(timestamp)] ❌ VERIFY: Error - GPG signature verification failed for '$gguf_first.sig'." >&2
                rm -f "$tmpf"
                kill -s TERM "$SCRIPT_PID"
                exit 4
              fi
              rm -f "$tmpf"
            else
              echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
            fi
          fi
          echo "[$(timestamp)] OK: ${gguf_first}.zbst"
        else
          # --verify without -z: expect .gguf present
          if [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first" && ! -L "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
            echo "[$(timestamp)] ❌ VERIFY: Expected first shard '$gguf_first' not found; when using --verify without -z the first shard must be .gguf." >&2
            kill -s TERM "$SCRIPT_PID"  # Kill parent
            exit 5
          fi
          if [[ "$SKIP_GPG" != true ]]; then
            if command -v gpg >/dev/null 2>&1; then
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ❌ VERIFY: Error - Signature file '$gguf_first.sig' is missing." >&2
                kill -s TERM "$SCRIPT_PID"
                exit 5
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
                echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
              else
                echo "[$(timestamp)] ❌ VERIFY: Error - GPG signature verification failed for '$gguf_first.sig'." >&2
                kill -s TERM "$SCRIPT_PID"
                exit 4
              fi
            else
              echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
            fi
          fi
          echo "[$(timestamp)] OK: $gguf_first"
        fi
      else
        echo "[$(timestamp)] MISSING: $gguf_first"
        touch "$FAIL_MARKER"
      fi
    else
      echo "[$(timestamp)] Skipping first-shard GPG verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
    fi
  ) &

  # 2) check each remaining shard, in parallel
  for idx in "${!TENSORS_TO_FETCH[@]}"; do
    wait_for_slot
    (
      tensor="${TENSORS_TO_FETCH[$idx]}"
      #echo "[$(timestamp)] Checking HASH for tensor='$tensor'"

      chunk_id=$(get_shard_id "$tensor")

      # If individual-tensors is enabled, skip shards not listed
      proceed=true
      if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
        if [[ -z "${IND_TENSOR_SET[$chunk_id]:-}" ]]; then
          proceed=false
        fi
      fi

      # If special node assignment is enabled, skip shards not assigned to this node
      if [[ "$proceed" == true && -n "$SPECIAL_NODE_ID" ]]; then
        if ! should_process_chunk "$chunk_id"; then
          #echo "[$(timestamp)] Skipping HASH of tensor='$tensor' chunk_id=$chunk_id not assigned to this node (xxhsum-based selection)."
          proceed=false
        else
          echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id assigned to this node (xxhsum-based selection) — proceeding to HASH"
        fi
      fi
      if [[ "$proceed" == true ]]; then
        for i in "${!PATTERNS[@]}"; do
          pat="${PATTERNS[$i]}"
          if [[ "$tensor" =~ $pat ]]; then
            qtype="${PATTERN_QTYPES[$i]^^]}"
            shardfile="${SHARD_FILENAMES[$idx]}"
            local_gguf="$LOCAL_MODEL_DIR/$shardfile"
            local_z="${local_gguf}.zbst"

            if [[ "$ARCHIVE_COMPRESS" == true ]]; then
              # In verify-only + -z mode: verify .gguf.zbst only (do not alter .gguf files).
              if safe_file_exists "$local_z" && ([[ "$SKIP_HASH" == true ]] || command -v _sha256sum &>/dev/null); then
                if [[ "$SKIP_HASH" == true ]]; then
                  # skip computing hash -> treat as OK
                  echo "[$(timestamp)] SKIP-HASH: treating ${shardfile}.zbst as valid (skipping stream sha256)."
                else
                  got=$(safe_stream_sha256_from_z "$local_z" || true)
                  got="${got%%[^0-9a-fA-F]*}"
                  exp=$(get_t_hash "$qtype" "$tensor")
                  if [[ "$got" != "$exp" ]]; then
                    echo "[$(timestamp)] WRONG HASH (stream): ${shardfile}.zbst ($got != $exp) - tensor: '$tensor' - qtype: '$qtype'"
                    touch "$FAIL_MARKER"
                  else
                    echo "[$(timestamp)] HASH OK (stream): ${shardfile}.zbst"
                  fi
                fi
              elif safe_file_exists "$local_gguf" && [[ "$SKIP_HASH" != true ]] && command -v _sha256sum &>/dev/null; then
                # Found .gguf while user requested -z verify-only: warn & treat as missing .zbst (do NOT compress/modify)
                echo "[$(timestamp)] WARNING: found ${shardfile} (.gguf) but --verify + -z expects ${shardfile}.zbst; treating as MISSING for verification purposes."
                touch "$FAIL_MARKER"
              elif [[ "$SKIP_HASH" == true ]] && safe_file_exists "$local_gguf"; then
                # If skip-hash and .gguf present but .zbst missing, we treat it as missing .zbst for verification semantics.
                echo "[$(timestamp)] WARNING: found ${shardfile} (.gguf) but --verify + -z expects ${shardfile}.zbst; treating as MISSING for verification purposes."
                touch "$FAIL_MARKER"
              else
                echo "[$(timestamp)] MISSING: ${shardfile}.zbst"
                touch "$FAIL_MARKER"
              fi
            else
              # normal verify-only (no -z): verify .gguf only (do not alter .zbst files)
              if safe_file_exists "$local_gguf" && ([[ "$SKIP_HASH" == true ]] || command -v _sha256sum &>/dev/null); then
                if [[ "$SKIP_HASH" == true ]]; then
                  echo "[$(timestamp)] SKIP-HASH: treating ${shardfile} as valid (skipping sha256)."
                else
                  got=$(safe_sha256sum "$local_gguf" 2>/dev/null || true)
                  got="${got%%[^0-9a-fA-F]*}"
                  exp=$(get_t_hash "$qtype" "$tensor")
                  if [[ "$got" != "$exp" ]]; then
                    echo "[$(timestamp)] WRONG HASH: $shardfile ($got != $exp) - tensor: '$tensor' - qtype: '$qtype'"
                    touch "$FAIL_MARKER"
                  else
                    echo "[$(timestamp)] HASH OK: $shardfile"
                  fi
                fi
              elif safe_file_exists "$local_z" && [[ "$SKIP_HASH" != true ]] && command -v _sha256sum &>/dev/null; then
                # Found .gguf.zbst while user requested non -z verify-only: warn & treat as missing .gguf
                echo "[$(timestamp)] WARNING: found ${shardfile}.zbst but --verify without -z expects ${shardfile} (.gguf); treating as MISSING for verification purposes."
                touch "$FAIL_MARKER"
              elif [[ "$SKIP_HASH" == true ]] && safe_file_exists "$local_z"; then
                # If skip-hash and only .zbst present while non -z verify requested: treat as missing .gguf
                echo "[$(timestamp)] WARNING: found ${shardfile}.zbst but --verify without -z expects ${shardfile} (.gguf); treating as MISSING for verification purposes."
                touch "$FAIL_MARKER"
              else
                echo "[$(timestamp)] MISSING: $shardfile"
                touch "$FAIL_MARKER"
              fi
            fi
            break
          fi
        done
      fi
    ) &
  done

  # wait for all verifications to finish
  wait

  # Cleanup
  if [[ "$SKIP_GPG" != true ]]; then
    if command -v gpg >/dev/null 2>&1; then
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
    fi
  fi
  # Remove qtype marker files
  if [[ "$VERIFY_READONLY" != true ]]; then
    rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || true
  fi

  # summary
  if [[ -f "$FAIL_MARKER" ]]; then
    if [[ "$SKIP_HASH" == true ]]; then
      echo "[$(timestamp)] ❌ VERIFY: some files missing"
    else
      echo "[$(timestamp)] ❌ VERIFY: some files missing or with hash mismatch"
    fi
    exit 1
  else
    if [[ "$SKIP_HASH" == true ]]; then
      echo "[$(timestamp)] ✅ VERIFY: all files present"
    else
      echo "[$(timestamp)] ✅ VERIFY: all files present and with valid hashes"
    fi
    rm -f "$FAIL_MARKER" 2>/dev/null || true
    exit 0
  fi
fi

# ------------------ MAIN DOWNLOAD LOOP (retry-until-success wrappers, no PID bookkeeping) -------------------

for idx in "${!TENSORS_TO_FETCH[@]}"; do
  wait_for_slot

  # compute tensor & chunk_id in parent so we can record them for the subshell
  tensor="${TENSORS_TO_FETCH[$idx]:-}"
  chunk_id="$(get_shard_id "$tensor")"

  # capture loop variables for the subshell to avoid race / duplication
  current_idx="$idx"
  current_tensor="$tensor"
  current_chunk="$chunk_id"

  (
    attempts=0
    # Each wrapper will keep trying until download_shard returns success (0)
    while true; do
      attempts=$((attempts + 1))
      if download_shard "$current_idx"; then
        # succeeded
        DEBUG "child idx=$current_idx download_shard succeeded (attempt $attempts)"
        break
      else
        rc=$?
        # use the parent-computed tensor/chunk_id (visible inside subshell)
        echo "[$(timestamp)] ⚠️ Warning: Tensor='${current_tensor}' chunk_id=${current_chunk} download failed with exit $rc (attempt $attempts). Retrying in 10s..." >&2
        sleep 10
      fi
    done
  ) &

  # Optionally show the child's PID in debug, but we do NOT save it anywhere.
  pid=$!
  DEBUG "spawned wrapper idx=$idx pid=$pid"
done

# Wait for all background wrapper jobs to finish
rc=0
set +e
# Single wait with no args waits for all child processes started in this shell.
# Its exit status is that of the last process waited for; treat any non-zero as error.
if ! wait; then
  status=$?
  echo "❌ Error: one or more download wrapper children exited with non-zero status (wait returned $status). Check child stderr logs for details (each child prints its tensor and chunk on retry/error)." >&2
  rc=1
fi
set -e

if (( rc != 0 )); then
  echo "❌ Error: one or more download wrappers exited abnormally." >&2
  exit $rc
fi

# ------------------ END MAIN DOWNLOAD LOOP (retry-until-success wrappers, no PID bookkeeping) -------------------

# ------------- FINAL FIRST-SHARD FETCH (non-verify) -----
# Decide which files to consider for first-shard detection depending on z mode
# Use shell globbing (nullglob) for portability and predictable behavior.
shopt -s nullglob 2>/dev/null || true
if [[ "$ARCHIVE_COMPRESS" == true ]]; then
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  # prefer already-decompressed .gguf if present, otherwise accept .gguf.zbst
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
else
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
fi
shopt -u nullglob 2>/dev/null || true

IFS= read -r first <<< "$(printf '%s\n' "${files[@]:-}" | head -n1 || true)"
if [[ -n "$first" && "$first" =~ -([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
  total="${BASH_REMATCH[2]}"
  gguf_first=$(basename "$(printf '%s\n' "$first" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
else
  # Attempt to build file name from .map file instead
  num_shards=${#SHARD_FILENAMES[@]}
  total="+$(printf "%05d" "$((num_shards + 1))")"
  gguf_first=$(basename "$(printf '%s\n' "${SHARD_FILENAMES[0]}" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
fi

# Determine whether we should perform first-shard GPG download/verification under special-node-mode (does it by default for BF16 models)
should_verify_first=true
_del=""
if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
  if ! should_process_chunk 1; then
    [[ "$RM_SKIPPED_SHARDS" == true ]] && _del=" and deleting (if present)" && rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first" "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig" || true
    should_verify_first=false
  fi
fi

# If individual-tensors is enabled and 1 isn't included, skip verifying/downloading first shard here
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true && -z "${IND_TENSOR_SET[1]:-}" ]]; then
  should_verify_first=false
fi

# Helper: attempt to redownload the first shard and its signature (unless verify-readonly).
# Returns 0 on success (downloaded & moved into LOCAL_MODEL_DIR), non-zero otherwise.
attempt_redownload_first() {
  if [[ "$VERIFY_READONLY" == true ]]; then
    return 1
  fi

  echo "[$(timestamp)] First shard appears corrupted or invalid — attempting to redownload first shard (and signature if GPG verification enabled)."

  # remove any possibly-broken files (both model dir and download dir)
  rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_MODEL_DIR/$gguf_first.sig \
        "$LOCAL_DOWNLOAD_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first.zbst" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig || true

  # Ensure download dir exists for redownload attempts
  mkdir -p "$LOCAL_DOWNLOAD_DIR" 2>/dev/null || true

  # download shard (keep retrying until success)
  until run_downloader_shard "${QTYPE}" 1 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first")"; do
    echo "[$(timestamp)] First shard download failed; retrying in 10s..." >&2
    sleep 10
  done

  # Move whichever artifact was produced (.gguf OR .gguf.zbst)
  if [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" ]]; then
    mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" "$LOCAL_MODEL_DIR/" || true
  elif [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" ]]; then
    mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" "$LOCAL_MODEL_DIR/" || true
  else
    echo "[$(timestamp)] ❌ Error: expected first shard in download dir after redownload but none found." >&2
    return 1
  fi

  # download signature if required
  if [[ "$SKIP_GPG" != true ]]; then
    until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
      echo "[$(timestamp)] First shard signature download failed; retrying in 10s..." >&2
      sleep 10
    done

    # Move signature artifact that was procured (.gguf.sig)
    if [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" ]]; then
      mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/" || true
    else
      echo "[$(timestamp)] ❌ Error: expected first shard signature in download dir after redownload but none found." >&2
      return 1
    fi
  else
    echo "[$(timestamp)] Redownload of first shard completed."
  fi

  return 0
}

if [[ "$should_verify_first" == true ]]; then
  if [[ "$VERIFY" != true ]]; then
    echo "[$(timestamp)] Fetching first shard separately"
    if [[ "$total" != "" ]] && [[ "$gguf_first" != "" ]]; then
      if [[ "$FORCE_REDOWNLOAD" == true ]]; then
        echo "[$(timestamp)] Force redownload: removing existing first shard (and gpg signature)"
        rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first" || true
        rm -f "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig" || true
        rm -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_DOWNLOAD_DIR/$gguf_first.zbst" || true
        sync || true
      fi
      if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first" ] && ! [ -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]; then
        until run_downloader_shard "${QTYPE}" 1 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first")"; do
          echo "[$(timestamp)] First shard download failed; retrying in 10s..."
          sleep 10
        done

        # Move whichever file was produced (.gguf or .gguf.zbst)
        if [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" ]]; then
          mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" "$LOCAL_MODEL_DIR/"
        elif [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" ]]; then
          mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" "$LOCAL_MODEL_DIR/"
        else
          echo "[$(timestamp)] ❌ Error: expected first shard in download dir but none found after download." >&2
          touch "$FAIL_MARKER"
          exit 1
        fi

        echo "[$(timestamp)] First shard saved"
        if [[ "$SKIP_GPG" != true ]]; then
          until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
            echo "[$(timestamp)] First shard signature download failed; retrying in 10s..."
            sleep 10
          done

          # Move file that was procurred .gguf.sig
          if [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" ]]; then
            mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/"
          else
            echo "[$(timestamp)] ❌ Error: expected first shard signature in download dir but none found after download." >&2
            touch "$FAIL_MARKER"
            exit 1
          fi

          echo "[$(timestamp)] First shard gpg signature saved"
        fi
      else
        echo "[$(timestamp)] First shard already exists"
        if [[ "$SKIP_GPG" != true ]]; then
          if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
            until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
              echo "[$(timestamp)] First shard gpg signature download failed; retrying in 10s..."
              sleep 10
            done
            mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/"
            echo "[$(timestamp)] First shard gpg signature saved"
          else
            echo "[$(timestamp)] First shard gpg signature already exists"
          fi
        fi
      fi

      # After ensuring first shard and signature present, perform GPG verification if required.
      if [[ "$SKIP_GPG" != true ]]; then
        if command -v gpg >/dev/null 2>&1; then
          # We'll attempt verification and on corruption/verify failure (when not verify-readonly) we will redownload and retry.
          # Prefer verifying the actual .gguf file if present (this avoids trying to decompress a missing/non-updated .gguf.zbst).
          if [[ -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
            # verify using the .gguf that we just ensured is present
            while :; do
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ⚠️ Signature file '$gguf_first.sig' is missing — treating as corrupted and will redownload." >&2
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: Signature file '$gguf_first.sig' is missing." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 5
                fi
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
                echo "[$(timestamp)] First shard gpg signature verification successful."
                break
              else
                echo "[$(timestamp)] ⚠️ GPG signature verification failed for '$gguf_first.sig' — treating as corrupted and will redownload." >&2
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 4
                fi
              fi
            done
          elif [[ -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]; then
            # If only a .zbst exists, decompress to temp (or to local model directory if -zd) and verify.
            while :; do
              if [[ "$VERIFY_READONLY" == true ]]; then
                tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
              elif [[ "$ARCHIVE_DECOMPRESS" == true ]] || ([[ "$ARCHIVE_COMPRESS" == true ]] && [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]); then
                tmpf="$LOCAL_MODEL_DIR/$gguf_first"
              else
                tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
              fi

              if [[ -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]] && ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
                echo "[$(timestamp)] ⚠️ decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption) — treating as corrupted and will redownload." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  # redownloaded, retry loop
                  continue
                else
                  # cannot redownload (readonly) -> mark failure and exit
                  echo "❌ Error: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption)." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 1
                fi
              fi

              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ⚠️ Signature file '$gguf_first.sig' is missing — treating as corrupted and will redownload." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: Signature file '$gguf_first.sig' is missing." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 5
                fi
              fi

              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
                if [[ "$tmpf" != "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
                  echo "[$(timestamp)] First shard gpg signature verification succeeded (via temp decompressed file)."
                  rm -f "$tmpf"
                else
                  echo "[$(timestamp)] First shard gpg signature verification succeeded."
                fi
                # If both .gguf and .gguf.zbst existed, remove .gguf to keep only .gguf.zbst in compress mode
                if [[ "$ARCHIVE_COMPRESS" == true && -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
                  if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]; then
                    echo "[$(timestamp)] z-compress validated \"$LOCAL_MODEL_DIR/$gguf_first\""
                    compress_gguf_to_archive "$LOCAL_MODEL_DIR/$gguf_first"
                  else
                    rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
                    echo "[$(timestamp)] Removed corresponding .gguf for '$gguf_first' because .gguf.zbst is present and valid"
                  fi
                fi
                break
              else
                echo "[$(timestamp)] ⚠️ First shard gpg signature verification failed for '$gguf_first.sig'." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 4
                fi
              fi
            done
          else
            echo "❌ Error: unable to find previous shards..." >&2
            touch "$FAIL_MARKER"
          fi
        else
          echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
        fi
      fi

      # After verification, if compression is enabled, ensure only .zbst remains (compress if necessary)
      if [[ "$ARCHIVE_COMPRESS" == true ]]; then
        if [[ -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
          echo "[$(timestamp)] z-compress verified first shard to .zbst"
          compress_gguf_to_archive "$LOCAL_MODEL_DIR/$gguf_first"
        fi
      fi

    else
      echo "❌ Error: unable to find previous corresponding map or shards..." >&2
      touch "$FAIL_MARKER"
    fi
  fi
else
  echo "[$(timestamp)] Skipping$_del first-shard signature download/verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
fi

# ------------- FINAL VERIFICATION & SHARD SEQUENCE --------
# If individual-tensors mode is enabled the user requested skipping full sequence verification.
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
  echo "[$(timestamp)] Skipping full shard sequence verification because --individual-tensors was provided."
else
  echo "[$(timestamp)] Verifying shard sequence completeness"
  # Build full_indices from the files list we assembled above (respecting z modes)
  full_indices=()
  if [[ "$ARCHIVE_COMPRESS" == true ]]; then
    # only consider .gguf.zbst for completeness
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    shopt -u nullglob 2>/dev/null || true
  elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
    # if z-decompress, prefer .gguf files (we expect decompressed .gguf). If .gguf absent, consider .gguf.zbst
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    if [[ ${#files[@]} -eq 0 ]]; then
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    fi
    shopt -u nullglob 2>/dev/null || true
  else
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    shopt -u nullglob 2>/dev/null || true
  fi

  for f in "${files[@]}"; do
    base=$(basename "$f")
    if [[ "$base" =~ -([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
      full_indices+=( "${BASH_REMATCH[1]}" )
      # capture total if not already set
      if [[ -z "${total:-}" ]]; then
        total="${BASH_REMATCH[2]}"
      fi
    fi
  done

  # sort the indices
  if [[ ${#full_indices[@]} -gt 0 ]]; then
    IFS=$'\n' full_indices=($(printf "%s\n" "${full_indices[@]}" | sort))
  else
    echo "❌ Error: no shard files found to verify." >&2
    touch "$FAIL_MARKER"
    exit 1
  fi

  # Convert first/last from the full list (zero-padded strings)
  full_first_z="${full_indices[0]}"
  full_last_z="${full_indices[-1]}"
  full_first=$((10#$full_first_z))
  full_last=$((10#$full_last_z))

  if [[ -z "$SPECIAL_NODE_ID" ]]; then
    # unchanged legacy behavior when not using special-node mode:
    last_index=${full_last_z}
    first_index=${full_first_z}
    count_expected=$((10#$last_index - 10#$first_index + 1))

    if [[ ${#full_indices[@]} -ne $count_expected ]]; then
      echo "❌ Error - $((count_expected - ${#full_indices[@]})) missing shard(s) between $first_index and $last_index. Verify recipe or rerun." >&2
      echo "Missing indices:"
      seq -f "%05g" "$full_first" "$full_last" \
        | grep -Fvx -f <(printf "%s\n" "${full_indices[@]}")
      touch "$FAIL_MARKER"
      exit 1
    fi

    echo "[$(timestamp)] All shards from ${first_index} to ${last_index} are present."
  else
    # SPECIAL_NODE_ID mode: compute which indices this node SHOULD process
    expected_for_node=()
    # iterate the *full* range and test each index with should_process_chunk
    while IFS= read -r idx_z; do
      # convert to integer for should_process_chunk
      if should_process_chunk "$idx_z"; then
        expected_for_node+=("$idx_z")
      fi
    done < <(seq -f "%05g" "$full_first" "$full_last")

    # Build actual present-for-node list by filtering full_indices with should_process_chunk
    present_for_node=()
    for idx_z in "${full_indices[@]}"; do
      if should_process_chunk "$idx_z"; then
        present_for_node+=("$idx_z")
      fi
    done

    count_expected=${#expected_for_node[@]}
    count_present=${#present_for_node[@]}

    if (( count_present != count_expected )); then
      # find missing expected entries (expected - present)
      echo "❌ Error - $((count_expected - count_present)) missing shard(s) assigned to this node between ${full_first_z} and ${full_last_z}. Verify recipe or rerun." >&2
      echo "Missing indices (zero-padded):"
      # print expected list and subtract present list
      printf "%s\n" "${expected_for_node[@]}" | grep -Fvx -f <(printf "%s\n" "${present_for_node[@]}")
      touch "$FAIL_MARKER"
      exit 1
    fi

    # For logging, show the node-specific first/last
    first_index="${expected_for_node[0]}"
    last_index="${expected_for_node[-1]}"
    echo "[$(timestamp)] All assigned shards for this node present: ${first_index} .. ${last_index} (count=${count_present})"
  fi
fi

if [[ "$SKIP_GPG" != true ]]; then
  # Decide whether to run final GPG verification for first-shard
  should_verify_first=true
  if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
    if ! should_process_chunk 1; then
      should_verify_first=false
    fi
  fi
  if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true && -z "${IND_TENSOR_SET[1]:-}" ]]; then
    should_verify_first=false
  fi
  if [[ "$should_verify_first" == true ]]; then
    if command -v gpg >/dev/null 2>&1; then
      if ([[ "$ARCHIVE_COMPRESS" == false ]] && [ ! -f "$LOCAL_MODEL_DIR/$gguf_first" ]) || ([[ "$ARCHIVE_COMPRESS" == true ]] && [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]); then
          echo "❌ Error: Shard file '$gguf_first' not found."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 5
      fi
      if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
          echo "❌ Error: Signature file '$gguf_first.sig' is missing."
          echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 5
      fi

      # If .zbst exists and we're in compress mode, decompress to a temp file for gpg verify
      if [[ -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]; then
        # create temp file in writable temp workspace (respect verify-readonly)
        if [[ "$VERIFY_READONLY" == true ]]; then
          tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
        else
          tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
        fi
        if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
          echo "❌ Error: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption)." >&2
          rm -f "$tmpf"
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 1
        fi
        if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful (via temp decompressed file)."
            # If both .gguf and .gguf.zbst existed, remove .gguf to keep only .gguf.zbst in compress mode
            if [[ "$ARCHIVE_COMPRESS" == true && -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
              rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
              echo "[$(timestamp)] Removed corresponding .gguf for '$gguf_first' because .gguf.zbst is present and valid"
            fi
        else
            echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'."
            rm -f "$tmpf"
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            touch "$FAIL_MARKER"
            exit 4
        fi
        rm -f "$tmpf"
        # If z-decompress requested, convert to regular .gguf permanently
        if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
          if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_MODEL_DIR/$gguf_first"; then
            echo "❌ Error: decompression failed while converting '$LOCAL_MODEL_DIR/$gguf_first.zbst' -> '$LOCAL_MODEL_DIR/$gguf_first' (data corruption)." >&2
            rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
            touch "$FAIL_MARKER"
            exit 1
          fi
          rm -f "$LOCAL_MODEL_DIR/$gguf_first.zbst"
          echo "[$(timestamp)] z-decompress: converted first shard .zbst -> .gguf"
        fi
      else
        if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
        else
            echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'."
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            touch "$FAIL_MARKER"
            exit 4
        fi
      fi
    else
      echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
    fi
  else
    echo "[$(timestamp)] Skipping final first-shard GPG verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
  fi
  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
fi

# Remove qtype marker files
rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || True

echo
rm -f "$FAIL_MARKER" 2>/dev/null || true
echo "✅ Download and verification complete. Enjoy!"
