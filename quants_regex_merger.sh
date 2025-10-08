#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quants_regex_merger.sh is a basic tool to combine tensor  **#
#** regex for llama-quantize consumption.                     **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Oct-05-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.           ₚᵣₒₘₚₜₑ𝒹 ₜₒ ₘᵢₛᵦₑₕₐᵥₑ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail
shopt -s lastpipe

# Toggle debug by exporting DEBUG=1
_debug() {
  [[ "${DEBUG:-0}" -ne 1 ]] && return
  printf '[DEBUG] %s\n' "$*" >&2
}

# Initialize array before any use
declare -a OUTPUTS=()

# Default behavior: create file
NO_FILE=0
MODEL_NAME=""
MODEL_LINK=""
PPL=""

raw_ppl=""
# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --no-file)
      NO_FILE=1
      shift
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --model-link)
      MODEL_LINK="$2"
      shift 2
      ;;
    --add-ppl)
      raw_ppl="$2"
      # Format PPL to exactly 4 decimal places (e.g. 2 → 2.0000, 2.6 → 2.6000)
      PPL=$(printf "%.4f" "$2")
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--no-file] [--model-name NAME] [--add-ppl VALUE]"
      echo
      echo "  --no-file         Do not write output to a file; just print."
      echo "  --model-name NAME Optional. Prepends NAME to the output filename."
      echo "  --add-ppl VALUE   Optional. Adds VALUE_PPL right after username in the filename."
      echo
      echo "Example output filename:"
      echo "  MODEL.USER.PPL_PPL.TOTALGB_GGUF-GPUGB_GPU-CPUGB_CPU.HASH1-HASH2.recipe"
      echo "  DeepSeek-R1-0528.THIREUS.2.6348_PPL.242GB-GGUF-11GB_GPU-231GB_CPU.3bf94a4-94e09c4.recipe"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-file] [--model-name NAME] [--add-ppl VALUE]"
      exit 1
      ;;
  esac
done

if [[ "$raw_ppl" != "" ]] && ! [[ "$raw_ppl" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  echo "Error: --add-ppl value must be numeric" >&2
  exit 1
fi

# Override echo to capture output
echo() {
  local msg="$*"
  OUTPUTS+=("$msg")
  builtin echo "$msg"
}

# Output of: quant_assign.py ppl_results_guessed.csv --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight' --gpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq6_k iq5_k_r4
# But also: cat DeepSeek-V3-0324-THIREUS-BF16-SPECIAL_SPLIT/tensors.map | cut -d: -f 3,5 | sed 's/:dtype//g' | sed 's/\./\\./g'
# If someone pipes into us, read all of stdin into $custom…
if [ ! -t 0 ]; then
  _debug "Reading custom from stdin"
  custom="$(cat)"
else
  # …otherwise use the built‑in default
  custom="
^v\.blk\.0\.attn_out\.bias$=f32
^v\.blk\.0\.attn_out\.weight$=bf16
^v\.blk\.0\.attn_q\.bias$=f32
^v\.blk\.0\.attn_k\.bias$=f32
^v\.blk\.0\.attn_v\.bias$=f32
^v\.blk\.0\.attn_q\.weight$=bf16
^v\.blk\.0\.attn_k\.weight$=bf16
^v\.blk\.0\.attn_v\.weight$=bf16
^v\.blk\.0\.ffn_up\.bias$=f32
^v\.blk\.0\.ffn_up\.weight$=bf16
^v\.blk\.0\.ffn_down\.bias$=f32
^v\.blk\.0\.ffn_down\.weight$=bf16
^v\.blk\.0\.ln1\.bias$=f32
^v\.blk\.0\.ln1\.weight$=f32
^v\.blk\.0\.ln2\.bias$=f32
^v\.blk\.0\.ln2\.weight$=f32
^v\.blk\.1\.attn_out\.bias$=f32
^v\.blk\.1\.attn_out\.weight$=bf16
^v\.blk\.1\.attn_q\.bias$=f32
^v\.blk\.1\.attn_k\.bias$=f32
^v\.blk\.1\.attn_v\.bias$=f32
^v\.blk\.1\.attn_q\.weight$=bf16
^v\.blk\.1\.attn_k\.weight$=bf16
^v\.blk\.1\.attn_v\.weight$=bf16
^v\.blk\.1\.ffn_up\.bias$=f32
^v\.blk\.1\.ffn_up\.weight$=bf16
^v\.blk\.1\.ffn_down\.bias$=f32
^v\.blk\.1\.ffn_down\.weight$=bf16
^v\.blk\.1\.ln1\.bias$=f32
^v\.blk\.1\.ln1\.weight$=f32
^v\.blk\.1\.ln2\.bias$=f32
^v\.blk\.1\.ln2\.weight$=f32
^v\.blk\.10\.attn_out\.bias$=f32
^v\.blk\.10\.attn_out\.weight$=bf16
^v\.blk\.10\.attn_q\.bias$=f32
^v\.blk\.10\.attn_k\.bias$=f32
^v\.blk\.10\.attn_v\.bias$=f32
^v\.blk\.10\.attn_q\.weight$=bf16
^v\.blk\.10\.attn_k\.weight$=bf16
^v\.blk\.10\.attn_v\.weight$=bf16
^v\.blk\.10\.ffn_up\.bias$=f32
^v\.blk\.10\.ffn_up\.weight$=bf16
^v\.blk\.10\.ffn_down\.bias$=f32
^v\.blk\.10\.ffn_down\.weight$=bf16
^v\.blk\.10\.ln1\.bias$=f32
^v\.blk\.10\.ln1\.weight$=f32
^v\.blk\.10\.ln2\.bias$=f32
^v\.blk\.10\.ln2\.weight$=f32
^v\.blk\.11\.attn_out\.bias$=f32
^v\.blk\.11\.attn_out\.weight$=bf16
^v\.blk\.11\.attn_q\.bias$=f32
^v\.blk\.11\.attn_k\.bias$=f32
^v\.blk\.11\.attn_v\.bias$=f32
^v\.blk\.11\.attn_q\.weight$=bf16
^v\.blk\.11\.attn_k\.weight$=bf16
^v\.blk\.11\.attn_v\.weight$=bf16
^v\.blk\.11\.ffn_up\.bias$=f32
^v\.blk\.11\.ffn_up\.weight$=bf16
^v\.blk\.11\.ffn_down\.bias$=f32
^v\.blk\.11\.ffn_down\.weight$=bf16
^v\.blk\.11\.ln1\.bias$=f32
^v\.blk\.11\.ln1\.weight$=f32
^v\.blk\.11\.ln2\.bias$=f32
^v\.blk\.11\.ln2\.weight$=f32
^v\.blk\.12\.attn_out\.bias$=f32
^v\.blk\.12\.attn_out\.weight$=bf16
^v\.blk\.12\.attn_q\.bias$=f32
^v\.blk\.12\.attn_k\.bias$=f32
^v\.blk\.12\.attn_v\.bias$=f32
^v\.blk\.12\.attn_q\.weight$=bf16
^v\.blk\.12\.attn_k\.weight$=bf16
^v\.blk\.12\.attn_v\.weight$=bf16
^v\.blk\.12\.ffn_up\.bias$=f32
^v\.blk\.12\.ffn_up\.weight$=bf16
^v\.blk\.12\.ffn_down\.bias$=f32
^v\.blk\.12\.ffn_down\.weight$=bf16
^v\.blk\.12\.ln1\.bias$=f32
^v\.blk\.12\.ln1\.weight$=f32
^v\.blk\.12\.ln2\.bias$=f32
^v\.blk\.12\.ln2\.weight$=f32
^v\.blk\.13\.attn_out\.bias$=f32
^v\.blk\.13\.attn_out\.weight$=bf16
^v\.blk\.13\.attn_q\.bias$=f32
^v\.blk\.13\.attn_k\.bias$=f32
^v\.blk\.13\.attn_v\.bias$=f32
^v\.blk\.13\.attn_q\.weight$=bf16
^v\.blk\.13\.attn_k\.weight$=bf16
^v\.blk\.13\.attn_v\.weight$=bf16
^v\.blk\.13\.ffn_up\.bias$=f32
^v\.blk\.13\.ffn_up\.weight$=bf16
^v\.blk\.13\.ffn_down\.bias$=f32
^v\.blk\.13\.ffn_down\.weight$=bf16
^v\.blk\.13\.ln1\.bias$=f32
^v\.blk\.13\.ln1\.weight$=f32
^v\.blk\.13\.ln2\.bias$=f32
^v\.blk\.13\.ln2\.weight$=f32
^v\.blk\.14\.attn_out\.bias$=f32
^v\.blk\.14\.attn_out\.weight$=bf16
^v\.blk\.14\.attn_q\.bias$=f32
^v\.blk\.14\.attn_k\.bias$=f32
^v\.blk\.14\.attn_v\.bias$=f32
^v\.blk\.14\.attn_q\.weight$=bf16
^v\.blk\.14\.attn_k\.weight$=bf16
^v\.blk\.14\.attn_v\.weight$=bf16
^v\.blk\.14\.ffn_up\.bias$=f32
^v\.blk\.14\.ffn_up\.weight$=bf16
^v\.blk\.14\.ffn_down\.bias$=f32
^v\.blk\.14\.ffn_down\.weight$=bf16
^v\.blk\.14\.ln1\.bias$=f32
^v\.blk\.14\.ln1\.weight$=f32
^v\.blk\.14\.ln2\.bias$=f32
^v\.blk\.14\.ln2\.weight$=f32
^v\.blk\.15\.attn_out\.bias$=f32
^v\.blk\.15\.attn_out\.weight$=bf16
^v\.blk\.15\.attn_q\.bias$=f32
^v\.blk\.15\.attn_k\.bias$=f32
^v\.blk\.15\.attn_v\.bias$=f32
^v\.blk\.15\.attn_q\.weight$=bf16
^v\.blk\.15\.attn_k\.weight$=bf16
^v\.blk\.15\.attn_v\.weight$=bf16
^v\.blk\.15\.ffn_up\.bias$=f32
^v\.blk\.15\.ffn_up\.weight$=bf16
^v\.blk\.15\.ffn_down\.bias$=f32
^v\.blk\.15\.ffn_down\.weight$=bf16
^v\.blk\.15\.ln1\.bias$=f32
^v\.blk\.15\.ln1\.weight$=f32
^v\.blk\.15\.ln2\.bias$=f32
^v\.blk\.15\.ln2\.weight$=f32
^v\.blk\.16\.attn_out\.bias$=f32
^v\.blk\.16\.attn_out\.weight$=bf16
^v\.blk\.16\.attn_q\.bias$=f32
^v\.blk\.16\.attn_k\.bias$=f32
^v\.blk\.16\.attn_v\.bias$=f32
^v\.blk\.16\.attn_q\.weight$=bf16
^v\.blk\.16\.attn_k\.weight$=bf16
^v\.blk\.16\.attn_v\.weight$=bf16
^v\.blk\.16\.ffn_up\.bias$=f32
^v\.blk\.16\.ffn_up\.weight$=bf16
^v\.blk\.16\.ffn_down\.bias$=f32
^v\.blk\.16\.ffn_down\.weight$=bf16
^v\.blk\.16\.ln1\.bias$=f32
^v\.blk\.16\.ln1\.weight$=f32
^v\.blk\.16\.ln2\.bias$=f32
^v\.blk\.16\.ln2\.weight$=f32
^v\.blk\.17\.attn_out\.bias$=f32
^v\.blk\.17\.attn_out\.weight$=bf16
^v\.blk\.17\.attn_q\.bias$=f32
^v\.blk\.17\.attn_k\.bias$=f32
^v\.blk\.17\.attn_v\.bias$=f32
^v\.blk\.17\.attn_q\.weight$=bf16
^v\.blk\.17\.attn_k\.weight$=bf16
^v\.blk\.17\.attn_v\.weight$=bf16
^v\.blk\.17\.ffn_up\.bias$=f32
^v\.blk\.17\.ffn_up\.weight$=bf16
^v\.blk\.17\.ffn_down\.bias$=f32
^v\.blk\.17\.ffn_down\.weight$=bf16
^v\.blk\.17\.ln1\.bias$=f32
^v\.blk\.17\.ln1\.weight$=f32
^v\.blk\.17\.ln2\.bias$=f32
^v\.blk\.17\.ln2\.weight$=f32
^v\.blk\.18\.attn_out\.bias$=f32
^v\.blk\.18\.attn_out\.weight$=bf16
^v\.blk\.18\.attn_q\.bias$=f32
^v\.blk\.18\.attn_k\.bias$=f32
^v\.blk\.18\.attn_v\.bias$=f32
^v\.blk\.18\.attn_q\.weight$=bf16
^v\.blk\.18\.attn_k\.weight$=bf16
^v\.blk\.18\.attn_v\.weight$=bf16
^v\.blk\.18\.ffn_up\.bias$=f32
^v\.blk\.18\.ffn_up\.weight$=bf16
^v\.blk\.18\.ffn_down\.bias$=f32
^v\.blk\.18\.ffn_down\.weight$=bf16
^v\.blk\.18\.ln1\.bias$=f32
^v\.blk\.18\.ln1\.weight$=f32
^v\.blk\.18\.ln2\.bias$=f32
^v\.blk\.18\.ln2\.weight$=f32
^v\.blk\.19\.attn_out\.bias$=f32
^v\.blk\.19\.attn_out\.weight$=bf16
^v\.blk\.19\.attn_q\.bias$=f32
^v\.blk\.19\.attn_k\.bias$=f32
^v\.blk\.19\.attn_v\.bias$=f32
^v\.blk\.19\.attn_q\.weight$=bf16
^v\.blk\.19\.attn_k\.weight$=bf16
^v\.blk\.19\.attn_v\.weight$=bf16
^v\.blk\.19\.ffn_up\.bias$=f32
^v\.blk\.19\.ffn_up\.weight$=bf16
^v\.blk\.19\.ffn_down\.bias$=f32
^v\.blk\.19\.ffn_down\.weight$=bf16
^v\.blk\.19\.ln1\.bias$=f32
^v\.blk\.19\.ln1\.weight$=f32
^v\.blk\.19\.ln2\.bias$=f32
^v\.blk\.19\.ln2\.weight$=f32
^v\.blk\.2\.attn_out\.bias$=f32
^v\.blk\.2\.attn_out\.weight$=bf16
^v\.blk\.2\.attn_q\.bias$=f32
^v\.blk\.2\.attn_k\.bias$=f32
^v\.blk\.2\.attn_v\.bias$=f32
^v\.blk\.2\.attn_q\.weight$=bf16
^v\.blk\.2\.attn_k\.weight$=bf16
^v\.blk\.2\.attn_v\.weight$=bf16
^v\.blk\.2\.ffn_up\.bias$=f32
^v\.blk\.2\.ffn_up\.weight$=bf16
^v\.blk\.2\.ffn_down\.bias$=f32
^v\.blk\.2\.ffn_down\.weight$=bf16
^v\.blk\.2\.ln1\.bias$=f32
^v\.blk\.2\.ln1\.weight$=f32
^v\.blk\.2\.ln2\.bias$=f32
^v\.blk\.2\.ln2\.weight$=f32
^v\.blk\.20\.attn_out\.bias$=f32
^v\.blk\.20\.attn_out\.weight$=bf16
^v\.blk\.20\.attn_q\.bias$=f32
^v\.blk\.20\.attn_k\.bias$=f32
^v\.blk\.20\.attn_v\.bias$=f32
^v\.blk\.20\.attn_q\.weight$=bf16
^v\.blk\.20\.attn_k\.weight$=bf16
^v\.blk\.20\.attn_v\.weight$=bf16
^v\.blk\.20\.ffn_up\.bias$=f32
^v\.blk\.20\.ffn_up\.weight$=bf16
^v\.blk\.20\.ffn_down\.bias$=f32
^v\.blk\.20\.ffn_down\.weight$=bf16
^v\.blk\.20\.ln1\.bias$=f32
^v\.blk\.20\.ln1\.weight$=f32
^v\.blk\.20\.ln2\.bias$=f32
^v\.blk\.20\.ln2\.weight$=f32
^v\.blk\.21\.attn_out\.bias$=f32
^v\.blk\.21\.attn_out\.weight$=bf16
^v\.blk\.21\.attn_q\.bias$=f32
^v\.blk\.21\.attn_k\.bias$=f32
^v\.blk\.21\.attn_v\.bias$=f32
^v\.blk\.21\.attn_q\.weight$=bf16
^v\.blk\.21\.attn_k\.weight$=bf16
^v\.blk\.21\.attn_v\.weight$=bf16
^v\.blk\.21\.ffn_up\.bias$=f32
^v\.blk\.21\.ffn_up\.weight$=bf16
^v\.blk\.21\.ffn_down\.bias$=f32
^v\.blk\.21\.ffn_down\.weight$=bf16
^v\.blk\.21\.ln1\.bias$=f32
^v\.blk\.21\.ln1\.weight$=f32
^v\.blk\.21\.ln2\.bias$=f32
^v\.blk\.21\.ln2\.weight$=f32
^v\.blk\.22\.attn_out\.bias$=f32
^v\.blk\.22\.attn_out\.weight$=bf16
^v\.blk\.22\.attn_q\.bias$=f32
^v\.blk\.22\.attn_k\.bias$=f32
^v\.blk\.22\.attn_v\.bias$=f32
^v\.blk\.22\.attn_q\.weight$=bf16
^v\.blk\.22\.attn_k\.weight$=bf16
^v\.blk\.22\.attn_v\.weight$=bf16
^v\.blk\.22\.ffn_up\.bias$=f32
^v\.blk\.22\.ffn_up\.weight$=bf16
^v\.blk\.22\.ffn_down\.bias$=f32
^v\.blk\.22\.ffn_down\.weight$=bf16
^v\.blk\.22\.ln1\.bias$=f32
^v\.blk\.22\.ln1\.weight$=f32
^v\.blk\.22\.ln2\.bias$=f32
^v\.blk\.22\.ln2\.weight$=f32
^v\.blk\.23\.attn_out\.bias$=f32
^v\.blk\.23\.attn_out\.weight$=bf16
^v\.blk\.23\.attn_q\.bias$=f32
^v\.blk\.23\.attn_k\.bias$=f32
^v\.blk\.23\.attn_v\.bias$=f32
^v\.blk\.23\.attn_q\.weight$=bf16
^v\.blk\.23\.attn_k\.weight$=bf16
^v\.blk\.23\.attn_v\.weight$=bf16
^v\.blk\.23\.ffn_up\.bias$=f32
^v\.blk\.23\.ffn_up\.weight$=bf16
^v\.blk\.23\.ffn_down\.bias$=f32
^v\.blk\.23\.ffn_down\.weight$=bf16
^v\.blk\.23\.ln1\.bias$=f32
^v\.blk\.23\.ln1\.weight$=f32
^v\.blk\.23\.ln2\.bias$=f32
^v\.blk\.23\.ln2\.weight$=f32
^v\.blk\.24\.attn_out\.bias$=f32
^v\.blk\.24\.attn_out\.weight$=bf16
^v\.blk\.24\.attn_q\.bias$=f32
^v\.blk\.24\.attn_k\.bias$=f32
^v\.blk\.24\.attn_v\.bias$=f32
^v\.blk\.24\.attn_q\.weight$=bf16
^v\.blk\.24\.attn_k\.weight$=bf16
^v\.blk\.24\.attn_v\.weight$=bf16
^v\.blk\.24\.ffn_up\.bias$=f32
^v\.blk\.24\.ffn_up\.weight$=bf16
^v\.blk\.24\.ffn_down\.bias$=f32
^v\.blk\.24\.ffn_down\.weight$=bf16
^v\.blk\.24\.ln1\.bias$=f32
^v\.blk\.24\.ln1\.weight$=f32
^v\.blk\.24\.ln2\.bias$=f32
^v\.blk\.24\.ln2\.weight$=f32
^v\.blk\.25\.attn_out\.bias$=f32
^v\.blk\.25\.attn_out\.weight$=bf16
^v\.blk\.25\.attn_q\.bias$=f32
^v\.blk\.25\.attn_k\.bias$=f32
^v\.blk\.25\.attn_v\.bias$=f32
^v\.blk\.25\.attn_q\.weight$=bf16
^v\.blk\.25\.attn_k\.weight$=bf16
^v\.blk\.25\.attn_v\.weight$=bf16
^v\.blk\.25\.ffn_up\.bias$=f32
^v\.blk\.25\.ffn_up\.weight$=bf16
^v\.blk\.25\.ffn_down\.bias$=f32
^v\.blk\.25\.ffn_down\.weight$=bf16
^v\.blk\.25\.ln1\.bias$=f32
^v\.blk\.25\.ln1\.weight$=f32
^v\.blk\.25\.ln2\.bias$=f32
^v\.blk\.25\.ln2\.weight$=f32
^v\.blk\.26\.attn_out\.bias$=f32
^v\.blk\.26\.attn_out\.weight$=bf16
^v\.blk\.26\.attn_q\.bias$=f32
^v\.blk\.26\.attn_k\.bias$=f32
^v\.blk\.26\.attn_v\.bias$=f32
^v\.blk\.26\.attn_q\.weight$=bf16
^v\.blk\.26\.attn_k\.weight$=bf16
^v\.blk\.26\.attn_v\.weight$=bf16
^v\.blk\.26\.ffn_up\.bias$=f32
^v\.blk\.26\.ffn_up\.weight$=bf16
^v\.blk\.26\.ffn_down\.bias$=f32
^v\.blk\.26\.ffn_down\.weight$=bf16
^v\.blk\.26\.ln1\.bias$=f32
^v\.blk\.26\.ln1\.weight$=f32
^v\.blk\.26\.ln2\.bias$=f32
^v\.blk\.26\.ln2\.weight$=f32
^v\.blk\.3\.attn_out\.bias$=f32
^v\.blk\.3\.attn_out\.weight$=bf16
^v\.blk\.3\.attn_q\.bias$=f32
^v\.blk\.3\.attn_k\.bias$=f32
^v\.blk\.3\.attn_v\.bias$=f32
^v\.blk\.3\.attn_q\.weight$=bf16
^v\.blk\.3\.attn_k\.weight$=bf16
^v\.blk\.3\.attn_v\.weight$=bf16
^v\.blk\.3\.ffn_up\.bias$=f32
^v\.blk\.3\.ffn_up\.weight$=bf16
^v\.blk\.3\.ffn_down\.bias$=f32
^v\.blk\.3\.ffn_down\.weight$=bf16
^v\.blk\.3\.ln1\.bias$=f32
^v\.blk\.3\.ln1\.weight$=f32
^v\.blk\.3\.ln2\.bias$=f32
^v\.blk\.3\.ln2\.weight$=f32
^v\.blk\.4\.attn_out\.bias$=f32
^v\.blk\.4\.attn_out\.weight$=bf16
^v\.blk\.4\.attn_q\.bias$=f32
^v\.blk\.4\.attn_k\.bias$=f32
^v\.blk\.4\.attn_v\.bias$=f32
^v\.blk\.4\.attn_q\.weight$=bf16
^v\.blk\.4\.attn_k\.weight$=bf16
^v\.blk\.4\.attn_v\.weight$=bf16
^v\.blk\.4\.ffn_up\.bias$=f32
^v\.blk\.4\.ffn_up\.weight$=bf16
^v\.blk\.4\.ffn_down\.bias$=f32
^v\.blk\.4\.ffn_down\.weight$=bf16
^v\.blk\.4\.ln1\.bias$=f32
^v\.blk\.4\.ln1\.weight$=f32
^v\.blk\.4\.ln2\.bias$=f32
^v\.blk\.4\.ln2\.weight$=f32
^v\.blk\.5\.attn_out\.bias$=f32
^v\.blk\.5\.attn_out\.weight$=bf16
^v\.blk\.5\.attn_q\.bias$=f32
^v\.blk\.5\.attn_k\.bias$=f32
^v\.blk\.5\.attn_v\.bias$=f32
^v\.blk\.5\.attn_q\.weight$=bf16
^v\.blk\.5\.attn_k\.weight$=bf16
^v\.blk\.5\.attn_v\.weight$=bf16
^v\.blk\.5\.ffn_up\.bias$=f32
^v\.blk\.5\.ffn_up\.weight$=bf16
^v\.blk\.5\.ffn_down\.bias$=f32
^v\.blk\.5\.ffn_down\.weight$=bf16
^v\.blk\.5\.ln1\.bias$=f32
^v\.blk\.5\.ln1\.weight$=f32
^v\.blk\.5\.ln2\.bias$=f32
^v\.blk\.5\.ln2\.weight$=f32
^v\.blk\.6\.attn_out\.bias$=f32
^v\.blk\.6\.attn_out\.weight$=bf16
^v\.blk\.6\.attn_q\.bias$=f32
^v\.blk\.6\.attn_k\.bias$=f32
^v\.blk\.6\.attn_v\.bias$=f32
^v\.blk\.6\.attn_q\.weight$=bf16
^v\.blk\.6\.attn_k\.weight$=bf16
^v\.blk\.6\.attn_v\.weight$=bf16
^v\.blk\.6\.ffn_up\.bias$=f32
^v\.blk\.6\.ffn_up\.weight$=bf16
^v\.blk\.6\.ffn_down\.bias$=f32
^v\.blk\.6\.ffn_down\.weight$=bf16
^v\.blk\.6\.ln1\.bias$=f32
^v\.blk\.6\.ln1\.weight$=f32
^v\.blk\.6\.ln2\.bias$=f32
^v\.blk\.6\.ln2\.weight$=f32
^v\.blk\.7\.attn_out\.bias$=f32
^v\.blk\.7\.attn_out\.weight$=bf16
^v\.blk\.7\.attn_q\.bias$=f32
^v\.blk\.7\.attn_k\.bias$=f32
^v\.blk\.7\.attn_v\.bias$=f32
^v\.blk\.7\.attn_q\.weight$=bf16
^v\.blk\.7\.attn_k\.weight$=bf16
^v\.blk\.7\.attn_v\.weight$=bf16
^v\.blk\.7\.ffn_up\.bias$=f32
^v\.blk\.7\.ffn_up\.weight$=bf16
^v\.blk\.7\.ffn_down\.bias$=f32
^v\.blk\.7\.ffn_down\.weight$=bf16
^v\.blk\.7\.ln1\.bias$=f32
^v\.blk\.7\.ln1\.weight$=f32
^v\.blk\.7\.ln2\.bias$=f32
^v\.blk\.7\.ln2\.weight$=f32
^v\.blk\.8\.attn_out\.bias$=f32
^v\.blk\.8\.attn_out\.weight$=bf16
^v\.blk\.8\.attn_q\.bias$=f32
^v\.blk\.8\.attn_k\.bias$=f32
^v\.blk\.8\.attn_v\.bias$=f32
^v\.blk\.8\.attn_q\.weight$=bf16
^v\.blk\.8\.attn_k\.weight$=bf16
^v\.blk\.8\.attn_v\.weight$=bf16
^v\.blk\.8\.ffn_up\.bias$=f32
^v\.blk\.8\.ffn_up\.weight$=bf16
^v\.blk\.8\.ffn_down\.bias$=f32
^v\.blk\.8\.ffn_down\.weight$=bf16
^v\.blk\.8\.ln1\.bias$=f32
^v\.blk\.8\.ln1\.weight$=f32
^v\.blk\.8\.ln2\.bias$=f32
^v\.blk\.8\.ln2\.weight$=f32
^v\.blk\.9\.attn_out\.bias$=f32
^v\.blk\.9\.attn_out\.weight$=bf16
^v\.blk\.9\.attn_q\.bias$=f32
^v\.blk\.9\.attn_k\.bias$=f32
^v\.blk\.9\.attn_v\.bias$=f32
^v\.blk\.9\.attn_q\.weight$=bf16
^v\.blk\.9\.attn_k\.weight$=bf16
^v\.blk\.9\.attn_v\.weight$=bf16
^v\.blk\.9\.ffn_up\.bias$=f32
^v\.blk\.9\.ffn_up\.weight$=bf16
^v\.blk\.9\.ffn_down\.bias$=f32
^v\.blk\.9\.ffn_down\.weight$=bf16
^v\.blk\.9\.ln1\.bias$=f32
^v\.blk\.9\.ln1\.weight$=f32
^v\.blk\.9\.ln2\.bias$=f32
^v\.blk\.9\.ln2\.weight$=f32
^v\.deepstack\.0\.fc1\.bias$=f32
^v\.deepstack\.0\.fc1\.weight$=bf16
^v\.deepstack\.0\.fc2\.bias$=f32
^v\.deepstack\.0\.fc2\.weight$=bf16
^v\.deepstack\.0\.norm\.bias$=f32
^v\.deepstack\.0\.norm\.weight$=f32
^v\.deepstack\.1\.fc1\.bias$=f32
^v\.deepstack\.1\.fc1\.weight$=bf16
^v\.deepstack\.1\.fc2\.bias$=f32
^v\.deepstack\.1\.fc2\.weight$=bf16
^v\.deepstack\.1\.norm\.bias$=f32
^v\.deepstack\.1\.norm\.weight$=f32
^v\.deepstack\.2\.fc1\.bias$=f32
^v\.deepstack\.2\.fc1\.weight$=bf16
^v\.deepstack\.2\.fc2\.bias$=f32
^v\.deepstack\.2\.fc2\.weight$=bf16
^v\.deepstack\.2\.norm\.bias$=f32
^v\.deepstack\.2\.norm\.weight$=f32
^mm\.0\.bias$=f32
^mm\.0\.weight$=bf16
^mm\.2\.bias$=f32
^mm\.2\.weight$=bf16
^v\.post_ln\.bias$=f32
^v\.post_ln\.weight$=f32
^v\.patch_embd\.weight$=f32
^v\.patch_embd\.weight\.1$=f32
^v\.position_embd\.weight$=f32
"
fi

# --------------- DETECT & DEFINE SHA256 HELPER ---------------
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
  # fallback stub: always errors out
  sha256tool=()
  args=()
fi

# _sha256sum reads either from file (if you pass an arg) or from stdin
_sha256sum() {
  if (( $# > 0 )); then
    # file‑mode: pass filename as $1
    "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
  else
    # stdin‑mode: read data from pipe
    "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
  fi
}

build_range_regex() {
  local S=$1 E=$2
  # declare every variable we use
  local parts=() full_decades=() partial=()
  local run_start run_prev d u_lo u_hi low2 start_d end_d joined

  _debug "    build_range_regex S=$S E=$E"

  # (1) single digits
  if (( S <= 9 )); then
    local hi=$(( E<9?E:9 ))
    if (( S==0 && hi==9 )); then
      parts+=("[0-9]"); _debug "      add [0-9]"
    elif (( S==hi )); then
      parts+=("$S");     _debug "      add $S"
    else
      parts+=("[${S}-${hi}]"); _debug "      add [${S}-${hi}]"
    fi
  fi

  # (2) decades 10–99
  if (( E >= 10 )); then
    low2=$(( S<10?10:S ))
    start_d=$(( low2/10 ))
    end_d=$(( E/10 ))
    _debug "      decades from $start_d to $end_d"

    for ((d=start_d; d<=end_d; d++)); do
      u_lo=$(( d==start_d? low2%10:0 ))
      u_hi=$(( d==end_d  ? E%10  :9 ))
      if (( u_lo==0 && u_hi==9 )); then
        full_decades+=("$d")
        _debug "        full decade $d"
      else
        if (( u_lo==u_hi )); then
          partial+=("${d}${u_lo}")
          _debug "        partial single ${d}${u_lo}"
        else
          partial+=("${d}[${u_lo}-${u_hi}]")
          _debug "        partial range ${d}[${u_lo}-${u_hi}]"
        fi
      fi
    done

    # collapse full_decades runs
    if (( ${#full_decades[@]} )); then
      IFS=$'\n' sorted_fd=($(printf '%s\n' "${full_decades[@]}" | sort -n))
      unset IFS
      run_start=${sorted_fd[0]}; run_prev=$run_start

      for d in "${sorted_fd[@]:1}"; do
        if (( d == run_prev+1 )); then
          run_prev=$d
        else
          if (( run_start==run_prev )); then
            parts+=("${run_start}[0-9]")
            _debug "          flush ${run_start}[0-9]"
          else
            parts+=("[${run_start}-${run_prev}][0-9]")
            _debug "          flush [${run_start}-${run_prev}][0-9]"
          fi
          run_start=$d; run_prev=$d
        fi
      done
      # final flush
      if (( run_start==run_prev )); then
        parts+=("${run_start}[0-9]")
        _debug "          final ${run_start}[0-9]"
      else
        parts+=("[${run_start}-${run_prev}][0-9]")
        _debug "          final [${run_start}-${run_prev}][0-9]"
      fi
    fi

    # append partial pieces
    for p in "${partial[@]}"; do
      parts+=("$p")
      _debug "        append partial $p"
    done
  fi

  # (3) fallback for E>99
  if (( E > 99 )); then
    parts=()
    for ((i=S; i<=E; i++)); do parts+=("$i"); done
  fi

  # (4) safe join
  joined=$(printf "|%s" "${parts[@]}")
  joined=${joined:1}
  _debug "    build_range_regex returns %s" "$joined"
  printf "%s" "$joined"
}


# --------------------------------------------------------------------------------
# shorten_regex_list(): read stdin, collapse consecutive blk.N (and similar) lines
# --------------------------------------------------------------------------------
shorten_regex_list() {
  local -a lines
  declare -A groups
  declare -A prefixes_seen

  # Read input line by line
  while IFS= read -r line; do
    if [[ $line =~ ^\^?(blk)\\.([0-9]+)\\.(.+)\$?$ ]]; then
      # Extract prefix, block number and suffix
      block_prefix="${BASH_REMATCH[1]}"
      block_num="${BASH_REMATCH[2]}"
      suffix="${BASH_REMATCH[3]}"
      groups["${block_prefix}_${suffix}"]+="$block_num "
      _debug "Bucket $block_num → prefix $block_prefix suffix $suffix"
    elif [[ $line =~ ^\^?(mm)\\.([0-9]+)\\.(.+)\$?$ ]]; then
      # Extract prefix, block number and suffix
      block_prefix="${BASH_REMATCH[1]}"
      block_num="${BASH_REMATCH[2]}"
      suffix="${BASH_REMATCH[3]}"
      groups["${block_prefix}_${suffix}"]+="$block_num "
      _debug "Bucket $block_num → prefix $block_prefix suffix $suffix"
    elif [[ $line =~ ^\^?(v\\.blk)\\.([0-9]+)\\.(.+)\$?$ ]]; then
      # Extract prefix, block number and suffix
      block_prefix="${BASH_REMATCH[1]}"
      block_num="${BASH_REMATCH[2]}"
      suffix="${BASH_REMATCH[3]}"
      groups["${block_prefix}_${suffix}"]+="$block_num "
      _debug "Bucket $block_num → prefix $block_prefix suffix $suffix"
    elif [[ $line =~ ^\^?(v\\.deepstack)\\.([0-9]+)\\.(.+)\$?$ ]]; then
      # Extract prefix, block number and suffix
      block_prefix="${BASH_REMATCH[1]}"
      block_num="${BASH_REMATCH[2]}"
      suffix="${BASH_REMATCH[3]}"
      groups["${block_prefix}_${suffix}"]+="$block_num "
      _debug "Bucket $block_num → prefix $block_prefix suffix $suffix"
    else
      # Non-blk (mm, v, deepstack) line: output immediately
      printf '%s\n' "$line"
    fi
  done

  # collect unique prefixes
  declare -A prefixes_seen
  for key in "${!groups[@]}"; do
    prefix="${key%%_*}"        # part before first underscore
    prefixes_seen["$prefix"]=1
  done

  # iterate prefixes then suffixes
  for prefix in "${!prefixes_seen[@]}"; do
    _debug "Processing prefix: $prefix"
    for key in "${!groups[@]}"; do
      if [[ $key == "${prefix}_"* ]]; then
        _debug "Processing key: $key"
        # compute suffix by slicing off prefix + the underscore
        len=$(( ${#prefix} + 1 ))
        suffix="${key:len}"                 # plain substring — no globbing
        nums_str="${groups[$key]% }"        # Get the numbers for this suffix
        _debug "Processing suffix: $suffix (value: $nums_str)"

        # Split nums_str into array
        read -ra nums <<<"$nums_str"
        # Sort and uniq
        IFS=$'\n' sorted=($(printf '%s\n' "${nums[@]}" | sort -n | uniq))
        unset IFS

        _debug "Processing sorted suffix: ${sorted[*]}"

        # If no numbers, skip
        if (( ${#sorted[@]} == 0 )); then
          continue
        fi

        # Break into consecutive runs
        runs=()
        run_start=${sorted[0]}
        run_prev=${sorted[0]}

        for (( i=1; i<${#sorted[@]}; i++ )); do
          num=${sorted[i]}
          if (( num == run_prev + 1 )); then
            run_prev=$num
          else
            runs+=("$run_start $run_prev")
            run_start=$num
            run_prev=$num
          fi
        done
        runs+=("$run_start $run_prev")

        # Build the regex parts for the runs
        parts=()
        for run in "${runs[@]}"; do
          read s e <<<"$run"
          if (( s == e )); then
            parts+=("$s")
            _debug "Run: single number $s"
          else
            part=$(build_range_regex "$s" "$e")
            parts+=("$part")
            _debug "Run: consecutive $s to $e -> $part"
          fi
        done

        # Join parts with '|'
        block_regex=$(IFS='|'; echo "${parts[*]}")

        # If block_regex contains '|', wrap in non-capturing group
        if [[ "$block_regex" == *"|"* ]]; then
          #block_regex="(?:$block_regex)"
          block_regex="($block_regex)"
          _debug "Wrapped block_regex: $block_regex"
        fi

        printf '^%s\\.%s\\.%s\n' "$prefix" "$block_regex" "$suffix"

      fi
    done
  done

}

# -----------------------------------------------------------------------------------------
# optimise_regex_list(): read stdin, merges the residual consecutive bracket-prefix/suffix
# -----------------------------------------------------------------------------------------
optimise_regex_list() {
  local line pr inner suffix
  local -a parts plain extras final_parts nums ks
  declare -A by_suffix by_prefix pmap

  while IFS= read -r line; do
    # only lines of the form blk\.(...)\.<suffix> (and similar for mm, v and deepstack)
    if ([[ $line == *\|\[* ]] || [[ $line == *\]\|* ]]); then
    #_debug "line-> $line"
      for prefix in "blk" "mm" "v\\.blk" "v\\.deepstack"; do
      #_debug "prefix-> $prefix"
        lit1="^${prefix}\\."   # literal: ^ then prefix then backslash+dot
        lit2="${prefix}\\."    # literal: prefix then backslash+dot
        # check literal string-starts-with using substring comparison (no pattern/regex)
        if [[ "${line:0:${#lit1}}" == "$lit1" || "^${line:0:${#lit2}}" == "$lit2" ]]; then
          _debug "Original line: $line"

          # 1) strip leading 'blk\.(' and similar prefix
          # Check if line starts with lit1 + '(' or lit2 + '('
          if [[ "${line:0:$(( ${#lit1} + 1 ))}" == "${lit1}(" ]]; then
            pr="${line:$(( ${#lit1} + 1 ))}"   # strip ^prefix\.(
          elif [[ "${line:0:$(( ${#lit2} + 1 ))}" == "${lit2}(" ]]; then
            pr="${line:$(( ${#lit2} + 1 ))}"
          else
            pr="$line"
          fi

          # 2) extract up to ')\.' as inner
          inner=${pr%%')\.'*}
          # 3) suffix is what follows ")\."
          suffix=${pr#*')\.'}

          _debug "  pr         = '$pr'"
          _debug "  inner      = '$inner'"
          _debug "  suffix     = '$suffix'"

          # 4) split alternation into parts[]
          IFS='|' read -r -a parts <<<"$inner"
          _debug "  parts      = ${parts[*]}"

          # clear collectors
          plain=(); extras=()
          by_suffix=(); by_prefix=()

          # 5) classify each part
          for p in "${parts[@]}"; do
            if [[ $p =~ ^([0-9]+)(\[[0-9]+-[0-9]+\])$ ]]; then
              local num=${BASH_REMATCH[1]}
              local su=${BASH_REMATCH[2]}
              by_suffix["$su"]+="$num "
              _debug "    by_suffix[$su] += $num"
            elif [[ $p =~ ^(\[[0-9]+-[0-9]+\])([0-9]+)$ ]]; then
              local prf=${BASH_REMATCH[1]}
              local num=${BASH_REMATCH[2]}
              by_prefix["$prf"]+="$num "
              _debug "    by_prefix[$prf] += $num"
            else
              plain+=("$p")
              _debug "    plain += $p"
            fi
          done

          # 6) merge runs in by_suffix (X[lo-hi])
          for su in "${!by_suffix[@]}"; do
            # build and sort unique nums array
            read -r -a nums <<<"${by_suffix[$su]}"
            IFS=$'\n' nums=($(printf '%s\n' "${nums[@]}" | sort -n | uniq))
            unset IFS

            # skip if no entries
            if [[ ${#nums[@]} -eq 0 ]]; then
              _debug "  no entries for suffix $su, skipping"
              continue
            fi

            _debug "  merging by_suffix[$su]: ${nums[*]}"

            # declare before assignment
            local start prev n
            start=${nums[0]}
            prev=$start

            for n in "${nums[@]:1}"; do
              if (( n == prev+1 )); then
                prev=$n
              else
                if (( start < prev )); then
                  extras+=("[${start}-${prev}]$su")
                  _debug "    extras += [${start}-${prev}]$su"
                else
                  extras+=("${start}$su")
                  _debug "    extras += ${start}$su"
                fi
                start=$n; prev=$n
              fi
            done
            # flush last
            if (( start < prev )); then
              extras+=("[${start}-${prev}]$su")
              _debug "    extras += [${start}-${prev}]$su"
            else
              extras+=("${start}$su")
              _debug "    extras += ${start}$su"
            fi
          done

          for prf in "${!by_prefix[@]}"; do
            # build and sort unique nums array
            read -r -a nums <<<"${by_prefix[$prf]}"
            IFS=$'\n' nums=($(printf '%s\n' "${nums[@]}" | sort -n | uniq))
            unset IFS

            # skip if no entries
            if [[ ${#nums[@]} -eq 0 ]]; then
              _debug "    no entries for prefix $prf, skipping"
              continue
            fi

            _debug "  merging by_prefix[$prf]: ${nums[*]}"

            # declare before assignment
            local start prev n
            start=${nums[0]}
            prev=$start

            for n in "${nums[@]:1}"; do
              if (( n == prev+1 )); then
                prev=$n
              else
                if (( start < prev )); then
                  extras+=("$prf[${start}-${prev}]")
                  _debug "    extras += $prf[${start}-${prev}]"
                else
                  extras+=("$prf$start")
                  _debug "    extras += $prf$start"
                fi
                start=$n; prev=$n
              fi
            done
            # flush last
            if (( start < prev )); then
              extras+=("$prf[${start}-${prev}]")
              _debug "    extras += $prf[${start}-${prev}]"
            else
              extras+=("$prf$start")
              _debug "    extras += $prf$start"
            fi
          done

          # 8) combine plain + extras
          final_parts=( "${plain[@]}" "${extras[@]}" )
          _debug "  final_parts = ${final_parts[*]}"

          # --- sorting logic ---
          sorted_stream=$(
            for p in "${final_parts[@]}"; do
              s="$p"
              key=0
              while [[ -n $s ]]; do
                if [[ $s =~ ^([0-9]+)(.*) ]]; then
                  chunk="${BASH_REMATCH[1]}"
                  s="${BASH_REMATCH[2]}"
                elif [[ $s =~ ^\[([0-9]+)-([0-9]+)\](.*) ]]; then
                  chunk="${BASH_REMATCH[1]}"
                  s="${BASH_REMATCH[3]}"
                else
                  key=999999999999
                  break
                fi
                digits=${#chunk}
                multiplier=1
                for ((i=0;i<digits;i++)); do
                  multiplier=$((multiplier * 10))
                done
                key=$(( key * multiplier + chunk ))
              done
              printf '%s\t%s\n' "$key" "$p"
            done | sort -t$'\t' -k1,1n -k2,2
          )

          # read sorted result back
          final_parts=()
          while IFS=$'\t' read -r _key item; do
            [[ -z $item ]] && continue
            final_parts+=( "$item" )
          done <<< "$sorted_stream"
          _debug "  final_parts sorted = ${final_parts[*]}"

          # 9) re-assemble
          printf '^%s\.(' "$prefix"
          ( IFS='|'; printf '%s' "${final_parts[*]}" )
          printf ')\.%s\n' "$suffix"
        fi
      done
    else
      printf '^%s\n' "${line#^}"
    fi
  done
}

# ---------------------------------------------------------------------------------------------
# expand_ranges(): read stdin, separate regex range entries if not supported by llama-quantize
# ---------------------------------------------------------------------------------------------
expand_ranges() {
  while IFS= read -r input; do
    local prefix body suffix

    # Check for parentheses
    if [[ "$input" =~ \(.*\) ]]; then
      prefix="${input%%(*}"
      body="${input#*\(}"
      body="${body%\)*}"
      suffix="${input##*\)}"
    else
      prefix=""
      body="${input}"
      suffix=""
    fi

    # Convert [a-b] to {a..b}
    body_expanded=$(echo "$body" | sed -E 's/\[([0-9]+)-([0-9]+)\]/{\1..\2}/g')

    # Always split on | and process each line
    echo "$body_expanded" | tr '|' '\n' | while IFS= read -r part; do
      # Use brace expansion only if there are braces
      if [[ "$part" == *\{*..*\}* ]]; then
        # Escape backslashes before eval
        escaped_part="${part//\\/\\\\}"
        eval "expanded=( $escaped_part )"
        for e in "${expanded[@]}"; do
          printf '%s%s%s\n' "$prefix" "$e" "$suffix"
        done
      else
        # No brace expansion needed
        printf '%s%s%s\n' "$prefix" "$part" "$suffix"
      fi
    done
  done
}

# -------------------------------------------------------------------
# reorder_and_group(): reorganise and group the final output
# -------------------------------------------------------------------
reorder_and_group() {
  # initialize arrays
  general=() gpu_shexp=() cpu_exps=() others=()

  # Read and classify lines
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    case "$line" in
      \^output\\.*|\^output_norm\\.*|\^token_embd\\.*|output\\.*|output_norm\\.*|token_embd\\.*)
        general+=("$line") ;;
      *shexp*)
        gpu_shexp+=("$line") ;;
      *exps*)
        cpu_exps+=("$line") ;;
      *)
        others+=("$line") ;;
    esac
  done

  # Helper: bucket lines by first digit in quant, sort descending
  bucket_by_bit() {
    local -n lines=$1
    local -a flat sorted
    local line quant bit
    for line in "${lines[@]}"; do
      quant="${line#*=}"
      if [[ $quant =~ ([0-9]) ]]; then
        bit="${BASH_REMATCH[1]}"
      else
        bit=0
      fi
      flat+=("$bit:$line")
    done
    IFS=$'\n' sorted=( $(printf '%s\n' "${flat[@]}" | sort -t: -k1,1nr) )
    unset IFS flat
    for item in "${sorted[@]}"; do
      echo "${item#*:}"
    done
  }

  # Helper: compute unique sorted quant bits for a group
  list_bits() {
    local -n lines=$1
    local bits=() bit
    for l in "${lines[@]}"; do
      q="${l#*=}"
      [[ $q =~ ([0-9]+) ]] && bits+=("${BASH_REMATCH[1]}")
    done
    printf '%s\n' "${bits[@]}" | sort -urn | tr '\n' ' '
  }

  # --- Output sections ---
  echo "## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/"
  [[ -n "$MODEL_NAME" ]] && echo "# Model name: $(basename ${MODEL_NAME})"
  [[ -n "$MODEL_LINK" ]] && echo "# Link to the original model: ${MODEL_LINK}"
  echo

  # Model head & embeddings
  if (( ${#general[@]} )); then
    echo "## Model head & embeddings — qbits: $(list_bits general)"
    for l in "${general[@]}"; do echo "$l"; done
    echo
  fi

  # Special attention weights
  attn_special=()
  for i in "${!others[@]}"; do
    if [[ "${others[i]}" =~ attn_k_b ]]; then
      attn_special+=("${others[i]}")
      unset 'others[i]'
    fi
  done
  if (( ${#attn_special[@]} )); then
    echo "## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: $(list_bits attn_special)"
    for l in "${attn_special[@]}"; do echo "$l"; done
    echo
  fi

  # Multi-headed attention parameters
  attn_group=()
  for i in "${!others[@]}"; do
    l="${others[i]}"
    if [[ "$l" =~ attn_.* ]] && [[ "$l" != *attn_k_b* ]]; then
      attn_group+=("$l")
      unset 'others[i]'
    fi
  done
  if (( ${#attn_group[@]} )); then
    echo "## Multi-headed attention parameters — qbits: $(list_bits attn_group)"
    for l in "${attn_group[@]}"; do echo "$l"; done
    echo
  fi

  # Dense Feed-Forward Network weights (main up/down projections + dense gates for blk.[0-2])
  ffn_raw=()
  for i in "${!others[@]}"; do
    l="${others[i]}"

    # skip expert-formatted tensors (they are handled elsewhere)
    if [[ "$l" =~ exps ]] || [[ "$l" =~ shexp ]]; then
      continue
    fi

    # collect down/up projections anywhere
    if [[ "$l" == *ffn_down* ]] || [[ "$l" == *ffn_up* ]]; then
      ffn_raw+=( "$l" )
      unset 'others[i]'
      continue
    fi

    # collect dense gate weights specifically for blocks 0..2
    # the input lines contain literal patterns like '^blk\.[0-2]\.ffn_gate...'
    # use fixed-string grep to match that exact substring
    if printf '%s\n' "$l" | grep -F -q '^blk\.[0-2]\.ffn_gate'; then
      ffn_raw+=( "$l" )
      unset 'others[i]'
      continue
    fi
    if printf '%s\n' "$l" | grep -F -q '^v\.blk\.[0-2]\.ffn_gate'; then
      ffn_raw+=( "$l" )
      unset 'others[i]'
      continue
    fi
  done

  if (( ${#ffn_raw[@]} )); then
    echo "## Dense Feed-Forward Network weights — qbits: $(list_bits ffn_raw)"
    for l in "${ffn_raw[@]}"; do echo "$l"; done
    echo
  fi

  ln_post=() embeddings=() deepstack=() nextn=() moe_gating=() gate_raw=() misc=()
  for i in "${!others[@]}"; do
    l="${others[i]}"
    case "$l" in
      *post_ln*|*ln1*|*ln2* )
        ln_post+=("$l") ;;
      *patch_embd*|*position_embd* )
        embeddings+=("$l") ;;
      *deepstack* )
        deepstack+=("$l") ;;
      *nextn* )
        nextn+=("$l") ;;
      # MoE gating & routing: gate input + routing/probability biases
      *ffn_gate_inp*|*exp_probs_b* )
        moe_gating+=("$l") ;;
      # remaining gate-related tensors (e.g. dense/block-0..2 ffn_gate were captured earlier)
      *ffn_gate* )
        gate_raw+=("$l") ;;
      * )
        misc+=("$l") ;;
    esac
    unset 'others[i]'
  done

  # Print groups
  if (( ${#ln_post[@]} )); then
    echo "## LayerNorm / Post-LN parameters — qbits: $(list_bits ln_post)"
    for l in "${ln_post[@]}"; do echo "$l"; done
    echo
  fi

  if (( ${#embeddings[@]} )); then
    echo "## Embeddings & positional encodings — qbits: $(list_bits embeddings)"
    for l in "${embeddings[@]}"; do echo "$l"; done
    echo
  fi

  if (( ${#deepstack[@]} )); then
    echo "## Deepstack modules — qbits: $(list_bits deepstack)"
    for l in "${deepstack[@]}"; do echo "$l"; done
    echo
  fi

  if (( ${#nextn[@]} )); then
    echo "## NextN tensors — qbits: $(list_bits nextn)"
    for l in "${nextn[@]}"; do echo "$l"; done
    echo
  fi

  # MoE Gating & Routing (ffn_gate_inp, exp_probs_b, etc.)
  if (( ${#moe_gating[@]} )); then
    echo "## MoE Gating & Routing — qbits: $(list_bits moe_gating)"
    for l in "${moe_gating[@]}"; do echo "$l"; done
    echo
  fi

  # Gating network (other gate-related tensors not captured above)
  if (( ${#gate_raw[@]} )); then
    echo "## Gating network — qbits: $(list_bits gate_raw)"
    for l in "${gate_raw[@]}"; do echo "$l"; done
    echo
  fi

  # Misc / Other remaining tensors
  if (( ${#misc[@]} )); then
    echo "## Misc / Other tensors — qbits: $(list_bits misc)"
    for l in "${misc[@]}"; do echo "$l"; done
    echo
  fi

  # Shared experts section: ffn_*_shexp
  if (( ${#gpu_shexp[@]} )); then
    echo "## GPU-loaded - MoE Shared Experts Feed-Forward Network - ffn_*_shexp"
    gpu_down=() gpu_up=() gpu_gate=()
    for l in "${gpu_shexp[@]}"; do
      case "$l" in
        *ffn_down_*) gpu_down+=("$l") ;; 
        *ffn_up_*)   gpu_up+=("$l")   ;; 
        *ffn_gate_*) gpu_gate+=("$l") ;; 
      esac
    done
    (( ${#gpu_down[@]} )) && echo "# ffn_down_shexp — down-projection (shared experts) — qbits: $(list_bits gpu_down)" && bucket_by_bit gpu_down && echo
    (( ${#gpu_up[@]} ))   && echo "# ffn_up_shexp — up-projection (shared experts) — qbits: $(list_bits gpu_up)" && bucket_by_bit gpu_up && echo
    (( ${#gpu_gate[@]} )) && echo "# ffn_gate_shexp — gating network (shared experts) — qbits: $(list_bits gpu_gate)" && bucket_by_bit gpu_gate && echo
  fi

  # Single-expert FFN section: ffn_*_exps
  if (( ${#cpu_exps[@]} )); then
    echo "## CPU-friendly - MoE Per-expert Feed-Forward Network - ffn_*_exps"
    cpu_down=() cpu_up=() cpu_gate=()
    for l in "${cpu_exps[@]}"; do
      case "$l" in
        *ffn_down_*) cpu_down+=("$l") ;; 
        *ffn_up_*)   cpu_up+=("$l")   ;; 
        *ffn_gate_*) cpu_gate+=("$l") ;; 
      esac
    done
    (( ${#cpu_down[@]} )) && echo "# ffn_down_exps — down-projection (per-expert) — qbits: $(list_bits cpu_down)" && bucket_by_bit cpu_down && echo
    (( ${#cpu_up[@]} ))   && echo "# ffn_up_exps — up-projection (per-expert) — qbits: $(list_bits cpu_up)" && bucket_by_bit cpu_up && echo
    (( ${#cpu_gate[@]} )) && echo "# ffn_gate_exps — gating network (per-expert) — qbits: $(list_bits cpu_gate)" && bucket_by_bit cpu_gate && echo
  fi
}

extract_summaries() {
  awk '
    /^## Summary/ {
      if (seen++)        # if this isn’t the first Summary, print separator
        print ""
      print               # print the “## Summary…” line
      in_block = 1        # start capturing
      next
    }
    in_block && NF {      # if we’re in a block and the line is non-empty
      print               #   print it
      next
    }
    in_block && !NF {     # if we hit an empty line
      in_block = 0        #   end the block
    }
  ' 
}

#echo 'blk\.[0-3]\.attn_output\.weight=iq3_s' | expand_ranges
#exit

#echo "$custom" | grep -v '^#' | grep -v '^$' | shorten_regex_list | optimise_regex_list
#exit

#echo 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=f16' | expand_ranges
#exit

#echo "$custom" | grep -v '^#' | grep -v '^$' | sort -u
#exit

#echo 'blk\.(9|[0-2]1|[0-2]3|1[4-9]|2[0-2]|2[4-9]|3[0-2]|4[0-9]|3[4-9]|5[0-5]|58)\.attn_q_b\.weight=iq3_xxs_r4' | optimise_regex_list
#echo 'blk\.[0-3]\.attn_q_b\.weight=iq3_xxs' | optimise_regex_list
#echo 'blk\.(3|9|1[0-2]|1[4-9]|2[0-2]|2[4-9]|3[0-2]|4[0-9]|3[4-9]|5[0-5]|58)\.ffn_gate_shexp\.weight=iq3_s' | optimise_regex_list
#exit

#echo "$custom" | grep -v '^#' | grep -v '^$' | shorten_regex_list | optimise_regex_list | expand_ranges | sed -E '/=q[0-9]+_._r[0-9]+$/! s/(=q.*)/\U\1/' | sed -E '/=iq[0-9]+_._r[0-9]+$/! s/(=q.*)/\U\1/'

echo "$custom" | grep -v '^#' | grep -v '^$' | shorten_regex_list | optimise_regex_list | reorder_and_group

echo "$(cat <<< $custom | extract_summaries)"

# At script end: extract metrics and generate filename
# Safely join OUTPUTS (allow empty)
all="${OUTPUTS[*]:-}"

# Extract integer GPU GiB or default to 0
gpuGiB=$(printf "%s" "$all" | sed -E -n 's/^# GPU Total: ([0-9]+)\..*$/\1/p')
gpuGiB=${gpuGiB:-0}

# Extract integer CPU GiB or default to 0
cpuGiB=$(printf "%s" "$all" | sed -E -n 's/^# CPU Total: ([0-9]+)\..*$/\1/p')
cpuGiB=${cpuGiB:-0}

# Extract integer GPU+CPU GiB or default to 0
totalGiB=$(printf "%s" "$all" | sed -E -n 's/^# GPU\+CPU Total: ([0-9]+)\..*$/\1/p')
totalGiB=${totalGiB:-0}

# Extract BPW or default to 0
bpw=$(printf "%s" "$all" | sed -E -n 's/^# -Average BPW: ([0-9]+\.[0-9]+).*$/\1/p')
bpw=${bpw:-0}

# Extract SHA-256 and first 7 chars
gsha=$(printf "%s" "$all" | sed -E -n 's/^# - Script SHA-256: ([0-9a-f]{64})$/\1/p')
shaPart=${gsha:0:7}

# Extract full command block, concatenate lines, then first 7 chars
fullCmd=$(printf "%s\n" "$all" \
  | sed -E -n '1,/^# - Command used:/d; s/^# - Command used: //; s/\\$//g; p' \
  | tr -d '\\n')

# compute first 7 chars of SHA-256 of $fullCmd, if possible
if command -v _sha256sum >/dev/null 2>&1; then
  cmdPart=$(printf '%s' "$fullCmd" | _sha256sum | cut -c1-7)
else
  # fallback: just take the first 7 chars directly
  cmdPart='NA'
fi

# Add PPL if set
ppl_part=""
if [[ -n "$PPL" ]]; then
  ppl_part="${PPL}ppl."
fi

# Build dynamic filename
whoami=$(whoami)
filename="${whoami^^}-${bpw}bpw-${ppl_part}${totalGiB}GB-GGUF_${gpuGiB}GB-GPU_${cpuGiB}GB-CPU.${shaPart}_${cmdPart}.recipe"

echo 
echo "## THE END!"

# Prepend model name if set
if [[ -n "$MODEL_NAME" ]]; then
  filename_prefix="${MODEL_NAME}."
  filename="${filename_prefix}$filename"
fi

# Output to file or show intended filename
if [[ "$NO_FILE" -eq 0 ]]; then
  printf "%s\n" "${OUTPUTS[@]:-}" > "$filename"
  echo "# Saved recipe to file: $filename"
else
  echo "# --no-file: would have written to $filename"
fi
