#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** csv_list.sh is a helper tool that collects the calibrated **#
#** data files of models listed in the models directory.      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jan-25-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.          ᵣₑₚᵣₒₘₚₜ ᵤₙₜᵢₗ ₛₐₜᵢₛ𝒻ᵢₑ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Use gfind on macOS, regular find on Linux
if [[ "$(uname)" == "Darwin" ]] && command -v gfind &> /dev/null; then
    FIND_CMD="gfind"
else
    FIND_CMD="find"
fi

# Find CSV files and save to csv_list.txt
$FIND_CMD . -name "*.csv" \
  -not -path "*/maps/*" \
  -not -path "*/outdated_results/*" \
  -not -path "*/benchmark_logs/*" \
  -not -path "*/benchmark_files/*" \
  -not -path "*/group0_logs/*" \
  -not -path "*/group0_files/*" \
  -printf "%P\n" | grep -v 'bpw_kld_results.csv' > csv_list.txt
  