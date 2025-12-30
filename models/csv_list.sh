#!/bin/bash

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
  -printf "%P\n" > csv_list.txt