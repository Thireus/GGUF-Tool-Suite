#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** model_tensor_sizes.py is a tool that helps identify which **#
#** tensors are the heaviest, thus to be benchmarked.         **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Oct-23-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.                 ₖₗD ₐₗₗ ₜₕₑ 𝓌ₐᵧ! **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

"""
sum_recipe_sizes.py — compute total tensor sizes per regex from a recipe and map file.

Usage:
  ./sum_recipe_sizes.py [--bytes] [--sort] RECIPE_FILE MAP_FILE

Options:
  --bytes       Show raw byte counts instead of human-readable units.
  --sort        Output only regex lines sorted by total size (heaviest -> lightest).
  -h, --help    Show this help and exit.

Example:
  chmod +x sum_recipe_sizes.py
  ./sum_recipe_sizes.py my.recipe my.map > my.recipe.sized
  ./sum_recipe_sizes.py --bytes --sort my.recipe my.map > my.recipe.sorted.sized
"""
from __future__ import annotations
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and prepend total tensor sizes per regex from a recipe and map file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--bytes", action="store_true", help="Show raw byte counts instead of human-readable units")
    p.add_argument("--sort", action="store_true", help="Output only regex lines sorted by total size (heaviest -> lightest)")
    p.add_argument("recipe_file", help=".recipe file path")
    p.add_argument("map_file", help=".map file path")
    return p.parse_args()

def human_readable(nbytes: int) -> str:
    """Return human-readable size (B, KB, MB, GB, TB)."""
    if nbytes < 1024:
        return f"{nbytes}B"
    size = float(nbytes)
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
    return f"{nbytes}B"

def parse_map_file(map_path: Path) -> Dict[str, int]:
    """Parse the .map file returning dict: tensor_name -> bytes."""
    tensors: Dict[str, int] = {}
    text = map_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        parts = line.split(":")
        # Expect at least 3 parts so parts[2] is tensor name
        if len(parts) < 3:
            continue
        name = parts[2].strip()
        m = re.search(r"bytes=(\d+)", line)
        if m:
            try:
                tensors[name] = int(m.group(1))
            except ValueError:
                # skip malformed numbers
                continue
    return tensors

def compile_regex(pattern: str):
    """Compile the pattern into a regex. If compile fails, return None."""
    try:
        return re.compile(pattern)
    except re.error as e:
        sys.stderr.write(f"Warning: invalid regex pattern: {pattern!r} -> {e}\n")
        return None

def total_bytes_for_pattern(pattern: str, tensors: Dict[str, int]) -> int:
    """Sum bytes of all tensors whose name matches the regex pattern."""
    regex = compile_regex(pattern)
    if regex is None:
        return 0
    # match anywhere (the patterns likely contain ^/$ anchors already)
    total = 0
    for name, size in tensors.items():
        if regex.search(name):
            total += size
    return total

def extract_regex_from_line(line: str) -> str:
    """Return the regex portion from a recipe line (part before first '=' or whitespace), trimmed."""
    if "=" in line:
        left = line.split("=", 1)[0]
    else:
        left = line
    return left.strip().split(" ", 1)[0].strip()

def read_recipe_lines(recipe_path: Path) -> List[str]:
    return recipe_path.read_text(encoding="utf-8", errors="ignore").splitlines()

def main():
    args = parse_args()
    recipe_path = Path(args.recipe_file)
    map_path = Path(args.map_file)

    if not recipe_path.exists():
        sys.exit(f"Error: recipe file '{recipe_path}' not found.")
    if not map_path.exists():
        sys.exit(f"Error: map file '{map_path}' not found.")

    tensors = parse_map_file(map_path)
    if not tensors:
        sys.stderr.write("Warning: no tensors parsed from map file (or map file missing bytes= entries).\n")

    recipe_lines = read_recipe_lines(recipe_path)

    # Collect regex lines and their totals
    regex_entries: List[Tuple[str, str, int]] = []
    # each tuple: (original_line, regex_pattern, total_bytes)
    for line in recipe_lines:
        stripped = line.strip()
        # skip comments/blank when gathering regex entries; we still want to preserve them when not --sort
        if not stripped or stripped.startswith("#"):
            continue
        pattern = extract_regex_from_line(line)
        if not pattern:
            continue
        total = total_bytes_for_pattern(pattern, tensors)
        regex_entries.append((line, pattern, total))

    # If --sort is requested, output only regex lines sorted by total bytes (desc)
    if args.sort:
        # sort by total desc, stable
        regex_entries.sort(key=lambda x: x[2], reverse=True)
        for orig_line, _, total in regex_entries:
            size_str = f"{total} B" if args.bytes else human_readable(total)
            print(f"{size_str:>10}  {orig_line}")
        return

    # Otherwise, preserve original file order and prepend sizes on regex lines; keep comments/blank lines
    # Create a lookup of pattern -> total for fast access (pattern is the lhs portion used earlier)
    pattern_to_total: Dict[str, int] = {p: t for (_, p, t) in ((e[0], e[1], e[2]) for e in regex_entries)}
    # Note: if the same exact pattern appears multiple times, it's fine — we use the same total each time.

    for line in recipe_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            print(line)
            continue
        pattern = extract_regex_from_line(line)
        if not pattern:
            print(line)
            continue
        total = pattern_to_total.get(pattern, 0)
        size_str = f"{total} B" if args.bytes else human_readable(total)
        print(f"{size_str:>10}  {line}")

if __name__ == "__main__":
    main()
