#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** model_tensor_sizes.py is a tool that helps identify which **#
#** tensors are the heaviest, thus to be benchmarked.         **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-29-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.                 ₖₗD ₐₗₗ ₜₕₑ 𝓌ₐᵧ! **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

"""
model_tensor_sizes.py — compute total tensor sizes per regex from a recipe and map file.

Usage:
  ./model_tensor_sizes.py [--bytes] [--sort] RECIPE_FILE MAP_FILE

Options:
  --bytes       Show raw byte counts instead of human-readable units.
  --sort        Output only regex lines sorted by total size (heaviest -> lightest).
  -h, --help    Show this help and exit.

Example:
  chmod +x model_tensor_sizes.py
  ./model_tensor_sizes.py my.recipe my.map > my.recipe.sized
  ./model_tensor_sizes.py --bytes --sort my.recipe my.map > my.recipe.sorted.sized
"""
from __future__ import annotations
import sys
import re
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Base bytes-per-weight table used to detect when tensors.map bytes are only the
# base quantized size and need the additional per-row scale bump applied.
BPW_TABLE: Dict[str, float] = {
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "Q8_0_R8": 8.5,
    "Q8_0": 8.5,
    "Q8_K_R8": 8.0625,
    "Q8_KV": 8,
    "F8": 8,
    "IQ6_K": 6.625,
    "Q6_K_R4": 6.5625,
    "Q6_K": 6.5625,
    "Q6_0_R4": 6.5,
    "Q6_0": 6.5,
    "Q5_1": 6,
    "Q5_K_R4": 5.5,
    "Q5_K": 5.5,
    "Q5_0_R4": 5.5,
    "Q5_0": 5.5,
    "IQ5_K_R4": 5.5,
    "IQ5_K": 5.5,
    "IQ5_KS_R4": 5.25,
    "IQ5_KS": 5.25,
    "Q4_1": 5,
    "Q4_K_R4": 4.5,
    "Q4_K": 4.5,
    "Q4_0_R8": 4.5,
    "Q4_0": 4.5,
    "IQ4_NL_R4": 4.5,
    "IQ4_NL": 4.5,
    "IQ4_K_R4": 4.5,
    "IQ4_K": 4.5,
    "IQ4_XS_R8": 4.25,
    "IQ4_XS": 4.25,
    "IQ4_KS_R4": 4.25,
    "IQ4_KS": 4.25,
    "IQ4_KT": 4,
    "IQ4_KSS": 4,
    "IQ3_KL": 4,
    "IQ3_M": 3.66,
    "Q3_K_R4": 3.4375,
    "Q3_K": 3.4375,
    "IQ3_S_R4": 3.4375,
    "IQ3_S": 3.4375,
    "IQ3_K_R4": 3.4375,
    "IQ3_K": 3.4375,
    "IQ3_XS": 3.3,
    "IQ3_KS": 3.1875,
    "IQ3_KT": 3.125,
    "IQ3_XXS_R4": 3.0625,
    "IQ3_XXS": 3.0625,
    "IQ2_M_R4": 2.7,
    "IQ2_M": 2.7,
    "IQ2_KL": 2.6875,
    "Q2_K_R4": 2.625,
    "Q2_K": 2.625,
    "IQ2_S": 2.5625,
    "IQ2_K_R4": 2.375,
    "IQ2_K": 2.375,
    "IQ2_XS_R4": 2.3125,
    "IQ2_XS": 2.3125,
    "IQ2_KS": 2.1875,
    "IQ2_KT": 2.125,
    "IQ2_XXS_R4": 2.0625,
    "IQ2_XXS": 2.0625,
    "IQ2_BN_R4": 2,
    "IQ2_BN": 2,
    "IQ1_M_R4": 1.75,
    "IQ1_M": 1.75,
    "IQ1_KT": 1.75,
    "IQ1_BN": 1.625,
    "IQ1_S": 1.5625,
    "IQ1_S_R4": 1.5,
}

# Additional per-row scale bump table. When a tensor dtype belongs here and the
# tensors.map bytes still equal the base elements*bpw/8 size, we add the scale
# overhead computed from the trailing dimensions.
ADDITIONAL_SCALE_FACTOR_TABLE: Dict[str, int] = {
    "IQ1_BN": 2,
    "IQ1_KT": 4,
    "IQ2_BN": 4,
    "IQ2_BN_R4": 4,
    "IQ2_KL": 2,
    "IQ2_KS": 2,
    "IQ2_KT": 4,
    "IQ3_KS": 2,
    "IQ3_KT": 4,
    "IQ4_KS": 4,
    "IQ4_KSS": 4,
    "IQ4_KS_R4": 4,
    "IQ4_KT": 4,
    "IQ5_KS": 4,
    "IQ5_KS_R4": 4,
    "Q8_KV": 8,
    "IQ1_S_R4": 2,
    "IQ1_M_R4": 2,
    "Q8_KV_R8": 4,
}

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

def parse_shape_from_line(line: str) -> Optional[List[int]]:
    """
    Parse shape=(...) from a tensors.map line.
    Returns a list of ints or None if no shape is present.
    """
    m = re.search(r"\bshape=\(([^)]*)\)", line, flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    if raw == "":
        return []
    dims: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            dims.append(int(tok))
        except ValueError:
            dims.append(0)
    return dims

def adjusted_bytes_for_tensor(dtype: Optional[str],
                              elements: Optional[int],
                              bytes_: Optional[int],
                              shape: Optional[List[int]]) -> Optional[int]:
    """
    If the dtype is one that can carry an additional per-row scale bump, and the
    reported bytes still match the base elements*bpw/8 value, add the bump.
    """
    if dtype is None or elements is None or bytes_ is None:
        return bytes_

    qtype_upper = dtype.upper()
    scale_factor = ADDITIONAL_SCALE_FACTOR_TABLE.get(qtype_upper)
    bpw = BPW_TABLE.get(qtype_upper)

    if scale_factor is None or bpw is None:
        return bytes_

    base_bytes = (elements * bpw) / 8.0
    if not math.isclose(float(bytes_), base_bytes, rel_tol=0.0, abs_tol=1e-9):
        return bytes_

    tail_product = 1
    if shape and len(shape) > 1:
        for dim in shape[1:]:
            tail_product *= int(dim)

    return int(bytes_ + (scale_factor * tail_product))

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
        dtype_m = re.search(r"\bdtype=([^:]+)", line, flags=re.IGNORECASE)
        elems_m = re.search(r"\belements=(\d+)", line, flags=re.IGNORECASE)
        bytes_m = re.search(r"\bbytes=(\d+)", line, flags=re.IGNORECASE)
        shape = parse_shape_from_line(line)

        dtype = dtype_m.group(1).strip() if dtype_m else None
        elements = int(elems_m.group(1)) if elems_m else None
        bytes_ = int(bytes_m.group(1)) if bytes_m else None

        adjusted = adjusted_bytes_for_tensor(dtype, elements, bytes_, shape)
        if adjusted is not None:
            tensors[name] = adjusted
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
