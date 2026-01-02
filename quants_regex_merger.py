#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** model_tensor_sizes.py is a tool that helps identify which **#
#** tensors are the heaviest, thus to be benchmarked.         **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jan-02-2026 -------------------- **#
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
#** Copyright © 2025 - Thireus.              Bₒₒₜₛₜᵣₐₚₚᵢₙ𝓰 ₛₖᵧₙₑₜ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# -*- coding: utf-8 -*-
"""
quants_regex_merger.py
A Python3 port of quants_regex_merger.sh (Thireus' GGUF Tool Suite)

Ported behavior notes:
- Attempts to follow original bash behavior closely (formatting, grouping, filename generation).
- Debug toggled via environment variable DEBUG=1.
- Reading from stdin: if stdin is piped, it's used; otherwise a built-in default snippet is used.
- Produces the same major output groups and attempts to keep regex escaping identical to the bash script.
- Saves to a dynamically-generated filename unless --no-file is used.

Usage: see --help
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import hashlib
import getpass
import itertools
from typing import List, Dict, Tuple

DEBUG = os.environ.get("DEBUG", "0") == "1"

def _debug(*args):
    if not DEBUG:
        return
    print("[DEBUG]", *args, file=sys.stderr)

# Capture outputs in list and also print them (like the bash script's echo override)
OUTPUTS: List[str] = []

def out(msg: str = ""):
    """Append to OUTPUTS and print to stdout (with newline)."""
    # ensure not to append trailing newline twice
    if msg.endswith("\n"):
        msg = msg[:-1]
    OUTPUTS.append(msg)
    print(msg)

# -----------------------
# sha256 helper detection
# -----------------------
# In Python we will compute sha256 via hashlib where needed
# but we also preserve the script logic of extracting existing hashes from outputs.

# -----------------------
# Utility: build_range_regex
# -----------------------
def build_range_regex(S: int, E: int) -> str:
    """
        Re-implement build_range_regex from the bash script.
        Returns a joined alternation (with |) describing numbers/ranges between S and E.
        """
    _debug("    build_range_regex S=%s E=%s", S, E)
    parts: List[str] = []
    full_decades: List[int] = []
    partial: List[str] = []

    # (1) single digits
    if S <= 9:
        hi = E if E < 9 else 9
        if S == 0 and hi == 9:
            parts.append("[0-9]")
            _debug("      add [0-9]")
        elif S == hi:
            parts.append(str(S))
            _debug("      add %s", S)
        else:
            parts.append(f"[{S}-{hi}]")
            _debug("      add [{S}-{hi}]", S, hi)

    # (2) decades 10–99
    if E >= 10:
        low2 = 10 if S < 10 else S
        start_d = low2 // 10
        end_d = E // 10
        _debug("      decades from %s to %s", start_d, end_d)

        for d in range(start_d, end_d + 1):
            u_lo = (low2 % 10) if d == start_d else 0
            u_hi = (E % 10) if d == end_d else 9
            if u_lo == 0 and u_hi == 9:
                full_decades.append(d)
                _debug("        full decade %s", d)
            else:
                if u_lo == u_hi:
                    partial.append(f"{d}{u_lo}")
                    _debug("        partial single %s%s", d, u_lo)
                else:
                    partial.append(f"{d}[{u_lo}-{u_hi}]")
                    _debug("        partial range %s[%s-%s]", d, u_lo, u_hi)

        # collapse full_decades runs
        if full_decades:
            sorted_fd = sorted(set(full_decades))
            run_start = run_prev = sorted_fd[0]
            for d in sorted_fd[1:]:
                if d == run_prev + 1:
                    run_prev = d
                else:
                    if run_start == run_prev:
                        parts.append(f"{run_start}[0-9]")
                        _debug("          flush %s[0-9]", run_start)
                    else:
                        parts.append(f"[{run_start}-{run_prev}][0-9]")
                        _debug("          flush [%s-%s][0-9]", run_start, run_prev)
                    run_start = run_prev = d
            # final flush
            if run_start == run_prev:
                parts.append(f"{run_start}[0-9]")
                _debug("          final %s[0-9]", run_start)
            else:
                parts.append(f"[{run_start}-{run_prev}][0-9]")
                _debug("          final [%s-%s][0-9]", run_start, run_prev)

        # append partial pieces
        for p in partial:
            parts.append(p)
            _debug("        append partial %s", p)

    # (3) fallback for E>99: enumerate every number
    if E > 99:
        parts = [str(i) for i in range(S, E + 1)]

    # (4) safe join
    joined = "|".join(parts)
    _debug("    build_range_regex returns %s", joined)
    return joined

# --------------------------------------------------------------------------------
# shorten_regex_list(): collapse consecutive blk.N (and similar) lines
# --------------------------------------------------------------------------------
def shorten_regex_list(lines: List[str]) -> List[str]:
    groups: Dict[str, List[int]] = {}
    out_lines: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Patterns: optional leading ^, prefix then \.<number>\.<suffix>
        m = re.match(r'^\^?(blk)\\.([0-9]+)\\\.(.+)\$?$', line)
        if not m:
            m = re.match(r'^\^?(mm)\\.([0-9]+)\\\.(.+)\$?$', line)
        if not m:
            m = re.match(r'^\^?(v\\.blk)\\.([0-9]+)\\\.(.+)\$?$', line)
        if not m:
            m = re.match(r'^\^?(v\\.deepstack)\\.([0-9]+)\\\.(.+)\$?$', line)
        if m:
            prefix = m.group(1)
            block_num = int(m.group(2))
            suffix = m.group(3)
            key = f"{prefix}_{suffix}"
            groups.setdefault(key, []).append(block_num)
            _debug("Bucket %s → prefix %s suffix %s", block_num, prefix, suffix)
        else:
            # Non-blk (mm, v, deepstack) line: output immediately
            out_lines.append(line)

    # collect unique prefixes
    prefixes_seen = {k.split("_", 1)[0] for k in groups.keys()}

    # iterate prefixes then suffixes
    for prefix in sorted(prefixes_seen):
        _debug("Processing prefix: %s", prefix)
        for key in sorted(groups.keys()):
            if not key.startswith(prefix + "_"):
                continue
            _debug("Processing key: %s", key)
            # compute suffix by slicing off prefix + the underscore
            suffix = key[len(prefix) + 1 :]
            nums = sorted(set(groups[key]))
            _debug("Processing suffix: %s (value: %s)", suffix, " ".join(map(str, nums)))

            if not nums:
                continue

            # Break into consecutive runs
            runs: List[Tuple[int, int]] = []
            run_start = run_prev = nums[0]
            for num in nums[1:]:
                if num == run_prev + 1:
                    run_prev = num
                else:
                    runs.append((run_start, run_prev))
                    run_start = run_prev = num
            runs.append((run_start, run_prev))

            # Build the regex parts for the runs
            parts: List[str] = []
            for s, e in runs:
                if s == e:
                    parts.append(str(s))
                    _debug("Run: single number %s", s)
                else:
                    part = build_range_regex(s, e)
                    parts.append(part)
                    _debug("Run: consecutive %s to %s -> %s", s, e, part)

            block_regex = "|".join(parts)

            # If block_regex contains '|', wrap in non-capturing group? original wrapped in parentheses
            if "|" in block_regex:
                block_regex = f"({block_regex})"
                _debug("Wrapped block_regex: %s", block_regex)

            out_lines.append(f"^{prefix}\\.{block_regex}\\.{suffix}")

    return out_lines

# -----------------------------------------------------------------------------------------
# optimise_regex_list(): merges consecutive bracket-prefix/suffix pieces
# -----------------------------------------------------------------------------------------
def optimise_regex_list(lines: List[str]) -> List[str]:
    out_lines: List[str] = []
    prefixes = ["blk", "mm", "v\\.blk", "v\\.deepstack"]

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Determine whether the line needs the complex optimisation:
        if ("|" in line) and ("[" in line or "]" in line):
            processed = False
            for prefix in prefixes:
                lit1 = f"^{prefix}\\."
                lit2 = f"{prefix}\\."
                if line.startswith(lit1 + "(") or line.startswith(lit2 + "("):
                    _debug("Original line: %s", line)
                    # strip leading 'prefix\.(' (either with caret or without)
                    if line.startswith(lit1 + "("):
                        pr = line[len(lit1) + 1 :]
                    elif line.startswith(lit2 + "("):
                        pr = line[len(lit2) + 1 :]
                    else:
                        pr = line

                    # extract up to ')\.' as inner
                    idx = pr.find(")\\.")
                    if idx != -1:
                        inner = pr[:idx]
                        suffix = pr[idx + 3 :]
                    else:
                        # fallback - try ') .' patterns or fail gracefully
                        # try to locate ')'
                        idx2 = pr.find(")")
                        if idx2 != -1 and idx2 + 1 < len(pr) and pr[idx2 + 1] == "\\" and pr[idx2 + 2] == ".":
                            inner = pr[:idx2]
                            suffix = pr[idx2 + 3 :]
                        else:
                            # can't parse: skip and output normalized '^line'
                            out_lines.append("^" + line.lstrip("^"))
                            processed = True
                            break

                    _debug("  pr         = '%s'", pr)
                    _debug("  inner      = '%s'", inner)
                    _debug("  suffix     = '%s'", suffix)

                    parts = inner.split("|")
                    _debug("  parts      = %s", parts)

                    plain: List[str] = []
                    extras: List[str] = []
                    by_suffix: Dict[str, List[int]] = {}
                    by_prefix: Dict[str, List[int]] = {}

                    # classify each part
                    for p in parts:
                        p = p.strip()
                        m1 = re.match(r"^([0-9]+)(\[[0-9]+-[0-9]+\])$", p)
                        m2 = re.match(r"^(\[[0-9]+-[0-9]+\])([0-9]+)$", p)
                        if m1:
                            num = int(m1.group(1))
                            su = m1.group(2)
                            by_suffix.setdefault(su, []).append(num)
                            _debug("    by_suffix[%s] += %s", su, num)
                        elif m2:
                            prf = m2.group(1)
                            num = int(m2.group(2))
                            by_prefix.setdefault(prf, []).append(num)
                            _debug("    by_prefix[%s] += %s", prf, num)
                        else:
                            plain.append(p)
                            _debug("    plain += %s", p)

                    # merge by_suffix
                    for su, nums in by_suffix.items():
                        nums_sorted = sorted(set(nums))
                        if not nums_sorted:
                            _debug("  no entries for suffix %s, skipping", su)
                            continue
                        _debug("  merging by_suffix[%s]: %s", su, nums_sorted)
                        start = prev = nums_sorted[0]
                        for n in nums_sorted[1:]:
                            if n == prev + 1:
                                prev = n
                            else:
                                if start < prev:
                                    extras.append(f"[{start}-{prev}]{su}")
                                    _debug("    extras += [%s-%s]%s", start, prev, su)
                                else:
                                    extras.append(f"{start}{su}")
                                    _debug("    extras += %s%s", start, su)
                                start = prev = n
                        # flush last
                        if start < prev:
                            extras.append(f"[{start}-{prev}]{su}")
                            _debug("    extras += [%s-%s]%s", start, prev, su)
                        else:
                            extras.append(f"{start}{su}")
                            _debug("    extras += %s%s", start, su)

                    # merge by_prefix
                    for prf, nums in by_prefix.items():
                        nums_sorted = sorted(set(nums))
                        if not nums_sorted:
                            _debug("    no entries for prefix %s, skipping", prf)
                            continue
                        _debug("  merging by_prefix[%s]: %s", prf, nums_sorted)
                        start = prev = nums_sorted[0]
                        for n in nums_sorted[1:]:
                            if n == prev + 1:
                                prev = n
                            else:
                                if start < prev:
                                    extras.append(f"{prf}[{start}-{prev}]")
                                    _debug("    extras += %s[%s-%s]", prf, start, prev)
                                else:
                                    extras.append(f"{prf}{start}")
                                    _debug("    extras += %s%s", prf, start)
                                start = prev = n
                        # flush last
                        if start < prev:
                            extras.append(f"{prf}[{start}-{prev}]")
                            _debug("    extras += %s[%s-%s]", prf, start, prev)
                        else:
                            extras.append(f"{prf}{start}")
                            _debug("    extras += %s%s", prf, start)

                    final_parts = plain + extras
                    _debug("  final_parts = %s", final_parts)

                    # sorting logic (compute numeric key)
                    def compute_key(s: str) -> int:
                        key = 0
                        rest = s
                        # parse sequence of chunks
                        while rest:
                            m_num = re.match(r"^([0-9]+)(.*)$", rest)
                            m_range = re.match(r"^\[([0-9]+)-([0-9]+)\](.*)$", rest)
                            if m_num:
                                chunk = m_num.group(1)
                                rest = m_num.group(2)
                            elif m_range:
                                chunk = m_range.group(1)
                                rest = m_range.group(3)
                            else:
                                # non-numeric trailing content: push to very high key
                                return 999999999999
                            digits = len(chunk)
                            multiplier = 10 ** digits
                            key = key * multiplier + int(chunk)
                        return key

                    sorted_stream = sorted(
                        ((compute_key(item), item) for item in final_parts),
                        key=lambda x: (x[0], x[1]),
                    )

                    final_sorted = [item for (_, item) in sorted_stream]
                    _debug("  final_parts sorted = %s", final_sorted)

                    # re-assemble
                    assembled = "|".join(final_sorted)
                    out_line = f"^{prefix}\\.({assembled})\\.{suffix}"
                    out_lines.append(out_line)
                    processed = True
                    break  # break prefix loop
            if not processed:
                # no matching prefix or couldn't process - normalise and prefix with ^
                out_lines.append("^" + line.lstrip("^"))
        else:
            # no '|'/'[' combination that needs special handling
            out_lines.append("^" + line.lstrip("^"))

    return out_lines

# ---------------------------------------------------------------------------------------------
# expand_ranges(): separate regex range entries if not supported by llama-quantize
# ---------------------------------------------------------------------------------------------
def expand_ranges(lines: List[str]) -> List[str]:
    expanded_lines: List[str] = []

    for input_line in lines:
        input_line = input_line.rstrip()
        if not input_line:
            continue

        if "(" in input_line and ")" in input_line:
            # prefix: before first '('
            prefix = input_line.split("(", 1)[0]
            # body: between first '(' and last ')'
            after_first = input_line.split("(", 1)[1]
            body = after_first.rsplit(")", 1)[0]
            # suffix: after last ')'
            suffix = input_line.rsplit(")", 1)[1]
        else:
            prefix = ""
            body = input_line
            suffix = ""

        # Convert [a-b] to {a..b}
        body_expanded = re.sub(r"\[([0-9]+)-([0-9]+)\]", r"{\1..\2}", body)

        # split on |
        parts = body_expanded.split("|")
        for part in parts:
            part = part.strip()
            if "{" in part and ".." in part and "}" in part:
                # Expand braces; support multiple {a..b} with cartesian product
                # Use re.split to capture groups of start/end
                segments = re.split(r"\{([0-9]+)\.\.([0-9]+)\}", part)
                # segments: [text0, start1, end1, text1, start2, end2, text2, ...]
                # Build list of choices
                choices = [""]
                i = 0
                while i < len(segments):
                    seg = segments[i]
                    if i % 3 == 0:
                        # literal text
                        choices = [c + seg for c in choices]
                        i += 1
                    else:
                        # seg is start, next is end
                        start = int(segments[i])
                        end = int(segments[i + 1])
                        # generate replacements
                        new_choices = []
                        for val in range(start, end + 1):
                            for c in choices:
                                new_choices.append(c + str(val))
                        choices = new_choices
                        i += 2
                for e in choices:
                    expanded_lines.append(f"{prefix}{e}{suffix}")
            else:
                expanded_lines.append(f"{prefix}{part}{suffix}")

    return expanded_lines

# -------------------------------------------------------------------
# reorder_and_group(): reorganise and group the final output
# -------------------------------------------------------------------
def reorder_and_group(lines: List[str], model_name: str = "", model_link: str = "") -> List[str]:
    # initialize arrays
    general: List[str] = []
    gpu_shexp: List[str] = []
    cpu_exps: List[str] = []
    others: List[str] = []

    # Read and classify lines
    for line in lines:
        if not line or line.strip().startswith("#"):
            continue
        if re.match(r'^\^output\\.*', line) or re.match(r'^\^output_norm\\.*', line) or re.match(r'^\^token_embd\\.*', line) \
           or line.startswith("output\\.") or line.startswith("output_norm\\.") or line.startswith("token_embd\\."):
            general.append(line)
        elif "shexp" in line:
            gpu_shexp.append(line)
        elif "exps" in line:
            cpu_exps.append(line)
        else:
            others.append(line)

    # Helper: bucket lines by first digit in quant, sort descending
    def bucket_by_bit(input_lines: List[str]) -> List[str]:
        flat = []
        for l in input_lines:
            quant = l.split("=", 1)[-1]
            m = re.search(r"([0-9])", quant)
            bit = int(m.group(1)) if m else 0
            flat.append((bit, l))
        # sort by bit desc then keep original order within same bit
        flat.sort(key=lambda x: (-x[0], x[1]))
        return [item for (_, item) in flat]

    # Helper: compute unique sorted quant bits for a group
    def list_bits(input_lines: List[str]) -> str:
        bits = []
        for l in input_lines:
            q = l.split("=", 1)[-1]
            m = re.search(r"([0-9]+)", q)
            if m:
                bits.append(int(m.group(1)))
        # unique sorted descending
        if not bits:
            return "0"
        uniq = sorted(set(bits), reverse=True)
        return " ".join(str(b) for b in uniq)

    out_lines: List[str] = []

    # --- Output sections ---
    out_lines.append("## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/")
    if model_name:
        out_lines.append(f"# Model name: {os.path.basename(model_name)}")
    if model_link:
        out_lines.append(f"# Link to the original model: {model_link}")
    out_lines.append("")

    # Model head & embeddings
    if general:
        out_lines.append(f"## Model head & embeddings — qbits: {list_bits(general)}")
        out_lines.extend(general)
        out_lines.append("")

    # Special attention weights
    attn_special = []
    remaining = []
    for l in others:
        if "attn_k_b" in l:
            attn_special.append(l)
        else:
            remaining.append(l)
    others = remaining

    if attn_special:
        out_lines.append("## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: " + list_bits(attn_special))
        out_lines.extend(attn_special)
        out_lines.append("")

    # Multi-headed attention parameters
    attn_group = []
    remaining = []
    for l in others:
        if re.search(r"attn_.*", l) and "attn_k_b" not in l:
            attn_group.append(l)
        else:
            remaining.append(l)
    others = remaining

    if attn_group:
        out_lines.append("## Multi-headed attention parameters — qbits: " + list_bits(attn_group))
        out_lines.extend(attn_group)
        out_lines.append("")

    # Dense Feed-Forward Network weights (main up/down projections + dense gates for blk.[0-2])
    ffn_raw = []
    remaining = []
    for l in others:
        if "exps" in l or "shexp" in l:
            # handled elsewhere
            remaining.append(l)
            continue
        if "ffn_down" in l or "ffn_up" in l:
            ffn_raw.append(l)
            continue
        # dense gate weights for blocks 0..2 (literal substring match)
        if l.startswith("^blk\\.[0-2].ffn_gate") or "^v\\.blk\\.[0-2]" in l and "ffn_gate" in l:
            ffn_raw.append(l)
            continue
        # detect literal substring '^blk\.[0-2]\.ffn_gate' anywhere
        if "^blk\\.[0-2]\\.ffn_gate" in l or "^v\\.blk\\.[0-2]\\.ffn_gate" in l:
            ffn_raw.append(l)
            continue
        remaining.append(l)
    others = remaining

    if ffn_raw:
        out_lines.append("## Dense Feed-Forward Network weights — qbits: " + list_bits(ffn_raw))
        out_lines.extend(ffn_raw)
        out_lines.append("")

    ln_post = []
    embeddings = []
    deepstack = []
    nextn = []
    moe_gating = []
    gate_raw = []
    misc = []

    for l in others:
        if any(x in l for x in ("post_ln", "ln1", "ln2")):
            ln_post.append(l)
        elif any(x in l for x in ("patch_embd", "position_embd")):
            embeddings.append(l)
        elif "deepstack" in l:
            deepstack.append(l)
        elif "nextn" in l:
            nextn.append(l)
        elif "ffn_gate_inp" in l or "exp_probs_b" in l:
            moe_gating.append(l)
        elif "ffn_gate" in l:
            gate_raw.append(l)
        else:
            misc.append(l)

    if ln_post:
        out_lines.append("## LayerNorm / Post-LN parameters — qbits: " + list_bits(ln_post))
        out_lines.extend(ln_post)
        out_lines.append("")

    if embeddings:
        out_lines.append("## Embeddings & positional encodings — qbits: " + list_bits(embeddings))
        out_lines.extend(embeddings)
        out_lines.append("")

    if deepstack:
        out_lines.append("## Deepstack modules — qbits: " + list_bits(deepstack))
        out_lines.extend(deepstack)
        out_lines.append("")

    if nextn:
        out_lines.append("## NextN tensors — qbits: " + list_bits(nextn))
        out_lines.extend(nextn)
        out_lines.append("")

    if moe_gating:
        out_lines.append("## MoE Gating & Routing — qbits: " + list_bits(moe_gating))
        out_lines.extend(moe_gating)
        out_lines.append("")

    if gate_raw:
        out_lines.append("## Gating network — qbits: " + list_bits(gate_raw))
        out_lines.extend(gate_raw)
        out_lines.append("")

    if misc:
        out_lines.append("## Misc / Other tensors — qbits: " + list_bits(misc))
        out_lines.extend(misc)
        out_lines.append("")

    # Shared experts section: ffn_*_shexp
    if gpu_shexp:
        out_lines.append("## GPU-loaded - MoE Shared Experts Feed-Forward Network - ffn_*_shexp")
        gpu_down = [l for l in gpu_shexp if "ffn_down_" in l]
        gpu_up = [l for l in gpu_shexp if "ffn_up_" in l]
        gpu_gate = [l for l in gpu_shexp if "ffn_gate_" in l]
        if gpu_down:
            out_lines.append("# ffn_down_shexp — down-projection (shared experts) — qbits: " + list_bits(gpu_down))
            out_lines.extend(bucket_by_bit(gpu_down))
            out_lines.append("")
        if gpu_up:
            out_lines.append("# ffn_up_shexp — up-projection (shared experts) — qbits: " + list_bits(gpu_up))
            out_lines.extend(bucket_by_bit(gpu_up))
            out_lines.append("")
        if gpu_gate:
            out_lines.append("# ffn_gate_shexp — gating network (shared experts) — qbits: " + list_bits(gpu_gate))
            out_lines.extend(bucket_by_bit(gpu_gate))
            out_lines.append("")

    # Single-expert FFN section: ffn_*_exps
    if cpu_exps:
        out_lines.append("## CPU-friendly - MoE Per-expert Feed-Forward Network - ffn_*_exps")
        cpu_down = [l for l in cpu_exps if "ffn_down_" in l]
        cpu_up = [l for l in cpu_exps if "ffn_up_" in l]
        cpu_gate = [l for l in cpu_exps if "ffn_gate_" in l]
        if cpu_down:
            out_lines.append("# ffn_down_exps — down-projection (per-expert) — qbits: " + list_bits(cpu_down))
            out_lines.extend(bucket_by_bit(cpu_down))
            out_lines.append("")
        if cpu_up:
            out_lines.append("# ffn_up_exps — up-projection (per-expert) — qbits: " + list_bits(cpu_up))
            out_lines.extend(bucket_by_bit(cpu_up))
            out_lines.append("")
        if cpu_gate:
            out_lines.append("# ffn_gate_exps — gating network (per-expert) — qbits: " + list_bits(cpu_gate))
            out_lines.extend(bucket_by_bit(cpu_gate))
            out_lines.append("")

    return out_lines

# extract_summaries
def extract_summaries(custom_text: str) -> List[str]:
    lines = custom_text.splitlines()
    result: List[str] = []
    in_block = False
    seen = 0
    for ln in lines:
        if ln.startswith("## Summary"):
            if seen:
                result.append("")  # separator blank line if not first
            result.append(ln)
            in_block = True
            seen += 1
            continue
        if in_block:
            if ln.strip() == "":
                in_block = False
            else:
                result.append(ln)
    return result

# -----------------------
# MAIN
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combine tensor regex for llama-quantize consumption (Python port)."
    )
    parser.add_argument("--no-file", dest="no_file", action="store_true", help="Do not write output to a file; just print.")
    parser.add_argument("--model-name", dest="model_name", type=str, default="", help="Optional. Prepends NAME to the output filename.")
    parser.add_argument("--model-link", dest="model_link", type=str, default="", help="Optional. Link to original model.")
    parser.add_argument("--add-ppl", dest="add_ppl", type=str, default="", help="Optional. Adds VALUE_PPL right after username in the filename.")
    parser.add_argument("--input", dest="input_file", type=str, default="", help="Optional. Read custom regex block from FILE instead of stdin.")
    args = parser.parse_args()

    # Validate add_ppl numeric
    PPL = ""
    raw_ppl = args.add_ppl
    if raw_ppl:
        if not re.match(r"^[0-9]*\.?[0-9]+$", raw_ppl):
            print("Error: --add-ppl value must be numeric", file=sys.stderr)
            sys.exit(1)
        # Format PPL to exactly 4 decimal places
        try:
            PPL = f"{float(raw_ppl):.4f}"
        except Exception:
            PPL = f"{float(raw_ppl):.4f}"

    # If --input provided, read file; else if stdin piped, read; otherwise use default
    if args.input_file:
        _debug("Reading custom from input file: %s", args.input_file)
        try:
            with open(args.input_file, "r", encoding="utf-8") as fh:
                custom = fh.read()
        except Exception as e:
            print(f"Error: could not read input file {args.input_file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # If stdin piped, read; otherwise use default
        if not sys.stdin.isatty():
            _debug("Reading custom from stdin")
            custom = sys.stdin.read()
        else:
            custom = r"""
  ^blk\.15\.attn_norm\.weight$=f32
^blk\.15\.exp_probs_b\.bias$=f32
^blk\.15\.ffn_gate_inp\.weight$=f32
^blk\.15\.ffn_down_shexp\.weight$=bf16
^blk\.15\.ffn_gate_shexp\.weight$=bf16
^blk\.15\.ffn_up_shexp\.weight$=bf16
^blk\.15\.ffn_norm\.weight$=f32
^blk\.15\.attn_kv_a_norm\.weight$=f32
^blk\.15\.attn_kv_a_mqa\.weight$=bf16
^blk\.15\.attn_k_b\.weight$=bf16
^blk\.15\.attn_v_b\.weight$=bf16
^blk\.15\.attn_output\.weight$=bf16
^blk\.15\.attn_q_a_norm\.weight$=f32
^blk\.15\.attn_q_a\.weight$=bf16
^blk\.15\.attn_q_b\.weight$=bf16
^blk\.39\.attn_norm\.weight$=f32
^blk\.39\.exp_probs_b\.bias$=f32
^blk\.39\.ffn_gate_inp\.weight$=f32
^blk\.39\.ffn_down_shexp\.weight$=bf16
^blk\.39\.ffn_gate_shexp\.weight$=bf16
^blk\.39\.ffn_up_shexp\.weight$=bf16
^blk\.39\.ffn_norm\.weight$=f32
^blk\.39\.attn_kv_a_norm\.weight$=f32
^blk\.39\.attn_kv_a_mqa\.weight$=bf16
^blk\.39\.attn_k_b\.weight$=bf16
^blk\.39\.attn_v_b\.weight$=bf16
^blk\.39\.attn_output\.weight$=bf16
^blk\.39\.attn_q_a_norm\.weight$=f32
^blk\.39\.attn_q_a\.weight$=bf16
^blk\.39\.attn_q_b\.weight$=bf16
^blk\.58\.attn_norm\.weight$=f32
^blk\.58\.exp_probs_b\.bias$=f32
^blk\.58\.ffn_gate_inp\.weight$=f32
^blk\.58\.ffn_down_shexp\.weight$=bf16
^blk\.58\.ffn_gate_shexp\.weight$=bf16
^blk\.58\.ffn_up_shexp\.weight$=bf16
^blk\.58\.ffn_norm\.weight$=f32
^blk\.58\.attn_kv_a_norm\.weight$=f32
^blk\.58\.attn_kv_a_mqa\.weight$=bf16
^blk\.58\.attn_k_b\.weight$=bf16
^blk\.58\.attn_v_b\.weight$=bf16
^blk\.58\.attn_output\.weight$=bf16
^blk\.58\.attn_q_a_norm\.weight$=f32
^blk\.58\.attn_q_a\.weight$=bf16
^blk\.58\.attn_q_b\.weight$=bf16
^blk\.5\.attn_norm\.weight$=f32
^blk\.5\.exp_probs_b\.bias$=f32
^blk\.5\.ffn_gate_inp\.weight$=f32
^blk\.5\.ffn_down_shexp\.weight$=bf16
^blk\.5\.ffn_gate_shexp\.weight$=bf16
^blk\.5\.ffn_up_shexp\.weight$=bf16
^blk\.5\.ffn_norm\.weight$=f32
^blk\.5\.attn_kv_a_norm\.weight$=f32
^blk\.5\.attn_kv_a_mqa\.weight$=bf16
^blk\.5\.attn_k_b\.weight$=bf16
^blk\.5\.attn_v_b\.weight$=bf16
^blk\.5\.attn_output\.weight$=bf16
^blk\.5\.attn_q_a_norm\.weight$=f32
^blk\.5\.attn_q_a\.weight$=bf16
^blk\.5\.attn_q_b\.weight$=bf16
^blk\.46\.attn_norm\.weight$=f32
^blk\.46\.exp_probs_b\.bias$=f32
^blk\.46\.ffn_gate_inp\.weight$=f32
^blk\.46\.ffn_down_shexp\.weight$=bf16
^blk\.46\.ffn_gate_shexp\.weight$=bf16
^blk\.46\.ffn_up_shexp\.weight$=bf16
^blk\.46\.ffn_norm\.weight$=f32
^blk\.46\.attn_kv_a_norm\.weight$=f32
^blk\.46\.attn_kv_a_mqa\.weight$=bf16
^blk\.46\.attn_k_b\.weight$=bf16
^blk\.46\.attn_v_b\.weight$=bf16
^blk\.46\.attn_output\.weight$=bf16
^blk\.46\.attn_q_a_norm\.weight$=f32
^blk\.46\.attn_q_a\.weight$=bf16
^blk\.46\.attn_q_b\.weight$=bf16
^blk\.7\.attn_norm\.weight$=f32
^blk\.7\.exp_probs_b\.bias$=f32
^blk\.7\.ffn_gate_inp\.weight$=f32
^blk\.7\.ffn_down_shexp\.weight$=bf16
^blk\.7\.ffn_gate_shexp\.weight$=bf16
^blk\.7\.ffn_up_shexp\.weight$=bf16
^blk\.7\.ffn_norm\.weight$=f32
^blk\.7\.attn_kv_a_norm\.weight$=f32
^blk\.7\.attn_kv_a_mqa\.weight$=bf16
^blk\.7\.attn_k_b\.weight$=bf16
^blk\.7\.attn_v_b\.weight$=bf16
^blk\.7\.attn_output\.weight$=bf16
^blk\.7\.attn_q_a_norm\.weight$=f32
^blk\.7\.attn_q_a\.weight$=bf16
^blk\.7\.attn_q_b\.weight$=bf16
^blk\.11\.attn_norm\.weight$=f32
^blk\.11\.exp_probs_b\.bias$=f32
^blk\.11\.ffn_gate_inp\.weight$=f32
^blk\.11\.ffn_down_shexp\.weight$=bf16
^blk\.11\.ffn_gate_shexp\.weight$=bf16
^blk\.11\.ffn_up_shexp\.weight$=bf16
^blk\.11\.ffn_norm\.weight$=f32
^blk\.11\.attn_kv_a_norm\.weight$=f32
^blk\.11\.attn_kv_a_mqa\.weight$=bf16
^blk\.11\.attn_k_b\.weight$=bf16
^blk\.11\.attn_v_b\.weight$=bf16
^blk\.11\.attn_output\.weight$=bf16
^blk\.11\.attn_q_a_norm\.weight$=f32
^blk\.11\.attn_q_a\.weight$=bf16
^blk\.11\.attn_q_b\.weight$=bf16
^blk\.40\.attn_norm\.weight$=f32
^blk\.40\.exp_probs_b\.bias$=f32
^blk\.40\.ffn_gate_inp\.weight$=f32
^blk\.40\.ffn_down_shexp\.weight$=bf16
^blk\.40\.ffn_gate_shexp\.weight$=bf16
^blk\.40\.ffn_up_shexp\.weight$=bf16
^blk\.40\.ffn_norm\.weight$=f32
^blk\.40\.attn_kv_a_norm\.weight$=f32
^blk\.40\.attn_kv_a_mqa\.weight$=bf16
^blk\.40\.attn_k_b\.weight$=bf16
^blk\.40\.attn_v_b\.weight$=bf16
^blk\.40\.attn_output\.weight$=bf16
^blk\.40\.attn_q_a_norm\.weight$=f32
^blk\.40\.attn_q_a\.weight$=bf16
^blk\.40\.attn_q_b\.weight$=bf16
^blk\.52\.attn_norm\.weight$=f32
^blk\.52\.exp_probs_b\.bias$=f32
^blk\.52\.ffn_gate_inp\.weight$=f32
^blk\.52\.ffn_down_shexp\.weight$=bf16
^blk\.52\.ffn_gate_shexp\.weight$=bf16
^blk\.52\.ffn_up_shexp\.weight$=bf16
^blk\.52\.ffn_norm\.weight$=f32
^blk\.52\.attn_kv_a_norm\.weight$=f32
^blk\.52\.attn_kv_a_mqa\.weight$=bf16
^blk\.52\.attn_k_b\.weight$=bf16
^blk\.52\.attn_v_b\.weight$=bf16
^blk\.52\.attn_output\.weight$=bf16
^blk\.52\.attn_q_a_norm\.weight$=f32
^blk\.52\.attn_q_a\.weight$=bf16
^blk\.52\.attn_q_b\.weight$=bf16
^blk\.32\.attn_norm\.weight$=f32
^blk\.32\.exp_probs_b\.bias$=f32
^blk\.32\.ffn_gate_inp\.weight$=f32
^blk\.32\.ffn_down_shexp\.weight$=bf16
^blk\.32\.ffn_gate_shexp\.weight$=bf16
^blk\.32\.ffn_up_shexp\.weight$=bf16
^blk\.32\.ffn_norm\.weight$=f32
^blk\.32\.attn_kv_a_norm\.weight$=f32
^blk\.32\.attn_kv_a_mqa\.weight$=bf16
^blk\.32\.attn_k_b\.weight$=bf16
^blk\.32\.attn_v_b\.weight$=bf16
^blk\.32\.attn_output\.weight$=bf16
^blk\.32\.attn_q_a_norm\.weight$=f32
^blk\.32\.attn_q_a\.weight$=bf16
^blk\.32\.attn_q_b\.weight$=bf16
^blk\.18\.attn_norm\.weight$=f32
^blk\.18\.exp_probs_b\.bias$=f32
^blk\.18\.ffn_gate_inp\.weight$=f32
^blk\.18\.ffn_down_shexp\.weight$=bf16
^blk\.18\.ffn_gate_shexp\.weight$=bf16
^blk\.18\.ffn_up_shexp\.weight$=bf16
^blk\.18\.ffn_norm\.weight$=f32
^blk\.18\.attn_kv_a_norm\.weight$=f32
^blk\.18\.attn_kv_a_mqa\.weight$=bf16
^blk\.18\.attn_k_b\.weight$=bf16
^blk\.18\.attn_v_b\.weight$=bf16
^blk\.18\.attn_output\.weight$=bf16
^blk\.18\.attn_q_a_norm\.weight$=f32
^blk\.18\.attn_q_a\.weight$=bf16
^blk\.18\.attn_q_b\.weight$=bf16
^blk\.30\.attn_norm\.weight$=f32
^blk\.30\.exp_probs_b\.bias$=f32
^blk\.30\.ffn_gate_inp\.weight$=f32
^blk\.30\.ffn_down_shexp\.weight$=bf16
^blk\.30\.ffn_gate_shexp\.weight$=bf16
^blk\.30\.ffn_up_shexp\.weight$=bf16
^blk\.30\.ffn_norm\.weight$=f32
^blk\.30\.attn_kv_a_norm\.weight$=f32
^blk\.30\.attn_kv_a_mqa\.weight$=bf16
^blk\.30\.attn_k_b\.weight$=bf16
^blk\.30\.attn_v_b\.weight$=bf16
^blk\.30\.attn_output\.weight$=bf16
^blk\.30\.attn_q_a_norm\.weight$=f32
^blk\.30\.attn_q_a\.weight$=bf16
^blk\.30\.attn_q_b\.weight$=bf16
^blk\.49\.attn_norm\.weight$=f32
^blk\.49\.exp_probs_b\.bias$=f32
^blk\.49\.ffn_gate_inp\.weight$=f32
^blk\.49\.ffn_down_shexp\.weight$=bf16
^blk\.49\.ffn_gate_shexp\.weight$=bf16
^blk\.49\.ffn_up_shexp\.weight$=bf16
^blk\.49\.ffn_norm\.weight$=f32
^blk\.49\.attn_kv_a_norm\.weight$=f32
^blk\.49\.attn_kv_a_mqa\.weight$=bf16
^blk\.49\.attn_k_b\.weight$=bf16
^blk\.49\.attn_v_b\.weight$=bf16
^blk\.49\.attn_output\.weight$=bf16
^blk\.49\.attn_q_a_norm\.weight$=f32
^blk\.49\.attn_q_a\.weight$=bf16
^blk\.49\.attn_q_b\.weight$=bf16
^blk\.54\.attn_norm\.weight$=f32
^blk\.54\.exp_probs_b\.bias$=f32
^blk\.54\.ffn_gate_inp\.weight$=f32
^blk\.54\.ffn_down_shexp\.weight$=bf16
^blk\.54\.ffn_gate_shexp\.weight$=bf16
^blk\.54\.ffn_up_shexp\.weight$=bf16
^blk\.54\.ffn_norm\.weight$=f32
^blk\.54\.attn_kv_a_norm\.weight$=f32
^blk\.54\.attn_kv_a_mqa\.weight$=bf16
^blk\.54\.attn_k_b\.weight$=bf16
^blk\.54\.attn_v_b\.weight$=bf16
^blk\.54\.attn_output\.weight$=bf16
^blk\.54\.attn_q_a_norm\.weight$=f32
^blk\.54\.attn_q_a\.weight$=bf16
^blk\.54\.attn_q_b\.weight$=bf16
^blk\.38\.attn_norm\.weight$=f32
^blk\.38\.exp_probs_b\.bias$=f32
^blk\.38\.ffn_gate_inp\.weight$=f32
^blk\.38\.ffn_down_shexp\.weight$=bf16
^blk\.38\.ffn_gate_shexp\.weight$=bf16
^blk\.38\.ffn_up_shexp\.weight$=bf16
^blk\.38\.ffn_norm\.weight$=f32
^blk\.38\.attn_kv_a_norm\.weight$=f32
^blk\.38\.attn_kv_a_mqa\.weight$=bf16
^blk\.38\.attn_k_b\.weight$=bf16
^blk\.38\.attn_v_b\.weight$=bf16
^blk\.38\.attn_output\.weight$=bf16
^blk\.38\.attn_q_a_norm\.weight$=f32
^blk\.38\.attn_q_a\.weight$=bf16
^blk\.38\.attn_q_b\.weight$=bf16
^blk\.2\.attn_norm\.weight$=f32
^blk\.2\.exp_probs_b\.bias$=f32
^blk\.2\.ffn_gate_inp\.weight$=f32
^blk\.2\.ffn_down_shexp\.weight$=bf16
^blk\.2\.ffn_gate_shexp\.weight$=bf16
^blk\.2\.ffn_up_shexp\.weight$=bf16
^blk\.2\.ffn_norm\.weight$=f32
^blk\.2\.attn_kv_a_norm\.weight$=f32
^blk\.2\.attn_kv_a_mqa\.weight$=bf16
^blk\.2\.attn_k_b\.weight$=bf16
^blk\.2\.attn_v_b\.weight$=bf16
^blk\.2\.attn_output\.weight$=bf16
^blk\.2\.attn_q_a_norm\.weight$=f32
^blk\.2\.attn_q_a\.weight$=bf16
^blk\.2\.attn_q_b\.weight$=bf16
^blk\.12\.attn_norm\.weight$=f32
^blk\.12\.exp_probs_b\.bias$=f32
^blk\.12\.ffn_gate_inp\.weight$=f32
^blk\.12\.ffn_down_shexp\.weight$=bf16
^blk\.12\.ffn_gate_shexp\.weight$=bf16
^blk\.12\.ffn_up_shexp\.weight$=bf16
^blk\.12\.ffn_norm\.weight$=f32
^blk\.12\.attn_kv_a_norm\.weight$=f32
^blk\.12\.attn_kv_a_mqa\.weight$=bf16
^blk\.12\.attn_k_b\.weight$=bf16
^blk\.12\.attn_v_b\.weight$=bf16
^blk\.12\.attn_output\.weight$=bf16
^blk\.12\.attn_q_a_norm\.weight$=f32
^blk\.12\.attn_q_a\.weight$=bf16
^blk\.12\.attn_q_b\.weight$=bf16
^blk\.1\.attn_norm\.weight$=f32
^blk\.1\.exp_probs_b\.bias$=f32
^blk\.1\.ffn_gate_inp\.weight$=f32
^blk\.1\.ffn_down_shexp\.weight$=bf16
^blk\.1\.ffn_gate_shexp\.weight$=bf16
^blk\.1\.ffn_up_shexp\.weight$=bf16
^blk\.1\.ffn_norm\.weight$=f32
^blk\.1\.attn_kv_a_norm\.weight$=f32
^blk\.1\.attn_kv_a_mqa\.weight$=bf16
^blk\.1\.attn_k_b\.weight$=bf16
^blk\.1\.attn_v_b\.weight$=bf16
^blk\.1\.attn_output\.weight$=bf16
^blk\.1\.attn_q_a_norm\.weight$=f32
^blk\.1\.attn_q_a\.weight$=bf16
^blk\.1\.attn_q_b\.weight$=bf16
^blk\.50\.attn_norm\.weight$=f32
^blk\.50\.exp_probs_b\.bias$=f32
^blk\.50\.ffn_gate_inp\.weight$=f32
^blk\.50\.ffn_down_shexp\.weight$=bf16
^blk\.50\.ffn_gate_shexp\.weight$=bf16
^blk\.50\.ffn_up_shexp\.weight$=bf16
^blk\.50\.ffn_norm\.weight$=f32
^blk\.50\.attn_kv_a_norm\.weight$=f32
^blk\.50\.attn_kv_a_mqa\.weight$=bf16
^blk\.50\.attn_k_b\.weight$=bf16
^blk\.50\.attn_v_b\.weight$=bf16
^blk\.50\.attn_output\.weight$=bf16
^blk\.50\.attn_q_a_norm\.weight$=f32
^blk\.50\.attn_q_a\.weight$=bf16
^blk\.50\.attn_q_b\.weight$=bf16
^blk\.36\.attn_norm\.weight$=f32
^blk\.36\.exp_probs_b\.bias$=f32
^blk\.36\.ffn_gate_inp\.weight$=f32
^blk\.36\.ffn_down_shexp\.weight$=bf16
^blk\.36\.ffn_gate_shexp\.weight$=bf16
^blk\.36\.ffn_up_shexp\.weight$=bf16
^blk\.36\.ffn_norm\.weight$=f32
^blk\.36\.attn_kv_a_norm\.weight$=f32
^blk\.36\.attn_kv_a_mqa\.weight$=bf16
^blk\.36\.attn_k_b\.weight$=bf16
^blk\.36\.attn_v_b\.weight$=bf16
^blk\.36\.attn_output\.weight$=bf16
^blk\.36\.attn_q_a_norm\.weight$=f32
^blk\.36\.attn_q_a\.weight$=bf16
^blk\.36\.attn_q_b\.weight$=bf16
^blk\.31\.attn_norm\.weight$=f32
^blk\.31\.exp_probs_b\.bias$=f32
^blk\.31\.ffn_gate_inp\.weight$=f32
^blk\.31\.ffn_down_shexp\.weight$=bf16
^blk\.31\.ffn_gate_shexp\.weight$=bf16
^blk\.31\.ffn_up_shexp\.weight$=bf16
^blk\.31\.ffn_norm\.weight$=f32
^blk\.31\.attn_kv_a_norm\.weight$=f32
^blk\.31\.attn_kv_a_mqa\.weight$=bf16
^blk\.31\.attn_k_b\.weight$=bf16
^blk\.31\.attn_v_b\.weight$=bf16
^blk\.31\.attn_output\.weight$=bf16
^blk\.31\.attn_q_a_norm\.weight$=f32
^blk\.31\.attn_q_a\.weight$=bf16
^blk\.31\.attn_q_b\.weight$=bf16
^blk\.14\.attn_norm\.weight$=f32
^blk\.14\.exp_probs_b\.bias$=f32
^blk\.14\.ffn_gate_inp\.weight$=f32
^blk\.14\.ffn_down_shexp\.weight$=bf16
^blk\.14\.ffn_gate_shexp\.weight$=bf16
^blk\.14\.ffn_up_shexp\.weight$=bf16
^blk\.14\.ffn_norm\.weight$=f32
^blk\.14\.attn_kv_a_norm\.weight$=f32
^blk\.14\.attn_kv_a_mqa\.weight$=bf16
^blk\.14\.attn_k_b\.weight$=bf16
^blk\.14\.attn_v_b\.weight$=bf16
^blk\.14\.attn_output\.weight$=bf16
^blk\.14\.attn_q_a_norm\.weight$=f32
^blk\.14\.attn_q_a\.weight$=bf16
^blk\.14\.attn_q_b\.weight$=bf16
^blk\.23\.attn_norm\.weight$=f32
^blk\.23\.exp_probs_b\.bias$=f32
^blk\.23\.ffn_gate_inp\.weight$=f32
^blk\.23\.ffn_down_shexp\.weight$=bf16
^blk\.23\.ffn_gate_shexp\.weight$=bf16
^blk\.23\.ffn_up_shexp\.weight$=bf16
^blk\.23\.ffn_norm\.weight$=f32
^blk\.23\.attn_kv_a_norm\.weight$=f32
^blk\.23\.attn_kv_a_mqa\.weight$=bf16
^blk\.23\.attn_k_b\.weight$=bf16
^blk\.23\.attn_v_b\.weight$=bf16
^blk\.23\.attn_output\.weight$=bf16
^blk\.23\.attn_q_a_norm\.weight$=f32
^blk\.23\.attn_q_a\.weight$=bf16
^blk\.23\.attn_q_b\.weight$=bf16
^blk\.21\.attn_norm\.weight$=f32
^blk\.21\.exp_probs_b\.bias$=f32
^blk\.21\.ffn_gate_inp\.weight$=f32
^blk\.21\.ffn_down_shexp\.weight$=bf16
^blk\.21\.ffn_gate_shexp\.weight$=bf16
^blk\.21\.ffn_up_shexp\.weight$=bf16
^blk\.21\.ffn_norm\.weight$=f32
^blk\.21\.attn_kv_a_norm\.weight$=f32
^blk\.21\.attn_kv_a_mqa\.weight$=bf16
^blk\.21\.attn_k_b\.weight$=bf16
^blk\.21\.attn_v_b\.weight$=bf16
^blk\.21\.attn_output\.weight$=bf16
^blk\.21\.attn_q_a_norm\.weight$=f32
^blk\.21\.attn_q_a\.weight$=bf16
^blk\.21\.attn_q_b\.weight$=bf16
^blk\.0\.attn_norm\.weight$=f32
^blk\.0\.ffn_down\.weight$=bf16
^blk\.0\.ffn_gate\.weight$=bf16
^blk\.0\.ffn_up\.weight$=bf16
^blk\.0\.ffn_norm\.weight$=f32
^blk\.0\.attn_kv_a_norm\.weight$=f32
^blk\.0\.attn_kv_a_mqa\.weight$=bf16
^blk\.0\.attn_k_b\.weight$=bf16
^blk\.0\.attn_v_b\.weight$=bf16
^blk\.0\.attn_output\.weight$=bf16
^blk\.0\.attn_q_a_norm\.weight$=f32
^blk\.0\.attn_q_a\.weight$=bf16
^blk\.0\.attn_q_b\.weight$=bf16
^blk\.25\.attn_norm\.weight$=f32
^blk\.25\.exp_probs_b\.bias$=f32
^blk\.25\.ffn_gate_inp\.weight$=f32
^blk\.25\.ffn_down_shexp\.weight$=bf16
^blk\.25\.ffn_gate_shexp\.weight$=bf16
^blk\.25\.ffn_up_shexp\.weight$=bf16
^blk\.25\.ffn_norm\.weight$=f32
^blk\.25\.attn_kv_a_norm\.weight$=f32
^blk\.25\.attn_kv_a_mqa\.weight$=bf16
^blk\.25\.attn_k_b\.weight$=bf16
^blk\.25\.attn_v_b\.weight$=bf16
^blk\.25\.attn_output\.weight$=bf16
^blk\.25\.attn_q_a_norm\.weight$=f32
^blk\.25\.attn_q_a\.weight$=bf16
^blk\.25\.attn_q_b\.weight$=bf16
^blk\.43\.attn_norm\.weight$=f32
^blk\.43\.exp_probs_b\.bias$=f32
^blk\.43\.ffn_gate_inp\.weight$=f32
^blk\.43\.ffn_down_shexp\.weight$=bf16
^blk\.43\.ffn_gate_shexp\.weight$=bf16
^blk\.43\.ffn_up_shexp\.weight$=bf16
^blk\.43\.ffn_norm\.weight$=f32
^blk\.43\.attn_kv_a_norm\.weight$=f32
^blk\.43\.attn_kv_a_mqa\.weight$=bf16
^blk\.43\.attn_k_b\.weight$=bf16
^blk\.43\.attn_v_b\.weight$=bf16
^blk\.43\.attn_output\.weight$=bf16
^blk\.43\.attn_q_a_norm\.weight$=f32
^blk\.43\.attn_q_a\.weight$=bf16
^blk\.43\.attn_q_b\.weight$=bf16
^blk\.28\.attn_norm\.weight$=f32
^blk\.28\.exp_probs_b\.bias$=f32
^blk\.28\.ffn_gate_inp\.weight$=f32
^blk\.28\.ffn_down_shexp\.weight$=bf16
^blk\.28\.ffn_gate_shexp\.weight$=bf16
^blk\.28\.ffn_up_shexp\.weight$=bf16
^blk\.28\.ffn_norm\.weight$=f32
^blk\.28\.attn_kv_a_norm\.weight$=f32
^blk\.28\.attn_kv_a_mqa\.weight$=bf16
^blk\.28\.attn_k_b\.weight$=bf16
^blk\.28\.attn_v_b\.weight$=bf16
^blk\.28\.attn_output\.weight$=bf16
^blk\.28\.attn_q_a_norm\.weight$=f32
^blk\.28\.attn_q_a\.weight$=bf16
^blk\.28\.attn_q_b\.weight$=bf16
^blk\.53\.attn_norm\.weight$=f32
^blk\.53\.exp_probs_b\.bias$=f32
^blk\.53\.ffn_gate_inp\.weight$=f32
^blk\.53\.ffn_down_shexp\.weight$=bf16
^blk\.53\.ffn_gate_shexp\.weight$=bf16
^blk\.53\.ffn_up_shexp\.weight$=bf16
^blk\.53\.ffn_norm\.weight$=f32
^blk\.53\.attn_kv_a_norm\.weight$=f32
^blk\.53\.attn_kv_a_mqa\.weight$=bf16
^blk\.53\.attn_k_b\.weight$=bf16
^blk\.53\.attn_v_b\.weight$=bf16
^blk\.53\.attn_output\.weight$=bf16
^blk\.53\.attn_q_a_norm\.weight$=f32
^blk\.53\.attn_q_a\.weight$=bf16
^blk\.53\.attn_q_b\.weight$=bf16
^blk\.13\.attn_norm\.weight$=f32
^blk\.13\.exp_probs_b\.bias$=f32
^blk\.13\.ffn_gate_inp\.weight$=f32
^blk\.13\.ffn_down_shexp\.weight$=bf16
^blk\.13\.ffn_gate_shexp\.weight$=bf16
^blk\.13\.ffn_up_shexp\.weight$=bf16
^blk\.13\.ffn_norm\.weight$=f32
^blk\.13\.attn_kv_a_norm\.weight$=f32
^blk\.13\.attn_kv_a_mqa\.weight$=bf16
^blk\.13\.attn_k_b\.weight$=bf16
^blk\.13\.attn_v_b\.weight$=bf16
^blk\.13\.attn_output\.weight$=bf16
^blk\.13\.attn_q_a_norm\.weight$=f32
^blk\.13\.attn_q_a\.weight$=bf16
^blk\.13\.attn_q_b\.weight$=bf16
^blk\.60\.attn_norm\.weight$=f32
^blk\.60\.exp_probs_b\.bias$=f32
^blk\.60\.ffn_gate_inp\.weight$=f32
^blk\.60\.ffn_down_shexp\.weight$=bf16
^blk\.60\.ffn_gate_shexp\.weight$=bf16
^blk\.60\.ffn_up_shexp\.weight$=bf16
^blk\.60\.ffn_norm\.weight$=f32
^blk\.60\.attn_kv_a_norm\.weight$=f32
^blk\.60\.attn_kv_a_mqa\.weight$=bf16
^blk\.60\.attn_k_b\.weight$=bf16
^blk\.60\.attn_v_b\.weight$=bf16
^blk\.60\.attn_output\.weight$=bf16
^blk\.60\.attn_q_a_norm\.weight$=f32
^blk\.60\.attn_q_a\.weight$=bf16
^blk\.60\.attn_q_b\.weight$=bf16
^blk\.4\.attn_norm\.weight$=f32
^blk\.4\.exp_probs_b\.bias$=f32
^blk\.4\.ffn_gate_inp\.weight$=f32
^blk\.4\.ffn_down_shexp\.weight$=bf16
^blk\.4\.ffn_gate_shexp\.weight$=bf16
^blk\.4\.ffn_up_shexp\.weight$=bf16
^blk\.4\.ffn_norm\.weight$=f32
^blk\.4\.attn_kv_a_norm\.weight$=f32
^blk\.4\.attn_kv_a_mqa\.weight$=bf16
^blk\.4\.attn_k_b\.weight$=bf16
^blk\.4\.attn_v_b\.weight$=bf16
^blk\.4\.attn_output\.weight$=bf16
^blk\.4\.attn_q_a_norm\.weight$=f32
^blk\.4\.attn_q_a\.weight$=bf16
^blk\.4\.attn_q_b\.weight$=bf16
^blk\.57\.attn_norm\.weight$=f32
^blk\.57\.exp_probs_b\.bias$=f32
^blk\.57\.ffn_gate_inp\.weight$=f32
^blk\.57\.ffn_down_shexp\.weight$=bf16
^blk\.57\.ffn_gate_shexp\.weight$=bf16
^blk\.57\.ffn_up_shexp\.weight$=bf16
^blk\.57\.ffn_norm\.weight$=f32
^blk\.57\.attn_kv_a_norm\.weight$=f32
^blk\.57\.attn_kv_a_mqa\.weight$=bf16
^blk\.57\.attn_k_b\.weight$=bf16
^blk\.57\.attn_v_b\.weight$=bf16
^blk\.57\.attn_output\.weight$=bf16
^blk\.57\.attn_q_a_norm\.weight$=f32
^blk\.57\.attn_q_a\.weight$=bf16
^blk\.57\.attn_q_b\.weight$=bf16
^blk\.45\.attn_norm\.weight$=f32
^blk\.45\.exp_probs_b\.bias$=f32
^blk\.45\.ffn_gate_inp\.weight$=f32
^blk\.45\.ffn_down_shexp\.weight$=bf16
^blk\.45\.ffn_gate_shexp\.weight$=bf16
^blk\.45\.ffn_up_shexp\.weight$=bf16
^blk\.45\.ffn_norm\.weight$=f32
^blk\.45\.attn_kv_a_norm\.weight$=f32
^blk\.45\.attn_kv_a_mqa\.weight$=bf16
^blk\.45\.attn_k_b\.weight$=bf16
^blk\.45\.attn_v_b\.weight$=bf16
^blk\.45\.attn_output\.weight$=bf16
^blk\.45\.attn_q_a_norm\.weight$=f32
^blk\.45\.attn_q_a\.weight$=bf16
^blk\.45\.attn_q_b\.weight$=bf16
^blk\.20\.attn_norm\.weight$=f32
^blk\.20\.exp_probs_b\.bias$=f32
^blk\.20\.ffn_gate_inp\.weight$=f32
^blk\.20\.ffn_down_shexp\.weight$=bf16
^blk\.20\.ffn_gate_shexp\.weight$=bf16
^blk\.20\.ffn_up_shexp\.weight$=bf16
^blk\.20\.ffn_norm\.weight$=f32
^blk\.20\.attn_kv_a_norm\.weight$=f32
^blk\.20\.attn_kv_a_mqa\.weight$=bf16
^blk\.20\.attn_k_b\.weight$=bf16
^blk\.20\.attn_v_b\.weight$=bf16
^blk\.20\.attn_output\.weight$=bf16
^blk\.20\.attn_q_a_norm\.weight$=f32
^blk\.20\.attn_q_a\.weight$=bf16
^blk\.20\.attn_q_b\.weight$=bf16
^blk\.59\.attn_norm\.weight$=f32
^blk\.59\.exp_probs_b\.bias$=f32
^blk\.59\.ffn_gate_inp\.weight$=f32
^blk\.59\.ffn_down_shexp\.weight$=bf16
^blk\.59\.ffn_gate_shexp\.weight$=bf16
^blk\.59\.ffn_up_shexp\.weight$=bf16
^blk\.59\.ffn_norm\.weight$=f32
^blk\.59\.attn_kv_a_norm\.weight$=f32
^blk\.59\.attn_kv_a_mqa\.weight$=bf16
^blk\.59\.attn_k_b\.weight$=bf16
^blk\.59\.attn_v_b\.weight$=bf16
^blk\.59\.attn_output\.weight$=bf16
^blk\.59\.attn_q_a_norm\.weight$=f32
^blk\.59\.attn_q_a\.weight$=bf16
^blk\.59\.attn_q_b\.weight$=bf16
^blk\.6\.attn_norm\.weight$=f32
^blk\.6\.exp_probs_b\.bias$=f32
^blk\.6\.ffn_gate_inp\.weight$=f32
^blk\.6\.ffn_down_shexp\.weight$=bf16
^blk\.6\.ffn_gate_shexp\.weight$=bf16
^blk\.6\.ffn_up_shexp\.weight$=bf16
^blk\.6\.ffn_norm\.weight$=f32
^blk\.6\.attn_kv_a_norm\.weight$=f32
^blk\.6\.attn_kv_a_mqa\.weight$=bf16
^blk\.6\.attn_k_b\.weight$=bf16
^blk\.6\.attn_v_b\.weight$=bf16
^blk\.6\.attn_output\.weight$=bf16
^blk\.6\.attn_q_a_norm\.weight$=f32
^blk\.6\.attn_q_a\.weight$=bf16
^blk\.6\.attn_q_b\.weight$=bf16
^blk\.26\.attn_norm\.weight$=f32
^blk\.26\.exp_probs_b\.bias$=f32
^blk\.26\.ffn_gate_inp\.weight$=f32
^blk\.26\.ffn_down_shexp\.weight$=bf16
^blk\.26\.ffn_gate_shexp\.weight$=bf16
^blk\.26\.ffn_up_shexp\.weight$=bf16
^blk\.26\.ffn_norm\.weight$=f32
^blk\.26\.attn_kv_a_norm\.weight$=f32
^blk\.26\.attn_kv_a_mqa\.weight$=bf16
^blk\.26\.attn_k_b\.weight$=bf16
^blk\.26\.attn_v_b\.weight$=bf16
^blk\.26\.attn_output\.weight$=bf16
^blk\.26\.attn_q_a_norm\.weight$=f32
^blk\.26\.attn_q_a\.weight$=bf16
^blk\.26\.attn_q_b\.weight$=bf16
^blk\.41\.attn_norm\.weight$=f32
^blk\.41\.exp_probs_b\.bias$=f32
^blk\.41\.ffn_gate_inp\.weight$=f32
^blk\.41\.ffn_down_shexp\.weight$=bf16
^blk\.41\.ffn_gate_shexp\.weight$=bf16
^blk\.41\.ffn_up_shexp\.weight$=bf16
^blk\.41\.ffn_norm\.weight$=f32
^blk\.41\.attn_kv_a_norm\.weight$=f32
^blk\.41\.attn_kv_a_mqa\.weight$=bf16
^blk\.41\.attn_k_b\.weight$=bf16
^blk\.41\.attn_v_b\.weight$=bf16
^blk\.41\.attn_output\.weight$=bf16
^blk\.41\.attn_q_a_norm\.weight$=f32
^blk\.41\.attn_q_a\.weight$=bf16
^blk\.41\.attn_q_b\.weight$=bf16
^blk\.56\.attn_norm\.weight$=f32
^blk\.56\.exp_probs_b\.bias$=f32
^blk\.56\.ffn_gate_inp\.weight$=f32
^blk\.56\.ffn_down_shexp\.weight$=bf16
^blk\.56\.ffn_gate_shexp\.weight$=bf16
^blk\.56\.ffn_up_shexp\.weight$=bf16
^blk\.56\.ffn_norm\.weight$=f32
^blk\.56\.attn_kv_a_norm\.weight$=f32
^blk\.56\.attn_kv_a_mqa\.weight$=bf16
^blk\.56\.attn_k_b\.weight$=bf16
^blk\.56\.attn_v_b\.weight$=bf16
^blk\.56\.attn_output\.weight$=bf16
^blk\.56\.attn_q_a_norm\.weight$=f32
^blk\.56\.attn_q_a\.weight$=bf16
^blk\.56\.attn_q_b\.weight$=bf16
^blk\.22\.attn_norm\.weight$=f32
^blk\.22\.exp_probs_b\.bias$=f32
^blk\.22\.ffn_gate_inp\.weight$=f32
^blk\.22\.ffn_down_shexp\.weight$=bf16
^blk\.22\.ffn_gate_shexp\.weight$=bf16
^blk\.22\.ffn_up_shexp\.weight$=bf16
^blk\.22\.ffn_norm\.weight$=f32
^blk\.22\.attn_kv_a_norm\.weight$=f32
^blk\.22\.attn_kv_a_mqa\.weight$=bf16
^blk\.22\.attn_k_b\.weight$=bf16
^blk\.22\.attn_v_b\.weight$=bf16
^blk\.22\.attn_output\.weight$=bf16
^blk\.22\.attn_q_a_norm\.weight$=f32
^blk\.22\.attn_q_a\.weight$=bf16
^blk\.22\.attn_q_b\.weight$=bf16
^blk\.29\.attn_norm\.weight$=f32
^blk\.29\.exp_probs_b\.bias$=f32
^blk\.29\.ffn_gate_inp\.weight$=f32
^blk\.29\.ffn_down_shexp\.weight$=bf16
^blk\.29\.ffn_gate_shexp\.weight$=bf16
^blk\.29\.ffn_up_shexp\.weight$=bf16
^blk\.29\.ffn_norm\.weight$=f32
^blk\.29\.attn_kv_a_norm\.weight$=f32
^blk\.29\.attn_kv_a_mqa\.weight$=bf16
^blk\.29\.attn_k_b\.weight$=bf16
^blk\.29\.attn_v_b\.weight$=bf16
^blk\.29\.attn_output\.weight$=bf16
^blk\.29\.attn_q_a_norm\.weight$=f32
^blk\.29\.attn_q_a\.weight$=bf16
^blk\.29\.attn_q_b\.weight$=bf16
^blk\.19\.attn_norm\.weight$=f32
^blk\.19\.exp_probs_b\.bias$=f32
^blk\.19\.ffn_gate_inp\.weight$=f32
^blk\.19\.ffn_down_shexp\.weight$=bf16
^blk\.19\.ffn_gate_shexp\.weight$=bf16
^blk\.19\.ffn_up_shexp\.weight$=bf16
^blk\.19\.ffn_norm\.weight$=f32
^blk\.19\.attn_kv_a_norm\.weight$=f32
^blk\.19\.attn_kv_a_mqa\.weight$=bf16
^blk\.19\.attn_k_b\.weight$=bf16
^blk\.19\.attn_v_b\.weight$=bf16
^blk\.19\.attn_output\.weight$=bf16
^blk\.19\.attn_q_a_norm\.weight$=f32
^blk\.19\.attn_q_a\.weight$=bf16
^blk\.19\.attn_q_b\.weight$=bf16
^blk\.33\.attn_norm\.weight$=f32
^blk\.33\.exp_probs_b\.bias$=f32
^blk\.33\.ffn_gate_inp\.weight$=f32
^blk\.33\.ffn_down_shexp\.weight$=bf16
^blk\.33\.ffn_gate_shexp\.weight$=bf16
^blk\.33\.ffn_up_shexp\.weight$=bf16
^blk\.33\.ffn_norm\.weight$=f32
^blk\.33\.attn_kv_a_norm\.weight$=f32
^blk\.33\.attn_kv_a_mqa\.weight$=bf16
^blk\.33\.attn_k_b\.weight$=bf16
^blk\.33\.attn_v_b\.weight$=bf16
^blk\.33\.attn_output\.weight$=bf16
^blk\.33\.attn_q_a_norm\.weight$=f32
^blk\.33\.attn_q_a\.weight$=bf16
^blk\.33\.attn_q_b\.weight$=bf16
^blk\.51\.attn_norm\.weight$=f32
^blk\.51\.exp_probs_b\.bias$=f32
^blk\.51\.ffn_gate_inp\.weight$=f32
^blk\.51\.ffn_down_shexp\.weight$=bf16
^blk\.51\.ffn_gate_shexp\.weight$=bf16
^blk\.51\.ffn_up_shexp\.weight$=bf16
^blk\.51\.ffn_norm\.weight$=f32
^blk\.51\.attn_kv_a_norm\.weight$=f32
^blk\.51\.attn_kv_a_mqa\.weight$=bf16
^blk\.51\.attn_k_b\.weight$=bf16
^blk\.51\.attn_v_b\.weight$=bf16
^blk\.51\.attn_output\.weight$=bf16
^blk\.51\.attn_q_a_norm\.weight$=f32
^blk\.51\.attn_q_a\.weight$=bf16
^blk\.51\.attn_q_b\.weight$=bf16
^blk\.10\.attn_norm\.weight$=f32
^blk\.10\.exp_probs_b\.bias$=f32
^blk\.10\.ffn_gate_inp\.weight$=f32
^blk\.10\.ffn_down_shexp\.weight$=bf16
^blk\.10\.ffn_gate_shexp\.weight$=bf16
^blk\.10\.ffn_up_shexp\.weight$=bf16
^blk\.10\.ffn_norm\.weight$=f32
^blk\.10\.attn_kv_a_norm\.weight$=f32
^blk\.10\.attn_kv_a_mqa\.weight$=bf16
^blk\.10\.attn_k_b\.weight$=bf16
^blk\.10\.attn_v_b\.weight$=bf16
^blk\.10\.attn_output\.weight$=bf16
^blk\.10\.attn_q_a_norm\.weight$=f32
^blk\.10\.attn_q_a\.weight$=bf16
^blk\.10\.attn_q_b\.weight$=bf16
^blk\.47\.attn_norm\.weight$=f32
^blk\.47\.exp_probs_b\.bias$=f32
^blk\.47\.ffn_gate_inp\.weight$=f32
^blk\.47\.ffn_down_shexp\.weight$=bf16
^blk\.47\.ffn_gate_shexp\.weight$=bf16
^blk\.47\.ffn_up_shexp\.weight$=bf16
^blk\.47\.ffn_norm\.weight$=f32
^blk\.47\.attn_kv_a_norm\.weight$=f32
^blk\.47\.attn_kv_a_mqa\.weight$=bf16
^blk\.47\.attn_k_b\.weight$=bf16
^blk\.47\.attn_v_b\.weight$=bf16
^blk\.47\.attn_output\.weight$=bf16
^blk\.47\.attn_q_a_norm\.weight$=f32
^blk\.47\.attn_q_a\.weight$=bf16
^blk\.47\.attn_q_b\.weight$=bf16
^blk\.35\.attn_norm\.weight$=f32
^blk\.35\.exp_probs_b\.bias$=f32
^blk\.35\.ffn_gate_inp\.weight$=f32
^blk\.35\.ffn_down_shexp\.weight$=bf16
^blk\.35\.ffn_gate_shexp\.weight$=bf16
^blk\.35\.ffn_up_shexp\.weight$=bf16
^blk\.35\.ffn_norm\.weight$=f32
^blk\.35\.attn_kv_a_norm\.weight$=f32
^blk\.35\.attn_kv_a_mqa\.weight$=bf16
^blk\.35\.attn_k_b\.weight$=bf16
^blk\.35\.attn_v_b\.weight$=bf16
^blk\.35\.attn_output\.weight$=bf16
^blk\.35\.attn_q_a_norm\.weight$=f32
^blk\.35\.attn_q_a\.weight$=bf16
^blk\.35\.attn_q_b\.weight$=bf16
^blk\.8\.attn_norm\.weight$=f32
^blk\.8\.exp_probs_b\.bias$=f32
^blk\.8\.ffn_gate_inp\.weight$=f32
^blk\.8\.ffn_down_shexp\.weight$=bf16
^blk\.8\.ffn_gate_shexp\.weight$=bf16
^blk\.8\.ffn_up_shexp\.weight$=bf16
^blk\.8\.ffn_norm\.weight$=f32
^blk\.8\.attn_kv_a_norm\.weight$=f32
^blk\.8\.attn_kv_a_mqa\.weight$=bf16
^blk\.8\.attn_k_b\.weight$=bf16
^blk\.8\.attn_v_b\.weight$=bf16
^blk\.8\.attn_output\.weight$=bf16
^blk\.8\.attn_q_a_norm\.weight$=f32
^blk\.8\.attn_q_a\.weight$=bf16
^blk\.8\.attn_q_b\.weight$=bf16
^blk\.3\.attn_norm\.weight$=f32
^blk\.3\.exp_probs_b\.bias$=f32
^blk\.3\.ffn_gate_inp\.weight$=f32
^blk\.3\.ffn_down_shexp\.weight$=bf16
^blk\.3\.ffn_gate_shexp\.weight$=bf16
^blk\.3\.ffn_up_shexp\.weight$=bf16
^blk\.3\.ffn_norm\.weight$=f32
^blk\.3\.attn_kv_a_norm\.weight$=f32
^blk\.3\.attn_kv_a_mqa\.weight$=bf16
^blk\.3\.attn_k_b\.weight$=bf16
^blk\.3\.attn_v_b\.weight$=bf16
^blk\.3\.attn_output\.weight$=bf16
^blk\.3\.attn_q_a_norm\.weight$=f32
^blk\.3\.attn_q_a\.weight$=bf16
^blk\.3\.attn_q_b\.weight$=bf16
^blk\.24\.attn_norm\.weight$=f32
^blk\.24\.exp_probs_b\.bias$=f32
^blk\.24\.ffn_gate_inp\.weight$=f32
^blk\.24\.ffn_down_shexp\.weight$=bf16
^blk\.24\.ffn_gate_shexp\.weight$=bf16
^blk\.24\.ffn_up_shexp\.weight$=bf16
^blk\.24\.ffn_norm\.weight$=f32
^blk\.24\.attn_kv_a_norm\.weight$=f32
^blk\.24\.attn_kv_a_mqa\.weight$=bf16
^blk\.24\.attn_k_b\.weight$=bf16
^blk\.24\.attn_v_b\.weight$=bf16
^blk\.24\.attn_output\.weight$=bf16
^blk\.24\.attn_q_a_norm\.weight$=f32
^blk\.24\.attn_q_a\.weight$=bf16
^blk\.24\.attn_q_b\.weight$=bf16
^blk\.34\.attn_norm\.weight$=f32
^blk\.34\.exp_probs_b\.bias$=f32
^blk\.34\.ffn_gate_inp\.weight$=f32
^blk\.34\.ffn_down_shexp\.weight$=bf16
^blk\.34\.ffn_gate_shexp\.weight$=bf16
^blk\.34\.ffn_up_shexp\.weight$=bf16
^blk\.34\.ffn_norm\.weight$=f32
^blk\.34\.attn_kv_a_norm\.weight$=f32
^blk\.34\.attn_kv_a_mqa\.weight$=bf16
^blk\.34\.attn_k_b\.weight$=bf16
^blk\.34\.attn_v_b\.weight$=bf16
^blk\.34\.attn_output\.weight$=bf16
^blk\.34\.attn_q_a_norm\.weight$=f32
^blk\.34\.attn_q_a\.weight$=bf16
^blk\.34\.attn_q_b\.weight$=bf16
^blk\.9\.attn_norm\.weight$=f32
^blk\.9\.exp_probs_b\.bias$=f32
^blk\.9\.ffn_gate_inp\.weight$=f32
^blk\.9\.ffn_down_shexp\.weight$=bf16
^blk\.9\.ffn_gate_shexp\.weight$=bf16
^blk\.9\.ffn_up_shexp\.weight$=bf16
^blk\.9\.ffn_norm\.weight$=f32
^blk\.9\.attn_kv_a_norm\.weight$=f32
^blk\.9\.attn_kv_a_mqa\.weight$=bf16
^blk\.9\.attn_k_b\.weight$=bf16
^blk\.9\.attn_v_b\.weight$=bf16
^blk\.9\.attn_output\.weight$=bf16
^blk\.9\.attn_q_a_norm\.weight$=f32
^blk\.9\.attn_q_a\.weight$=bf16
^blk\.9\.attn_q_b\.weight$=bf16
^blk\.48\.attn_norm\.weight$=f32
^blk\.48\.exp_probs_b\.bias$=f32
^blk\.48\.ffn_gate_inp\.weight$=f32
^blk\.48\.ffn_down_shexp\.weight$=bf16
^blk\.48\.ffn_gate_shexp\.weight$=bf16
^blk\.48\.ffn_up_shexp\.weight$=bf16
^blk\.48\.ffn_norm\.weight$=f32
^blk\.48\.attn_kv_a_norm\.weight$=f32
^blk\.48\.attn_kv_a_mqa\.weight$=bf16
^blk\.48\.attn_k_b\.weight$=bf16
^blk\.48\.attn_v_b\.weight$=bf16
^blk\.48\.attn_output\.weight$=bf16
^blk\.48\.attn_q_a_norm\.weight$=f32
^blk\.48\.attn_q_a\.weight$=bf16
^blk\.48\.attn_q_b\.weight$=bf16
^blk\.17\.attn_norm\.weight$=f32
^blk\.17\.exp_probs_b\.bias$=f32
^blk\.17\.ffn_gate_inp\.weight$=f32
^blk\.17\.ffn_down_shexp\.weight$=bf16
^blk\.17\.ffn_gate_shexp\.weight$=bf16
^blk\.17\.ffn_up_shexp\.weight$=bf16
^blk\.17\.ffn_norm\.weight$=f32
^blk\.17\.attn_kv_a_norm\.weight$=f32
^blk\.17\.attn_kv_a_mqa\.weight$=bf16
^blk\.17\.attn_k_b\.weight$=bf16
^blk\.17\.attn_v_b\.weight$=bf16
^blk\.17\.attn_output\.weight$=bf16
^blk\.17\.attn_q_a_norm\.weight$=f32
^blk\.17\.attn_q_a\.weight$=bf16
^blk\.17\.attn_q_b\.weight$=bf16
^blk\.44\.attn_norm\.weight$=f32
^blk\.44\.exp_probs_b\.bias$=f32
^blk\.44\.ffn_gate_inp\.weight$=f32
^blk\.44\.ffn_down_shexp\.weight$=bf16
^blk\.44\.ffn_gate_shexp\.weight$=bf16
^blk\.44\.ffn_up_shexp\.weight$=bf16
^blk\.44\.ffn_norm\.weight$=f32
^blk\.44\.attn_kv_a_norm\.weight$=f32
^blk\.44\.attn_kv_a_mqa\.weight$=bf16
^blk\.44\.attn_k_b\.weight$=bf16
^blk\.44\.attn_v_b\.weight$=bf16
^blk\.44\.attn_output\.weight$=bf16
^blk\.44\.attn_q_a_norm\.weight$=f32
^blk\.44\.attn_q_a\.weight$=bf16
^blk\.44\.attn_q_b\.weight$=bf16
^blk\.55\.attn_norm\.weight$=f32
^blk\.55\.exp_probs_b\.bias$=f32
^blk\.55\.ffn_gate_inp\.weight$=f32
^blk\.55\.ffn_down_shexp\.weight$=bf16
^blk\.55\.ffn_gate_shexp\.weight$=bf16
^blk\.55\.ffn_up_shexp\.weight$=bf16
^blk\.55\.ffn_norm\.weight$=f32
^blk\.55\.attn_kv_a_norm\.weight$=f32
^blk\.55\.attn_kv_a_mqa\.weight$=bf16
^blk\.55\.attn_k_b\.weight$=bf16
^blk\.55\.attn_v_b\.weight$=bf16
^blk\.55\.attn_output\.weight$=bf16
^blk\.55\.attn_q_a_norm\.weight$=f32
^blk\.55\.attn_q_a\.weight$=bf16
^blk\.55\.attn_q_b\.weight$=bf16
^blk\.42\.attn_norm\.weight$=f32
^blk\.42\.exp_probs_b\.bias$=f32
^blk\.42\.ffn_gate_inp\.weight$=f32
^blk\.42\.ffn_down_shexp\.weight$=bf16
^blk\.42\.ffn_gate_shexp\.weight$=bf16
^blk\.42\.ffn_up_shexp\.weight$=bf16
^blk\.42\.ffn_norm\.weight$=f32
^blk\.42\.attn_kv_a_norm\.weight$=f32
^blk\.42\.attn_kv_a_mqa\.weight$=bf16
^blk\.42\.attn_k_b\.weight$=bf16
^blk\.42\.attn_v_b\.weight$=bf16
^blk\.42\.attn_output\.weight$=bf16
^blk\.42\.attn_q_a_norm\.weight$=f32
^blk\.42\.attn_q_a\.weight$=bf16
^blk\.42\.attn_q_b\.weight$=bf16
^blk\.16\.attn_norm\.weight$=f32
^blk\.16\.exp_probs_b\.bias$=f32
^blk\.16\.ffn_gate_inp\.weight$=f32
^blk\.16\.ffn_down_shexp\.weight$=bf16
^blk\.16\.ffn_gate_shexp\.weight$=bf16
^blk\.16\.ffn_up_shexp\.weight$=bf16
^blk\.16\.ffn_norm\.weight$=f32
^blk\.16\.attn_kv_a_norm\.weight$=f32
^blk\.16\.attn_kv_a_mqa\.weight$=bf16
^blk\.16\.attn_k_b\.weight$=bf16
^blk\.16\.attn_v_b\.weight$=bf16
^blk\.16\.attn_output\.weight$=bf16
^blk\.16\.attn_q_a_norm\.weight$=f32
^blk\.16\.attn_q_a\.weight$=bf16
^blk\.16\.attn_q_b\.weight$=bf16
^output\.weight$=bf16
^token_embd\.weight$=bf16
^output_norm\.weight$=f32
^blk\.37\.attn_norm\.weight$=f32
^blk\.37\.exp_probs_b\.bias$=f32
^blk\.37\.ffn_gate_inp\.weight$=f32
^blk\.37\.ffn_down_shexp\.weight$=bf16
^blk\.37\.ffn_gate_shexp\.weight$=bf16
^blk\.37\.ffn_up_shexp\.weight$=bf16
^blk\.37\.ffn_norm\.weight$=f32
^blk\.37\.attn_kv_a_norm\.weight$=f32
^blk\.37\.attn_kv_a_mqa\.weight$=bf16
^blk\.37\.attn_k_b\.weight$=bf16
^blk\.37\.attn_v_b\.weight$=bf16
^blk\.37\.attn_output\.weight$=bf16
^blk\.37\.attn_q_a_norm\.weight$=f32
^blk\.37\.attn_q_a\.weight$=bf16
^blk\.37\.attn_q_b\.weight$=bf16
^blk\.27\.attn_norm\.weight$=f32
^blk\.27\.exp_probs_b\.bias$=f32
^blk\.27\.ffn_gate_inp\.weight$=f32
^blk\.27\.ffn_down_shexp\.weight$=bf16
^blk\.27\.ffn_gate_shexp\.weight$=bf16
^blk\.27\.ffn_up_shexp\.weight$=bf16
^blk\.27\.ffn_norm\.weight$=f32
^blk\.27\.attn_kv_a_norm\.weight$=f32
^blk\.27\.attn_kv_a_mqa\.weight$=bf16
^blk\.27\.attn_k_b\.weight$=bf16
^blk\.27\.attn_v_b\.weight$=bf16
^blk\.27\.attn_output\.weight$=bf16
^blk\.27\.attn_q_a_norm\.weight$=f32
^blk\.27\.attn_q_a\.weight$=bf16
^blk\.27\.attn_q_b\.weight$=bf16
^blk\.15\.ffn_down_exps\.weight$=bf16
^blk\.15\.ffn_gate_exps\.weight$=bf16
^blk\.15\.ffn_up_exps\.weight$=bf16
^blk\.39\.ffn_down_exps\.weight$=bf16
^blk\.39\.ffn_gate_exps\.weight$=bf16
^blk\.39\.ffn_up_exps\.weight$=bf16
^blk\.58\.ffn_down_exps\.weight$=bf16
^blk\.58\.ffn_gate_exps\.weight$=bf16
^blk\.58\.ffn_up_exps\.weight$=bf16
^blk\.5\.ffn_down_exps\.weight$=bf16
^blk\.5\.ffn_gate_exps\.weight$=bf16
^blk\.5\.ffn_up_exps\.weight$=bf16
^blk\.46\.ffn_down_exps\.weight$=bf16
^blk\.46\.ffn_gate_exps\.weight$=bf16
^blk\.46\.ffn_up_exps\.weight$=bf16
^blk\.7\.ffn_down_exps\.weight$=bf16
^blk\.7\.ffn_gate_exps\.weight$=bf16
^blk\.7\.ffn_up_exps\.weight$=bf16
^blk\.11\.ffn_down_exps\.weight$=bf16
^blk\.11\.ffn_gate_exps\.weight$=bf16
^blk\.11\.ffn_up_exps\.weight$=bf16
^blk\.40\.ffn_down_exps\.weight$=bf16
^blk\.40\.ffn_gate_exps\.weight$=bf16
^blk\.40\.ffn_up_exps\.weight$=bf16
^blk\.52\.ffn_down_exps\.weight$=bf16
^blk\.52\.ffn_gate_exps\.weight$=bf16
^blk\.52\.ffn_up_exps\.weight$=bf16
^blk\.32\.ffn_down_exps\.weight$=bf16
^blk\.32\.ffn_gate_exps\.weight$=bf16
^blk\.32\.ffn_up_exps\.weight$=bf16
^blk\.18\.ffn_down_exps\.weight$=bf16
^blk\.18\.ffn_gate_exps\.weight$=bf16
^blk\.18\.ffn_up_exps\.weight$=bf16
^blk\.30\.ffn_down_exps\.weight$=bf16
^blk\.30\.ffn_gate_exps\.weight$=bf16
^blk\.30\.ffn_up_exps\.weight$=bf16
^blk\.49\.ffn_down_exps\.weight$=bf16
^blk\.49\.ffn_gate_exps\.weight$=bf16
^blk\.49\.ffn_up_exps\.weight$=bf16
^blk\.54\.ffn_down_exps\.weight$=bf16
^blk\.54\.ffn_gate_exps\.weight$=bf16
^blk\.54\.ffn_up_exps\.weight$=bf16
^blk\.38\.ffn_down_exps\.weight$=bf16
^blk\.38\.ffn_gate_exps\.weight$=bf16
^blk\.38\.ffn_up_exps\.weight$=bf16
^blk\.2\.ffn_down_exps\.weight$=bf16
^blk\.2\.ffn_gate_exps\.weight$=bf16
^blk\.2\.ffn_up_exps\.weight$=bf16
^blk\.12\.ffn_down_exps\.weight$=bf16
^blk\.12\.ffn_gate_exps\.weight$=bf16
^blk\.12\.ffn_up_exps\.weight$=bf16
^blk\.1\.ffn_down_exps\.weight$=bf16
^blk\.1\.ffn_gate_exps\.weight$=bf16
^blk\.1\.ffn_up_exps\.weight$=bf16
^blk\.50\.ffn_down_exps\.weight$=bf16
^blk\.50\.ffn_gate_exps\.weight$=bf16
^blk\.50\.ffn_up_exps\.weight$=bf16
^blk\.36\.ffn_down_exps\.weight$=bf16
^blk\.36\.ffn_gate_exps\.weight$=bf16
^blk\.36\.ffn_up_exps\.weight$=bf16
^blk\.31\.ffn_down_exps\.weight$=bf16
^blk\.31\.ffn_gate_exps\.weight$=bf16
^blk\.31\.ffn_up_exps\.weight$=bf16
^blk\.14\.ffn_down_exps\.weight$=bf16
^blk\.14\.ffn_gate_exps\.weight$=bf16
^blk\.14\.ffn_up_exps\.weight$=bf16
^blk\.23\.ffn_down_exps\.weight$=bf16
^blk\.23\.ffn_gate_exps\.weight$=bf16
^blk\.23\.ffn_up_exps\.weight$=bf16
^blk\.21\.ffn_down_exps\.weight$=bf16
^blk\.21\.ffn_gate_exps\.weight$=bf16
^blk\.21\.ffn_up_exps\.weight$=bf16
^blk\.25\.ffn_down_exps\.weight$=bf16
^blk\.25\.ffn_gate_exps\.weight$=bf16
^blk\.25\.ffn_up_exps\.weight$=bf16
^blk\.43\.ffn_down_exps\.weight$=bf16
^blk\.43\.ffn_gate_exps\.weight$=bf16
^blk\.43\.ffn_up_exps\.weight$=bf16
^blk\.28\.ffn_down_exps\.weight$=bf16
^blk\.28\.ffn_gate_exps\.weight$=bf16
^blk\.28\.ffn_up_exps\.weight$=bf16
^blk\.53\.ffn_down_exps\.weight$=bf16
^blk\.53\.ffn_gate_exps\.weight$=bf16
^blk\.53\.ffn_up_exps\.weight$=bf16
^blk\.13\.ffn_down_exps\.weight$=bf16
^blk\.13\.ffn_gate_exps\.weight$=bf16
^blk\.13\.ffn_up_exps\.weight$=bf16
^blk\.60\.ffn_down_exps\.weight$=bf16
^blk\.60\.ffn_gate_exps\.weight$=bf16
^blk\.60\.ffn_up_exps\.weight$=bf16
^blk\.4\.ffn_down_exps\.weight$=bf16
^blk\.4\.ffn_gate_exps\.weight$=bf16
^blk\.4\.ffn_up_exps\.weight$=bf16
^blk\.57\.ffn_down_exps\.weight$=bf16
^blk\.57\.ffn_gate_exps\.weight$=bf16
^blk\.57\.ffn_up_exps\.weight$=bf16
^blk\.45\.ffn_down_exps\.weight$=bf16
^blk\.45\.ffn_gate_exps\.weight$=bf16
^blk\.45\.ffn_up_exps\.weight$=bf16
^blk\.20\.ffn_down_exps\.weight$=bf16
^blk\.20\.ffn_gate_exps\.weight$=bf16
^blk\.20\.ffn_up_exps\.weight$=bf16
^blk\.59\.ffn_down_exps\.weight$=bf16
^blk\.59\.ffn_gate_exps\.weight$=bf16
^blk\.59\.ffn_up_exps\.weight$=bf16
^blk\.6\.ffn_down_exps\.weight$=bf16
^blk\.6\.ffn_gate_exps\.weight$=bf16
^blk\.6\.ffn_up_exps\.weight$=bf16
^blk\.26\.ffn_down_exps\.weight$=bf16
^blk\.26\.ffn_gate_exps\.weight$=bf16
^blk\.26\.ffn_up_exps\.weight$=bf16
^blk\.41\.ffn_down_exps\.weight$=bf16
^blk\.41\.ffn_gate_exps\.weight$=bf16
^blk\.41\.ffn_up_exps\.weight$=bf16
^blk\.56\.ffn_down_exps\.weight$=bf16
^blk\.56\.ffn_gate_exps\.weight$=bf16
^blk\.56\.ffn_up_exps\.weight$=bf16
^blk\.22\.ffn_down_exps\.weight$=bf16
^blk\.22\.ffn_gate_exps\.weight$=bf16
^blk\.22\.ffn_up_exps\.weight$=bf16
^blk\.29\.ffn_down_exps\.weight$=bf16
^blk\.29\.ffn_gate_exps\.weight$=bf16
^blk\.29\.ffn_up_exps\.weight$=bf16
^blk\.19\.ffn_down_exps\.weight$=bf16
^blk\.19\.ffn_gate_exps\.weight$=bf16
^blk\.19\.ffn_up_exps\.weight$=bf16
^blk\.33\.ffn_down_exps\.weight$=bf16
^blk\.33\.ffn_gate_exps\.weight$=bf16
^blk\.33\.ffn_up_exps\.weight$=bf16
^blk\.51\.ffn_down_exps\.weight$=bf16
^blk\.51\.ffn_gate_exps\.weight$=bf16
^blk\.51\.ffn_up_exps\.weight$=bf16
^blk\.10\.ffn_down_exps\.weight$=bf16
^blk\.10\.ffn_gate_exps\.weight$=bf16
^blk\.10\.ffn_up_exps\.weight$=bf16
^blk\.47\.ffn_down_exps\.weight$=bf16
^blk\.47\.ffn_gate_exps\.weight$=bf16
^blk\.47\.ffn_up_exps\.weight$=bf16
^blk\.35\.ffn_down_exps\.weight$=bf16
^blk\.35\.ffn_gate_exps\.weight$=bf16
^blk\.35\.ffn_up_exps\.weight$=bf16
^blk\.8\.ffn_down_exps\.weight$=bf16
^blk\.8\.ffn_gate_exps\.weight$=bf16
^blk\.8\.ffn_up_exps\.weight$=bf16
^blk\.3\.ffn_down_exps\.weight$=bf16
^blk\.3\.ffn_gate_exps\.weight$=bf16
^blk\.3\.ffn_up_exps\.weight$=bf16
^blk\.24\.ffn_down_exps\.weight$=bf16
^blk\.24\.ffn_gate_exps\.weight$=bf16
^blk\.24\.ffn_up_exps\.weight$=bf16
^blk\.34\.ffn_down_exps\.weight$=bf16
^blk\.34\.ffn_gate_exps\.weight$=bf16
^blk\.34\.ffn_up_exps\.weight$=bf16
^blk\.9\.ffn_down_exps\.weight$=bf16
^blk\.9\.ffn_gate_exps\.weight$=bf16
^blk\.9\.ffn_up_exps\.weight$=bf16
^blk\.48\.ffn_down_exps\.weight$=bf16
^blk\.48\.ffn_gate_exps\.weight$=bf16
^blk\.48\.ffn_up_exps\.weight$=bf16
^blk\.17\.ffn_down_exps\.weight$=bf16
^blk\.17\.ffn_gate_exps\.weight$=bf16
^blk\.17\.ffn_up_exps\.weight$=bf16
^blk\.44\.ffn_down_exps\.weight$=bf16
^blk\.44\.ffn_gate_exps\.weight$=bf16
^blk\.44\.ffn_up_exps\.weight$=bf16
^blk\.55\.ffn_down_exps\.weight$=bf16
^blk\.55\.ffn_gate_exps\.weight$=bf16
^blk\.55\.ffn_up_exps\.weight$=bf16
^blk\.42\.ffn_down_exps\.weight$=bf16
^blk\.42\.ffn_gate_exps\.weight$=bf16
^blk\.42\.ffn_up_exps\.weight$=bf16
^blk\.16\.ffn_down_exps\.weight$=bf16
^blk\.16\.ffn_gate_exps\.weight$=bf16
^blk\.16\.ffn_up_exps\.weight$=bf16
^blk\.37\.ffn_down_exps\.weight$=bf16
^blk\.37\.ffn_gate_exps\.weight$=bf16
^blk\.37\.ffn_up_exps\.weight$=bf16
^blk\.27\.ffn_down_exps\.weight$=bf16
^blk\.27\.ffn_gate_exps\.weight$=bf16
^blk\.27\.ffn_up_exps\.weight$=bf16
"""

    # Prepare the line pipeline:
    # grep -v '^#' | grep -v '^$' | shorten_regex_list | optimise_regex_list | reorder_and_group

    # Split custom into lines and remove comments/blank
    input_lines = [ln for ln in custom.splitlines() if ln and not ln.strip().startswith("#")]
    _debug("Input lines count: %s", len(input_lines))

    # shorten_regex_list
    shortened = shorten_regex_list(input_lines)
    _debug("Shortened lines count: %s", len(shortened))

    # optimise_regex_list
    optimised = optimise_regex_list(shortened)
    _debug("Optimised lines count: %s", len(optimised))

    # reorder_and_group
    grouped = reorder_and_group(optimised, model_name=args.model_name, model_link=args.model_link)

    # Print grouped outputs (this mirrors the bash pipeline which prints to stdout)
    for ln in grouped:
        out(ln)

    # Then print summaries extracted from the original custom block (if any)
    summaries = extract_summaries(custom)
    if summaries:
        for ln in summaries:
            out(ln)

    # At script end: extract metrics and generate filename
    all_text = "\n".join(OUTPUTS)

    # Extract integer GPU GiB or default to 0
    m = re.search(r"^# GPU Total: ([0-9]+)\..*$", all_text, flags=re.M)
    gpuGiB = int(m.group(1)) if m else 0

    # Extract integer CPU GiB or default to 0
    m = re.search(r"^# CPU Total: ([0-9]+)\..*$", all_text, flags=re.M)
    cpuGiB = int(m.group(1)) if m else 0

    # Extract integer GPU+CPU GiB or default to 0
    m = re.search(r"^# GPU\+CPU Total: ([0-9]+)\..*$", all_text, flags=re.M)
    totalGiB = int(m.group(1)) if m else 0

    # Extract BPW or default to 0
    m = re.search(r"^# -Average BPW: ([0-9]+\.[0-9]+).*$", all_text, flags=re.M)
    bpw = m.group(1) if m else "0"

    # Extract SHA-256 and first 7 chars
    m = re.search(r"^# - Script SHA-256: ([0-9a-f]{64})$", all_text, flags=re.M)
    gsha = m.group(1) if m else ""
    shaPart = gsha[:7] if gsha else "0000000"

    # Extract full command block from the *last* "# - Command used:" entry,
    # concatenate lines, then take first 7 chars later as before.
    #
    # sed equivalent (but for the last occurrence):
    # take lines AFTER the last line matching '^# - Command used:',
    # remove '# - Command used: ' from lines if present,
    # then remove trailing backslashes and join without newlines.

    lines = all_text.splitlines()

    # Step 1: find the index of the LAST marker line
    last_marker_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("# - Command used:"):
            last_marker_idx = i

    # Step 2: collect lines only after the last marker (if any)
    fullCmd_lines = []
    if last_marker_idx is not None:
        for ln in lines[last_marker_idx + 1:]:
            # remove any leading '# - Command used: ' if present
            l = re.sub(r"^# - Command used:\s*", "", ln)

            # remove trailing backslashes (only at end of line)
            l = re.sub(r"\\+$", "", l)

            fullCmd_lines.append(l)

    # The bash pipeline then restricted to a block of comment lines and transformed them:
    #   sed -n '/^#/,/^# /p'   (select a contiguous comment block)
    #   sed -E 's/^# //;s/\\$//' (remove leading '# ' and trailing backslashes)
    #   tr '\n' ' ' | sed 's/  */ /g;s/^ //;s/ $//'  (join with spaces and normalize)
    #
    # Implement the same behavior in Python: find the first contiguous block of lines that start with '#'
    # and take all of those lines (stop when a non-comment line is encountered).
    block_lines = []
    in_block = False
    for l in fullCmd_lines:
        if not in_block:
            if l.startswith("#"):
                in_block = True
                block_lines.append(l)
            else:
                # skip lines until the first comment line
                continue
        else:
            if l.startswith("#"):
                block_lines.append(l)
            else:
                # stop at the first non-comment line after the block
                break

    # Now remove a leading '# ' (only that exact sequence) and strip any trailing backslashes again,
    # join with spaces, compress multiple spaces and trim.
    transformed = []
    for l in block_lines:
        l = re.sub(r"^# ", "", l)
        l = re.sub(r"\\+$", "", l)
        transformed.append(l)

    if transformed:
        fullCmd = " ".join(transformed)
        # compress runs of whitespace to a single space and trim ends (mimics sed 's/  */ /g;s/^ //;s/ $//')
        fullCmd = re.sub(r"\s+", " ", fullCmd).strip()
    else:
        fullCmd = ""

    if fullCmd:
        cmdPart = hashlib.sha256(fullCmd.encode("utf-8")).hexdigest()[:7]
    else:
        cmdPart = "NA"

    # Add PPL if set
    ppl_part = ""
    if PPL:
        ppl_part = f"{PPL}ppl."

    # Build dynamic filename
    whoami = getpass.getuser()
    filename = f"{whoami.upper()}-{bpw}bpw-{ppl_part}{totalGiB}GB-GGUF_{gpuGiB}GB-GPU_{cpuGiB}GB-CPU.{shaPart}_{cmdPart}.recipe"

    out("")  # blank line
    out("## THE END!")

    # Prepend model name if set
    if args.model_name:
        filename = f"{args.model_name}.{filename}"

    if not args.no_file:
        # Write OUTPUTS to file
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write("\n".join(OUTPUTS) + "\n")
            out(f"# Saved recipe to file: {filename}")
        except Exception as e:
            print(f"Error: could not write file {filename}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        out(f"# --no-file: would have written to {filename}")

if __name__ == "__main__":
    main()
