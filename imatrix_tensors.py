#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** extract_tensors_strict.py extracts tensor names from      **#
#** llama.cpp/ik_llama.cpp imatrix file.                      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Feb-07-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.              ᵢ𝓰ₙₒᵣₐₙ𝒸ₑ ᵢₛ ᵦₗᵢₛₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
Strict extraction of tensor names from a llama.cpp / ik_llama imatrix file.

Searches for byte sequences between:
    START = b'\x00\x00\x00'
    END   = b'\x48\x03\x00\x00\x00'

Each found segment (the raw bytes between START and END) is treated as a
single candidate name. After optional stripping of leading/trailing NULs
and whitespace, the candidate is accepted only if every byte belongs to the
allowed character set (default: a-z0-9_.-). If any unauthorized byte is
present, the whole segment is skipped.

Usage examples:
    python3 extract_tensors_strict.py model.imatrix
    python3 extract_tensors_strict.py --allow-upper --unique --show-offsets model.imatrix
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Iterable, Tuple

START = b'\x00\x00\x00'
END = b'\x48\x03\x00\x00\x00'  # hex 4803000000

def find_segments(data: bytes, start: bytes = START, end: bytes = END) -> Iterable[Tuple[int,int,bytes]]:
    """Yield (start_index, end_index, segment_bytes) for each occurrence of start...end.
       Overlapping occurrences are allowed (advance search by 1 after each start).
    """
    pos = 0
    dlen = len(data)
    while True:
        s_idx = data.find(start, pos)
        if s_idx == -1:
            break
        e_idx = data.find(end, s_idx + len(start))
        if e_idx == -1:
            pos = s_idx + 1
            continue
        segment = data[s_idx + len(start): e_idx]
        yield s_idx, e_idx, segment
        pos = s_idx + 1

def make_allowed_set(allow_upper: bool) -> set:
    """Return a set of allowed byte values (integers). Default = lowercase a-z, digits, underscore, dot, hyphen."""
    allowed = set()
    for c in b'abcdefghijklmnopqrstuvwxyz0123456789_.-':
        allowed.add(c)
    if allow_upper:
        for c in b'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            allowed.add(c)
    return allowed

def clean_segment(segment: bytes, strip_nulls: bool) -> bytes:
    """Optionally strip leading/trailing NUL and common whitespace bytes."""
    if strip_nulls:
        return segment.strip(b'\x00 \t\r\n')
    else:
        return segment

def open_data(path: str, use_mmap: bool=False) -> bytes:
    if not use_mmap:
        with open(path, 'rb') as f:
            return f.read()
    import mmap
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # Note: we purposely leave file/mmap alive until program exit
    return mm

def main(argv=None):
    p = argparse.ArgumentParser(description="Strictly extract tensor names from an imatrix file.")
    p.add_argument('file', help='Path to the imatrix file')
    p.add_argument('--allow-upper', action='store_true', help='Allow A-Z in names (default: lowercase only)')
    p.add_argument('--min-length', type=int, default=1, help='Minimum name length after stripping (default: 1)')
    p.add_argument('--max-length', type=int, default=200, help='Maximum name length (default: 200)')
    p.add_argument('--no-strip-nulls', action='store_true', help='Do not strip leading/trailing NUL/whitespace bytes (default: strip them)')
    p.add_argument('--unique', action='store_true', help='Only print unique names (preserve first appearance)')
    p.add_argument('--mmap', action='store_true', help='Use memory-mapped IO for large files')
    p.add_argument('--show-offsets', action='store_true', help='Print 0xstart-0xend alongside each name')
    p.add_argument('-o', '--output', help='Write results to this file instead of stdout')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose progress info')
    
    p.add_argument('--map-file', help='Path to a .map file to enrich with imatrix hashes')
    p.add_argument('--output-map-file', help='Path where the enriched map file will be written (required if --map-file is used)')

    args = p.parse_args(argv)

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    if args.map_file and not args.output_map_file:
        print("ERROR: --map-file requires --output-map-file to be specified", file=sys.stderr)
        sys.exit(2)

    if args.map_file and not os.path.isfile(args.map_file):
        print(f"ERROR: map file not found: {args.map_file}", file=sys.stderr)
        sys.exit(2)

    strip_nulls = not args.no_strip_nulls
    allowed = make_allowed_set(args.allow_upper)

    data = open_data(args.file, use_mmap=args.mmap)
    if args.verbose:
        try:
            dlen = len(data)
        except Exception:
            dlen = "unknown"
        print(f"Loaded {dlen:,} bytes from {args.file}", file=sys.stderr)

    seen = set()
    results = []

    # Track all accepted tensor names (case-sensitive, exact names) for map enrichment.
    found_names = set()

    seg_count = 0
    accepted = 0
    skipped = 0

    for s_idx, e_idx, segment in find_segments(data, START, END):
        seg_count += 1
        cleaned = clean_segment(segment, strip_nulls)
        if not cleaned:
            skipped += 1
            continue
        if len(cleaned) < args.min_length or len(cleaned) > args.max_length:
            skipped += 1
            continue
        # Check every byte is allowed
        bad = False
        for b in cleaned:
            if b not in allowed:
                bad = True
                break
        if bad:
            skipped += 1
            continue
        # Accept: decode as ascii (safe because all bytes are in ASCII ranges)
        try:
            name = cleaned.decode('ascii')
        except Exception:
            # should not happen given allowed set, but guard anyway
            skipped += 1
            continue

        # Record in the global found_names set (used for map matching). This preserves all accepted names.
        found_names.add(name)

        # For output results, apply unique option if requested
        if args.unique:
            if name in seen:
                # do not count as duplicate accepted for final printed results,
                # but we already recorded it in found_names.
                continue
            seen.add(name)

        accepted += 1
        if args.show_offsets:
            results.append(f"{name}\t0x{s_idx:x}-0x{e_idx:x}")
        else:
            results.append(name)
        # optional verbose progress
        if args.verbose and seg_count % 1000 == 0:
            print(f"...scanned {seg_count:,} segments, accepted {accepted:,}, skipped {skipped:,}", file=sys.stderr)

    # output tensor names
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out_f:
            for item in results:
                out_f.write(item + '\n')
    else:
        for item in results:
            print(item)

    if args.verbose:
        print(f"Done. Segments scanned: {seg_count:,}; accepted: {accepted:,}; skipped: {skipped:,}", file=sys.stderr)

    # If requested, enrich the map file with imatrix sha256 hash metadata.
    if args.map_file:
        import hashlib

        # Compute sha256 of the imatrix file we just processed.
        try:
            # data might be an mmap or bytes; hashlib accepts both.
            sha256 = hashlib.sha256(data).hexdigest()
        except Exception:
            # As a fallback, read file from disk into memory to compute hash.
            with open(args.file, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()

        if args.verbose:
            print(f"Computed imatrix sha256: {sha256}", file=sys.stderr)

        # Read original map file and process line-by-line.
        out_lines = []
        with open(args.map_file, 'r', encoding='utf-8') as mf:
            for raw_line in mf:
                line = raw_line.rstrip('\n')
                if not line.strip():
                    # Preserve blank lines
                    out_lines.append(line)
                    continue
                # Split off any existing ":imatrix=..." suffix so we can replace/remove it reliably.
                base_line, *im_part = line.split(':imatrix=', 1)
                # base_line now contains everything up to but not including any existing imatrix metadata.

                # Parse base_line into expected 7 fields:
                # shard_name:shard_hash:tensor_name:tensor_shape:quantization_type:tensor_elements:tensor_bytes
                parts = base_line.split(':', 6)  # max 7 fields
                if len(parts) < 7:
                    # Unexpected format — preserve the sanitized base_line (without imatrix) to avoid duplicating old imatrix data.
                    # This follows the rule: if tensor name isn't demonstrably present, do not add imatrix metadata.
                    out_lines.append(base_line)
                    continue

                tensor_name = parts[2]

                # Exact case-sensitive full-match required.
                if tensor_name in found_names:
                    # Add (or replace) imatrix metadata after the tensor_bytes field.
                    new_line = base_line + f":imatrix={sha256}"
                    out_lines.append(new_line)
                else:
                    # Ensure any existing imatrix metadata is removed.
                    out_lines.append(base_line)

        # Write enriched map file
        with open(args.output_map_file, 'w', encoding='utf-8') as outf:
            for idx, ol in enumerate(out_lines):
                # restore trailing newlines
                outf.write(ol)
                # write newline for all but possibly the last line if original file didn't have final newline;
                # to be consistent and safe, always write newline
                outf.write('\n')

        if args.verbose:
            print(f"Wrote enriched map file to {args.output_map_file}", file=sys.stderr)

if __name__ == '__main__':
    main()
