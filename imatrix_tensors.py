#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** extract_tensors_strict.py extracts tensor names from      **#
#** llama.cpp/ik_llama.cpp imatrix file.                      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Feb-10-2026 -------------------- **#
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
    END   = b'.weight'  # user-configurable via --end-marker (default ".weight")

Each found segment (the raw bytes between START and END) is treated as a
single candidate name. After optional stripping of leading/trailing NULs
and whitespace, the candidate is accepted only if every byte belongs to the
allowed character set (default: a-z0-9_.-). If any unauthorized byte is
present, the whole segment is skipped.

Usage examples:
    python3 extract_tensors_strict.py model.imatrix
    python3 extract_tensors_strict.py --show-offsets --end-marker ".weight,.bias" model.imatrix
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Iterable, Tuple, List

START = b'\x00\x00\x00'

def parse_markers(marker_arg: str) -> List[bytes]:
    """
    Parse a comma-separated list of end-marker specifications into a list of bytes objects.

    Supported formats for each marker:
      - plain text (e.g. ".weight") -> encoded as UTF-8 bytes
      - hex string with 0x prefix (e.g. 0xD500000000) -> bytes.fromhex(...)
      - backslash escapes (e.g. "\\xD5\\x00\\x00") -> processed with unicode_escape and encoded as latin-1

    Whitespace around comma-separated items is ignored. Empty items are skipped.
    """
    markers: List[bytes] = []
    parts = [p.strip() for p in marker_arg.split(',') if p.strip() != '']
    for p in parts:
        # hex form starting with 0x
        if p.lower().startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in p[2:]):
            try:
                hb = bytes.fromhex(p[2:])
                markers.append(hb)
                continue
            except Exception:
                # fall through to other parsing attempts
                pass
        # backslash-escaped bytes like \xD5\x00
        if '\\x' in p or '\\u' in p or '\\' in p:
            try:
                # decode escape sequences into unicode string, then map to latin-1 bytes (0-255)
                decoded = bytes(p, 'utf-8').decode('unicode_escape')
                markers.append(decoded.encode('latin-1'))
                continue
            except Exception:
                # fall through to plain utf-8
                pass
        # default: treat as UTF-8 text
        markers.append(p.encode('utf-8'))
    return markers

def find_segments(data: bytes, start: bytes = START, ends: List[bytes] | None = None) -> Iterable[Tuple[int,int,bytes]]:
    """Yield (start_index, end_index, segment_bytes) for each occurrence of start...end.
       Overlapping occurrences are allowed (advance search by 1 after each start).

       This function searches for any of the provided end markers (full match, no wildcards).
    """
    if ends is None:
        ends = [b'.weight']
    pos = 0
    dlen = len(data)
    start_len = len(start)

    while True:
        s_idx = data.find(start, pos)
        if s_idx == -1:
            break
        # find the earliest end marker occurrence after the start
        best_e_idx = None
        for end in ends:
            search_from = s_idx + start_len
            e_found = data.find(end, search_from)
            if e_found != -1:
                # e_found is the start index of the end marker; segment end is e_found
                if best_e_idx is None or e_found < best_e_idx:
                    best_e_idx = e_found
        if best_e_idx is None:
            pos = s_idx + 1
            continue
        segment = data[s_idx + start_len: best_e_idx]
        yield s_idx, best_e_idx, segment
        pos = s_idx + 1

def find_segments_by_end(data: bytes, start: bytes = START, ends: List[bytes] | None = None,
                         max_scan_back: int = 4096) -> Iterable[Tuple[int,int,bytes]]:
    """
    Faster end-first search.

    This function locates occurrences of any end marker in the data (scanning forward),
    then for each occurrence scans backward at most `max_scan_back` bytes
    looking for the *last* occurrence of `start` before that `end`. If such a
    `start` is found and lies before the `end`, the segment between them is
    yielded as (s_idx, e_idx, segment_bytes).

    Rationale:
      - For large files it's typically faster to find the relatively short end markers
        and then search a limited backward window for the `start` marker rather than
        scanning for starts and calling find(end) repeatedly.
      - `max_scan_back` should be set to (max_length + len(start) + small slack)
        to avoid scanning unnecessarily large windows.

    Notes:
      - Overlapping occurrences of end are allowed (we advance the search by 1
        after each found end marker).
      - If the `start` marker is not found within the backward window, this
        end occurrence is skipped (per user's requested strategy).
    """
    if ends is None:
        ends = [b'.weight']
    pos = 0
    dlen = len(data)
    start_len = len(start)

    # Ensure max_scan_back is positive
    if max_scan_back < start_len:
        max_scan_back = start_len

    while True:
        # Find the earliest next occurrence among all end markers
        next_idx = None
        next_end = None
        for end in ends:
            idx = data.find(end, pos)
            if idx != -1:
                if next_idx is None or idx < next_idx:
                    next_idx = idx
                    next_end = end
        if next_idx is None or next_end is None:
            break
        e_idx = next_idx  # full end start index is the index where the chosen end marker begins
        # window to scan backwards (inclusive of possible start)
        window_start = e_idx - max_scan_back
        if window_start < 0:
            window_start = 0
        window = data[window_start:e_idx+len(next_end)]  # slice up to the end marker ends
        # find last occurrence of start within window
        rel_s = window.rfind(start)
        if rel_s != -1:
            s_idx = window_start + rel_s
            # segment is the bytes between the end of start marker and the end of end marker
            segment = data[s_idx + start_len: e_idx+len(next_end)]
            yield s_idx, e_idx, segment
            # advance past this end occurrence to allow overlapping ends
            pos = next_idx + 1
            continue
        # if no start found within backward window, skip this end and continue searching
        pos = next_idx + 1

def make_allowed_set() -> set:
    """Return a set of allowed byte values (integers). Default = lowercase a-z, digits, underscore, dot, hyphen, parentheses and space."""
    allowed = set()
    for c in b'abcdefghijklmnopqrstuvwxyz0123456789_.-() ':
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
    p.add_argument('--min-length', type=int, default=6, help='Minimum name length after stripping (default: 6)')
    p.add_argument('--max-length', type=int, default=200, help='Maximum name length (default: 200)')
    p.add_argument('--no-strip-nulls', action='store_true', help='Do not strip leading/trailing NUL/whitespace bytes (default: strip them)')
    p.add_argument('--mmap', action='store_true', help='Use memory-mapped IO for large files')
    p.add_argument('--show-offsets', action='store_true', help='Print 0xstart-0xend alongside each name')
    p.add_argument('-o', '--output', help='Write results to this file instead of stdout')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose progress info')
    p.add_argument('-q', '--quiet', action='store_true', help='Quiet: suppress stdout output (still writes to --output if specified)')

    p.add_argument('--map-file', help='Path to a .map file to enrich with imatrix hashes')
    p.add_argument('--output-map-file', help='Path where the enriched map file will be written (required if --map-file is used)')

    # Performance tuning option (new): how many bytes to scan backwards from each END marker.
    p.add_argument('--max-back', type=int, default=0,
                   help='(Advanced) Maximum bytes scanned backwards from each END occurrence when using the fast end-first search. '
                        'Default: computed from --max-length (+32 slack). Set to 0 to use default.')

    # User-definable end marker(s). Multiple markers may be provided comma-separated.
    p.add_argument('--end-marker', type=str, default='.weight',
                   help='Comma-separated end markers (strings or escaped bytes). Default: ".weight".')

    # Tensor name filtering: list of regexes (comma-separated). Default matches names ending with ".weight"
    # and also variants like "tensor.weight (characteristic)".
    p.add_argument('--tensor-regex', type=str,
                   default=r'.*\.weight,.*\.weight \(.*\)',
                   help='Comma-separated regex patterns to match final tensor names against (default: ".*\\.weight,.*\\.weight \\(.*\\)"). '
                        'A segment is accepted only if the decoded name FULLMATCHES at least one pattern.')

    # Stripping regexes to remove parts from resulting tensor names (applied after matching)
    p.add_argument('--strip-tensor-regex', type=str,
                   default=r' \(.+\)',
                   help='Comma-separated regex patterns to remove from accepted tensor names before storing (default removes the trailing parenthetical " (.*)").')

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
    allowed = make_allowed_set()

    # Compile tensor regex patterns (fullmatch will be used)
    try:
        regex_parts = [patt.strip() for patt in args.tensor_regex.split(',') if patt.strip() != '']
        if not regex_parts:
            # empty -> no matches allowed
            compiled_regexes = []
        else:
            compiled_regexes = [re.compile(patt) for patt in regex_parts]
    except re.error as e:
        print(f"ERROR: invalid --tensor-regex pattern: {e}", file=sys.stderr)
        sys.exit(2)

    # Compile strip-tensor-regex patterns (these are applied to accepted names BEFORE storing)
    try:
        strip_parts = [patt.strip() for patt in args.strip_tensor_regex.split(',') if patt.strip() != '']
        if not strip_parts:
            compiled_strip_regexes = []
        else:
            compiled_strip_regexes = [re.compile(patt) for patt in strip_parts]
    except re.error as e:
        print(f"ERROR: invalid --strip-tensor-regex pattern: {e}", file=sys.stderr)
        sys.exit(2)

    # Parse end markers
    end_markers = parse_markers(args.end_marker)
    if args.verbose:
        marker_list_str = ', '.join([repr(m) for m in end_markers])
        print(f"Using end markers: {marker_list_str}", file=sys.stderr)

    data = open_data(args.file, use_mmap=args.mmap)
    if args.verbose:
        try:
            dlen = len(data)
        except Exception:
            dlen = "unknown"
        print(f"Loaded {dlen:,} bytes from {args.file}", file=sys.stderr)

    results = []

    # Track all accepted tensor names (case-sensitive, exact names) for map enrichment.
    found_names = set()

    seg_count = 0
    accepted = 0
    skipped = 0

    # Compute max_scan_back for the fast algorithm. Default = max_length + len(START) + slack(32)
    if args.max_back and args.max_back > 0:
        max_scan_back = args.max_back
    else:
        max_scan_back = args.max_length + len(START) + 32

    # Use the fast end-first search routine (much faster on large files).
    for s_idx, e_idx, segment in find_segments_by_end(data, START, end_markers, max_scan_back=max_scan_back):
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

        # New requirement: check tensor name matches one of the provided regex patterns
        # (using fullmatch semantics). If no compiled patterns exist, treat as no-match.
        matched = False
        for cre in compiled_regexes:
            if cre.fullmatch(name):
                matched = True
                break
        if not matched:
            skipped += 1
            continue

        # Apply strip-tensor-regex substitutions at the very end (before storing results and found_names)
        stored_name = name
        if compiled_strip_regexes:
            for scre in compiled_strip_regexes:
                stored_name = scre.sub('', stored_name)
            # optional: normalize surrounding whitespace after stripping
            stored_name = stored_name.strip()

        # Record in the global found_names set (used for map matching). This preserves all accepted names.
        # Note: per new requirement, we store the stripped version.
        found_names.add(stored_name)

        accepted += 1
        if args.show_offsets:
            out_item = f"{stored_name}\t0x{s_idx:x}-0x{e_idx:x}"
        else:
            out_item = stored_name

        results.append(out_item)

        # optional verbose progress
        if args.verbose and seg_count % 1000 == 0:
            print(f"...scanned {seg_count:,} end-markers, accepted {accepted:,}, skipped {skipped:,}", file=sys.stderr)

    # output tensor names (respecting -q/--quiet)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out_f:
            for item in results:
                out_f.write(item + '\n')
    else:
        if not args.quiet:
            for item in results:
                print(item)

    if args.verbose:
        print(f"Done. End-markers scanned (approx): {seg_count:,}; accepted: {accepted:,}; skipped: {skipped:,}", file=sys.stderr)

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
