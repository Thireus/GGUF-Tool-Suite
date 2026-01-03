#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** tensor_metrics.py my failed initial attempt at finding    **#
#** patterns to tensor quantisation sensitiveness. Don't use. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jan-03-2026 -------------------- **#
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
#** Copyright © 2025 - Thireus.           ₚₗₑₐₛₑ, 𝒹ₒₙ'ₜ ₕᵢᵣₑ ₘₑ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Dependencies: pip install tabulate pandas sentencepiece numpy==1.26.4
# Dependencies: pip install "gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py"

import sys
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

# --- Use the GGUFReader from gguf-py ---
try:
    from gguf.gguf_reader import GGUFReader
except ImportError:
    sys.stderr.write(
        "Error: could not import GGUFReader.\n"
        "Ensure ik_llama.cpp/gguf-py is in PYTHONPATH or install via pip.\n"
    )
    sys.exit(1)

# --- BPW lookup table for GGUF quant dtypes ---
BPW_TABLE = {
    'F32': 32, 'F16': 16, 'BF16': 16, 'F8': 8,
    # fill known Q/IQ formats
    'Q8_0': 8.25,
    'Q8_0_R8': 8.5,
    'Q8_KV': 8,
    'Q6_K': 6.5625,
    'Q6_0': 6.5,
    'IQ6_K': 6.5,
    'Q6_0_R4': 6.5,
    'Q5_K': 5.5,
    'Q5_0': 5.1875,
    'Q5_1': 5.375,
    'IQ5_KS_R4': 5.25,
    'IQ5_KS': 5.25,
    'IQ5_K': 5.5,
    'IQ5_K_R4': 5.5,
    'Q5_0_R4': 5.5,
    'IQ4_XS': 4.25,
    'Q4_K': 4.5,
    'Q4_1': 4.375,
    'IQ4_KS_R4': 4.25,
    'IQ4_KS': 4.25,
    'IQ4_NL': 4.1,
    'IQ4_KT': 4,
    'Q4_0': 4.1875,
    'IQ4_K': 4.5,
    'IQ4_K_R4': 4.5,
    'IQ4_KSS': 4,
    'Q4_0_R8': 4.5,
    'IQ4_XS_R8': 4.25,
    'IQ4_NL_R4': 4.5,
    'IQ3_K': 3.4375,
    'IQ3_S': 3.44,
    'IQ3_XXS': 3.06,
    'IQ3_KT': 3.125,
    'Q3_K': 3.4375,
    'IQ3_K_R4': 3.44,
    'IQ3_XXS_R4': 3.06,
    'IQ3_S_R4': 3.44,
    'IQ3_KL': 4,
    'IQ3_M': 3.66,
    'IQ3_XS': 3.3,
    'IQ2_KS': 2.1875,
    'IQ2_BN': 2,
    'IQ2_KT': 2.125,
    'IQ2_S': 2.5,
    'IQ2_XXS': 2.06,
    'IQ2_XS': 2.31,
    'IQ2_K': 2.375,
    'Q2_K': 2.625,
    'IQ2_K_R4': 2.375,
    'IQ2_BN_R4': 2,
    'IQ2_XXS_R4': 2.06,
    'IQ2_XS_R4': 2.31,
    'IQ2_M': 2.7,
    'IQ2_M_R4': 2.7,
    'IQ1_M': 1.75,
    'IQ1_S': 1.56,
    'IQ1_S_R4': 1.5,
    'IQ1_BN': 1.62,
    'IQ1_KT': 1.75,
    'IQ1_M_R4': 1.75
}

# Default bits-per-word for any unmatched quantizable format
def get_bpw(key):
    return BPW_TABLE.get(key.upper())


def quantize_codes(codes, native_bits, target_bits):
    """
    Simulate bit-reduction of integer codes by right-shifting out bits.
    """
    if target_bits >= native_bits:
        return codes.copy()
    shift = native_bits - int(target_bits)
    q = (codes >> shift) << shift
    return q


def simulate_quant_loss(arr, dtype_name, native_bits, targets):
    """
    For each target in targets (bits), simulate quantization loss and return dict of %noise.
    """
    # get original codes as integers
    # reuse compute_code_metrics extraction behavior
    # for simplicity, if float data, view as codes
    if dtype_name in ('BF16', 'F16', 'F32', 'F8') or re.match(r'I?Q\d+', dtype_name):
        # flatten raw codes via compute_code_metrics hack
        # here we recover codes via same logic as compute
        # but assume arr has been viewed already in compute function
        # re-extract codes
        codes = None
        # BF16
        if dtype_name == 'BF16':
            if arr.dtype == np.uint16:
                codes = arr.ravel()
            elif arr.dtype == np.float32:
                codes = (arr.view(np.uint32) >> 16).astype(np.uint16).ravel()
            else:
                codes = arr.view(np.uint8).view(np.uint16).ravel()
        elif dtype_name == 'F16':
            codes = arr.view(np.uint16).ravel()
        elif dtype_name == 'F32':
            codes = arr.view(np.uint32).ravel()
        elif dtype_name == 'F8':
            codes = arr.view(np.uint8).ravel()
        else:
            m = re.match(r'(I?Q)(\d+)', dtype_name)
            if m:
                bits = int(m.group(2))
                codes = arr.view(np.uint8).ravel() & ((1 << bits) - 1)
        if codes is None:
            return {}
    else:
        return {}

    var = np.mean(codes.astype(np.float32)**2)
    losses = {}
    nat = int(native_bits)
    for t in targets:
        tb = int(t)
        try:
            qc = quantize_codes(codes, nat, tb)
            mse = np.mean((codes.astype(np.float32) - qc.astype(np.float32))**2)
            losses[f'loss_{tb}b_%noise'] = 100 * mse / var if var>0 else None
        except Exception:
            losses[f'loss_{tb}b_%noise'] = None
    return losses

def compute_code_metrics(arr: np.ndarray, dtype_name: str):
    """
    Compute entropy, occupancy, sparsity, and range metrics for an array of quantized codes or floats.
    Supports:
      - BF16: float32 (high half), uint16, raw uint8
      - F16: float16
      - F32: float32 raw patterns
      - F8: 8-bit float-like stored as uint8
      - Qx_y or IQx_y: arbitrary x-bit quantizations
    """
    # BF16
    if dtype_name == 'BF16':
        if arr.dtype == np.float32:
            codes = (arr.view(np.uint32) >> 16).astype(np.uint16).ravel()
            range_span = arr.max() - arr.min()
        elif arr.dtype == np.uint16:
            codes = arr.ravel()
            float_vals = (codes.astype(np.uint32) << 16).view(np.float32)
            range_span = float_vals.max() - float_vals.min()
        elif arr.dtype == np.uint8:
            codes = arr.view(np.uint16).ravel()
            float_vals = (codes.astype(np.uint32) << 16).view(np.float32)
            range_span = float_vals.max() - float_vals.min()
        else:
            raise ValueError(f"BF16 expected float32, uint16, or uint8 but got {arr.dtype}")
        bits = 16
    # F16
    elif dtype_name == 'F16':
        if arr.dtype != np.float16:
            raise ValueError(f"F16 expected float16 but got {arr.dtype}")
        codes = arr.view(np.uint16).ravel()
        range_span = float(arr.astype(np.float32).max() - arr.astype(np.float32).min())
        bits = 16
    # F32
    elif dtype_name == 'F32':
        if arr.dtype != np.float32:
            raise ValueError(f"F32 expected float32 but got {arr.dtype}")
        codes = arr.view(np.uint32).ravel()
        range_span = float(arr.max() - arr.min())
        bits = 32
    # F8 (8-bit float variants)
    elif dtype_name == 'F8':
        # stored as uint8
        if arr.dtype != np.uint8:
            raise ValueError(f"F8 expected uint8 storage but got {arr.dtype}")
        codes = arr.ravel()
        # Without spec, treat codes as indices; range via codes
        range_span = int(codes.max() - codes.min())
        bits = 8
    else:
        # Qx or IQx
        m = re.match(r'(I?Q)(\d+)', dtype_name)
        if m:
            bw = int(m.group(2))
            bits = bw
            codes = arr.view(np.uint8).ravel() & ((1 << bits) - 1)
            range_span = int(codes.max() - codes.min())
        else:
            raise ValueError(f"Unsupported tensor_type '{dtype_name}' for metrics computation")

    # compute entropy
    vals, cnts = np.unique(codes, return_counts=True)
    p = cnts / cnts.sum()
    entropy_bits = -float(np.sum(p * np.log2(p)))

    occupancy_ratio = len(vals) / (2**bits)
    zero_fraction = float(np.count_nonzero(codes == 0) / codes.size)

    metrics = {
        'entropy_bits': entropy_bits,
        'occupancy_ratio': occupancy_ratio,
        'zero_fraction': zero_fraction,
        'range_span': range_span,
    }

    # subspace occupancy for BF16/F16
    if bits == 16 and dtype_name in ('BF16', 'F16'):
        exp_mask, man_mask = 0x7F80, 0x007F
        exps = (vals & exp_mask) >> 7
        mans = vals & man_mask
        metrics['exponent_occupancy'] = len(np.unique(exps)) / 2**8
        metrics['mantissa_occupancy'] = len(np.unique(mans)) / 2**7

    # debug info
    sys.stderr.write(f"    Unique codes: {len(vals)} / {2**bits}\n")
    native_bits = arr.dtype.itemsize * 8
    bpw = get_bpw(dtype_name)
    metrics['bpw'] = bpw if bpw is not None else native_bits
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute quantization & entropy metrics for GGUF tensors"
    )
    parser.add_argument('gguf_file')
    parser.add_argument('-o', '--output-csv', default=None)
    parser.add_argument('-p', '--pattern', default=None)
    parser.add_argument('-t', '--targets',
                        help="Comma-separated list of dtypes or bitwidths (or ALL)",
                        default=None)
    args = parser.parse_args()

    pattern = re.compile(args.pattern) if args.pattern else None
    reader = GGUFReader(Path(args.gguf_file))
    tensors = reader.tensors
    total = len(tensors)
    records = []

    for idx, tensor in enumerate(reader.tensors, 1):
        name = tensor.name
        sys.stderr.write(f"[{idx}/{total}] Processing tensor '{name}'...\n")
        if pattern and not pattern.search(name):
            sys.stderr.write("  Skipped by pattern.\n")
            continue
        try:
            arr = tensor.data
        except:
            sys.stderr.write("  Could not load data, skipped.\n")
            continue
        dtype_name = tensor.tensor_type.name
        # supported types
        if not re.match(r'(BF16|F16|F32|F8|I?Q\d+)', dtype_name):
            sys.stderr.write(f"  Unsupported tensor type '{dtype_name}', skipped.\n")
            continue
        try:
            base_metrics = compute_code_metrics(arr, dtype_name)
        except Exception as e:
            sys.stderr.write(f"  Error computing metrics: {e}, skipped.\n")
            continue
        # parse targets
        targets = []
        if args.targets:
            if args.targets.upper() == 'ALL':
                ref_bpw = get_bpw(dtype_name.upper())
                if ref_bpw is None:
                    raise ValueError(f"Unknown dtype_name: {dtype_name}")
                targets = sorted({
                    int(bpw) for k in BPW_TABLE
                    if (bpw := get_bpw(k)) is not None and bpw < ref_bpw
                })
            else:
                for tok in args.targets.split(','):
                    tok = tok.strip()
                    if tok.isdigit():
                        targets.append(int(tok))
                    else:
                        bpw = get_bpw(tok)
                        if bpw is None:
                            sys.stderr.write(f"Warning: unknown quant dtype '{tok}', skipped.\n")
                        else:
                            targets.append(int(bpw))
            targets = sorted(set(targets))
        if targets:
            loss_metrics = simulate_quant_loss(arr, dtype_name,
                                               base_metrics['bpw'], targets)
            base_metrics.update(loss_metrics)
        base_metrics.update({'tensor_name': name,
                             'shape': tuple(int(d) for d in tensor.shape),
                             'dtype': dtype_name})
        records.append(base_metrics)
        sys.stderr.write("  Metrics computed.\n")

    if not records:
        sys.stderr.write("No tensors selected after filtering.\n")
        sys.exit(0)

    df = pd.DataFrame(records)

    # reorder columns
    base_cols = ['tensor_name', 'dtype', 'bpw', 'shape', 'entropy_bits',
                'occupancy_ratio', 'zero_fraction', 'range_span']
    # include any loss columns dynamically
    loss_cols = [c for c in df.columns if c.startswith('loss_')]
    other_cols = [c for c in df.columns if c not in base_cols + loss_cols]
    cols = base_cols + other_cols + loss_cols
    # filter to existing columns
    cols = [c for c in cols if c in df.columns]

    df = df[cols]
    print(df.to_markdown(index=False))
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        sys.stderr.write(f"Metrics saved to {args.output_csv}\n")

if __name__ == '__main__':
    main()
