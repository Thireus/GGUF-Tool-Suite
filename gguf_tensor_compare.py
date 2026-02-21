#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** gguf_tensor_compare.py a tool to compare the quality of   **#
#** quantized tensors mathematically.                         **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Feb-21-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.    ₐₗₗ ᵧₒᵤᵣ ᵣₐₘ ₐᵣₑ ᵦₑₗₒₙ𝓰 ₜₒ ᵤₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
gguf_tensor_compare.py

Compare multiple single-tensor baseline GGUF files against matching single-tensor
quantized GGUF files. Computes value-space metrics, optional per-channel metrics,
and—when possible—KLD/logit-based metrics.

How it works (two modes for KLD):
 1. **Per-tensor-as-logits mode** (--tensors-are-logits): the script will load
    the baseline and quantized tensors (which must be precomputed logits or
    probabilities), optionally reshape them using --logits-shape (format: N,L,V
    or L,V or V), and compute average KL(softmax(baseline) || softmax(quant)).
    If the tensors are already probabilities, pass --tensors-are-probs.

 2. **Model-based logit-KL** (--compute-logit-kl): unchanged from previous
    behavior; attempts to compute logits by running a model (Transformers or
    llama.cpp). This is useful when you have a model binary or HF model.

Usage examples:
  # treat GGUF tensors as logits (no model required)
  python gguf_tensor_compare.py --baseline-dir ./b --quant-dir ./q \
      --pair-by name --tensors-are-logits --logits-shape "10,128,32000" --out with_kl.csv

  # if tensors are probabilities already
  python gguf_tensor_compare.py --baseline-files b1.gguf b2.gguf \
      --quant-files q1.gguf q2.gguf --tensors-are-logits --tensors-are-probs --out probs_kl.csv

Notes and limitations:
 - This only works if the GGUF single-tensor files actually contain the
   *logits/probabilities* computed on the same calibration prompts in the same
   layout (same N,L,V). If they are model weights (e.g. weight matrices), you
   cannot derive logits from them without a model/runtime.
 - If you don't know whether your tensors are logits, inspect their shapes and
   content: logits typically have 3D shapes (batch, seq_len, vocab) or 2D
   (seq_len, vocab) depending on how they were saved. Values can be large and
   unbounded (logits) or in [0,1] and sum-to-1 along vocab axis (probabilities).
 - Use per-channel and value-space metrics to triage tensors first; the logits
   mode is useful when you already have saved model outputs per-tensor.

Dependencies:
  numpy, pandas, tqdm
  optional: gguf (pygguf) or gguf-parser to read GGUF files
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import json
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

# Try available GGUF readers
_has_gguf = False
_has_gguf_parser = False
GGUFReader = None

# Try to import the reader similar to canonical usage
try:
    from gguf.gguf_reader import GGUFReader  # type: ignore
    _has_gguf = True
except Exception:
    try:
        import gguf  # fallback module
        GGUFReader = getattr(gguf, "GGUFReader", None)
        if GGUFReader is not None:
            _has_gguf = True
    except Exception:
        _has_gguf = False

try:
    from gguf_parser import GGUFParser
    _has_gguf_parser = True
except Exception:
    _has_gguf_parser = False

from tqdm import tqdm

EPS = 1e-12


def js_divergence(a: np.ndarray, b: np.ndarray, nbins: int = 256, eps: float = EPS) -> float:
    mn = float(min(np.min(a), np.min(b)))
    mx = float(max(np.max(a), np.max(b)))
    if mn == mx:
        return 0.0
    bins = np.linspace(mn, mx, nbins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = pa + eps
    pb = pb + eps
    pa = pa / np.sum(pa)
    pb = pb / np.sum(pb)
    m = 0.5 * (pa + pb)
    kld_pa_m = float(np.sum(pa * np.log(pa / m)))
    kld_pb_m = float(np.sum(pb * np.log(pb / m)))
    js = 0.5 * (kld_pa_m + kld_pb_m)
    return js


def value_metrics(a: np.ndarray, b: np.ndarray, nbins: int = 256) -> Dict[str, Any]:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    diff = a - b
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rms_signal = float(math.sqrt(np.mean(a * a))) if a.size > 0 else 0.0
    rel_rmse = float(rmse / (rms_signal + EPS))

    denom = (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    cosine = float(np.dot(a, b) / denom)

    pearson = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else 1.0

    var_signal = float(np.var(a))
    var_error = float(np.var(diff)) + EPS
    sqnr_db = 10.0 * math.log10(max(var_signal, 1e-12) / var_error)

    js = js_divergence(a, b, nbins=nbins)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
        "pearson": pearson,
        "sqnr_db": sqnr_db,
        "js": js,
    }


def per_channel_metrics(a: np.ndarray, b: np.ndarray, axis: int = 0, nbins: int = 256) -> Dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError("shapes must match for per-channel metrics")
    if axis < 0 or axis >= a.ndim:
        raise ValueError("invalid axis for per-channel metrics")

    nch = a.shape[axis]
    rels = np.zeros(nch, dtype=np.float64)
    jss = np.zeros(nch, dtype=np.float64)
    for i in range(nch):
        sa = np.take(a, i, axis=axis)
        sb = np.take(b, i, axis=axis)
        diff = sa.astype(np.float64).ravel() - sb.astype(np.float64).ravel()
        mse = np.mean(diff * diff)
        rmse = math.sqrt(mse)
        rms_signal = math.sqrt(np.mean(sa.astype(np.float64).ravel() ** 2)) if sa.size > 0 else 0.0
        rels[i] = rmse / (rms_signal + EPS)
        try:
            jss[i] = js_divergence(sa.astype(np.float64).ravel(), sb.astype(np.float64).ravel(), nbins=nbins)
        except Exception:
            jss[i] = 0.0

    idx_sort = np.argsort(-rels)
    entries = [(int(int(i)), float(rels[i]), float(jss[i])) for i in idx_sort]
    return {
        'per_channel_count': int(nch),
        'per_channel_rel_rmse_mean': float(np.mean(rels)),
        'per_channel_rel_rmse_max': float(np.max(rels)),
        'per_channel_js_mean': float(np.mean(jss)),
        'per_channel_js_max': float(np.max(jss)),
        'per_channel_sorted': entries,
    }


def load_single_tensor_from_gguf(path: str) -> Any:
    """
    Load the first tensor *entry* from a GGUF file and return the ReaderTensor
    (i.e. reader.tensors[0]) object.

    This follows the gguf.GGUFReader API: reader = GGUFReader(path);
    reader.tensors is a list of tensor entries.
    """
    if _has_gguf:
        try:
            if GGUFReader is not None:
                reader = GGUFReader(path)
            else:
                # fallback: try to get class from gguf module
                import gguf as _gguf_mod  # type: ignore
                reader_cls = getattr(_gguf_mod, "GGUFReader", None)
                if reader_cls is None:
                    raise RuntimeError("GGUFReader class not found in gguf module")
                reader = reader_cls(path)
        except Exception as e:
            raise RuntimeError(f"failed to open GGUF file with GGUFReader: {e}")

        tensors = getattr(reader, "tensors", None)
        if not tensors:
            raise RuntimeError(f"No tensors found in GGUF reader for file {path}")

        # Return the first tensor entry unconditionally.
        return tensors[0]

    # fallback: gguf_parser (metadata only)
    if _has_gguf_parser:
        parser = GGUFParser(path)
        parser.parse()
        tensors = parser.tensors
        names = [t.name for t in tensors]
        raise RuntimeError(
            "gguf-parser is available but does not provide dequantized arrays in this environment.\n"
            "Install a GGUF reader that returns numpy arrays (e.g. `pip install gguf`).\n"
            f"File {path} contains tensors: {names}"
        )

    raise RuntimeError(
        "No GGUF reader available. Please install one of:\n"
        "  pip install gguf    # preferred\n"
        "  pip install gguf-parser\n    "
    )


def discover_files_from_dir(d: str) -> List[str]:
    files = [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.lower().endswith('.gguf')]
    return files


def parse_composite(spec: str) -> Dict[str, float]:
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    out = {}
    for part in parts:
        if ':' not in part:
            raise ValueError(f"invalid composite part: {part}")
        k, v = part.split(':', 1)
        out[k.strip()] = float(v.strip())
    total = sum(out.values())
    if total == 0:
        raise ValueError("composite weights sum to zero")
    for k in out:
        out[k] = out[k] / total
    return out


def build_pairs(baseline_files: List[str], quant_files: List[str], pair_by: str = 'order') -> List[Tuple[str, str]]:
    if pair_by == 'order':
        if len(baseline_files) != len(quant_files):
            raise ValueError("when pairing by order, the two lists must have equal length")
        return list(zip(baseline_files, quant_files))
    elif pair_by == 'name':
        base_map = {os.path.splitext(os.path.basename(p))[0]: p for p in baseline_files}
        quant_map = {os.path.splitext(os.path.basename(p))[0]: p for p in quant_files}
        keys = sorted(set(base_map.keys()) & set(quant_map.keys()))
        if not keys:
            raise ValueError("no matching basenames found between dirs/lists")
        pairs = [(base_map[k], quant_map[k]) for k in keys]
        return pairs
    else:
        raise ValueError(f"unknown pair_by value: {pair_by}")


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    # expect comma-separated ints, e.g. "10,128,32000" or "128,32000"
    parts = [s.strip() for s in shape_str.split(',') if s.strip()]
    return tuple(int(p) for p in parts)


def compute_avg_kl_from_precomputed_tensors(a: np.ndarray, b: np.ndarray, tensors_are_probs: bool = False) -> float:
    """
    Given two arrays containing logits or probabilities (same shape), compute
    average KL(softmax(a) || softmax(b)) across all items. If tensors_are_probs
    is True, treat them as probability distributions already and compute KL(p||q).
    """
    if a.shape != b.shape:
        raise ValueError(f"logit/prob tensors must have same shape: {a.shape} vs {b.shape}")

    # If tensors are 1D or 2D or 3D, flatten batch and seq dims if present but keep vocab as last dim
    if a.ndim == 1:
        # treat as single-vocab logits (degenerate): KL is zero
        return 0.0
    # assuming last axis is vocab
    V = a.shape[-1]
    # reshape to (N, V)
    flat_a = a.reshape(-1, V)
    flat_b = b.reshape(-1, V)

    if tensors_are_probs:
        p = np.clip(flat_a, 1e-12, 1.0)
        q = np.clip(flat_b, 1e-12, 1.0)
    else:
        # convert logits -> probs with softmax per row
        a2 = flat_a - np.max(flat_a, axis=-1, keepdims=True)
        b2 = flat_b - np.max(flat_b, axis=-1, keepdims=True)
        p = np.exp(a2)
        q = np.exp(b2)
        p = p / (np.sum(p, axis=-1, keepdims=True) + 1e-12)
        q = q / (np.sum(q, axis=-1, keepdims=True) + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)

    kl = p * (np.log(p) - np.log(q))
    per_row = np.sum(kl, axis=-1)
    return float(np.mean(per_row))


def tensor_shape_tuple(tensor: Any) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(dim) for dim in tensor.shape)
    except Exception:
        try:
            return tuple(tensor.shape)
        except Exception:
            return None


def tensor_n_elements(tensor: Any, shape: Optional[Tuple[int, ...]]) -> int:
    if hasattr(tensor, "n_elements") and tensor.n_elements is not None:
        try:
            return int(tensor.n_elements)
        except Exception:
            pass
    # fallback: product of dims if available
    if shape:
        prod = 1
        for d in shape:
            prod *= int(d)
        return prod
    raise RuntimeError("Unable to determine tensor element count")


def tensor_n_bytes(tensor: Any) -> Optional[int]:
    if hasattr(tensor, "n_bytes") and tensor.n_bytes is not None:
        try:
            return int(tensor.n_bytes)
        except Exception:
            pass
    # fallback try to get data.nbytes
    try:
        return int(getattr(tensor, "data").nbytes)
    except Exception:
        return None


def tensor_type_name(tensor: Any) -> Optional[str]:
    tt = getattr(tensor, "tensor_type", None)
    if tt is not None:
        try:
            return tt.name
        except Exception:
            try:
                return str(tt)
            except Exception:
                return None
    return None


def dequantize_tensor_to_float_array(tensor: Any, n_elements: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try to produce a dequantized float32 NumPy array for *common* non-packed formats.

    Returns (array_or_None, error_msg_or_None).
    - Supported conversions:
        * BF16 stored as uint16 words -> float32
        * float16 -> float32
        * float32/float64 -> float32 (copy/astype)
    - Does NOT attempt to dequantize packed GGML quant formats (Q4/Q8/etc).
      Those remain as byte-packed layouts and require their own dequantizers.
    """
    tt_name = (tensor_type_name(tensor) or "").lower()
    raw = getattr(tensor, "data", None)
    if raw is None:
        return None, "tensor has no .data"

    arr = np.asarray(raw)

    # If the logical sizes already match, simply convert to float32 where appropriate.
    if arr.size == n_elements:
        # BF16 handling: many GGUF readers expose BF16 as uint16 buffer.
        if arr.dtype == np.uint16 and ('bf16' in tt_name or 'bfloat16' in tt_name):
            try:
                # convert uint16 bf16 -> float32 by shifting into high 16 bits of uint32
                arr_u16 = arr.astype(np.uint16).ravel()
                arr_u32 = arr_u16.astype(np.uint32) << 16
                float32_arr = arr_u32.view(np.float32).copy()
                return float32_arr.reshape(arr.shape), None
            except Exception as e:
                return None, f"bf16->float32 conversion failed: {e}"

        # float16 -> float32
        if arr.dtype == np.float16:
            try:
                return arr.astype(np.float32), None
            except Exception as e:
                return None, f"float16->float32 conversion failed: {e}"

        # float32/float64 -> float32
        if arr.dtype in (np.float32, np.float64):
            try:
                return arr.astype(np.float32), None
            except Exception as e:
                return None, f"float->float32 conversion failed: {e}"

        # uint8 / uint16 but tensor_type is float -> try to reinterpret? No.
        # For now, if it's already numeric but not float, try casting if it makes sense.
        if np.issubdtype(arr.dtype, np.integer) and ('f' in tt_name or 'float' in tt_name):
            try:
                return arr.astype(np.float32), None
            except Exception as e:
                return None, f"integer->float conversion failed: {e}"

        # If arr is already float-like but different shape or dtype, return cast
        if np.issubdtype(arr.dtype, np.floating):
            try:
                return arr.astype(np.float32), None
            except Exception as e:
                return None, f"float cast failed: {e}"

        # Otherwise, we can't trust this buffer for direct numeric comparison
        return None, f"unsupported raw dtype {arr.dtype} for non-packed tensor_type {tt_name}"

    # If arr.size != n_elements, it may be a packed quant format (cannot dequant here),
    # or the reader returned raw byte buffer. Try BF16 case where arr.dtype==np.uint8 but bytes==2*n_elements
    # Some readers may expose BF16 as a uint8 buffer instead of uint16 words.
    if arr.dtype == np.uint8 and arr.size == n_elements * 2 and ('bf16' in tt_name or 'bfloat16' in tt_name):
        try:
            u16 = arr.view(np.uint16) if arr.flags['C_CONTIGUOUS'] else arr.copy().view(np.uint16)
            u16 = u16.astype(np.uint16)
            u32 = u16.astype(np.uint32) << 16
            float32_arr = u32.view(np.float32).copy()
            return float32_arr.reshape((n_elements,)), None
        except Exception as e:
            return None, f"bf16-from-bytes conversion failed: {e}"

    # Nothing we can do automatically for packed quant formats (Q4/Q8 etc)
    return None, "data buffer size does not match element count; likely packed quant bytes (dequantizer required)"


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare multiple single-tensor GGUF baseline/quant pairs and rank by sensitivity")
    p.add_argument('--baseline-gguf', type=str, default=None, help='comma-separated list of baseline gguf files (alternative to --baseline-files/--baseline-dir)')
    p.add_argument('--quant-gguf', type=str, default=None, help='comma-separated list of quant gguf files (alternative to --quant-files/--quant-dir)')

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument('--baseline-files', nargs='+', help='baseline gguf files (list)')
    group.add_argument('--baseline-dir', help='directory with baseline gguf files')
    p2 = p.add_mutually_exclusive_group(required=False)
    p2.add_argument('--quant-files', nargs='+', help='quantized gguf files (list)')
    p2.add_argument('--quant-dir', help='directory with quantized gguf files')

    p.add_argument('--pair-by', choices=['order', 'name'], default='order', help='how to pair baseline/quant files')
    p.add_argument('--nbins', type=int, default=256, help='histogram bins for JS divergence')
    p.add_argument('--rank-by', default='js', help='metric name to rank by (default: js)')
    p.add_argument('--composite', default=None, help='composite score spec: e.g. "js:0.7,rel_rmse:0.3"')
    p.add_argument('--out', default='gguf_compare_results.csv', help='CSV output path')

    # per-channel
    p.add_argument('--per-channel-axis', type=int, choices=[0, 1], default=None, help='compute per-channel metrics along this axis (0 or 1)')
    p.add_argument('--topk-channels', type=int, default=5, help='how many worst channels to report')

    # tensors-as-logits mode
    p.add_argument('--tensors-are-logits', action='store_true', help='treat GGUF tensors as precomputed logits/probabilities and compute KL directly')
    p.add_argument('--tensors-are-probs', action='store_true', help='if set together with --tensors-are-logits, treat tensors as probabilities (not logits)')
    p.add_argument('--logits-shape', type=str, default=None, help='optional shape to reshape logits to before KL, format: N,L,V or L,V or V')

    # legacy model-based kl (kept but not mandatory)
    p.add_argument('--compute-logit-kl', action='store_true', help='compute avg KL between baseline and variant logits using a model and a calibration file (legacy)')
    p.add_argument('--model-path', type=str, default=None, help='path or name for transformers model (legacy)')
    p.add_argument('--tokenizer-path', type=str, default=None, help='tokenizer path (legacy)')
    p.add_argument('--calib-file', type=str, default=None, help='calibration prompts file (legacy)')
    p.add_argument('--mapping-file', type=str, default=None, help='mapping gguf tensor name -> model param name (legacy)')
    p.add_argument('--device', type=str, default=None, help='device for model inference (cpu or cuda)')
    p.add_argument('--batch-size', type=int, default=8, help='batch size for calibration inference')
    p.add_argument('--max-length', type=int, default=512, help='max tokens for tokenizer')
    p.add_argument('--model-bin', type=str, default=None, help='path to a llama.cpp .bin (legacy)')

    args = p.parse_args(argv)

    # gather baseline_files / quant_files
    if args.baseline_gguf:
        baseline_files = [x.strip() for x in args.baseline_gguf.split(',') if x.strip()]
    elif args.baseline_dir:
        baseline_files = discover_files_from_dir(args.baseline_dir)
    elif args.baseline_files:
        baseline_files = list(args.baseline_files)
    else:
        p.error("one of --baseline-gguf, --baseline-files or --baseline-dir is required")

    if args.quant_gguf:
        quant_files = [x.strip() for x in args.quant_gguf.split(',') if x.strip()]
    elif args.quant_dir:
        quant_files = discover_files_from_dir(args.quant_dir)
    elif args.quant_files:
        quant_files = list(args.quant_files)
    else:
        p.error("one of --quant-gguf, --quant-files or --quant-dir is required")

    pairs = build_pairs(baseline_files, quant_files, pair_by=args.pair_by)
    print(f"Found {len(pairs)} pairs to evaluate")

    # calibration prompts for legacy model KL (if used)
    prompts = []
    if args.compute_logit_kl:
        if not args.calib_file:
            raise ValueError("--compute-logit-kl requires --calib-file")
        with open(args.calib_file, 'r', encoding='utf-8') as f:
            prompts = [l.strip() for l in f if l.strip()]
        if len(prompts) == 0:
            raise ValueError("calibration file is empty")

    logits_shape = None
    if args.logits_shape:
        logits_shape = parse_shape(args.logits_shape)

    rows: List[Dict[str, Any]] = []

    for bi, qi in pairs:
        try:
            btensor = load_single_tensor_from_gguf(bi)
            qtensor = load_single_tensor_from_gguf(qi)
        except Exception as e:
            print(f"Error opening GGUF files for pair ({bi}, {qi}): {e}", file=sys.stderr)
            continue

        # compute stable metadata values
        bname = getattr(btensor, "name", None)
        qname = getattr(qtensor, "name", None)
        bshape = tensor_shape_tuple(btensor)
        qshape = tensor_shape_tuple(qtensor)
        try:
            belems = tensor_n_elements(btensor, bshape)
            qelems = tensor_n_elements(qtensor, qshape)
        except Exception as e:
            print(f"Unable to determine element counts for pair {bi} vs {qi}: {e}", file=sys.stderr)
            continue

        bbytes = tensor_n_bytes(btensor)
        qbytes = tensor_n_bytes(qtensor)
        btype = tensor_type_name(btensor)
        qtype = tensor_type_name(qtensor)

        # Compare names first
        if (bname is None) or (qname is None):
            print(f"Skipping pair because tensor name missing for {bi} or {qi}", file=sys.stderr)
            continue
        if bname != qname:
            print(f"Skipping pair because tensor names differ: baseline '{bname}' != quant '{qname}' for files {bi} vs {qi}", file=sys.stderr)
            continue

        # Compare logical element counts
        if belems != qelems:
            # Clear instruction when elements differ (common when comparing BF16->Q4_0)
            print(
                f"Skipping pair because logical element counts differ: {bi} has {belems} vs {qi} has {qelems}. Dequantization required to compare.\n"
                "This often happens when the baseline is BF16 (dequantized) and the quant file is a packed GGML quant (e.g. Q4_0).\n"
                "Convert the packed Q4_0 GGUF to a dequantized BF16 GGUF (or other float format) before comparing.\n"
                "Example (replace with your conversion tool/flags; `llama-quantize` is commonly used):\n"
                "  llama-quantize input_q4_0.gguf output_bf16.gguf --to bf16\n"
                "After converting the quant file to BF16 (so both files contain the same logical element counts and shapes), re-run this script to compare.\n",
                file=sys.stderr,
            )
            continue

        # Attempt to obtain numeric arrays that match the logical element count.
        barr, berr = dequantize_tensor_to_float_array(btensor, belems)
        qarr, qerr = dequantize_tensor_to_float_array(qtensor, qelems)

        if barr is None or qarr is None:
            # If either side could not be dequantized automatically, print helpful reason and skip.
            print(
                f"Skipping pair {bi} vs {qi}: cannot obtain dequantized numeric arrays.\n"
                f"Baseline: dtype/type={tensor_type_name(btensor)} n_bytes={bbytes} dequant_err={berr}\n"
                f"Quant:    dtype/type={tensor_type_name(qtensor)} n_bytes={qbytes} dequant_err={qerr}\n\n"
                "Likely cause: the quant GGUF uses a packed GGML quant format (e.g. Q4_0, Q4_1, Q8_0).\n"
                "This script does not automatically dequantize packed GGML quant formats.\n\n"
                "Action: convert the quant GGUF to a dequantized float format (BF16/FP16/FP32) before running this script.\n"
                "Common approach: use a conversion/quantize tool (for example `llama-quantize`) to convert Q4_0 -> BF16.\n"
                "Example (replace with the exact tool/flags you have):\n"
                "  llama-quantize input_q4_0.gguf output_bf16.gguf --to bf16\n\n"
                "After conversion, re-run this script with the BF16 output file as the quant file —\n"
                "the script will accept BF16 (it converts BF16 to float32 internally) and perform the comparisons.\n",
                file=sys.stderr,
            )
            continue

        # reshape to logical shapes if available
        try:
            if bshape:
                barr = barr.reshape(bshape)
            if qshape:
                qarr = qarr.reshape(qshape)
        except Exception:
            barr = barr.ravel()
            qarr = qarr.ravel()

        # Now compute metrics
        try:
            metrics = value_metrics(barr, qarr, nbins=args.nbins)
        except Exception as e:
            print(f"Value-space metrics failed for pair ({bi}, {qi}): {e}", file=sys.stderr)
            continue

        row = {
            'baseline_file': bi,
            'quant_file': qi,
            'tensor_name': bname,
        }
        row.update(metrics)

        # per-channel
        if args.per_channel_axis is not None:
            try:
                pcm = per_channel_metrics(barr, qarr, axis=args.per_channel_axis, nbins=args.nbins)
                row['per_channel_count'] = pcm['per_channel_count']
                row['per_channel_rel_rmse_mean'] = pcm['per_channel_rel_rmse_mean']
                row['per_channel_rel_rmse_max'] = pcm['per_channel_rel_rmse_max']
                row['per_channel_js_mean'] = pcm['per_channel_js_mean']
                row['per_channel_js_max'] = pcm['per_channel_js_max']
                topk = pcm['per_channel_sorted'][:args.topk_channels]
                row['per_channel_topk'] = json.dumps(topk)
            except Exception as e:
                print(f"Per-channel metrics failed for pair {bi}: {e}", file=sys.stderr)

        # KL from precomputed logits inside GGUF tensors (if requested)
        if args.tensors_are_logits:
            try:
                a = barr
                b = qarr
                if logits_shape is not None and a.shape != tuple(logits_shape):
                    a = a.reshape(logits_shape)
                    b = b.reshape(logits_shape)
                kl = compute_avg_kl_from_precomputed_tensors(a, b, tensors_are_probs=args.tensors_are_probs)
                row['logit_kl_from_tensors'] = kl
            except Exception as e:
                print(f"Failed to compute KL from precomputed tensors for pair {bi}: {e}", file=sys.stderr)
                row['logit_kl_from_tensors'] = None

        # legacy model-based KL (best effort)
        if args.compute_logit_kl:
            print("Model-based logit-KL requested but model-based KL path is legacy and may require extra setup; skipping in this run.", file=sys.stderr)
            row['logit_kl'] = None

        rows.append(row)
        print(f"Processed: {os.path.basename(bi)} -> {os.path.basename(qi)}  js={metrics['js']:.6g} rel_rmse={metrics['rel_rmse']:.6g} logit_kl_from_tensors={row.get('logit_kl_from_tensors')}")

    if not rows:
        print("No successful pairs processed.")
        return

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
    except Exception:
        import csv
        keys = list(rows[0].keys())
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote basic CSV to {args.out} (pandas not installed)")
        return

    # ranking / composite
    if args.composite:
        comp = parse_composite(args.composite)
        missing = [k for k in comp.keys() if k not in df.columns]
        if missing:
            raise ValueError(f"composite references unknown metrics: {missing}")
        df['_composite_score'] = 0.0
        for k, w in comp.items():
            df['_composite_score'] += df[k].astype(float) * float(w)
        rank_col = '_composite_score'
    else:
        if args.rank_by not in df.columns:
            raise ValueError(f"rank-by metric '{args.rank_by}' not present. Available: {list(df.columns)}")
        rank_col = args.rank_by

    df_sorted = df.sort_values(by=rank_col, ascending=False).reset_index(drop=True)
    df_sorted.index.name = 'rank'

    df_sorted.to_csv(args.out, index=True)
    print(f"Wrote ranked results to {args.out}")
    display_cols = ['baseline_file', 'quant_file', 'tensor_name', rank_col]
    if 'logit_kl_from_tensors' in df_sorted.columns:
        display_cols.append('logit_kl_from_tensors')
    print(df_sorted[display_cols].head(20).to_string())


if __name__ == '__main__':
    main()
