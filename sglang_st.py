#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_st.py reads and writes safetensors shards and      **#
#** their scale tensors for the SGLang path.                  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Sep-10-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.          ₛₕₐᵣ𝒹ₑ𝒹, ₙₒₜ ₛₜᵢᵣᵣₑ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_st.py - a minimal safetensors reader and writer, with no dependency on
the `safetensors` package.

WHY NOT THE PACKAGE.  This suite's whole claim on the SGLang lane is that its
byte model is exact: `sglang_native.algo_bytes()` reproduces a published
checkpoint's language-model byte count TO THE BYTE, and `sglang_write.py`
reproduces the weight bytes RadixArk published.  A claim like that is only auditable
if the file layout is ours and visible:

    <u64 header_len> <header json, space-padded to a multiple of 8> <tensor data>

with the tensor data written in header order and nothing else in the file.  The
package would do the same thing, correctly, and hide it.  It is also one fewer
dependency in a venv that deliberately has neither torch nor safetensors on the
assign/predict path.

`dtype` strings are the safetensors names.  F8_E4M3 and U8 both come back as
uint8 because numpy has no float8 - the CODEC knows which is which, the
container does not need to.
"""

from __future__ import annotations

import glob
import json
import os
import struct

import numpy as np

NP_OF = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32, 'F64': np.float64,
         'F8_E4M3': np.uint8, 'F8_E5M2': np.uint8, 'U8': np.uint8, 'I8': np.int8,
         'I16': np.int16, 'I32': np.int32, 'I64': np.int64, 'U16': np.uint16,
         'U32': np.uint32, 'U64': np.uint64, 'BOOL': np.bool_}
ITEMSIZE = {k: np.dtype(v).itemsize for k, v in NP_OF.items()}


def header(path):
    """(header dict, offset of the first tensor byte)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def index(paths):
    """{name: (path, entry, data_base)} over a list of shard files."""
    out = {}
    for p in paths:
        h, base = header(p)
        for k, v in h.items():
            if k != '__metadata__':
                out[k] = (p, v, base)
    return out


def index_dir(d, pattern='*.safetensors'):
    return index(sorted(glob.glob(os.path.join(d, pattern))))


def read(idx, name):
    p, e, base = idx[name]
    a, b = e['data_offsets']
    with open(p, 'rb') as f:
        f.seek(base + a)
        buf = f.read(b - a)
    return np.frombuffer(buf, NP_OF[e['dtype']]).reshape(e['shape'])


def raw(idx, name):
    """The tensor's bytes, untouched - what a byte-for-byte cross-check compares."""
    p, e, base = idx[name]
    a, b = e['data_offsets']
    with open(p, 'rb') as f:
        f.seek(base + a)
        return f.read(b - a)


def lm_bytes(idx, exclude=('mtp.', 'model.visual.')):
    """Language-model tensor bytes, the quantity every table in this lane quotes.

    Stated as an EXCLUSION - everything that is not the vision tower and not the
    MTP head - because the language model's own prefix is a property of the
    architecture (`model.language_model.` on qwen3_5, plain `model.` on
    glm4_moe) while what is not the language model is a short, closed list.  The
    caller passes the MTP prefixes it derived from the arch; the default is only
    a sane fallback for a checkpoint nobody has an arch table for.
    """
    return sum(e['data_offsets'][1] - e['data_offsets'][0]
               for n, (_p, e, _b) in idx.items() if not n.startswith(tuple(exclude)))


class ShardWriter:
    """Streams tensors into a shard without holding it in memory.

    The header is sized from the plan first (so every offset is known before a
    byte of data is written), then the data is appended in plan order.  The
    writer refuses a tensor that arrives out of order or at the wrong size,
    because both are silent corruptions otherwise.
    """

    def __init__(self, path, plan, metadata=None):
        """plan: [(name, dtype_str, shape, nbytes)] in write order."""
        self.path = path
        hdr, off = {}, 0
        for name, dt, shape, nb in plan:
            hdr[name] = {'dtype': dt, 'shape': list(shape),
                         'data_offsets': [off, off + nb]}
            off += nb
        if metadata:
            hdr['__metadata__'] = metadata
        blob = json.dumps(hdr, separators=(',', ':')).encode()
        blob += b' ' * ((-len(blob)) % 8)
        self.f = open(path, 'wb')
        self.f.write(struct.pack('<Q', len(blob)))
        self.f.write(blob)
        self.base = 8 + len(blob)
        self.total = self.base + off
        self.expect = [(n, nb) for n, _, _, nb in plan]
        self.i = 0

    def write(self, name, arr):
        n, nb = self.expect[self.i]
        if n != name:
            raise RuntimeError(f'{self.path}: expected {n}, got {name}')
        b = np.ascontiguousarray(arr).tobytes()
        if len(b) != nb:
            raise RuntimeError(f'{self.path}: {name} is {len(b)} B, planned {nb} B')
        self.f.write(b)
        self.i += 1
        return b

    def close(self):
        if self.i != len(self.expect):
            raise RuntimeError(f'{self.path}: wrote {self.i}/{len(self.expect)} tensors')
        self.f.close()
        got = os.path.getsize(self.path)
        if got != self.total:
            raise RuntimeError(f'{self.path}: {got} B on disk, planned {self.total}')
        return got


def selftest(verbose=True) -> int:
    """Round-trip the container on a temporary file."""
    import tempfile
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)
        if verbose:
            print(('  ok   ' if ok else '  FAIL ') + msg)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'model-00001-of-00001.safetensors')
        a = np.arange(12, dtype=np.uint16).reshape(3, 4)
        b = np.float32([1.5])
        plan = [('a.weight', 'BF16', (3, 4), a.nbytes),
                ('a.weight_scale_2', 'F32', (), b.nbytes)]
        w = ShardWriter(p, plan, {'format': 'pt'})
        w.write('a.weight', a)
        w.write('a.weight_scale_2', b)
        n = w.close()
        chk(n == os.path.getsize(p), f'writer: planned size == on-disk size ({n})')
        idx = index_dir(d)
        chk(sorted(idx) == ['a.weight', 'a.weight_scale_2'],
            'reader: both tensors indexed')
        chk((read(idx, 'a.weight') == a).all(), 'round-trip: BF16 payload identical')
        chk(raw(idx, 'a.weight') == a.tobytes(), 'round-trip: raw bytes identical')
        chk((header(p)[1] - 8) % 8 == 0,
            f'header: json blob padded to an 8-byte boundary ({header(p)[1] - 8} B)')
        try:
            w2 = ShardWriter(os.path.join(d, 'x.safetensors'), plan)
            w2.write('a.weight_scale_2', b)
            chk(False, 'writer: out-of-order write must raise')
        except RuntimeError:
            chk(True, 'writer: an out-of-order write raises instead of corrupting')
        try:
            w3 = ShardWriter(os.path.join(d, 'y.safetensors'), plan)
            w3.write('a.weight', a[:2])
            chk(False, 'writer: wrong-size write must raise')
        except RuntimeError:
            chk(True, 'writer: a wrong-size write raises instead of corrupting')
    if verbose:
        print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


if __name__ == '__main__':
    import sys
    sys.exit(selftest())
