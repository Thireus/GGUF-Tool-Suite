#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_gguf.py reads the owner's BF16 GGUF split and      **#
#** gives back the HF tensors it was converted from.          **#
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
#** Copyright © 2026 - Thireus.         ᵣₑ𝓌ᵢₙ𝒹 ₜₕₑ 𝒸ₒₙᵥₑᵣₛᵢₒₙ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_gguf.py - read the owner's BF16 GGUF split and hand the checkpoint
writer back the HF tensors it was made from.

CPU ONLY.  Reads no more than the tensors it is asked for, one shard at a time,
and never merges a split.

WHY A SECOND SOURCE AT ALL
--------------------------
`sglang_write.py` builds from a BF16 safetensors snapshot.  Anybody who has made
GGUF recipes with this suite already has the same weights in another form - the
SPECIAL_SPLIT set that `quant_downloader.sh` fetches, one tensor per GGUF shard,
first shard carrying the metadata - and it is usually the only copy on the
machine, because the snapshot was deleted after the conversion (docs/Convert
model to BF16.md says so in as many words).  Making that set a source costs one
inversion table and saves re-downloading 50 GB, or 700 GB on GLM.

WHAT HAS TO BE INVERTED, AND WHY IT IS DATA
-------------------------------------------
`convert_hf_to_gguf.py` is not a re-container: it renames, permutes, splits,
stacks, offsets and re-types.  Every one of those is invertible, and every one
of them is a property of the MODEL ARCHITECTURE rather than of the tensor, so
they live in the tables below next to the names they belong to - the same shape
`sglang_native.ARCHS` has, and cross-checked against it by the selftest so the
two cannot drift.

The transformations, per architecture, all read off llama.cpp's converter (the
`conversion/` modules behind `convert_hf_to_gguf.py`) and every one of them
MEASURED byte-for-byte against the BF16 snapshot before it was written down here:

  qwen3_5 (`conversion/qwen.py`, `_LinearAttentionVReorderBase`,
           `Qwen3NextModel`, `_QwenMtpMixin`; `conversion/qwen3vl.py`,
           `Qwen3VLVisionModel`):
    1. TRANSPOSED NAMING, NO TRANSPOSED DATA.  GGUF records `ne` fastest-first,
       safetensors records the shape slowest-first, so a linear that is
       [N, K] in HF is ne=[K, N] in GGUF and the BYTES ARE THE SAME.  Inverting
       a plain linear is therefore a rename and nothing else.
    2. ZERO-CENTRED NORMS.  Every `*norm.weight` EXCEPT `linear_attn.norm`
       is written as HF + 1 (`qwen.py`, Qwen3NextModel.modify_tensors).  The
       addition happens in float32 - `base.py:999` widens bf16 to f32 before
       modify_tensors - so subtracting 1 recovers the bf16 exactly, and this
       file proves that per tensor rather than assuming it.
    3. A_log -> -exp.  `ssm_a` holds -exp(A_log); the inverse is log(-x).
    4. THE V-HEAD REGROUPING.  Linear attention has num_v_heads > num_k_heads
       (48 and 16 here), and the converter retiles the V heads from K-major
       ([G0_v0..v2, G1_v0..v2, ...]) to V-major ([G0_v0, G1_v0, ..., G0_v1, ...])
       so ggml_repeat can broadcast.  It touches SIX tensors and in THREE
       different places, which is the trap: `in_proj_z`, `in_proj_a`,
       `in_proj_b`, `A_log` and `dt_bias` are regrouped over their ROWS, the
       TAIL of `in_proj_qkv` and of `conv1d` is regrouped over the rows after
       the q and k blocks, and `out_proj` is regrouped over its COLUMNS,
       because the V heads are its INPUT.  A permutation applied on the wrong
       axis is silent: the shape is unchanged and the checkpoint loads.
    5. conv1d IS SQUEEZED from [C, 1, K] to [C, K].
    6. THE MTP HEAD IS RENAMED INTO THE LAYER STACK.  `mtp.layers.0.*` becomes
       `blk.<n_layer>.*` and the four loose MTP tensors become `nextn.*`
       (`_QwenMtpMixin.filter_tensors`), which is why it needs its own table
       and cannot be an ARCHS layer entry.
    7. THE VISION TOWER IS A SEPARATE FILE with its own names, and its Conv3D
       patch embedding is SPLIT ALONG THE TEMPORAL DIMENSION into two Conv2Ds
       (`qwen3vl.py`, `v.patch_embd.weight` and `.weight.1`).

  glm4_moe (`conversion/glm.py`, `Glm4MoeModel`):
    1. The same transposed naming, and NOTHING else on the dense tensors: no
       norm offset, no permutation (the comment at glm.py:200 says why - GLM4V
       MoE is already in Neox order).
    2. STACKED EXPERTS.  A layer's 160 routed experts are `torch.stack`ed into
       one tensor per projection, so `blk.N.ffn_gate_exps.weight` is ne=
       [K, N, E] and expert e is the e-th contiguous [N, K] block of it.  The
       inverse is a SLICE, not a merge: this reader seeks to the expert it
       wants and reads N*K elements, so a 2.5 GB stacked tensor never enters
       memory to recover a 15 MB expert.

  BOTH:
    F32 WIDTH IS A CONVERTER CONVENTION, NOT THE MODEL'S.  `base.py:1017`
    forces every 1-D tensor and every `*_norm.weight` to F32 regardless of
    --outtype, so a GGUF norm is 4 B/element where the HF checkpoint it came
    from is 2.  Narrowing back is exact and is CHECKED to be exact (see
    `_narrow_bf16`); the exceptions - tensors the HF checkpoint really does
    keep in F32 - are declared once, in `sglang_native.ARCHS[...]['f32_tensors']`,
    which is the same declaration the size model already reads.

WHAT THE GGUF CANNOT GIVE BACK
------------------------------
Weights, all of them, exactly.  Not the checkpoint's small text files: a GGUF
carries the tokenizer as ggml arrays and the hyperparameters as ggml keys, which
is a DIFFERENT encoding of the same facts and not the bytes SGLang reads.  The
one exception is the chat template, which a GGUF stores verbatim.  `hf_files()`
below is the inventory, and it VALIDATES what it is given against the GGUF's own
metadata rather than trusting it.

NO DEPENDENCY ON gguf-py.  The same argument `sglang_st.py` makes for
safetensors: the whole claim of this lane is that the byte model is exact, and a
claim like that is only auditable if the container is ours and visible.  A GGUF
header is a length-prefixed key/value list and a tensor directory, which is 120
lines; `gguf-py` would do it correctly and hide it, and it is one more thing a
venv on the assign path would have to carry.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sglang_native as SN          # noqa: E402


# =============================================================================
# 0. THE CONTAINER
# =============================================================================
#
#   "GGUF" <u32 version> <u64 n_tensors> <u64 n_kv>
#   n_kv   x  <string key> <u32 type> <value>
#   n_tens x  <string name> <u32 n_dims> <u64 ne[n_dims]> <u32 ggml_type>
#             <u64 offset>
#   padding to general.alignment, then the tensor data at those offsets.
#
# Only the three types a BF16 split can hold are decoded.  A quantised split
# (q8_0, iq4_ks, ...) is REFUSED rather than dequantised: this source exists to
# recover the exact BF16 the checkpoint was made from, and a quantised shard
# cannot do that.

GGUF_MAGIC = b'GGUF'
GGML_ALIGN_DEFAULT = 32

# ggml_type -> (safetensors dtype name, numpy dtype)
GGML_TYPES = {0: ('F32', np.float32), 1: ('F16', np.float16), 30: ('BF16', np.uint16)}
GGML_TYPE_NAMES = {0: 'f32', 1: 'f16', 30: 'bf16'}
# what a tensors.map row's `dtype=` says, in the same terms
_GGML_OF_MAP = {v: k for k, v in GGML_TYPE_NAMES.items()}

_KV_SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f',
              7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
_KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_STRING, _KV_ARRAY = 8, 9


class _Cursor:
    """A byte cursor over a header buffer.  Raises on a truncated read."""

    def __init__(self, buf, path):
        self.b, self.o, self.path = buf, 0, path

    def take(self, n):
        if self.o + n > len(self.b):
            raise ValueError(f'{self.path}: header truncated at {self.o}')
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode('utf-8', errors='replace')

    def value(self, t):
        if t == _KV_STRING:
            return self.string()
        if t == _KV_ARRAY:
            et, n = self.u32(), self.u64()
            if et in _KV_SCALAR:                 # bulk-decode a numeric array
                sz = _KV_SIZE[et]
                raw = self.take(sz * n)
                return np.frombuffer(raw, np.dtype(_KV_SCALAR[et][1:])).tolist()
            return [self.value(et) for _ in range(n)]
        if t not in _KV_SCALAR:
            raise ValueError(f'{self.path}: unknown GGUF value type {t}')
        return struct.unpack(_KV_SCALAR[t], self.take(_KV_SIZE[t]))[0]


class GgufFile:
    """One GGUF shard: its metadata, its tensor directory, and reads into it.

    The header is parsed once and the file is then only ever seeked into, so a
    shard costs one open per tensor read and nothing is held.
    """

    # A data shard's header is a few hundred bytes; only the metadata shard's is
    # large (a 248k-token vocabulary).  Read a prefix and grow once rather than
    # pull a 100 MB shard through memory to learn the shape of the tensor in it.
    HEADER_PROBE = 1 << 17

    def __init__(self, path):
        self.path = path
        size = os.path.getsize(path)
        want = min(size, self.HEADER_PROBE)
        while True:
            with open(path, 'rb') as f:
                buf = f.read(want)
            if buf[:4] != GGUF_MAGIC:
                raise ValueError(f'{path}: not a GGUF file')
            try:
                self._parse(buf)
                return
            except ValueError:
                if want >= size:
                    raise
                want = size

    def _parse(self, buf):
        c = _Cursor(buf, self.path)
        c.take(4)
        self.version = c.u32()
        n_tensors, n_kv = c.u64(), c.u64()
        kv = {}
        for _ in range(n_kv):
            k = c.string()
            kv[k] = c.value(c.u32())
        tensors = {}
        for _ in range(n_tensors):
            name = c.string()
            ne = [c.u64() for _ in range(c.u32())]
            tensors[name] = (ne, c.u32(), c.u64())
        align = int(kv.get('general.alignment', GGML_ALIGN_DEFAULT))
        self.kv, self.tensors = kv, tensors
        self.data_start = (c.o + align - 1) // align * align

    # -- metadata ---------------------------------------------------------- #
    @property
    def arch(self):
        return self.kv.get('general.architecture')

    def akv(self, suffix, default=None):
        """A key of this file's own architecture: `akv('block_count')`."""
        return self.kv.get(f'{self.arch}.{suffix}', default)

    # -- tensor data ------------------------------------------------------- #
    def entry(self, name):
        ne, tt, off = self.tensors[name]
        if tt not in GGML_TYPES:
            raise SystemExit(
                f'{self.path}: {name} is ggml type {tt}, which this reader does '
                f'not decode.  A source split must be the BF16 one - a quantised '
                f'split cannot give back the weights the checkpoint is made from.')
        return ne, tt, off

    def read(self, name, first=None, count=None):
        """The tensor as numpy, in GGUF order (numpy shape = reversed ne).

        `first`/`count` read a contiguous run of ELEMENTS instead of the whole
        tensor, which is how a stacked-expert tensor gives up one expert without
        the other 159 ever being touched.
        """
        ne, tt, off = self.entry(name)
        dtype = GGML_TYPES[tt][1]
        item = np.dtype(dtype).itemsize
        total = int(np.prod(ne)) if ne else 1
        first = 0 if first is None else int(first)
        count = total if count is None else int(count)
        if first < 0 or first + count > total:
            raise ValueError(f'{self.path}: {name}[{first}:{first+count}] is '
                             f'outside its {total} elements')
        with open(self.path, 'rb') as f:
            f.seek(self.data_start + off + first * item)
            buf = f.read(count * item)
        if len(buf) != count * item:
            raise ValueError(f'{self.path}: {name} is short by '
                             f'{count * item - len(buf)} B')
        a = np.frombuffer(buf, dtype)
        return a.reshape(tuple(reversed(ne))) if count == total else a


# =============================================================================
# 1. WHAT THE CONVERSION DID, AS DATA
# =============================================================================
#
# An entry is  gguf suffix -> (hf leaf, ops)  where
#
#   `hf leaf` is the tensor name the HF checkpoint carries, relative to the part
#             prefix, with `{bid}` for the layer and `{eid}` for the expert.  It
#             is spelled in full rather than derived from `ARCHS[...]['tensors']`
#             because the two answer different questions - ARCHS names the MODULE
#             a recipe assigns, which on glm4_moe is the FUSED name no file ever
#             carries - and because a name written twice can be CHECKED, which
#             `selftest` does for every entry the two share.
#   `ops`     the transformations convert_hf_to_gguf.py applied, in the order it
#             applied them.  `invert()` undoes them in reverse.
#
# THE OPS.  Each names one thing the converter does and nothing else:
#
#   'plus_one'     GGUF = HF + 1, computed in f32 (a zero-centred RMSNorm gamma)
#   'neg_exp'      GGUF = -exp(HF)
#   'v_rows'       V heads regrouped K-major -> V-major over the FIRST axis
#   'v_rows_1'     ... the same, with one row per V head instead of head_v_dim
#   'v_rows_tail'  ... the same, but only over the rows after the q and k blocks
#   'v_cols'       ... the same, over the LAST axis (an out projection, whose
#                  INPUT is the V heads)
#   'conv_kernel'  HF [C, 1, K] squeezed to [C, K]
#   'stack'        one tensor per expert stacked into [E, N, K]
#   'patch_t0' /   an HF Conv3D [O, C, 2, H, W] split along the temporal axis
#   'patch_t1'     into two Conv2D tensors [O, C, H, W]
#
# Everything absent from a table is a name this reader REFUSES, because a tensor
# we cannot name is a tensor we would silently drop, and a checkpoint missing a
# tensor fails on the GPU rather than here.

INVERT_QWEN3_5 = {
    'arch': 'qwen3_5',
    # `general.architecture` of a split this table can read.
    'gguf_arch': ('qwen35',),
    'parts': {
        # ---- the main split: the language model ------------------------------
        'text': {
            'gguf_arch': ('qwen35',),
            'layer_prefix': 'model.language_model.layers.{bid}.',
            'globals': {
                'token_embd.weight':  ('model.language_model.embed_tokens.weight', ()),
                'output.weight':      ('lm_head.weight', ()),
                'output_norm.weight': ('model.language_model.norm.weight', ('plus_one',)),
            },
            # The `mtp-` companion repeats these so it is loadable on its own
            # (the `keep` list in `_QwenMtpMixin.filter_tensors`).  They are the
            # main split's tensors, byte for byte, so they are taken from the
            # main split and skipped there.
            'mtp_duplicates': ('token_embd.weight', 'output.weight',
                               'output_norm.weight'),
            'tensors': {
                'attn_norm.weight':           ('input_layernorm.weight', ('plus_one',)),
                'post_attention_norm.weight': ('post_attention_layernorm.weight', ('plus_one',)),
                'attn_q.weight':              ('self_attn.q_proj.weight', ()),
                'attn_k.weight':              ('self_attn.k_proj.weight', ()),
                'attn_v.weight':              ('self_attn.v_proj.weight', ()),
                'attn_output.weight':         ('self_attn.o_proj.weight', ()),
                'attn_q_norm.weight':         ('self_attn.q_norm.weight', ('plus_one',)),
                'attn_k_norm.weight':         ('self_attn.k_norm.weight', ('plus_one',)),
                'ffn_gate.weight':            ('mlp.gate_proj.weight', ()),
                'ffn_up.weight':              ('mlp.up_proj.weight', ()),
                'ffn_down.weight':            ('mlp.down_proj.weight', ()),
                # linear attention.  `linear_attn.norm` is the ONE norm the
                # converter does not offset, and it is excluded by name in
                # qwen.py rather than by any rule - so it is excluded by name here.
                'ssm_norm.weight':            ('linear_attn.norm.weight', ()),
                'attn_qkv.weight':            ('linear_attn.in_proj_qkv.weight', ('v_rows_tail',)),
                'attn_gate.weight':           ('linear_attn.in_proj_z.weight', ('v_rows',)),
                'ssm_alpha.weight':           ('linear_attn.in_proj_a.weight', ('v_rows_1',)),
                'ssm_beta.weight':            ('linear_attn.in_proj_b.weight', ('v_rows_1',)),
                'ssm_out.weight':             ('linear_attn.out_proj.weight', ('v_cols',)),
                'ssm_conv1d.weight':          ('linear_attn.conv1d.weight',
                                               ('v_rows_tail', 'conv_kernel')),
                'ssm_a':                      ('linear_attn.A_log', ('v_rows_1', 'neg_exp')),
                'ssm_dt.bias':                ('linear_attn.dt_bias', ('v_rows_1',)),
            },
        },
        # ---- the mtp- companion split: the NEXTN draft head ------------------
        # `_QwenMtpMixin.filter_tensors` moves `mtp.layers.<i>.*` to
        # `model.layers.<n_layer + i>.*` and renames the four loose tensors, so
        # the head arrives inside the layer stack under a layer id the main
        # split does not have.  `{mid}` below is that offset put back.
        'mtp': {
            'gguf_arch': ('qwen35',),
            'dir_prefix': 'mtp-',
            'layer_prefix': 'mtp.layers.{mid}.',
            'globals': {},
            'tensors': {
                'attn_norm.weight':           ('input_layernorm.weight', ('plus_one',)),
                'post_attention_norm.weight': ('post_attention_layernorm.weight', ('plus_one',)),
                'attn_q.weight':              ('self_attn.q_proj.weight', ()),
                'attn_k.weight':              ('self_attn.k_proj.weight', ()),
                'attn_v.weight':              ('self_attn.v_proj.weight', ()),
                'attn_output.weight':         ('self_attn.o_proj.weight', ()),
                'attn_q_norm.weight':         ('self_attn.q_norm.weight', ('plus_one',)),
                'attn_k_norm.weight':         ('self_attn.k_norm.weight', ('plus_one',)),
                'ffn_gate.weight':            ('mlp.gate_proj.weight', ()),
                'ffn_up.weight':              ('mlp.up_proj.weight', ()),
                'ffn_down.weight':            ('mlp.down_proj.weight', ()),
                # the four that leave the layer stack on the way back
                'nextn.eh_proj.weight':          ('!mtp.fc.weight', ()),
                'nextn.enorm.weight':            ('!mtp.pre_fc_norm_embedding.weight',
                                                  ('plus_one',)),
                'nextn.hnorm.weight':            ('!mtp.pre_fc_norm_hidden.weight',
                                                  ('plus_one',)),
                'nextn.shared_head_norm.weight': ('!mtp.norm.weight', ('plus_one',)),
            },
        },
        # ---- the mmproj- companion split: the vision tower -------------------
        'vision': {
            'gguf_arch': ('clip',),
            # what `convert_hf_to_gguf.py --mmproj` names the directory it goes
            # into, which is how the companion is found beside the main split
            'dir_prefix': 'mmproj-',
            'layer_prefix': 'model.visual.blocks.{bid}.',
            'blk_prefix': 'v.blk.',
            'globals': {
                'v.patch_embd.weight':    ('model.visual.patch_embed.proj.weight',
                                           ('patch_t0',)),
                'v.patch_embd.weight.1':  ('model.visual.patch_embed.proj.weight',
                                           ('patch_t1',)),
                'v.patch_embd.bias':      ('model.visual.patch_embed.proj.bias', ()),
                'v.position_embd.weight': ('model.visual.pos_embed.weight', ()),
                'v.post_ln.weight':       ('model.visual.merger.norm.weight', ()),
                'v.post_ln.bias':         ('model.visual.merger.norm.bias', ()),
                'mm.0.weight':            ('model.visual.merger.linear_fc1.weight', ()),
                'mm.0.bias':              ('model.visual.merger.linear_fc1.bias', ()),
                'mm.2.weight':            ('model.visual.merger.linear_fc2.weight', ()),
                'mm.2.bias':              ('model.visual.merger.linear_fc2.bias', ()),
            },
            'tensors': {
                'attn_qkv.weight':  ('attn.qkv.weight', ()),
                'attn_qkv.bias':    ('attn.qkv.bias', ()),
                'attn_out.weight':  ('attn.proj.weight', ()),
                'attn_out.bias':    ('attn.proj.bias', ()),
                'ffn_up.weight':    ('mlp.linear_fc1.weight', ()),
                'ffn_up.bias':      ('mlp.linear_fc1.bias', ()),
                'ffn_down.weight':  ('mlp.linear_fc2.weight', ()),
                'ffn_down.bias':    ('mlp.linear_fc2.bias', ()),
                'ln1.weight':       ('norm1.weight', ()),
                'ln1.bias':         ('norm1.bias', ()),
                'ln2.weight':       ('norm2.weight', ()),
                'ln2.bias':         ('norm2.bias', ()),
            },
        },
    },
}

INVERT_GLM4_MOE = {
    'arch': 'glm4_moe',
    'gguf_arch': ('glm4moe',),
    'parts': {
        # One part and no companions: GLM's MTP head is layer 92 of the same
        # stack (`Glm4MoeModel.filter_tensors` keeps it there) and the model has
        # no vision tower in the checkpoints this suite has calibration for.
        'text': {
            'gguf_arch': ('glm4moe',),
            'layer_prefix': 'model.layers.{bid}.',
            'globals': {
                'token_embd.weight':  ('model.embed_tokens.weight', ()),
                'output.weight':      ('lm_head.weight', ()),
                'output_norm.weight': ('model.norm.weight', ()),
            },
            'tensors': {
                'attn_norm.weight':           ('input_layernorm.weight', ()),
                'post_attention_norm.weight': ('post_attention_layernorm.weight', ()),
                'attn_q.weight':              ('self_attn.q_proj.weight', ()),
                'attn_k.weight':              ('self_attn.k_proj.weight', ()),
                'attn_v.weight':              ('self_attn.v_proj.weight', ()),
                'attn_q.bias':                ('self_attn.q_proj.bias', ()),
                'attn_k.bias':                ('self_attn.k_proj.bias', ()),
                'attn_v.bias':                ('self_attn.v_proj.bias', ()),
                'attn_output.weight':         ('self_attn.o_proj.weight', ()),
                'attn_q_norm.weight':         ('self_attn.q_norm.weight', ()),
                'attn_k_norm.weight':         ('self_attn.k_norm.weight', ()),
                'ffn_gate.weight':            ('mlp.gate_proj.weight', ()),
                'ffn_up.weight':              ('mlp.up_proj.weight', ()),
                'ffn_down.weight':            ('mlp.down_proj.weight', ()),
                'ffn_gate_inp.weight':        ('mlp.gate.weight', ()),
                'exp_probs_b.bias':           ('mlp.gate.e_score_correction_bias', ()),
                'ffn_gate_shexp.weight':      ('mlp.shared_experts.gate_proj.weight', ()),
                'ffn_up_shexp.weight':        ('mlp.shared_experts.up_proj.weight', ()),
                'ffn_down_shexp.weight':      ('mlp.shared_experts.down_proj.weight', ()),
                'ffn_gate_exps.weight':       ('mlp.experts.{eid}.gate_proj.weight', ('stack',)),
                'ffn_up_exps.weight':         ('mlp.experts.{eid}.up_proj.weight', ('stack',)),
                'ffn_down_exps.weight':       ('mlp.experts.{eid}.down_proj.weight', ('stack',)),
                # the MTP head, which stays in the layer stack on this arch
                'nextn.embed_tokens.weight':     ('embed_tokens.weight', ()),
                'nextn.eh_proj.weight':          ('eh_proj.weight', ()),
                'nextn.enorm.weight':            ('enorm.weight', ()),
                'nextn.hnorm.weight':            ('hnorm.weight', ()),
                'nextn.shared_head_norm.weight': ('shared_head.norm.weight', ()),
                'nextn.shared_head_head.weight': ('shared_head.head.weight', ()),
            },
        },
    },
}

INVERTERS = {'qwen3_5': INVERT_QWEN3_5, 'glm4_moe': INVERT_GLM4_MOE}

# `general.architecture` -> the suite arch key whose table reads that split.
GGUF_ARCH_KEYS = {g: t['arch'] for t in INVERTERS.values() for g in t['gguf_arch']}

_BLK_RE = re.compile(r'^blk\.(\d+)\.(.+)$')
_VBLK_RE = re.compile(r'^v\.blk\.(\d+)\.(.+)$')


# =============================================================================
# 2. THE ARITHMETIC, AND ITS PROOF
# =============================================================================
#
# Three of the inversions are not permutations, so none of them is taken on
# trust.  Each is checked ON THE TENSOR, at the moment it is inverted:
#
#   the F32 -> BF16 narrowing   must widen back to the SAME f32 bits.  That is
#       the proof that the GGUF's F32 really is a widened BF16 and that nothing
#       is being rounded away; a source that fails it is not the BF16 split.
#   the `plus_one` subtraction  must add back to the same f32 bits.
#   the `neg_exp` logarithm     must land WELL INSIDE its bf16 cell (a quarter
#       of a ULP), which makes the recovered value the unique nearest one no
#       matter whose exp() and log() were used - numpy's here, torch's in the
#       converter.  A 1-ULP difference between the two libraries is 2^-24
#       relative; a quarter of a bf16 ULP is 2^-10.  The margin is 16,384x.

def _widen_bf16(u16):
    """bf16 bits -> f32 (exact; bf16 is the top half of f32)."""
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _narrow_bf16(a32, what):
    """f32 -> bf16 bits, round-half-to-even, and PROVE it lost nothing.

    Every f32 this reader narrows came from a bf16 that the converter widened
    (`base.py:999`), so the narrowing must be exact.  If it is not, the source
    is not what it claims to be and saying so here beats writing a checkpoint
    that is quietly not the model.
    """
    a32 = np.ascontiguousarray(a32, np.float32)
    u = a32.view(np.uint32)
    out = ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) >> 16).astype(np.uint16)
    bad = _widen_bf16(out).view(np.uint32) != u
    if bad.any():
        i = int(np.argmax(bad))
        raise SystemExit(
            f'{what}: element {i} is {float(a32.reshape(-1)[i])!r}, which is not '
            f'a BF16 value.  This reader recovers the BF16 the checkpoint was '
            f'converted from; a source carrying anything else is not that split.')
    return out


def _round_bf16(a32):
    """f32 -> bf16 bits, round-half-to-even, no questions asked."""
    u = np.ascontiguousarray(a32, np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) >> 16).astype(np.uint16)


def _undo_plus_one(x32, what, stats):
    """Recover the BF16 gamma from a GGUF norm that holds f32(gamma + 1).

    THE ONE PLACE THE CONVERSION LOSES SOMETHING, and it is worth being exact
    about what.  `gamma + 1` is computed and stored in float32, and whether that
    sum keeps every bit of the bf16 is a question about TWO spacings, not about
    one threshold.  A bf16 carries 8 mantissa bits, so its lowest bit sits at
    |gamma| * 2^-8; float32's spacing at the sum is 2^-23 above 1.0 and 2^-24
    below it.  A POSITIVE gamma is therefore exact from 2^-16 up (its lowest
    bit clears 2^-23 from there) and a NEGATIVE one from 2^-17 up (1 - |gamma|
    lands in the binade below, where float32 is one bit finer) - the sum lands
    in a different binade each way.  Below that the bottom bits are rounded
    off: several adjacent bf16 values give the same f32 sum and no reader can
    tell which one was there.  Both thresholds are exhaustive, not estimated:
    over all 65,536 bf16 values the largest gamma that does NOT come back
    exactly is one bf16 step under each of them.

    Nothing is lost for INFERENCE - the GGUF holds a correctly rounded (1 +
    gamma) and llama.cpp uses exactly that - but a checkpoint rebuilt from it
    can differ from the original in those elements, by at most half the float32
    spacing just above 1.0, 2^-24 = 5.96e-08 absolute - a bound the same
    exhaustive sweep attains.
    So this returns the NEAREST bf16 that reproduces the stored sum, and COUNTS
    the elements where that choice was not unique, which is what `--build`
    reports and BUILD.json records.  MEASURED on Qwen3.8-27B, whose 168
    zero-centred norms hold 694,784 gammas: 46 of them are ambiguous (an upper
    bound - 36 actually come back different from the snapshot), none above
    2^-16 = 1.526e-05 in magnitude, and the spread over the candidates is
    5.96e-08, exactly as the arithmetic says it must be.
    """
    one = np.float32(1.0)
    y = x32 - one                       # exact: Sterbenz, for x in [0.5, 2]
    b = _round_bf16(y)
    xu = x32.view(np.uint32)

    def hits(bits):
        return (_widen_bf16(bits) + one).view(np.uint32) == xu

    ok = hits(b)
    if not ok.all():
        # the nearest bf16 does not reproduce the sum; one of its neighbours must
        for step in (np.uint16(1), np.uint16(0xFFFF)):
            alt = (b + step).astype(np.uint16)
            take = (~ok) & hits(alt)
            b = np.where(take, alt, b).astype(np.uint16)
            ok = ok | take
        if not ok.all():
            i = int(np.argmax(~ok))
            raise SystemExit(
                f'{what}: element {i} ({float(x32.reshape(-1)[i])!r}) is not '
                f'1 + <a BF16>; this tensor did not come from a BF16 checkpoint '
                f'through convert_hf_to_gguf.py.')
    # AMBIGUITY, MEASURED AS A SPREAD.  An element is ambiguous when a
    # NEIGHBOURING bf16 also reproduces the stored sum, and what that costs is
    # not an error we can compute (the original is gone) but an interval we can:
    # the distance to the furthest candidate that also fits.  That is the honest
    # bound on how far the rebuilt checkpoint can be from the original one.
    base = _widen_bf16(b).astype(np.float64)
    spread = np.zeros(base.shape, np.float64)
    amb = np.zeros(base.shape, bool)
    for step in (np.uint16(1), np.uint16(0xFFFF)):
        alt = (b + step).astype(np.uint16)
        h = hits(alt)
        amb |= h
        spread = np.where(h, np.maximum(
            spread, np.abs(_widen_bf16(alt).astype(np.float64) - base)), spread)
    if amb.any() and stats is not None:
        stats.setdefault('inexact', {})[what] = (
            int(amb.sum()), int(amb.size), float(spread.max()),
            float(np.abs(base[amb]).max()))
    return b


def _snap_bf16(a32, what, margin=0.25):
    """f32 -> bf16 bits when the value is only APPROXIMATELY a bf16.

    Used by `neg_exp` alone.  Requires the value to sit within `margin` of a
    ULP of the bf16 it rounds to, so the choice cannot depend on which library
    computed the exponential.
    """
    a32 = np.ascontiguousarray(a32, np.float32)
    u = a32.view(np.uint32)
    out = ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) >> 16).astype(np.uint16)
    back = _widen_bf16(out)
    ulp = np.abs(_widen_bf16((out.astype(np.uint32) + 1).astype(np.uint16)) - back)
    ulp = np.where(ulp > 0, ulp, np.float32(np.finfo(np.float32).tiny))
    off = np.abs(back.astype(np.float64) - a32.astype(np.float64)) / ulp
    if (off > margin).any():
        i = int(np.argmax(off))
        raise SystemExit(
            f'{what}: element {i} lands {float(off.reshape(-1)[i]):.3f} ULP from '
            f'the nearest BF16, past the {margin} ULP the inversion is allowed. '
            f'The recovered value would depend on whose exp() ran.')
    return out


def _v_perm(nk, r, hd):
    """The index the converter GATHERED with, and the index that undoes it.

    `_reorder_v_heads` views the axis as [nk, r, hd] and swaps the first two,
    which is a pure permutation, so the inverse is the argsort of it - or,
    equivalently, the same view with nk and r exchanged.  Both are computed and
    the selftest checks they agree, because getting this backwards is exactly
    the failure the shape does not catch.
    """
    fwd = np.arange(nk * r * hd).reshape(nk, r, hd).transpose(1, 0, 2).reshape(-1)
    inv = np.empty_like(fwd)
    inv[fwd] = np.arange(fwd.size)
    return fwd, inv


class Geometry:
    """The linear-attention head counts, read off the GGUF's own metadata.

    Nothing here is a constant: `ssm.group_count` is num_k_heads,
    `ssm.time_step_rank` is num_v_heads and `ssm.inner_size / num_v_heads` is
    the value head dim, which is how `Qwen3NextModel.set_gguf_parameters` wrote
    them.  When the two head counts are equal the converter's regrouping is a
    no-op (`if num_k_heads != num_v_heads`, qwen.py) and so is every op here.
    """

    def __init__(self, gg):
        self.nk = gg.akv('ssm.group_count')
        self.nv = gg.akv('ssm.time_step_rank')
        inner = gg.akv('ssm.inner_size')
        self.hk = gg.akv('ssm.state_size')
        self.hv = (inner // self.nv) if (inner and self.nv) else None
        self.r = (self.nv // self.nk) if (self.nk and self.nv) else 1
        self._cache = {}

    @property
    def active(self):
        return bool(self.nk and self.nv and self.nk != self.nv)

    def need(self, what):
        if not (self.nk and self.nv and self.hv and self.hk):
            raise SystemExit(
                f'{what}: this tensor is regrouped by V head, and the GGUF '
                f'metadata does not carry the head counts (ssm.group_count, '
                f'ssm.time_step_rank, ssm.inner_size, ssm.state_size) needed to '
                f'undo it.')

    def inv(self, head_dim):
        if head_dim not in self._cache:
            self._cache[head_dim] = _v_perm(self.nk, self.r, head_dim)[1]
        return self._cache[head_dim]

    @property
    def qk_rows(self):
        """The q and k rows of in_proj_qkv / of the conv1d kernel, which the
        regrouping does NOT touch: `head_k_dim * num_k_heads * 2` in both."""
        return self.hk * self.nk * 2


def invert(a, ops, geo, what, hf_dtype, stats=None):
    """Undo `ops` (applied left to right by the converter) and return HF bytes.

    Returns a numpy array in the HF layout and HF width; `hf_dtype` is 'BF16'
    (uint16 carrying bf16 bits) or 'F32'.
    """
    for op in reversed(ops):
        if op == 'conv_kernel':
            a = a.reshape(a.shape[0], 1, *a.shape[1:])
        elif op == 'v_rows':
            if geo.active:
                geo.need(what)
                a = a[geo.inv(geo.hv)]
        elif op == 'v_rows_1':
            if geo.active:
                geo.need(what)
                a = a[geo.inv(1)]
        elif op == 'v_rows_tail':
            if geo.active:
                geo.need(what)
                q = geo.qk_rows
                a = np.concatenate([a[:q], a[q:][geo.inv(geo.hv)]], axis=0)
        elif op == 'v_cols':
            if geo.active:
                geo.need(what)
                a = a[..., geo.inv(geo.hv)]
        elif op in ('plus_one', 'neg_exp'):
            pass                     # numeric, applied below on the f32 view
        elif op in ('stack', 'patch_t0', 'patch_t1'):
            pass                     # structural, resolved by the source view
        else:
            raise SystemExit(f'{what}: unknown inversion op {op!r}')

    numeric = [op for op in ops if op in ('plus_one', 'neg_exp')]
    if numeric:
        x = np.ascontiguousarray(a, np.float32)
        for op in reversed(numeric):
            if op == 'plus_one':
                bits = _undo_plus_one(x, what, stats)
                if hf_dtype == 'F32':
                    x = _widen_bf16(bits).reshape(x.shape)
                    continue
                return bits.reshape(x.shape)
            if (x >= 0).any():                               # neg_exp
                raise SystemExit(f'{what}: -exp(A_log) must be negative '
                                 f'everywhere; this tensor is not.')
            x = np.log(-x.astype(np.float64)).astype(np.float32)
        if hf_dtype == 'F32':
            return x
        return _snap_bf16(x, what)

    if hf_dtype == 'F32':
        return np.ascontiguousarray(a, np.float32)
    if a.dtype == np.uint16:                     # already bf16 bits, untouched
        return np.ascontiguousarray(a)
    return _narrow_bf16(a, what).reshape(a.shape)


# =============================================================================
# 3. THE SPLIT
# =============================================================================
#
# THE TENSOR LIST COMES FROM tensors.map WHEN THERE IS ONE.  `quant_downloader.sh`
# fetches `tensors.<qtype>.map` beside the shards and leaves a generic
# `tensors.map` symlink pointing at it (quant_downloader.sh:3356-3374), so a
# downloaded split describes itself and the writer needs no `--bf16-map` at all.
# The precedence between the two is stated where it is applied,
# `sglang_write.recipe_universe()`.

_MAP_NAMES = ('tensors.map', 'tensors.bf16.map')


class Split:
    """A SPECIAL_SPLIT directory, or the first shard of one."""

    def __init__(self, path):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            self.dir = os.path.dirname(path)
        elif os.path.isdir(path):
            self.dir = path
        else:
            raise SystemExit(f'{path}: no such file or directory')
        self.shards = sorted(glob.glob(os.path.join(self.dir, '*.gguf')))
        if not self.shards:
            raise SystemExit(f'{self.dir}: no *.gguf shards here')
        self.first = GgufFile(self.shards[0])
        self.arch = self.first.arch
        self.map_path = next((p for p in (os.path.join(self.dir, n)
                                          for n in _MAP_NAMES) if os.path.exists(p)),
                             None)
        self._entries = None
        self._open = {}

    # -- the tensor directory ---------------------------------------------- #
    def entries(self):
        """{gguf name: (shard path, ne, ggml type)}, in the split's own order.

        FROM tensors.map WHEN THERE IS ONE, and not only for the names: a map
        row carries `shape=` and `dtype=` too, which is every fact this reader
        needs to PLAN a build.  That is 1 file read instead of 851 header parses,
        so `--dry-run` and the shard plan cost nothing, and a shard is opened
        only when its tensor is actually read.

        Either way the result is checked against the tensor count the metadata
        shard declares, so a half-downloaded split is a refusal rather than a
        checkpoint with a hole in it.
        """
        if self._entries is not None:
            return self._entries
        out = {}
        if self.map_path:
            for name, shape, _elems, dtype, shard, _sha in SN.read_bf16_map(
                    self.map_path):
                tt = _GGML_OF_MAP.get((dtype or '').lower())
                if tt is None:
                    raise SystemExit(
                        f'{self.map_path}: {name} is dtype={dtype!r}.  A source '
                        f'split must be the BF16 one - a quantised split cannot '
                        f'give back the weights the checkpoint is made from.')
                p = os.path.join(self.dir, shard)
                if not os.path.exists(p):
                    raise SystemExit(
                        f'{self.map_path} lists {shard} for {name}, which is not '
                        f'in {self.dir}.  Finish the download '
                        f'(quant_downloader.sh) before building from this split.')
                out[name] = (p, list(shape), tt)
        else:
            for p in self.shards:
                gg = GgufFile(p)
                for name, (ne, tt, _off) in gg.tensors.items():
                    out[name] = (p, list(ne), tt)
        want = self.first.kv.get('split.tensors.count')
        if want is not None and int(want) != len(out):
            raise SystemExit(
                f'{self.dir}: the metadata shard declares {int(want)} tensors and '
                f'{"tensors.map" if self.map_path else "the shard headers"} '
                f'account for {len(out)}.  This split is incomplete.')
        self._entries = out
        return out

    def universe(self):
        """The GGUF tensor names, in map order - what expands a merged recipe."""
        return list(self.entries())

    def gguf(self, path):
        if path not in self._open:
            self._open[path] = GgufFile(path)
        return self._open[path]

    def read(self, name, first=None, count=None):
        path, ne, tt = self.entries()[name]
        gg = self.gguf(path)
        # tensors.map is a SEPARATE artefact from the shards it describes (it is
        # produced by monitor_and_clean.sh afterwards and signed on its own), so
        # the one time a shard is opened is the one time to check the two agree.
        h_ne, h_tt, _off = gg.entry(name)
        if list(h_ne) != list(ne) or h_tt != tt:
            raise SystemExit(
                f'{os.path.basename(path)}: {name} is {GGML_TYPE_NAMES.get(h_tt, h_tt)} '
                f'{list(h_ne)} in the shard and '
                f'{GGML_TYPE_NAMES.get(tt, tt)} {list(ne)} in '
                f'{os.path.basename(self.map_path or "the headers")}. Rebuild the '
                f'map (monitor_and_clean.sh) before building from this split.')
        return gg.read(name, first, count)

    def label(self):
        return os.path.basename(self.dir.rstrip('/'))


def dir_prefixes(arch_key=None):
    """{part: the directory prefix its companion split is named with}."""
    tables = ([INVERTERS[arch_key]] if arch_key in INVERTERS
              else list(INVERTERS.values()))
    return {part: spec['dir_prefix'] for t in tables
            for part, spec in t['parts'].items() if spec.get('dir_prefix')}


def part_name_prefixes(arch_key=None):
    """{part: the HF-name prefix that part owns}, for the split-out parts.

    Derived from the table, not spelled out here: the longest prefix common to
    the part's layer names and to every name it owns outside the layer stack.
    On qwen3_5 that comes out as `mtp.` for the draft head - its four escaped
    names leave the layer stack but keep the prefix - and `model.visual.` for
    the tower.  It is what lets a tensor that never arrived be attributed to the
    companion that should have carried it, so a refusal can name the directory
    to fetch instead of just counting the hole.
    """
    tables = ([INVERTERS[arch_key]] if arch_key in INVERTERS
              else list(INVERTERS.values()))
    out = {}
    for t in tables:
        for part, spec in t['parts'].items():
            if not spec.get('dir_prefix'):
                continue
            names = [spec['layer_prefix'].split('{')[0]]
            names += [hf for hf, _ops in spec['globals'].values()]
            names += [leaf[1:] for leaf, _ops in spec['tensors'].values()
                      if leaf.startswith('!')]
            pre = os.path.commonprefix(names)
            pre = pre[: pre.rfind('.') + 1]
            if pre:
                out[part] = pre
    return out


def find_companions(main: Split, arch_key=None):
    """The `mmproj-` and `mtp-` splits that sit beside a main one.

    The suite's own naming (docs/Convert model to BF16.md): a companion is the
    same directory name with a part prefix, in the same parent, and the prefix
    is declared by the part rather than spelled here.  Found rather than
    configured, and what is found is always PRINTED, because a companion that is
    silently absent is a checkpoint missing its vision tower.
    """
    parent = os.path.dirname(main.dir.rstrip('/'))
    base = main.label()
    out = []
    for pre in sorted(set(dir_prefixes(arch_key).values())):
        cand = os.path.join(parent, pre + base)
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, '*.gguf')):
            out.append(cand)
    return out


# =============================================================================
# 4. THE SOURCE VIEW - a split, presented as an HF safetensors index
# =============================================================================
#
# `sglang_write.py` walks a source as `{hf name: (path, {dtype, shape}, base)}`
# and reads it with `read()` / `raw()`.  This class is that, over a split, so
# the writer's plan, its fused-group checks, its shard packing and its
# `--verify` are the SAME CODE for both sources and cannot drift apart.
#
# THE ORDER MATTERS, and it is not cosmetic.  The writer packs its output shards
# by walking the source in order, so two sources in different orders give the
# same tensors in differently-cut files.  A snapshot's order is
# `sorted(shard file), then sorted(name) within it` - which is exactly what
# `model.safetensors.index.json` records, so when that file is among the
# `--hf-files` the split reproduces the snapshot's order and the two builds are
# identical down to the shard sha256.  Without it the order is `sorted(name)`,
# which is deterministic, and the writer says so.

class GgufSource:
    """A BF16 GGUF split (plus its companions) as an HF tensor index."""

    def __init__(self, main: Split, companions=(), hf_index=None, arch_key=None,
                 log=None):
        self.log = log or (lambda *a: None)
        self.arch_key = arch_key or GGUF_ARCH_KEYS.get(main.arch)
        if self.arch_key not in INVERTERS:
            raise SystemExit(
                f'{main.label()}: general.architecture is {main.arch!r}, which no '
                f'inversion table reads.  The registered ones are '
                f'{", ".join(sorted(GGUF_ARCH_KEYS))}.')
        self.table = INVERTERS[self.arch_key]
        self.arch = SN.ARCHS[self.arch_key]
        self.f32_suffixes = set(self.arch.get('f32_tensors', ()))
        self._entries = {}          # hf name -> a plan for producing it
        self._splits = []
        self.parts = {}
        # what could not be recovered EXACTLY, filled in as tensors are read.
        # Only `_undo_plus_one` ever puts anything here; see its docstring.
        self.stats = {}

        self._add(main)
        for c in companions:
            self._add(c if isinstance(c, Split) else Split(c))

        order = list(self._entries)
        if hf_index:
            rank = _hf_index_order(hf_index)
            missing = [n for n in order if n not in rank]
            if missing:
                raise SystemExit(
                    f'{os.path.basename(hf_index)} does not list '
                    f'{len(missing)} of the tensors this split carries, e.g. '
                    f'{missing[:3]}.  It is not the index of this model.')
            order.sort(key=lambda n: rank[n])
            self.order_from = os.path.basename(hf_index)
        else:
            order.sort()
            self.order_from = ('sorted tensor names - no '
                               'model.safetensors.index.json in --hf-files, so '
                               'the output shard boundaries are this reader\'s '
                               'own rather than the snapshot\'s')
        self._entries = {n: self._entries[n] for n in order}

    # -- construction ------------------------------------------------------ #
    #
    # A SPLIT IS NOT A PART.  What part of the model a tensor belongs to is a
    # property of the TENSOR, not of the file it arrived in, and the two come
    # apart in both directions: the vision tower is always its own file, while
    # the MTP head is a separate `mtp-` split when the conversion ran with
    # `--no-mtp` (which is what docs/Convert model to BF16.md tells you to do)
    # and a handful of extra layers inside the main one when it did not.  So the
    # part is decided per tensor, by the same rule in both cases: a layer id at
    # or past `block_count - nextn_predict_layers` is the draft head.

    def _spec_of(self, split: Split):
        """(the language-model spec, the mtp spec or None, the mtp layer base)."""
        parts = self.table['parts']
        if split.arch in parts.get('vision', {}).get('gguf_arch', ()):
            return parts['vision'], None, None
        text = parts['text']
        if split.arch not in text['gguf_arch']:
            raise SystemExit(
                f'{split.label()}: general.architecture {split.arch!r} is not one '
                f'the {self.arch_key} inversion table reads '
                f'({", ".join(sorted(set(text["gguf_arch"]) | set(parts.get("vision", {}).get("gguf_arch", ()))))})')
        mtp = parts.get('mtp')
        if mtp is None:
            # An arch that keeps its draft head in the layer stack - glm4_moe -
            # names those tensors in its own table and needs no second one.
            return text, None, None
        n_next = int(split.first.akv('nextn_predict_layers', 0) or 0)
        if not n_next:
            return text, None, None
        base = int(split.first.akv('block_count', 0) or 0) - n_next
        return text, mtp, base

    def _add(self, split: Split):
        text, mtp, mtp_base = self._spec_of(split)
        vision = text is self.table['parts'].get('vision')
        geo = Geometry(split.first)
        blk_re = _VBLK_RE if text.get('blk_prefix') else _BLK_RE
        names = split.universe()
        # A split whose every layer is a draft-head layer is the `mtp-` companion,
        # and it repeats the embedding, the head and the final norm so it is
        # loadable on its own (the `keep` list in `_QwenMtpMixin.filter_tensors`).
        # Those are the main split's tensors byte for byte; taking them from here
        # too would present one HF name twice.
        lids = [int(m.group(1)) for m in (blk_re.match(n) for n in names) if m]
        mtp_only = bool(mtp) and bool(lids) and min(lids) >= mtp_base
        seen_parts, n_named = set(), 0
        for gname in names:
            m = blk_re.match(gname)
            if m:
                bid, suffix = int(m.group(1)), m.group(2)
                use_mtp = bool(mtp) and bid >= mtp_base
                spec = mtp if use_mtp else text
                part = 'vision' if vision else ('mtp' if use_mtp else 'text')
                ent = spec['tensors'].get(suffix)
                if ent is None:
                    raise SystemExit(self._unknown(split, gname, part))
                leaf, ops = ent
                mid = bid - mtp_base if use_mtp else bid
                if leaf.startswith('!'):            # leaves the layer stack
                    hf = leaf[1:].format(mid=mid)
                else:
                    hf = spec['layer_prefix'].format(bid=bid, mid=mid) + leaf
                gsuffix = suffix
            else:
                part = 'vision' if vision else 'text'
                if mtp_only and gname in text.get('mtp_duplicates', ()):
                    continue
                ent = text['globals'].get(gname)
                if ent is None:
                    raise SystemExit(self._unknown(split, gname, part))
                hf, ops = ent
                gsuffix = gname
            self._register(split, geo, gname, gsuffix, hf, ops)
            seen_parts.add(part)
            n_named += 1
        for part in sorted(seen_parts):
            other = self.parts.get(part)
            if other is not None and other is not split:
                raise SystemExit(f'{split.label()} and {other.label()} both carry '
                                 f'the {part!r} part of this model')
            self.parts[part] = split
        self._splits.append(split)
        self.log(f'  {split.label()}: {n_named} tensor(s), '
                 f'part(s) {", ".join(sorted(seen_parts))}, arch {split.arch!r}, '
                 + (f'tensor list from {os.path.basename(split.map_path)}'
                    if split.map_path else 'tensor list from the shard headers'))

    def _unknown(self, split, gname, part):
        return (f'{split.label()}: {gname!r} is not in the {self.arch_key}/{part} '
                f'inversion table, so this reader cannot say which HF tensor it '
                f'came from.  A tensor it cannot name is one it would silently '
                f'drop, and a checkpoint missing a tensor fails on the GPU '
                f'instead of here.  Add the entry to sglang_gguf.py.')

    def _register(self, split, geo, gname, gsuffix, hf, ops):
        _path, ne, tt = split.entries()[gname]
        if tt not in GGML_TYPES:
            raise SystemExit(f'{split.label()}: {gname} is ggml type {tt}, which '
                             f'this reader does not decode')
        hf_dtype = 'F32' if gsuffix in self.f32_suffixes else 'BF16'
        if tt == 30:
            hf_dtype = 'BF16'                    # a BF16 shard is already narrow
        shape = list(reversed(ne))
        if 'conv_kernel' in ops:
            shape = [shape[0], 1] + shape[1:]
        if 'stack' in ops:
            # one GGUF tensor, E HF tensors: [E, N, K] -> E x [N, K]
            n_exp, n, k = shape[0], shape[1], shape[2]
            for eid in range(n_exp):
                self._put(hf.format(eid=eid),
                          dict(split=split, geo=geo, gguf=gname, ops=(),
                               dtype=hf_dtype, shape=[n, k],
                               first=eid * n * k, count=n * k))
            return
        if 'patch_t0' in ops or 'patch_t1' in ops:
            # two GGUF tensors, one HF tensor: [O,C,H,W] x 2 -> [O,C,2,H,W]
            e = self._entries.get(hf)
            slot = 0 if 'patch_t0' in ops else 1
            if e is None:
                e = dict(split=split, geo=geo, gguf=None, ops=('patch',),
                         dtype=hf_dtype,
                         shape=[shape[0], shape[1], 2] + shape[2:],
                         halves=[None, None])
                self._entries[hf] = e
            e['halves'][slot] = gname
            return
        self._put(hf, dict(split=split, geo=geo, gguf=gname, ops=ops,
                           dtype=hf_dtype, shape=shape))

    def _put(self, hf, plan):
        if hf in self._entries:
            raise SystemExit(f'{hf}: two GGUF tensors claim the same HF name')
        self._entries[hf] = plan

    # -- the ST index interface -------------------------------------------- #
    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __contains__(self, k):
        return k in self._entries

    def __getitem__(self, name):
        p = self._entries[name]
        return (p['split'].dir, {'dtype': p['dtype'], 'shape': p['shape']}, 0)

    def keys(self):
        return self._entries.keys()

    def items(self):
        return ((n, self[n]) for n in self._entries)

    def get(self, name, default=None):
        return self[name] if name in self._entries else default

    # -- the per-tensor unit a SPECIAL_TENSOR file holds -------------------- #
    #
    # A GGUF split ships ONE TENSOR PER FILE and a `tensors.map` that names it,
    # and `quant_downloader.sh` assembles any recipe from those files because
    # the recipe and the map speak the same GGUF names.  An SGLang split has to
    # ship the same unit to reuse the same downloader - so the unit is the GGUF
    # TENSOR, and the file it names carries every HF tensor that came out of it.
    #
    # The map is not one-to-one in either direction and both exceptions land
    # here rather than in the writer:
    #
    #   one GGUF tensor, many HF tensors   'stack': a MoE layer's E experts are
    #                                      one [E, N, K] tensor in GGUF and E
    #                                      modules in HF.  All E go in the file.
    #   two GGUF tensors, one HF tensor    'patch_t0'/'patch_t1': the temporal
    #                                      patch embedding is one Conv3D in HF
    #                                      and two Conv2D halves in GGUF.  The
    #                                      FIRST half's file carries it and the
    #                                      second's is empty - an empty file
    #                                      rather than a missing one, because
    #                                      the downloader checks that the shard
    #                                      numbering has no hole in it.
    #   one GGUF tensor, no HF tensor      the `mtp_duplicates` the companion
    #                                      repeats so it is loadable alone; the
    #                                      main split owns them, so the
    #                                      companion's files for them are empty
    #                                      for the same reason.

    def gguf_groups(self):
        """[(split, gguf name, [hf names])], each split in its own map order."""
        by_gguf = {}
        for hf, p in self._entries.items():
            for g in ([p['gguf']] if p['gguf']
                      else [h for h in p['halves'] if h]):
                by_gguf.setdefault((p['split'].dir, g), []).append(hf)
        out, claimed = [], set()
        for sp in self._splits:
            for g in sp.universe():
                hf = [h for h in sorted(set(by_gguf.get((sp.dir, g), [])))
                      if h not in claimed]
                claimed.update(hf)
                out.append((sp, g, hf))
        left = set(self._entries) - claimed
        if left:
            raise SystemExit(f'{len(left)} tensor(s) belong to no GGUF tensor of '
                             f'any split, e.g. {sorted(left)[:3]}')
        return out

    # -- the model's own GGUF tensor set, which expands a merged recipe ---- #
    def universe(self):
        """The language model's GGUF tensor names, in the split's own order.

        A recipe is written against the model folder's `tensors.bf16.map`, which
        is the MAIN split's list - not the vision tower's and not the draft
        head's, which are converted separately and have their own maps.  So this
        deliberately returns only the text part's names.
        """
        return self.parts['text'].universe()

    # -- reading ----------------------------------------------------------- #
    def read(self, name):
        p = self._entries[name]
        what = f'{name} (from {p["gguf"] or p["halves"]})'
        if p['ops'] == ('patch',):
            if any(h is None for h in p['halves']):
                raise SystemExit(f'{name}: the split carries only one half of the '
                                 f'temporal patch embedding')
            a = np.stack([p['split'].read(g) for g in p['halves']], axis=2)
            return invert(a, (), p['geo'], what, p['dtype'], self.stats)
        a = p['split'].read(p['gguf'], p.get('first'), p.get('count'))
        if p.get('count') is not None:
            a = a.reshape(p['shape'])
        return invert(a, p['ops'], p['geo'], what, p['dtype'], self.stats)

    def raw(self, name):
        """The tensor's HF bytes - what a byte-for-byte comparison compares."""
        return np.ascontiguousarray(self.read(name)).tobytes()

    # -- what the writer reports ------------------------------------------- #
    def inexact(self):
        """(tensors, elements, of, worst spread, worst magnitude), ambiguous.

        Zero on every tensor except a zero-centred norm whose gamma is smaller
        than one f32 ULP at 1.0 - `_undo_plus_one` says why, and this is what
        makes that visible in the build log and in BUILD.json instead of only in
        a byte comparison somebody may never run.
        """
        d = self.stats.get('inexact', {})
        if not d:
            return 0, 0, 0, 0.0, 0.0
        return (len(d), sum(v[0] for v in d.values()),
                sum(v[1] for v in d.values()), max(v[2] for v in d.values()),
                max(v[3] for v in d.values()))

    def describe(self):
        n_t, n_e, n_of, spread, mag = self.inexact()
        return {'kind': 'gguf-split', 'arch': self.arch_key,
                'order': self.order_from,
                'parts': {k: SN.publishable_path(v.dir) for k, v in
                          sorted(self.parts.items())},
                'ambiguous_tensors': n_t, 'ambiguous_elements': n_e,
                'ambiguous_of_elements': n_of,
                'ambiguous_worst_spread': spread,
                'ambiguous_worst_magnitude': mag}


def _hf_index_order(path):
    """{tensor name: rank} in the order a safetensors snapshot presents them.

    `sglang_st.index_dir` reads `sorted(glob('*.safetensors'))` and takes each
    file's header in order, and every safetensors writer emits a sorted header,
    so the snapshot order is `(shard file, name)` - which this recovers from the
    index alone, no snapshot needed.  MEASURED on Qwen3.8-27B: the order derived
    here is the identical 1,199-name list `index_dir` returns.
    """
    return hf_index_order(json.load(open(path, encoding='utf-8')))


def hf_index_order(doc):
    """The same, from the parsed document - what an assembler has in hand.

    `sglang_write.py --assemble` gets the index out of the split's metadata
    shard rather than off a disk it has no snapshot on, so the ranking rule
    lives here once and both callers use it.
    """
    wm = doc['weight_map']
    names = sorted(wm, key=lambda n: (wm[n], n))
    return {n: i for i, n in enumerate(names)}


# =============================================================================
# 5. THE NON-TENSOR FILES
# =============================================================================
#
# A checkpoint is not only weights.  What a GGUF split can and cannot give back:
#
#   config.json                    NO.  The GGUF carries the hyperparameters as
#                                  ggml keys (block_count, embedding_length,
#                                  head counts, rope base, rms eps), which is a
#                                  different encoding of SOME of them and none
#                                  of the rest: no class name, no vision_config,
#                                  no dtype, no rope_parameters dict.  The
#                                  writer needs the real file - it edits it,
#                                  adding quantization_config - so this one is
#                                  required.  The GGUF keys are used to CHECK it.
#   chat_template.jinja            YES, verbatim (`tokenizer.chat_template`).
#                                  Regenerated when absent; byte-compared when
#                                  present.
#   tokenizer.json                 NO.  The GGUF has the token strings, the
#   tokenizer_config.json          merges and the token types, which rebuild a
#   vocab.json / merges.txt        LLAMA.CPP tokenizer, not a `tokenizers`
#   special_tokens_map.json        one - no normalizer, no pre-tokenizer, no
#                                  decoder, no post-processor, no added-token
#                                  flags.  The vocabulary SIZE and the special
#                                  token ids are checked against the GGUF.
#   generation_config.json         NO, though the GGUF carries the sampling
#                                  defaults (`general.sampling.*`) and the
#                                  eos/bos/pad ids, which are checked.
#   preprocessor_config.json       NO.  Vision/video preprocessing lives in the
#   video_preprocessor_config.json mmproj GGUF as clip.* keys (image mean, std,
#                                  patch size) but not as the file transformers
#                                  reads.
#   model.safetensors.index.json   NOT NEEDED to serve.  Used, when present, to
#                                  reproduce the snapshot's tensor ORDER so the
#                                  two builds match shard for shard.
#
# All of them are small; all of them are in the model's own repository; none of
# them is a weight.  `--hf-files` is where the writer looks for them.

# (file, required, what a missing one costs)
HF_FILE_INVENTORY = (
    ('config.json', True,
     'the writer edits it into the checkpoint config; nothing can stand in for it'),
    ('generation_config.json', False,
     'serving defaults (temperature, top_p, eos); SGLang falls back to its own'),
    ('tokenizer.json', False, 'the fast tokenizer; a text server needs it'),
    ('tokenizer_config.json', False, 'tokenizer class, special tokens, padding'),
    ('vocab.json', False, 'the slow-tokenizer vocabulary, when the model ships one'),
    ('merges.txt', False, 'the slow-tokenizer BPE merges, when the model ships one'),
    ('special_tokens_map.json', False, 'special-token aliases, when the model ships one'),
    ('chat_template.jinja', False,
     'REGENERATED from the GGUF metadata when absent'),
    ('preprocessor_config.json', False, 'image preprocessing, on a vision model'),
    ('video_preprocessor_config.json', False, 'video preprocessing, on a vision model'),
    ('model.safetensors.index.json', False,
     'the snapshot tensor ORDER, which makes the two builds match shard for shard'),
    ('LICENSE', False, 'the model licence'),
)


def hf_files(directory, source: 'GgufSource', log=print):
    """Inventory `--hf-files`, validate it against the GGUF, fill what can be.

    Returns {filename: path or None} plus a list of generated (name, text)
    pairs the writer should write.  Every check here compares a fact the GGUF
    metadata states against the same fact in the file, so a config.json from a
    DIFFERENT model is caught before a 15 GB write rather than on the GPU.
    """
    present, generated, problems = {}, [], []
    for name, required, why in HF_FILE_INVENTORY:
        p = os.path.join(directory, name) if directory else None
        if p and os.path.exists(p):
            present[name] = p
        elif required:
            problems.append(f'{name} is missing from --hf-files: {why}')
        else:
            present[name] = None

    # A COMPANION IS NOT A CHECKPOINT.  Everything below is a fact of the TEXT
    # part - the hyperparameters config.json is checked against, the tokenizer,
    # the chat template - and an `mmproj-` or `mtp-` split carries none of
    # them.  Pointed at one, this used to die on `KeyError: 'text'` several
    # frames from the command line that caused it; say which part is missing
    # and which directory has it instead.
    main = source.parts.get('text')
    if main is None:
        pres = dir_prefixes(source.arch_key)
        held = ', '.join(sorted(source.parts)) or 'no part this reader names'
        want = [os.path.join(os.path.dirname(sp.dir.rstrip('/')),
                             sp.label()[len(pres[part]):])
                for part, sp in sorted(source.parts.items())
                if pres.get(part) and sp.label().startswith(pres[part])]
        raise SystemExit(
            f'--source carries the {held} part of this model and not its '
            f'text part, which is the one that has the hyperparameters, the '
            f'tokenizer and the chat template - there is nothing here to '
            f'check --hf-files against or to write a config.json from. Point '
            f'--source at the MAIN split'
            + (f', {want[0]}' if want else
               ' - the same directory name without the part prefix, in the '
               'same parent') + '; its '
            + ' / '.join(sorted(set(pres.values())))
            + ' companions are found beside it and are not passed on their own.')
    kv = main.first.kv
    tmpl = kv.get('tokenizer.chat_template')
    if tmpl is not None:
        if present.get('chat_template.jinja'):
            have = open(present['chat_template.jinja'], encoding='utf-8').read()
            if have != tmpl:
                log('  note: chat_template.jinja differs from the GGUF\'s own '
                    'tokenizer.chat_template; the file is used, as given')
        else:
            generated.append(('chat_template.jinja', tmpl))
            log('  chat_template.jinja regenerated from the GGUF metadata '
                f'({len(tmpl)} chars)')

    if present.get('config.json'):
        problems += check_config(present['config.json'], source)
    return present, generated, problems


def _cfg_get(cfg, key):
    """A hyperparameter, whether the config nests its text half or not."""
    if key in cfg:
        return cfg[key]
    tc = cfg.get('text_config') or {}
    return tc.get(key)


def check_config(path, source: 'GgufSource'):
    """The GGUF's own hyperparameters against the config.json it is given.

    NOT a schema check - a MODEL check.  Every row is a number the converter
    wrote into the GGUF from this very config, so a mismatch means the two are
    not the same model.  A key the GGUF does not carry is skipped rather than
    guessed at.
    """
    cfg = json.load(open(path, encoding='utf-8'))
    gg = source.parts['text'].first
    n_mtp = int(gg.akv('nextn_predict_layers', 0) or 0)
    rows = (
        ('block_count', 'num_hidden_layers', -n_mtp),
        ('embedding_length', 'hidden_size', 0),
        ('feed_forward_length', 'intermediate_size', 0),
        ('attention.head_count', 'num_attention_heads', 0),
        ('attention.head_count_kv', 'num_key_value_heads', 0),
        ('context_length', 'max_position_embeddings', 0),
        ('expert_count', 'n_routed_experts', 0),
    )
    out = []
    for gk, ck, adj in rows:
        gv = gg.akv(gk)
        cv = _cfg_get(cfg, ck)
        if gv is None or cv is None:
            continue
        if int(gv) + adj != int(cv):
            out.append(f'config.json {ck}={cv} but the GGUF says {gk}={gv}'
                       + (f' ({-adj} of them the MTP head)' if adj else ''))
    toks = gg.kv.get('tokenizer.ggml.tokens')
    vocab = _cfg_get(cfg, 'vocab_size')
    if toks is not None and vocab is not None and len(toks) != int(vocab):
        out.append(f'config.json vocab_size={vocab} but the GGUF carries '
                   f'{len(toks)} tokens')
    arch_key = SN.arch_from_hf_config(cfg)
    if arch_key and arch_key != source.arch_key:
        out.append(f'config.json describes {arch_key} and the split is '
                   f'{source.arch_key}')
    return out


# =============================================================================
# 6. OPENING A SOURCE
# =============================================================================

def looks_like_split(path):
    """True when `--source` points at a GGUF split rather than a snapshot."""
    if os.path.isfile(path):
        return path.endswith('.gguf')
    if os.path.isdir(path):
        return (not glob.glob(os.path.join(path, '*.safetensors'))
                and bool(glob.glob(os.path.join(path, '*.gguf'))))
    return False


def open_split(path, companions=None, hf_files_dir=None, arch_key=None,
               log=print):
    """A `GgufSource` over the split at `path` and every companion beside it."""
    main = Split(path)
    found = list(companions) if companions else find_companions(
        main, arch_key or GGUF_ARCH_KEYS.get(main.arch))
    hf_index = None
    if hf_files_dir:
        p = os.path.join(hf_files_dir, 'model.safetensors.index.json')
        hf_index = p if os.path.exists(p) else None
    src = GgufSource(main, found, hf_index=hf_index, arch_key=arch_key, log=log)
    log(f'  tensor order: {src.order_from}')
    return src


def expected_parts(cfg, arch_key=None):
    """Which parts this config needs as SPLITS OF THEIR OWN.

    TWO DIFFERENT FACTS, and a gate that uses only the first is wrong.  The
    CONFIG says what the model has: a `vision_config` means a vision tower, an
    MTP layer count means a draft head.  The INVERSION TABLE says how the
    converter lays that out, and on that the two architectures disagree - on
    qwen3_5 the tower and the draft head are companion sets of their own
    (`mmproj-`, `mtp-`, which is what docs/Convert model to BF16.md tells you to
    write), while on glm4_moe the draft head is layer 92 of the same stack
    (`Glm4MoeModel.filter_tensors` keeps it there), so no `mtp-` split exists,
    none can be passed, and none is missing.  Only a part this arch's table
    gives a `dir_prefix` can arrive as a companion, so only those are demanded:
    asking for a split the converter never writes would refuse a build that is
    already complete.  With no `arch_key` the config alone is answered for.
    """
    parts = {'text'}
    if 'vision_config' in cfg or 'vision_config' in (cfg.get('text_config') or {}):
        parts.add('vision')
    n = (_cfg_get(cfg, 'mtp_num_hidden_layers')
         or _cfg_get(cfg, 'num_nextn_predict_layers') or 0)
    if int(n or 0) > 0:
        parts.add('mtp')
    if arch_key is not None:
        parts &= {'text'} | set(dir_prefixes(arch_key))
    return parts


# =============================================================================
# 7. SELFTEST
# =============================================================================

def selftest(verbose=True) -> int:
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)
        if verbose:
            print(('  ok   ' if ok else '  FAIL ') + msg)

    # -- the permutation, both ways ---------------------------------------- #
    nk, r, hd = 4, 3, 2
    fwd, inv = _v_perm(nk, r, hd)
    x = np.arange(nk * r * hd)
    chk((x[fwd][inv] == x).all(), 'v-head regrouping: the inverse undoes it')
    chk((fwd == np.arange(nk * r * hd).reshape(nk, r, hd)
         .transpose(1, 0, 2).reshape(-1)).all(),
        'v-head regrouping: the forward index IS the converter\'s reshape/swap')
    chk((inv == np.arange(r * nk * hd).reshape(r, nk, hd)
         .transpose(1, 0, 2).reshape(-1)).all(),
        'v-head regrouping: the inverse is the same view with nk and r exchanged '
        '- the two derivations agree')

    # -- the numeric inversions and their guards ---------------------------- #
    src = np.array([0.05029297, -0.0625, 0.87109375, -1.5], np.float32)
    bits = _narrow_bf16(src, 't')
    chk((_widen_bf16(bits) == src).all(), 'bf16 narrowing: exact on bf16 values')
    try:
        _narrow_bf16(np.float32([1.0000001]), 't')
        chk(False, 'bf16 narrowing: a non-bf16 value must be refused')
    except SystemExit:
        chk(True, 'bf16 narrowing: a value that is not a BF16 is refused, not '
                  'silently rounded')
    geo = Geometry.__new__(Geometry)
    geo.nk = geo.nv = 1
    geo.r = 1
    geo._cache = {}
    got = invert(src + np.float32(1.0), ('plus_one',), geo, 't', 'BF16')
    chk((got == bits).all(), 'plus_one: (x + 1) - 1 recovers the bf16 exactly')
    a_log = np.float32([-3.203125, -2.65625, -4.65625])
    ggufd = (-np.exp(a_log.astype(np.float64))).astype(np.float32)
    got = invert(ggufd, ('neg_exp',), geo, 't', 'BF16')
    chk((_widen_bf16(got) == a_log).all(),
        'neg_exp: log(-x) recovers A_log, and the value lands inside its bf16 cell')
    try:
        invert(np.float32([1.0]), ('neg_exp',), geo, 't', 'BF16')
        chk(False, 'neg_exp: a non-negative -exp() must be refused')
    except SystemExit:
        chk(True, 'neg_exp: a tensor that cannot be -exp(x) is refused')

    # -- the tables agree with the size model's name map -------------------- #
    # ARCHS names the MODULE a recipe assigns; the tables here name the TENSOR a
    # file carries.  Wherever both speak - a linear or an embedding of the layer
    # stack - the disk name DiskNames derives must be the HF leaf declared here,
    # or one of the two is wrong.
    for key, table in sorted(INVERTERS.items()):
        arch = SN.ARCHS[key]
        dn = SN.DiskNames(arch)
        spec = table['parts']['text']
        n = 0
        for gsuf, (hf_leaf, ops) in sorted(spec['tensors'].items()):
            ent = arch['tensors'].get(gsuf)
            if ent is None or ent[1] not in ('linear', 'embedding'):
                continue
            disk = arch.get('hf_disk', {}).get(gsuf, (ent[0], None))[0]
            want = disk.replace('{eid}', '0') + '.weight'
            got = hf_leaf.replace('{eid}', '0')
            n += 1
            chk(got == want,
                f'{key}: {gsuf} -> {got} agrees with the size model\'s disk name '
                f'({want})')
        chk(n > 0, f'{key}: the inversion table and ARCHS share {n} tensor name(s)')
        for gname, (hf, ops) in sorted(spec['globals'].items()):
            ent = arch['globals'].get(gname)
            if ent is None:
                continue
            chk(hf == ent[0] + '.weight',
                f'{key}: global {gname} -> {hf} agrees with ARCHS ({ent[0]})')
        chk(set(spec['tensors']) >= set(arch['tensors']),
            f'{key}: every ARCHS tensor suffix has an inversion '
            f'({sorted(set(arch["tensors"]) - set(spec["tensors"]))[:4]})')
        chk(set(spec['globals']) >= set(arch['globals']),
            f'{key}: every ARCHS global has an inversion')

    # -- expert unstacking is a slice, and the slice is the right one ------- #
    e, nn, kk = 3, 2, 4
    stacked = np.arange(e * nn * kk, dtype=np.uint16).reshape(e, nn, kk)
    chk(all((stacked.reshape(-1)[i * nn * kk:(i + 1) * nn * kk].reshape(nn, kk)
             == stacked[i]).all() for i in range(e)),
        'stacked experts: expert i is the i-th contiguous [N, K] block, so one '
        'expert is a seek and a read - the stack is never materialised')

    # -- the per-tensor unit, at all three places the map is not one-to-one - #
    class _FakeSplit:
        def __init__(self, d, names):
            self.dir, self._n = d, list(names)

        def universe(self):
            return list(self._n)

    sp_a = _FakeSplit('/a', ['ffn_gate_exps.weight', 'token_embd.weight'])
    sp_b = _FakeSplit('/b', ['v.patch_embd.weight', 'v.patch_embd.weight.1',
                             'token_embd.weight'])
    gs = GgufSource.__new__(GgufSource)
    gs._splits = [sp_a, sp_b]
    gs._entries = {
        'mlp.experts.0.gate_proj.weight': dict(split=sp_a,
                                               gguf='ffn_gate_exps.weight'),
        'mlp.experts.1.gate_proj.weight': dict(split=sp_a,
                                               gguf='ffn_gate_exps.weight'),
        'model.embed_tokens.weight': dict(split=sp_a, gguf='token_embd.weight'),
        'model.visual.patch_embed.proj.weight': dict(
            split=sp_b, gguf=None,
            halves=['v.patch_embd.weight', 'v.patch_embd.weight.1']),
    }
    got = [(sp.dir, g, hf) for sp, g, hf in gs.gguf_groups()]
    chk(got == [('/a', 'ffn_gate_exps.weight',
                 ['mlp.experts.0.gate_proj.weight',
                  'mlp.experts.1.gate_proj.weight']),
                ('/a', 'token_embd.weight', ['model.embed_tokens.weight']),
                ('/b', 'v.patch_embd.weight',
                 ['model.visual.patch_embed.proj.weight']),
                ('/b', 'v.patch_embd.weight.1', []),
                ('/b', 'token_embd.weight', [])],
        'gguf_groups: E stacked experts land in one file, the temporal patch '
        'pair lands in the FIRST half\'s and leaves the second empty, and a '
        'companion\'s repeated global is empty because the main split owns it '
        f'({got})')
    chk(sorted(n for _s, _g, hf in gs.gguf_groups() for n in hf)
        == sorted(gs._entries),
        'gguf_groups: every HF tensor is claimed exactly once, so an assembler '
        'that unions the files gets the checkpoint and no duplicate')

    # -- the snapshot order is recoverable from the index alone ------------- #
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'i.json')
        json.dump({'weight_map': {'b': 'model-00002.safetensors',
                                  'a': 'model-00002.safetensors',
                                  'z': 'model-00001.safetensors'}},
                  open(p, 'w', encoding='utf-8'))
        rank = _hf_index_order(p)
        chk(sorted(rank, key=rank.get) == ['z', 'a', 'b'],
            'snapshot order: (shard file, then name) - which is what index_dir '
            'returns and what makes the two builds match shard for shard')

    # -- the container refuses what it must --------------------------------- #
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'x.gguf'), 'wb').write(b'NOPE' + b'\0' * 32)
        try:
            GgufFile(os.path.join(d, 'x.gguf'))
            chk(False, 'container: a non-GGUF file must be refused')
        except ValueError:
            chk(True, 'container: a file without the GGUF magic is refused')

    # -- the parts a config demands ----------------------------------------- #
    chk(expected_parts({'vision_config': {}, 'text_config':
                        {'mtp_num_hidden_layers': 1}}) == {'text', 'vision', 'mtp'},
        'expected parts: a vision_config means a vision tower and an MTP layer '
        'count means a draft head - derived from the config, not listed')
    chk(expected_parts({'num_hidden_layers': 92}) == {'text'},
        'expected parts: a plain text model needs only the main split')

    # -- and the same question asked of a real config, per architecture ------ #
    #
    # THE CONFIG IS NOT THE LAYOUT.  GLM-4.7's config.json declares
    # num_nextn_predict_layers 1 and the converter still puts that head in the
    # main stack, so a gate that reads only the config demands an `mtp-` split
    # that does not exist and refuses a complete source.  Both halves are
    # asserted on the real files: the config that says there IS a draft head,
    # and the model's own GGUF tensor list that says which split carries it.
    def _real_cfg(repo, fallback):
        snap = SN._hf_snapshot(repo)
        p = os.path.join(snap, 'config.json') if snap else None
        if p and os.path.exists(p):
            return json.load(open(p, encoding='utf-8')), repo + ' config.json'
        return fallback, 'a literal fixture (' + repo + ' is not cached here)'

    gcfg, gwhy = _real_cfg('zai-org/GLM-4.7',
                           {'architectures': ['Glm4MoeForCausalLM'],
                            'num_hidden_layers': 92,
                            'num_nextn_predict_layers': 1})
    chk(int(_cfg_get(gcfg, 'num_nextn_predict_layers') or 0) == 1
        and expected_parts(gcfg) == {'text', 'mtp'}
        and expected_parts(gcfg, 'glm4_moe') == {'text'},
        f'expected parts: {gwhy} declares a draft head, and glm4_moe keeps '
        f'it in the main stack - so no companion is demanded and a GLM '
        f'split of part(s) text passes the gate')

    qcfg, qwhy = _real_cfg('Qwen/Qwen3.8-27B',
                           {'vision_config': {},
                            'text_config': {'mtp_num_hidden_layers': 1}})
    chk(expected_parts(qcfg, 'qwen3_5') == {'text', 'mtp', 'vision'},
        f'expected parts: {qwhy} declares both, and qwen3_5 splits both out - '
        f'the mtp- and mmproj- companions are still required')

    gmap = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'models', 'GLM-4.7', 'group0', 'tensors.bf16.map')
    if os.path.exists(gmap):
        gnames = [ln.split(':')[2] for ln in open(gmap, encoding='utf-8')
                  if ln.strip()]
        tspec = INVERTERS['glm4_moe']['parts']['text']
        unknown = []
        for n in gnames:
            m = _BLK_RE.match(n)
            ent = (tspec['tensors'].get(m.group(2)) if m
                   else tspec['globals'].get(n))
            if ent is None:
                unknown.append(n)
        nextn = sum(1 for n in gnames if n.startswith('blk.92.'))
        chk(not unknown and nextn > 0 and dir_prefixes('glm4_moe') == {},
            f'expected parts: all {len(gnames)} GGUF names of the real GLM-4.7 '
            f'map - the {nextn} blk.92.* draft-head rows included - are '
            f'named by the glm4_moe table\'s ONE part, and that arch '
            f'declares no companion prefix at all'
            + (f' (unknown: {unknown[:3]})' if unknown else ''))
    else:
        chk(False, f'expected parts: {gmap} is missing, so the real GLM tensor '
                   f'name set could not be checked')

    # -- --source pointed at a companion, which carries no text part ------- #
    #
    # `mmproj-` and `mtp-` splits are downloaded beside the main one and are
    # easy to name by mistake.  Everything hf_files() checks is a fact of the
    # text part, so there is nothing here to check anything against; it used to
    # be a bare KeyError: 'text' several frames from the command line.
    class _Companion:
        arch_key = 'qwen3_5'

        class _Sp:
            dir = '/models/mmproj-Qwen3.8-27B-THIREUS-BF16-SPECIAL_SPLIT'

            def label(self):
                return os.path.basename(self.dir)

        def __init__(self):
            self.parts = {'vision': self._Sp()}

    try:
        hf_files(None, _Companion(), log=lambda *a: None)
        chk(False, '--source: a companion split must be refused by hf_files')
    except SystemExit as e:
        m = str(e)
        chk('vision' in m and 'text part' in m
            and '/models/Qwen3.8-27B-THIREUS-BF16-SPECIAL_SPLIT' in m
            and 'mmproj-' in m,
            '--source: a companion split names the part it holds, the part it '
            'does not, and the main directory to point at instead - not '
            f'KeyError({chr(39)}text{chr(39)}) ({m[:60]}...)')

    if verbose:
        print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', help='a SPECIAL_SPLIT directory, or its first shard')
    ap.add_argument('--hf-files', default=None,
                    help='directory holding the model\'s small non-tensor files')
    ap.add_argument('--list', action='store_true',
                    help='print the HF tensors the split can give back')
    ap.add_argument('--meta', action='store_true',
                    help='print the split\'s GGUF metadata keys')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.split:
        ap.print_help()
        return 0
    if a.meta:
        gg = Split(a.split).first
        for k, v in gg.kv.items():
            s = repr(v)
            print(f'  {k} = {s if len(s) <= 120 else s[:120] + f"... ({len(v)})"}')
        return 0
    src = open_split(a.split, hf_files_dir=a.hf_files)
    if a.hf_files:
        _p, _g, problems = hf_files(a.hf_files, src)
        for m in problems:
            print(f'  [problem] {m}')
    print(f'  {len(src)} HF tensors')
    if a.list:
        for n in src:
            e = src[n][1]
            print(f'    {n:70s} {e["dtype"]:5s} {tuple(e["shape"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
