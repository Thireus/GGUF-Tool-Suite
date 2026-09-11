#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_write.py turns a recipe into a checkpoint SGLang   **#
#** can load, then verifies it against the BF16 source.       **#
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
#** Copyright © 2026 - Thireus.       𝒸ₕₑ𝒸ₖₚₒᵢₙₜ ₒ𝒻 ₙₒ ᵣₑₜᵤᵣₙ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_write.py - write, verify and cross-check an SGLang `modelopt_mixed`
checkpoint from a BF16 source and a `quant_assign.py` recipe.

CPU ONLY.  Opens no GPU, starts no server, loads no model.

    # build, from an HF BF16 snapshot
    sglang_write.py --build --source <BF16 snapshot> --recipe <x.recipe> \\
        --input-scales-from <a calibrated NVFP4 checkpoint> \\
        --out <output checkpoint dir> --name <name>

    # build, from the BF16 GGUF SPECIAL_SPLIT instead - same checkpoint
    sglang_write.py --build --source <BF16 split dir> --hf-files <small files> \\
        --recipe <x.recipe> --input-scales-from <...> --out <...> --name <...>

    # audit, before any GPU sees it
    sglang_write.py --verify     --ckpt <out> --source <either source>
    sglang_write.py --crosscheck --ckpt <out> --against <published ckpt>

    # ship it: one per-tensor repository per SGLang qtype, and back again
    sglang_write.py --split --qtype sgl_nvfp4 --source <BF16 split> \\
        --hf-files <small files> --input-scales-from <...> --out <dir>
    sglang_write.py --assemble <downloaded dir> [<companion dir> ...] \\
        --out <checkpoint dir>

`--arch` is optional and is read from whatever is given (config.json,
general.architecture, the tensor names); every witness must agree.

TWO SOURCES, ONE CHECKPOINT
---------------------------
`--source` takes either copy of the BF16 weights a machine is likely to have:
the HF safetensors snapshot, or the owner's BF16 GGUF SPECIAL_SPLIT that
`quant_downloader.sh` fetches.  `sglang_gguf.py` inverts every transformation
`convert_hf_to_gguf.py` applied - the name map, the zero-centred norm offset,
the V-head regrouping on six tensors and three axes, `-exp` on A_log, the
squeezed conv kernel, the stacked experts, the temporal split of the patch
embedding - and hands back the same HF tensors, so everything below this line is
one code path and the two cannot drift.  The split does not carry the vision
tower or the draft head; those are their own `mmproj-` and `mtp-` splits and are
picked up from beside the main one.  It does not carry the checkpoint's small
text files either, which is what `--hf-files` is for; `sglang_gguf` says
precisely which of them are needed, checks them against the GGUF's own
metadata, and regenerates the chat template from it.

MEASURED on Qwen3.8-27B: the two sources give the same 1,199 tensors, in the
same order, at the same widths, and the F recipe built from each comes out
byte-identical in 2,378 of its 2,404 tensors - every quantised one included.
The 26 that differ are all zero-centred norms, and they differ in 36 of the
694,784 gammas this model has, by at most 5.96e-08: `gamma + 1` stored in
float32 cannot say which tiny bf16 gamma it came from.  The reader flags exactly
that as it reads (46 elements, an upper bound on the 36) rather than leaving it
to be discovered.  See `sglang_gguf._undo_plus_one`.

HOW A CHECKPOINT SHIPS: ONE REPOSITORY PER QTYPE, ONE FILE PER TENSOR
---------------------------------------------------------------------
THE UNIT IS THE GGUF TENSOR, because that is what a recipe names.  The suite
already ships GGUF models as one file per tensor in a repository called
`<MODEL>-<MAINTAINER>-<QTYPE>-SPECIAL_SPLIT`, with a `tensors.map` giving each
tensor's file, sha256, shape and byte count, and `quant_downloader.sh` cooks any
recipe out of those repositories by fetching, per tensor, the file from the
repository of the qtype the recipe assigned it.  `--split` ships an SGLang
checkpoint the same way: `<MODEL>-<MAINTAINER>-SGL_NVFP4-SPECIAL_SPLIT` holds the
whole model at NVFP4, one `...-SPECIAL_TENSOR-NNNNN-of-MMMMM.safetensors` per
GGUF tensor, and that file carries every HF tensor the module needs at that
format - `weight`, `weight_scale`, `weight_scale_2`, `input_scale` or
`weight_scale_inv`, whichever the format registers.  NNNNN is the model's OWN
GGUF chunk id for that tensor and MMMMM its chunk total, so an SGL repository is
numbered exactly like the BF16 GGUF one it was made from and a recipe written
against `tensors.bf16.map` addresses it unchanged.  Shard 00001 is the metadata
shard, as in a GGUF split: no weights, and in the main split the checkpoint's
small text files - config.json, the tokenizer, chat_template.jinja, the
preprocessor configs and the source model.safetensors.index.json - as U8 tensors
named `__hf_file__/<name>`, so everything a checkpoint needs travels through the
one downloader, hashed and signed like every other file.

THE DOWNLOADER NEEDED ONE EXTENSION AND NOT A SECOND SCRIPT.  For a qtype
matching `sgl_*`, `quant_downloader.sh` and `tensor_downloader.sh` swap `.gguf`
for `.safetensors` in the names they build and the map lines they parse, check
the safetensors header instead of the GGUF magic, and refuse the GGUF-only
extras (gguf_info.py verification, quantise-from-bf16, the computed maps) that
have nothing to inspect here.  The map format, the sha256 of every file, the GPG
signature on the map and on shard 00001, the resume, the .zbst handling,
`--verify`, `--individual-tensors` and the node sharding are all untouched -
they are about FILES, not about what is in them.  One rule is new: a tensor no
recipe pattern matches is fetched from the repository of `--qtype`, because in
the SGLang container BF16 IS ENCODED BY ABSENCE (a module missing from
`quantized_layers` resolves to `UnquantizedLinearMethod`), so `--qtype sgl_bf16`
is what makes a recipe that names only its quantised modules - which is what
`quant_assign.py --ignore-f32` emits - resolve to a whole model.

AND THE BYTES ARE THE SAME BYTES.  A per-tensor file is written by the same
`quantise()` the recipe path uses, from the same weight amax and the same
calibrated activation amax, and neither of those depends on the recipe: both
maxima are taken over EVERY shard of the fused group, and a recipe cannot assign
part of a group anyway.  So the file `--split` writes for a module at NVFP4 is
byte-for-byte the file `--build` would write for it under any recipe that
chooses NVFP4 there, and `--assemble` - which re-plans the shards in the source
snapshot's own tensor order, the order the metadata shard carries - hands back
the identical checkpoint.  MEASURED on Qwen3.8-27B, F recipe: 2,404 of 2,404
tensors identical, and the four shard files identical byte for byte when the two
runs are given the same `--name` - that name is what `__metadata__.producer`
records, and it is the only thing that differs when they are not.  The split
also CHECKS itself as it writes: `sglang_native.plan_tensor` - the function
`quant_assign.py` prices a recipe with - predicts each tensor's bytes, and a
file whose payload disagrees is a refusal, so the assigner's byte model and the
writer's output cannot drift apart unnoticed.

WHAT A WRITER HAS TO GET RIGHT THAT A CODEC CANNOT KNOW
------------------------------------------------------
The codecs in `sglang_codecs.py` are each proven byte-for-byte against a
published checkpoint.  Three further rules live here, because they are
properties of the RECIPE and the MODEL, not of one tensor:

  1. THE FUSED-SHARD ws2 RULE.  `ModelOptFp4LinearMethod` does
     `weight_scale_2 = layer.weight_scale_2.max()` over the shards of a fused
     linear and applies that one alpha to the whole GEMM WITHOUT requantising
     (modelopt_quant.py:1806-1810).  A shard written with its own smaller ws2 is
     dequantised with the larger one and comes out scaled up by the ratio - a
     gross error, and SILENT.  Every NVFP4 shard of a fused module therefore
     gets ONE ws2, from the amax of the whole group.  RadixArk does the same.
     (FP8 does not need it - `requantize_with_max_scale` rescales each shard -
     and FP8_PB_WO does not either, its scales being per 128x128 block.)
  2. THE FUSED-SHARD ALGO RULE.  `_resolve_quant_algo` RAISES "Mixed quant_algo
     within fused layer" if the shards disagree (modelopt_quant.py:963-969).
     Checked before a byte is written, not discovered on the GPU.
  3. ACTIVATION SCALES.  NVFP4 on a linear is W4A4: the activation is quantised
     with a STATIC `input_scale` that is a property of the CALIBRATION DATA, not
     of the weights, and cannot be computed from a checkpoint.  Absent, SGLang
     silently fills 1.0 and quantises activations on the wrong scale.  We
     harvest ModelOpt's own calibrated values from a published checkpoint of the
     same model and convert between the two conventions, which differ only by
     the factor 6:
         FP8   : input_scale = amax_act / 448
         NVFP4 : input_scale = amax_act / (6*448)
     That also makes a head-to-head against that checkpoint maximally fair:
     identical activation calibration, only the weight allocation differs.
     FP8_PB_WO is dynamic-activation and takes no input_scale at all.

AND ONE ENCODING RULE THAT LOOKS LIKE AN OMISSION.  BF16 IS ENCODED BY ABSENCE.
A module missing from `quantized_layers` resolves to `UnquantizedLinearMethod`
(modelopt_quant.py:1027).  No `exclude_modules` entry is needed or wanted - that
list is matched by `is_layer_skipped` and is for whole subtrees (RadixArk uses
it only for `mtp*`).

THE STRONGEST CHECK AVAILABLE WITHOUT A GPU is `--crosscheck`: wherever our
recipe and a published checkpoint chose the same algorithm for the same
module, our written bytes must EQUAL that checkpoint's.  It exercises the whole
pipeline - source read, fused-group amax, weight_scale_2 sharing, packing, shard
writing - on real files.  Across the eight recipes2 checkpoints that is
1,415 / 1,415 modules byte-identical.

TORCH.  Only the codecs need it, and they import it lazily with a precise
error.  `--verify`'s structural half and all of `--crosscheck` are pure numpy,
so a checkpoint can be audited in the suite venv even where it cannot be built.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sglang_gguf as GG            # noqa: E402
import sglang_native as SN          # noqa: E402
import sglang_st as ST              # noqa: E402

WRITEABLE = ('NVFP4', 'W4A16_NVFP4', 'FP8', 'FP8_PB_WO')
ALGOS = WRITEABLE + ('BF16',)
# The small files a checkpoint needs beside its weights.  They are COPIED, never
# generated, because every one of them is an upstream artefact whose exact bytes
# are what transformers and SGLang parse - the inventory, and what a GGUF split
# can and cannot give back, is `sglang_gguf.HF_FILE_INVENTORY`.
COPY_FILES = tuple(n for n, _req, _why in GG.HF_FILE_INVENTORY
                   if n not in ('config.json', 'model.safetensors.index.json'))


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


# =============================================================================
# THE SOURCE - A BF16 SNAPSHOT, OR THE BF16 GGUF SPLIT IT WAS CONVERTED INTO
# =============================================================================
#
# `--source` is whichever copy of the BF16 weights the machine has.  A snapshot
# is read by `sglang_st`; a SPECIAL_SPLIT is read by `sglang_gguf`, which undoes
# every transformation `convert_hf_to_gguf.py` applied and hands back the same
# HF tensors under the same names, in the same layouts, at the same widths.
#
# Everything downstream of these functions is one code path.  That is the
# point: the plan, the fused-group checks, the shard packing, the amax pass and
# `--verify` cannot behave differently for the two sources, because they cannot
# tell them apart.

def open_source(a, log_fn=log):
    """The tensor index for `--source`, whichever kind it is."""
    if GG.looks_like_split(a.source):
        return GG.open_split(a.source, companions=a.gguf_companion,
                             hf_files_dir=a.hf_files or None,
                             arch_key=getattr(a, '_arch_key', None), log=log_fn)
    return ST.index_dir(a.source)


def src_read(idx, name):
    """The tensor as numpy, in HF layout and HF width.

    The only way anything below reads a weight.  A snapshot answers from
    `sglang_st`, a split from `sglang_gguf` after inverting the conversion, and
    the caller cannot tell which - which is the whole point.
    """
    r = getattr(idx, 'read', None)
    return r(name) if r is not None else ST.read(idx, name)


def nontensor_dir(a):
    """Where the checkpoint's small text files come from, and why.

    A snapshot carries them itself.  A GGUF split does not - it carries the
    tokenizer as ggml arrays and the hyperparameters as ggml keys, which is a
    different encoding of some of the same facts and not the bytes SGLang reads
    - so `--hf-files` names a directory holding them, and the GGUF's own
    metadata is used to CHECK that directory belongs to this model.
    """
    if a.hf_files:
        return a.hf_files
    if GG.looks_like_split(a.source):
        raise SystemExit(
            '--source is a GGUF split, which carries weights and metadata but '
            'not the checkpoint\'s small text files (config.json first of all, '
            'which the writer edits into the output).  Pass --hf-files <dir> '
            'holding them; they are the non-LFS files of the model\'s own '
            'repository and come to a few hundred kB:\n    '
            + '\n    '.join(f'{n:32s} {"required" if r else "optional"}  {w}'
                             for n, r, w in GG.HF_FILE_INVENTORY))
    return a.source


# =============================================================================
# THE MODEL'S TENSOR SET, AND WHERE IT COMES FROM
# =============================================================================
#
# A published recipe is regex-compacted (`quants_regex_merger`), so expanding it
# needs the model's tensor list.  There are three places to get one and this is
# the precedence, strongest first:
#
#   1. `--bf16-map`               what the user typed, and the only one that can
#                                 name a set the source does not have.
#   2. the split's own tensors.map  when `--source` IS a split.  `quant_downloader.sh`
#                                 fetches `tensors.<qtype>.map` beside the shards
#                                 and symlinks it as `tensors.map`
#                                 (quant_downloader.sh:3356-3374), so a downloaded
#                                 split describes itself and `--bf16-map` becomes
#                                 optional.  Absent a map file the shard headers
#                                 give the same list, more slowly.
#   3. nothing                    an HF snapshot carries HF names, not GGUF ones,
#                                 so it cannot supply this.  A raw per-tensor
#                                 recipe still builds; a compacted one errors,
#                                 naming the flag.
#
# The two map files ARE the same file: on the shipped Qwen3.8-27B split
# `tensors.map` is a symlink to `tensors.bf16.map`, and both are byte-identical
# to `models/Qwen3.8-27B/group0/tensors.bf16.map`, which is where the recipe was
# made.  So this is a convenience, not a second source of truth.

def recipe_universe(a, src_idx):
    """([gguf tensor names] or None, where they came from)."""
    if getattr(a, 'bf16_map', None):
        return ([r[0] for r in SN.read_bf16_map(a.bf16_map)],
                f'--bf16-map {os.path.basename(a.bf16_map)}')
    if hasattr(src_idx, 'universe'):
        split = src_idx.parts['text']
        return (src_idx.universe(),
                (os.path.basename(split.map_path) if split.map_path
                 else 'the shard headers') + " of the split's own directory")
    return None, 'not needed (a raw per-tensor recipe) or not available'


# =============================================================================
# WHICH ARCHITECTURE, ASKED OF WHATEVER THE COMMAND LINE POINTS AT
# =============================================================================
#
# `--arch` used to default to qwen3_5, which is right for one model and silently
# wrong for the next.  Every artefact a mode is given already states its own
# architecture - a checkpoint and a snapshot in `config.json`, a GGUF split in
# `general.architecture` and again, more strictly, in its tensor names - so the
# flag becomes an OVERRIDE and the default is to read.
#
# Every available witness is consulted and they must AGREE.  Two witnesses that
# disagree mean the command line has put two different models together (a
# checkpoint verified against the wrong snapshot, a split beside somebody else's
# config.json), which is worth a refusal of its own and is not otherwise caught
# until the tensor names fail to line up minutes later.

def arch_witnesses(a):
    """[(what said it, arch key or None, detail)] over everything given."""
    out = []

    def from_config(path, what):
        if path and os.path.exists(os.path.join(path, 'config.json')):
            cfg = json.load(open(os.path.join(path, 'config.json'),
                                 encoding='utf-8'))
            names = cfg.get('architectures') or cfg.get('model_type')
            out.append((what, SN.arch_from_hf_config(cfg), f'{names}'))

    from_config(getattr(a, 'ckpt', None), 'the checkpoint\'s config.json')
    from_config(getattr(a, 'hf_files', None), '--hf-files config.json')
    src = getattr(a, 'source', None)
    if src and GG.looks_like_split(src):
        split = GG.Split(src)
        out.append(('the split\'s general.architecture',
                    SN.arch_from_gguf_arch(split.arch), repr(split.arch)))
        # the strict witness: an arch qualifies only by naming EVERY tensor
        out.append(('the split\'s tensor names',
                    SN.detect_arch(split.universe()),
                    f'{len(split.universe())} name(s)'))
    elif src and os.path.isdir(src):
        from_config(src, 'the source config.json')
        ix = os.path.join(src, 'model.safetensors.index.json')
        if os.path.exists(ix):
            names = list(json.load(open(ix, encoding='utf-8'))['weight_map'])
            out.append(('the snapshot\'s tensor names',
                        SN.detect_arch_hf(names), f'{len(names)} name(s)'))
    if getattr(a, 'bf16_map', None):
        names = [r[0] for r in SN.read_bf16_map(a.bf16_map)]
        out.append(('--bf16-map tensor names', SN.detect_arch(names),
                    f'{len(names)} name(s)'))
    # An SGL split states its architecture in every metadata shard it has, and
    # the main one carries the model's config.json too - so --assemble is asked
    # the same question the other modes are, and answers with two witnesses.
    for d in getattr(a, 'assemble', None) or []:
        for f in sorted(glob.glob(os.path.join(d, '*' + SPLIT_EXT))):
            hdr, base = ST.header(f)
            m = hdr.get('__metadata__') or {}
            if m.get('kind') != 'metadata':
                continue
            if m.get('arch'):
                out.append((f'{os.path.basename(d)} metadata shard',
                            m['arch'] if m['arch'] in SN.ARCHS else None,
                            repr(m['arch'])))
            k = HF_FILE + 'config.json'
            if k in hdr:
                o = hdr[k]['data_offsets']
                with open(f, 'rb') as fh:
                    fh.seek(base + o[0])
                    cfg = json.loads(fh.read(o[1] - o[0]))
                out.append((f'{os.path.basename(d)} config.json',
                            SN.arch_from_hf_config(cfg),
                            f'{cfg.get("architectures") or cfg.get("model_type")}'))
            break
    return out


def resolve_arch(a, ap):
    """(the arch key, the sentence that says how it was decided)."""
    if a.arch:
        return a.arch, 'given by --arch'
    wit = [(w, k, d) for w, k, d in arch_witnesses(a) if k]
    keys = {k for _w, k, _d in wit}
    if len(keys) == 1:
        key = keys.pop()
        return key, ('detected from ' + ', '.join(w for w, _k, _d in wit))
    if len(keys) > 1:
        ap.error('the sources given do not describe one model: '
                 + '; '.join(f'{w} says {k} ({d})' for w, k, d in wit)
                 + '. Fix the command line, or force one with --arch.')
    ap.error(
        'could not tell which architecture this model is. Nothing given names '
        'one: ' + ('; '.join(f'{w} ({d})' for w, _k, d in arch_witnesses(a))
                   or 'no config.json, no GGUF metadata and no tensor list')
        + '. The registered ones are ' + ', '.join(sorted(SN.ARCHS))
        + '; pass --arch <name> to force one, or add an ARCHS entry to '
          'sglang_native.py if none of them is this model (docs/sglang.md, '
          'Adding a new model).')


# =============================================================================
# MODULES vs TENSORS, AND THE SCALE GROUPS - ALL DERIVED FROM THE ARCH
# =============================================================================
#
# A recipe assigns MODULES; a checkpoint carries TENSORS, and the two are not
# the same name whenever a module is a fused linear or a FusedMoE.
# `sglang_native.DiskNames` owns that map (and the scale groups it implies);
# everything below works in DISK names, because that is what is read and
# written.  Only `quantized_layers` is written back in module names.

def disk_assignment(assign, src_idx, dn):
    """({disk prefix: algo}, [modules the source does not carry]).

    Built by walking the SOURCE, not by generating names, so an expert count or
    a layer count never appears here: whatever tensors the checkpoint has that
    classify onto an assigned module get that module's algo.
    """
    out, seen = {}, set()
    for name in src_idx:
        if not name.endswith('.weight'):
            continue
        pre = name[: -len('.weight')]
        c = dn.classify(pre)
        if c is None:
            continue
        algo = assign.get(c[0])
        if algo is not None:
            out[pre] = algo
            seen.add(c[0])
    return out, sorted(set(assign) - seen)


def check_fusion(disk_assign, dn):
    """Every scale group uniform, and complete.  Returns (groups, problems)."""
    groups = {}
    for p, a in disk_assign.items():
        c = dn.classify(p)
        if c and c[1]:
            groups.setdefault(c[1], {})[p] = a
    bad = []
    for gid, members in sorted(groups.items()):
        algos = set(members.values())
        missing = set(dn.group_members(gid)) - set(members)
        if len(algos) > 1:
            bad.append(f'{gid[0]}.{gid[1]}: shards disagree {sorted(algos)}')
        elif missing and algos:
            # a shard left BF16 while its siblings are quantised is the same fault:
            # the loader collects the algos of ALL shards and BF16 is one of them
            bad.append(f'{gid[0]}.{gid[1]}: {sorted(algos)} but BF16 (absent) for '
                       f'{sorted(x.rsplit(".", 1)[-1] for x in missing)}')
    return groups, bad


def needs_input_scale(pre, algo):
    """Does this module consume a CALIBRATED static activation scale?

    NVFP4 on a linear is W4A4 and FP8 here is W8A8-static, so both do.  NVFP4 on
    a token embedding is a gather served by `ModelOptNvFp4EmbeddingMethod`
    (modelopt_quant.py:700-793), which registers no `input_scale` at all - there
    is no activation to quantise - so it does not.  FP8_PB_WO is
    dynamic-activation and takes none.
    """
    return algo in ('NVFP4', 'FP8') and not pre.endswith('embed_tokens')


# =============================================================================
# RECIPE -> {hf prefix: ALGO}
# =============================================================================

def load_assignment(path, arch, universe=None):
    """Accept either a `.recipe` (GGUF names) or an `hf_quant_config.json`.

    A recipe is the primary form: it is what `quant_assign.py` emits and it
    carries the GGUF names the whole suite speaks.  A config is accepted so a
    hand-edited or published config can be rebuilt verbatim.

    A recipe is regex - the published form folds each tensor family into one
    alternation (`quants_regex_merger`).  `universe` is the model's tensor set
    (the GGUF names of its tensors.bf16.map); given it, `parse_recipe` expands
    every pattern over it exactly as the GGUF build path does, so a merged recipe
    yields the identical assignment as its per-tensor original.  Without it a raw
    recipe still loads and a compacted one errors instead of silently collapsing.
    """
    if path.endswith('.json'):
        c = json.load(open(path, encoding='utf-8'))
        q = c.get('quantization', c)
        out = {}
        for k, v in q['quantized_layers'].items():
            a = v['quant_algo'].upper()
            if a not in ALGOS:
                raise SystemExit(f'{path}: unknown quant_algo {a!r} for {k}')
            out[k] = a
        return out, q
    rec = SN.parse_recipe(path, universe)
    # A tensor the architecture pins to BF16 (sglang_native.PIN_BF16_ADVISORY:
    # the model code reads it as BF16) is refused at any other type here, as
    # the preset refuses to assign it and --split refuses to write it: a
    # hand-written recipe must not be the one door to an unloadable checkpoint.
    pinned = [(n, q) for n, q in rec.items()
              if str(q).lower() != 'sgl_bf16' and SN.pinned_bf16(n, arch)]
    if pinned:
        n0, q0 = pinned[0]
        raise SystemExit(
            f'{path}: {len(pinned)} tensor(s) the architecture pins to BF16 are '
            f'assigned another type, e.g. {n0}={q0}: {SN.pinned_bf16(n0, arch)}. '
            f'Give them sgl_bf16 (the preset does) or drop them from the recipe.')
    layers, problems = SN.recipe_to_quantized_layers(rec, arch)
    if problems:
        raise SystemExit('recipe -> quantized_layers failed:\n  '
                         + '\n  '.join(problems))
    out = {k: v['quant_algo'].upper() for k, v in layers.items()}
    return out, None


# =============================================================================
# THE OUTPUT PLAN
# =============================================================================

def plan_outputs(src_idx, assign, has_input_scale):
    """[(name, dtype, shape, nbytes)] in source order, plus {prefix: algo}.

    Every entry a `create_weights()` registers and nothing more.  An extra
    tensor is as fatal as a missing one: the loader iterates the checkpoint and
    a name it does not expect is an error, not spare data.
    """
    plan, owner = [], {}

    def add(name, dt, shape):
        nb = int(np.prod(shape)) * ST.ITEMSIZE[dt] if shape else ST.ITEMSIZE[dt]
        plan.append((name, dt, tuple(shape), nb))

    for name in src_idx:
        _p, e, _b = src_idx[name]
        dt, shape = e['dtype'], tuple(e['shape'])
        if not name.endswith('.weight'):
            add(name, dt, shape)
            continue
        pre = name[: -len('.weight')]
        algo = assign.get(pre)
        if algo is None or algo == 'BF16':
            add(name, dt, shape)
            continue
        n, k = shape
        owner[pre] = algo
        if algo in ('NVFP4', 'W4A16_NVFP4'):
            add(name, 'U8', (n, k // 2))
            add(pre + '.weight_scale', 'F8_E4M3', (n, k // 16))
            add(pre + '.weight_scale_2', 'F32', ())
            # THE EMBEDDING EXCEPTION.  NVFP4 on a linear is W4A4 and registers
            # an input_scale; NVFP4 on the token embedding is served by
            # ModelOptNvFp4EmbeddingMethod, which is GATHER-ONLY
            # (modelopt_quant.py:700-780) and registers weight, weight_scale and
            # weight_scale_2 and NOTHING ELSE - a gather has no activation to
            # quantise.  An input_scale there is an UNEXPECTED tensor.
            if algo == 'NVFP4' and has_input_scale(pre):
                add(pre + '.input_scale', 'F32', ())
        elif algo == 'FP8':
            add(name, 'F8_E4M3', (n, k))
            add(pre + '.weight_scale', 'F32', ())
            if has_input_scale(pre):
                add(pre + '.input_scale', 'F32', ())
        elif algo == 'FP8_PB_WO':
            add(name, 'F8_E4M3', (n, k))
            add(pre + '.weight_scale_inv', 'F32',
                ((n + 127) // 128, (k + 127) // 128))
        else:
            raise SystemExit(f'{pre}: writing {algo} is not implemented')
    return plan, owner


def shard_split(plan, max_bytes):
    """Group the plan into shards, never splitting one module's tensors.

    A module's weight and its scales must land in the same file: the loader
    resolves them together and a split is a load-time surprise.
    """
    shards, cur, cur_b, unit, unit_b = [], [], 0, [], 0
    for item in plan:
        base = item[0].rsplit('.', 1)[0]
        if unit and unit[0][0].rsplit('.', 1)[0] != base:
            if cur_b + unit_b > max_bytes and cur:
                shards.append(cur)
                cur, cur_b = [], 0
            cur += unit
            cur_b += unit_b
            unit, unit_b = [], 0
        unit.append(item)
        unit_b += item[3]
    if unit:
        if cur_b + unit_b > max_bytes and cur:
            shards.append(cur)
            cur, cur_b = [], 0
        cur += unit
    if cur:
        shards.append(cur)
    return shards


# =============================================================================
# THE THREE NUMBERS A MODULE'S BYTES DEPEND ON, AND NOTHING ELSE
# =============================================================================
#
# A quantised tensor's bytes are decided by the algo, the weight amax of its
# FUSED GROUP and the calibrated activation amax of that same group.  All three
# are computed here, once, and `--build` and `--split` both call these - which
# is what makes the claim below structural rather than lucky:
#
#   THE BYTES A UNIFORM SPLIT WRITES FOR A MODULE ARE THE BYTES `--build`
#   WRITES FOR IT UNDER ANY RECIPE THAT SELECTS THE SAME ALGO.
#
# It holds because none of the three depends on the recipe.  `dn.group_members`
# returns EVERY shard of a fused group, not the assigned ones, so both maxima
# are taken over the same set whichever tool asks; and a recipe cannot assign
# part of a group anyway - `check_fusion` refuses that, because SGLang raises on
# it.  Break that and a downloaded checkpoint stops matching a built one, which
# is why the three live in one place.

def harvest_act_amax(path, groups, dn, Q, log_fn=log):
    """{disk prefix: calibrated activation amax}, one value per fused group.

    The reference states `input_scale`, which is the amax divided by the
    format's own maximum; the amax is what is recipe-independent, so it is what
    is carried and the division is redone at write time for the format actually
    chosen.  A fused GEMM sees ONE input, so every shard of a group takes the
    group's maximum - q/k/v of a QKV linear, gate/up of a merged one, and
    gate/up of ONE expert inside a FusedMoE.
    """
    act_amax = {}
    if not path:
        return act_amax
    ri = ST.index_dir(path, 'model-*.safetensors')
    for k in ri:
        if k.endswith('.input_scale'):
            pre = k[: -len('.input_scale')]
            v = float(ST.read(ri, k).reshape(-1)[0])
            # the source's own convention for that module: NVFP4 iff it has a ws2
            nv = (pre + '.weight_scale_2') in ri
            act_amax[pre] = v * (Q.NVFP4_WS2_DENOM if nv else Q.E4M3_MAX)
    log_fn(f'  harvested {len(act_amax)} calibrated activation amax from '
           f'{os.path.basename(path.rstrip("/"))}')
    for gid in groups:
        names = dn.group_members(gid)
        vals = [act_amax[n] for n in names if n in act_amax]
        if vals:
            m = max(vals)
            for n in names:
                if n in act_amax:
                    act_amax[n] = m
    return act_amax


def weight_amax(cache_path, assign, src_idx, Q, log_fn=log):
    """{tensor name: |w|max} for every NVFP4 tensor, cached across recipes.

    Recipe-independent by construction - it is a property of the BF16 source -
    so one cache serves a whole family and a split and a build share it.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding='utf-8'))
    need = [p for p, al in assign.items()
            if al in ('NVFP4', 'W4A16_NVFP4') and p + '.weight' not in cache]
    if need:
        log_fn(f'  computing weight amax for {len(need)} tensors '
               f'({len(cache)} already cached)')
        for i, p in enumerate(sorted(need)):
            w = Q.as_f32_tensor(src_read(src_idx, p + '.weight'))
            cache[p + '.weight'] = float(w.abs().max())
            del w
            if (i + 1) % 50 == 0:
                log_fn(f'    amax {i+1}/{len(need)}')
        if cache_path:
            json.dump(cache, open(cache_path, 'w', encoding='utf-8'), indent=0)
    return cache


def ws2_table(assign, cache, dn, Q):
    """{disk prefix: weight_scale_2} - one value per fused group.

    THE FUSED-SHARD ws2 RULE of the header, applied.  `ModelOptFp4LinearMethod`
    takes the max over the shards and dequantises the whole GEMM with it, so a
    shard written with its own smaller ws2 comes out scaled up by the ratio.
    """
    ws2 = {}
    for p, al in assign.items():
        if al not in ('NVFP4', 'W4A16_NVFP4'):
            continue
        gid = dn.classify(p)[1]
        if gid:
            am = max(cache[n + '.weight'] for n in dn.group_members(gid)
                     if n + '.weight' in cache)
        else:
            am = cache[p + '.weight']
        ws2[p] = Q.nvfp4_weight_scale_2(am)
    return ws2


def quantise(pre, algo, src32, want, ws2, act_amax, Q):
    """(the weight array, {companion tensor: array}) for one module.

    `want` is the set of names the plan asked for, which is how the embedding
    exception arrives here: no `input_scale` was planned for a gather, so none
    is produced.
    """
    out = {}
    if algo in ('NVFP4', 'W4A16_NVFP4'):
        s2 = ws2[pre]
        arr, codes = Q.nvfp4_quantize(src32, s2)
        out[pre + '.weight_scale'] = codes
        out[pre + '.weight_scale_2'] = np.float32(s2)
        if (pre + '.input_scale') in want:
            out[pre + '.input_scale'] = np.float32(
                np.float64(act_amax[pre]) / Q.NVFP4_WS2_DENOM)
    elif algo == 'FP8':
        arr, s = Q.fp8_static_quantize(src32)
        out[pre + '.weight_scale'] = np.float32(s)
        if (pre + '.input_scale') in want:
            out[pre + '.input_scale'] = np.float32(
                np.float64(act_amax[pre]) / Q.E4M3_MAX)
    elif algo == 'FP8_PB_WO':
        arr, s = Q.fp8_block_quantize(src32)
        out[pre + '.weight_scale_inv'] = s
    else:
        raise SystemExit(f'{pre}: writing {algo} is not implemented')
    return arr, out


def uncalibrated(assign, groups, act_amax, dn):
    """The modules that need a calibrated activation scale and have none.

    Taken as a CLOSURE over the fused groups: a group shares one input_scale, so
    one uncalibrated shard makes the whole group uncalibrated, and dropping only
    that shard would leave a mixed fused module - the very fault check_fusion
    exists to catch.
    """
    bare = {p for p, al in assign.items()
            if needs_input_scale(p, al) and p not in act_amax}
    for _gid, members in groups.items():
        if bare & set(members):
            bare |= set(members)
    return sorted({dn.classify(p)[0] for p in bare}), bare


# =============================================================================
# BUILD
# =============================================================================

def build(a, arch):
    import sglang_codecs as Q
    t0 = time.time()
    nt_dir = nontensor_dir(a)          # refuses first, before 851 shards are read
    src_idx = open_source(a)
    src_kind = 'gguf-split' if hasattr(src_idx, 'describe') else 'hf-snapshot'
    nt_generated, nt_problems = [], []
    if src_kind == 'gguf-split':
        _present, nt_generated, nt_problems = GG.hf_files(nt_dir, src_idx, log)
        cfg0 = json.load(open(os.path.join(nt_dir, 'config.json'), encoding='utf-8')) \
            if os.path.exists(os.path.join(nt_dir, 'config.json')) else {}
        # THE ARCHITECTURE DECIDES WHAT A COMPANION IS, not the config alone:
        # GLM-4.7 declares num_nextn_predict_layers 1 and keeps that head in the
        # main stack, so there is no `mtp-` split to ask for and asking would
        # refuse a source that is complete.  `expected_parts` intersects the
        # config with the parts this arch's table splits out, so every name in
        # `gap` has a real directory prefix to print.
        want = GG.expected_parts(cfg0, src_idx.arch_key)
        gap = sorted(want - set(src_idx.parts))
        if gap:
            pre = GG.dir_prefixes(src_idx.arch_key)
            nt_problems.append(
                'config.json describes a model with a ' + ' and a '.join(gap)
                + ' and no split here carries it. The suite converts those into '
                  'their own companion sets (docs/Convert model to BF16.md), '
                  'named '
                + ', '.join(f'{pre[g]}<split> for the {g}' for g in gap)
                + ' beside the main one. Pass --gguf-companion <dir> for each, '
                  'or the checkpoint would be missing them and fail on the GPU '
                  'rather than here.')
    if nt_problems:
        raise SystemExit('--source / --hf-files do not describe one model:\n  '
                         + '\n  '.join(nt_problems))
    universe, uni_why = recipe_universe(a, src_idx)
    log(f'source {SN.publishable_path(a.source)} ({src_kind}, '
        f'{len(src_idx)} tensors)')
    if universe is not None:
        log(f'  recipe tensor set: {len(universe)} name(s) from {uni_why}')
    modules, _q = load_assignment(a.recipe, arch, universe)
    dn = SN.DiskNames(arch)
    log(f'recipe {a.recipe}')
    log(f'  {len(modules)} quantized_layers  '
        + str({k: sum(1 for v in modules.values() if v == k)
               for k in sorted(set(modules.values()))}))

    assign, missing = disk_assignment(modules, src_idx, dn)
    if missing:
        raise SystemExit(f'{len(missing)} recipe modules absent from the source, '
                         f'e.g. {missing[:5]}')
    log(f'  {len(assign)} source tensors carry them '
        f'(a module is one tensor per shard and per expert on disk)')
    nonbf = [p for p in assign if src_idx[p + '.weight'][1]['dtype'] != 'BF16']
    if nonbf:
        raise SystemExit(f'source is not BF16 for: {nonbf[:5]}')

    groups, bad = check_fusion(assign, dn)
    if bad:
        raise SystemExit('FUSED-MODULE VIOLATION - SGLang would raise '
                         '"Mixed quant_algo within fused layer":\n  ' + '\n  '.join(bad))
    log(f'  scale groups checked: {len(groups)} groups, 0 violations')

    act_amax = harvest_act_amax(a.input_scales_from, groups, dn, Q)

    # ---- modules the reference cannot calibrate ---------------------------- #
    # The activation amax is a property of the CALIBRATION DATA and cannot be
    # computed from weights.  A reference checkpoint that does not carry a module
    # - an MTP head RadixArk drops, an lm_head Salyut1 left BF16 - gives
    # us nothing to write, and SGLang would then read whatever `create_weights`
    # left in `input_scale`.  Two honest answers, and no third: refuse, or leave
    # the module BF16 (absence IS the encoding) and say which ones and what it
    # cost.  Never a fabricated 1.0.
    uncal, _bare = uncalibrated(assign, groups, act_amax, dn)
    if uncal:
        if a.uncalibrated == 'refuse':
            _none = not a.input_scales_from
            raise SystemExit(
                f'{len(uncal)} module(s) have no calibrated activation scale'
                + (' and --input-scales-from was not given at all'
                   if _none else ' in --input-scales-from')
                + f'; NVFP4 is W4A4 and FP8 is W8A8-static and neither can be '
                  f'written without one. e.g. {uncal[:4]}.\n'
                  f'  THREE WAYS OUT, in order of preference:\n'
                  f'    --input-scales-from <a published NVFP4/FP8 checkpoint of '
                  f'THIS model>   harvest its calibrated scales (an activation '
                  f'amax cannot be computed from weights, so it must come from '
                  f'somewhere that measured it);\n'
                  f'    --uncalibrated bf16                                      '
                  f'         leave exactly those modules unquantised - absence IS '
                  f'the encoding - and have them named in the log and BUILD.json;\n'
                  f'    build a recipe that needs no activation scales at all, '
                  f'with `quant_assign.py ... --gpu-quants sgl_fp8_pb_wo '
                  f'sgl_bf16`: FP8_PB_WO is dynamic-activation, so nothing in '
                  f'that pool asks for a calibrated input_scale.')
        drop = set(uncal)
        dropped = {p: al for p, al in assign.items() if dn.classify(p)[0] in drop}
        for m in drop:
            modules.pop(m, None)
        assign = {p: al for p, al in assign.items() if p not in dropped}
        groups, bad = check_fusion(assign, dn)
        if bad:
            raise SystemExit('dropping the uncalibrated modules would leave a '
                             'mixed fused module:\n  ' + '\n  '.join(bad))
        log(f'  --uncalibrated bf16: {len(uncal)} module(s) / {len(dropped)} tensor(s) '
            f'left BF16 for want of a calibrated activation scale')
        for m in uncal[:12]:
            log(f'      uncalibrated -> BF16: {m}')
        if len(uncal) > 12:
            log(f'      ... and {len(uncal) - 12} more (all in BUILD.json)')

    def has_is(pre):
        return pre in act_amax

    plan, _owner = plan_outputs(src_idx, assign, has_is)
    total = sum(x[3] for x in plan)
    non_lm = dn.mtp_prefixes(src_idx) + SN.NON_LM_PREFIXES
    lm = sum(x[3] for x in plan if not x[0].startswith(non_lm))
    shards = shard_split(plan, a.shard_bytes)
    log(f'  output {len(plan)} tensors, {total:,} B ({total/2**30:.3f} GiB) '
        f'in {len(shards)} shards')
    log(f'  language-model bytes (excl. {", ".join(non_lm)}): {lm:,}')
    if a.dry_run:
        return 0

    # -- weight amax, for the fused NVFP4 ws2 -------------------------------- #
    cache = weight_amax(a.amax_cache, assign, src_idx, Q)
    ws2 = ws2_table(assign, cache, dn, Q)
    nshared = sum(1 for p in ws2 if dn.classify(p)[1])
    log(f'  weight_scale_2: {len(ws2)} tensors, {nshared} of them sharing a '
        f'fused-group value')

    # -- write ---------------------------------------------------------------#
    os.makedirs(a.out, exist_ok=True)
    weight_map, sha, sizes = {}, {}, {}
    nsh, done_el = len(shards), 0
    for si, items in enumerate(shards, 1):
        fn = f'model-{si:05d}-of-{nsh:05d}.safetensors'
        path = os.path.join(a.out, fn)
        w = ST.ShardWriter(path, items, {'format': 'pt', 'producer': a.name})
        # The header is part of the file's identity, so it goes into the hash -
        # and it is still in the writer's own buffer at this point, so it has to
        # be flushed before it can be read back.  A small header would otherwise
        # be hashed as nothing at all, silently.
        w.f.flush()
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            h.update(f.read(w.base))
        pending, names = {}, {i[0] for i in items}
        for name, _dt, _shape, nb in items:
            weight_map[name] = fn
            if name in pending:
                arr = pending.pop(name)
            else:
                pre = name[: -len('.weight')] if name.endswith('.weight') else None
                algo = assign.get(pre) if pre else None
                if algo is None or algo == 'BF16':
                    arr = src_read(src_idx, name)
                else:
                    src = Q.as_f32_tensor(src_read(src_idx, name))
                    done_el += src.numel()
                    arr, more = quantise(pre, algo, src, names, ws2, act_amax, Q)
                    pending.update(more)
                    del src
            b = np.ascontiguousarray(arr).tobytes()
            if len(b) != nb:
                raise SystemExit(f'{name}: produced {len(b)} B, planned {nb}')
            w.f.write(b)
            w.i += 1
            h.update(b)
            del arr, b
        if pending:
            raise SystemExit(f'{fn}: unwritten {sorted(pending)}')
        w.f.close()
        got = os.path.getsize(path)
        if got != w.total:
            raise SystemExit(f'{path}: {got} B on disk, planned {w.total}')
        sha[fn], sizes[fn] = h.hexdigest(), got
        log(f'  {fn}  {got/2**30:7.3f} GiB  {len(items):4d} tensors  '
            f'sha256 {sha[fn][:16]}...  ({si}/{nsh}, {done_el/1e9:.2f} G elements)')

    # -- config.json / hf_quant_config.json / index --------------------------#
    # keyed on MODULES, which is what `_resolve_quant_algo` looks up - the disk
    # names above are what the loader's weight_loader consumes, and the two are
    # only the same string on an arch whose modules are not fused
    quantized_layers = {}
    for p, al in sorted(modules.items()):
        e = {'quant_algo': al}
        if al in ('NVFP4', 'W4A16_NVFP4'):
            e['group_size'] = 16
        quantized_layers[p] = e
    qcfg = {
        'quant_method': 'modelopt_mixed',
        'quant_algo': 'MIXED_PRECISION',
        'kv_cache_scheme': {'dynamic': False, 'num_bits': 8, 'type': 'float'},
        'kv_cache_quant_algo': a.kv_cache_quant_algo,
        'ignore': list(a.exclude), 'exclude_modules': list(a.exclude),
        'quantized_layers': quantized_layers,
        # PROVENANCE, WITHOUT A RETURN ADDRESS.  This block ships inside the
        # checkpoint, so it names WHAT was read and never WHERE it sat on the
        # machine that read it - sglang_native.py, section 0, states the rule.
        'producer': {'name': 'GGUF-Tool-Suite/sglang_write.py', 'recipe': a.name,
                     'source': SN.publishable_path(a.source),
                     'source_kind': src_kind,
                     'non_tensor_files': SN.publishable_path(nt_dir),
                     'input_scales_from':
                         SN.publishable_path(a.input_scales_from)},
    }
    cfg = json.load(open(os.path.join(nt_dir, 'config.json'), encoding='utf-8'))
    cfg.setdefault('dtype', 'bfloat16')
    cfg['quantization_config'] = qcfg
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w', encoding='utf-8'),
              indent=2)
    json.dump({'producer': qcfg['producer'], 'quantization': qcfg},
              open(os.path.join(a.out, 'hf_quant_config.json'), 'w', encoding='utf-8'),
              indent=2)
    json.dump({'metadata': {'total_size': total}, 'weight_map': weight_map},
              open(os.path.join(a.out, 'model.safetensors.index.json'), 'w',
                   encoding='utf-8'), indent=2)
    copied = []
    for f in COPY_FILES:
        sp = os.path.join(nt_dir, f)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(a.out, f))
            copied.append(f)
    for f, text in nt_generated:
        with open(os.path.join(a.out, f), 'w', encoding='utf-8') as fh:
            fh.write(text)
        copied.append(f + ' (from the GGUF metadata)')
    log(f'  non-tensor files from {SN.publishable_path(nt_dir)}: config.json, '
        + ', '.join(copied))
    meta = {'name': a.name, 'source': SN.publishable_path(a.source),
            'source_kind': src_kind,
            'source_detail': src_idx.describe() if hasattr(src_idx, 'describe')
                             else {'kind': 'hf-snapshot'},
            'non_tensor_files': SN.publishable_path(nt_dir),
            'recipe': SN.publishable_path(a.recipe),
            'recipe_tensor_set': uni_why,
            'input_scales_from': SN.publishable_path(a.input_scales_from),
            'tensors': len(plan),
            'total_bytes': total, 'language_model_bytes': lm,
            'language_model_excludes': list(non_lm),
            'shards': [{'file': f, 'bytes': sizes[f], 'sha256': sha[f]}
                       for f in sorted(sizes)],
            'modules': len(quantized_layers),
            'module_algo_counts': {k: sum(1 for v in modules.values() if v == k)
                                   for k in sorted(set(modules.values()))},
            'algo_counts': {k: sum(1 for v in assign.values() if v == k)
                            for k in sorted(set(assign.values()))},
            'uncalibrated_bf16': uncal,
            'wall_seconds': round(time.time() - t0, 1)}
    # WHAT THE SOURCE COULD NOT GIVE BACK EXACTLY, if anything.  Only a GGUF
    # split can have such elements, and only in a zero-centred norm: the
    # conversion stores gamma + 1 in float32, whose ULP at 1.0 is larger than a
    # tiny gamma's own, so several bf16 gammas produce the same stored sum.  It
    # is reported here rather than left to somebody's byte comparison.
    if hasattr(src_idx, 'inexact'):
        n_t, n_e, n_of, spread, mag = src_idx.inexact()
        meta['source_detail'] = src_idx.describe()
        if n_e:
            log(f'  NOTE: {n_e} element(s) of {n_of:,} in {n_t} zero-centred '
                f'norm(s) had more than one BF16 that reproduces the value the '
                f'GGUF stores (gamma + 1 in float32). The nearest was taken; the '
                f'original can differ by at most {spread:.3e}, and every such '
                f'element is under {mag:.3e} in magnitude. Nothing else in this '
                f'checkpoint is affected - see sglang_gguf._undo_plus_one.')
        else:
            log('  every tensor recovered from the split bit for bit')
    json.dump(meta, open(os.path.join(a.out, 'BUILD.json'), 'w', encoding='utf-8'),
              indent=2)
    log(f'[done] {a.out}  {total:,} B  in {meta["wall_seconds"]:.0f} s')
    return 0


# =============================================================================
# THE PER-TENSOR SPLIT - HOW AN SGLANG CHECKPOINT SHIPS
# =============================================================================
#
# The design is stated in this file's header, under HOW A CHECKPOINT SHIPS.
# What follows is that design and nothing else: the unit is the GGUF TENSOR,
# because that is what a recipe names; each file carries every HF tensor that
# module needs at its format; shard 00001 is the metadata shard and in the main
# split it carries the checkpoint's small text files; and the bytes are written
# by the same quantiser the recipe path uses, from the same two maxima, so a
# downloaded checkpoint and a built one are the same file.

SPLIT_EXT = '.safetensors'
HF_FILE = '__hf_file__/'
# the companions a module can carry, in the order `plan_outputs` plans them -
# which is what makes an assembled shard identical to a built one
COMPANIONS = ('.weight_scale', '.weight_scale_2', '.input_scale',
              '.weight_scale_inv')
_SPLIT_DIR_RE = re.compile(r'^(?P<model>.+)-(?P<maintainer>[^-]+)-'
                           r'(?P<qtype>[^-]+)-SPECIAL_SPLIT$')
_GGUF_SHARD_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$')


def split_naming(source_dir, qtype):
    """(model, maintainer, output directory name) for a split at `qtype`."""
    base = os.path.basename(os.path.abspath(source_dir).rstrip('/'))
    m = _SPLIT_DIR_RE.match(base)
    if not m:
        raise SystemExit(
            f'{base}: a source split directory must be named '
            f'<MODEL>-<MAINTAINER>-<QTYPE>-SPECIAL_SPLIT, which is what '
            f'quant_downloader.sh fetches into and what the output repository '
            f'name is derived from.')
    model, maint = m.group('model'), m.group('maintainer')
    return model, maint, f'{model}-{maint}-{qtype.upper()}-SPECIAL_SPLIT'


def gguf_chunk(split, gguf_name):
    """(chunk id, chunk total) of a GGUF tensor, from the shard it sits in.

    The SGL split reuses these numbers rather than counting its own, so one
    tensor has ONE id across every repository of the model - which is what
    `quant_downloader.sh` assumes when it fetches tensor by tensor from
    several qtypes at once, and what keeps `download.conf`'s CHUNKS_TOTAL true
    for the SGL repositories as well.
    """
    m = _GGUF_SHARD_RE.search(os.path.basename(split.entries()[gguf_name][0]))
    if not m:
        raise SystemExit(f'{gguf_name}: cannot read a chunk id out of '
                         f'{os.path.basename(split.entries()[gguf_name][0])}')
    return int(m.group(1)), int(m.group(2))


def uniform_algos(split, qtype, arch):
    """{gguf name: (effective qtype, predicted bytes or None, why)}.

    `sglang_native.plan_tensor` decides - the same function `quant_assign.py`
    costs a recipe with and `convert_map_qtype.py` writes tensors.<qtype>.map
    with - so a uniform split and the map the assigner planned against agree on
    every tensor's type and size by construction, not by coincidence.

    A GGUF name the arch does not map is BF16 with no prediction: the vision
    tower is converted by its own path and named `v.blk.*`, which
    `sglang_native` deliberately does not model, and no published checkpoint
    calibrates it.

    AND THE PINS OVERRULE IT, for the tensors `PIN_BF16_ADVISORY` names: the
    model code reads those in BF16 whatever the file says (qwen3_5's in_proj_a
    / in_proj_b, glm4_moe's router and its bias), so a repository that held
    them at `qtype` would carry files no server can load.  A recipe cannot be
    trusted to avoid them - `sglang_preset.py` pins them, but this repository
    is also a pool a hand-written recipe draws single tensors out of - so the
    REPOSITORY is where the pin has to hold.  The size model is deliberately
    not changed to match: it prices what it is asked about, which is what makes
    it reproduce a published checkpoint exactly, and it is not consulted for a
    pinned tensor anyway because the preset assigns that tensor `sgl_bf16` and
    the assigner then prices it off tensors.sgl_bf16.map.  The one visible
    consequence is that the map written beside this repository says `sgl_bf16`
    on those rows where tensors.<qtype>.map says `<qtype>`; the map states the
    type each file actually holds, exactly as it does for `--uncalibrated
    bf16`, so a recipe that asks for `<qtype>` there still resolves - to a file
    the model can read.
    """
    ent = split.entries()
    fused = SN.fused_groups(list(ent), arch)
    shard_sizes = {}
    for members in fused.values():
        sizes = [ent[m][1][1] for m in members if len(ent[m][1]) > 1]
        for m in members:
            shard_sizes[m] = sizes
    out = {}
    for name, (_p, ne, _tt) in ent.items():
        elems = 1
        for d in ne:
            elems *= int(d)
        if SN.map_tensor(name, arch) is None:
            out[name] = (SN.BF16, None,
                         'not a tensor the arch\'s name map models')
            continue
        pin = SN.pinned_bf16(name, arch) if qtype != SN.BF16 else None
        algo = SN.BF16 if pin else qtype
        eff, nbytes, why = SN.plan_tensor(name, list(ne), elems, algo, arch,
                                          shard_sizes.get(name))
        out[name] = (eff, nbytes, f'pinned BF16 - {pin}' if pin else why)
    return out


def _shape_text(ne):
    return '(' + ', '.join(str(int(d)) for d in ne) + (',)' if len(ne) == 1 else ')')


def split(a, arch):
    """Write one whole-model SGL_<QTYPE> split per part, with its tensors.map."""
    import sglang_codecs as Q
    t0 = time.time()
    if not GG.looks_like_split(a.source):
        raise SystemExit(
            '--split reads the BF16 GGUF SPECIAL_SPLIT, not an HF snapshot: the '
            'per-tensor layout is keyed on GGUF TENSOR NAMES and numbered with '
            'the model\'s own GGUF chunk ids, and a snapshot carries neither. '
            'Point --source at the BF16 split (its companions are picked up '
            'beside it) and --hf-files at the snapshot for the small text files.')
    qtype = a.qtype.lower()
    if qtype not in SN.SGLANG_ALGOS:
        raise SystemExit(f'--qtype {a.qtype}: not an SGLang qtype. The '
                         f'registered ones are '
                         f'{", ".join(sorted(SN.SGLANG_ALGOS))}.')
    nt_dir = nontensor_dir(a)
    src_idx = open_source(a)
    present, nt_generated, nt_problems = GG.hf_files(nt_dir, src_idx, log)
    if nt_problems:
        raise SystemExit('--source / --hf-files do not describe one model:\n  '
                         + '\n  '.join(nt_problems))
    dn = SN.DiskNames(arch)
    units = src_idx.gguf_groups()
    splits = []
    for sp, _g, _h in units:
        if sp not in splits:
            splits.append(sp)
    part_of = {sp.dir: part for part, sp in src_idx.parts.items()}
    log(f'source {SN.publishable_path(a.source)} ({len(src_idx)} tensors in '
        f'{len(units)} GGUF tensor group(s) over {len(splits)} split(s))')

    # -- what each GGUF tensor becomes at this qtype ------------------------ #
    algo_of = {}
    for sp in splits:
        for n, v in uniform_algos(sp, qtype, arch).items():
            algo_of[(sp.dir, n)] = v

    # -- and which HF module prefixes that quantises ------------------------ #
    # Built by walking the source, exactly as `disk_assignment` does for a
    # recipe: an HF tensor the arch's DiskNames cannot classify is a tensor no
    # `quantized_layers` entry could name, so it stays BF16 and the map says so.
    assign, unnamed = {}, set()
    for sp, g, hfs in units:
        qa = SN.SGLANG_ALGOS[algo_of[(sp.dir, g)][0]].get('quant_algo')
        if not qa:
            continue
        pres = [h[: -len('.weight')] for h in hfs if h.endswith('.weight')]
        named = [p for p in pres if dn.classify(p) is not None]
        if not named:
            unnamed.add((sp.dir, g))
            continue
        if len(named) != len(pres):
            raise SystemExit(
                f'{g}: {len(pres) - len(named)} of its {len(pres)} HF tensors '
                f'have a module name and the rest do not, so half of it would '
                f'be quantised and half not. The arch table disagrees with '
                f'itself about this tensor.')
        for pre in named:
            assign[pre] = qa
    for key in unnamed:
        algo_of[key] = (SN.BF16, None, 'no quantized_layers entry can name it')

    groups, bad = check_fusion(assign, dn)
    if bad:
        raise SystemExit('FUSED-MODULE VIOLATION in a uniform split, which '
                         'means a shape rule split a fused group:\n  '
                         + '\n  '.join(bad))
    act_amax = harvest_act_amax(a.input_scales_from, groups, dn, Q)
    uncal, bare = uncalibrated(assign, groups, act_amax, dn)
    if uncal:
        if a.uncalibrated == 'refuse':
            raise SystemExit(
                f'{len(uncal)} module(s) of this split have no calibrated '
                f'activation scale' + ('' if a.input_scales_from else
                                       ' and --input-scales-from was not given')
                + f', e.g. {uncal[:4]}. A published checkpoint calibrates only '
                  f'what it chose to quantise, so its scales stop where its own '
                  f'choices did. Pass --uncalibrated bf16 to leave exactly '
                  f'those BF16 - the map records the type each file actually '
                  f'holds, so a recipe that asks for {qtype} there still '
                  f'resolves - or --input-scales-from a set that covers them, '
                  f'which is what sglang_calibrate.py writes (497 modules on '
                  f'Qwen3.8-27B, which covers this whole split). Note that '
                  f'PIN_BF16_ADVISORY has already taken the pinned modules out '
                  f'of the quantised set, so a scale set is only short here if '
                  f'it misses something this split really does quantise. '
                  f'RadixArk\'s 401 scales no longer fall short on this model: '
                  f'the 96 modules they leave out are in_proj_a/in_proj_b, '
                  f'which the pin holds at BF16 anyway.')
        for pre in bare:
            assign.pop(pre, None)
        for sp, g, hfs in units:
            if any(h.endswith('.weight') and h[: -len('.weight')] in bare
                   for h in hfs):
                _p, ne, _t = sp.entries()[g]
                elems = 1
                for d in ne:
                    elems *= int(d)
                algo_of[(sp.dir, g)] = SN.plan_tensor(g, list(ne), elems,
                                                      SN.BF16, arch) \
                    if SN.map_tensor(g, arch) else (SN.BF16, None, 'uncalibrated')
        groups, bad = check_fusion(assign, dn)
        if bad:
            raise SystemExit('dropping the uncalibrated modules would leave a '
                             'mixed fused module:\n  ' + '\n  '.join(bad))
        log(f'  --uncalibrated bf16: {len(uncal)} module(s) / {len(bare)} '
            f'tensor(s) left BF16 for want of a calibrated activation scale')

    # -- the files that carry nothing, and what their map line should say --- #
    # Two KINDS of GGUF tensor have no HF tensor of their own: the second half
    # of a temporal patch pair (the first half's file carries the whole
    # assembled Conv3D) and a global the `mtp-` companion repeats so it is
    # loadable on its own (the main split owns it).  On Qwen3.8-27B that is
    # FOUR files per qtype set - `v.patch_embd.weight.1` in `mmproj-`, and
    # `token_embd.weight`, `output_norm.weight`, `output.weight` in `mtp-`.
    # Their files are written EMPTY rather than skipped, because
    # quant_downloader.sh checks that the shard numbering has no hole in
    # it.  The map line then states the truth on both counts: the type the model
    # carries that tensor at - looked up from the split that does own it - and
    # bytes=0, because this repository spends none on it.
    owner = {}
    for sp, g, hfs in units:
        if hfs and g not in owner:
            owner[g] = algo_of[(sp.dir, g)][0]
    n_empty = 0
    for sp, g, hfs in units:
        if hfs:
            continue
        n_empty += 1
        algo_of[(sp.dir, g)] = (owner.get(g, algo_of[(sp.dir, g)][0]), 0,
                                'carried by another split of this model')

    census = {}
    for key, (eff, _b, _w) in algo_of.items():
        census[eff] = census.get(eff, 0) + 1
    log(f'  {qtype}: ' + str({k: census[k] for k in sorted(census)}))
    if n_empty:
        log(f'  {n_empty} file(s) written empty (bytes=0 in the map): a tensor '
            f'another split of this model owns')
    log(f'  {len(assign)} source tensor(s) quantised, {len(units) - len(assign)} '
        f'left as the source has them')
    if a.dry_run:
        return 0
    cache = weight_amax(a.amax_cache, assign, src_idx, Q)
    ws2 = ws2_table(assign, cache, dn, Q)
    log(f'  weight_scale_2: {len(ws2)} tensors, '
        f'{sum(1 for p in ws2 if dn.classify(p)[1])} sharing a fused-group value')

    def has_is(pre):
        return pre in act_amax

    # -- write, one directory per part -------------------------------------- #
    hf_blobs = []
    for nm, pth in sorted(present.items()):
        if pth:
            with open(pth, 'rb') as fh:
                hf_blobs.append((nm, fh.read()))
    for nm, text in nt_generated:
        hf_blobs.append((nm, text.encode('utf-8')))

    os.makedirs(a.out, exist_ok=True)
    report = []
    for sp in splits:
        part = part_of.get(sp.dir, 'text')
        model, maint, dirname = split_naming(sp.dir, qtype)
        outdir = os.path.join(a.out, dirname)
        os.makedirs(outdir, exist_ok=True)
        total = gguf_chunk(sp, sp.universe()[0])[1]

        def fname(cid):
            return (f'{model}-{maint}-{qtype.upper()}-SPECIAL_TENSOR-'
                    f'{cid:05d}-of-{total:05d}{SPLIT_EXT}')

        # shard 00001: the metadata shard, and in the main split the small files
        blobs = hf_blobs if part == 'text' else []
        meta = {'format': 'pt', 'producer': 'GGUF-Tool-Suite/sglang_write.py',
                'kind': 'metadata', 'part': part, 'sgl_qtype': qtype,
                'arch': src_idx.arch_key, 'gguf_arch': str(sp.arch),
                'model': model, 'maintainer': maint, 'chunks_total': str(total),
                'hf_files': ','.join(n for n, _b in blobs),
                'source': SN.publishable_path(sp.dir)}
        first = fname(1)
        plan = [(HF_FILE + n, 'U8', (len(b),), len(b)) for n, b in blobs]
        w = ST.ShardWriter(os.path.join(outdir, first), plan, meta)
        for _n, b in blobs:
            w.f.write(b)
            w.i += 1
        w.f.close()
        n_files, n_bytes, lines = 1, os.path.getsize(os.path.join(outdir, first)), []

        for u_sp, g, hfs in units:
            if u_sp is not sp:
                continue
            cid, _tot = gguf_chunk(sp, g)
            eff, want_bytes, _why = algo_of[(sp.dir, g)]
            fn = fname(cid)
            path = os.path.join(outdir, fn)
            mini = {h: src_idx[h] for h in hfs}
            plan, _own = plan_outputs(mini, assign, has_is)
            payload = sum(x[3] for x in plan)
            if want_bytes is not None and payload != want_bytes:
                raise SystemExit(
                    f'{g}: sglang_native.plan_tensor predicts {want_bytes} B at '
                    f'{eff} and the file holds {payload} B. The assigner\'s byte '
                    f'model and this writer disagree, which would make every '
                    f'recipe size wrong; fix one of them before shipping.')
            names = {i[0] for i in plan}
            w = ST.ShardWriter(path, plan, dict(
                meta, kind='tensor', gguf_tensor=g, sgl_qtype=eff,
                quant_algo=SN.SGLANG_ALGOS[eff].get('quant_algo') or 'BF16',
                hf_files='', hf_tensors=str(len(hfs))))
            w.f.flush()          # the header is hashed too - see build()
            h = hashlib.sha256()
            with open(path, 'rb') as fh:
                h.update(fh.read(w.base))
            pending = {}
            for name, _dt, _shape, nb in plan:
                if name in pending:
                    arr = pending.pop(name)
                else:
                    pre = (name[: -len('.weight')] if name.endswith('.weight')
                           else None)
                    al = assign.get(pre) if pre else None
                    if al is None or al == 'BF16':
                        arr = src_read(src_idx, name)
                    else:
                        s32 = Q.as_f32_tensor(src_read(src_idx, name))
                        arr, more = quantise(pre, al, s32, names, ws2, act_amax, Q)
                        pending.update(more)
                        del s32
                b = np.ascontiguousarray(arr).tobytes()
                if len(b) != nb:
                    raise SystemExit(f'{name}: produced {len(b)} B, planned {nb}')
                w.f.write(b)
                w.i += 1
                h.update(b)
                del arr, b
            if pending:
                raise SystemExit(f'{fn}: unwritten {sorted(pending)}')
            w.f.close()
            got = os.path.getsize(path)
            if got != w.total:
                raise SystemExit(f'{path}: {got} B on disk, planned {w.total}')
            _p, ne, _t = sp.entries()[g]
            elems = 1
            for d in ne:
                elems *= int(d)
            lines.append(f'{fn}:{h.hexdigest()}:{g}:shape={_shape_text(ne)}:'
                         f'dtype={eff}:elements={elems}:bytes={payload}')
            n_files += 1
            n_bytes += got
            if n_files % 100 == 0:
                log(f'  {dirname}: {n_files}/{len(sp.universe()) + 1} files, '
                    f'{n_bytes/2**30:.2f} GiB')
        body = '\n'.join(lines) + '\n'
        for nm in (f'tensors.{qtype}.map', 'tensors.map'):
            with open(os.path.join(outdir, nm), 'w', encoding='utf-8') as fh:
                fh.write(body)
        json.dump({'split': dirname, 'part': part, 'qtype': qtype,
                   'model': model, 'maintainer': maint, 'chunks_total': total,
                   'files': n_files, 'bytes': n_bytes,
                   'source': SN.publishable_path(sp.dir),
                   'non_tensor_files': [n for n, _b in blobs],
                   'arch': src_idx.arch_key,
                   'dtype_counts': {k: sum(1 for l in lines
                                           if f':dtype={k}:' in l)
                                    for k in sorted(census)}},
                  open(os.path.join(outdir, 'SPLIT.json'), 'w',
                       encoding='utf-8'), indent=2)
        log(f'  {dirname}: {n_files} files, {n_bytes:,} B '
            f'({n_bytes/2**30:.3f} GiB), tensors.{qtype}.map written')
        report.append((dirname, n_files, n_bytes))
    log(f'[done] {a.out}  {sum(r[2] for r in report):,} B in '
        f'{sum(r[1] for r in report)} files, {time.time() - t0:.0f} s')
    return 0


# =============================================================================
# ASSEMBLE - the downloaded per-tensor files back into a servable checkpoint
# =============================================================================
#
# The inverse of `--split`, and deliberately not a second writer: it re-plans
# the shards from the tensors that are actually there, in the tensor order the
# metadata shard carries, and copies bytes.  It quantises nothing, so it needs
# no torch and cannot introduce a difference of its own - what it hands back is
# what the split holds.  `quantized_layers` is read off each file's recorded
# algo and CHECKED against the tensors that file carries, so a config that says
# NVFP4 over a file holding BF16 is a refusal rather than a load-time surprise.

def assembly_gaps(cfg, arch_key, parts_present, index_names, have):
    """What the directories given to `--assemble` do not carry.

    THE CHECK THAT IS NOT ONE-SIDED.  That every tensor present is named by the
    index says the splits are all of one model; it says nothing at all about a
    split that was never passed.  A forgotten companion fails in the other
    direction, and it is the mistake a user will actually make, because the
    download is one `quant_downloader.sh` run per part with its own
    download.conf - so leaving one out costs nothing and looks like it worked.
    Both witnesses are here and they are independent:

      * the metadata shard's own model.safetensors.index.json - the SOURCE
        checkpoint's index, carried in verbatim - names every tensor the
        checkpoint must end up with, including the ones whose directory was not
        passed.  A name in it that no file carries is a HOLE.
      * `config.json` plus the parts the shards stamp themselves with says a
        whole part never arrived even when there is no index to consult.

    NEITHER WITNESS ACQUITS ON ITS OWN.  A part is here when a shard is STAMPED
    with it AND the index, where there is one, finds none of its tensors short.
    The index CONVICTS a part that is stamped - naming the tensors that never
    arrived - but it cannot acquit one that is not, because an index that never
    mentions that part at all (a metadata shard whose index lists only the text
    half, which is what a checkpoint converted part by part looks like) would
    otherwise wave a wholly absent companion through in silence - the one case
    this function exists for.

    Returns [(part or None, [missing names])] - `None` for tensors no companion
    of this architecture claims, which is a half-finished download rather than a
    missing directory.  A tensor is attributed to a part by the HF-name prefix
    the arch's inversion table gives it, so nothing here is spelled per model.
    """
    want = GG.expected_parts(cfg, arch_key)
    pre = GG.part_name_prefixes(arch_key)
    absent = [n for n in index_names if n not in have]
    out, taken = [], set()
    for part in sorted(want - {'text'}):
        p = pre.get(part)
        mine = [n for n in absent if p and n.startswith(p)]
        # The index acquits a part it names in full (a draft head converted
        # into the main split is stamped 'text' and lives in the index); a
        # stamp acquits a part the index never mentions.  Neither acquits a
        # part that is both unstamped and absent from the index.
        knows = bool(p) and any(n.startswith(p) for n in index_names)
        if not mine and (part in parts_present or knows):
            continue
        taken.update(mine)
        out.append((part, mine))
    rest = [n for n in absent if n not in taken]
    if rest:
        out.append((None, rest))
    return out


def assemble(a, arch):
    t0 = time.time()
    files, dirs = [], list(a.assemble)
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f'{d}: not a directory')
        got = sorted(glob.glob(os.path.join(d, '*' + SPLIT_EXT)))
        if not got:
            raise SystemExit(f'{d}: no *{SPLIT_EXT} files here. Point --assemble '
                             f'at the directory quant_downloader.sh downloaded '
                             f'into (one per part: the main split and its mtp-/'
                             f'mmproj- companions).')
        files += got
    idx = ST.index(files)
    blobs, algo_at, part_seen, owner, dup = {}, {}, {}, {}, []
    blob_from = {}                     # which split's file carried each of them
    meta_at = {}                       # and what that file stamped itself with
    for f in files:
        hdr, base = ST.header(f)
        meta = hdr.get('__metadata__') or {}
        meta_at[f] = meta
        part = meta.get('part', '?')
        part_seen[part] = part_seen.get(part, 0) + 1
        with open(f, 'rb') as fh:
            for k, v in hdr.items():
                if k != '__metadata__':
                    # ONE FILE PER TENSOR, so a name that arrives twice means two
                    # directories carry the same part - the index would silently
                    # keep whichever was read last, which is not a choice to make
                    # on somebody's behalf.
                    if k in owner:
                        dup.append((k, owner[k], f))
                    owner[k] = f
                if k.startswith(HF_FILE):
                    lo, hi = v['data_offsets']
                    fh.seek(base + lo)
                    blobs[k[len(HF_FILE):]] = fh.read(hi - lo)
                    blob_from[k[len(HF_FILE):]] = f
                elif k != '__metadata__' and k.endswith('.weight'):
                    # what the file says it holds; the tensors it carries are
                    # checked against that below
                    algo_at[k[: -len('.weight')]] = meta.get('quant_algo', 'BF16')
    if dup:
        raise SystemExit(
            f'{len(dup)} tensor(s) are carried by two files, so the directories '
            f'given overlap. e.g. {dup[0][0]} in '
            f'{os.path.basename(dup[0][1])} and {os.path.basename(dup[0][2])}. '
            f'Pass one directory per part.')
    log(f'{len(files)} per-tensor file(s) from {len(dirs)} split(s): '
        + ', '.join(f'{p} x{n}' for p, n in sorted(part_seen.items()))
        + f'; {len(blobs)} non-tensor file(s)')
    if 'config.json' not in blobs:
        raise SystemExit('no config.json in any metadata shard. The MAIN split '
                         'carries the checkpoint\'s small text files; pass its '
                         'directory to --assemble as well as the companions\'.')

    # -- the tensor order, which is what makes an assembled shard a built one -
    rank, order_why = None, 'sorted tensor names'
    if 'model.safetensors.index.json' in blobs:
        rank = GG.hf_index_order(json.loads(blobs['model.safetensors.index.json']))
        order_why = 'model.safetensors.index.json of the metadata shard'
    comp = {n for n in idx if not n.startswith(HF_FILE)
            and any(n.endswith(c) and n[: -len(c)] + '.weight' in idx
                    for c in COMPANIONS)}
    base_names = [n for n in idx if not n.startswith(HF_FILE) and n not in comp]
    if rank is not None:
        missing = [n for n in base_names if n not in rank]
        if missing:
            raise SystemExit(f'{len(missing)} tensor(s) are not in the index the '
                             f'metadata shard carries, e.g. {missing[:3]}; the '
                             f'splits are not all of one model.')
        base_names.sort(key=lambda n: rank[n])
    else:
        base_names.sort()

    # -- and the other direction: a part that never arrived ----------------- #
    ak = getattr(a, '_arch_key', None) or next(
        (k for k, v in SN.ARCHS.items() if v is arch), None)
    gaps = assembly_gaps(json.loads(blobs['config.json']), ak, set(part_seen),
                         list(rank or ()), set(base_names))
    if gaps:
        # THE MODEL NAME IS STAMPED, NOT READ OFF THE PATH.  What
        # models/<name>/download.conf is keyed on is the MODEL; what the
        # directory is called is whatever the person who ran
        # quant_downloader.sh typed, and a refusal that names a path nobody has
        # is worse than no path at all.  So take it from the metadata shard
        # that carried config.json - `--split` stamps every shard it writes
        # with the model it took from the source split's own name.  Only a
        # shard written before that stamp existed falls back to the directory.
        mshard = blob_from['config.json']
        model = (meta_at.get(mshard) or {}).get('model') or os.path.basename(
            os.path.dirname(os.path.abspath(mshard)))
        pre, lines = GG.dir_prefixes(ak), []
        for part, names in gaps:
            head = (f'the {part} part is missing' if part else
                    'these tensors are in no directory given')
            what = (f'{len(names)} tensor(s) the metadata shard\'s own '
                    f'model.safetensors.index.json names are in none of the '
                    f'directories given, e.g. {names[:3]}' if names else
                    'no shard here is stamped with it')
            fix = (f'Download it with GGUF_DOWNLOAD_CONF=models/'
                   f'{pre.get(part, "")}{model}/download.conf and pass that '
                   f'directory to --assemble too.' if part else
                   'Re-run the download for the part that owns them; '
                   'quant_downloader.sh --verify says what is short.')
            lines.append(f'{head}: {what}. {fix}')
        raise SystemExit(
            '--assemble would write an incomplete checkpoint:\n  '
            + '\n  '.join(lines)
            + '\n  A checkpoint with a hole in it loads and then fails on the '
              'GPU, which is exactly where this is meant not to happen.')
    plan = []
    for n in base_names:
        _p, e, _b = idx[n]
        plan.append((n, e['dtype'], tuple(e['shape']),
                     e['data_offsets'][1] - e['data_offsets'][0]))
        if not n.endswith('.weight'):
            continue
        pre = n[: -len('.weight')]
        for c in COMPANIONS:
            if pre + c in idx:
                _p2, e2, _b2 = idx[pre + c]
                plan.append((pre + c, e2['dtype'], tuple(e2['shape']),
                             e2['data_offsets'][1] - e2['data_offsets'][0]))
    total = sum(x[3] for x in plan)
    shards = shard_split(plan, a.shard_bytes)
    log(f'  {len(plan)} tensors, {total:,} B ({total/2**30:.3f} GiB) in '
        f'{len(shards)} shards; order from {order_why}')

    # -- quantized_layers, read off the files and checked against them ------ #
    dn = SN.DiskNames(arch)
    quantized_layers, problems = {}, []
    for pre, al in sorted(algo_at.items()):
        al = (al or 'BF16').upper()
        if al == 'BF16':
            continue
        want = {'NVFP4': ('.weight_scale', '.weight_scale_2'),
                'W4A16_NVFP4': ('.weight_scale', '.weight_scale_2'),
                'FP8': ('.weight_scale',),
                'FP8_PB_WO': ('.weight_scale_inv',)}.get(al)
        if want is None:
            problems.append(f'{pre}: unknown quant_algo {al!r}')
            continue
        for c in want:
            if pre + c not in idx:
                problems.append(f'{pre}: recorded {al} but carries no {c[1:]}')
        c = dn.classify(pre)
        if c is None:
            problems.append(f'{pre}: recorded {al} but no module name covers it')
            continue
        e = {'quant_algo': al}
        if al in ('NVFP4', 'W4A16_NVFP4'):
            e['group_size'] = 16
        prev = quantized_layers.get(c[0])
        if prev and prev['quant_algo'] != al:
            problems.append(f'{c[0]}: shards disagree ({prev["quant_algo"]} vs '
                            f'{al}) - SGLang raises "Mixed quant_algo within '
                            f'fused layer"')
        quantized_layers[c[0]] = e
    if problems:
        raise SystemExit('the downloaded files do not describe a loadable '
                         'checkpoint:\n  ' + '\n  '.join(problems[:20]))
    log(f'  quantized_layers {len(quantized_layers)} '
        + str({k: sum(1 for v in quantized_layers.values()
                      if v['quant_algo'] == k)
               for k in sorted({v['quant_algo']
                                for v in quantized_layers.values()})}))
    if a.dry_run:
        return 0

    os.makedirs(a.out, exist_ok=True)
    weight_map, sha, sizes = {}, {}, {}
    nsh = len(shards)
    for si, items in enumerate(shards, 1):
        fn = f'model-{si:05d}-of-{nsh:05d}.safetensors'
        path = os.path.join(a.out, fn)
        w = ST.ShardWriter(path, items, {'format': 'pt', 'producer': a.name})
        w.f.flush()              # the header is hashed too - see build()
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            h.update(fh.read(w.base))
        for name, _dt, _shape, nb in items:
            weight_map[name] = fn
            b = ST.raw(idx, name)
            if len(b) != nb:
                raise SystemExit(f'{name}: {len(b)} B on disk, planned {nb}')
            w.f.write(b)
            w.i += 1
            h.update(b)
            del b
        w.f.close()
        got = os.path.getsize(path)
        if got != w.total:
            raise SystemExit(f'{path}: {got} B on disk, planned {w.total}')
        sha[fn], sizes[fn] = h.hexdigest(), got
        log(f'  {fn}  {got/2**30:7.3f} GiB  {len(items):4d} tensors  '
            f'sha256 {sha[fn][:16]}...  ({si}/{nsh})')

    qcfg = {
        'quant_method': 'modelopt_mixed',
        'quant_algo': 'MIXED_PRECISION',
        'kv_cache_scheme': {'dynamic': False, 'num_bits': 8, 'type': 'float'},
        'kv_cache_quant_algo': a.kv_cache_quant_algo,
        'ignore': list(a.exclude), 'exclude_modules': list(a.exclude),
        'quantized_layers': quantized_layers,
        'producer': {'name': 'GGUF-Tool-Suite/sglang_write.py --assemble',
                     'recipe': a.name, 'source': 'per-tensor SGL splits',
                     'source_kind': 'sgl-split',
                     'non_tensor_files': 'the metadata shard',
                     'input_scales_from': None},
    }
    cfg = json.loads(blobs['config.json'])
    cfg.setdefault('dtype', 'bfloat16')
    cfg['quantization_config'] = qcfg
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w',
                        encoding='utf-8'), indent=2)
    json.dump({'producer': qcfg['producer'], 'quantization': qcfg},
              open(os.path.join(a.out, 'hf_quant_config.json'), 'w',
                   encoding='utf-8'), indent=2)
    json.dump({'metadata': {'total_size': total}, 'weight_map': weight_map},
              open(os.path.join(a.out, 'model.safetensors.index.json'), 'w',
                   encoding='utf-8'), indent=2)
    copied = []
    for f in COPY_FILES:
        if f in blobs:
            with open(os.path.join(a.out, f), 'wb') as fh:
                fh.write(blobs[f])
            copied.append(f)
    log(f'  non-tensor files from the metadata shard: config.json, '
        + ', '.join(copied))
    json.dump({'name': a.name, 'source': [SN.publishable_path(d) for d in dirs],
               'source_kind': 'sgl-split', 'tensor_order': order_why,
               'tensors': len(plan), 'total_bytes': total,
               'shards': [{'file': f, 'bytes': sizes[f], 'sha256': sha[f]}
                          for f in sorted(sizes)],
               'modules': len(quantized_layers),
               'module_algo_counts':
                   {k: sum(1 for v in quantized_layers.values()
                           if v['quant_algo'] == k)
                    for k in sorted({v['quant_algo']
                                     for v in quantized_layers.values()})},
               'wall_seconds': round(time.time() - t0, 1)},
              open(os.path.join(a.out, 'BUILD.json'), 'w', encoding='utf-8'),
              indent=2)
    log(f'[done] {a.out}  {total:,} B  in {time.time() - t0:.0f} s')
    return 0


# =============================================================================
# VERIFY - the loader's own preconditions, checked before any GPU sees the file
# =============================================================================

def verify(a, arch):
    """1 inventory, 2 per-algo tensor set, 3 shape rules, 4 fused agreement,
    5 index consistency, 6 dequantisation error within the FORMAT's own bound.

    Everything here is a rule taken from the SGLang loader, so a pass means the
    loader's preconditions hold - not that the checkpoint is good, but that it
    will load and be read the way it was written.
    """
    fail = []

    def check(cond, msg):
        if not cond:
            fail.append(msg)
        return cond

    dn = SN.DiskNames(arch)
    cfg = json.load(open(os.path.join(a.ckpt, 'config.json'), encoding='utf-8'))
    qc = cfg['quantization_config']
    ql = qc['quantized_layers']
    idx = ST.index_dir(a.ckpt, 'model-*.safetensors')
    src = open_source(a, log_fn=print)
    print(f'checkpoint {a.ckpt}')
    print(f'  quant_method {qc["quant_method"]!r}  quant_algo {qc["quant_algo"]!r}  '
          f'kv {qc.get("kv_cache_quant_algo")}  quantized_layers {len(ql)}')
    check(qc['quant_method'] == 'modelopt_mixed', 'quant_method must be modelopt_mixed')
    check(qc['quant_algo'] == 'MIXED_PRECISION', 'quant_algo must be MIXED_PRECISION')
    check(len(ql) > 0, 'quantized_layers must be non-empty')

    expected = {}
    for n, (_p, e, _b) in src.items():
        pre = n[: -len('.weight')] if n.endswith('.weight') else None
        # the config names MODULES, the file names TENSORS
        c = dn.classify(pre) if pre else None
        algo = (ql.get(c[0]) or {}).get('quant_algo', '').upper() if c else ''
        if not algo:
            expected[n] = (e['dtype'], tuple(e['shape']))
            continue
        N, K = e['shape']
        if algo in ('NVFP4', 'W4A16_NVFP4'):
            check(K % 16 == 0, f'{pre}: NVFP4 needs K%16==0, K={K}')
            expected[n] = ('U8', (N, K // 2))
            expected[pre + '.weight_scale'] = ('F8_E4M3', (N, K // 16))
            expected[pre + '.weight_scale_2'] = ('F32', ())
            if algo == 'NVFP4' and not pre.endswith('embed_tokens'):
                expected[pre + '.input_scale'] = ('F32', ())
        elif algo == 'FP8':
            expected[n] = ('F8_E4M3', (N, K))
            expected[pre + '.weight_scale'] = ('F32', ())
            expected[pre + '.input_scale'] = ('F32', ())
        elif algo == 'FP8_PB_WO':
            expected[n] = ('F8_E4M3', (N, K))
            expected[pre + '.weight_scale_inv'] = ('F32', ((N + 127) // 128,
                                                           (K + 127) // 128))
            if c[1] is not None:
                check(N % 128 == 0, f'{pre}: FP8_PB_WO fused shard output {N} '
                                    f'not divisible by 128')
        else:
            fail.append(f'{pre}: unsupported algo {algo}')
        # the suite's own legality rule must agree with the file on disk
        ok, why = SN.algo_legal(
            {'NVFP4': 'sgl_nvfp4', 'W4A16_NVFP4': 'sgl_nvfp4a16', 'FP8': 'sgl_fp8',
             'FP8_PB_WO': 'sgl_fp8_pb_wo'}.get(algo, 'sgl_bf16'), K, N,
            'embedding' if pre.endswith('embed_tokens') else 'linear',
            name='output.weight' if pre.endswith('lm_head') else None)
        check(ok, f'{pre}: sglang_native.algo_legal refuses {algo}: {why}')
    got = {n: (e['dtype'], tuple(e['shape'])) for n, (_p, e, _b) in idx.items()}
    miss = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    wrong = sorted(n for n in set(expected) & set(got) if expected[n] != got[n])
    check(not miss, f'{len(miss)} expected tensors missing, e.g. {miss[:4]}')
    check(not extra, f'{len(extra)} unexpected tensors, e.g. {extra[:4]}')
    check(not wrong, f'{len(wrong)} tensors with wrong dtype/shape, e.g. '
                     f'{[(n, expected[n], got[n]) for n in wrong[:3]]}')
    print(f'  tensors: {len(got)} on disk, {len(expected)} expected  '
          f'(missing {len(miss)}, extra {len(extra)}, mismatched {len(wrong)})')

    # the scale groups, over the TENSORS on disk: SGLang collapses each group
    # into one weight_scale_2 and one input_scale, so a group whose members
    # disagree is a silent gross error and is checked here, not on the GPU
    grp = {}
    for n in src:
        if not n.endswith('.weight'):
            continue
        pre = n[: -len('.weight')]
        c = dn.classify(pre)
        if c and c[1] and c[0] in ql:
            grp.setdefault(c[1], []).append(pre)
    nws2 = nis = nalgo = 0
    for gid, members in sorted(grp.items()):
        algos = {ql[dn.classify(m)[0]]['quant_algo'].upper() for m in members}
        check(len(algos) == 1, f'{gid[0]}.{gid[1]}: mixed quant_algo {sorted(algos)}')
        nalgo += 1
        check(set(members) == set(dn.group_members(gid)),
              f'{gid[0]}.{gid[1]}: only {sorted(members)} quantised, the rest '
              f'would be BF16')
        if algos & {'NVFP4', 'W4A16_NVFP4'}:
            v = {float(ST.read(idx, m + '.weight_scale_2').reshape(-1)[0])
                 for m in members}
            check(len(v) == 1, f'{gid[0]}.{gid[1]}: weight_scale_2 differs across '
                               'shards - SGLang takes max() and does NOT requantise')
            nws2 += 1
        if 'NVFP4' in algos:
            v = {float(ST.read(idx, m + '.input_scale').reshape(-1)[0])
                 for m in members if m + '.input_scale' in idx}
            check(len(v) <= 1, f'{gid[0]}.{gid[1]}: input_scale differs across shards')
            nis += 1
    print(f'  scale groups: {nalgo} uniform algo, {nws2} sharing weight_scale_2, '
          f'{nis} sharing input_scale')

    bad_s = [n for n in got if n.endswith(('.weight_scale_2', '.input_scale'))
             and not (float(ST.read(idx, n).reshape(-1)[0]) > 0)]
    check(not bad_s, f'{len(bad_s)} non-positive scalar scales, e.g. {bad_s[:3]}')

    ix = json.load(open(os.path.join(a.ckpt, 'model.safetensors.index.json'),
                        encoding='utf-8'))
    files = sorted({os.path.basename(p) for p, _e, _b in idx.values()})
    check(sorted(set(ix['weight_map'].values())) == files,
          'weight_map files != files on disk')
    check(set(ix['weight_map']) == set(got), 'weight_map names != tensors on disk')
    nb = sum(int(np.prod(s)) * ST.ITEMSIZE[d] if s else ST.ITEMSIZE[d]
             for d, s in got.values())
    check(ix['metadata']['total_size'] == nb,
          f'index total_size {ix["metadata"]["total_size"]} != {nb}')
    ondisk = sum(os.path.getsize(os.path.join(a.ckpt, f)) for f in files)
    print(f'  index: {len(files)} shards, tensor bytes {nb:,}, on disk {ondisk:,} '
          f'(+{ondisk-nb:,} B of headers)')
    non_lm = dn.mtp_prefixes(idx) + SN.NON_LM_PREFIXES
    print(f'  language-model bytes (excl. {", ".join(non_lm)}): '
          f'{ST.lm_bytes(idx, exclude=non_lm):,}')

    if a.sample:
        import sglang_codecs as Q
        import torch                                        # noqa: F401
        print('  dequantisation error vs the BF16 source:')
        # sampled over the TENSORS, since a module can be 160 of them
        cand = sorted({n[: -len('.weight')] for n in src if n.endswith('.weight')
                       and (dn.classify(n[: -len('.weight')]) or (None,))[0] in ql})
        step = max(1, len(cand) // a.sample)
        for pre in cand[::step][:a.sample]:
            algo = ql[dn.classify(pre)[0]]['quant_algo'].upper()
            w = Q.as_f32_tensor(src_read(src, pre + '.weight')).numpy()
            if algo in ('NVFP4', 'W4A16_NVFP4'):
                d = Q.nvfp4_dequantize(
                    ST.read(idx, pre + '.weight'), ST.read(idx, pre + '.weight_scale'),
                    float(ST.read(idx, pre + '.weight_scale_2').reshape(-1)[0]))
            elif algo == 'FP8_PB_WO':
                d = Q.fp8_block_dequantize(ST.read(idx, pre + '.weight'),
                                           ST.read(idx, pre + '.weight_scale_inv'))
            else:
                s = float(ST.read(idx, pre + '.weight_scale').reshape(-1)[0])
                d = Q.require_torch().from_numpy(
                    ST.read(idx, pre + '.weight').copy()).view(
                    Q.require_torch().float8_e4m3fn).float().numpy() * s
            e = np.abs(d - w).astype(np.float64)
            amax = float(np.abs(w).max())
            rel = float(np.sqrt((e ** 2).mean())) / float(
                np.sqrt((w.astype(np.float64) ** 2).mean()))
            print(f'    {algo:12s} {pre:62s} max|err|/amax {float(e.max())/amax:8.5f}  '
                  f'rms rel {rel:8.5f}')
            # FORMAT BOUNDS, NOT TASTE.  E4M3 keeps 3 mantissa bits so its worst
            # relative error is half a ULP = 2^-4 = 6.25 % of the element, hence
            # max|err| <= 0.0625*amax; a little headroom for the scale's own
            # rounding gives 0.07.  E2M1's coarsest step is 6-4 = 2 in units of
            # the block scale = block_amax/6, so half a step is 0.167*amax; with
            # the E4M3 rounding of that block scale, 0.20.  Past these means a
            # packing or scale bug, not quantisation.
            lim = {'NVFP4': 0.20, 'W4A16_NVFP4': 0.20,
                   'FP8_PB_WO': 0.07, 'FP8': 0.07}[algo]
            check(float(e.max()) / amax < lim,
                  f'{pre}: max relative error {float(e.max())/amax:.4f} exceeds '
                  f'{lim} for {algo}')
            del w, d, e

    print(f'\n  {"PASS" if not fail else "FAIL"} - {len(fail)} problem(s)')
    for m in fail:
        print('   *', m)
    return 1 if fail else 0


# =============================================================================
# CROSSCHECK - the strongest correctness evidence obtainable without a GPU
# =============================================================================

def _layout(idx, pre):
    """(the algo a checkpoint's own tensors say it used, the names to compare).

    Read off the FILE, not off a config: a quantised module is identified by the
    scale tensors beside its weight, which is the only description both our
    checkpoint and a published one are guaranteed to agree on - published
    checkpoints come in three different quant-config dialects, and one of them
    (`modelopt`) names no modules at all.
    """
    if pre + '.weight' not in idx:
        return None, []
    if pre + '.weight_scale_2' in idx:
        return 'NVFP4', ['.weight', '.weight_scale', '.weight_scale_2']
    if pre + '.weight_scale_inv' in idx:
        return 'FP8_PB_WO', ['.weight', '.weight_scale_inv']
    if pre + '.weight_scale' in idx and idx[pre + '.weight'][1]['dtype'] == 'F8_E4M3':
        return 'FP8', ['.weight', '.weight_scale']
    return None, []


def crosscheck(a):
    """Wherever our checkpoint and a published one chose the same algorithm for
    the same TENSOR, our written BYTES must equal theirs.

    Stronger than the codec test, because it exercises the whole pipeline -
    source read, fused-group amax, weight_scale_2 sharing, packing, shard
    writing - on real files rather than on one tensor in memory.  Tensors where
    the two chose different algorithms have no counterpart and are reported as
    'no counterpart', which is not a failure.

    Tensor, not module: on an architecture whose FusedMoE is 160 experts on
    disk, one module is 480 files' worth of bytes and collapsing them into a
    single verdict would hide 479 of them.
    """
    ours = ST.index_dir(a.ckpt, 'model-*.safetensors')
    ref = ST.index_dir(a.against, 'model-*.safetensors')
    n_ok = n_bad = n_skip = 0
    bytes_ok = 0
    worst = []
    for pre in sorted({n.rsplit('.', 1)[0] for n in ours
                       if n.endswith(('.weight_scale', '.weight_scale_2',
                                      '.weight_scale_inv'))}):
        mine, names = _layout(ours, pre)
        if mine is None:
            continue
        theirs, _ = _layout(ref, pre)
        if theirs != mine:
            n_skip += 1
            continue
        diffs = []
        for suf in names:
            b1, b2 = ST.raw(ours, pre + suf), ST.raw(ref, pre + suf)
            if b1 != b2:
                x = np.frombuffer(b1, np.uint8)
                y = np.frombuffer(b2, np.uint8)
                diffs.append(f'{suf} {100*float((x != y).mean()):.6f}% of '
                             f'{len(b1)} B differ')
            else:
                bytes_ok += len(b1)
        if diffs:
            n_bad += 1
            worst.append(f'{pre}: ' + '; '.join(diffs))
        else:
            n_ok += 1
    print(f'checkpoint {a.ckpt}')
    print(f'  reference                            : {a.against}')
    print(f'  tensors where our algo == reference\'s: {n_ok + n_bad}')
    print(f'  byte-identical                       : {n_ok}')
    print(f'  differing                            : {n_bad}')
    print(f'  no counterpart (different algo)      : {n_skip}')
    print(f'  bytes proven identical               : {bytes_ok:,}')
    for w in worst[:10]:
        print('   *', w)
    return 1 if n_bad else 0


# =============================================================================
# SELF-TEST
# =============================================================================

def selftest(verbose=True) -> int:
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)
        if verbose:
            print(('  ok   ' if ok else '  FAIL ') + msg)

    dq = SN.DiskNames(SN.ARCH_QWEN3_5)
    chk(dq.classify('model.language_model.layers.3.self_attn.q_proj')
        == ('model.language_model.layers.3.self_attn.q_proj',
            ('model.language_model.layers.3.self_attn', 'qkv_proj')),
        'qwen3_5: the disk name IS the module name, and the scale group is the '
        'arch fused key - this arch goes through DiskNames unchanged')
    chk(dq.group_members(('m.self_attn', 'qkv_proj'))
        == ['m.self_attn.q_proj', 'm.self_attn.k_proj', 'm.self_attn.v_proj'],
        'qkv_proj shard order preserved from the arch table')
    chk(dq.classify('model.language_model.layers.3.mlp.down_proj')[1] is None,
        'an unfused module is its own scale group (None)')

    base = 'model.language_model.layers.0.self_attn'
    ok_a = {f'{base}.q_proj': 'NVFP4', f'{base}.k_proj': 'NVFP4',
            f'{base}.v_proj': 'NVFP4'}
    _g, bad = check_fusion(ok_a, dq)
    chk(not bad, 'fusion check: a uniform group passes')
    mixed = dict(ok_a, **{f'{base}.k_proj': 'FP8'})
    _g, bad = check_fusion(mixed, dq)
    chk(len(bad) == 1 and 'disagree' in bad[0],
        'fusion check: a mixed group is refused (SGLang would raise)')
    partial = {f'{base}.q_proj': 'NVFP4', f'{base}.k_proj': 'NVFP4'}
    _g, bad = check_fusion(partial, dq)
    chk(len(bad) == 1 and 'BF16' in bad[0],
        'fusion check: a shard left BF16 while its siblings are quantised is the '
        'same fault')

    # --- the fused / stacked architecture: modules are not tensors ---------- #
    dg = SN.DiskNames(SN.ARCH_GLM4_MOE)
    chk(dg.classify('model.layers.5.self_attn.k_proj')
        == ('model.layers.5.self_attn.qkv_proj',
            ('model.layers.5.self_attn', 'qkv_proj')),
        'glm4_moe: a shard on disk resolves to the FUSED module the config names')
    chk(dg.classify('model.layers.5.mlp.experts.7.gate_proj')
        == ('model.layers.5.mlp.experts', ('model.layers.5.mlp.experts.7', 'w13')),
        'glm4_moe: expert 7 resolves to the one FusedMoE module, and to ITS OWN '
        'w13 scale group - not to a group shared by all 160')
    chk(dg.classify('model.layers.5.mlp.experts.7.down_proj')
        == ('model.layers.5.mlp.experts', None),
        'glm4_moe: an expert down_proj is w2, its own scale (w2_weight_scale_2 is '
        '[num_experts], modelopt_quant.py:2446)')
    chk(dg.group_members(('model.layers.5.mlp.experts.7', 'w13'))
        == ['model.layers.5.mlp.experts.7.gate_proj',
            'model.layers.5.mlp.experts.7.up_proj'],
        'glm4_moe: a w13 group is one expert\'s gate+up, measured against '
        'Salyut1/GLM-4.7-NVFP4')
    chk(dg.classify('model.layers.1.mlp.gate_proj')[0]
        == 'model.layers.1.mlp.gate_up_proj'
        and dg.classify('model.layers.1.mlp.shared_experts.gate_proj')[0]
        == 'model.layers.1.mlp.shared_experts.gate_up_proj',
        'glm4_moe: the dense MLP and the shared expert have the same leaf names '
        'and different modules - the parent prefix separates them')
    chk(dg.classify('model.layers.5.mlp.gate') is None,
        'glm4_moe: the router is role "other" and classifies to nothing')

    src = {'model.layers.5.self_attn.q_proj.weight': 1,
           'model.layers.5.self_attn.k_proj.weight': 1,
           'model.layers.5.self_attn.v_proj.weight': 1,
           'model.layers.5.mlp.experts.0.gate_proj.weight': 1,
           'model.layers.5.mlp.experts.0.up_proj.weight': 1,
           'model.layers.5.mlp.experts.0.down_proj.weight': 1,
           'model.layers.5.mlp.experts.1.gate_proj.weight': 1,
           'model.layers.5.mlp.experts.1.up_proj.weight': 1,
           'model.layers.5.mlp.experts.1.down_proj.weight': 1,
           'model.layers.5.input_layernorm.weight': 1}
    da, miss = disk_assignment({'model.layers.5.self_attn.qkv_proj': 'NVFP4',
                                'model.layers.5.mlp.experts': 'FP8'}, src, dg)
    chk(not miss and len(da) == 9,
        f'disk_assignment expands 2 modules onto the 9 tensors that carry them '
        f'({len(da)})')
    chk(all(v == 'FP8' for k, v in da.items() if '.experts.' in k),
        'disk_assignment gives every expert of a FusedMoE the module\'s one algo')
    _g, miss2 = disk_assignment({'model.layers.5.mlp.gate_up_proj': 'FP8'}, src, dg)
    chk(miss2 == ['model.layers.5.mlp.gate_up_proj'],
        'disk_assignment reports a module the source does not carry')
    g, bad = check_fusion(da, dg)
    chk(not bad and len(g) == 3,
        f'check_fusion sees 3 scale groups here - one qkv and one w13 per expert '
        f'({len(g)})')

    chk(dg.mtp_prefixes(['model.layers.92.eh_proj.weight',
                         'model.layers.92.mlp.experts.3.up_proj.weight',
                         'model.layers.5.self_attn.q_proj.weight'])
        == ('model.layers.92.', 'mtp.'),
        'the MTP head is DERIVED - the layer carrying a nextn-only tensor, its '
        'experts included - not a hard-coded layer number')
    chk(dq.mtp_prefixes(['model.language_model.layers.3.mlp.down_proj.weight'])
        == ('mtp.',),
        'an arch with no nextn tensors in its map yields only the mtp. prefix')

    chk(needs_input_scale('m.self_attn.q_proj', 'NVFP4')
        and needs_input_scale('m.self_attn.q_proj', 'FP8')
        and not needs_input_scale('model.embed_tokens', 'NVFP4')
        and not needs_input_scale('m.mlp.down_proj', 'FP8_PB_WO'),
        'W4A4 and static-W8A8 need a calibrated activation scale; a gather and '
        'a dynamic-activation format do not')

    # the output plan must register exactly what create_weights() does
    src = {'x.weight': ('f', {'dtype': 'BF16', 'shape': [256, 512]}, 0),
           'y.weight': ('f', {'dtype': 'BF16', 'shape': [256, 512]}, 0),
           'z.weight': ('f', {'dtype': 'BF16', 'shape': [256, 512]}, 0),
           'e.weight': ('f', {'dtype': 'BF16', 'shape': [1000, 512]}, 0),
           'n.bias': ('f', {'dtype': 'BF16', 'shape': [256]}, 0)}
    plan, _own = plan_outputs(src, {'x': 'NVFP4', 'y': 'FP8', 'z': 'FP8_PB_WO'},
                              lambda p: True)
    names = [p[0] for p in plan]
    chk(names[:4] == ['x.weight', 'x.weight_scale', 'x.weight_scale_2', 'x.input_scale'],
        f'plan: NVFP4 registers weight + scale + ws2 + input_scale ({names[:4]})')
    chk(dict((p[0], (p[1], p[2])) for p in plan)['x.weight'] == ('U8', (256, 256)),
        'plan: NVFP4 weight is U8 [N, K/2]')
    chk(dict((p[0], (p[1], p[2])) for p in plan)['x.weight_scale'] == ('F8_E4M3', (256, 32)),
        'plan: NVFP4 weight_scale is e4m3 [N, K/16]')
    chk(dict((p[0], (p[1], p[2])) for p in plan)['z.weight_scale_inv'] == ('F32', (2, 4)),
        'plan: FP8_PB_WO weight_scale_inv is f32 [ceil(N/128), ceil(K/128)]')
    chk('n.bias' in names and dict((p[0], p[1]) for p in plan)['n.bias'] == 'BF16',
        'plan: a non-weight tensor is copied through untouched')
    chk('e.weight_scale' not in names,
        'plan: a module absent from the recipe stays BF16 - absence IS the encoding')
    plan2, _o = plan_outputs({'e.weight': src['e.weight']}, {'e': 'NVFP4'},
                             lambda p: False)
    chk([p[0] for p in plan2] == ['e.weight', 'e.weight_scale', 'e.weight_scale_2'],
        'plan: the NVFP4 EMBEDDING registers no input_scale (a gather has no '
        'activation to quantise)')

    total = sum(p[3] for p in plan)
    sh = shard_split(plan, total // 2)
    chk(sum(len(s) for s in sh) == len(plan), 'shard split: keeps every tensor')
    chk(len(sh) > 1, f'shard split: a plan larger than the shard cap splits ({len(sh)})')
    owners = {}
    for i, s in enumerate(sh):
        for n, _d, _sp, _nb in s:
            owners.setdefault(n.rsplit('.', 1)[0], set()).add(i)
    chk(all(len(v) == 1 for v in owners.values()),
        'shard split: a module\'s weight and its scales never straddle two files')

    # --- the source door: which arch, and where the tensor set comes from -- #
    import argparse as _ap
    import tempfile

    def _args(**kw):
        d = dict(arch=None, source=None, ckpt=None, hf_files=None, bf16_map=None,
                 gguf_companion=[])
        d.update(kw)
        return _ap.Namespace(**d)

    class _Refuse(Exception):
        pass

    class _P:
        def error(self, m):
            raise _Refuse(m)

    with tempfile.TemporaryDirectory() as d:
        snap = os.path.join(d, 'snap')
        os.makedirs(snap)
        json.dump({'architectures': ['Qwen3_5ForConditionalGeneration']},
                  open(os.path.join(snap, 'config.json'), 'w', encoding='utf-8'))
        key, why = resolve_arch(_args(source=snap), _P())
        chk(key == 'qwen3_5' and 'config.json' in why,
            f'arch: a snapshot\'s config.json names it, no --arch needed ({why})')
        key, _w = resolve_arch(_args(source=snap, arch='glm4_moe'), _P())
        chk(key == 'glm4_moe', 'arch: --arch still overrides what was detected')
        ck = os.path.join(d, 'ck')
        os.makedirs(ck)
        json.dump({'architectures': ['Glm4MoeForCausalLM']},
                  open(os.path.join(ck, 'config.json'), 'w', encoding='utf-8'))
        try:
            resolve_arch(_args(source=snap, ckpt=ck), _P())
            chk(False, 'arch: two witnesses that disagree must be refused')
        except _Refuse as e:
            chk('do not describe one model' in str(e),
                'arch: a checkpoint verified against another model\'s snapshot '
                'is refused, naming both')
        try:
            resolve_arch(_args(source=os.path.join(d, 'nothing')), _P())
            chk(False, 'arch: no witness at all must be refused')
        except _Refuse as e:
            chk('--arch' in str(e) and 'registered ones are' in str(e),
                'arch: with nothing to read from, it refuses and names the flag '
                'and the registered archs')

        mp = os.path.join(d, 'tensors.bf16.map')
        with open(mp, 'w', encoding='utf-8') as fh:
            fh.write('s.gguf:0:blk.0.ffn_up.weight:shape=(2, 3):dtype=bf16:'
                     'elements=6:bytes=12\n')
        uni, why = recipe_universe(_args(bf16_map=mp), {})
        chk(uni == ['blk.0.ffn_up.weight'] and 'bf16-map' in why,
            'tensor set: --bf16-map wins and says so')
        uni, why = recipe_universe(_args(), {})
        chk(uni is None, 'tensor set: an HF snapshot cannot supply GGUF names')

    class _FakeSplit:
        map_path = '/x/tensors.map'

        def universe(self):
            return ['blk.0.ffn_up.weight']

    class _FakeSrc(dict):
        parts = {'text': _FakeSplit()}

        def universe(self):
            return _FakeSplit().universe()

    uni, why = recipe_universe(_args(), _FakeSrc())
    chk(uni == ['blk.0.ffn_up.weight'] and 'tensors.map' in why,
        'tensor set: a GGUF split supplies its own, so --bf16-map is optional '
        f'({why})')

    # --- the per-tensor split: naming, companion order, and a round trip --- #
    def _quiet(fn, *args):
        """Run a mode with its progress log silenced; the assertions are on
        the files it leaves behind, not on what it printed."""
        saved, globals()['log'] = log, lambda *a: None
        try:
            return fn(*args)
        finally:
            globals()['log'] = saved

    chk(split_naming('/x/Qwen3.8-27B-THIREUS-BF16-SPECIAL_SPLIT', 'sgl_nvfp4')
        == ('Qwen3.8-27B', 'THIREUS',
            'Qwen3.8-27B-THIREUS-SGL_NVFP4-SPECIAL_SPLIT'),
        'split naming: the qtype is swapped and the model and maintainer kept, '
        'so an SGL repository sits beside the GGUF ones under the same scheme')
    try:
        split_naming('/x/some-directory', 'sgl_fp8')
        chk(False, 'split naming: a directory that is not a split must be refused')
    except SystemExit:
        chk(True, 'split naming: a --source that is not a SPECIAL_SPLIT is '
                  'refused rather than guessed at')

    # --- the BF16 pins hold in the REPOSITORY, not only in a recipe -------- #
    #
    # The model code reads these in BF16 whatever the file says, so a uniform
    # repository that quantised them would carry files no server can load - and
    # a repository is drawn from one tensor at a time, so pinning them in the
    # preset alone is not enough.  What is deliberately NOT changed is the size
    # model: `plan_tensor` still prices a pinned tensor at the algo it is asked
    # about, which is what makes it reproduce a published checkpoint's byte
    # count, and the assigner never asks it for a pinned tensor anyway because
    # the preset has already assigned that tensor sgl_bf16.
    class _EntrySplit:
        """The one thing uniform_algos() asks a split for."""

        def __init__(self, ent):
            self._e = ent

        def entries(self):
            return self._e

    _ssm = 'blk.0.ssm_alpha.weight'
    _ent = {_ssm: ('p', (5120, 48), 30),
            'blk.0.ssm_beta.weight': ('p', (5120, 48), 30),
            'blk.0.ffn_down.weight': ('p', (17408, 5120), 30)}
    _ua = uniform_algos(_EntrySplit(_ent), 'sgl_nvfp4', SN.ARCH_QWEN3_5)
    chk(_ua[_ssm][:2] == (SN.BF16, 491520)
        and _ua['blk.0.ssm_beta.weight'][:2] == (SN.BF16, 491520),
        'split pins: --split sgl_nvfp4 keeps in_proj_a / in_proj_b at sgl_bf16 '
        'and at their BF16 size, so the map beside the repository says sgl_bf16 '
        f'and the file holds it ({_ua[_ssm][:2]})')
    chk('in_proj_ba' in _ua[_ssm][2] and _ua[_ssm][2].startswith('pinned BF16'),
        'split pins: the reason is the advisory\'s own, so the map and the '
        f'--pins argument cannot say different things ({_ua[_ssm][2]!r})')
    chk(_ua['blk.0.ffn_down.weight'][0] == 'sgl_nvfp4',
        'split pins: a tensor no pin names is quantised as asked')
    chk(SN.plan_tensor(_ssm, [5120, 48], 245760, 'sgl_nvfp4',
                       SN.ARCH_QWEN3_5)[:2] == ('sgl_nvfp4', 138248),
        'split pins: the SIZE MODEL is untouched - it still prices a pinned '
        'tensor at the algo asked about, which is how it reproduces a '
        'published checkpoint exactly')
    import tempfile as _tf
    with _tf.NamedTemporaryFile('w', suffix='.recipe', delete=False) as _fh:
        _fh.write('^blk\\.0\\.ssm_alpha\\.weight$=sgl_nvfp4\n^blk\\.0\\.attn_q\\.weight$=sgl_nvfp4\n')
        _tmp = _fh.name
    try:
        load_assignment(_tmp, SN.ARCH_QWEN3_5)
        chk(False, 'a recipe that quantises a pinned tensor is refused by --build')
    except SystemExit as _e:
        chk('pins to BF16' in str(_e) and 'ssm_alpha' in str(_e),
            'a recipe that quantises a pinned tensor is refused by --build, '
            'naming the tensor and the reason')
    finally:
        os.unlink(_tmp)
    chk(SN.pinned_bf16('blk.7.ssm_beta.weight', SN.ARCH_QWEN3_5)
        == SN.PIN_BF16_ADVISORY['qwen3_5'][0][1]
        and SN.pinned_bf16('blk.7.ffn_down.weight', SN.ARCH_QWEN3_5) is None,
        'split pins: PIN_BF16_ADVISORY is the one table both the preset and '
        'the split read')
    _gua = uniform_algos(_EntrySplit({'blk.3.ffn_gate_inp.weight':
                                      ('p', (5120, 160), 0)}),
                         'sgl_nvfp4', SN.ARCH_GLM4_MOE)
    chk(_gua['blk.3.ffn_gate_inp.weight'][:2] == (SN.BF16, 1638400)
        and 'router' in _gua['blk.3.ffn_gate_inp.weight'][2],
        'split pins: the GLM router is held BF16 at its HF width and says why '
        '- role "other" already left it there, so the pin costs this arch no '
        'bytes and gives the map line its reason')

    # THE ORDER THAT MAKES AN ASSEMBLED SHARD A BUILT ONE.  `--assemble` puts a
    # module back together by taking its weight and then whichever COMPANIONS
    # are present, in that list's order; that has to be the order plan_outputs
    # planned them in, or the shard offsets differ and only the tensors match.
    _pre = 'model.language_model.layers.0.mlp.down_proj'
    for _al, _want in (('NVFP4', ('.weight_scale', '.weight_scale_2',
                                  '.input_scale')),
                       ('W4A16_NVFP4', ('.weight_scale', '.weight_scale_2')),
                       ('FP8', ('.weight_scale', '.input_scale')),
                       ('FP8_PB_WO', ('.weight_scale_inv',))):
        _src = {_pre + '.weight': ('p', {'dtype': 'BF16', 'shape': [256, 128]}, 0)}
        _pl, _own = plan_outputs(_src, {_pre: _al}, lambda p: True)
        _got = tuple(n[len(_pre):] for n, _d, _s, _b in _pl)
        chk(_got == ('.weight',) + _want,
            f'{_al}: plan_outputs emits {list(_got)}')
        chk(tuple(c for c in COMPANIONS if c in _want) == _want,
            f'{_al}: COMPANIONS lists them in that same order, which is what '
            f'--assemble rebuilds the shard from')

    # --- a whole per-tensor split, assembled back ------------------------- #
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sd = os.path.join(d, 'M-T-SGL_NVFP4-SPECIAL_SPLIT')
        os.makedirs(sd)
        nrm = 'model.language_model.layers.0.input_layernorm.weight'
        wgt = _pre + '.weight'
        # the index deliberately ranks down_proj FIRST and sorts SECOND, so an
        # assembler that ignored it would be caught here
        cfgb = json.dumps({'architectures': ['Qwen3_5ForConditionalGeneration'],
                           'num_hidden_layers': 1}).encode()
        ixb = json.dumps({'weight_map': {wgt: 'model-00001.safetensors',
                                         nrm: 'model-00002.safetensors'}}).encode()
        blobs = [('config.json', cfgb), ('model.safetensors.index.json', ixb)]
        w = ST.ShardWriter(
            os.path.join(sd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00001-of-00003'
                             '.safetensors'),
            [(HF_FILE + n, 'U8', (len(b),), len(b)) for n, b in blobs],
            {'format': 'pt', 'kind': 'metadata', 'part': 'text',
             'arch': 'qwen3_5'})
        for _n, b in blobs:
            w.write(HF_FILE + _n, np.frombuffer(b, np.uint8))
        w.close()
        pk = np.arange(8, dtype=np.uint8).reshape(4, 2)
        sc = np.arange(4, dtype=np.uint8).reshape(4, 1)
        w = ST.ShardWriter(
            os.path.join(sd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00002-of-00003'
                             '.safetensors'),
            [(wgt, 'U8', (4, 2), pk.nbytes),
             (_pre + '.weight_scale', 'F8_E4M3', (4, 1), sc.nbytes),
             (_pre + '.weight_scale_2', 'F32', (), 4),
             (_pre + '.input_scale', 'F32', (), 4)],
            {'format': 'pt', 'kind': 'tensor', 'quant_algo': 'NVFP4',
             'gguf_tensor': 'blk.0.ffn_down.weight'})
        w.write(wgt, pk)
        w.write(_pre + '.weight_scale', sc)
        w.write(_pre + '.weight_scale_2', np.float32(0.5))
        w.write(_pre + '.input_scale', np.float32(0.25))
        w.close()
        g = np.arange(4, dtype=np.uint16)
        w = ST.ShardWriter(
            os.path.join(sd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00003-of-00003'
                             '.safetensors'),
            [(nrm, 'BF16', (4,), g.nbytes)],
            {'format': 'pt', 'kind': 'tensor', 'quant_algo': 'BF16',
             'gguf_tensor': 'blk.0.attn_norm.weight'})
        w.write(nrm, g)
        w.close()

        out = os.path.join(d, 'ckpt')
        ns = argparse.Namespace(assemble=[sd], out=out, name='t',
                                shard_bytes=1 << 30, kv_cache_quant_algo='FP8',
                                exclude=['mtp*'], dry_run=False)
        chk(_quiet(assemble, ns, SN.ARCH_QWEN3_5) == 0,
            'assemble: a three-file split assembles')
        got = ST.index_dir(out, 'model-*.safetensors')
        chk(list(got) == [wgt, _pre + '.weight_scale', _pre + '.weight_scale_2',
                          _pre + '.input_scale', nrm],
            f'assemble: the tensor order is the metadata shard\'s index order, '
            f'not sorted() - down_proj before the norm ({list(got)[:2]})')
        chk(ST.raw(got, wgt) == pk.tobytes()
            and ST.raw(got, nrm) == g.tobytes(),
            'assemble: the bytes are copied, not recomputed')
        qc = json.load(open(os.path.join(out, 'hf_quant_config.json'),
                            encoding='utf-8'))['quantization']
        chk(qc['quantized_layers'] == {_pre: {'quant_algo': 'NVFP4',
                                              'group_size': 16}},
            f'assemble: quantized_layers is read off the files and keyed on the '
            f'MODULE ({qc["quantized_layers"]})')
        ix = json.load(open(os.path.join(out, 'model.safetensors.index.json'),
                            encoding='utf-8'))
        chk(ix['metadata']['total_size'] == 8 + 4 + 4 + 4 + 8
            and set(ix['weight_map']) == set(got),
            'assemble: the written index accounts for every tensor')
        chk('quantization_config' in json.load(
            open(os.path.join(out, 'config.json'), encoding='utf-8')),
            'assemble: config.json comes from the metadata shard with the '
            'quantization_config edited in')

        # --- --assemble refuses a checkpoint with a hole in it --------- #
        #
        # The download is one quant_downloader.sh run per part, so leaving a
        # companion out is easy, silent, and fatal on the GPU rather than
        # here.  Both halves of the rule are asserted: the part nothing was
        # stamped with, and the part that arrived half-downloaded.
        qcfg = {'vision_config': {},
                'text_config': {'mtp_num_hidden_layers': 1}}
        ixn = ['model.language_model.layers.0.self_attn.q_proj.weight',
               'model.visual.blocks.0.attn.qkv.weight',
               'mtp.layers.0.self_attn.q_proj.weight']
        all3 = {'text', 'mtp', 'vision'}
        chk(assembly_gaps(qcfg, 'qwen3_5', {'text'}, ixn, {ixn[0]})
            == [('mtp', [ixn[2]]), ('vision', [ixn[1]])],
            'assemble completeness: the main split alone is refused, and each '
            'missing tensor is attributed to the companion that owns it by '
            'the HF-name prefix its part has in the inversion table')
        chk(assembly_gaps(qcfg, 'qwen3_5', all3, ixn, set(ixn)) == [],
            'assemble completeness: all three parts there and every tensor '
            'the index names present - no gap')
        chk(assembly_gaps(qcfg, 'qwen3_5', all3, ixn, set(ixn[:2]))
            == [('mtp', [ixn[2]])],
            'assemble completeness: a companion that arrived HALF-downloaded '
            'is a gap too - the stamp says the part is there and the index '
            'says a tensor of it is not')
        chk(assembly_gaps(qcfg, 'qwen3_5', {'text'}, [], set())
            == [('mtp', []), ('vision', [])],
            'assemble completeness: with no index in the metadata shard the '
            'part each shard stamps itself with is the only witness left, and '
            'a part nothing is stamped with is still a refusal')
        chk(assembly_gaps(qcfg, 'qwen3_5', all3, ixn + ['x.y'], set(ixn))
            == [(None, ['x.y'])],
            'assemble completeness: a missing tensor no companion of this '
            'arch claims is reported on its own - that is a short download, '
            'not a forgotten directory')
        chk(assembly_gaps({'num_hidden_layers': 92,
                           'num_nextn_predict_layers': 1}, 'glm4_moe',
                          {'text'}, ['model.layers.92.mlp.gate.weight'],
                          {'model.layers.92.mlp.gate.weight'}) == [],
            'assemble completeness: GLM declares a draft head and keeps it in '
            'the main stack, so one directory is the whole checkpoint and no '
            'companion is demanded')
        chk(assembly_gaps(qcfg, 'qwen3_5', {'text'}, ixn[:1], {ixn[0]})
            == [('mtp', []), ('vision', [])],
            'assemble completeness: an index that names ONLY the text part\'s '
            'tensors acquits nothing - it has nothing to say about a companion '
            'it never mentions, and that is precisely the metadata shard a '
            'part-by-part conversion leaves behind')

        # --- and the refusal names the MODEL, not the download directory --- #
        #
        # models/<name>/download.conf is keyed on the model; the directory is
        # called whatever the person who ran quant_downloader.sh typed.  This
        # split is in one called `dl-main` and stamps model=Qwen3.8-27B, and
        # the refusal has to print the second.  Its index names only the text
        # tensor, so it is the case above end to end as well.
        odd = os.path.join(d, 'dl-main')
        os.makedirs(odd)
        cfg2 = json.dumps({'architectures': ['Qwen3_5ForConditionalGeneration'],
                           'num_hidden_layers': 1, 'vision_config': {},
                           'text_config': {'mtp_num_hidden_layers': 1}}).encode()
        ix2 = json.dumps({'weight_map': {nrm: 'model-00001.safetensors'}}).encode()
        b2 = [('config.json', cfg2), ('model.safetensors.index.json', ix2)]
        w = ST.ShardWriter(
            os.path.join(odd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00001-of-00002'
                              '.safetensors'),
            [(HF_FILE + n, 'U8', (len(b),), len(b)) for n, b in b2],
            {'format': 'pt', 'kind': 'metadata', 'part': 'text',
             'arch': 'qwen3_5', 'model': 'Qwen3.8-27B'})
        for _n, b in b2:
            w.write(HF_FILE + _n, np.frombuffer(b, np.uint8))
        w.close()
        w = ST.ShardWriter(
            os.path.join(odd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00002-of-00002'
                              '.safetensors'),
            [(nrm, 'BF16', (4,), g.nbytes)],
            {'format': 'pt', 'kind': 'tensor', 'part': 'text',
             'quant_algo': 'BF16', 'model': 'Qwen3.8-27B',
             'gguf_tensor': 'blk.0.attn_norm.weight'})
        w.write(nrm, g)
        w.close()
        ns2 = argparse.Namespace(assemble=[odd], out=os.path.join(d, 'ckpt3'),
                                 name='t', shard_bytes=1 << 30,
                                 kv_cache_quant_algo='FP8', exclude=['mtp*'],
                                 dry_run=False)
        try:
            _quiet(assemble, ns2, SN.ARCH_QWEN3_5)
            chk(False, 'assemble: a split whose index names only its own part '
                       'must still be refused for the companions it lacks')
        except SystemExit as _e:
            _m = str(_e)
            chk('the mtp part is missing' in _m
                and 'the vision part is missing' in _m,
                'assemble completeness: end to end, a metadata shard whose '
                'index lists only the text tensors is refused and BOTH absent '
                'companions are named')
            chk('models/mtp-Qwen3.8-27B/download.conf' in _m
                and 'models/mmproj-Qwen3.8-27B/download.conf' in _m
                and 'dl-main' not in _m,
                'assemble: the download.conf to run is keyed on the model the '
                'metadata shard STAMPS, so the path printed exists even when '
                'the download went into a directory called something else')

        # THE MAP PROMISES THE FILE'S OWN sha256, header included - and a small
        # header is still in the writer's buffer when the hash starts, so
        # reading it back without flushing first would hash NOTHING and every
        # map line would be wrong in a way only a download reveals.
        _hp = os.path.join(d, 'hash-probe.safetensors')
        _w = ST.ShardWriter(_hp, [('t', 'BF16', (4,), 8)], {'format': 'pt'})
        _w.f.flush()
        _h = hashlib.sha256()
        with open(_hp, 'rb') as _fh:
            _h.update(_fh.read(_w.base))
        _b = np.arange(4, dtype=np.uint16).tobytes()
        _w.f.write(_b)
        _w.i += 1
        _h.update(_b)
        _w.f.close()
        with open(_hp, 'rb') as _fh:
            chk(_h.hexdigest() == hashlib.sha256(_fh.read()).hexdigest(),
                'map sha256: the digest taken while writing IS the digest of the '
                'finished file, header included - which is what the downloader '
                'checks a fetched file against')

        # and it refuses a file that claims an algo its tensors do not support
        bad = os.path.join(d, 'bad')
        os.makedirs(bad)
        shutil.copy2(os.path.join(sd, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00001-of-'
                                      '00003.safetensors'), bad)
        w = ST.ShardWriter(
            os.path.join(bad, 'M-T-SGL_NVFP4-SPECIAL_TENSOR-00002-of-00003'
                              '.safetensors'),
            [(wgt, 'U8', (4, 2), pk.nbytes)],
            {'format': 'pt', 'kind': 'tensor', 'quant_algo': 'NVFP4'})
        w.write(wgt, pk)
        w.close()
        ns.assemble = [bad]
        ns.out = os.path.join(d, 'ckpt2')
        try:
            _quiet(assemble, ns, SN.ARCH_QWEN3_5)
            chk(False, 'assemble: a file claiming NVFP4 with no scales must be '
                       'refused')
        except SystemExit:
            chk(True, 'assemble: a file that records NVFP4 and carries no '
                      'weight_scale is refused, not written into a config that '
                      'would fail on the GPU')

    chk(GG.selftest(verbose=False) == 0,
        'sglang_gguf: the inversion tables, their arithmetic and its guards')
    chk(ST.selftest(verbose=False) == 0, 'sglang_st: container round-trip')
    try:
        import sglang_codecs as Q
        Q.require_torch()
        chk(Q.selftest(verbose=False) == 0, 'sglang_codecs: all codec invariants')
    except SystemExit:
        chk(True, 'sglang_codecs: SKIPPED (no torch here - the assign path does '
                  'not need it; run the writer under an interpreter that has it)')

    if verbose:
        print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', action='store_true',
                    help='MODE: turn a .recipe into a loadable checkpoint. Needs '
                         '--source, --recipe and --out, and needs torch.')
    ap.add_argument('--verify', action='store_true',
                    help='MODE: audit a checkpoint against the BF16 source it '
                         'came from - snapshot or GGUF split - for shapes, '
                         'dtypes, scales, and a dequantised sample (--sample). '
                         'Needs --ckpt and --source.')
    ap.add_argument('--crosscheck', action='store_true',
                    help='MODE: compare a checkpoint against a published '
                         'one - wherever both chose the same algorithm for the '
                         'same module the bytes must be EQUAL. The strongest '
                         'check available without a GPU. Needs --ckpt/--against.')
    ap.add_argument('--split', action='store_true',
                    help='MODE: write the whole model at ONE SGLang qtype as a '
                         'per-tensor SPECIAL_SPLIT - one .safetensors per GGUF '
                         'tensor, named and numbered like the BF16 GGUF split, '
                         'with a tensors.<qtype>.map - so quant_downloader.sh '
                         'can fetch any recipe out of it. Needs --qtype, '
                         '--source (the BF16 GGUF split), --hf-files and --out.')
    ap.add_argument('--assemble', nargs='+', default=None, metavar='DIR',
                    help='MODE: turn the directories quant_downloader.sh '
                         'downloaded into - the main split and its mtp-/mmproj- '
                         'companions - back into a servable sharded checkpoint '
                         '(model.safetensors.index.json, hf_quant_config.json, '
                         'config and tokenizer). Copies bytes and quantises '
                         'nothing, so it needs no torch. Needs --out.')
    ap.add_argument('--qtype', default=None,
                    help='which SGLang qtype --split writes the whole model at '
                         '(sgl_bf16, sgl_nvfp4, sgl_fp8, ...). It names the '
                         'output repository and every file in it, and the map '
                         'records the type each file ACTUALLY holds, which is '
                         'this one wherever the shape rules and the calibration '
                         'allow it and sgl_bf16 where they do not.')
    ap.add_argument('--selftest', action='store_true',
                    help='MODE: run the shard-plan and fused-group invariants and '
                         'exit. CPU, offline, and torch-free (the codec half is '
                         'skipped, and says so, when torch is absent).')
    ap.add_argument('--source',
                    help='where the BF16 weights are: an HF safetensors '
                         'snapshot, or the owner\'s BF16 GGUF SPECIAL_SPLIT '
                         '(the directory, or its first shard). A split is read '
                         'through sglang_gguf.py, which undoes every '
                         'transformation convert_hf_to_gguf.py applied and '
                         'yields the same checkpoint; it needs --hf-files for '
                         'the small text files a GGUF does not carry, and it '
                         'picks up the mmproj-/mtp- companion splits beside it '
                         'on its own.')
    ap.add_argument('--hf-files', default=None, metavar='DIR',
                    help='directory holding the model\'s small non-tensor files '
                         '(config.json, tokenizer*, generation_config.json, the '
                         'preprocessor configs, model.safetensors.index.json). '
                         'REQUIRED when --source is a GGUF split, which carries '
                         'weights and ggml metadata but not those files; they '
                         'are the non-LFS files of the model\'s own repository. '
                         'Everything given is CHECKED against the GGUF metadata, '
                         'and chat_template.jinja is regenerated from it when '
                         'absent. Defaults to --source for a snapshot.')
    ap.add_argument('--gguf-companion', action='append', default=[],
                    metavar='DIR',
                    help='an extra SPECIAL_SPLIT holding a part the main one '
                         'does not (the mmproj- vision tower, the mtp- draft '
                         'head). Repeatable. Omit it and the siblings named '
                         'mmproj-<split> and mtp-<split> are found automatically; '
                         'a part config.json says the model has and no split '
                         'carries is a refusal, not a silent hole.')
    ap.add_argument('--recipe', help='.recipe from quant_assign.py, or an '
                                     'hf_quant_config.json')
    ap.add_argument('--bf16-map', default=None,
                    help="the model's tensors.bf16.map (GGUF tensor set). A "
                         "published recipe is regex-compacted "
                         "(quants_regex_merger); given this, each pattern is "
                         "expanded over the model's tensors exactly as the GGUF "
                         "build path does, so a merged recipe rebuilds the "
                         "identical checkpoint as its per-tensor original. A raw "
                         "per-tensor recipe needs it not; a compacted recipe "
                         "without it errors rather than collapsing to BF16. "
                         "OPTIONAL when --source is a GGUF split: that carries "
                         "the same list in its own tensors.map, and failing that "
                         "in its shard headers. This flag still overrides both.")
    ap.add_argument('--out', help='output checkpoint directory')
    ap.add_argument('--name', default='sgl', help='name recorded in BUILD.json')
    ap.add_argument('--ckpt', help='checkpoint to verify or cross-check')
    ap.add_argument('--against', help='reference checkpoint for --crosscheck')
    ap.add_argument('--input-scales-from', default=None,
                    help='a calibrated checkpoint of the same model to harvest '
                         'static activation scales from (NVFP4 is W4A4 and needs them)')
    ap.add_argument('--amax-cache', default=None,
                    help='JSON cache of per-tensor weight amax; recipe-independent, '
                         'so one cache serves a whole family')
    ap.add_argument('--uncalibrated', choices=('refuse', 'bf16'), default='refuse',
                    help='what to do with a module the --input-scales-from '
                         'reference does not calibrate (an MTP head RadixArk '
                         'drops, an lm_head Salyut1 left BF16). refuse (default) '
                         'stops; bf16 leaves those modules unquantised - absence IS '
                         'the encoding - and names them in the log and BUILD.json. '
                         'There is no third option: an activation amax cannot be '
                         'computed from weights, and a fabricated 1.0 is silent.')
    ap.add_argument('--arch', default=None, choices=sorted(SN.ARCHS),
                    help='which sglang_native.ARCHS entry names this model\'s '
                         'tensors and fused groups. OPTIONAL: it is read from '
                         'whatever the command line points at - config.json\'s '
                         '`architectures` in a snapshot, a checkpoint or '
                         '--hf-files, and general.architecture plus the tensor '
                         'names in a GGUF split - and every witness must agree. '
                         'Pass it to override them.')
    ap.add_argument('--shard-bytes', type=int, default=int(4.5 * 2 ** 30),
                    help='target bytes per output safetensors shard (default '
                         '4.5 GiB). A module is never split across shards.')
    ap.add_argument('--kv-cache-quant-algo', default='FP8',
                    help="what to record as the KV cache's quant_algo in the "
                         "written config (default FP8). It describes the CACHE, "
                         "not any weight, and changes no bytes on disk.")
    ap.add_argument('--exclude', nargs='*', default=['mtp*', 'mtp.layers.0*'],
                    help='exclude_modules / ignore list for whole subtrees')
    ap.add_argument('--sample', type=int, default=8,
                    help='tensors to dequantise in --verify (0 skips it and the '
                         'torch dependency with it)')
    ap.add_argument('--dry-run', action='store_true',
                    help='plan the build and print the shard layout and the '
                         'quantized_layers census, then stop before writing '
                         'anything (and before importing torch).')
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.build:
        for r in ('source', 'recipe', 'out'):
            if not getattr(a, r):
                ap.error(f'--build needs --{r}')
    elif a.split:
        for r in ('source', 'qtype', 'out'):
            if not getattr(a, r):
                ap.error(f'--split needs --{r}')
    elif a.assemble:
        if not a.out:
            ap.error('--assemble needs --out')
    elif a.verify:
        for r in ('ckpt', 'source'):
            if not getattr(a, r):
                ap.error(f'--verify needs --{r}')
    elif a.crosscheck:
        for r in ('ckpt', 'against'):
            if not getattr(a, r):
                ap.error(f'--crosscheck needs --{r}')
    else:
        ap.print_help()
        return 0

    # ONE detection for every mode, so a checkpoint is never audited under a
    # different arch than it was written with.
    a._arch_key, why = resolve_arch(a, ap)
    arch = SN.ARCHS[a._arch_key]
    log(f'architecture {a._arch_key} ({why})')

    if a.build:
        return build(a, arch)
    if a.split:
        return split(a, arch)
    if a.assemble:
        return assemble(a, arch)
    if a.verify:
        return verify(a, arch)
    return crosscheck(a)


if __name__ == '__main__':
    sys.exit(main())
