#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_calibrate.py measures the static activation scales **#
#** NVFP4 and FP8 need, from a text you chose yourself.       **#
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
#** Copyright © 2026 - Thireus.    ₐₘₐₓ ᵢₛ ₙₒₜ ᵢₙ ₜₕₑ 𝓌ₑᵢ𝓰ₕₜₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""
sglang_calibrate.py - measure the static activation scales an NVFP4 (W4A4) or
per-tensor FP8 (W8A8-static) checkpoint needs, on a calibration text you chose
yourself, and write them in the two forms this suite consumes.

ONE GPU, named with --gpu, and no other.  The BF16 source is STREAMED: the
module tree is built without any weights in it, and each decoder layer's
parameters are read from the source, put on the card, walked and freed before
the next layer arrives.  Host RAM therefore holds a couple of layers whatever
the model weighs, and a 355B MoE calibrates on the same 32 GB card a 27B dense
one does.

    # measure, snapshotting the running amax at several token budgets at once
    sglang_calibrate.py --gpu 3 --source <BF16 snapshot> \\
        --text imatrix-calibration-corpus-v02.txt --tokens 131072 \\
        --snapshots 8192,32768,131072 --out-root scales --name ubergarm

    # the same, from the BF16 GGUF SPECIAL_SPLIT the downloader fetches
    sglang_calibrate.py --gpu 3 --source <BF16 split> --hf-files <small files> \\
        --text imatrix-calibration-corpus-v02.txt --out-root scales --name x

    # what our numbers look like next to somebody else's
    sglang_calibrate.py --compare scales/ubergarm-32768 \\
        --against <a published NVFP4/FP8 checkpoint>

    # the driver is the thing worth distrusting, so make it prove itself
    sglang_calibrate.py --selftest

WHAT THIS COMPUTES, AND WHY IT CANNOT BE READ OFF THE WEIGHTS
------------------------------------------------------------
A weight scale is a closed-form property of bytes we already hold: the tensor
IS the population, and its amax is a `.abs().max()` away.  An ACTIVATION amax
is a property of the input data, which does not exist until a forward pass
runs.  That is the whole asymmetry, and it is why `sglang_write.py` has had to
borrow `input_scale` from somebody else's checkpoint with `--input-scales-from`.

This tool ends that dependency.  It runs the BF16 source over a calibration
text and takes, per quantised linear, the running maximum of `|x|` over every
activation that enters it - the same statistic ModelOpt's calibrator takes, by
the same means (a forward pass with quantisation switched off and a hook on the
input), and differing only in the text we feed it and in our right to choose it.

    FP8   : input_scale = amax / 448
    NVFP4 : input_scale = amax / (6*448)

Both conventions are written to the JSON.  The safetensors carries the FP8 one,
which is what `--input-scales-from` reads back as `amax` for a module that has
no `weight_scale_2` beside it (sglang_write.py:340-350) - so the round trip is
exact, and no fictitious `weight_scale_2` has to be invented to carry a number
that is not about weights at all.

WHICH TEXT, AND ONE TEXT THAT IS FORBIDDEN
------------------------------------------
The amax IS the clipping threshold: an activation above it is clamped, in both
formats, silently.  Fitting that threshold to the evaluation text and then
reporting perplexity on the same text flatters the number.  So calibrate on the
imatrix corpus the suite already trusts for exactly this job, or on any general
text you like - and never on `wiki.test.raw`, which is what we measure with.
This tool refuses that file by name; the refusal is not a formality.

WHICH COPY OF THE BF16 WEIGHTS, AND WHY IT IS THE WRITER THAT OPENS IT
----------------------------------------------------------------------
`--source` takes whichever copy the machine has: the HF safetensors snapshot,
or the owner's BF16 GGUF SPECIAL_SPLIT together with `--hf-files` for the small
text files a split does not carry (config.json and the tokenizer).  Both are
opened with `sglang_write.open_source` and read with `sglang_write.src_read` -
the writer's own door, over `sglang_gguf.GgufSource` and `sglang_st.index_dir`.
The tool that MEASURES a scale and the tool that WRITES it therefore see the
same tensors under the same names and cannot drift apart; a name this reads is
a name that harvest looks up.

WHY LAYER-BY-LAYER, AND WHY IT IS THE SAME ARITHMETIC
-----------------------------------------------------
Sequences are independent (no cache, no cross-sequence state), so layer L sees
exactly the same inputs whether the model is walked layer-major or batch-major.
Walking it layer-major means one layer's weights on the card at a time - about
0.6 GB against a 52 GB model, 7.9 GB against a 668 GB one - and one sequential
pass over the checkpoint.  The claim is checked, not asserted: --selftest runs
a real model both ways and requires the per-module amax to agree exactly, and
--check-monolithic does the same on the model you are actually calibrating.

Layer-major is also what makes the streaming possible, and the streaming is what
removes the size limit: nothing but the layer being walked, the preamble and the
hidden states is ever resident, so the host never holds the model at all.  The
same walk reads a snapshot and a GGUF split, because both answer `src_read`.

The model's own preamble builds the rotary embeddings and the attention masks:
they are captured from a probe pass rather than reimplemented here, so a hybrid
stack gets the mask its own code chose for each layer type, not one this script
guessed.

THE EXPERTS THAT ARE NOT nn.Linear
-----------------------------------
transformers 5.16 keeps a MoE layer's routed experts as two stacked Parameters -
`gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I] - and loops over the experts
inside one module's forward (`Glm4MoeExperts`, modeling_glm4_moe.py:342).  A
hook on nn.Linear never sees them, which on GLM-4.7 would leave 42,987 of its
43,365 quantised modules uncalibrated.  They are recorded per expert instead,
from the routing that module is handed: the tokens routed to expert e give ONE
amax for its `gate_proj` and `up_proj` (both read that same input, and SGLang
collapses the pair into one `w13` scale anyway) and one for its `down_proj`.
They are filed under the names a checkpoint carries -
`model.layers.N.mlp.experts.E.gate_proj` and its two siblings - which is what
`sglang_write.harvest_act_amax` looks a scale up by, what `hf_disk` in
`sglang_native.ARCHS['glm4_moe']` maps `ffn_*_exps` onto, and what Salyut1's
published GLM-4.7-NVFP4 index states.  The number of routed tokens is recorded
beside each expert, because an expert that saw forty tokens has an amax and a
thin one, and thin coverage should be visible rather than inferred.

THE DRAFT HEAD IS NOT CALIBRATED, AND THE REASON IS NOT AN OVERSIGHT
--------------------------------------------------------------------
transformers builds no MTP/nextn module for either model in this lane.
`Glm4MoeForCausalLM` instantiates `num_hidden_layers` layers - 0..91 on GLM-4.7
- and lists `model.layers.92.*` in `_keys_to_ignore_on_load_unexpected`; the
Qwen3.8-27B class builds 1,184 parameters against its snapshot's 1,199, and the
15 it leaves out are exactly `mtp.*`.  A module that is not built cannot be
hooked, and running the head anyway would mean writing its forward - the
enorm / hnorm / eh_proj / shared_head chain - into a calibration tool, which is
a second implementation of the model and the one thing the probe-pass design
exists to avoid.  So the head comes out uncalibrated - and so does anything the
class DOES build but the walk never stages, a VLM's vision tower above all,
which is not in the decoder stack this goes down.  The two are different facts
and are counted, named and reported APART in the log and in CALIBRATION.json;
`sglang_write.py --uncalibrated` then decides what happens to them, which is
what it already does, because every published checkpoint drops the head as well.

WHAT THIS NEEDS torch FOR, AND WHAT IT DOES NOT
-----------------------------------------------
Measuring a scale needs torch and a model.  READING one back - `--compare`,
`--help` - is json and numpy, so those run in the suite's own CPU venv, which
has no torch in it.  The import is therefore allowed to fail and every torch
path says so by name instead of dying on an import nobody asked it to do.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import re
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --------------------------------------------------------------------------- #
# The GPU has to be chosen BEFORE torch is imported.  This box runs production
# work on its other cards and a stray context on one of them is not a small
# mistake, so the choice is made by masking the others out of the process
# entirely rather than by trusting every later .to(device) to say the right
# number.
# --------------------------------------------------------------------------- #
def _early_gpu(argv):
    for i, t in enumerate(argv):
        if t == '--gpu' and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith('--gpu='):
            return t.split('=', 1)[1]
    return None


_GPU = _early_gpu(sys.argv[1:])
if _GPU is not None:
    # `--gpu cpu` hides every card; `--gpu N` shows exactly that one.
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if str(_GPU).lower() == 'cpu' else str(_GPU)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np                                              # noqa: E402

# torch is needed to MEASURE a scale and is not needed to READ one back.
# `--compare` and `--help` are json+numpy work, and the suite's own CPU venv
# (.venv) deliberately has no torch in it, so this import is allowed to fail
# there: the failure is carried rather than raised, and every path that touches
# a tensor calls `require_torch()` first and names what it wanted torch for.
# Only torch's own absence is tolerated - an installed-but-broken torch, which
# reports a different missing module, still raises here.
try:
    import torch                                                # noqa: E402
    from torch import nn                                        # noqa: E402
except ImportError as _exc:                                     # noqa: E402
    if getattr(_exc, 'name', None) != 'torch':
        raise
    torch = nn = None
    _NO_TORCH = str(_exc)
else:
    _NO_TORCH = None

import sglang_gguf as GG                                        # noqa: E402
import sglang_st as ST                                          # noqa: E402
import sglang_write as SW                                       # noqa: E402

# =============================================================================
# CONSTANTS
# =============================================================================
E4M3_MAX = 448.0                    # finfo(float8_e4m3fn).max
E2M1_MAX = 6.0                      # the largest FP4 code
NVFP4_WS2_DENOM = E2M1_MAX * E4M3_MAX          # 2688.0, exact in binary

# The evaluation text.  Calibrating on it would set the clipping threshold from
# the very tokens we later report perplexity over.  Matched by content hash as
# well as by name, because a copy under another name is the same mistake.
FORBIDDEN_SHA256 = {
    # wikitext-2 raw test, the suite's perplexity text, matched by content so a
    # copy under another name is refused too
    '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08': 'wiki.test.raw',
}
FORBIDDEN_BASENAMES = ('wiki.test.raw', 'wikitext-2-raw/wiki.test.raw')

# Log2 histogram used for the percentiles.  Wide enough that nothing an
# activation can plausibly be falls off either end, fine enough that a bin is
# 4.4% wide.
HIST_LO, HIST_HI, HIST_BINS = -24.0, 16.0, 640
HIST_CHUNK = 1 << 23                # elements per histogram kernel launch


def log(*a):
    print(*a, flush=True)


def require_torch(what):
    """Refuse a torch path in an interpreter that has none, saying which path."""
    if torch is None:
        raise SystemExit(
            '%s needs torch and this interpreter has none (%s).  Reading scale '
            'sets back - --compare, --help - is json and numpy work and runs '
            'here; measuring them needs the torch venv.' % (what, _NO_TORCH))


# =============================================================================
# THE RECORDER
# =============================================================================
class Recorder(object):
    """Running amax (and a log2 histogram) of |x| entering one module.

    Snapshots are prefixes of the batch order, so one pass yields every token
    budget at once: a batch is folded into every tier it still belongs to.  The
    running maxima live on the GPU and are read back once, at the end - a
    `.item()` per module per batch would put a device sync in the hot path and
    serialise the whole walk.
    """

    __slots__ = ('n_tiers', 'amax', 'hist', 'count', 'tokens', 'active', 'stride')

    def __init__(self, n_tiers, device, use_hist=True, stride=1):
        self.n_tiers = n_tiers
        self.amax = [torch.zeros((), dtype=torch.float32, device=device)
                     for _ in range(n_tiers)]
        self.hist = ([torch.zeros(HIST_BINS, dtype=torch.float64, device=device)
                      for _ in range(n_tiers)] if use_hist else None)
        self.count = [0] * n_tiers
        self.tokens = [0] * n_tiers
        self.active = list(range(n_tiers))
        self.stride = stride

    def observe(self, x):
        observe_into((self,), x)

    def add_tokens(self, n):
        """How many rows this module was given - only a routed expert has a
        number here, and it is the one that says whether its amax is thin."""
        for t in self.active:
            self.tokens[t] += int(n)

    def percentiles(self, tier, qs=(0.99, 0.999, 0.9999)):
        if self.hist is None or self.count[tier] == 0:
            return {}
        h = self.hist[tier].cpu().numpy()
        total = h.sum()
        if total <= 0:
            return {}
        cum = np.cumsum(h)
        w = (HIST_HI - HIST_LO) / HIST_BINS
        out = {}
        for q in qs:
            k = int(np.searchsorted(cum, q * total, side='left'))
            k = min(k, HIST_BINS - 1)
            out['p%g' % (q * 100)] = float(2.0 ** (HIST_LO + (k + 1) * w))
        return out


def observe_into(recs, x):
    """Fold |x| into several Recorders, reducing it ONCE.

    Two Recorders share a reduction whenever two modules share an input, which
    is exactly the routed expert's `gate_proj` and `up_proj`: they are two names
    for one activation, they must carry one number, and reducing the tensor
    twice to say so would double the cost of the MoE walk for nothing.
    """
    if not torch.is_tensor(x) or x.numel() == 0:
        return
    a = x.detach()
    if not a.is_floating_point():
        return
    m = a.abs().amax().float()
    # COUNTED HERE, NOT IN THE HISTOGRAM BRANCH.  `elements` says how large the
    # population behind this amax was; that is a fact about the measurement and
    # not about whether percentiles were asked for.  Counting it inside the
    # histogram made every --no-percentiles set report `"elements": 0` beside a
    # real number.  percentiles() takes its own total from the histogram itself
    # (h.sum()), so --hist-stride subsampling does not distort this count.
    seen = a.numel()
    for r in recs:
        for t in r.active:
            torch.maximum(r.amax[t], m, out=r.amax[t])
            r.count[t] += seen
    want = [r for r in recs if r.hist is not None and r.active]
    if not want:
        return
    stride = max(1, min(r.stride for r in want))
    v = a.reshape(-1)
    if stride > 1:
        v = v[::stride]
    n = v.numel()
    acc = None
    for i in range(0, n, HIST_CHUNK):
        c = v[i:i + HIST_CHUNK].abs().float()   # abs() allocates: never in place on the caller's storage
        c.clamp_(min=2.0 ** HIST_LO, max=2.0 ** HIST_HI)
        c.log2_()
        h = torch.histc(c, bins=HIST_BINS, min=HIST_LO, max=HIST_HI)
        acc = h.double() if acc is None else acc.add_(h.double())
    for r in want:
        for t in r.active:
            r.hist[t].add_(acc)


# =============================================================================
# MODEL PLUMBING
# =============================================================================
def load_bf16(source, dtype=None, attn='sdpa'):
    """The BF16 source, whole, in host RAM, with the checkpoint's own class.

    `config.architectures[0]` and not an Auto* mapping: the module NAMES have to
    be the checkpoint's names, because that is the key a scale is filed under
    and the key `--input-scales-from` looks it up by.
    """
    require_torch('--in-ram')
    dtype = torch.bfloat16 if dtype is None else dtype
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(source)
    arch = (cfg.architectures or [None])[0]
    if arch is None or not hasattr(transformers, arch):
        raise SystemExit('%s: config names no usable architecture (%r)' % (source, arch))
    cls = getattr(transformers, arch)
    log('  class %s, dtype %s, attn %s' % (arch, dtype, attn))
    model = cls.from_pretrained(source, dtype=dtype, attn_implementation=attn)
    model.eval()
    return model, cfg, arch


def find_stack(model):
    """(text model, its decoder-layer ModuleList, the ModuleList's dotted name).

    The longest `*.layers` ModuleList in the model.  On a VLM that is the text
    stack and not the vision tower, whose blocks are neither the longest nor
    called `layers`; on a plain LM there is only one candidate.
    """
    cand = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.ModuleList) and n.split('.')[-1] == 'layers' and len(m)]
    if not cand:
        raise SystemExit('no decoder-layer ModuleList found')
    name, layers = max(cand, key=lambda p: len(p[1]))
    parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
    return parent, layers, name


def layer_types(text_model, n):
    lt = getattr(getattr(text_model, 'config', None), 'layer_types', None)
    if lt and len(lt) >= n:
        return list(lt[:n])
    return ['uniform'] * n


def linear_children(root, name_of):
    """[(dotted name, module)] for every nn.Linear under `root`."""
    return [(name_of[id(m)], m) for m in root.modules()
            if isinstance(m, nn.Linear) and id(m) in name_of]


def _join(prefix, name):
    return (prefix + '.' + name) if prefix else name


def _owner(root, dotted):
    """(the module that owns the leaf, the leaf's own name)."""
    if '.' not in dotted:
        return root, dotted
    parent, leaf = dotted.rsplit('.', 1)
    return root.get_submodule(parent), leaf


# =============================================================================
# THE SOURCE - WHICHEVER COPY OF THE BF16 WEIGHTS THIS BOX HAS
# =============================================================================
class Source(object):
    """The BF16 weights, opened and read through the WRITER's own door.

    `sglang_write.open_source` decides what `--source` is and hands back either
    `sglang_st`'s index over an HF snapshot or a `sglang_gguf.GgufSource` over a
    BF16 SPECIAL_SPLIT; `sglang_write.src_read` reads one tensor out of either,
    in HF layout, at HF width, under the HF name.  Nothing below can tell which
    it got, and that is the point: the tool that measures a scale and the tool
    that writes it read the same bytes by the same route, so a name this finds
    is a name `harvest_act_amax` will look up.
    """

    def __init__(self, source, hf_files=None, companions=None, log_fn=log):
        a = argparse.Namespace(source=source, hf_files=hf_files or None,
                               gguf_companion=list(companions or []))
        self.path = source
        self.hf_files = hf_files or None
        self.kind = 'gguf-split' if GG.looks_like_split(source) else 'snapshot'
        self.idx = SW.open_source(a, log_fn=log_fn)
        # A split carries weights and metadata but not config.json or the
        # tokenizer; the writer already owns that rule, so ask it rather than
        # restate it.
        self.files = SW.nontensor_dir(a)
        log('source: %s, %d tensors, %s' % (self.kind, len(self.idx), source))
        log('  small files (config, tokenizer): %s' % self.files)

    def __contains__(self, name):
        return name in self.idx

    def read(self, name, device=None):
        t = as_torch(SW.src_read(self.idx, name))
        return t.to(device) if device is not None else t

    def provenance(self):
        """What has to be recorded for this set to be reproducible."""
        d = {'source_kind': self.kind, 'source': self.path,
             'hf_files': self.hf_files, 'source_tensors': len(self.idx)}
        if self.kind == 'gguf-split':
            parts = getattr(self.idx, 'parts', {}) or {}
            d['split_parts'] = {k: v.dir for k, v in sorted(parts.items())}
            mp = getattr(parts.get('text'), 'map_path', None)
            d['tensors_map'] = mp
            d['tensors_map_sha256'] = sha256_file(mp) if mp else None
            d['tensor_order'] = getattr(self.idx, 'order_from', None)
        return d


def as_torch(arr):
    """A source tensor, as torch, in the checkpoint's own dtype.

    `sglang_st` returns BF16 as uint16 because numpy has no bfloat16 - the bytes
    are the tensor's own, so the view IS the conversion and nothing is rounded.
    Everything else (the F32 router bias, an F16 anything) comes back as itself.
    """
    a = np.ascontiguousarray(arr)
    if not a.flags.writeable:
        a = a.copy()
    t = torch.from_numpy(a)
    return t.view(torch.bfloat16) if t.dtype == torch.uint16 else t


# =============================================================================
# THE MODULE TREE, WITHOUT ITS WEIGHTS
# =============================================================================
#
# `with torch.device('meta')` would put the BUFFERS on the meta device as well,
# and some buffers are COMPUTED rather than loaded - the rotary inverse
# frequencies first of all, which `__init__` derives from the config with
# `torch.arange`.  A meta `inv_freq` is not a missing number, it is a silently
# wrong one.  So what is redirected is the set of constructors that ALLOCATE
# storage, which is what every parameter is made with and what no derived buffer
# is; arithmetic on real tensors stays real and lands on the CPU as usual.
#
# MEASURED: on GLM-4.7 this builds the whole 92-layer tree in 0.7 s and leaves
# exactly 89 meta buffers, all of them `mlp.gate.e_score_correction_bias`, which
# the checkpoint carries and the streamer therefore fills like a parameter.

_ALLOCATORS = ('empty', 'zeros', 'ones', 'full', 'rand', 'randn')


@contextlib.contextmanager
def weightless():
    """Allocate parameters on the meta device, and leave everything else real."""
    old = {n: getattr(torch, n) for n in _ALLOCATORS}

    def redirect(fn):
        def wrapper(*a, **kw):
            kw['device'] = torch.device('meta')
            return fn(*a, **kw)
        return wrapper

    for n in _ALLOCATORS:
        setattr(torch, n, redirect(old[n]))
    try:
        yield
    finally:
        for n in _ALLOCATORS:
            setattr(torch, n, old[n])


def build_skeleton(files, dtype=None, attn='sdpa'):
    """(a weightless model with the checkpoint's own class, its config, its arch).

    `config.architectures[0]` and not an Auto* mapping, for the same reason
    `load_bf16` insists on it: the module NAMES have to be the checkpoint's
    names, because that is the key a scale is filed under and the key
    `--input-scales-from` looks it up by.
    """
    require_torch('building the module tree')
    dtype = torch.bfloat16 if dtype is None else dtype
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(files)
    arch = (cfg.architectures or [None])[0]
    if arch is None or not hasattr(transformers, arch):
        raise SystemExit('%s: config names no usable architecture (%r)'
                         % (files, arch))
    cls = getattr(transformers, arch)
    with weightless():
        model = cls._from_config(cfg, dtype=dtype, attn_implementation=attn)
    model.eval()
    log('  class %s, dtype %s, attn %s, experts %s'
        % (arch, dtype, model.config._attn_implementation,
           getattr(model.config, '_experts_implementation', None)))
    return model, cfg, arch


# =============================================================================
# PUTTING ONE PIECE OF THE MODEL ON THE CARD, AND TAKING IT OFF AGAIN
# =============================================================================
class InRam(object):
    """The model is already in host RAM: a piece moves with `.to()`."""

    kind = 'in-ram'

    def __init__(self, device):
        self.device = device
        self.bytes = 0
        self.tensors = 0

    def to_device(self, module, prefix):
        module.to(self.device)

    def off_device(self, module, prefix):
        module.to('cpu')


class Streamed(object):
    """A piece of the model is READ when it is needed and dropped afterwards.

    Every parameter under `prefix` is looked up in the source by its own dotted
    name, which IS the checkpoint's name for everything except one thing:
    transformers 5.16 keeps a MoE layer's routed experts stacked, as
    `mlp.experts.gate_up_proj` [E, 2I, H] and `mlp.experts.down_proj` [E, H, I],
    while both the BF16 source and every published NVFP4 checkpoint store one
    tensor per expert per role.  The stack is assembled here one expert at a
    time, straight into the card, so the host never holds it - and it is
    assembled by transformers' OWN rule for this family (`conversion_mapping.py`
    maps glm4_moe onto the qwen2_moe converter: `MergeModulelist(dim=0)` over
    the experts, then `Concatenate(dim=1)` of gate before up), not by a guess.
    """

    kind = 'streamed'

    def __init__(self, src, device):
        self.src = src
        self.device = device
        self.bytes = 0
        self.tensors = 0
        self.reads = 0

    # -- bringing it in ---------------------------------------------------- #
    def to_device(self, module, prefix):
        for name, p in list(module.named_parameters(recurse=True,
                                                    remove_duplicate=False)):
            owner, leaf = _owner(module, name)
            t = self._tensor_for(_join(prefix, name), owner, leaf, p)
            owner._parameters[leaf] = nn.Parameter(t, requires_grad=False)
            self.tensors += 1
            self.bytes += t.numel() * t.element_size()
        for name, b in list(module.named_buffers(recurse=True,
                                                 remove_duplicate=False)):
            owner, leaf = _owner(module, name)
            full = _join(prefix, name)
            if full in self.src:
                owner._buffers[leaf] = self.src.read(full, self.device)
                self.reads += 1
            elif b.device.type == 'meta':
                raise SystemExit(
                    '%s is a buffer the source does not carry and the module '
                    'did not compute at build time, so there is nothing to put '
                    'in it.  Calibrating with it left empty would be a silent '
                    'wrong answer.' % full)
            else:
                owner._buffers[leaf] = b.to(self.device)

    def _tensor_for(self, full, owner, leaf, old):
        if full in self.src:
            self.reads += 1
            return self.src.read(full, self.device)
        return self._stack_experts(full, owner, leaf, old)

    def _stack_experts(self, full, owner, leaf, old):
        """One stacked expert Parameter, filled expert by expert on the card."""
        roles = {'gate_up_proj': ('gate_proj', 'up_proj'),
                 'up_proj': ('up_proj',),
                 'down_proj': ('down_proj',)}.get(leaf)
        if roles is None or old.dim() != 3:
            raise SystemExit(
                '%s: the source has no tensor of this name and it is not a '
                'stacked expert parameter this reader knows how to assemble '
                '(shape %s).  The source is %s.'
                % (full, tuple(old.shape), self.src.path))
        if getattr(owner, 'is_transposed', False):
            raise SystemExit('%s: this experts module stores its weights '
                             'transposed, which the assembler does not do '
                             'yet.' % full)
        if len(roles) > 1 and not getattr(owner, 'is_concatenated', True):
            raise SystemExit('%s: this experts module interleaves gate and up '
                             'rather than concatenating them, which the '
                             'assembler does not do yet.' % full)
        base = full.rsplit('.', 1)[0]
        n_exp = int(old.shape[0])
        width = int(old.shape[1]) // len(roles)
        out = torch.empty(tuple(old.shape), dtype=old.dtype, device=self.device)
        for e in range(n_exp):
            for j, role in enumerate(roles):
                nm = '%s.%d.%s.weight' % (base, e, role)
                if nm not in self.src:
                    raise SystemExit(
                        '%s needs %s and the source does not have it; the '
                        'source carries %d expert tensor(s) for this module.'
                        % (full, nm, e * len(roles) + j))
                out[e, j * width:(j + 1) * width].copy_(self.src.read(nm))
                self.reads += 1
        return out

    # -- and letting it go -------------------------------------------------- #
    def off_device(self, module, prefix):
        for name, p in list(module.named_parameters(recurse=True,
                                                    remove_duplicate=False)):
            owner, leaf = _owner(module, name)
            owner._parameters[leaf] = nn.Parameter(
                torch.empty(p.shape, dtype=p.dtype, device='meta'),
                requires_grad=False)
        for name, b in list(module.named_buffers(recurse=True,
                                                 remove_duplicate=False)):
            owner, leaf = _owner(module, name)
            if _join(prefix, name) in self.src:
                owner._buffers[leaf] = torch.empty(b.shape, dtype=b.dtype,
                                                   device='meta')
            else:
                owner._buffers[leaf] = b.to('cpu')


# =============================================================================
# THE ROUTED EXPERTS, WHICH ARE NOT nn.Linear
# =============================================================================
def fused_experts(root, name_of):
    """[(dotted name, module)] for every stacked-expert container under `root`.

    Found by SHAPE, not by class name: a module holding a 3-D `gate_up_proj`
    (or `up_proj`) Parameter beside a 3-D `down_proj` one is the stacked-expert
    container whatever its architecture calls it, and a model without one - any
    dense model, Qwen3.8-27B included - yields an empty list and is walked
    exactly as before.
    """
    out = []
    for m in root.modules():
        if id(m) not in name_of:
            continue
        gu = getattr(m, 'gate_up_proj', None)
        if not isinstance(gu, nn.Parameter):
            gu = getattr(m, 'up_proj', None)
        dn = getattr(m, 'down_proj', None)
        if (isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter)
                and gu.dim() == 3 and dn.dim() == 3):
            out.append((name_of[id(m)], m))
    return out


def walked_prefixes(model):
    """The dotted prefixes of everything the walk actually stages and hooks.

    The walk is not a forward over the whole model.  It runs the preamble
    (embeddings, rotary, and the final norm the tail needs), then the LONGEST
    `*.layers` stack one layer at a time, then the tail.  Whatever else the
    class builds is built, holds real source tensors, and is never handed an
    input - so no hook can see it either.  That is a DIFFERENT fact from "the
    class builds no module for this", which is why `unbuilt` reports the two
    apart.
    """
    name_of = {id(m): n for n, m in model.named_modules()}
    text_model, _layers, stack_name = find_stack(model)
    out = {stack_name + '.'}
    for attr in ('embed_tokens', 'rotary_emb', 'norm'):
        m = getattr(text_model, attr, None)
        if isinstance(m, nn.Module) and id(m) in name_of:
            out.add(name_of[id(m)] + '.')
    try:
        ol = model.get_output_embeddings()
    except Exception:                                           # noqa: BLE001
        ol = None
    if (ol is not None and isinstance(ol, nn.Linear) and id(ol) in name_of
            and hasattr(text_model, 'norm')):
        out.add(name_of[id(ol)] + '.')
    return tuple(sorted(out))


def unbuilt(model, src):
    """(source tensors no module claims, source tensors no WALKED module claims).

    TWO WAYS A SOURCE TENSOR GETS NO SCALE, AND THEY ARE NOT THE SAME FACT.

    NOT BUILT is where the DRAFT HEAD shows up, and it is worth stating rather
    than discovering.  transformers does not build one: `Glm4MoeForCausalLM`
    makes `num_hidden_layers` layers - 0..91 on GLM-4.7 - and puts
    `model.layers.92.*` in `_keys_to_ignore_on_load_unexpected`, and the
    Qwen3.8-27B class builds 1,184 parameters against its snapshot's 1,199, the
    15 missing ones being exactly `mtp.*`.  A module that does not exist cannot
    be hooked, and running the head anyway would mean writing its forward - the
    enorm / hnorm / eh_proj / shared_head chain - into a calibration tool, which
    is a SECOND implementation of the model and the one thing the probe-pass
    design exists to avoid.

    BUILT BUT NOT WALKED is the other one, and reporting only the first hid it:
    on `Qwen3_5ForConditionalGeneration` the whole `model.visual.*` tower is
    built, carries 333 source tensors, and sits outside the text stack the walk
    goes down, so it is as uncalibrated as the draft head while looking, to a
    count of unclaimed tensors, perfectly calibrated.  Nothing in this lane
    quantises a vision tower (`sglang_native.NON_LM_PREFIXES`), so this is an
    accounting fix and not a measurement one - but an accounting number that is
    wrong is worse than one that is missing.

    Either way the modules come out uncalibrated, `--uncalibrated` in the writer
    decides what happens to them, and both counts are printed so neither is ever
    a surprise.
    """
    have = set(n for n, _ in model.named_parameters(remove_duplicate=False))
    have |= set(n for n, _ in model.named_buffers(remove_duplicate=False))
    walked = walked_prefixes(model)
    exp = re.compile(r'^(.*\.experts)\.\d+\.(gate|up|down)_proj\.weight$')
    orphan, unwalked = [], []
    for n in src.idx:
        owner = n if n in have else None
        if owner is None:
            m = exp.match(n)
            if m:
                for leaf in ('.gate_up_proj', '.up_proj', '.down_proj'):
                    if m.group(1) + leaf in have:
                        owner = m.group(1) + leaf
                        break
        if owner is None:
            orphan.append(n)
        elif not owner.startswith(walked):
            unwalked.append(n)
    return sorted(orphan), sorted(unwalked)


def _gated(mod, proj):
    """The down_proj input: the module's own gating, applied to its own output."""
    fn = getattr(mod, '_apply_gate', None)
    if fn is not None and getattr(mod, 'has_gate', True):
        return fn(proj)
    if not getattr(mod, 'has_gate', True):
        return mod.act_fn(proj)
    gate, up = proj.chunk(2, dim=-1)
    return mod.act_fn(gate) * up


def attach_experts(name, mod, recs, n_tiers, device, use_hist, stride):
    """Hook one stacked-expert module so every expert gets its own two amax.

    WHY A PRE-HOOK AND NOT A REPLACEMENT FORWARD.  The experts module is handed
    the batch and the routing and loops over the experts inside itself, and
    which loop it runs depends on `config._experts_implementation` - eager,
    grouped_mm, batched_mm, a hub kernel.  Copying one of those into this tool
    would make the measurement depend on a transformers internal and would go
    stale silently.  So the module's own forward runs untouched and this reads
    only what it was GIVEN: the rows routed to expert e are the input to that
    expert's gate_proj and up_proj, and its down_proj input is the gated product
    of them, which is recomputed here with the module's own slice of the stacked
    weight and its own `_apply_gate`.  That is the only arithmetic repeated, it
    is two thirds of one expert's forward, and it buys independence from which
    kernel the model chose.

    gate_proj and up_proj get ONE reduction between them because they read one
    tensor - which is also why SGLang collapses the pair into a single `w13`
    scale, and why every published checkpoint carries them equal.
    """
    gu = getattr(mod, 'gate_up_proj', None)
    fused_gate = isinstance(gu, nn.Parameter)
    if not fused_gate:
        gu = mod.up_proj
    # THE TWO REFUSALS `Streamed._stack_experts` MAKES ABOUT THE WEIGHTS, AND A THIRD FOR BIASES,
    # made here about the ARITHMETIC.  This hook recomputes the down_proj input
    # from the module's own slice of the stacked weight, so an experts module
    # that stores that weight transposed would be multiplied the wrong way
    # round, one that interleaves gate and up rather than concatenating them
    # would be gated on the wrong halves, and one carrying per-expert biases
    # (gpt_oss) would have them dropped from the product.  All three are silent
    # wherever the shapes happen to line up, which is exactly why they are
    # refused here rather than left to be noticed in a scale.
    if getattr(mod, 'is_transposed', False):
        raise SystemExit(
            '%s: this experts module stores its weights transposed, so the '
            'down_proj input recomputed here would be the wrong orientation.  '
            'Refusing rather than measuring a wrong number.' % name)
    if fused_gate and not getattr(mod, 'is_concatenated', True):
        raise SystemExit(
            '%s: this experts module interleaves gate and up rather than '
            'concatenating them, so the gating applied here would pair the '
            'wrong halves.  Refusing rather than measuring a wrong number.'
            % name)
    if getattr(mod, 'has_bias', False):
        raise SystemExit(
            '%s: this experts module carries per-expert biases, which the '
            'gate_up product recomputed here does not add, so its down_proj '
            'scale would be measured on the wrong tensor.  Refusing rather '
            'than measuring a wrong number.' % name)
    n_exp = int(getattr(mod, 'num_experts', gu.shape[0]))
    trio = []
    for e in range(n_exp):
        trio.append(tuple(
            recs.setdefault('%s.%d.%s' % (name, e, role),
                            Recorder(n_tiers, device, use_hist, stride))
            for role in ('gate_proj', 'up_proj', 'down_proj')))

    def hook(m, args, kwargs):
        hs = args[0] if args else kwargs.get('hidden_states')
        idx = (args[1] if len(args) > 1 else
               kwargs.get('top_k_index', kwargs.get('topk_indices')))
        if not torch.is_tensor(hs) or not torch.is_tensor(idx):
            raise SystemExit(
                '%s: cannot tell which argument is the routing (%d positional, '
                'keywords %s), so the per-expert scales would be silently '
                'wrong.' % (name, len(args), sorted(kwargs)))
        h = hs.reshape(-1, hs.shape[-1])
        top = idx.reshape(h.shape[0], -1)
        w = getattr(m, 'gate_up_proj', None)
        if not isinstance(w, nn.Parameter):
            w = m.up_proj
        k = top.shape[1]
        tok = (torch.arange(h.shape[0], device=h.device)
               .unsqueeze(1).expand(-1, k).reshape(-1))
        eid = top.reshape(-1)
        # An expert-parallel sentinel is an id past the end of the table; it is
        # not an expert and it must not become one.
        keep = eid < n_exp
        eid, tok = eid[keep], tok[keep]
        order = torch.argsort(eid, stable=True)
        eid, tok = eid[order], tok[order]
        counts = torch.bincount(eid, minlength=n_exp).tolist()
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            rows = tok[start:start + n]
            start += n
            rg, ru, rd = trio[e]
            xe = h.index_select(0, rows)
            observe_into((rg, ru), xe)
            observe_into((rd,), _gated(m, torch.nn.functional.linear(xe, w[e])))
            for r in (rg, ru, rd):
                r.add_tokens(n)
            del xe

    return mod.register_forward_pre_hook(hook, with_kwargs=True), [
        r for t in trio for r in t]


def emitted_names(model):
    """Every module name this tool will file a scale under, known BEFORE the walk.

    The same two enumerations `hook_module` runs - the nn.Linears of a decoder
    layer, and the per-expert trio of a stacked-expert container - plus the LM
    head the tail calibrates.  Having the list up front is what lets the naming
    contract be checked at startup instead of after a walk that costs hours.
    """
    name_of = {id(m): n for n, m in model.named_modules()}
    text_model, layers, _stack = find_stack(model)
    out = set()
    for layer in layers:
        for nm, _m in linear_children(layer, name_of):
            out.add(nm)
        for nm, mod in fused_experts(layer, name_of):
            gu = getattr(mod, 'gate_up_proj', None)
            if not isinstance(gu, nn.Parameter):
                gu = mod.up_proj
            n_exp = int(getattr(mod, 'num_experts', gu.shape[0]))
            for e in range(n_exp):
                for role in ('gate_proj', 'up_proj', 'down_proj'):
                    out.add('%s.%d.%s' % (nm, e, role))
    try:
        ol = model.get_output_embeddings()
    except Exception:                                           # noqa: BLE001
        ol = None
    if (ol is not None and isinstance(ol, nn.Linear) and id(ol) in name_of
            and hasattr(text_model, 'norm')):
        out.add(name_of[id(ol)])
    return sorted(out)


def check_naming(model, arch, cfg, log_fn=log):
    """Refuse before the walk if a name it would emit is one the writer cannot
    look up.  Returns (the names, the arch key, the scale-group ids).

    THE CONTRACT THIS TURNS INTO AN ASSERTION.  A measured scale is worth
    nothing unless `sglang_write.harvest_act_amax` can find it, and what that
    looks a scale up by is the DISK name `sglang_native.DiskNames.classify`
    accepts for the architecture.  Several of the names emitted here - the
    per-expert trio above all - are spelled out by hand, because the arch table
    maps a GGUF tensor onto an HF module and not the other way round, so there
    is nothing to derive them FROM.  The hand-written spelling is therefore
    CHECKED against that table rather than trusted: every name the walk will
    file a scale under is classified before the walk starts, and a name the
    table does not claim stops the run in a second instead of after hours.

    Two families are excluded, both because nothing in this lane quantises
    them: the vision tower (`sglang_native.NON_LM_PREFIXES`, which has no ARCHS
    entry at all) and the MTP/draft head, whose prefixes are derived from the
    arch table itself rather than named here.  An architecture no table claims
    is reported and not refused - the tool still measures, there is simply no
    contract to check it against.
    """
    import sglang_native as SN
    names = emitted_names(model)
    key = SN.arch_from_hf_config({'architectures': [arch],
                                  'model_type': getattr(cfg, 'model_type', None)})
    if key is None:
        log_fn('  naming: no sglang_native arch table claims %s, so the %d '
               'module name(s) this emits cannot be checked against one'
               % (arch, len(names)))
        return names, None, set()
    dn = SN.DiskNames(SN.ARCHS[key])
    skip = tuple(SN.NON_LM_PREFIXES) + tuple(dn.mtp_prefixes(names))
    bad, gids = [], set()
    for n in names:
        if n.startswith(skip):
            continue
        c = dn.classify(n)
        if c is None:
            bad.append(n)
        elif c[1] is not None:
            gids.add(c[1])
    if bad:
        raise SystemExit(
            'the naming contract with sglang_write.harvest_act_amax is broken: '
            '%d of the %d module name(s) this would emit are not names '
            'sglang_native.DiskNames(%r).classify accepts, e.g. %s.  A scale '
            'filed under a name the writer cannot look up does nothing, so '
            'this refuses before the walk rather than after it.'
            % (len(bad), len(names), key, ', '.join(bad[:4])))
    log_fn('  naming: every one of %d module name(s) this emits is classified '
           'by the %s table, into %d scale group(s)'
           % (len(names), key, len(gids)))
    return names, key, gids


def expert_coverage(recs, tier):
    """{experts module: how many tokens its thinnest and median expert saw}.

    A running max over forty tokens is a different object from one over forty
    thousand, and the difference is invisible in the scale itself.  So it is
    stated: per MoE layer, the min, median and max routed-token count across its
    experts, and the number of experts that saw none at all.
    """
    by_layer = {}
    for nm, r in recs.items():
        if not nm.endswith('.down_proj'):
            continue
        m = re.match(r'^(.*\.experts)\.(\d+)\.down_proj$', nm)
        if m:
            by_layer.setdefault(m.group(1), []).append(int(r.tokens[tier]))
    out = {}
    for k in sorted(by_layer):
        v = sorted(by_layer[k])
        out[k] = {'experts': len(v), 'min': v[0], 'median': q(v, .5),
                  'max': v[-1], 'total': sum(v),
                  'idle': sum(1 for x in v if x == 0)}
    return out


def coverage_summary(cover):
    """The per-layer table said in one paragraph, for CALIBRATION.json."""
    if not cover:
        return None
    mins = sorted(c['min'] for c in cover.values())
    meds = sorted(c['median'] for c in cover.values())
    maxs = sorted(c['max'] for c in cover.values())
    return {'moe_layers': len(cover),
            'experts_per_layer': max(c['experts'] for c in cover.values()),
            'idle_experts': sum(c['idle'] for c in cover.values()),
            'routed_tokens': sum(c['total'] for c in cover.values()),
            'layer_min': {'min': mins[0], 'median': q(mins, .5), 'max': mins[-1]},
            'layer_median': {'min': meds[0], 'median': q(meds, .5), 'max': meds[-1]},
            'layer_max': {'min': maxs[0], 'median': q(maxs, .5), 'max': maxs[-1]}}


# =============================================================================
# THE PROBE PASS
# =============================================================================
def capture_preamble(text_model, layers, types, input_ids):
    """Run the model's own preamble and keep what a decoder layer is given.

    Every layer is temporarily an identity that records its keyword arguments,
    so the rotary embeddings, the position ids and - the part worth the trouble
    - the attention mask THE MODEL chose for each layer type are captured rather
    than reconstructed.  Nothing raises: the pass runs to the end through 64
    identities, which costs one layernorm.
    """
    got = {'pe': None, 'pid': None, 'hidden': None}
    masks = {}
    saved = [l.forward for l in layers]

    def make(idx):
        def probe(hidden_states, **kw):
            if got['hidden'] is None:
                got['hidden'] = hidden_states
                got['pe'] = kw.get('position_embeddings')
                got['pid'] = kw.get('position_ids')
            masks.setdefault(types[idx], kw.get('attention_mask'))
            return hidden_states
        return probe

    try:
        for i, l in enumerate(layers):
            l.forward = make(i)
        with torch.no_grad():
            text_model(input_ids=input_ids, use_cache=False)
    finally:
        for l, f in zip(layers, saved):
            l.forward = f
    if got['hidden'] is None:
        raise SystemExit('probe pass never reached a decoder layer')
    return got['hidden'], got['pe'], got['pid'], masks


def _to(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, device) for v in x)
    return x


# =============================================================================
# THE WALK
# =============================================================================
def hook_module(root, name_of, recs, n_tiers, device, use_hist, stride):
    """Hook everything under `root` that carries a calibrated activation.

    ([handles], [the Recorders this subtree writes into]).  The nn.Linears are
    the ordinary case; a stacked-expert container is the one that needs its own
    reader, and both end up filing under the checkpoint's own names.
    """
    handles, mine = [], []
    for nm, mod in linear_children(root, name_of):
        r = recs.setdefault(nm, Recorder(n_tiers, device, use_hist, stride))
        mine.append(r)
        handles.append(mod.register_forward_pre_hook(
            (lambda rec: (lambda _m, args: rec.observe(args[0])))(r)))
    for nm, mod in fused_experts(root, name_of):
        h, rs = attach_experts(nm, mod, recs, n_tiers, device, use_hist, stride)
        handles.append(h)
        mine.extend(rs)
    return handles, mine


def calibrate(model, ids, device, tiers, batch=8, use_hist=True, stride=1,
              progress=True, stage=None):
    """{module name: Recorder} over `ids` [n_seq, seq_len], layer by layer.

    `tiers` is a list of sequence counts, ascending, each a whole number of
    batches; tier t is the first tiers[t] sequences.  `stage` decides where a
    piece of the model comes from when it is wanted: `InRam` moves it off the
    host, `Streamed` reads it out of the source and drops it again.
    """
    stage = stage or InRam(device)
    name_of = {id(m): n for n, m in model.named_modules()}
    text_model, layers, stack_name = find_stack(model)
    types = layer_types(text_model, len(layers))
    n_seq, seq_len = ids.shape
    nb = (n_seq + batch - 1) // batch
    tier_batches = [int(math.ceil(t / float(batch))) for t in tiers]

    log('  stack %s: %d layers, types %s, weights %s' % (
        stack_name, len(layers), sorted(set(types)), stage.kind))
    log('  %d sequences x %d tokens, batch %d -> %d forwards per layer'
        % (n_seq, seq_len, batch, nb))

    # -- preamble: embeddings and rotary on the card, hidden states kept there -
    head = [text_model.embed_tokens] if hasattr(text_model, 'embed_tokens') else []
    if hasattr(text_model, 'rotary_emb'):
        head.append(text_model.rotary_emb)
    if hasattr(text_model, 'norm'):
        head.append(text_model.norm)      # the probe pass runs the model's tail
    for m in head:
        stage.to_device(m, name_of[id(m)])
    hidden, pes, pids, masks = [], [], [], []
    t0 = time.time()
    for b in range(nb):
        chunk = ids[b * batch:(b + 1) * batch].to(device)
        h, pe, pid, mk = capture_preamble(text_model, layers, types, chunk)
        hidden.append(h)
        pes.append(_to(pe, device))
        pids.append(_to(pid, device))
        masks.append({k: _to(v, device) for k, v in mk.items()})
        del chunk
    for m in head:
        stage.off_device(m, name_of[id(m)])
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    log('  preamble: %d batches in %.1f s, masks for %s'
        % (nb, time.time() - t0, sorted(masks[0])))

    # -- the layers ---------------------------------------------------------- #
    recs = {}
    t0 = time.time()
    for i, layer in enumerate(layers):
        stage.to_device(layer, name_of[id(layer)])
        handles, mine = hook_module(layer, name_of, recs, len(tiers), device,
                                    use_hist, stride)
        with torch.no_grad():
            for b in range(nb):
                active = [t for t, tb in enumerate(tier_batches) if b < tb]
                for r in mine:
                    r.active = active
                out = layer(hidden[b], position_embeddings=pes[b],
                            attention_mask=masks[b].get(types[i]),
                            position_ids=pids[b],
                            past_key_values=None, use_cache=False)
                hidden[b] = out[0] if isinstance(out, tuple) else out
        for h in handles:
            h.remove()
        stage.off_device(layer, name_of[id(layer)])
        if progress and (i % 8 == 0 or i == len(layers) - 1):
            el = time.time() - t0
            done = (i + 1) / float(len(layers))
            log('    layer %3d/%d  %6.1f s elapsed  %5.2f s/layer  eta %5.0f s'
                '  %d modules' % (i + 1, len(layers), el, el / (i + 1),
                                  el / done - el, len(recs)))
    wall = time.time() - t0

    # -- the tail: the LM head's input is the final norm's output ------------- #
    head_lin = None
    try:
        ol = model.get_output_embeddings()
        if isinstance(ol, nn.Linear) and id(ol) in name_of:
            head_lin = name_of[id(ol)]
    except Exception:
        head_lin = None
    if head_lin and hasattr(text_model, 'norm'):
        nrm = text_model.norm
        stage.to_device(nrm, name_of[id(nrm)])
        r = recs.setdefault(head_lin, Recorder(len(tiers), device, use_hist, stride))
        with torch.no_grad():
            for b in range(nb):
                r.active = [t for t, tb in enumerate(tier_batches) if b < tb]
                r.observe(nrm(hidden[b]))
        stage.off_device(nrm, name_of[id(nrm)])
        log('  tail: %s calibrated from %s.norm' % (head_lin, stack_name.rsplit('.', 1)[0]))
    elif head_lin:
        log('  tail: %s NOT calibrated (no final norm found)' % head_lin)

    del hidden, pes, pids, masks
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return recs, wall


def monolithic(model, ids, device, batch=1, use_hist=False):
    """The same statistic the ordinary way: whole model, one forward per batch.

    Only for --check-monolithic and --selftest.  It is the reference the
    layer-by-layer walk has to reproduce, and it needs the whole model on one
    device - which is why it is a check on a model that fits and not the way the
    walk works.
    """
    name_of = {id(m): n for n, m in model.named_modules()}
    text_model, layers, _ = find_stack(model)
    recs = {}
    handles, _mine = hook_module(layers, name_of, recs, 1, device, use_hist, 1)
    model.to(device)
    try:
        with torch.no_grad():
            for b in range(0, ids.shape[0], batch):
                text_model(input_ids=ids[b:b + batch].to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return recs


# =============================================================================
# TEXT
# =============================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load_tokens(text_path, tokenizer, n_tokens, seq_len, allow_eval_text=False):
    """[n_seq, seq_len] of the FIRST n_tokens of the text, and its provenance.

    The first N tokens and not a sample: it makes the budgets nested, so one
    walk snapshots 8k, 32k and 128k of the same text and the three sets differ
    only in how much of it was seen.
    """
    real = os.path.realpath(text_path)
    digest = sha256_file(real)
    base = os.path.basename(real)
    if not allow_eval_text and (base in FORBIDDEN_BASENAMES or digest in FORBIDDEN_SHA256):
        raise SystemExit(
            '%s is the evaluation text.  Calibrating on it would fit the '
            'clipping threshold to the very tokens the perplexity is reported '
            'over.  Pick another text.' % base)
    want_chars = n_tokens * 32 + (1 << 20)
    with open(real, 'rb') as f:
        raw = f.read(want_chars)
    text = raw.decode('utf-8', 'ignore')
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids = enc['input_ids']
    if len(ids) < n_tokens:
        raise SystemExit('%s yields %d tokens, %d asked for' % (base, len(ids), n_tokens))
    t = torch.tensor(ids[:n_tokens], dtype=torch.long).reshape(-1, seq_len)
    return t, {'path': base, 'sha256': digest, 'bytes_read': len(raw),
               'tokens': int(n_tokens), 'seq_len': int(seq_len),
               'sequences': int(t.shape[0])}


# =============================================================================
# OUTPUT
# =============================================================================
def save_safetensors(path, tensors, metadata):
    """A minimal `model-*.safetensors` of F32 scalars, written by hand.

    By hand because the only thing that has to be true of it is that
    `sglang_st.index_dir` can read it, and that is a header and a payload.
    """
    header, off = {}, 0
    blobs = []
    for k in sorted(tensors):
        b = np.float32(tensors[k]).tobytes()
        header[k] = {'dtype': 'F32', 'shape': [], 'data_offsets': [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(',', ':')).encode('utf-8')
    pad = (-len(hb)) % 8
    hb += b' ' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def emit(out_dir, recs, tier, meta):
    os.makedirs(out_dir, exist_ok=True)
    mods, empty = {}, []
    for nm in sorted(recs):
        r = recs[nm]
        amax = float(r.amax[tier].item())
        # An amax of exactly zero is not a small scale, it is NO measurement -
        # a routed expert this budget never sent a token to, or a module the
        # walk never reached.  Writing it would hand the checkpoint an
        # `input_scale` of 0 and the kernel a division by it, so the module is
        # left OUT and the writer's `--uncalibrated` decides what to do about
        # it, which is the estate's existing answer to a missing scale.
        if amax == 0.0:
            empty.append(nm)
            continue
        e = {'amax': amax,
             'input_scale': amax / E4M3_MAX,
             'input_scale_nvfp4': amax / NVFP4_WS2_DENOM,
             'elements': int(r.count[tier])}
        if r.tokens[tier]:
            e['routed_tokens'] = int(r.tokens[tier])
        e.update(r.percentiles(tier))
        mods[nm] = e
    cover = expert_coverage(recs, tier)
    if empty:
        log('  %d module(s) had no activation at this budget and are NOT '
            'written, e.g. %s' % (len(empty), empty[:3]))
    doc = {'meta': meta, 'modules': mods}
    if cover:
        doc['experts'] = cover
    with open(os.path.join(out_dir, 'scales.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    save_safetensors(
        os.path.join(out_dir, 'model-00001-of-00001.safetensors'),
        {nm + '.input_scale': mods[nm]['input_scale'] for nm in mods},
        {'convention': 'input_scale = activation amax / 448 (per-tensor FP8); '
                       'no weight_scale_2 accompanies it, so a reader recovers '
                       'amax as input_scale * 448',
         'modules': str(len(mods)),
         'tokens': str(meta.get('tokens')),
         'text': str(meta.get('text', {}).get('path'))})
    with open(os.path.join(out_dir, 'CALIBRATION.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return mods


# =============================================================================
# READING SCALES BACK - OURS OR ANYBODY'S
# =============================================================================
def _decode_scalar(dtype, buf, name):
    """One published scalar, in whatever dtype its OWN header says it is.

    The header is not decoration.  modelopt writes F32 today, but a checkpoint
    that stored its `input_scale` as BF16 and was read as F32 would not raise -
    it would pair up two neighbouring scalars into one plausible-looking wrong
    number.  So the dtype decides, and one this does not know stops the read.
    BF16 has no numpy dtype, so it is widened by its own definition: those 16
    bits ARE the top half of the F32.
    """
    width = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8}.get(dtype)
    if width is None:
        raise SystemExit(
            '%s is stored as %s, and this reader knows F32, F16, BF16 and F64.  '
            'Decoding it as one of those anyway would produce a number that '
            'looks like a scale and is not one.' % (name, dtype))
    if len(buf) != width:
        raise SystemExit(
            '%s: the header says one %s scalar (%d bytes) but the data range '
            'holds %d bytes; the file does not describe itself.'
            % (name, dtype, width, len(buf)))
    if dtype == 'BF16':
        u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
        return float(u.view(np.float32).reshape(-1)[0])
    np_of = {'F32': np.float32, 'F16': np.float16, 'F64': np.float64}[dtype]
    return float(np.frombuffer(buf, np_of).reshape(-1)[0])


def read_amax(path, prefer_json=True):
    """{module: activation amax} from our scales.json, our safetensors, or a
    published checkpoint.

    A published checkpoint states its convention per module the way
    sglang_write.py reads it: an `input_scale` with a `weight_scale_2` beside it
    is NVFP4's (amax/2688), one without is per-tensor FP8's (amax/448).
    `scales.json` holds the amax unrounded; the safetensors holds it as the
    F32 `input_scale` the writer reads, which is what a build sees.  `compare`
    passes prefer_json=False when only one side has the json, so both sides are
    read the same way and a rounding difference is not reported as a measurement.
    """
    if (prefer_json and os.path.isdir(path)
            and os.path.exists(os.path.join(path, 'scales.json'))):
        d = json.load(open(os.path.join(path, 'scales.json'), encoding='utf-8'))
        return {k: v['amax'] for k, v in d['modules'].items()}
    if path.endswith('.json'):
        d = json.load(open(path, encoding='utf-8'))
        d = d.get('modules', d)
        return {k: (v['amax'] if isinstance(v, dict) else float(v)) for k, v in d.items()}
    files = sorted(glob.glob(os.path.join(path, 'model-*.safetensors')))
    if not files:
        raise SystemExit('%s: no scales.json and no model-*.safetensors' % path)
    idx = {}
    for p in files:
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != '__metadata__':
                idx[k] = (p, v, 8 + n)
    # One open per SHARD, not one per scale: a published MoE checkpoint carries
    # 43,364 of them and reopening the file for each is minutes of nothing.
    out = {}
    per_file = {}
    for k, (p, v, base) in idx.items():
        if k.endswith('.input_scale'):
            per_file.setdefault(p, []).append((k, v, base))
    for p, items in per_file.items():
        with open(p, 'rb') as f:
            for k, v, base in sorted(items, key=lambda it: it[1]['data_offsets'][0]):
                a, b = v['data_offsets']
                f.seek(base + a)
                val = _decode_scalar(v['dtype'], f.read(b - a), k)
                pre = k[: -len('.input_scale')]
                out[pre] = val * (NVFP4_WS2_DENOM
                                  if (pre + '.weight_scale_2') in idx
                                  else E4M3_MAX)
    return out


def family(name):
    """`layers.31.mlp.gate_proj` -> `layers.N.mlp.gate_proj`."""
    return re.sub(r'\.\d+\.', '.N.', name)


def q(xs, p):
    if not xs:
        return float('nan')
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def diff_stats(a_recs, b_recs, tier=0):
    """How far two sets of Recorders disagree, module by module.

    An amax is a maximum: it does not average a mistake away, so `worst` is the
    number that matters and the quantiles are there to say whether a large
    `worst` is one module or all of them.
    """
    keys = sorted(set(a_recs) & set(b_recs))
    ds, worst, wm = [], 0.0, ''
    for k in keys:
        x = float(a_recs[k].amax[tier].item())
        y = float(b_recs[k].amax[tier].item())
        d = abs(x - y) / max(abs(y), 1e-30)
        ds.append(d)
        if d > worst:
            worst, wm = d, k
    return {'n': len(keys), 'worst': worst, 'worst_module': wm,
            'median': q(ds, .5), 'p95': q(ds, .95),
            'only_a': len(set(a_recs) - set(b_recs)),
            'only_b': len(set(b_recs) - set(a_recs))}


def compare(a_path, b_path, top=12, by_family=True):
    has_json = [os.path.isdir(x) and os.path.exists(os.path.join(x, 'scales.json'))
                for x in (a_path, b_path)]
    like = not (has_json[0] != has_json[1])
    if not like:
        log('  one side has scales.json and the other does not: both are read from '
            'their safetensors (the F32 input_scale a build sees), like for like')
    A, B = read_amax(a_path, prefer_json=like), read_amax(b_path, prefer_json=like)
    common = sorted(set(A) & set(B))
    log('')
    log('  %s' % a_path)
    log('    vs %s' % b_path)
    log('  modules: %d here, %d there, %d in common (%d only here, %d only there)'
        % (len(A), len(B), len(common), len(set(A) - set(B)), len(set(B) - set(A))))
    if not common:
        return {}
    ratios = {m: A[m] / B[m] for m in common if B[m] > 0}
    r = list(ratios.values())
    lr = [math.log(x) for x in r]
    stats = {'n': len(r), 'median': q(r, .5), 'p5': q(r, .05), 'p95': q(r, .95),
             'min': min(r), 'max': max(r),
             'geomean': math.exp(sum(lr) / len(lr)),
             'within_10pct': sum(1 for x in r if 0.9 <= x <= 1.1) / len(r),
             'within_25pct': sum(1 for x in r if 0.8 <= x <= 1.25) / len(r)}
    log('  ratio (ours/theirs): median %.4f  p5 %.4f  p95 %.4f  min %.4f  max %.4f'
        % (stats['median'], stats['p5'], stats['p95'], stats['min'], stats['max']))
    log('  geomean %.4f   within +-10%%: %.1f%%   within +-25%%: %.1f%%'
        % (stats['geomean'], 100 * stats['within_10pct'], 100 * stats['within_25pct']))
    # The RELATIVE DIFFERENCE, which is the number to quote when the two sets
    # are supposed to be the SAME measurement - two sources of one model's
    # weights, the same text, the same budget.  Zero is the claim; anything else
    # has to be explained, so the module that is furthest out is named.
    rel = max(((abs(A[m] - B[m]) / max(abs(B[m]), 1e-30)), m) for m in common)
    stats['worst_rel'] = rel[0]
    # When every module agrees the max still names one, and a module name beside
    # a zero reads as a finding when it is only the last key among the zeros.
    # `diff_stats` reports '' for that case; so does this, and both print '-'.
    stats['worst_rel_module'] = rel[1] if rel[0] > 0.0 else ''
    stats['identical'] = sum(1 for m in common if A[m] == B[m])
    log('  worst relative difference %.6g on %s (%d of %d modules identical)'
        % (rel[0], stats['worst_rel_module'] or '-', stats['identical'],
           len(common)))
    worst = sorted(ratios.items(), key=lambda kv: -abs(math.log(kv[1])))[:top]
    log('  furthest apart:')
    for m, x in worst:
        log('    %-62s %8.4f   ours %10.4g  theirs %10.4g' % (m[-62:], x, A[m], B[m]))
    if by_family:
        fam, cov = {}, {}
        for m, x in ratios.items():
            fam.setdefault(family(m), []).append(x)
            # COVERAGE, the quantity that predicts damage.  The failure is
            # one-sided - an amax that is too small clips and one that is too
            # large costs almost nothing - so what matters per module is not
            # |log ratio| but how far each set falls SHORT of the largest amax
            # either of them measured.  A running max is monotone in evidence,
            # so max(ours, theirs) is the least-bad stand-in for the truth.
            hi = max(A[m], B[m])
            cov.setdefault(family(m), []).append((A[m] / hi, B[m] / hi))
        log('  by family (ratio ours/theirs, then coverage of max(ours,theirs)):')
        log('    %-44s %6s %8s %8s %8s %8s %8s %6s'
            % ('family', 'n', 'median', 'p5', 'p95', 'cov ours', 'cov thrs',
               '2x low'))
        for f in sorted(fam):
            v, c = fam[f], cov[f]
            ga = math.exp(sum(math.log(max(x, 1e-30)) for x, _ in c) / len(c))
            gb = math.exp(sum(math.log(max(y, 1e-30)) for _, y in c) / len(c))
            log('    %-44s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %6d'
                % (f[-44:], len(v), q(v, .5), q(v, .05), q(v, .95), ga, gb,
                   sum(1 for x, _ in c if x < 0.5)))
        allc = [p for v in cov.values() for p in v]
        stats['coverage_ours'] = math.exp(
            sum(math.log(max(x, 1e-30)) for x, _ in allc) / len(allc))
        stats['coverage_theirs'] = math.exp(
            sum(math.log(max(y, 1e-30)) for _, y in allc) / len(allc))
        log('    %-44s %6d %8s %8s %8s %8.4f %8.4f %6d'
            % ('ALL', len(allc), '', '', '', stats['coverage_ours'],
               stats['coverage_theirs'], sum(1 for x, _ in allc if x < 0.5)))
    stats['worst'] = [(m, x, A[m], B[m]) for m, x in worst]
    return stats


# =============================================================================
# SELFTEST
# =============================================================================
def selftest():
    """The layer-by-layer walk against the ordinary forward, on a real model.

    A small one, built and randomly initialised here, but a real transformers
    model running its real code: if the walk feeds a layer anything other than
    what the monolithic forward feeds it, the per-module amax will not match,
    and an amax is a maximum - it does not average a mistake away.
    """
    require_torch('--selftest')
    import transformers
    ok = [0, 0]

    def chk(cond, what):
        ok[1] += 1
        ok[0] += bool(cond)
        log('  %s %s' % ('PASS' if cond else 'FAIL', what))

    chk(abs(NVFP4_WS2_DENOM - 2688.0) < 1e-12, 'NVFP4 denominator is 6*448 = 2688')
    chk(abs(E4M3_MAX - 448.0) < 1e-12, 'FP8 E4M3 max is 448')

    # the header this tool writes is the header sglang_st.index_dir reads
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_safetensors(os.path.join(td, 'model-00001-of-00001.safetensors'),
                         {'a.input_scale': 0.25, 'b.input_scale': 1.5},
                         {'convention': 'test'})
        back = read_amax(td)
        chk(abs(back['a'] - 0.25 * E4M3_MAX) < 1e-6
            and abs(back['b'] - 1.5 * E4M3_MAX) < 1e-4,
            'safetensors round trip recovers amax = input_scale * 448')
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import sglang_st as ST
            idx = ST.index_dir(td, 'model-*.safetensors')
            chk(sorted(idx) == ['a.input_scale', 'b.input_scale']
                and idx['a.input_scale'][1]['dtype'] == 'F32'
                and tuple(idx['a.input_scale'][1]['shape']) == ()
                and abs(float(ST.read(idx, 'b.input_scale').reshape(-1)[0]) - 1.5) < 1e-6,
                "sglang_st reads it: F32 scalars named '<module>.input_scale'")
        except ImportError:
            log('  SKIP sglang_st not importable from here')

    # -- a published input_scale is decoded in ITS OWN dtype ----------------- #
    def hand_written(path, entries):
        """A safetensors written byte by byte, so the header can say anything."""
        head, off, blobs = {}, 0, []
        for k in sorted(entries):
            dt, b = entries[k]
            head[k] = {'dtype': dt, 'shape': [], 'data_offsets': [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        hb = json.dumps(head, separators=(',', ':')).encode('utf-8')
        hb += b' ' * ((-len(hb)) % 8)
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(hb)))
            f.write(hb)
            for b in blobs:
                f.write(b)

    with tempfile.TemporaryDirectory() as td:
        one = 0.25          # exact in F32, F16 and BF16, so the check is exact
        hand_written(os.path.join(td, 'model-00001-of-00001.safetensors'), {
            'f32.input_scale': ('F32', np.float32(one).tobytes()),
            'f16.input_scale': ('F16', np.float16(one).tobytes()),
            'bf16.input_scale': ('BF16', np.uint16(
                np.uint32(np.float32(one).view(np.uint32)) >> 16).tobytes())})
        back = read_amax(td)
        chk(sorted(back) == ['bf16', 'f16', 'f32']
            and all(back[k] == one * E4M3_MAX for k in back),
            'a published input_scale is decoded in the dtype its own header '
            'states (F32, F16, BF16)')
        hand_written(os.path.join(td, 'model-00002-of-00002.safetensors'),
                     {'i32.input_scale': ('I32', np.int32(3).tobytes())})
        try:
            read_amax(td)
            refused = False
        except SystemExit:
            refused = True
        chk(refused, 'an input_scale in a dtype this cannot decode is refused, '
                     'not read as float32 anyway')

    # -- nothing is named as the worst when nothing differs ------------------ #
    with tempfile.TemporaryDirectory() as td:
        save_safetensors(os.path.join(td, 'model-00001-of-00001.safetensors'),
                         {'x.input_scale': 0.5, 'y.input_scale': 0.25},
                         {'convention': 'test'})
        same = compare(td, td, top=2, by_family=False)
        chk(same['worst_rel'] == 0.0 and same['worst_rel_module'] == '',
            'two identical sets name no worst module, because a module name '
            'beside a zero reads as a finding')

    # THE SELFTEST DOES NOT HELP ITSELF TO A GPU.  Without --gpu nothing has
    # masked CUDA_VISIBLE_DEVICES, so `cuda` here means device 0 - somebody
    # else's production card on this box.  The toy models are a few hundred KB
    # and run on the CPU in seconds, so the CPU is the default and a card is
    # touched only when one was named, in which case it is the only one this
    # process can see.
    dev = torch.device('cuda' if (_GPU is not None and torch.cuda.is_available())
                       else 'cpu')
    log('  device: %s%s' % (dev.type,
                            '' if _GPU is None else '  (--gpu %s)' % _GPU))
    chk(dev.type == 'cpu' or _GPU is not None,
        'the selftest runs on the CPU unless --gpu named a card')
    cfgs = []
    cfgs.append(('llama', transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=512)))
    if hasattr(transformers, 'Qwen3_5TextConfig'):
        try:
            cfgs.append(('qwen3_5 hybrid', transformers.Qwen3_5TextConfig(
                hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                vocab_size=256, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16,
                layer_types=['linear_attention', 'full_attention',
                             'linear_attention', 'full_attention'])))
        except Exception as e:                                  # noqa: BLE001
            log('  SKIP qwen3_5 toy config: %s' % e)
    # The MoE case, which is the one nn.Linear hooks cannot see: three layers,
    # the first dense (first_k_dense_replace=1) so both kinds of MLP are walked,
    # and six experts with two of them routed per token so some are thin.
    if hasattr(transformers, 'Glm4MoeConfig'):
        try:
            cfgs.append(('glm4_moe fused experts', transformers.Glm4MoeConfig(
                hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, vocab_size=128, n_routed_experts=6, n_shared_experts=1,
                num_experts_per_tok=2, first_k_dense_replace=1, n_group=1,
                topk_group=1, max_position_embeddings=128)))
        except Exception as e:                                  # noqa: BLE001
            log('  SKIP glm4_moe toy config: %s' % e)

    for label, cfg in cfgs:
        torch.manual_seed(0)
        cls = transformers.AutoModelForCausalLM
        try:
            model = cls.from_config(cfg, dtype=torch.float32)
        except Exception as e:                                  # noqa: BLE001
            log('  SKIP %s: %s' % (label, e))
            continue
        model.eval()
        vocab = int(getattr(cfg, 'vocab_size', 256))
        ids = torch.randint(0, vocab, (4, 32), dtype=torch.long)
        try:
            mono = monolithic(model, ids, dev, batch=2)
            model.to('cpu')
            seq, _ = calibrate(model, ids, dev, [4], batch=2, use_hist=False,
                               progress=False)
        except Exception as e:                                  # noqa: BLE001
            log('  FAIL %s: %s' % (label, e))
            ok[1] += 1
            continue
        d = diff_stats(seq, mono)
        chk(d['n'] == len(mono) and d['n'] > 0,
            '%s: the walk sees every module the forward sees (%d)' % (label, d['n']))
        chk(d['worst'] == 0.0,
            '%s: per-module amax identical to the monolithic forward '
            '(worst %.3g on %s)' % (label, d['worst'], d['worst_module'] or '-'))
        # The histogram is OFF in this walk, which is what --no-percentiles
        # does; `elements` is a fact about the measurement either way.
        chk(seq and all(r.count[0] > 0 for r in seq.values()),
            '%s: every module records how many elements its amax was taken '
            'over, with the histogram off (%d module(s))' % (label, len(seq)))
        n_exp = len([k for k in mono if re.search(r'\.experts\.\d+\.', k)])
        if n_exp:
            per = int(getattr(cfg, 'num_local_experts', 0)) * 3
            moe = int(cfg.num_hidden_layers) - int(cfg.first_k_dense_replace)
            chk(n_exp == per * moe,
                '%s: every expert of every MoE layer is recorded, three names '
                'each (%d = %d experts x 3 x %d layers)'
                % (label, n_exp, per // 3, moe))
            gu = [k for k in seq if k.endswith('.gate_proj')
                  and re.search(r'\.experts\.\d+\.', k)]
            chk(all(float(seq[k].amax[0]) ==
                    float(seq[k[:-len('gate_proj')] + 'up_proj'].amax[0])
                    for k in gu),
                '%s: gate_proj and up_proj of one expert carry ONE number, as '
                'the w13 scale group requires' % label)
            chk(all(seq[k].tokens[0] > 0 for k in gu),
                '%s: every expert was routed at least one token, so no amax is '
                'a zero standing in for a measurement' % label)

        # -- the same walk, STREAMED off a checkpoint on disk ---------------- #
        with tempfile.TemporaryDirectory() as td:
            try:
                model.save_pretrained(td, safe_serialization=True)
                src = Source(td, log_fn=lambda *a: None)
                sk, _cfg2, _arch2 = build_skeleton(td, dtype=torch.float32)
                st, _ = calibrate(sk, ids, dev, [4], batch=2, use_hist=False,
                                  progress=False, stage=Streamed(src, dev))
            except Exception as e:                              # noqa: BLE001
                log('  FAIL %s streamed: %s' % (label, e))
                ok[1] += 1
                del model
                continue
        ds = diff_stats(st, mono)
        chk(ds['n'] == len(mono) and ds['only_b'] == 0
            and sorted(st) == sorted(seq),
            '%s: the streamed walk sees exactly the modules the whole-model '
            'walk sees (%d)' % (label, ds['n']))
        chk(ds['worst'] == 0.0,
            '%s: streamed off a save_pretrained checkpoint, per-module amax '
            'identical to the whole-model forward (worst %.3g on %s)'
            % (label, ds['worst'], ds['worst_module'] or '-'))
        chk(all(p.device.type == 'meta'
                for _n, p in sk.named_parameters(remove_duplicate=False)),
            '%s: the streamed model holds no weights when the walk is over'
            % label)

        # -- the naming contract, and the two refusals, on the arch that has -- #
        # -- a table: the toy is spelled `model.layers.N.*` exactly as GLM-4.7 -#
        if label.startswith('glm4_moe'):
            import sglang_native as SN
            try:
                names, key, gids = check_naming(model, 'Glm4MoeForCausalLM',
                                                cfg, log_fn=lambda *_a: None)
            except SystemExit as e:                             # noqa: BLE001
                names, key, gids = [], None, set()
                log('    %s' % e)
            dn = SN.DiskNames(SN.ARCHS['glm4_moe'])
            n_exp = int(cfg.n_routed_experts)
            moe = int(cfg.num_hidden_layers) - int(cfg.first_k_dense_replace)
            chk(key == 'glm4_moe' and len(names) == n_exp * 3 * moe
                + 7 * int(cfg.num_hidden_layers) + 1
                and all(dn.classify(n) is not None for n in names),
                '%s: every module name this emits (%d) is one '
                'sglang_native.DiskNames.classify accepts, so the writer can '
                'look every scale up' % (label, len(names)))
            chk(bool(gids)
                and dn.classify('model.layers.1.mlp.experts.0.gate_up_proj') is None
                and dn.classify('model.layers.1.mlp.experts.0.gate_proj') is not None,
                '%s: and the assertion has teeth - the FUSED per-expert '
                'spelling is rejected, the per-role one is not' % label)

            name_of = {id(m): n for n, m in model.named_modules()}
            fx = fused_experts(model, name_of)
            fired = []
            for flag in ('is_transposed', 'is_concatenated', 'has_bias'):
                nm, em = fx[0]
                had, old = hasattr(em, flag), getattr(em, flag, None)
                setattr(em, flag, False if flag == 'is_concatenated' else True)
                try:
                    h, _rs = attach_experts(nm, em, {}, 1, dev, False, 1)
                    h.remove()
                    fired.append(False)
                except SystemExit:
                    fired.append(True)
                if had:
                    setattr(em, flag, old)
                else:
                    delattr(em, flag)
            chk(fx and all(fired),
                '%s: an experts module that is transposed, interleaved or '
                'biased is REFUSED, because the down_proj input recomputed '
                'here would be wrong and silently so' % label)

            # what the class does not build, and what it builds outside the walk
            model.side_tower = nn.Linear(4, 4)
            probe = argparse.Namespace(idx={
                'model.layers.0.self_attn.q_proj.weight': 1,
                'model.layers.0.mlp.down_proj.weight': 1,
                'side_tower.weight': 1,
                'model.layers.99.mlp.down_proj.weight': 1})
            orph, unw = unbuilt(model, probe)
            del model.side_tower
            chk(orph == ['model.layers.99.mlp.down_proj.weight']
                and unw == ['side_tower.weight'],
                '%s: uncalibrated is reported as TWO kinds - not built at all, '
                'and built but never staged by the walk' % label)
        del model, sk

    log('  %d/%d' % (ok[0], ok[1]))
    return 0 if ok[0] == ok[1] else 1


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Measure the static activation scales NVFP4 and per-tensor '
                    'FP8 need, on a calibration text of your own choosing.')
    ap.add_argument('--gpu', help='CUDA device index; every other GPU is masked '
                                  'out of this process before torch is imported. '
                                  '`cpu` hides them all and walks on the CPU. '
                                  'A walk never picks a device on its own.')
    ap.add_argument('--source', help='the BF16 weights: an HF snapshot '
                                     'directory, or a BF16 GGUF SPECIAL_SPLIT '
                                     '(directory or first shard)')
    ap.add_argument('--hf-files', default=None,
                    help="directory holding the checkpoint's small text files "
                         '(config.json, the tokenizer). Required when --source '
                         'is a GGUF split, which does not carry them; it is '
                         'also where the tokenizer comes from when given')
    ap.add_argument('--gguf-companion', action='append', default=[],
                    help='an extra GGUF split to read beside --source '
                         '(repeatable); found automatically when it sits beside '
                         'the main one')
    ap.add_argument('--in-ram', action='store_true',
                    help='load the whole model into host RAM with '
                         'from_pretrained instead of streaming it off the '
                         'source. Snapshot only, and it needs RAM for the whole '
                         'BF16 model; --check-monolithic requires it')
    ap.add_argument('--text', help='calibration text (never the evaluation text)')
    ap.add_argument('--tokens', type=int, default=32768,
                    help='total calibration tokens (default 32768)')
    ap.add_argument('--seq-len', type=int, default=512,
                    help='tokens per sequence (default 512, the imatrix ctx)')
    ap.add_argument('--batch', type=int, default=8, help='sequences per forward')
    ap.add_argument('--snapshots', default=None,
                    help='comma-separated token budgets to emit from the one '
                         'walk, e.g. 8192,32768,131072 (default: --tokens)')
    ap.add_argument('--out', default=None, metavar='DIR',
                    help='directory to write THE scale set into (one snapshot: '
                         '--tokens); the form the guide uses')
    ap.add_argument('--out-root', default=None,
                    help='directory to write sets under, one per snapshot')
    ap.add_argument('--name', default=None,
                    help='set name; each snapshot lands in <out-root>/<name>-<tokens>')
    ap.add_argument('--no-percentiles', action='store_true',
                    help='skip the log2 histogram (amax only, a little faster)')
    ap.add_argument('--hist-stride', type=int, default=1,
                    help='subsample stride for the histogram only; the amax is '
                         'always over every element')
    ap.add_argument('--attn', default='sdpa', help="attn_implementation (default sdpa)")
    ap.add_argument('--check-monolithic', type=int, default=0, metavar='N',
                    help='prove the walk on THIS model: run the first N '
                         'sequences both ways on --check-device and require the '
                         'same amax, then measure what the GPU walk does to the '
                         'same numbers')
    ap.add_argument('--check-device', default='cpu',
                    help='device the two ways are compared on (default cpu: the '
                         'whole model has to fit on it at once)')
    ap.add_argument('--compare', action='append', default=[],
                    help='a scales set (or published checkpoint) to compare')
    ap.add_argument('--against', default=None, help='what to compare it with')
    ap.add_argument('--top', type=int, default=12, help='rows in the outlier table')
    ap.add_argument('--allow-eval-text', action='store_true',
                    help=argparse.SUPPRESS)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.compare:
        if not a.against:
            raise SystemExit('--compare needs --against')
        for c in a.compare:
            compare(c, a.against, top=a.top)
        return 0

    if not (a.source and a.text and (a.out or (a.out_root and a.name))):
        raise SystemExit('need --source, --text, and --out DIR (or --out-root '
                         'and --name for several snapshots of one walk)')
    if a.out and (a.out_root or a.name or (a.snapshots and ',' in a.snapshots)):
        raise SystemExit('--out names one directory for one set: use --out-root '
                         'and --name to emit several snapshots of one walk')

    snaps = ([int(x) for x in a.snapshots.split(',')] if a.snapshots else [a.tokens])
    snaps = sorted(set(snaps))
    if snaps[-1] > a.tokens:
        raise SystemExit('a snapshot (%d) exceeds --tokens (%d)' % (snaps[-1], a.tokens))
    per_batch = a.batch * a.seq_len
    bad = [s for s in snaps if s % per_batch]
    if bad:
        raise SystemExit('snapshot %s is not a whole number of batches (%d tokens); '
                         'adjust --batch or --seq-len' % (bad, per_batch))
    if a.tokens % per_batch:
        raise SystemExit('--tokens (%d) must be a whole number of batches (%d '
                         'tokens); adjust --batch or --seq-len' % (a.tokens, per_batch))
    if a.check_monolithic and not a.in_ram:
        raise SystemExit(
            '--check-monolithic runs the whole model in one forward and needs '
            'it all on one device at once, which the streamed walk deliberately '
            'never has.  Add --in-ram (snapshot sources only).')
    # Beside it, and for the same reason: a refusal belongs before the work, not
    # after the tokenizer has loaded and the whole budget has been tokenised.
    # `looks_like_split` is the predicate `Source` itself decides `kind` with, so
    # this cannot disagree with the source that is opened a moment later.
    if a.in_ram and GG.looks_like_split(a.source):
        raise SystemExit(
            '--in-ram loads the model with transformers\' own from_pretrained, '
            'which reads an HF snapshot and not a GGUF split.  Drop --in-ram '
            'and the split streams like any other source.')

    require_torch('a calibration walk')
    if a.gpu is None:
        raise SystemExit('pass --gpu N (the CUDA device to use; every other one '
                         'is hidden from this process), or --gpu cpu to walk on '
                         'the CPU.  A walk never picks a device on its own.')
    dev = torch.device('cuda' if (str(a.gpu).lower() != 'cpu'
                                  and torch.cuda.is_available()) else 'cpu')
    if dev.type == 'cuda':
        log('device: %s (%d visible)' % (torch.cuda.get_device_name(0),
                                         torch.cuda.device_count()))

    import transformers
    src = Source(a.source, a.hf_files, a.gguf_companion)
    log('tokenizer: %s' % src.files)
    tok = transformers.AutoTokenizer.from_pretrained(src.files)
    ids, tmeta = load_tokens(a.text, tok, a.tokens, a.seq_len, a.allow_eval_text)
    log('text: %s sha256 %s -> %d tokens in %d sequences'
        % (tmeta['path'], tmeta['sha256'][:16], tmeta['tokens'], tmeta['sequences']))

    t0 = time.time()
    if a.in_ram:
        model, cfg, arch = load_bf16(a.source, attn=a.attn)
        stage = InRam(dev)
    else:
        model, cfg, arch = build_skeleton(src.files, attn=a.attn)
        stage = Streamed(src, dev)
    t_load = time.time() - t0
    log('model ready in %.1f s (%s)' % (t_load, stage.kind))

    # What the class does not build cannot be hooked, and neither can what it
    # builds outside the stack this walks; say which and how many before the
    # walk rather than leaving either to be found in the module count.
    orphan, unwalked = unbuilt(model, src)
    orphan_fam = sorted(set(family(n) for n in orphan))
    unwalked_fam = sorted(set(family(n) for n in unwalked))
    if orphan:
        log('  %d source tensor(s) belong to no module this class builds, so '
            'they go uncalibrated: %s' % (len(orphan), ', '.join(orphan_fam[:6])))
    if unwalked:
        log('  %d source tensor(s) belong to modules the class builds but the '
            'walk never stages, so they go uncalibrated too: %s'
            % (len(unwalked), ', '.join(unwalked_fam[:6])))

    # And the naming contract, checked structurally rather than by hand: a name
    # this would emit that the writer cannot look up stops the run here.
    check_naming(model, arch, cfg)

    tiers = [s // a.seq_len for s in snaps]
    recs, wall = calibrate(model, ids, dev, tiers, batch=a.batch,
                           use_hist=not a.no_percentiles, stride=a.hist_stride,
                           stage=stage)
    tps = a.tokens / wall
    log('walk: %d tokens in %.1f s = %.1f tok/s (%d modules)'
        % (a.tokens, wall, tps, len(recs)))
    if stage.kind == 'streamed':
        log('  streamed %d parameter(s) / %d source tensor(s), %.1f GiB, '
            'nothing kept' % (stage.tensors, stage.reads,
                              stage.bytes / float(1 << 30)))

    checked = None
    if a.check_monolithic:
        n = a.check_monolithic
        cdev = torch.device(a.check_device)
        t1 = time.time()
        log('check-monolithic: %d sequences, both ways, on %s' % (n, a.check_device))
        mono = monolithic(model, ids[:n], cdev, batch=1)
        model.to('cpu')
        same, _ = calibrate(model, ids[:n], cdev, [n], batch=min(a.batch, n),
                            use_hist=False, progress=False)
        d_same = diff_stats(same, mono)
        log('  walk vs forward, both on %s: worst %.3g (%s), median %.3g, '
            'p95 %.3g, over %d modules%s'
            % (a.check_device, d_same['worst'], d_same['worst_module'] or '-',
               d_same['median'], d_same['p95'], d_same['n'],
               '  IDENTICAL' if d_same['worst'] == 0.0 else ''))
        cross = None
        if dev != cdev:
            log('  and the same %d sequences walked on %s, for the kernel drift'
                % (n, dev))
            other, _ = calibrate(model, ids[:n], dev, [n], batch=min(a.batch, n),
                                 use_hist=False, progress=False)
            cross = diff_stats(other, mono)
            log('  %s walk vs %s forward: worst %.3g (%s), median %.3g, p95 %.3g'
                % (dev, a.check_device, cross['worst'],
                   cross['worst_module'] or '-', cross['median'], cross['p95']))
        checked = {'sequences': n, 'device': a.check_device,
                   'walk_vs_forward_same_device': d_same,
                   'walk_on_%s_vs_forward' % dev.type: cross,
                   'seconds': time.time() - t1}

    outs = []
    for t, s in zip(range(len(tiers)), snaps):
        d = a.out if a.out else os.path.join(a.out_root, '%s-%d' % (a.name, s))
        cover = expert_coverage(recs, t)
        meta = {'tool': os.path.basename(__file__),
                'source': a.source, 'architecture': arch,
                'text': dict(tmeta, tokens=s, sequences=s // a.seq_len),
                'tokens': s, 'seq_len': a.seq_len, 'batch': a.batch,
                'snapshot_of': {'walk_tokens': a.tokens, 'snapshots': snaps},
                'torch': torch.__version__,
                'transformers': transformers.__version__,
                'device': (torch.cuda.get_device_name(0) if dev.type == 'cuda'
                           else 'cpu'),
                'attn_implementation': a.attn,
                'experts_implementation': getattr(
                    model.config, '_experts_implementation', None),
                'weights': stage.kind,
                'histogram': (not a.no_percentiles),
                'hist_stride': a.hist_stride,
                'walk_seconds': wall, 'walk_tokens_per_s': tps,
                'load_seconds': t_load,
                'check_monolithic': checked,
                'expert_coverage': coverage_summary(cover),
                'uncalibrated': {
                    'source_tensors': len(orphan) + len(unwalked),
                    'families': sorted(set(orphan_fam + unwalked_fam)),
                    'not_built': {
                        'source_tensors': len(orphan), 'families': orphan_fam,
                        'why': 'the model class builds no module for these, so '
                               'no hook can see them; the draft head is the '
                               'usual case and every published checkpoint '
                               'drops it too'},
                    'built_but_not_walked': {
                        'source_tensors': len(unwalked),
                        'families': unwalked_fam,
                        'why': 'the class builds these but the walk never '
                               'stages them - it goes down the longest decoder '
                               'stack with its preamble and tail, so a vision '
                               'tower is built, never given an input, and '
                               'never hooked'}},
                'convention': {'fp8': 'input_scale = amax / 448',
                               'nvfp4': 'input_scale = amax / 2688',
                               'safetensors': 'fp8'}}
        meta.update(src.provenance())
        mods = emit(d, recs, t, meta)
        outs.append((d, len(mods)))
        log('wrote %s  (%d modules)' % (d, len(mods)))
        if cover:
            mins = sorted(c['min'] for c in cover.values())
            meds = sorted(c['median'] for c in cover.values())
            log('  experts: %d MoE layer(s); thinnest expert anywhere %d tokens, '
                'median of the per-layer minima %d, median of the per-layer '
                'medians %d' % (len(cover), mins[0], q(mins, .5), q(meds, .5)))

    if a.against:
        for d, _n in outs:
            compare(d, a.against, top=a.top)
    return 0


if __name__ == '__main__':
    sys.exit(main())
