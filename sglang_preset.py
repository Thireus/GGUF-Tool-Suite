#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** sglang_preset.py is the one-line preset: it fills in      **#
#** every flag the SGLang path needs so you only pass a size. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Sep-11-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.       ₒₙₑ 𝒻ₗₐ𝓰, 𝓏ₑᵣₒ ₜₕₒᵤ𝓰ₕₜₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#
"""What `--speed-profile sglang` decides for you, and how each decision is made.

THE PROBLEM THIS FILE SOLVES.  The canonical SGLang command in docs/sglang.md
used to be fourteen flags long, and every one of them was either a constant
(`--harmonization-technique 3`), a fact about the ENGINE the user cannot be
expected to know (`--gpu-quants sgl_nvfp4 sgl_fp8 sgl_fp8_pb_wo sgl_bf16`), a
fact about the MODEL that the tool can read off its own inputs
(`--harmonize-tensors`, the BF16 pins, the architecture) or a number nobody
could guess (`--prefill-budget 0.8366153327562383`).  None of those is a
preference.  A preference is the byte target, and that is the one thing the user
does know.  So:

    ../../quant_assign.py kld_results.csv --speed-profile sglang \
        --gpu-tensors-max-size 17410000000B

EVERY DEFAULT HERE IS OVERRIDABLE BY PASSING THE FLAG.  The preset fills in a
flag only when it is absent from `sys.argv`, so the full explicit command
reproduces exactly what it always did - that is the regression gate in
docs/sglang.md SS6, and it is checked on every change to this file.

FOUR DECISIONS ARE MADE FROM DATA RATHER THAN FROM A TABLE, and this file is
where each is derived:

  1. THE ARCHITECTURE, from the tensor names (`sglang_native.detect_arch`).
  2. NEXTN/EAGLE COMPATIBILITY, from MTP/NEXTN tensor names in the GGUF map and,
     when the GGUF map does not carry the draft head, from the HF source's own
     config.  See `resolve_nextn()` for why the embedding is the only tensor
     that matters.
  3. THE PREFILL BUDGET, from the FILL rule: the strictest budget in the sweep
     whose recipe actually spends the size cap.  `fill_pick()` is the rule; the
     sweep that feeds it is one assigner pass per point and lives in
     `quant_assign.py`, because only the assigner can produce a point.
  4. THE DECODE BUDGET: none.  It is reported against the pool floor and not
     capped - see section 5 for why an automatic one was removed.

CPU only.  Opens no GPU, loads no weights, needs no torch, downloads nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sglang_native as SGN                                    # noqa: E402


# =============================================================================
# 1. THE POOL
# =============================================================================
#
# The four types a `modelopt_mixed` checkpoint can be built from with the
# writer this branch ships.  The other three members of
# `sglang_native.SGLANG_POOL_ORDER` are deliberately out:
#
#   sgl_nvfp4a16   same bytes as NVFP4 and 1.81x less damage, so NO BYTE BUDGET
#                  CAN EVER BUY IT - it is only reachable by a zero-byte upgrade
#                  pass, which is the parked frontier branch's speed axis.
#                  Including it here would put a type in the pool that the
#                  single-recipe optimiser can never choose.
#   sgl_int4_g128  a DIFFERENT CONTAINER (compressed-tensors).  Naming it makes
#                  every sgl_fp8* illegal in the same checkpoint
#                  (`container_for`), and compressed-tensors has no embedding
#                  method at all, so the embedding lands at BF16 and the recipe
#                  cannot get smaller.
#   sgl_mxfp8      8.25 bpw against sgl_fp8's 8.0 for the same kernel class and
#                  no measured quality gain (deg 0.005332 vs 0.005301).
#
# Pass --gpu-quants explicitly to use any of them.
PRESET_POOL: Tuple[str, ...] = ('sgl_nvfp4', 'sgl_fp8', 'sgl_fp8_pb_wo', 'sgl_bf16')

# The preset's non-negotiable constants.  Each is a property of this ENGINE
# path, not a preference: harmonisation technique 3 is the one that assigns one
# qtype to a whole fused group (the others equalise losses, which is the older
# and softer half of the feature and does not express the constraint).
PRESET_HARMONIZATION_TECHNIQUE = 3


# =============================================================================
# 2. WHERE THE MODEL'S OWN FILES LIVE
# =============================================================================
#
# THE download.conf TRAP.  `tensor_downloader.sh:115` reads
# `${GGUF_DOWNLOAD_CONF:-<suite root>/download.conf}`, and the suite root's
# `download.conf` is a SYMLINK the owner repoints by hand at whichever model is
# being worked on.  Run the assigner on GLM-4.7 while that symlink points at
# Qwen and you silently get Qwen's `tensors.map`, a plausible recipe and no
# error - observed once, 2026-09-06 07:37, in the glm47 notes.
#
# Under the preset there is no symlink to get wrong: the model is the directory
# the user's own calibration CSV sits in, so the config is resolved from that.

def model_dir_for(csv_path: str, degradation_csv: Optional[str] = None) -> str:
    """The model folder: the directory holding the calibration CSV.

    `<model>/kld_results.csv` is what the canonical command names, so its parent
    IS the model folder.  When the caller passes only a degradation CSV that
    sits in `<model>/group0/`, its grandparent is used instead.
    """
    if csv_path:
        d = os.path.dirname(os.path.abspath(csv_path))
        if d:
            return d
    if degradation_csv:
        d = os.path.dirname(os.path.abspath(degradation_csv))
        if os.path.basename(d).startswith('group'):
            return os.path.dirname(d)
        return d
    return os.getcwd()


def tensor_names_for(model_dir: str, csv_path: str):
    """(names, where they came from): every tensor name this model has.

    `<model>/group0/tensors.bf16.map` is the COMPLETE list - norms, 1-D tensors
    and, when the conversion kept it, the MTP/NEXTN draft head.  The calibration
    CSV's header is the assignable subset, used when no map is on disk.

    ONE FUNCTION, because two used to disagree: `--explain` read only the CSV
    header and reported `926 tensor name(s)` and `3 MTP/NEXTN tensor(s)` on
    GLM-4.7 where the run itself said 1761 and 6.  Same decisions, different
    numbers, in the tool advertised as printing every decision.
    """
    m = os.path.join(model_dir, 'group0', 'tensors.bf16.map')
    if os.path.isfile(m):
        try:
            names = [r[0] for r in SGN.read_bf16_map(m)]
            if names:
                return names, m
        except (OSError, ValueError, IndexError):
            pass
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as fh:
            header = fh.readline().rstrip('\n').split(',')
        return [n for n in header[1:] if n], csv_path
    except OSError:
        return [], None


def download_conf_for(model_dir: str) -> Optional[str]:
    """`<model folder>/download.conf`, or None."""
    p = os.path.join(model_dir, 'download.conf')
    return p if os.path.isfile(p) else None


_CONF_NAME_RE = re.compile(r'^\s*MODEL_NAME\s*=\s*["\']?([^"\'\s#]+)', re.M)


def model_name_from_conf(conf_path: Optional[str]) -> Optional[str]:
    """MODEL_NAME out of a download.conf, without sourcing it."""
    if not conf_path or not os.path.isfile(conf_path):
        return None
    try:
        with open(conf_path, 'r', encoding='utf-8', errors='replace') as fh:
            m = _CONF_NAME_RE.search(fh.read())
    except OSError:
        return None
    return m.group(1) if m else None


# The `sgl_*` rows live in their OWN file, `group0/kld_results_sglang.csv`.
# They are a different KIND of number from the rest of `group0/kld_results.csv`:
# the GGUF rows are weight-only llama-perplexity measurements at full coverage,
# while four of the seven `sgl_*` rows carry an ACTIVATION term the engine's
# kernels add at runtime and every one of them is divided by the share of the
# model's sensitivity mass its checkpoint covers.  Keeping them in one file
# invited exactly one mistake - reading `sgl_nvfp4` 0.0669 as "2.4x worse than
# q3_K 0.0280" - which the cross-calibration had to undo.
# Separate files, one kind of number each, and the preset says which it read.
DEG_SGLANG_BASENAME = 'kld_results_sglang.csv'
DEG_MERGED_BASENAME = 'kld_results.csv'

# THE MEASURED PER-FORMAT ROWS, CARRIED BY THE TOOL.  Measured in SGLang on
# Qwen3.8-27B on Sep-07-2026: the in-engine KLD of a checkpoint that is entirely
# one format, each divided by the share of that model's sensitivity mass the
# checkpoint covers.  They are the same seven rows as
# `models/Qwen3.8-27B/group0/kld_results_sglang.csv`, and the selftest holds the
# constant and that file identical so the two cannot drift.
#
# ON EVERY MODEL THESE ROWS SET THE PREDICTED NUMBER AND NOT THE ASSIGNMENT.
# The pool is three rungs (4.5, 8 and 16 bits) and the assigner only ever asks
# which of two types is worse, so a per-model table moves the degradation the
# footer prints and leaves the recipe exactly where it was.  MEASURED over a
# 27-size sweep on Qwen3.8-27B with in-engine numbers (docs/sglang.md, "Adding a
# new model"): the measured rows, the same rows interpolated from the model's
# own GGUF table and the round-to-nearest estimate give the same recipe at 25 of
# 27 sizes, and the two that differ measure the same in the engine.  So no model
# has to benchmark its own.
DEG_BUILTIN_MODEL = 'Qwen3.8-27B'
DEG_BUILTIN_ROWS: Tuple[Tuple[str, float], ...] = (
    ('sgl_bf16', 0.000000),
    ('sgl_fp8', 0.005301),
    ('sgl_fp8_pb_wo', 0.005291),
    ('sgl_mxfp8', 0.005332),
    ('sgl_int4_g128', 0.054767),
    ('sgl_nvfp4a16', 0.036930),
    ('sgl_nvfp4', 0.066930),
)

# How those rows name themselves on stderr and in the recipe footer.
DEG_BUILTIN_LABEL = 'built-in rows measured on ' + DEG_BUILTIN_MODEL

# One paragraph, on stderr, whenever a FILE prices the pool instead of the rows
# above.  The file is honoured; the warning is there because measuring one is
# work nobody owes, and a person who did it should know what it bought.
DEG_FILE_WARNING = (
    'this per-format degradation table is not needed. A measured table changes '
    'only the predicted degradation the footer prints, not the recipe '
    '(measured on ' + DEG_BUILTIN_MODEL + ' across 27 sizes), so there is no '
    'need to benchmark one for a model. It is used as asked, and the footer '
    'names it.')


def builtin_degradation_values() -> Dict[str, float]:
    """The built-in rows as the assigner's own {qtype: degradation} mapping."""
    return {q: v for q, v in DEG_BUILTIN_ROWS}


def committed_degradation_csv() -> Optional[str]:
    """The suite's own copy of those rows on disk, when the checkout has it.

    `models/<DEG_BUILTIN_MODEL>/group0/kld_results_sglang.csv`, falling back to
    that folder's `kld_results.csv` for a tree still in the pre-split layout.
    NO RUN READS IT - the rows are built in, which is what lets the preset work
    from a model folder that can see no `models/` directory at all.  The
    selftest reads it, to hold it equal to `DEG_BUILTIN_ROWS`.
    """
    for base in (DEG_SGLANG_BASENAME, DEG_MERGED_BASENAME):
        p = os.path.join(_HERE, 'models', DEG_BUILTIN_MODEL, 'group0', base)
        if csv_has_sgl_rows(p):
            return p
    return None


def degradation_source_for(model_dir: str):
    """(path, why) for the per-FORMAT degradation table; `None` means built-in.

    THE SEARCH ORDER, most specific first:

      1. `<model>/group0/kld_results_sglang.csv` - this model's own measured
         rows, in the dedicated file.  Used, and warned about
         (`DEG_FILE_WARNING`): all it can change is the predicted number.
      2. `<model>/group0/kld_results.csv` when it still carries `sgl_*` rows -
         the pre-split layout, or a user's own merged table.  Used the same way
         and with the same warning, so a tree that has not been migrated is
         never silently ignored.
      3. `DEG_BUILTIN_ROWS`, and no file at all.  A model folder holding its own
         `kld_results.csv` and `group0/tensors.bf16.map` is the whole input:
         nothing is read from `models/`, so the preset runs from anywhere.

    There is no failure case any more, which is the point - this table is not
    something a new model has to produce.  It is still not interchangeable with
    the per-TENSOR sensitivity CSV the user passes positionally: that one IS per
    model, and it is the one that shapes the recipe.
    """
    own = os.path.join(model_dir, 'group0', DEG_SGLANG_BASENAME)
    if csv_has_sgl_rows(own):
        return own, ("this model's own "
                     + os.path.join('group0', DEG_SGLANG_BASENAME)
                     + ', which overrides the ' + DEG_BUILTIN_LABEL)
    merged = os.path.join(model_dir, 'group0', DEG_MERGED_BASENAME)
    if csv_has_sgl_rows(merged):
        return merged, ("this model's own "
                        + os.path.join('group0', DEG_MERGED_BASENAME)
                        + ' (the pre-split layout: the sgl_* rows are still '
                          'merged into the GGUF table), which overrides the '
                        + DEG_BUILTIN_LABEL)
    return None, ('no file needed - the rows are a property of the FORMAT, '
                  'measured in SGLang on ' + DEG_BUILTIN_MODEL + ', and on '
                  'every model they set the predicted degradation and not the '
                  'assignment')


def degradation_csv_for(model_dir: str) -> Optional[str]:
    """Just the path from `degradation_source_for`, for callers that want it.

    `None` is not a failure here: it is the built-in rows, which is the normal
    answer for a model that has no table of its own.
    """
    return degradation_source_for(model_dir)[0]


def csv_has_sgl_rows(path: Optional[str]) -> bool:
    """Does this degradation CSV name at least two of the pool's own types?

    Two, not one: a table with a single `sgl_*` row cannot express a CHOICE, and
    a run against it would silently put the whole model on that one type.
    """
    if not path or not os.path.isfile(path):
        return False
    seen = set()
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                q = line.split(',', 1)[0].strip()
                if q.startswith('sgl_'):
                    seen.add(q)
    except OSError:
        return False
    return len(seen) >= 2


# =============================================================================
# 3. THE HF SOURCE, AND WHAT IT KNOWS THAT A GGUF MAP DOES NOT
# =============================================================================
#
# A GGUF conversion may or may not carry the MTP/NEXTN draft head.  MEASURED on
# this box, 2026-09-06:
#
#   GLM-4.7          `models/GLM-4.7/group0/tensors.bf16.map` HAS the head:
#                    6 `blk.92.nextn.*` rows, 3 of them also in kld_results.csv.
#   Qwen3.8-27B      `models/Qwen3.8-27B/group0/tensors.bf16.map` has NONE - the
#                    MTP layer was split into a separate companion GGUF
#                    (an `mtp-Qwen3.8-27B` companion split) and never reached
#                    the suite's map.  Its HF config says `mtp_num_hidden_layers: 1`
#                    and `mtp_use_dedicated_embeddings: false`.
#
# So neither source alone is sufficient and the detection reads both.  The HF
# source is found by exact model name in the local HF cache; nothing is
# downloaded and a missing cache is not an error, it is one of the three answers
# `resolve_nextn()` can give.

# Config keys that DECLARE a speculative draft head, across the families the
# suite has seen.  A key counts only when its value is truthy: several models
# ship `num_nextn_predict_layers: 0`, which means "there is no head".
HF_NEXTN_KEYS = (
    'num_nextn_predict_layers',      # glm4_moe, deepseek_v3
    'mtp_num_hidden_layers',         # qwen3_5
    'num_mtp_layers',
    'n_future_tokens',
    'num_speculative_tokens',
)

# Keys that describe MoE routing.  Read for the same reason and from the same
# file: a routed expert is streamed only top_k/n_experts of the time, and
# without that the decode budget prices an MoE as if it were dense.
HF_MOE_TOPK_KEYS = ('num_experts_per_tok', 'moe_top_k', 'num_experts_per_token')
HF_MOE_N_KEYS = ('n_routed_experts', 'num_experts', 'num_local_experts',
                 'moe_num_experts')


def hf_cache_roots() -> List[str]:
    """The HF hub caches to look in, most specific first."""
    roots = []
    for env in ('HF_HUB_CACHE',):
        v = os.environ.get(env)
        if v:
            roots.append(v)
    home = os.environ.get('HF_HOME')
    roots.append(os.path.join(home, 'hub') if home
                 else os.path.join(os.path.expanduser('~'), '.cache',
                                   'huggingface', 'hub'))
    return [r for r in roots if os.path.isdir(r)]


def find_hf_source(model_name: Optional[str],
                   roots: Optional[Sequence[str]] = None) -> Optional[str]:
    """A local HF snapshot of EXACTLY this model, or None.

    Matched on the repo NAME only (`models--<org>--<name>`), never on a prefix:
    `Qwen3.8-27B-NVFP4` and `Qwen3.8-27B-FP8` are quantised derivatives whose
    configs may not describe the source at all, and a wrong config here would
    turn a compatibility decision into a guess.  A snapshot qualifies only if it
    has a `config.json` to read.
    """
    if not model_name:
        return None
    best = None
    for root in (roots if roots is not None else hf_cache_roots()):
        for repo in sorted(glob.glob(os.path.join(root, 'models--*'))):
            leaf = os.path.basename(repo)[len('models--'):]
            if '--' not in leaf:
                continue
            if leaf.rsplit('--', 1)[-1] != model_name:
                continue
            for snap in sorted(glob.glob(os.path.join(repo, 'snapshots', '*'))):
                if os.path.isfile(os.path.join(snap, 'config.json')):
                    best = best or snap
    return best


def _walk_config(obj, out: Dict[str, object], prefix: str = '') -> None:
    """Flatten a config.json, so `text_config.mtp_num_hidden_layers` is found."""
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            _walk_config(v, out, prefix + k + '.')
        else:
            out.setdefault(k, v)
            out[prefix + k] = v


def hf_facts(hf_dir: Optional[str]) -> Dict[str, object]:
    """What the HF source says about a draft head and about MoE routing.

    Returns `{'ok': bool, 'nextn': bool, 'evidence': [str], 'moe_top_k': int|None,
    'moe_n_experts': int|None, 'source': str|None}`.  `ok` False means nothing
    could be inspected, which is NOT the same answer as "inspected, none found".
    """
    out: Dict[str, object] = {'ok': False, 'nextn': False, 'evidence': [],
                              'moe_top_k': None, 'moe_n_experts': None,
                              'source': hf_dir}
    if not hf_dir or not os.path.isdir(hf_dir):
        return out
    cfg_path = os.path.join(hf_dir, 'config.json')
    flat: Dict[str, object] = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as fh:
                _walk_config(json.load(fh), flat)
            out['ok'] = True
        except (OSError, ValueError):
            pass
    ev: List[str] = out['evidence']                              # type: ignore
    for k in HF_NEXTN_KEYS:
        v = flat.get(k)
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and v:
            out['nextn'] = True
            ev.append(f'config.json {k}={v}')
    for k in HF_MOE_TOPK_KEYS:
        v = flat.get(k)
        if isinstance(v, int) and v > 0 and out['moe_top_k'] is None:
            out['moe_top_k'] = v
    for k in HF_MOE_N_KEYS:
        v = flat.get(k)
        if isinstance(v, int) and v > 1 and out['moe_n_experts'] is None:
            out['moe_n_experts'] = v

    # The weight index names the draft head's own tensors when the config is
    # silent about it.  Read only when needed: it is a big JSON.
    if not out['nextn']:
        idx = os.path.join(hf_dir, 'model.safetensors.index.json')
        if os.path.isfile(idx):
            try:
                with open(idx, 'r', encoding='utf-8') as fh:
                    keys = list((json.load(fh).get('weight_map') or {}).keys())
                out['ok'] = True
                hits = SGN.nextn_tensor_names(keys)
                if hits:
                    out['nextn'] = True
                    ev.append(f'{len(hits)} MTP/NEXTN tensor(s) in '
                              f'model.safetensors.index.json (e.g. {hits[0]})')
            except (OSError, ValueError):
                pass
    return out


# THE TWO ENVIRONMENT VARIABLES THE PRESET OWNS.
#
# `GGUF_DOWNLOAD_CONF` (tensor_downloader.sh:115) picks which model's tensors.map
# is fetched; `SGL_ARCH` (convert_map_qtype.py) picks which ARCHS entry
# synthesises the sgl_* maps.  Under the preset both are DERIVED, so a value
# left over in a shell can only be a mistake - and it was a silent one: running
# the documented Qwen3.8-27B command with a stale GGUF_DOWNLOAD_CONF pointing at
# GLM-4.7 fetched GLM's map and returned a 662 GiB, 926-line recipe for a
# 17.41 GB request, at exit 0.
PRESET_OWNED_ENV = ('GGUF_DOWNLOAD_CONF', 'SGL_ARCH')


def env_conflict(var: str, have: Optional[str], want) -> Optional[str]:
    """The refusal for a stale environment variable, or None when it agrees.

    Agreement is by resolved path for a path-valued variable and by string
    otherwise, so `./download.conf` and an absolute path are the same answer.
    """
    if not want or not have:
        return None
    if have == str(want):
        return None
    if os.path.sep in str(want) and \
            os.path.abspath(have) == os.path.abspath(str(want)):
        return None
    return (f"{var} is set in the environment to {have!r}, but "
            f"--speed-profile sglang resolved {str(want)!r}. Under the preset "
            f"that variable is LEGACY: the preset owns it, because a stale value "
            f"silently builds the recipe from ANOTHER MODEL and exits 0 "
            f"(measured: a 662 GiB, 926-line recipe for a 17.41 GB request). "
            f"Unset it, or make it agree.")


# =============================================================================
# 4. THE NEXTN / EAGLE DECISION
# =============================================================================

def companion_split_for(model_dir: Optional[str], model_name: Optional[str],
                        part: str = 'mtp') -> Optional[str]:
    """`models/<part>-<MODEL>/download.conf` beside this model folder, or None.

    A draft head converted with --no-mtp lives in its own `mtp-<MODEL>` split,
    and this suite carries that split's folder next to the model's.  It is a
    witness that needs no HF cache and no download, so a recipe made from a
    fresh clone says the same thing about the draft head as one made here.
    """
    if not model_dir or not model_name:
        return None
    p = os.path.join(os.path.dirname(os.path.abspath(model_dir)),
                     f'{part}-{model_name}', 'download.conf')
    return p if os.path.isfile(p) else None


def nextn_evidence(gguf_names=None,
                   hf_dir: Optional[str] = None,
                   companion: Optional[str] = None) -> Tuple[Optional[bool], str]:
    """Is there an MTP/NEXTN draft head?  (True / False / None, and the evidence).

    None means NOTHING COULD BE INSPECTED, which is not the same answer as "no".
    Separated from the decision below because the decision is now `off` by
    default and the DETECTION is still wanted: a user who quantises the
    embedding on a model that has a draft head deserves one line saying which
    engine that recipe then needs.

    Witnesses, in order: the tensor map (a head converted into the main split),
    the `mtp-<MODEL>` companion folder of this suite (a head converted apart),
    then the HF source's config.json.  The first two travel with the suite.
    """
    hits = SGN.nextn_tensor_names(gguf_names or [])
    if hits:
        return True, (f'{len(hits)} MTP/NEXTN tensor(s) in the tensor map '
                      f'(e.g. {hits[0]})')
    if companion:
        rel = os.path.join(*os.path.normpath(companion).split(os.sep)[-3:])
        return True, ('no MTP/NEXTN tensor in the map, but this suite carries '
                      f'the draft head\'s own split ({rel})')
    facts = hf_facts(hf_dir)
    if facts['nextn']:
        return True, ('no MTP/NEXTN tensor in the map, but the HF source says '
                      + '; '.join(facts['evidence']))              # type: ignore
    if facts['ok']:
        return False, (f'inspected the tensor map and the HF source '
                       f'({facts["source"]}) and found no MTP/NEXTN head')
    return None, ('no MTP/NEXTN tensor in the map and no HF source could be '
                  'inspected')


def resolve_nextn(mode: str, gguf_names=None,
                  hf_dir: Optional[str] = None,
                  companion: Optional[str] = None) -> Tuple[bool, str]:
    """(on, one-line reason) for the NEXTN/EAGLE-compatible BF16 embedding.

    WHAT THE PIN IS.  Speculative decoding - MTP/NEXTN and EAGLE alike - runs a
    draft head that SHARES the target model's embedding table, and a STOCK
    SGLang hands it that table as a bare parameter.  A packed NVFP4 embedding is
    not a bare parameter (uint8 codes + e4m3 block scales + a global
    weight_scale_2), so on a stock engine the draft breaks.  Pinning the
    embedding to BF16 avoids that, and costs only that one tensor: `lm_head` at
    NVFP4 runs NEXTN fine (measured), and the embedding is a gather, so the pin
    costs bytes and neither prefill nor decode time.

    IT IS OFF BY DEFAULT, and that is a MEASUREMENT, not a preference.  The
    engine these numbers were taken on carries the shared-embedding fix (an
    SGLang pull request for it is pending), so the stock-engine breakage the pin
    exists to avoid does not arise; and AT EQUAL SIZE the pin measured +53 %
    KLD, because 1.70 GiB spent on an embedding is 1.70 GiB not spent on the
    weights that carry the model.  Paying 53 % of your quality for a workaround
    you do not need is the wrong default.

    THE THREE ANSWERS.
      * `off` (DEFAULT) - the embedding is quantised like anything else.  When a
        draft head IS detected the run still says so, in one line, naming the
        engine that recipe needs.
      * `on`  - pin it.  For a STOCK engine, and for EAGLE, which has drafts for
        models with no MTP head at all - so this must be settable everywhere.
      * `auto` - on when a draft head is found in the tensor map or the HF
        source, and on when nothing could be inspected at all.  This is the old
        default, kept for anyone deploying on an unpatched engine.
    """
    if mode == 'on':
        return True, ('forced on by --nextn-optimization on (a stock engine, or '
                      'EAGLE, which needs it even where the model has no MTP head)')
    found, why = nextn_evidence(gguf_names, hf_dir, companion)
    if mode == 'off':
        return False, ('off (the default): the embedding is quantised like any '
                       'other tensor - ' + why)
    if found is True:
        return True, f'auto: {why}'
    if found is False:
        return False, f'auto: {why}'
    return True, ('auto: ' + why + ' - defaulting ON, because on an unpatched '
                  'engine a NEXTN-incompatible recipe cannot be fixed later '
                  'without re-quantising')


# The reminder a run prints when it leaves the embedding quantised on a model
# that HAS a draft head.  One line, once, in the preset block: the recipe is
# fine, it just names the engine it needs for speculation.
NEXTN_REMINDER = (
    'NEXTN/EAGLE head detected and the token embedding stays quantized: '
    'speculation then needs an engine carrying the shared-embedding fix '
    '(SGLang PR pending) - pass --nextn-optimization on for a stock engine.')


# =============================================================================
# 5. THE TWO BUDGETS, WITHOUT A BENCHMARK
# =============================================================================
#
# ONE SENTENCE: SPEND WHAT YOU ASKED FOR, THEN BE AS FAST AS POSSIBLE.
#
# The user hands this tool one number, a byte target, and it is not a hint - it
# is the size of the card the checkpoint has to live on.  A run that comes back
# 2.5 GB under it has not been conservative, it has ignored the only thing it
# was told.  Both budget defaults follow from that:
#
#   --prefill-budget auto   the FILL rule below: among the sweep's points, the
#                           STRICTEST prefill budget whose recipe actually
#                           spends the target.  Speed is bought with whatever
#                           the byte target does not need, never with bytes.
#   --decode-budget auto    UNCONSTRAINED, and reported.  See below.
#
# WHAT WAS HERE BEFORE, AND WHY IT IS GONE.  The first cut of this file chose
# the prefill budget by the KNEE of the predicted quality-vs-prefill curve, and
# the decode budget by a PROPORTIONAL rule (decode bytes may grow as the size
# cap does over the pool's byte floor).  Both are defensible in the abstract and
# both were wrong here, measured:
#
#   * the knee treats prefill speed as a co-equal objective the user never
#     stated.  On Qwen3.8-27B at 19.04 GB it picked a recipe of 16.50 GB -
#     2.5 GB under target - trading 47 % more predicted KLD for about 8 points
#     of prefill throughput, and reported "all budgets met".  Two different byte
#     targets (17.41 and 19.04 GB) collapsed onto the same recipe.
#   * the proportional decode rule is ILL-POSED ON A MoE.  It equates a growth
#     in checkpoint bytes with a growth in per-token ACTIVE bytes, and on
#     GLM-4.7 `mid` those move at completely different rates: active bytes grow
#     35 % while total bytes grow 5 %.  The rule then binds on a quantity the
#     user has no lever for, and the run reports BUDGET NOT MET for a recipe
#     that is exactly the size it was asked for.
#
# For a dense model the SIZE CAP IS ALREADY THE DECODE LEVER - decode bytes are
# the checkpoint minus the embedding - so an automatic decode budget adds
# nothing there either.  Hence: report it, do not cap it.  `--decode-budget
# <fraction>` and `--decode-max-bytes` are unchanged for anyone who wants one.

# How full is full.  Below this fraction of the requested cap a recipe has not
# spent what it was given; at or above it, the remainder is smaller than the
# smallest thing the pool can buy with it.  A documented constant rather than a
# hidden one, and `--fill-fraction` overrides it.
FILL_FRACTION = 0.98


def fill_pick(rows: Sequence[dict], cap_bytes: Optional[float],
              fill_fraction: float = FILL_FRACTION) -> Tuple[int, str]:
    """(index, reason) - the FILL rule over a prefill sweep.

    `rows` are the sweep's points, ordered by ascending prefill budget (loosest
    first), each carrying `budget`, `bytes` and `deg`.  `cap_bytes` is what the
    user asked for, on the same basis as `bytes`.

    THE RULE.  Take every point whose recipe reaches at least `fill_fraction` of
    the cap, and among those take the STRICTEST budget - the fastest recipe that
    still spends the money.  Speed is then bought only with bytes the target did
    not need.

    WHEN NOTHING REACHES IT - the cap is at or below the pool's byte floor, or
    the pool simply cannot spend that many bytes at any speed - take the point
    that gets closest, loosest first, and SAY SO.  That is not a failure: it is
    the honest answer to "spend 19 GB" from a pool whose recipes stop at 18.9.

    Ties at equal bytes go to the lower predicted degradation, which is the
    objective the whole tool is written around.

    AND THE BUDGET RECORDED MUST BE ONE THE RECIPE MEETS.  A point's `met` says
    whether its recipe came in under the prefill cap that produced it; where the
    pool cannot reach the floor at this byte target, the strictest budgets are
    ones the recipe fails while returning the same recipe as a looser one.
    Recording those would put a bound in the header that the artefact does not
    satisfy, so among the points that fill the target the met ones win first.

    FILLING IS A WINDOW, NOT A THRESHOLD.  `bytes >= want` alone also calls a
    point that OVERSHOOTS the cap "filling the budget": the cap is the size of
    the card, so a recipe above it has not filled the target, it has missed it in
    the direction that does not fit.  MEASURED on GLM-4.7 `mid`: an assigner
    defect put every one of the twelve points at 226,066,179,852 B against a
    211,275,244,537 B cap and this rule reported "fills the budget at the
    strictest prefill budget: ... 107.0 %" and shipped it.  So the window is
    `want <= bytes <= cap`; below it, the largest point that still FITS wins and
    says nothing was filled; and only when NO point fits - the cap sits at or
    below the pool's byte floor, where exactly one recipe exists - is the
    smallest overshoot taken, and named as one.
    """
    n = len(rows)
    if n == 0:
        raise ValueError('fill_pick() needs at least one point')
    want = (float(cap_bytes) * float(fill_fraction)) if cap_bytes else None
    if want:
        elig = [i for i in range(n)
                if want <= float(rows[i].get('bytes') or 0) <= float(cap_bytes)]
        if elig:
            i = max(elig, key=lambda k: (rows[k].get('met') is not False,
                                         rows[k]['budget'],
                                         -float(rows[k].get('deg') or 0)))
            return i, (f'fills the budget at the strictest prefill budget'
                       + ('' if rows[i].get('met') is not False
                          else ' (none of the filling points meets its own cap)')
                       + f': '
                       f'{float(rows[i]["bytes"]):,.0f} B is '
                       f'{100.0 * float(rows[i]["bytes"]) / float(cap_bytes):.1f}% '
                       f'of the {float(cap_bytes):,.0f} B asked for, at or above '
                       f'the {100.0 * fill_fraction:.0f}% fill fraction, and no '
                       f'stricter budget in the sweep also reaches it')
    # Nothing filled the window.  Among the points that FIT, the one that spends
    # the most; and if none fits, the one that overshoots least - the cap is then
    # at or below the pool's byte floor and that point is the only recipe there.
    under = ([i for i in range(n)
              if float(rows[i].get('bytes') or 0) <= float(cap_bytes)]
             if cap_bytes else list(range(n)))
    if under:
        i = max(under, key=lambda k: (float(rows[k].get('bytes') or 0),
                                      -float(rows[k].get('deg') or 0),
                                      -rows[k]['budget']))
        return i, (f'NOTHING IN THE SWEEP FILLS THE BUDGET'
                   + (f' ({100.0 * fill_fraction:.0f}% of {float(cap_bytes):,.0f} B '
                      f'= {want:,.0f} B)' if want else '')
                   + f': this is the point that spends the most, '
                     f'{float(rows[i].get("bytes") or 0):,.0f} B. The cap is at or '
                     f'below the pool byte floor, or the pool cannot spend that many '
                     f'bytes at any prefill budget - the run reports both floors '
                     f'above')
    i = min(range(n), key=lambda k: (float(rows[k].get('bytes') or 0),
                                     float(rows[k].get('deg') or 0),
                                     -rows[k]['budget']))
    return i, (f'EVERY POINT IN THE SWEEP IS OVER THE SIZE CAP '
               f'({float(cap_bytes):,.0f} B): this is the smallest of them, '
               f'{float(rows[i].get("bytes") or 0):,.0f} B '
               f'({100.0 * float(rows[i].get("bytes") or 0) / float(cap_bytes):.1f}% '
               f'of it). The cap is below the pool byte floor - the pinned '
               f'tensors alone weigh more than that - so exactly one recipe '
               f'exists here and it is over target; the run reports both floors '
               f'above')


def sweep_table(rows: Sequence[dict], chosen: int,
                cap_bytes: Optional[float] = None) -> List[str]:
    """The sweep, as a small table with the chosen point marked."""
    out = [f'     {"prefill-budget":<14} {"prefill index":<13} {"predicted deg":<13} '
           f'{"bytes":>15} {"bpw":>7} {"fill":>6}',
           f'     {"-"*14} {"-"*13} {"-"*13} {"-"*15} {"-"*7} {"-"*6}']
    for i, r in enumerate(rows):
        mark = '->' if i == chosen else '  '
        b = r.get('budget')
        cb = 'unbudgeted' if b is None else ('%.6f' % b)
        cp = 'n/a' if r.get('prefill') is None else ('%.6f' % r['prefill'])
        cd = 'n/a' if r.get('deg') is None else ('%.6f' % r['deg'])
        cn = '{:,}'.format(int(r.get('bytes') or 0))
        cw = 'n/a' if r.get('bpw') is None else ('%.4f' % r['bpw'])
        cf = ('n/a' if not cap_bytes else
              '%.1f%%' % (100.0 * float(r.get('bytes') or 0) / float(cap_bytes)))
        out.append(f'  {mark} {cb:<14} {cp:<13} {cd:<13} {cn:>15} {cw:>7} {cf:>6}')
    return out


# =============================================================================
# 6. THE PLAN
# =============================================================================

def preset_plan(csv_path: str, names, arch_key: Optional[str] = None,
                degradation_csv: Optional[str] = None,
                pool: Sequence[str] = PRESET_POOL,
                nextn_mode: str = 'off',
                hf_source: Optional[str] = None) -> dict:
    """Everything `--speed-profile sglang` decides, with the reason for each.

    Pure: reads files, touches no global state, returns a dict.  The caller maps
    it onto argparse and is responsible for not overriding anything the user
    typed.  Raises ValueError with a message naming the flag to pass when a
    decision cannot be made.
    """
    model_dir = model_dir_for(csv_path, degradation_csv)
    conf = download_conf_for(model_dir)
    model_name = model_name_from_conf(conf)

    names = list(names)
    if arch_key is None:
        arch_key = SGN.detect_arch(names)
        arch_reason = f'auto-detected from {len(names)} tensor name(s)'
    else:
        arch_reason = 'given by --sgl-arch'
    if arch_key is None or arch_key not in SGN.ARCHS:
        raise ValueError(
            'could not identify the model architecture from the tensor names. '
            'An arch qualifies only if it names EVERY tensor, because a tensor '
            'we cannot name is one that would silently stay BF16 and blow the '
            'byte budget in a run that reports success. The registered ones are: '
            + ', '.join(sorted(SGN.ARCHS))
            + '. If one of those IS this model, pass --sgl-arch <name>. IF NONE '
              'OF THEM FITS, this model needs its own ARCHS entry before the '
              'SGLang path can size it - that is a small declarative table and '
              'no GPU at all (docs/sglang.md SS9), and `sglang_native.py '
              '--smoke-hf <an HF checkpoint of this model>` is the pre-flight '
              'that needs no table. Forcing a wrong --sgl-arch does not produce '
              'a wrong recipe, it produces a refusal further in.')
    arch = SGN.ARCHS[arch_key]

    # WHICH per-format table prices the pool, and whether one is needed at all.
    # The rows are a property of the FORMAT and the measured ones are built in,
    # so this cannot fail and never asks anybody for a benchmark.  A file - the
    # flag, this model's own dedicated one, or a merged table still carrying the
    # rows - is honoured, and warned about, because all it can move is the
    # predicted number the footer prints.
    if degradation_csv:
        deg_csv = degradation_csv
        deg_why = 'given by --quant-degradation-csv'
    else:
        deg_csv, deg_why = degradation_source_for(model_dir)

    if hf_source is None:
        hf_source = find_hf_source(model_name)
    facts = hf_facts(hf_source)
    companion = companion_split_for(model_dir, model_name)
    nextn_on, nextn_reason = resolve_nextn(nextn_mode, names, hf_source, companion)
    nextn_found, _nextn_why = nextn_evidence(names, hf_source, companion)

    emb = SGN.embedding_tensors(arch)
    # The draft head's own token embedding table, as this model's map names it:
    # every per-layer tensor the arch types as an embedding, i.e.
    # blk.N.nextn.embed_tokens.weight on glm4_moe and nothing at all on an arch
    # whose table declares no per-layer embedding.  Asked for by role, like the
    # global one, so nothing here spells a tensor name.
    draft_emb: List[str] = []
    for _n in names:
        _info = SGN.map_tensor(_n, arch)
        if _info and _info[1] == 'embedding' and _info[3] is not None:
            draft_emb.append(_n)
    draft_emb.sort()
    dense = SGN.dense_embedding_algo(pool)
    emb_pins: List[str] = []
    draft_pins: List[str] = []
    # Two rules here, and they are not the same kind of rule.
    #
    # The arch's own covers the global table alone (`emb`; `embedding_tensors`
    # deliberately returns no per-layer one).  An arch whose SGLang model file
    # builds VocabParallelEmbedding without a quant_config can load nothing but
    # bf16 there (sglang_native.embedding_algos_for / algo_legal), so that pin is
    # emitted whatever --nextn-optimization says and its reason is the model
    # file, not speculation.
    #
    # --nextn-optimization's own covers every table the draft head reads: the
    # global one it shares and, where the model has one, its own per-layer table.
    # glm4_moe_nextn.py builds that one without a quant_config too, so a packed
    # one does not load either - but only once the draft module is instantiated,
    # and that happens only under speculation, which is exactly what this flag
    # declares.  So with the flag on both tables are pinned, and with it off the
    # per-layer one is left to the optimiser like any other tensor.
    arch_emb = SGN.embedding_algos(arch)
    emb_bf16_only = tuple(arch_emb) == (SGN.BF16,)
    emb_reason = None
    if emb_bf16_only or nextn_on:
        if not dense:
            raise ValueError(
                ('this architecture can load nothing but a DENSE (unpacked) '
                 'token embedding' if emb_bf16_only else
                 'the NEXTN/EAGLE-compatible embedding needs a DENSE embedding '
                 'type')
                + ' and --gpu-quants has none (only '
                + SGN.BF16 + ' qualifies). Add it'
                + ('.' if emb_bf16_only else ', or pass '
                   '--no-nextn-optimization.'))
        emb_pins = [f'^{re.escape(n)}$={dense}' for n in emb]
        if nextn_on:
            draft_pins = [f'^{re.escape(n)}$={dense}' for n in draft_emb]
    if emb_bf16_only:
        emb_reason = (
            f"{arch_key}'s SGLang model file builds the embedding without a "
            f'quant_config, so SGLang loads it as bf16 only and a packed one '
            f'does not load at all - not a budget choice')

    return dict(
        model_dir=model_dir,
        download_conf=conf,
        model_name=model_name,
        arch=arch_key,
        arch_reason=arch_reason,
        degradation_csv=deg_csv,
        degradation_csv_reason=deg_why,
        # The three below are the built-in branch: no path to name, the rows
        # themselves for the assigner, and no warning.  With a file it is the
        # other way round - the path is the record, and the warning fires.
        degradation_label=(None if deg_csv else DEG_BUILTIN_LABEL),
        degradation_rows=(None if deg_csv else builtin_degradation_values()),
        degradation_warning=(DEG_FILE_WARNING if deg_csv else None),
        pool=list(pool),
        harmonize=SGN.harmonize_argument(names, arch),
        pins=[f'{pat}={SGN.BF16}'
              for pat, _why in SGN.PIN_BF16_ADVISORY.get(arch_key, [])],
        pin_reasons=[why for _pat, why in SGN.PIN_BF16_ADVISORY.get(arch_key, [])],
        assign_qtype=SGN.BF16,
        nextn=nextn_on,
        nextn_reason=nextn_reason,
        nextn_found=nextn_found,
        # Printed only when the pin is OFF and a head IS there: the recipe is
        # good, it just names the engine speculation needs.  It still is with
        # the arch pinning the global table anyway, because the table the
        # reminder is about is then the draft head's own, and that one is
        # quantised like anything else while the flag is off.
        nextn_reminder=(NEXTN_REMINDER if (nextn_found and not nextn_on)
                        else None),
        embedding=emb,
        embedding_algo=dense,
        embedding_algos=list(arch_emb),
        embedding_bf16_only=emb_bf16_only,
        embedding_pin_reason=emb_reason,
        embedding_pins=emb_pins,
        draft_embedding=draft_emb,
        draft_embedding_pins=draft_pins,
        hf_source=hf_source,
        moe_top_k=facts['moe_top_k'],
        moe_n_experts=facts['moe_n_experts'],
        harmonization_technique=PRESET_HARMONIZATION_TECHNIQUE,
    )


# =============================================================================
# 7. THE SELF-TEST
# =============================================================================

_MODELS = os.path.join(_HERE, 'models')


def selftest(verbose: bool = True) -> int:
    fails = []

    def pub(path):
        """A path shortened against the suite root, for readable test detail."""
        if not path:
            return str(path)
        try:
            return os.path.relpath(str(path), _HERE)
        except ValueError:                                     # pragma: no cover
            return str(path)

    def ok(cond, label, extra=''):
        (print(f'  ok   {label}' + (f'  ({extra})' if extra else ''))
         if cond and verbose else None)
        if not cond:
            fails.append(label)
            print(f'  FAIL {label}' + (f'  ({extra})' if extra else ''))

    # -- 1. the pool ---------------------------------------------------------
    ok(SGN.container_for(PRESET_POOL) == 'modelopt_mixed',
       'pool: the preset pool is one container, and it is the one that can '
       'quantise an embedding')
    ok(SGN.dense_embedding_algo(PRESET_POOL) == SGN.BF16,
       'pool: the preset pool contains a dense embedding type')
    ok(all(q in SGN.SGLANG_ALGOS for q in PRESET_POOL),
       'pool: every preset qtype exists in the size model')

    # -- 2. the NEXTN decision. OFF IS THE DEFAULT, and it is a measurement:
    # an engine with the shared-embedding fix does not need the pin, and at
    # equal size the pin cost +53 % KLD.
    on, why = resolve_nextn('off', ['blk.92.nextn.eh_proj.weight'])
    ok(on is False and 'default' in why,
       'nextn: off is the DEFAULT and beats the evidence', why[:56])
    ok(nextn_evidence(['blk.92.nextn.eh_proj.weight'])[0] is True,
       'nextn: detection still runs when the pin is off - that is what the '
       'reminder is made of')
    on, why = resolve_nextn('auto', ['blk.92.nextn.eh_proj.weight'])
    ok(on and 'tensor map' in why,
       'nextn: auto is still available and MTP tensors in the map turn it ON',
       why[:56])
    on, why = resolve_nextn('auto', ['blk.1.attn_q.weight'], hf_dir=None)
    ok(on and 'no HF source' in why,
       'nextn: auto with nothing inspectable is ON, and SAYS SO', why[:56])
    on, why = resolve_nextn('on', ['blk.1.attn_q.weight'])
    ok(on is True,
       'nextn: it can be forced ON for a model with no MTP head at all - EAGLE '
       'drafts exist there')
    ok(nextn_evidence(['blk.1.attn_q.weight'])[0] is None,
       'nextn: "could not look" is a third answer, not a no')
    _f, _w = nextn_evidence(['blk.1.attn_q.weight'], None,
                            '/x/models/mtp-M/download.conf')
    ok(_f is True and 'models/mtp-M/download.conf' in _w,
       'nextn: the mtp-<MODEL> folder of this suite is a witness that needs no '
       'HF cache', _w[-40:])
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        os.makedirs(os.path.join(_td, 'models', 'M'))
        os.makedirs(os.path.join(_td, 'models', 'mtp-M'))
        open(os.path.join(_td, 'models', 'mtp-M', 'download.conf'), 'w').close()
        ok(companion_split_for(os.path.join(_td, 'models', 'M'), 'M') is not None
           and companion_split_for(os.path.join(_td, 'models', 'M'), 'N') is None,
           'nextn: the companion is looked up by THIS model\'s name beside its folder')

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'config.json'), 'w') as fh:
            json.dump({'model_type': 'x', 'text_config':
                       {'mtp_num_hidden_layers': 1}}, fh)
        on, why = resolve_nextn('auto', ['blk.1.attn_q.weight'], hf_dir=td)
        ok(on and 'mtp_num_hidden_layers=1' in why,
           'nextn: a NESTED config key is found (qwen3_5 puts it in text_config)',
           why[-60:])
        ok(nextn_evidence(['blk.1.attn_q.weight'], td)[0] is True,
           'nextn: and the HF source is inspected for the reminder too, not only '
           'for auto')
        with open(os.path.join(td, 'config.json'), 'w') as fh:
            json.dump({'model_type': 'x', 'num_nextn_predict_layers': 0}, fh)
        on, why = resolve_nextn('auto', ['blk.1.attn_q.weight'], hf_dir=td)
        ok(on is False and 'found no MTP' in why,
           'nextn: a ZERO-valued key means no head, and a negative needs that '
           'evidence to turn it off', why[:60])
        with open(os.path.join(td, 'model.safetensors.index.json'), 'w') as fh:
            json.dump({'weight_map': {'mtp.layers.0.eh_proj.weight': 'a.st'}}, fh)
        on, why = resolve_nextn('auto', ['blk.1.attn_q.weight'], hf_dir=td)
        ok(on and 'index.json' in why,
           'nextn: the weight index is read when the config is silent', why[-50:])
        ok(find_hf_source('Nope', roots=[td]) is None,
           'nextn: find_hf_source refuses a cache that has no such repo')

    # exact-name matching: a quantised derivative must not answer for the source
    with tempfile.TemporaryDirectory() as td:
        for repo in ('models--Org--M', 'models--Other--M-NVFP4'):
            snap = os.path.join(td, repo, 'snapshots', 'abc')
            os.makedirs(snap)
            open(os.path.join(snap, 'config.json'), 'w').write('{}')
        got = find_hf_source('M', roots=[td])
        ok(got is not None and 'models--Org--M' + os.sep in got + os.sep,
           'nextn: the HF source is matched on the EXACT repo name, so a '
           'quantised derivative cannot answer for the source', str(got))

    # -- 3. the decode rule --------------------------------------------------
    # -- 3/4. THE FILL RULE ---------------------------------------------------
    # Rows are ordered loosest budget first, exactly as the sweep hands them
    # over.  Bytes here are the recipe's own assigned+pinned total.
    def _row(b, by, dg):
        return dict(budget=b, bytes=by, deg=dg, prefill=1.0, bpw=5.0)

    CAP = 19_044_000_000
    # The real Qwen3.8-27B 19.04 GB sweep, which the knee used to answer 16.50 GB
    # to: four distinct recipes, and only the loosest ones spend the target.
    real = ([_row(0.40 + 0.05 * k, 18_857_851_508, 0.032204) for k in range(9)]
            + [_row(0.890119, 18_022_894_732, 0.036925),
               _row(0.945059, 16_499_838_092, 0.047410),
               _row(1.000000, 15_161_429_132, 0.063954)])
    i_, why = fill_pick(real, CAP)
    ok(i_ == 8 and abs(real[i_]['budget'] - 0.80) < 1e-9,
       'fill: the STRICTEST budget that still spends the target - not the knee, '
       'which took 16.50 GB of a 19.04 GB request', f'index {i_}, {why[:40]}')
    ok('fills the budget at the strictest' in why,
       'fill: and it says which rule fired')
    # A cap EVERY point overshoots (15.0 GB against a sweep that bottoms out at
    # 15.16 GB): the cap is below the pool's floor, exactly one recipe exists
    # there, and the honest answer is the smallest of them - never the largest,
    # and never called "filled".
    i_, why = fill_pick(real, 15_000_000_000)
    ok(i_ == 11 and 'OVER THE SIZE CAP' in why,
       'fill: a cap below every point takes the SMALLEST overshoot and says the '
       'sweep never fitted', f'index {i_}, {why[:44]}')
    # AND THE DEFECT THIS WINDOW EXISTS FOR.  GLM-4.7 `mid` once the lm_head was
    # pinned to bf16: an assigner defect put all twelve points at
    # 226,066,179,852 B against a 211,275,244,537 B cap, and `bytes >= want` with
    # no upper bound called that "fills the budget at the strictest prefill
    # budget: ... 107.0 %" and shipped it.  Overshooting the size of the card is
    # not filling it.
    _over = [_row(0.410206 + 0.053618 * k, 226_066_179_852, 0.019953)
             for k in range(12)]
    i_, why = fill_pick(_over, 211_275_244_537)
    ok('fills the budget' not in why and 'OVER THE SIZE CAP' in why,
       'fill: a point 107% of the cap is never reported as filling it',
       why[:52])
    # One point over the cap, one just under it: the one that FITS wins, however
    # much closer to the target the other one looks.
    _mixed = [_row(0.5, 226_066_179_852, 0.019953),
              _row(0.9, 211_263_901_452, 0.023230)]
    i_, why = fill_pick(_mixed, 211_275_244_537)
    ok(i_ == 1 and 'fills the budget at the strictest' in why,
       'fill: and the point that fits is picked over the one that overshoots',
       f'index {i_}')
    # A cap nothing reaches (at or below the floor, or beyond the pool): the
    # point that spends the most, and it SAYS the budget was not filled.
    i_, why = fill_pick(real, 40_000_000_000)
    ok(i_ == 0 and 'NOTHING IN THE SWEEP FILLS THE BUDGET' in why,
       'fill: an unreachable cap takes the point that spends the most, and says so',
       f'index {i_}')
    # Tie-break at equal bytes: the lower predicted degradation.
    tie = [_row(0.5, 1000, 0.05), _row(0.6, 1000, 0.04), _row(0.7, 900, 0.01)]
    i_, _w = fill_pick(tie, 40_000)          # nothing fills it -> max bytes
    ok(i_ == 1, 'fill: at equal bytes the tie goes to the lower predicted '
                'degradation', f'index {i_}')
    ok(fill_pick(real, None)[0] == 0,
       'fill: with no byte cap at all there is nothing to fill, so the point '
       'that spends the most is taken')
    ok(abs(FILL_FRACTION - 0.98) < 1e-12,
       'fill: FILL_FRACTION is a documented constant (0.98), not a hidden one')
    _just_under = [_row(0.9, int(CAP * 0.979), 0.03), _row(1.0, int(CAP * 0.5), 0.06)]
    ok(fill_pick(_just_under, CAP)[0] == 0,
       'fill: 97.9% of the cap does NOT count as filled at the default fraction')
    ok(fill_pick(_just_under, CAP, 0.97)[0] == 0,
       'fill: and --fill-fraction 0.97 changes that answer, so the constant is '
       'really the knob')
    # THE RECORDED BUDGET MUST BE ONE THE RECIPE MEETS.  GLM-4.7 at 211.3 GB:
    # all twelve points fill the target, but the strict ones are budgets their
    # own recipe fails while returning what a looser one returns.
    glm = ([_row(0.40 + 0.05 * k, 211_273_840_024, 0.022821) for k in range(7)]
           + [_row(0.780509, 210_468_614_992, 0.022951)]
           + [_row(0.835382 + 0.055 * k, 208_416_029_560, 0.024374) for k in range(4)])
    for k, r in enumerate(glm):
        r['met'] = k <= 7
    i_, why = fill_pick(glm, 211_275_244_537)
    ok(i_ == 7 and glm[i_]['met'],
       'fill: among the points that fill the target, one the recipe MEETS wins '
       'over a stricter one it fails', f'index {i_}')
    for r in glm:
        r['met'] = False
    i_, why = fill_pick(glm, 211_275_244_537)
    ok(i_ == 11 and 'none of the filling points meets its own cap' in why,
       'fill: and when none of them meets it, the strictest is taken and the '
       'reason says so', f'index {i_}')
    ok(len(sweep_table([_row(1.0, 5, 0.03)], 0, CAP)) == 3,
       'fill: the sweep table renders, header included')
    ok('->' in sweep_table([_row(1.0, 5, 0.03)], 0, CAP)[2],
       'fill: with the chosen row marked')

    # -- 5. the plan, on this suite's own models -----------------------------
    for model, arch, expect_nextn_src, emb_bf16_only in (
            ('Qwen3.8-27B', 'qwen3_5', 'draft head\'s own split', False),
            ('GLM-4.7', 'glm4_moe', 'tensor map', True)):
        csv_path = os.path.join(_MODELS, model, 'kld_results.csv')
        if not os.path.isfile(csv_path):
            print(f'  skip preset plan for {model} (no {csv_path})')
            continue
        import csv as _csv
        with open(csv_path, newline='') as fh:
            names = next(_csv.reader(fh))[1:]
        try:
            # No degradation flag: the plan resolves its own table, and on a
            # model with no file of its own that is the built-in rows.
            plan = preset_plan(csv_path, names)
        except ValueError as e:
            ok(False, f'{model}: preset plan builds', str(e)[:70])
            continue
        ok(plan['arch'] == arch, f'{model}: architecture auto-detected',
           str(plan['arch']))
        ok(plan['download_conf'] and os.path.basename(
            os.path.dirname(plan['download_conf'])) == model,
           f'{model}: download.conf resolved NEXT TO the CSV, not from the '
           'suite-root symlink', str(plan['download_conf']))
        ok(plan['model_name'] == model,
           f'{model}: and it names this model', str(plan['model_name']))
        ok(plan['nextn_found'] is True and expect_nextn_src in plan['nextn_reason'],
           f'{model}: NEXTN head detected from the {expect_nextn_src}',
           plan['nextn_reason'][:64])
        ok(plan['embedding_bf16_only'] is emb_bf16_only
           and (SGN.BF16 in plan['embedding_algos']),
           f'{model}: the arch says whether its model file can load a packed '
           f'embedding', ' '.join(plan['embedding_algos']))
        if emb_bf16_only:
            # The arch rule is not the NEXTN choice: it fires with the pin off.
            ok(plan['nextn'] is False
               and plan['embedding_pins'] == [r'^token_embd\.weight$=sgl_bf16']
               and 'quant_config' in (plan['embedding_pin_reason'] or ''),
               f'{model}: the embedding is pinned bf16 by the arch even with '
               f'--nextn-optimization off, and the reason names the model file',
               str(plan['embedding_pin_reason'])[:58])
        else:
            ok(plan['nextn'] is False and plan['embedding_pins'] == []
               and plan['embedding_pin_reason'] is None,
               f'{model}: OFF by default - the embedding is quantised like '
               f'anything else, and no pin is emitted')
        ok(plan['nextn_reminder'] == NEXTN_REMINDER,
           f'{model}: and because a draft head IS there, exactly one reminder '
           f'line naming the engine speculation needs - the arch pin does not '
           f'silence it, because the table left quantised is the draft\'s own')
        _on = preset_plan(csv_path, names, nextn_mode='on')
        ok(_on['embedding_pins'] == [r'^token_embd\.weight$=sgl_bf16'],
           f'{model}: --nextn-optimization on still pins it, from the arch role '
           f'column', str(_on['embedding_pins']))
        ok(_on['nextn_reminder'] is None,
           f'{model}: and the reminder is silent when the pin IS on')
        # The two pins have two different scopes, and the difference is only
        # visible on a model that has a per-layer embedding: the arch rule
        # covers the global table alone, while --nextn-optimization covers the
        # draft head's own table too - it is what pays for the draft module, and
        # that module builds its table without a quant_config just as the main
        # one does.  With the flag off nothing pins it: the module is not built.
        _mtp = sorted(n for n in names
                      if SGN.NEXTN_NAME_RE.search(n) and 'embed_tokens' in n)
        ok(_mtp == plan['draft_embedding'],
           f'{model}: the draft head\'s table is found by role, and it is the '
           f'MTP embedding the map carries', ' '.join(_mtp) or '(none)')
        if _mtp:
            _want = sorted(f'^{re.escape(n)}$=' + SGN.BF16 for n in _mtp)
            ok(_on['draft_embedding_pins'] == _want,
               f'{model}: --nextn-optimization on pins the draft head\'s own '
               f'table bf16 as well - glm4_moe_nextn.py builds it without a '
               f'quant_config, so a stock engine cannot load a packed one',
               ' '.join(_on['draft_embedding_pins']))
            ok(plan['draft_embedding_pins'] == []
               and not any(SGN.NEXTN_NAME_RE.search(p)
                           for p in plan['embedding_pins']),
               f'{model}: and with the flag off it is pinned by nothing - the '
               f'draft module is never instantiated, so the optimiser spends '
               f'those bytes where they buy quality')
            ok(SGN.embedding_algos_for(_mtp[0], SGN.ARCHS[arch])
               == SGN.EMBEDDING_ALGOS_DEFAULT,
               f'{model}: so it keeps the format ceiling in the size model, '
               f'which prices a checkpoint that may never speculate',
               ' '.join(SGN.embedding_algos_for(_mtp[0], SGN.ARCHS[arch])))
        else:
            ok(_on['draft_embedding_pins'] == [],
               f'{model}: --nextn-optimization on pins no draft table, because '
               f'this arch declares no per-layer embedding')
        ok(all(p.endswith('=' + SGN.BF16) for p in plan['pins']),
           f'{model}: the BF16 pins come from the arch advisory table')
        ok(plan['harmonize'] == SGN.harmonize_argument(names, SGN.ARCHS[arch]),
           f'{model}: harmonize groups come from the arch fused table')

    # -- 5b. WHICH per-format table, and why ---------------------------------
    #
    # THE THREE BRANCHES, and the warning that goes with two of them.  The
    # measured rows are BUILT IN, so the normal answer is "no file at all";
    # this model's own dedicated file overrides them; a merged table still
    # carrying the rows overrides them too, so a tree that has not been migrated
    # is never silently ignored.  Nothing in any of the three reads `models/`,
    # which is what lets the preset run from a model folder with no suite tree
    # in reach.
    #
    # THE DRIFT GUARD FIRST.  The constant and the committed Qwen file are the
    # same seven rows; a change to one that is not a change to the other would
    # be a silent divergence between what the tool computes and what the repo
    # ships, so a checkout that has the file holds the two together.
    _committed = committed_degradation_csv()
    if _committed:
        _rows = {}
        with open(_committed, encoding='utf-8') as fh:
            for _line in fh:
                _q, _, _v = _line.partition(',')
                _q, _v = _q.strip(), _v.strip()
                if _q.startswith('sgl_'):
                    _rows[_q] = float(_v)
        ok(_rows == builtin_degradation_values(),
           'built-in rows: the constant IS ' + pub(_committed) + ', row for row '
           'and value for value - the two cannot drift', f'{len(_rows)} row(s)')

    ok(set(PRESET_POOL) <= set(builtin_degradation_values()),
       'built-in rows: every type the preset pool can assign is priced by them '
       '- which is the whole reason a model needs no table of its own',
       ' '.join(PRESET_POOL))

    qwen_sgl = os.path.join(_MODELS, DEG_BUILTIN_MODEL, 'group0',
                            DEG_SGLANG_BASENAME)
    qwen_deg = os.path.join(_MODELS, DEG_BUILTIN_MODEL, 'group0',
                            DEG_MERGED_BASENAME)
    if os.path.isfile(qwen_sgl):
        ok(csv_has_sgl_rows(qwen_sgl) is True,
           'Qwen3.8-27B: the DEDICATED group0/kld_results_sglang.csv carries the '
           'sgl_* rows')
        _p, _why = degradation_source_for(os.path.join(_MODELS,
                                                       DEG_BUILTIN_MODEL))
        ok(_p == qwen_sgl and 'own' in _why,
           'Qwen3.8-27B: a model that HAS a file of its own uses THAT file, not '
           'the built-in rows and not the GGUF table', pub(_p))
        ok(csv_has_sgl_rows(qwen_deg) is False,
           'Qwen3.8-27B: the GGUF table no longer carries them - one kind of '
           'number per file (a weight-only llama-perplexity row and a '
           'coverage-normalised in-engine row are not the same quantity)')
    elif os.path.isfile(qwen_deg):
        ok(csv_has_sgl_rows(qwen_deg) is True,
           'Qwen3.8-27B: pre-split tree - the sgl_* rows are still in the GGUF '
           'table, and the search finds them there')

    glm_deg = os.path.join(_MODELS, 'GLM-4.7', 'group0', DEG_MERGED_BASENAME)
    if os.path.isfile(glm_deg):
        ok(csv_has_sgl_rows(glm_deg) is False,
           'GLM-4.7: its own group0 table carries no sgl_* rows')
        ok(not os.path.isfile(os.path.join(_MODELS, 'GLM-4.7', 'group0',
                                           DEG_SGLANG_BASENAME)),
           'GLM-4.7: and it has no dedicated file of its own either')
        _p, _why = degradation_source_for(os.path.join(_MODELS, 'GLM-4.7'))
        ok(_p is None and 'no file needed' in _why and DEG_BUILTIN_MODEL in _why,
           'GLM-4.7: so the BUILT-IN rows price the pool - no file, no models/ '
           'lookup, nothing for this model to measure', _why[:58])
        ok('not the assignment' in _why,
           'GLM-4.7: and the reason says what those rows decide and what they '
           'do not, so a prediction never reads as a measurement of this model')

    # the three branches as the PLAN reports them, in a directory that is not a
    # model folder of this suite at all
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _g = os.path.join(_td, 'group0')
        os.makedirs(_g)
        _csv = os.path.join(_td, 'kld_results.csv')
        with open(_csv, 'w') as fh:
            fh.write('QTYPE,blk.0.attn_q.weight\n')
        _names = ['blk.0.attn_q.weight']

        _p, _why = degradation_source_for(_td)
        ok(_p is None and 'no file needed' in _why,
           'branch 3: a model folder with nothing of its own resolves to the '
           'built-in rows, and that is not a fallback but the normal answer')
        _pl = preset_plan(_csv, _names, arch_key='qwen3_5')
        ok(_pl['degradation_csv'] is None
           and _pl['degradation_rows'] == builtin_degradation_values()
           and _pl['degradation_warning'] is None,
           'branch 3: the plan carries the seven rows themselves, no path and '
           'no warning - this is the case that needs no benchmark')
        ok(_pl['degradation_label'] == DEG_BUILTIN_LABEL
           and DEG_BUILTIN_MODEL in DEG_BUILTIN_LABEL,
           'branch 3: and the footer names them, with the model they were '
           'measured on', _pl['degradation_label'])

        with open(os.path.join(_g, DEG_MERGED_BASENAME), 'w') as fh:
            fh.write('QTYPE,group0\nq8_0,0.000549\nsgl_bf16,0.0\n'
                     'sgl_nvfp4,0.09\nsgl_fp8,0.009\n')
        _p, _why = degradation_source_for(_td)
        ok(_p == os.path.join(_g, DEG_MERGED_BASENAME) and 'pre-split' in _why,
           'branch 2: a merged table that still has sgl_* rows is USED, not '
           'silently overridden by the built-in ones')
        _pl = preset_plan(_csv, _names, arch_key='qwen3_5')
        ok(_pl['degradation_rows'] is None
           and _pl['degradation_warning'] == DEG_FILE_WARNING,
           'branch 2: and it is warned about, because all it can move is the '
           'predicted number')

        with open(os.path.join(_g, DEG_SGLANG_BASENAME), 'w') as fh:
            fh.write('# mine\nQTYPE,group0\nsgl_bf16,0.0\nsgl_nvfp4,0.07\n')
        _p, _why = degradation_source_for(_td)
        ok(_p == os.path.join(_g, DEG_SGLANG_BASENAME) and 'own' in _why,
           'branch 1: with both on disk the DEDICATED file wins')
        _pl = preset_plan(_csv, _names, arch_key='qwen3_5')
        ok(_pl['degradation_csv'] == os.path.join(_g, DEG_SGLANG_BASENAME)
           and _pl['degradation_rows'] is None
           and _pl['degradation_warning'] == DEG_FILE_WARNING,
           'branch 1: the plan names the file the footer will record, and warns '
           'the same way')

        _pl = preset_plan(_csv, _names, arch_key='qwen3_5',
                          degradation_csv='/somewhere/else.csv')
        ok(_pl['degradation_csv'] == '/somewhere/else.csv'
           and _pl['degradation_csv_reason'] == 'given by --quant-degradation-csv'
           and _pl['degradation_warning'] == DEG_FILE_WARNING,
           'the explicit --quant-degradation-csv still wins over every branch, '
           'and carries the same warning')

    ok('not the recipe' in DEG_FILE_WARNING and '27 sizes' in DEG_FILE_WARNING
       and DEG_BUILTIN_MODEL in DEG_FILE_WARNING
       and 'not needed' in DEG_FILE_WARNING,
       'warning: it says the table is not needed, what a measured one changes, '
       'what it does not, and where the built-in rows came from')

    # unknown architecture: refuse, name the flag AND the way out when no arch fits
    try:
        preset_plan('/nonexistent/kld_results.csv', ['not.a.tensor.name'])
        ok(False, 'unknown arch: refuses')
    except ValueError as e:
        ok('--sgl-arch' in str(e),
           'unknown arch: refuses and names the override flag')
        ok('SS9' in str(e) and '--smoke-hf' in str(e),
           'unknown arch: and says what to do when NEITHER arch fits - an ARCHS '
           'entry, docs SS9, with --smoke-hf as the pre-flight')

    # AND NOTHING IS READ FROM `models/`.  With the suite root pointed at a
    # directory that does not exist, a model with no table of its own still
    # prices its pool - which is the case the shipped recipes are re-proved
    # from: one model folder, no suite tree in reach, no flag.
    _saved = globals()['_HERE']
    try:
        globals()['_HERE'] = '/nonexistent-suite'
        _plan = preset_plan(os.path.join(_MODELS, 'GLM-4.7', 'kld_results.csv'),
                            ['blk.0.attn_q.weight'], arch_key='glm4_moe')
        ok(_plan['degradation_csv'] is None
           and _plan['degradation_rows'] == builtin_degradation_values(),
           'no suite tree in reach: the pool is still priced, by the built-in '
           'rows, and there is nothing left to refuse for')
    except ValueError as e:
        ok(False, 'no suite tree in reach: the pool is still priced', str(e)[:70])
    finally:
        globals()['_HERE'] = _saved

    # -- 6. the environment variables the preset owns ------------------------
    ok(env_conflict('SGL_ARCH', None, 'qwen3_5') is None,
       'env: an unset variable is not a conflict')
    ok(env_conflict('SGL_ARCH', 'qwen3_5', 'qwen3_5') is None,
       'env: a variable that AGREES is silent')
    _m = env_conflict('SGL_ARCH', 'glm4_moe', 'qwen3_5')
    ok(_m and 'LEGACY' in _m and 'glm4_moe' in _m and 'qwen3_5' in _m,
       'env: a stale SGL_ARCH is refused, naming both values', (_m or '')[:52])
    _m = env_conflict('GGUF_DOWNLOAD_CONF', '/a/GLM-4.7/download.conf',
                      '/b/Qwen3.8-27B/download.conf')
    ok(_m and '662 GiB' in _m,
       'env: a stale GGUF_DOWNLOAD_CONF is refused, with the measured damage')
    ok(env_conflict('GGUF_DOWNLOAD_CONF', './download.conf',
                    os.path.join(os.getcwd(), 'download.conf')) is None,
       'env: the same file by two paths is not a conflict')
    ok(set(PRESET_OWNED_ENV) == {'GGUF_DOWNLOAD_CONF', 'SGL_ARCH'},
       'env: both knobs the docs used to deny are owned by the preset')

    # -- 7. what quant_assign.py does with all of that -----------------------
    # Imported lazily: quant_assign imports THIS module, so it is fully loaded by
    # the time selftest() runs and the cycle cannot bite.
    try:
        import quant_assign as QA
    except Exception as e:                                     # pragma: no cover
        print(f'  skip quant_assign cases ({e})')
        QA = None
    if QA is not None:
        # THE DEGRADATION LOADER AND THE COMMENT.  The dedicated table carries a
        # provenance header, and `pd.read_csv` without `comment='#'` PROMOTES a
        # leading `#` line to the header: the real header becomes a data row,
        # every value is dropped as unparseable, and the run continues with an
        # EMPTY table at exit 0.  So this is a correctness test, not cosmetics.
        if _committed:
            _vals = QA.load_quant_degradation_values(_committed)
            ok(len([q for q in _vals if q.startswith('sgl_')]) >= 6,
               'loader: the dedicated table reads back all its sgl_* rows '
               'THROUGH its leading comment block', f'{len(_vals)} row(s)')
            ok(abs(_vals.get('sgl_nvfp4', -1) - 0.066930) < 1e-9
               and _vals.get('sgl_bf16') == 0.0,
               'loader: and the values are the measured ones, not shifted by a '
               'column')
            ok({q: v for q, v in _vals.items() if q.startswith('sgl_')}
               == builtin_degradation_values(),
               'loader: and read THROUGH the loader the committed file is the '
               'built-in constant exactly - a run with the file and a run '
               'without it price the pool with the same seven numbers')
        with _tf.NamedTemporaryFile('w', suffix='.csv', delete=False) as _fh:
            _fh.write('# a leading comment\nQTYPE,group0\nsgl_bf16,0.0\n'
                      'sgl_nvfp4,0.0669\n')
            _tmp_csv = _fh.name
        try:
            _vals = QA.load_quant_degradation_values(_tmp_csv)
            ok(_vals.get('sgl_nvfp4') == 0.0669 and _vals.get('sgl_bf16') == 0.0,
               'loader: a leading comment is a comment, not the header')
        finally:
            os.remove(_tmp_csv)

        # THE UNIT TRAP, both directions.
        _ref = 53_786_705_920                       # Qwen3.8-27B all-BF16 total
        for txt, want in (('17410000000B', 17_410_000_000),
                          ('17.41GB', 17_410_000_000),
                          ('16.2GiB', int(16.2 * 1024 ** 3)),
                          ('16', 16 * 1024 ** 3),
                          ('80%', int(0.8 * _ref))):
            got = QA._resolve_max_size(txt, _ref, 'gpu', '--gpu-tensors-max-size')
            ok(abs(got - want) <= 1,
               f'size: {txt!r} -> {want:,} B', f'{got:,.0f}')
        for bad, why in (('17.41B', 'seventeen BYTES, not gigabytes'),
                         ('200GIB', 'more than twice the largest recipe')):
            try:
                QA._resolve_max_size(bad, _ref, 'gpu', '--gpu-tensors-max-size')
                ok(False, f'size: {bad!r} is REFUSED ({why})')
            except SystemExit as e:
                ok(e.code == 2, f'size: {bad!r} is REFUSED ({why})')
        # A bare number keeps the contract every GGUF recipe was made under:
        # GiB, never refused, however large - it is warned about, not stopped.
        import io as _io, contextlib as _ctx
        _err = _io.StringIO()
        with _ctx.redirect_stderr(_err):
            _got = QA._resolve_max_size('17410000000', _ref, 'gpu',
                                        '--gpu-tensors-max-size')
        ok(_got == 17410000000 * 1024 ** 3 and 'Warning' in _err.getvalue(),
           'size: a bare number is never refused (GiB, as always), only warned about',
           _err.getvalue()[:60])
        for bad in ('banana', '12 GG', ''):
            try:
                QA._size_arg(bad)
                ok(False, f'size: {bad!r} is rejected at parse time')
            except argparse.ArgumentTypeError:
                ok(True, f'size: {bad!r} is rejected at parse time')
        ok(QA._size_arg(' 17410000000B ') == '17410000000B',
           'size: a value with spaces round it still parses')

        # THE BUDGET VERDICT.
        _rep = dict(feasible=False, applied=283, p_max=1.0075, p_budget=1.0,
                    prefill_after=1.4701, P_floor=1.0070, prefill_ratio=0.821,
                    d_max=2.5e10, decode_after=2.6e10, D_floor=1.9e10,
                    decode_ratio=0.778)
        _v = QA.budget_verdict_lines(_rep, 'gpu')
        ok(_v and _v[0].startswith('BUDGET NOT MET'),
           'verdict: it starts with the words BUDGET NOT MET, not "feasible=False"')
        ok(any('WHAT TO DO' in l for l in _v),
           'verdict: and it says what to do about it')
        ok(any('1.4701' in l and '1.0075' in l for l in _v),
           'verdict: naming the index it reached and the cap it missed')
        ok(all('feasible=' not in l for l in _v),
           'verdict: not `feasible=False`, which is not a phrase anyone typed')

        # --sgl-arch REACHES THE MAP SYNTHESISER.  The flag the architecture
        # refusal tells you to pass used to stop at the preset; the subprocess
        # read only SGL_ARCH, re-detected, failed, and the parent died on a
        # missing file.  Checked end to end, on the CLI both sides actually use.
        import subprocess as _sp
        _h = _sp.run([sys.executable, os.path.join(_HERE, 'convert_map_qtype.py'),
                      '--help'], capture_output=True, text=True).stdout
        ok('--sgl-arch' in _h,
           'arch: convert_map_qtype.py accepts --sgl-arch, not only the env var')
        _map = os.path.join(_MODELS, 'Qwen3.8-27B', 'group0', 'tensors.bf16.map')
        if os.path.isfile(_map):
            import tempfile as _tf, shutil as _sh
            with _tf.TemporaryDirectory() as _td:
                _in = os.path.join(_td, 'tensors.bf16.map')
                _sh.copy(_map, _in)
                # A WRONG SGL_ARCH in the environment, overridden by the flag.
                _env = dict(os.environ, SGL_ARCH='glm4_moe')
                _r = _sp.run([sys.executable,
                              os.path.join(_HERE, 'convert_map_qtype.py'), _in,
                              '--qtype', 'sgl_bf16', '--sgl-arch', 'qwen3_5',
                              '--ignore-imatrix-rules', '--with-imatrix'],
                             capture_output=True, text=True, env=_env)
                ok(_r.returncode == 0
                   and os.path.isfile(os.path.join(_td, 'tensors.sgl_bf16.map')),
                   'arch: --sgl-arch BEATS a stale SGL_ARCH and the map is written',
                   (_r.stderr or _r.stdout).strip().splitlines()[-1][:60]
                   if (_r.stderr or _r.stdout).strip() else '')
                _r2 = _sp.run([sys.executable,
                               os.path.join(_HERE, 'convert_map_qtype.py'), _in,
                               '--qtype', 'sgl_bf16', '--sgl-arch', 'glm4_moe',
                               '--ignore-imatrix-rules', '--with-imatrix'],
                              capture_output=True, text=True,
                              env=dict(os.environ, SGL_ARCH='qwen3_5'))
                ok(_r2.returncode != 0,
                   'arch: and a WRONG --sgl-arch refuses rather than guessing '
                   '(the flag really is what decides)')
        ok(QA.CONVERT_SGL_ARCH is None,
           'arch: unset until a run sets it, so nothing else changes behaviour')

    print('SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})')
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='What --speed-profile sglang decides, and why.')
    ap.add_argument('--selftest', action='store_true',
                    help='run the preset, detection and fill-rule invariants and exit')
    ap.add_argument('--explain', metavar='CSV',
                    help='print the plan for a model\'s calibration CSV')
    ap.add_argument('--sgl-arch', default=None, choices=sorted(SGN.ARCHS))
    ap.add_argument('--quant-degradation-csv', default=None)
    ap.add_argument('--nextn-optimization', default='off',
                    choices=('auto', 'on', 'off'))
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.explain:
        # THE SAME NAME SOURCE THE RUN USES, via the same function.
        names, whence = tensor_names_for(
            model_dir_for(a.explain, a.quant_degradation_csv), a.explain)
        plan = preset_plan(a.explain, names, arch_key=a.sgl_arch,
                           degradation_csv=a.quant_degradation_csv,
                           nextn_mode=a.nextn_optimization)
        plan['names_from'] = whence
        plan['n_names'] = len(names)
        # The plan holds REAL paths, because the run has to open those files.
        # What is printed is a record, and a record is published: it goes out
        # through the one renderer (sglang_native.py, section 0), so --explain
        # and a recipe footer say the same thing about the same run.
        for k in sorted(plan):
            print(f'{k:24s} {SGN.publishable_path(plan[k])}')
        return 0
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
