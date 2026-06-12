# `quant_assign.py` — Per-Tensor Quantization Recipe Assignment

`quant_assign.py` decides, for every tensor in a model, **which GGUF quant type (qtype) it should be quantized to**, so that the whole model fits a size budget (your VRAM/RAM) while losing as little quality (perplexity / KLD) as possible. Its output — a list of `^tensor_regex$=qtype` lines plus a metadata header/footer — is a **recipe** that the rest of the GGUF Tool Suite turns into a downloadable, ready-to-run quantized model.

It is the "brain" of the suite: the calibration data tells it *how sensitive each tensor is*, the tensor maps tell it *how big each tensor is at each qtype*, and `quant_assign.py` solves the resulting knapsack-like problem with one of three algorithms.

> 🎉 A web-based port is available at **<https://gguf.thireus.com/quant_assign.html>** — same logic, no install. This document describes the command-line tool, which is the reference implementation.

---

## Table of contents

1. [Where it sits in the pipeline](#1-where-it-sits-in-the-pipeline)
2. [Inputs and core concepts](#2-inputs-and-core-concepts)
3. [Tensor classes, quant pools & size budgets](#3-tensor-classes-quant-pools--size-budgets)
4. [Assignment methods — overview & how to choose](#4-assignment-methods--overview--how-to-choose)
5. [Auto method — deep dive](#5-auto-method--deep-dive)
6. [Harmonization & synergy](#6-harmonization--synergy)
7. [Fallback & on-the-fly map computation](#7-fallback--on-the-fly-map-computation)
8. [The output recipe (format, footer, filename)](#8-the-output-recipe-format-footer-filename)
9. [Complete parameter reference](#9-complete-parameter-reference)
10. [Worked examples](#10-worked-examples)
11. [Tips & troubleshooting](#11-tips--troubleshooting)

---

## 1. Where it sits in the pipeline

`quant_assign.py` runs **after** you have benchmarked a model (to learn per-tensor sensitivity) and **before** you download/cook the quantized shards. It only emits text — it never touches the weights itself.

```
                 ┌────────────────────────────────────────────────┐
 benchmarking →  │  kld_results.csv     (per-tensor sensitivity)  │
 (several days   │  group0/kld_results.csv (per-qtype degradation)│ ──┐
  of GPU/CPU)    │  tensors.<qtype>.map (per-tensor byte sizes)   │   │
                 └────────────────────────────────────────────────┘   │
                                                                      ▼
                                                         ┌────────────────────┐
   your VRAM/RAM budget + qtype pool (CLI flags) ─────►  │   quant_assign.py  │
                                                         └────────┬───────────┘
                                                                  │  recipe (stdout)
                                                                  ▼
                                                  ┌─────────────────────────────┐
                                                  │ quants_regex_merger.sh      │  ← compacts the
                                                  │  (compact regexes + header) │     regex lines,
                                                  └────────┬────────────────────┘     names the file
                                                           │  .recipe file
                                                           ▼
                                                  ┌─────────────────────────────┐
                                                  │ quant_downloader.sh         │  ← fetches only the
                                                  │  (download matching shards) │     needed shards
                                                  └────────┬────────────────────┘
                                                           ▼
                                                  ik_llama.cpp / llama.cpp (run it)
```

Companion tools you will see referenced:

| Tool | Role |
|---|---|
| `benchmark_each_tensor.sh` / `collect_ppl_results.sh` | Produce the calibration CSV (`kld_results.csv` / `ppl_results.csv`). |
| `group0_enricher.py` | Fill gaps in the per-qtype degradation CSV (`group0/kld_results_partial.csv`). |
| `convert_map_qtype.py` | Compute a `tensors.<qtype>.map` from `tensors.bf16.map` when one is missing (see `--compute-*-map`). |
| `tensor_downloader.sh` | Fetch the `.map` files (and their GPG signatures) `quant_assign.py` needs. |
| `quants_regex_merger.sh` | Compact the raw recipe and write the final, named `.recipe` file (usually piped). |
| `quant_downloader.sh` | Download exactly the shards a recipe calls for. |
| `recipe_to_colab_params.py` | Turn a recipe back into Colab pipeline parameters. |

### Quick start

```bash
# Copy the chosen model's calibration data + download.conf into the working dir first
cp -f models/DeepSeek-R1-0528/download.conf .
cp -f models/DeepSeek-R1-0528/kld_results.csv .         # prefer kld_results.csv over ppl_results.csv

python quant_assign.py kld_results.csv \
  --gpu-tensors '.*' \
  --gpu-quants q8_0 iq6_k iq5_k_r4 iq4_k iq3_k iq2_k \
  --gpu-tensors-max-size 95% \
  --use-auto-quant-assign \
  --quant-degradation-csv group0/kld_results.csv \
  | ./quants_regex_merger.sh \
      --model-name "recipe_examples/ik_llama.cpp_recipes/DeepSeek-R1-0528" \
      --model-link "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"
```

> ⚠️ qtype names are **case-sensitive**: `q*_K` / `q*_KV` use a capital `K`/`KV`; everything else (`iq4_nl`, `q8_0`, …) is lowercase. See the README for the full per-framework qtype lists.

---

## 2. Inputs and core concepts

| Concept | What it is | Where it comes from |
|---|---|---|
| **Per-tensor sensitivity** | One number per tensor: how much quality degrades if *this* tensor alone is pushed to the reference (worst) qtype. High = important (e.g. `token_embd`, `output`). | The positional **calibration CSV** (`kld_results.csv` or `ppl_results.csv`) — rows = measured qtypes, columns = tensors. |
| **Degradation curve** | One number per qtype: the global quality cost of using that qtype everywhere (lower bpw → higher cost). | `group0/kld_results.csv` via `--quant-degradation-csv`; otherwise hardcoded Qwen3-4B-Thinking-2507 defaults. |
| **bpw** | Bits-per-weight of a qtype (e.g. `f32`=32, `q8_0`=8.5, `iq4_nl`=4.5, `iq3_xxs`=3.0625, `iq2_xs`=2.3125, `iq1_s`=1.5625). | Built-in `BPW_TABLE` (in `quant_assign.py`). |
| **Tensor size** | Exact byte size of each tensor at each qtype. | `tensors.<qtype>.map` files (fetched per qtype; GPG-verified unless `--skip-gpg`). |
| **Budget** | Target size, split GPU vs CPU. | `--gpu-tensors-max-size`, `--cpu-tensors-max-size`. |
| **Quant pool** | The candidate qtypes a tensor may use. | `--gpu-quants`, `--cpu-quants`. |
| **Class** | Whether a tensor is GPU-friendly or CPU-friendly (different pools/budgets). | `--gpu-tensors`, `--cpu-tensors`. |

### The calibration CSV (`csv_file`, positional — required)

The single required positional argument. Each **row** is a measured reference qtype; each **column** is a tensor name; each **cell** is the quality metric (KLD or PPL) observed when that tensor is dropped to that qtype while the rest of the model stays at baseline. A higher value means the tensor is **more sensitive** to quantization.

- **`kld_results.csv`** — Kullback–Leibler Divergence per tensor. **Preferred**: produces better recipes, and is required (in practice) by the greedy and auto methods.
- **`ppl_results.csv`** — perplexity per tensor. The older format; still works but is suboptimal.
- `--qtype` selects which row (reference qtype) to read; by default the lowest quant (smallest numeric prefix, e.g. `iq1_*` before `iq2_*`) that does **not** end in `_bn`.
- `--tensors-from-csv` makes the tensor list come from the CSV columns; by default the tensor list comes from the `.map` file instead (so tensors present in the map but missing from the CSV are still handled, via the assign-qtype fallback).
- **`0.0` vs `404` cells.** A cell of exactly `0.0` means the tensor *was measured* and has no effect → it is genuinely safe to crush to the smallest qtype. A cell of **`404`** is the **unmeasured** sentinel (a tensor type benchmarked with no imatrix data — e.g. a brand-new architecture's `indexer.*` / `nextn.*`). A `404` is **not** read as a literal sensitivity of 404; it inherits the kld of measured tensors of the same **type** (same name across layers), else the same **class** (first path component, e.g. the whole `indexer` group), and only if the whole class is unmeasured does it fall back to `--cpu/gpu-assign-qtype`. This prevents an unmeasured tensor from being mistaken for the most-important tensor in the model and hogging budget that belongs to `token_embd`/`output`.

> Generating a calibration CSV takes **days of GPU+CPU time** for large models — this is why the repo ships pre-computed CSVs under `models/<model>/`.

### The tensor maps (`tensors.<qtype>.map`)

These give the **exact byte size** of every tensor at a given qtype — what makes the size budget accurate rather than a bpw estimate. `quant_assign.py` fetches the maps it needs (one per qtype in your pools) into a temp dir and **GPG-verifies** them against `trusted-keys.asc` (skip with `--skip-gpg`). Missing maps can be computed locally with `--compute-missing-map` / `--compute-all-map` (see §7).

### The degradation curve (`--quant-degradation-csv`)

Used by the **greedy** and **auto** methods. A small CSV with a `QTYPE` column and a `group0` column giving each qtype's *global* degradation cost (values may be absolute or `%`). Typically `models/<model>/group0/kld_results.csv`. If omitted, hardcoded Qwen3-4B-Thinking-2507 values are used (and you will likely need to hand-tune `--exponential-factor`). Fill gaps with `group0_enricher.py`. *(The old `--quant-degradation-equation` flag is deprecated and ignored.)*

### The central quantity: `predicted_damage`

```
predicted_damage = Σ_tensor  sensitivity[tensor] · curve[assigned_qtype[tensor]]
```

Putting a high-sensitivity tensor on a low (high-curve) qtype spikes the damage. This single formula captures most of "what makes a recipe good or bad", and the methods below are different strategies for minimizing it under the size budget.

---

## 3. Tensor classes, quant pools & size budgets

These knobs are **shared by all three methods** and decide *what* is being optimized before *how*.

### Classes — `--gpu-tensors` / `--cpu-tensors`

Space-separated **regex** patterns (full-match — the whole tensor name must match) that split tensors into two classes, each with its own pool and budget. The classic MoE pattern keeps attention/dense on the GPU and offloads the big expert FFNs to the CPU:

```bash
--gpu-tensors '.*' \
--cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_(down|up|gate)_exps\.weight'
```

- A tensor is tested against the CPU patterns, then the GPU patterns.
- **Unmatched tensors default to the GPU class.** If you define `--cpu-tensors` but not `--gpu-tensors`, everything else silently becomes GPU-friendly — be explicit with `--gpu-tensors '.*'` to be safe.

### Pools — `--gpu-quants` / `--cpu-quants`

The ordered list of candidate qtypes for each class, e.g. `--gpu-quants q8_0 iq6_k iq5_k_r4 iq4_k iq3_k iq2_k`. The methods choose, per tensor, one qtype from its class pool. (Order matters mainly for the default method's outlier handling; greedy/auto treat the pool as a set ordered by bpw.) A class with an empty pool is skipped entirely.

### Budgets — `--gpu-tensors-max-size` / `--cpu-tensors-max-size`

The size target per class, in one of two forms:

- **Absolute GiB**: `--cpu-tensors-max-size 230` → 230 GiB.
- **Percent**: `--gpu-tensors-max-size 95%` → 95% of the class's *maximum* size (the size if every tensor in the class used the highest-bpw qtype in its pool, including f32 and pre-assigned tensors).

The effective optimization budget is the target **minus** what is already spoken for: IQR outliers, f32 tensors (unless `--ignore-f32`), and force-assigned / missing tensors are deducted first, and the remainder is what the algorithm distributes.

### Reference & fallback qtypes

| Flag | Purpose | Default |
|---|---|---|
| `--qtype` | Which CSV row (reference qtype) to read as sensitivity. | lowest quant (smallest numeric prefix) not ending in `_bn`. |
| `--gpu-assign-qtype` / `--cpu-assign-qtype` | qtype given to tensors that are **missing from the CSV** or otherwise unmeasured. | highest-bpw qtype in the class pool. |
| `--gpu-assign-tensors` / `--cpu-assign-tensors` | `regex=qtype` rules that **force** specific tensors to a qtype (first match wins). These are locked and excluded from optimization. | none |
| `--ignore-f32` | Exclude f32 tensors from the size totals. | off (f32 counted). |
| `--gpu-irq-k` / `--cpu-irq-k` | IQR multiplier *k* for outlier detection (Tukey bounds `Q1−k·IQR` / `Q3+k·IQR`). | 1.5 |

> `--gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'` is a common one: `attn_k_b` is a single-quant kernel that `llama-quantize` handles specially, so it is pinned.

---

## 4. Assignment methods — overview & how to choose

There are three algorithms. Pick one with a flag; if you pick none, you get the default method.

| | Default (spread) | Greedy | **Auto** (recommended) |
|---|---|---|---|
| Flag | *(none)* | `--use-greedy-quant-assign` | `--use-auto-quant-assign` |
| Needs degradation curve | no | yes (KLD) | yes (KLD) |
| Hand-tuning needed | `--exponential-factor` | `--exponential-factor` (0.3–5) | none — self-tunes |
| Optimum-seeking | no | yes | yes (+ safety) |
| Honours `--tolerance` | yes (±5% default) | **no** | **no** (always exact size) |
| Best for | quick/legacy runs | when you understand the curve | almost everything |

### 4.1 Default — spread / midpoint

Used when you pass **neither** method flag. It is the original, simplest method: it detects IQR outliers (pinning extremes to the lowest/highest qtype), computes a midpoint sensitivity for the class, then **spreads** the remaining tensors across the pool around that midpoint. It sweeps a *stretch* factor (1.0→10.0) and binary-searches the midpoint until the total lands within `--tolerance` of the budget. `--exponential-factor` (default **8** here) controls how aggressively the spread pushes toward the extreme qtypes.

It needs no degradation data, but it is **not optimum-seeking** — it does not know the relative cost of qtypes, only their ordering. It still works and is not deprecated, but greedy and (especially) auto produce better recipes, so it is now rarely used.

```bash
python quant_assign.py ppl_results.csv \
  --gpu-tensors '.*' --gpu-quants q8_0 q6_K q5_K q4_K \
  --gpu-tensors-max-size 80% --tolerance 0.05 --exponential-factor 8
```

### 4.2 Greedy — `--use-greedy-quant-assign`

A priority-queue optimizer. Every candidate move (downgrade one tensor by one qtype step) is scored by **degradation added per byte saved**:

```
score = sensitivity[t]^p · (deg[to_qtype] − deg[from_qtype]) / (bytes[from] − bytes[to])
```

where each tensor's sensitivity is first raised to the **loss exponent `p`** (= `--exponential-factor`) to reshape the curve. It repeatedly applies the cheapest-per-byte downgrades from a min-heap until the budget is met, then **promotes** tensors (a max-heap of upgrades) while any headroom remains. It needs per-tensor degradation (KLD — perplexity-only works poorly) plus a per-qtype degradation curve.

`--exponential-factor` reshapes the curve into a more linear space; its default depends on context:

- **with** `--quant-degradation-csv` → **1.0** (the curve is assumed to already match the model);
- **without** a curve CSV → auto-estimated as `y = 0.5·ln(bf16_total_GiB)` (fallback **3.0**).

Recommended manual range is **0.3–5.0**. `--per-tensor-degradation-scaling` (0 = off, ~0.0–0.5) additionally inflates the degradation of above-average-sensitivity tensors so they are protected.

```bash
python quant_assign.py kld_results.csv \
  --gpu-tensors '.*' --gpu-quants q8_0 iq6_k iq5_k_r4 iq4_k iq3_k \
  --gpu-tensors-max-size 110 \
  --use-greedy-quant-assign \
  --quant-degradation-csv group0/kld_results.csv \
  --exponential-factor 1.5
```

### 4.3 Auto — `--use-auto-quant-assign` (recommended)

The data-adaptive method: everything is tuned from the calibration data — no need to hand-pick an exponent or a qtype window. It enumerates many candidate recipes (rank-mapped pool windows, budget-aware splits, constrained-greedy variants), auto-sweeps the score exponents, and ranks them with a cliff-aware meta-score, with safety guardrails on top. It **always targets the exact budget** (it ignores `--tolerance`; pass `--info` to be reminded if you set both). Requires `--quant-degradation-csv` (strongly recommended) or it falls back to hardcoded defaults.

This is the default choice for new recipes. Because it has the most moving parts, it gets its own section below.

---

## 5. Auto method — deep dive

### 5.1 The core pipeline

Per class (GPU / CPU) the auto method, conceptually:

1. **Pins zero-kld outliers** at the smallest qtype (tensors the data says cost nothing to crush).
2. **Detects high-sensitivity outliers** (e.g. `token_embd`) via IQR and aims each at a budget-progressive **Pareto target**, so the most-sensitive tensors walk smoothly up the Pareto frontier as the budget grows.
3. **Enumerates every contiguous sub-window** of the qtype pool and rank-maps tensors uniformly across each window (the baseline candidate set), plus budget-aware 2-qtype splits.
4. Adds **constrained-greedy candidates** that exclude the worst-degradation qtypes one tier at a time, letting the meta pick the optimal "cap".
5. **Auto-tunes the score exponents (p, q)** over a coarse-then-fine grid to discover promising candidates.
6. Ranks all candidates with a cliff-aware **meta-score** (roughly `Σ (loss + mean_loss) · deg^p_meta`, where `p_meta = log2(max_pool_deg / max_loss) + 1`, clamped to ≥ 1, adjusts the steepness automatically — the `+1` amplifies the penalty on catastrophic qtypes), tie-broken by how much the outliers overshoot their target.
7. Runs a **promotion pass (Phase C)** to consume leftover headroom.

The chosen `(p, q)` are disclosed in the recipe footer (see §8). When the **greedy candidate wins** the auto sweep, a second pass runs — that is where the adaptive combo selection lives.

| Flag | Effect |
|---|---|
| `--auto-no-pareto-filter` | Disable the per-tensor Pareto pruning (drops qtypes that are *both* larger and more-degrading than another available one). Use only if a "worse-looking" qtype actually runs better on your hardware. |
| `--exponential-factor p` | Pin the loss exponent **p** instead of sweeping it. |
| `--auto-deg-exponent q` | Pin the degradation exponent **q** instead of sweeping it. |
| `--auto-force-combo N` | Force the greedy 2nd-pass combo (see §5.4). |

### 5.2 The greedy-winner second pass & adaptive combo selection

When greedy wins, re-running *pure* greedy on the pristine inputs is usually best — but sometimes a small **alteration** to the inputs does better, and sometimes far worse. Instead of hard-coding one choice, the auto method **enumerates a small lattice of alteration combos, predicts which is best, and picks it safely.**

**The alteration toggles (combos).** Three toggles are applied to the greedy inputs before re-running it:

- **`class`** — scale each bulk tensor's loss by its per-class factor.
- **`pos`** — boost the loss of first/last edge-layer tensors.
- **`tier2`** — demote "tier-2" statistical-outlier tensors to the floor qtype. This is the **GLM-style "free win"**: when the calibration data *over-rates* those outliers, crushing them frees budget for many bulk upgrades and lowers PPL. When the data is accurate, crushing them *hurts*.

(`pareto`, which prunes Pareto-dominated qtypes, also exists but is **inert** in every test so far — it never changes the recipe.)

**What makes a recipe good or bad (the learned patterns).** Benchmarking 38+ recipes across model families produced these patterns:

- **`predicted_damage` tracks PPL for "data-trustworthy" models** (Qwen): lower damage → lower PPL.
- **It is INVERTED for the GLM `tier2` win**: those recipes have the *highest* predicted damage (they crush "sensitive" tensors) yet the *best* PPL, because the sensitivity data over-rated those tensors.
- **Disaster signatures** (recipes that blow up):
  1. **`token_embd` crushed** to the floor (e.g. Qwen-397B `tier2`: `q8_0`→`iq1_kt`).
  2. **A top-sensitivity body tensor crushed** (e.g. Qwen-0.8B `tier2`: an early `attn_qkv` to `iq1_s`).
  3. **Starvation**: floor-demotion into a *tight* bpw body (`std_bpw / range_bpw < 0.275`) — no high-bpw reservoir to absorb the freed budget.
- **Good `tier2`-win signature**: floor-demotion into a *broad / bimodal*, left-skewed histogram (rich high-bpw body, few tensors below 2.5 bpw).

**The selector.** Per class, it runs greedy for each combo, then:

*Tier 1 — safety vetoes* (remove disasters; each real disaster is caught by ≥2):

- **V1 — token_embd crush**: veto if `token_embd` drops > 0.5 bpw vs the `none` baseline.
- **V2 — critical-tensor damage**: veto if a top-sensitivity tensor's single-tensor damage increase, normalized by the baseline total, exceeds a threshold.
- **V3 — starvation shape**: veto if it floor-demotes into a tight body (`std_bpw/range_bpw < 0.275`). *V3 alone perfectly separates the disasters in every test.*

*Tier 2 — selection among survivors:*

- **2A — regime-gated `tier2` win-promotion.** A per-class statistic `HBR = (sens[token_embd] + sens[output]) / max(other sensitivities)` decides the regime. If `HBR < 5` (the data likely over-rates its outliers, GLM-style) and a *clean floor-pusher* survivor exists (broad body, left-skewed, few sub-2.5 tensors), promote it — preferring the **gentlest** such combo (plain `tier2`), so it never drifts onto aggressive variants that hit unsupported `iq1_m` at tight budgets.
- **2A.5 — gated `class+pos` upgrade.** `class+pos` is the best conservative combo on several models (GLM-2.5107, Qwen3.5-122B, Qwen3.6-27B at tight budgets) but a *disaster* on a few (Qwen3.5-0.8B, DeepSeek) where it still survives the vetoes — and `predicted_damage` can't tell them apart. Two **per-class** features cleanly do: **`rel_crit`** (does it crush a critical tensor?) and the **quant-pool floor** (does the model actually use sub-2-bit quants?). So prefer `class+pos` only when it survives the vetoes **and `rel_crit < 0.05` and the pool floor < 2.0**; otherwise fall through.
- **2B — conservative.** Otherwise pick the lowest-`predicted_damage` survivor, tie-broken by highest sensitivity-weighted bpw (protect sensitive tensors), preferring `none`. A meta-guard still protects cliff models.

**Why it is safe:** the win-promoter only ever runs on veto *survivors*, so it can never select a disaster — even if its regime gate is wrong, the worst case is a little regret, never a blow-up. Models where greedy does **not** win the sweep (e.g. gemma) never reach this code at all.

**Configuration (hardcoded module constants near the top of `quant_assign.py`):**

- `ADAPT_ENABLED = True` — run the adaptive selector (set `False` for a faithful pure-greedy `none` fallback).
- `ADAPT_LATTICE = ['class','pos','tier2']` — which toggles the selector enumerates.
- `ADAPT_ALLOW_TIER2 = True` — allow the `tier2` win-promotion.
- `ADAPT_ALLOW_CLASSPOS = True` — allow the gated `class+pos` upgrade.
- `ADAPT_STARVE_PTS = 0.4` — the **ultra-tight "starvation" fallback**. At very low budgets (sub-~1.75 bpw on some MoE models) the budget is so tight that *every* combo floor-demotes into a tight body and the V3 starvation veto fires on all of them. In that degenerate all-vetoed case the lattice and `predicted_damage` become uninformative (the lowest-damage recipe can have the *worst* PPL), so instead of falling back to the under-protective `none` greedy the selector adopts a **sensitivity-protected greedy** — per-tensor degradation scaling `(loss/mean)^ADAPT_STARVE_PTS` (equivalent to `--use-greedy-quant-assign --per-tensor-degradation-scaling`). The footer then discloses the combo as `none+pts<k>`. Set `0` to restore the plain `none` fallback. This branch is reachable *only* when all combos are vetoed, so recipes that have any surviving combo are unaffected. (Measured: Qwen3.6-35B-A3B 1.7030bpw `none` 10.49 ppl → `pts` 10.06 ppl.)

To override one combo at runtime without editing source, use `--auto-force-combo N`.

### 5.3 Forcing a combo — `--auto-force-combo N` (troubleshooting)

By default the selector picks automatically. For troubleshooting you can **force** a specific combo and disable adaptive selection. `N` is a bitmask over `{class=1, pos=2, tier2=4, pareto=8}`:

| N | combo | | N | combo |
|---|---|---|---|---|
| **0** | **none** (pure greedy) | | 8 | pareto *(= 0)* |
| **1** | class | | 9 | pareto+class *(= 1)* |
| **2** | pos | | 10 | pareto+pos *(= 2)* |
| **3** | class+pos | | 11 | pareto+class+pos *(= 3)* |
| **4** | tier2 | | 12 | pareto+tier2 *(= 4)* |
| **5** | class+tier2 | | 13 | pareto+class+tier2 *(= 5)* |
| **6** | pos+tier2 | | 14 | pareto+pos+tier2 *(= 6)* |
| **7** | class+pos+tier2 | | 15 | pareto+class+pos+tier2 *(= 7)* |

`pareto` is **inert**, so 8–15 produce recipes identical to 0–7 — in practice use **0–7**. The flag only applies to `--use-auto-quant-assign` and only takes effect when greedy wins the auto sweep.

- **When forced**, an info log states the choice, e.g. `[AUTO] --auto-force-combo 4 (=tier2): forcing this greedy 2nd-pass alteration set; ADAPTIVE selection DISABLED.`
- **When *not* forced**, the adaptively-chosen combo number is disclosed per class in the recipe footer, e.g. `#   Greedy 2nd-pass combo [GPU]: 4 (tier2) — adaptively selected; reproduce/override with --auto-force-combo 4`

This lets you (a) reproduce exactly what the adaptive selector chose, and (b) A/B-test any other combo against it.

### 5.4 Measurement results (perplexity, lower = better)

The selector was validated by generating every combo and measuring real PPL. **Adaptive pick** is what the algorithm chose *without seeing the PPLs*.

`greedy (none)` is pure greedy — the method the original recipes used. **Δ vs greedy**: ✅ beats, `=` tie (±0.005), ✗ worse. **gap** is to the *global optimum* — the best of **all** measured combos, including the `pareto` variants the selector does not enumerate.

| model (bpw) | greedy `none` | adaptive pick | PPL | Δ vs greedy | global optimum | opt PPL | gap |
|---|---|---|---|---|---|---|---|
| GLM-4.7-Flash 2.5107 | 10.9702 | **class+pos** | 10.9266 | −0.044 ✅ | class+pos | 10.9266 | **optimal** |
| GLM-4.7-Flash 3.4836 | 10.6441 | **tier2** (win-promote) | 9.4516 | −1.19 ✅ | tier2 | 9.4516 | **optimal** |
| GLM-4.7-Flash 4.7669 | 10.0338 | tier2 (win-promote) | 8.8605 | −1.17 ✅ | pareto+pos+tier2 | 8.7940 | +0.066 |
| Qwen3.5-0.8B 4.2578 | 19.9252 | **pos** | 19.6428 | −0.282 ✅ | pos | 19.6428 | **optimal** |
| Qwen3.5-122B 2.5523 | 5.4124 | **class+pos** | 5.4094 | −0.003 = | class+pos | 5.4094 | **optimal** |
| Qwen3.5-397B 6.8060 | 3.4875 | **class+pos** | 3.4861 | −0.001 = | pos (=class+pos) | 3.4861 | **optimal** |
| DeepSeek-V3.1-T 2.6115 | 3.7958 | **none** | 3.7958 | 0 = | tier2 | 3.7954 | optimal (Δ0.0004) |
| Qwen3.6-27B 1.7005 | 10.9961 | **class+pos** | 10.9464 | −0.050 ✅ | class+pos | 10.9464 | **optimal** |
| Qwen3.6-27B 1.9980 | 8.8739 | **class+pos** | 8.8470 | −0.027 ✅ | class+pos | 8.8470 | **optimal** |
| Qwen3.6-27B 2.5507 | 7.8141 | class+pos | 7.8149 | +0.001 = | class | 7.7945 | +0.020 |
| Qwen3.6-27B 3.4009 | 7.0643 | **class+pos** | 7.0303 | −0.034 ✅ | pareto+class+pos | 7.0299 | optimal (Δ0.0004) |
| Qwen3.6-27B 4.2512 | 6.9148 | **none** | 6.9148 | 0 = | none | 6.9148 | **optimal** |
| Qwen3.6-27B 5.1014 | 6.9030 | **class+pos** | 6.9014 | −0.002 = | pareto+class+pos | 6.9012 | optimal (Δ0.0002) |
| Qwen3.6-27B 6.8018 | 6.9004 | class+pos | 6.9057 | +0.005 ✗ | pareto | 6.8975 | +0.008 |

**11/14 hit the optimum (or within ±0.0005); the adaptive beats-or-ties classic greedy everywhere except one noise-level budget (Qwen3.6 6.80, +0.005).** The 3 gaps to the global optimum:

- **GLM-4.7669 (+0.066):** the win-promoter deliberately picks the *gentlest* `tier2` over `pos+tier2` (the aggressive variants hit unloadable `iq1_m` at tighter budgets) — and the true optimum is a `pareto` combo the selector doesn't enumerate.
- **Qwen3.6 2.55 (+0.020):** picks `class+pos` (≈greedy) where `class` alone is best — a small overshoot at that one budget.
- **Qwen3.6 6.80 (+0.008):** at the highest budget `class+pos` slightly over-reaches; within measurement noise.

(DeepSeek is in the win-promote *regime* by HBR but its pools never go below 2.0 bpw, so the win-promoter and the gated `class+pos` upgrade both correctly stay inactive → conservative `none`.)

**GLM-4.7-Flash (win-promote regime — `tier2` is a big win):**

| combo | 3.4836bpw | 4.7669bpw |
|---|---|---|
| none | 10.6441 | 10.0338 |
| pos | 10.6341 | 9.9779 |
| **tier2** | **9.4516** ⟵ adaptive | **8.8605** ⟵ adaptive |
| pos+tier2 | *(iq1_m, n/a)* | 8.7951 |

**Qwen3.5 (conservative regime — `tier2` is a disaster, correctly vetoed):**

| combo | Qwen-0.8B | Qwen-397B |
|---|---|---|
| none | 19.9252 | 3.4875 ⟵ adaptive |
| **pos** | **19.6428** ⟵ adaptive | 3.4861 |
| tier2 | **32.478** 💥 (V2/V3 vetoed) | **4.3515** 💥 (V1/V3 vetoed) |
| pos+tier2 | 33.206 💥 | 4.3520 💥 |

The vetoes removed every `tier2` blow-up; the conservative selector then picked the genuinely-best safe combo.

---

### 5.5 Small-class protection floor (baked into the auto)

Single-tensor-drop kld **under-measures small, heavily-reused tensor classes** (attention projections, shared experts, dense FFN): their *absolute* kld is comparable to the giant MoE experts, yet they are 30–500× smaller — so they are *cheap* to keep at a high qtype, but the rank-based allocation crushes them and pours the budget into the experts. On models with a lightning-indexer / MLA attention block (e.g. GLM-5.1) this is catastrophic at tight budgets.

This is **baked into the auto method** (no flag), but **gated on calibration flatness**: it applies only when the measured class-mean klds span less than `AUTO_PROTECT_FLAT_MAX` (12×) — i.e. when the calibration is too *flat* to rank classes (GLM-5.1: 7.6×). On models whose calibration has real contrast (GLM-4.7-Flash 30×, Qwen3.6 87–252×) the rank allocation is trustworthy and the **entire legacy selector is kept byte-identical** — validation showed the floor regresses such models badly at tight budgets (Qwen3.6-35B-A3B 1.70 bpw: 10.06 → 19.14; Flash 3.48: 9.45 → 10.65). The same flatness gate covers every other mechanism in §5.5/§5.6 (V1 te-ratio exemption, V3 regime-skip, reworked win-promote, rebalance).

When the gate is open, the auto detects protected classes **purely from data** — a class whose mean parameter count is `< AUTO_PROTECT_SIZE_FRAC` (0.10) of the largest class **and** that has measured kld — and **floors** their allowed qtypes at `min(AUTO_PROTECT_K · target_bpw, AUTO_PROTECT_CAP_BPW)` (K = 3.0, cap ≈ iq6_k). No architecture names, no per-model knob; `target_bpw` makes it self-scaling across budgets. Tune via the `AUTO_PROTECT_*` module constants.

Key properties:
- **Self-limiting.** A *floor* only lifts crushed classes; where the recipe already exceeds it, dropping the unused low qtypes is a no-op — so well-calibrated models are unaffected.
- **Robust by construction.** The floor restricts `tensor_quants` *before* the pristine snapshots, so the window pass, the greedy 2nd-pass and the consistency-guard re-fits all inherit it. As a backstop, a **floor-enforcement** step re-fits via a floored greedy if any protected tensor still ends below the floor (a deep internal phase can violate it at isolated budgets, e.g. 25%).
- **Unused-tensor crushing is fallback-aware.** 0-kld (and same-class-inherited-0 404) tensors are pinned to the qtype with the smallest *advertised map size*, which already accounts for qtype→qtype fallback (see §7).

Measured (GLM-5.1, 565-chunk PPL): at 1.928 bpw the plain auto = 4.7027; with the floor = **4.5965**, *beating a hand-tuned recipe (4.6654)* fully automatically. K = 3.0 is the empirical optimum (K = 2.5 → 4.6162, K = 3.5 → 4.6124).

**Scope.** The floor addresses budgets where *attention* is the limiting class (tight budgets). At higher budgets (≥ `AUTO_T2_SAT_BPW` = 3.2 size-weighted bpw) a separate mechanism takes over — the high-budget rebalance (§5.6).

### 5.6 High-budget tier2 rebalance (flat-calibration models, win-promote path)

At plentiful budgets the tier2 greedy wins the selector but mis-shapes the recipe in ways that were each isolated and PPL-measured on GLM-5.1 @3.649 bpw (565 chunks). The rebalance (`_t2_rebalance`, fires only on **flat-calibration models** (§5.5 gate) inside the win-promote regime when the **job-level** size-weighted bpw ≥ 3.2, so all validated lower budgets and all contrast-calibration models are byte-identical) applies, in order:

1. **Expert-ratio cap (M2).** The kld-driven imbalance between the highest-kld bulk-expert class and its siblings overshoots with budget; the measured optimum decays as `r*(b) = clamp(1.37 − 0.10·(b − 2.976), 1.28, 1.45)` (ratio sweep: 1.49 → 2.9336, 1.32 → 2.8641). The demotion volume is solved analytically so the **post-respend** ratio lands on r*.
2. **Uniform 2-tier refill.** Within a bulk class the kld spread is noise; every measured winner is uniform-per-class while every greedy patchwork lost by ~0.06. Each side is refilled as one base qtype + edge-first layer upgrades to the next rung (mild edge preference — *stronger* edge-weighting measured worse).
3. **Head trim (M3).** The rank allocation pushes measured non-bulk classes to a q8_0/q6_K patchwork the data can't justify (~0.012 PPL). They are uniformized down to the floor-cap rung; classes whose **mean kld does not exceed the bulk experts' own** were floored structurally (small size), not by sensitivity, and sit one rung lower still (sub-floor; `AUTO_T2_SUBFLOOR` keeps the floor-enforcement from undoing it). Purely kld-based — no tensor names.
4. **Outlier lift (M1).** tier2-demoted *measured* outliers (e.g. token_embd) are lifted back toward their pre-demotion qtype, capped at ~1.25× budget bpw using **actual map sizes** (so fallback qtypes can't bypass the cap). 0-kld unused tensors stay crushed.
5. **Refill ladder hygiene.** The per-class qtype ladder uses actual map bytes (fallback-aware), the lowest-degradation name per size, and **statistical-tie pruning**: group0 is a single measurement per qtype, and when two rungs sit within `AUTO_LADDER_TIE_BYTES` (3%) in size and `AUTO_LADDER_TIE_DEG` (20%) in degradation the comparison is a coin flip that does not transfer (PPL-proven twice: q6_K-over-iq6_k and iq3_kt-over-iq3_ks were both wrong, costing 0.006/0.016). Within a tie the non-trellis ik quant is preferred over legacy/trellis — a preference over the quantizer's fixed qtype vocabulary, never over tensor names.

Guards: the rebalance requires ≥2 multi-tensor (≥8 members) bulk classes and skips when the highest-kld bulk class is harmonized with a sibling (e.g. GLM-4.7-Flash's gate↔up) — Flash's documented recipes regenerate byte-identically.

Measured (GLM-5.1 @3.649): plain tier2 2.9336 → rebalance iterations 2.9208/2.8820/2.8803 → final **2.8620**, beating the best hand-tuned recipe (2.8780) and the best manually-balanced reference (2.8641). With §5.5 + §5.6 the auto wins all seven validated GLM-5.1 budgets against hand-tuned recipes.

---

## 6. Harmonization & synergy

Two complementary features that couple related tensors. **Harmonization** can *force* identical qtypes; **synergy** only *nudges* losses to encourage similar qtypes.

### Harmonization — `--harmonize-tensors` / `--harmonization-technique`

Forces groups of related tensors to share a qtype **per layer**. The argument is a list of comma-joined regex groups; the **default** group fuses the MoE up/gate experts:

```
--harmonize-tensors '^blk\..*\.ffn_up_exps.*,blk\..*\.ffn_gate_exps.*'
```

This matters for speed: with **ik_llama.cpp** (`fmoe`), `ffn_up_exps` and `ffn_gate_exps` can be **fused** only when they share a qtype, giving **up to ~20% PP/TG speedup** on MoE models. The trade-off is slightly less dynamic-quant flexibility. Leave it at the default when targeting ik_llama.cpp.

`--harmonization-technique` chooses how the per-layer qtype is reconciled across the group:

| Value | Technique | Effect |
|---|---|---|
| 0 | disabled | no harmonization |
| 1 | max | uses the highest qtype — calibration accuracy not degraded (safest) |
| 2 | mean | a compromise |
| **3** | **min** *(default)* | uses the lowest qtype — degrades calibration accuracy but gives the **best results** in practice |

Disable harmonization with `--harmonize-tensors ""` or `--harmonization-technique 0`.

### Synergy — `--synergistic-tensors` / `--synergy-strength`

Softly pulls each group's per-tensor losses toward a size-weighted group average, encouraging (not enforcing) similar quantization within a layer. Only valid with greedy/auto. The **default** group covers all three expert matrices (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`), but it is **off by default** because `--synergy-strength` defaults to **0** (0 = disabled, 1 = fully averaged). Set a strength to activate it, e.g. `--synergy-strength 0.3`.

---

## 7. Fallback & on-the-fly map computation

**Fallback** handles tensors whose requested qtype isn't actually available in the map (a tensor that can't be quantized to that qtype "falls back" to whatever dtype the map really has).

- By default, `quant_assign.py` **inspects the `.map` files** and uses the real per-tensor sizes, substituting the map's actual qtype where there is a mismatch.
- `--no-fallback` disables that inspection and instead **guesses** sizes as if every tensor had been cleanly quantized to the requested qtype (using its bpw). Also forwarded to `convert_map_qtype.py`.

**On-the-fly maps**: if a `tensors.<qtype>.map` is missing you can compute it locally from `tensors.bf16.map`:

| Flag | Effect |
|---|---|
| `--compute-missing-map` | Compute only the maps that are missing. |
| `--compute-all-map` | Compute *all* non-bf16 maps (mutually exclusive with the above). |

Computed maps are **not** GPG-checked; their qtypes are annotated with a leading `!` in the recipe so you can tell them apart. The following are forwarded to `convert_map_qtype.py`:

| Flag | Effect |
|---|---|
| `--with-imatrix` | Declare an imatrix is available (satisfies imatrix checks). |
| `--ignore-imatrix-rules` | Skip imatrix-related checks. |
| `--fallback-quants iq2_xs IQ3_S …` | Whitelist qtypes allowed as fallbacks (case-insensitive). |
| `--fallback-quants-forbidden '^(iq1_|Q8_K$)' '.*_bn$'` | Regexes for qtypes that must **not** be used as fallbacks. |

---

## 8. The output recipe (format, footer, filename)

`quant_assign.py` writes the recipe to **stdout**; you normally pipe it into `quants_regex_merger.sh`, which compacts the regexes and writes the final `.recipe` file.

A recipe has three parts:

**1. Body — semantic groups of `^regex$=qtype` lines**, e.g.:

```
## Model head & embeddings — qbits: 32 4 3
^output_norm\.weight$=f32
^token_embd\.weight$=iq4_xs
^output\.weight$=iq3_s

## Multi-headed attention parameters — qbits: 32 8 6 4 3 2
^blk\.([0-9]|1[0-2]|42)\.attn_q_b\.weight$=q6_K
...
```

Tensors are grouped by function (head/embeddings, attention, dense FFN, MoE experts, gating, …) and, in the merged recipe, split into GPU-loaded vs CPU-friendly sections based on your `--gpu-tensors`/`--cpu-tensors` patterns.

**2. Header / summary** — comment lines recording how the recipe was made: per-class size totals and percentages, per-qtype tensor counts and bpw (with markers: `+` pre-assigned/f32, `*` fallback, `:` dynamic bpw, `!` computed map), and the SHA-256 of the script, the calibration CSV, the degradation CSV and every `.map` file, plus the GPG status.

**3. Footer — hidden parameters + command**, the values that are auto-determined and otherwise hard to reproduce:

```
# - GPG signatures: PASSED
# - Hidden parameters (not passed as CLI args):
#   Greedy 2nd-pass combo [GPU]: 4 (tier2) — adaptively selected; reproduce/override with --auto-force-combo 4
#   Loss exponent (p) [GPU]: 8
#   Degradation exponent (q) [GPU]: 1
# - Command used:
# /…/quant_assign.py kld_results.csv --gpu-tensors '.*' --gpu-quants … --use-auto-quant-assign …
```

- **Greedy 2nd-pass combo** — only shown when the auto method's greedy 2nd pass ran and the combo was *not* forced via `--auto-force-combo`.
- **Loss / Degradation exponents (p, q)** — the auto-swept score exponents (per class).
- **Exponential factor** — shown when it was auto-computed.

**Filename convention** (set by `quants_regex_merger.sh`):

```
<model>.<USER>-<bpw>bpw[-<ppl>ppl].<total>GB-GGUF_<gpu>GB-GPU_<cpu>GB-CPU.<scriptHash>_<cmdHash>.recipe
```

e.g. `GLM-4.7-Flash.THIREUS-2.5107bpw-0.0000ppl.8GB-GGUF_8GB-GPU_0GB-CPU.ca15905_13e9735.recipe`. The two short hashes are the **script SHA-256** (first 7 chars) and the **command SHA-256** (first 7 chars), so a recipe records exactly which script version and which command produced it.

---

## 9. Complete parameter reference

Run `python quant_assign.py --help` for the authoritative, always-current text. Grouped summary:

### Method selection

| Flag | Effect |
|---|---|
| *(none)* | Default spread/midpoint method. |
| `--use-greedy-quant-assign` | Greedy priority-queue method (needs KLD + degradation curve). |
| `--use-auto-quant-assign` | Data-adaptive auto method (recommended). Mutually exclusive with greedy. |

### Inputs & data

| Flag | Effect |
|---|---|
| `csv_file` *(positional)* | Calibration CSV (`kld_results.csv` preferred, or `ppl_results.csv`). |
| `--qtype` | Reference qtype/row to read from the CSV. Default: lowest non-`_bn`. |
| `--quant-degradation-csv` | Per-qtype degradation curve (`group0/kld_results.csv`). |
| `--tensors-from-csv` | Take tensor list from the CSV instead of the `.map` file. |
| `--skip-gpg` | Skip GPG verification of fetched maps. |
| `--ignore-f32` | Exclude f32 tensors from size totals. |

### Classes, pools & budgets

| Flag | Effect |
|---|---|
| `--gpu-tensors` / `--cpu-tensors` | Regexes defining each class (unmatched → GPU). |
| `--gpu-quants` / `--cpu-quants` | Ordered candidate qtype pool per class. |
| `--gpu-tensors-max-size` / `--cpu-tensors-max-size` | Budget per class (GiB or `%`). |
| `--gpu-assign-qtype` / `--cpu-assign-qtype` | qtype for unmeasured/missing tensors. Default: highest in pool. |
| `--gpu-assign-tensors` / `--cpu-assign-tensors` | `regex=qtype` force-assignments (first match wins). |
| `--gpu-irq-k` / `--cpu-irq-k` | IQR outlier multiplier *k*. Default 1.5. |

### Scoring / tuning

| Flag | Effect | Applies to |
|---|---|---|
| `--exponential-factor` | Loss exponent **p** / curve reshaping. Default 8 (default method), 1 or auto (greedy), swept (auto). | all |
| `--auto-deg-exponent` | Pin the degradation exponent **q** (else auto-swept). | auto |
| `--tolerance` | Relative GiB acceptance band (default 0.05). **Honoured only by the default method** (greedy and auto target the budget directly). | default |
| `--per-tensor-degradation-scaling` | Protect sensitive tensors (exponent, 0=off, ~0.0–0.5). | greedy, auto |
| `--auto-no-pareto-filter` | Disable Pareto pruning of per-tensor qtypes. | auto |
| `--auto-force-combo N` | Force greedy 2nd-pass combo (0–7; troubleshooting). | auto |

### Grouping

| Flag | Effect |
|---|---|
| `--harmonize-tensors` | List-of-regex-groups forced to share a qtype per layer. Default: `ffn_up_exps`+`ffn_gate_exps`. |
| `--harmonization-technique {0,1,2,3}` | 0=off, 1=max, 2=mean, **3=min (default)**. |
| `--synergistic-tensors` | List-of-regex-groups whose losses are softly averaged. Default: up/gate/down exps. |
| `--synergy-strength` | 0 (off, default) … 1 (fully averaged). |

### Fallback & map computation

| Flag | Effect |
|---|---|
| `--no-fallback` | Don't inspect maps for dtype mismatches; guess sizes from bpw. |
| `--compute-missing-map` / `--compute-all-map` | Compute maps via `convert_map_qtype.py` (`!`-annotated, mutually exclusive). |
| `--with-imatrix` / `--ignore-imatrix-rules` | Forwarded imatrix controls. |
| `--fallback-quants` / `--fallback-quants-forbidden` | Forwarded fallback whitelist / forbidden regexes. |

### Logging

| Flag | Effect |
|---|---|
| `--info` | Info-level logs (e.g. the `--tolerance`-ignored note, forced-combo note). |
| `--debug` | Verbose debug logs (trace candidate selection, vetoes, picked combo). |

### Hardcoded module constants (edit source to change)

`ADAPT_ENABLED`, `ADAPT_LATTICE`, `ADAPT_ALLOW_TIER2`, `ADAPT_ALLOW_CLASSPOS`, `ADAPT_STARVE_PTS` — near the top of `quant_assign.py`; control the auto method's adaptive combo selector and the ultra-tight starvation fallback (see §5.2).

`CONSISTENCY_GUARD`, `CONSISTENCY_PROBE_DELTA`, `CONSISTENCY_TOPK`, `CONSISTENCY_MARGIN_BPW` — the **cross-budget consistency guard** (auto only). After computing the recipe at the target, the auto method also computes it at a slightly lower and higher target (`±CONSISTENCY_PROBE_DELTA`; maps are already loaded so this is cheap). If a sensitive tensor (the top-`CONSISTENCY_TOPK` by sensitivity, plus `token_embd`/`output`) is assigned a qtype ≥ `CONSISTENCY_MARGIN_BPW` bpw **below what BOTH neighbours give it**, that is a budget-non-monotonic anomaly (e.g. a tensor sitting high at a lower *and* a higher budget but collapsing at one in between); it is pinned to the lower neighbour's qtype and the recipe is re-fit. Requiring both neighbours to agree means monotonic series are left byte-identical. Set `CONSISTENCY_GUARD = False` to disable (it costs two extra assignment passes per recipe).

---

## 10. Worked examples

**Auto (recommended), single GPU budget, MoE with fused experts:**

```bash
python quant_assign.py kld_results.csv \
  --gpu-tensors '.*' \
  --gpu-quants q8_0 iq6_k iq5_k_r4 iq4_k iq3_k iq2_k iq2_xs \
  --gpu-tensors-max-size 95% \
  --use-auto-quant-assign \
  --quant-degradation-csv group0/kld_results.csv \
  | ./quants_regex_merger.sh --model-name "<model>" --model-link "<hf-url>"
```

**Auto with a GPU/CPU split (big MoE: experts on CPU):**

```bash
python quant_assign.py kld_results.csv \
  --gpu-tensors '.*' \
  --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_(down|up|gate)_exps\.weight' \
  --gpu-quants q8_0 iq6_k iq5_k_r4 \
  --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 \
  --gpu-tensors-max-size 90% \
  --cpu-tensors-max-size 230 \
  --use-auto-quant-assign \
  --quant-degradation-csv group0/kld_results.csv \
  --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
```

**Greedy with an explicit curve and a protective exponent:**

```bash
python quant_assign.py kld_results.csv \
  --gpu-tensors '.*' --gpu-quants q8_0 iq6_k iq5_k_r4 iq4_k iq3_k \
  --gpu-tensors-max-size 110 \
  --use-greedy-quant-assign \
  --quant-degradation-csv group0/kld_results.csv \
  --exponential-factor 1.5 --per-tensor-degradation-scaling 0.3
```

**Default method (legacy, no curve needed):**

```bash
python quant_assign.py ppl_results.csv \
  --gpu-tensors '.*' --gpu-quants q8_0 q6_K q5_K q4_K \
  --gpu-tensors-max-size 80% --tolerance 0.05 --exponential-factor 8
```

**Reproduce / A-B test an auto combo (troubleshooting):**

```bash
# read the chosen combo from the recipe footer, then force it (or another) to compare
python quant_assign.py kld_results.csv … --use-auto-quant-assign --auto-force-combo 4 --info
```

---

## 11. Tips & troubleshooting

- **Prefer `kld_results.csv` over `ppl_results.csv`.** KLD gives per-qtype, per-tensor degradation granularity that greedy/auto need; PPL-only works poorly with those methods.
- **Always pass `--quant-degradation-csv`** for greedy/auto. Without it the hardcoded Qwen3-4B-Thinking-2507 curve is used and you will have to hand-tune `--exponential-factor`.
- **Two or three qtypes per tensor group is usually enough.** Huge pools mostly slow things down; the methods rarely need a dozen options to find the curve.
- **`--gpu-tensors '.*'` is the safe catch-all.** Unmatched tensors silently default to GPU; be explicit to avoid surprises.
- **Only the default method honours `--tolerance`.** Greedy ignores it; auto always targets the exact budget (run with `--info` for a reminder if you pass `--tolerance` alongside `--use-auto-quant-assign`).
- **Recipe blew up in PPL?** Check the footer's combo and the body for a crushed `token_embd` or a top-sensitivity body tensor at the floor — the classic disaster signatures (§5.2). Force `--auto-force-combo 0` (pure greedy) to get the safe baseline, then compare.
- **Use `--debug`** to trace the auto selector: the HBR regime, which combos were vetoed (V1/V2/V3), and which one was finally picked per class.
- **Fused-expert speed on ik_llama.cpp** comes from harmonizing `ffn_up_exps`+`ffn_gate_exps` (the default). Don't disable harmonization unless you have a reason to.
- **Reproducibility:** the recipe footer records the full command, and the filename's two hashes pin the script version and the exact command — keep recipes to reproduce a build later.
