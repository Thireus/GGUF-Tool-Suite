# `quant_assign.py` — Quantization Recipe Assignment

`quant_assign.py` decides, for every tensor in a model, **which GGUF quant type (qtype) it should be quantized to**, so that the whole model fits a size budget while losing as little quality (perplexity / KLD) as possible. Its output — a list of `^tensor_regex$=qtype` lines plus a footer — is a *recipe* that the companion `quants_regex_merger.sh` compacts and that downstream tooling uses to build the quantized GGUF.

This document explains the inputs, the three assignment methods (with a deep dive on the **auto** method and its **adaptive combo selection**), how to force a specific combo for troubleshooting, and the measured perplexity (PPL) results that validate the approach.

---

## 1. Inputs and core concepts

| Concept | What it is | Where it comes from |
|---|---|---|
| **Per-tensor sensitivity** | One number per tensor: how much quality degrades if *this* tensor is pushed to the reference (worst) qtype. High = important (e.g. `token_embd`, `output`). | The positional CSV (`kld_results.csv` or `ppl_results.csv`) — a single data row, columns = tensors. |
| **Degradation curve** | One number per qtype: the global quality cost of that qtype (lower bpw → higher cost). | `group0/kld_results.csv` (`--quant-degradation-csv`), an equation (`--quant-degradation-equation`), or the `--exponential-factor` model. |
| **bpw** | Bits-per-weight of a qtype (e.g. `q8_0`=8.5, `iq4_nl`=4.5, `iq2_xs`=2.31, `iq1_s`=1.56). | Built-in `BPW_TABLE`. |
| **Budget** | Target size (`--gpu-tensors-max-size`, `--cpu-tensors-max-size`), optionally split GPU/CPU. | CLI. |
| **Quant pool** | The candidate qtypes a tensor may use (`--gpu-quants`, `--cpu-quants`). | CLI. |
| **Harmonization** | Forcing related tensors (e.g. `ffn_up_exps` + `ffn_gate_exps`) to share a qtype. | `--harmonize-tensors`, `--harmonization-technique`. |

A recipe's **predicted damage** is the central quantity:

```
predicted_damage = Σ_tensor  sensitivity[tensor] · curve[assigned_qtype[tensor]]
```

Putting a high-sensitivity tensor on a low (high-curve) qtype spikes the damage — this single formula captures most of "what makes a recipe good or bad".

---

## 2. Assignment methods

### 2.1 Default (spread / midpoint)

The legacy method. Spreads tensors across the pool around a midpoint qtype. Simple but not optimum-seeking. *(In current versions a method flag is required; the bare default is deprecated.)*

### 2.2 Greedy — `--use-greedy-quant-assign`

A priority-queue optimizer. Every candidate move (downgrade/upgrade one tensor by one qtype) is scored by **degradation added per byte saved**:

```
score = sensitivity[t] · (deg[to_qtype] − deg[from_qtype]) / (bytes[from] − bytes[to])
```

It repeatedly applies the cheapest-per-byte downgrades until the budget is met, then promotes with any leftover headroom. Needs per-tensor degradation data (KLD), not just perplexity. `--exponential-factor` reshapes the curve (1.0 with a matching `group0` csv; otherwise auto-estimated).

### 2.3 Auto — `--use-auto-quant-assign` (recommended)

The data-adaptive method. Everything is tuned from the calibration data — no need to hand-pick exponents or a qtype window. Per class (GPU / CPU) it:

1. **Pins zero-kld outliers** at the smallest qtype.
2. **Detects high-sensitivity outliers** (e.g. `token_embd`) via IQR and pins each to a budget-progressive Pareto target, so the most-sensitive tensors walk smoothly through the Pareto frontier as the budget grows.
3. **Enumerates every contiguous sub-window** of the qtype pool and rank-maps tensors uniformly across each window.
4. Adds **constrained-greedy candidates** that exclude the worst-degradation qtypes one tier at a time (the meta picks the optimal "cap").
5. **Auto-tunes the score exponents (p, q)** over a fine grid.
6. Ranks all candidates with a **meta-score** `Σ (loss + mean_loss) · deg^p_meta`, where `p_meta = log2(max_pool_deg / max_loss)` auto-adjusts cliff-awareness.
7. Runs a **promotion pass (Phase C)** to consume leftover headroom.

The chosen `(p, q)` are disclosed in the recipe's hidden-parameters footer. When the **greedy candidate wins** the auto sweep, the auto method runs a second pass described next — this is where the adaptive combo selection lives.

---

## 3. The greedy-winner second pass & adaptive combo selection

When greedy wins, re-running *pure* greedy on the pristine inputs is usually best — but sometimes a small **alteration** to the inputs does better, and sometimes far worse. Instead of hard-coding one choice, the auto method **enumerates a small lattice of alteration combos, predicts which is best, and picks it safely.**

### 3.1 The alteration toggles (combos)

Three toggles are applied to the greedy inputs before re-running it:

- **`class`** — scale each bulk tensor's loss by its per-class factor.
- **`pos`** — boost the loss of first/last edge-layer tensors.
- **`tier2`** — demote "tier-2" statistical-outlier tensors to the floor qtype. This is the **GLM-style "free win"**: when the calibration data *over-rates* those outliers, crushing them frees budget for many bulk upgrades and lowers PPL. When the data is accurate, crushing them *hurts*.

(`pareto`, which prunes Pareto-dominated qtypes, is also available but is **inert** in every test so far — it never changes the recipe.)

The combos are numbered as a bitmask — see §4.

### 3.2 What makes a recipe good or bad (the learned patterns)

Benchmarking 38+ recipes across model families produced these patterns:

- **`predicted_damage` tracks PPL for "data-trustworthy" models** (Qwen): lower damage → lower PPL.
- **It is INVERTED for the GLM `tier2` win**: those recipes have the *highest* predicted damage (they crush "sensitive" tensors) yet the *best* PPL, because the sensitivity data over-rated those tensors.
- **Disaster signatures** (recipes that blow up):
  1. **`token_embd` crushed** to the floor (e.g. Qwen-397B `tier2`: `q8_0`→`iq1_kt`).
  2. **A top-sensitivity body tensor crushed** (e.g. Qwen-0.8B `tier2`: an early `attn_qkv` to `iq1_s`).
  3. **Starvation**: floor-demotion into a *tight* bpw body (`std_bpw / range_bpw < 0.275`) — no high-bpw reservoir to absorb the freed budget.
- **Good `tier2`-win signature**: floor-demotion into a *broad / bimodal*, left-skewed histogram (rich high-bpw body, few tensors below 2.5 bpw).

### 3.3 The selector

Per class, the selector enumerates the combos, runs greedy for each, then:

**Tier 1 — safety vetoes** (remove disasters; each real disaster is caught by ≥2):

- **V1 — token_embd crush**: veto if `token_embd` drops > 0.5 bpw vs the `none` baseline.
- **V2 — critical-tensor damage**: veto if a top-sensitivity tensor's single-tensor damage increase, normalized by the baseline total, exceeds a threshold.
- **V3 — starvation shape**: veto if it floor-demotes into a tight body (`std_bpw/range_bpw < 0.275`). *V3 alone perfectly separates the disasters in every test, with a comfortable margin.*

**Tier 2 — selection among survivors:**

- **2A — regime-gated `tier2` win-promotion.** A per-class statistic `HBR = (sens[token_embd] + sens[output]) / max(other sensitivities)` decides the regime. If `HBR < 5` (the data likely over-rates its outliers, GLM-style) and a *clean floor-pusher* survivor exists (broad body, left-skewed, few sub-2.5 tensors), promote it — preferring the **gentlest** such combo (plain `tier2`), so it never drifts onto aggressive variants that hit unsupported `iq1_m` at tight budgets.
- **2B — conservative.** Otherwise pick the lowest `predicted_damage` survivor, tie-broken by highest sensitivity-weighted bpw (protect sensitive tensors), preferring `none`. The existing meta-guard still protects cliff models.

**Why it is safe:** the win-promoter only ever runs on veto *survivors*, so it can never select a disaster — even if its regime gate is wrong, the worst case is a little regret, never a blow-up. Models where greedy does **not** win the sweep (e.g. gemma) never reach this code at all.

### 3.4 Environment knobs

- `ADAPT=0` — disable adaptive selection (legacy single-combo path).
- `ADAPT_LATTICE=class,pos,tier2` — which toggles to enumerate (default).
- `ADAPT_NO_TIER2=1` — keep the vetoes + conservative selector but disable win-promotion (strictly safest; forgoes the `tier2` wins).
- `SP_ALTS=<subset>` (+`SP_FORCE=1`) — legacy single-combo ablation harness.

---

## 4. Forcing a combo — `--auto-force-combo N` (troubleshooting)

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
- **When *not* forced**, the adaptively-chosen combo number is disclosed per class in the recipe's hidden-parameters footer, e.g. `#   Greedy 2nd-pass combo [GPU]: 4 (tier2) — adaptively selected; reproduce/override with --auto-force-combo 4`

This lets you (a) reproduce exactly what the adaptive selector chose, and (b) A/B-test any other combo against it.

---

## 5. Measurement results (perplexity, lower = better)

The selector was validated by generating every combo and measuring real PPL. **Adaptive pick** is what the algorithm chose without seeing the PPLs.

### 5.1 Cross-family scorecard

| Model (budget) | regime (HBR) | adaptive pick | adaptive PPL | best measured | gap |
|---|---|---|---|---|---|
| GLM-4.7-Flash 2.5107 | win-promote (4.46) | `none` | 10.9702 | class+pos 10.9266 | +0.044 (≈noise) |
| GLM-4.7-Flash 3.4836 | win-promote (4.46) | **`tier2`** | **9.4516** | **`tier2` 9.4516** | **best** |
| GLM-4.7-Flash 4.7669 | win-promote (4.46) | **`tier2`** | 8.8605 | pos+tier2 8.7940 | +0.066 |
| Qwen3.5-0.8B 4.2578 | conservative (7.94) | **`pos`** | **19.6428** | **`pos` 19.6428** | **best** |
| Qwen3.5-122B 2.5523 | conservative (17.0) | `none` | 5.4124 | class+pos 5.4094 | +0.003 (≈noise) |
| Qwen3.5-397B 6.8060 | conservative (5.55) | `none` | 3.4875 | pos 3.4861 | +0.0014 (≈noise) |
| DeepSeek-V3.1-Terminus 2.6115 | win-promote\* (2.12) | `none` | 3.7958 | tier2 3.7954 | +0.0004 (≈noise) |

\* DeepSeek is in the win-promote *regime*, but its quant pools are all ≥2.1 bpw (no sub-2.0 floor), so no clean-floor-pusher exists and the win-promoter correctly stays inactive → conservative `none`.

**In every case the adaptive pick is the best or statistically tied for best, and never a worse combo.** It captured the GLM `tier2` wins (−1.2 PPL), the Qwen-0.8B `pos` win, and on Qwen it **vetoed the `tier2` disasters** (see below).

### 5.2 GLM-4.7-Flash (win-promote regime — `tier2` is a big win)

`tier2` demotes outliers the calibration data over-rates; on GLM this frees budget for the bulk and **improves PPL by >1.0**:

| combo | 3.4836bpw | 4.7669bpw |
|---|---|---|
| none | 10.6441 | 10.0338 |
| pos | 10.6341 | 9.9779 |
| **tier2** | **9.4516** ⟵ adaptive | **8.8605** ⟵ adaptive |
| pos+tier2 | *(iq1_m, n/a)* | 8.7951 |

### 5.3 Qwen3.5 (conservative regime — `tier2` is a disaster, correctly vetoed)

| combo | Qwen-0.8B | Qwen-397B |
|---|---|---|
| none | 19.9252 | 3.4875 ⟵ adaptive |
| **pos** | **19.6428** ⟵ adaptive | 3.4861 |
| tier2 | **32.478** 💥 (V2/V3 vetoed) | **4.3515** 💥 (V1/V3 vetoed) |
| pos+tier2 | 33.206 💥 | 4.3520 💥 |

The vetoes removed every `tier2` blow-up; the conservative selector then picked the genuinely-best safe combo (`pos` on 0.8B, `none` on 397B).

### 5.4 DeepSeek-V3.1-Terminus 2.6115bpw (new family — conservative path generalizes)

A huge MoE with GPU/CPU offload and `--exponential-factor` degradation. All 8 combos at ~2.663 bpw:

| combo | PPL | | combo | PPL |
|---|---|---|---|---|
| tier2 | 3.7954 (best) | | pos | 3.8725 |
| **none** ⟵ adaptive | **3.7958** | | pos+tier2 | 3.8766 |
| class | 3.8525 | | class+pos | 3.8826 |
| class+tier2 | 3.8534 | | class+pos+tier2 | 3.8842 |

The adaptive pick `none` is tied for best (Δ0.0004, ~1/50th of the ±0.023 stderr) and correctly avoided the `class`/`pos` combos that are +0.06 to +0.09 worse. `tier2` is neutral here (high-bpw pool, win-promoter inactive), so picking `none` over the marginally-better `tier2` costs nothing.

---

## 6. Reading the recipe footer

The footer lists parameters that are otherwise hard to reproduce because they are auto-determined:

```
# - Hidden parameters (not passed as CLI args):
#   Greedy 2nd-pass combo [GPU]: 4 (tier2) — adaptively selected; reproduce/override with --auto-force-combo 4
#   Loss exponent (p) [GPU]: 8
#   Degradation exponent (q) [GPU]: 1
```

- **Greedy 2nd-pass combo** — the combo the adaptive selector chose for that class (only shown when *not* forced via `--auto-force-combo`).
- **Loss/Degradation exponents (p, q)** — the auto-swept score exponents.
- **Exponential factor** — shown when `--exponential-factor` was auto-computed.

---

## 7. Quick reference

| Flag / env | Effect |
|---|---|
| `--use-auto-quant-assign` | Data-adaptive auto method (recommended). |
| `--use-greedy-quant-assign` | Greedy priority-queue method. |
| `--quant-degradation-csv group0/kld_results.csv` | Per-qtype degradation curve. |
| `--exponential-factor F` | Reshape the degradation curve (default 8 / auto). |
| `--auto-deg-exponent q` | Pin the degradation exponent (else auto-swept). |
| `--auto-no-pareto-filter` | Disable Pareto pruning of per-tensor qtypes. |
| `--auto-force-combo N` | Force greedy 2nd-pass combo N (0–7 distinct; troubleshooting). |
| `--harmonize-tensors` / `--harmonization-technique` | Tie related tensors to one qtype. |
| `ADAPT=0` | Disable adaptive combo selection. |
| `ADAPT_NO_TIER2=1` | Keep vetoes + conservative; disable `tier2` win-promotion. |
| `ADAPT_LATTICE=class,pos,tier2` | Toggles the adaptive selector enumerates. |
