#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** recipe_to_colab_params.py turns recipe files into Google  **#
#** Colab pipeline parameters for quant_recipe_pipeline.ipynb **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Nov-12-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.       ₗₑₐₜₕₑᵣ₋ₚₒ𝓌ₑᵣₑ𝒹 𝒸ₒₘₚᵤₜᵢₙ𝓰 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

"""
recipe_to_colab_params.py

Read a .recipe file and emit a Python snippet containing Google Colab pipeline parameters.

This version:
- Replaces occurrences of https://gguf.thireus.com with the GitHub URL.
- Conservatively extracts the "# - Command used:" block.
- Ensures a complete default parameter set is present when missing, while preserving any
  values discovered in the recipe.
"""

from __future__ import annotations
import argparse
import pathlib
import re
import shlex
from typing import List, Dict, Any, Optional

REPLACE_FROM = "https://gguf.thireus.com"
REPLACE_TO = "https://github.com/Thireus/GGUF-Tool-Suite/"

# Complete defaults requested
DEFAULTS: Dict[str, Any] = {
    "repo_url": "https://github.com/Thireus/GGUF-Tool-Suite.git",
    "model_name": "DeepSeek-R1-0528",
    "model_link": "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
    "cpu_tensors": [],
    "gpu_tensors": [r".*"],
    "cpu_quants": ["iq4_ks", "iq3_k", "iq2_ks", "iq1_m_r4"],
    "gpu_quants": ["q8_0", "iq5_k_r4", "iq6_k"],
    "cpu_tensors_max_size": "230",
    "gpu_tensors_max_size": "95%",
    "tolerance": 0.01,
    "exponential_factor": 8,
    "cpu_assign_qtype": "",
    "cpu_assign_tensors": [],
    "gpu_assign_qtype": "iq4_xs",
    "gpu_assign_tensors": [r"^blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight$=q8_0"],
    "harmonize_tensors": [[r"^blk\..*\.ffn_up_exps.*", r"blk\..*\.ffn_gate_exps.*"]],
    "harmonization_technique": 3,
    "csv_filename": "",
    "qtype": "",
    "debug": False,
    "info": False,
    "ignore_f32": False,
    "tensors_from_csv": False,
    "cpu_irq_k": 1.5,
    "gpu_irq_k": 1.5,
    "skip_gpg": False,
    "display_graphs": True,
}


def error(msg: str) -> None:
    raise SystemExit(msg)


def load_recipe(path: pathlib.Path) -> str:
    if not path.exists():
        error(f"Error: file not found: {path}")
    if path.suffix.lower() != ".recipe":
        error(f"Error: file must have a .recipe extension: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Replace the requested URL when found
    if REPLACE_FROM in text:
        text = text.replace(REPLACE_FROM, REPLACE_TO)
    return text


def find_first(pattern: str, text: str, flags=0) -> Optional[str]:
    m = re.search(pattern, text, flags | re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_command_used_block(text: str) -> Optional[str]:
    """
    Conservative extraction of the '# - Command used:' comment block.
    Only collects comment lines that look like parts of a shell command
    (contain --, =, .py, ./ or ../, or end with a backslash).
    Stops when encountering blank or narrative trailing comments like 'THE END'.
    """
    marker = re.search(r"^#\s*-?\s*Command used:\s*$", text, re.MULTILINE)
    if not marker:
        marker = re.search(r"^#\s*Command used:\s*$", text, re.MULTILINE)
    if not marker:
        return None
    start = marker.end()
    lines: List[str] = []
    for m in re.finditer(r"^#(.*)$", text[start:], re.MULTILINE):
        content = m.group(1).rstrip()
        stripped = content.strip()
        # stop on blank comment or explicit THE END marker (narrative)
        if stripped == "" or stripped.upper().startswith("THE END"):
            break

        # Heuristic: accept this comment line only if it looks like part of a shell command:
        # contains a flag (--), an assignment (=), a python/script reference (.py), a path (./ or ../),
        # or ends with a backslash continuation "\".
        if not re.search(r'(--|=|\.py|(^\./)|(^\.\./)|\\\s*$)', content):
            if lines:
                # we've started collecting command-lines but this line doesn't look like command => stop
                break
            else:
                # haven't seen a real command-like line yet, continue (skip noise)
                continue

        # store trimmed line without trailing backslash (we'll join safely)
        lines.append(content.rstrip("\\").strip())

    if not lines:
        return None
    joined = " ".join(lines)
    return joined.strip()


def tokenize_command(cmd: str) -> List[str]:
    """Split the command string into tokens and filter common noisy/trailer tokens."""
    try:
        tokens = shlex.split(cmd)
    except Exception:
        tokens = cmd.split()

    filtered: List[str] = []
    for t in tokens:
        # ignore lone comment markers
        if t == "#":
            continue
        # ignore pure punctuation tokens
        if re.fullmatch(r'^[\W_]+$', t):
            continue
        # ignore short narrative ALL-CAPS tokens like THE, END, NOTE, etc.
        if re.fullmatch(r'^[A-Z]{2,10}\W*$', t):
            continue
        filtered.append(t)
    return filtered


def collect_flag_values(tokens: List[str], flag: str) -> List[str]:
    """
    Return tokens belonging to the given flag.
    Values are all tokens after the flag until the next token beginning with '--' or end.
    """
    vals: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == flag:
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                vals.append(tokens[i])
                i += 1
        else:
            i += 1
    return vals


def pretty_py_list(items: List[str]) -> str:
    """Format a list of strings as a Python raw-string list literal."""
    if not items:
        return "[]"
    elems = []
    for it in items:
        s = it
        s_escaped = s.replace('"', '\\"')
        elems.append(f'r"{s_escaped}"')
    return "[{}]".format(", ".join(elems))


def emit_parameters(params: Dict[str, Any]) -> str:
    """Generate the Python snippet string from extracted params (including defaults)."""
    # Use values from params which already have defaults applied in parse step
    lines: List[str] = []

    lines.append('# @title ⚙️ Pipeline Parameters')
    lines.append(f'repo_url = "{params["repo_url"]}"         #@param {{type:"string"}}')
    lines.append(f'model_name = "{params["model_name"]}"                                     #@param {{type:"string"}}')
    lines.append(f'model_link = "{params["model_link"]}"  #@param {{type:"string"}}')
    lines.append("")
    lines.append("# regex lists as Python lists of strings - CPU/GPU-friendly tensor names can be found in *.recipe file of the model directory")
    lines.append(f'cpu_tensors = {pretty_py_list(params["cpu_tensors"])}   #@param {{type:"raw"}}')
    lines.append(f'gpu_tensors = {pretty_py_list(params["gpu_tensors"])}    #@param {{type:"raw"}}')
    lines.append("")
    lines.append("# quant types for cpu-friendly and gpu-friendly tensor assignments")
    lines.append('cpu_quants = [{}]   #@param {{type:"raw"}}'.format(", ".join('"{}"'.format(q) for q in params["cpu_quants"])))
    lines.append('gpu_quants = [{}]              #@param {{type:"raw"}}'.format(", ".join('"{}"'.format(q) for q in params["gpu_quants"])))
    lines.append("")
    lines.append("# sizes & tuning")
    lines.append(f'cpu_tensors_max_size = "{params["cpu_tensors_max_size"]}"    #@param {{type:"string"}}')
    lines.append(f'gpu_tensors_max_size = "{params["gpu_tensors_max_size"]}"    #@param {{type:"string"}}')
    lines.append(f'tolerance = {params["tolerance"]}                #@param {{type:"number"}}')
    lines.append(f'exponential_factor = {params["exponential_factor"]}          #@param {{type:"integer"}}')
    lines.append("")
    lines.append("# assignment override")
    lines.append(f'cpu_assign_qtype = "{params["cpu_assign_qtype"]}"        #@param {{type:"string"}}')
    lines.append(f'cpu_assign_tensors = {pretty_py_list(params["cpu_assign_tensors"])}        #@param {{type:"raw"}}')
    lines.append(f'gpu_assign_qtype = "{params["gpu_assign_qtype"]}"    #@param {{type:"string"}}')
    lines.append(f'gpu_assign_tensors = {pretty_py_list(params["gpu_assign_tensors"])} #@param {{type:"raw"}}')
    lines.append("")
    lines.append("# harmonization options (optional)")
    lines.append("# harmonize_tensors: list-of-lists of regex strings; each inner list declares a group whose matching tensors (within a class) will be qtype harmonized layer-wise.")
    lines.append("# Default harmonizes ffn_up_exps and ffn_gate_exps fused pairs used by ik_llama.cpp (speed boost ~15%).")
    # harmonize_tensors nested list printing
    ht = "[" + ", ".join(pretty_py_list(group) for group in params["harmonize_tensors"]) + "]"
    lines.append(f'harmonize_tensors = {ht}   #@param {{type:"raw"}}')
    lines.append("# harmonization_technique: 0=disabled, 1=max, 2=mean, 3=min (default)")
    lines.append(f'harmonization_technique = {params["harmonization_technique"]}    #@param {{type:"integer"}}')
    lines.append("")
    lines.append("# calibration data filename (\"kld_results.csv\" or \"kld_results_partial.csv\" or \"ppl_results.csv\" or \"ppl_results_partial.csv\" are automatically used by default in this order when empty)")
    lines.append(f'csv_filename = "{params["csv_filename"]}" #@param {{type:"string"}}')
    lines.append("")
    lines.append("# calibration data qtype (leave empty for auto-selection which will choose the lowest bpw) - list of available qtypes can be found in the calibration data file")
    lines.append(f'qtype = "{params["qtype"]}"                  #@param {{type:"string"}}')
    lines.append("")
    lines.append("# additional flags (advanced and optional)")
    lines.append(f'debug = {params["debug"]}               #@param {{type:"boolean"}}')
    lines.append(f'info = {params["info"]}                #@param {{type:"boolean"}}')
    lines.append(f'ignore_f32 = {params["ignore_f32"]}          #@param {{type:"boolean"}}')
    lines.append(f'tensors_from_csv = {params["tensors_from_csv"]}    #@param {{type:"boolean"}}')
    lines.append(f'cpu_irq_k = {params["cpu_irq_k"]}             #@param {{type:"number"}}')
    lines.append(f'gpu_irq_k = {params["gpu_irq_k"]}             #@param {{type:"number"}}')
    lines.append(f'skip_gpg = {params["skip_gpg"]}            #@param {{type:"boolean"}}')
    lines.append("")
    lines.append("# other pipeline parameters (optional)")
    lines.append(f'display_graphs = {params["display_graphs"]}       #@param {{type:"boolean"}}')

    return "\n".join(lines)


def parse_recipe_to_params(recipe_text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # basic metadata
    mn = find_first(r"^#\s*Model name:\s*(.+)$", recipe_text, flags=re.MULTILINE)
    if mn:
        params['model_name'] = mn

    ml = find_first(r"^#\s*Link to the original model:\s*(\S+)", recipe_text, flags=re.MULTILINE)
    if ml:
        params['model_link'] = ml

    # repo_url best-effort: look for the tool suite URL in the header (after replacement this will be the GitHub URL if present)
    repo_candidate = find_first(r"Quant mix recipe created using [^-\n]*-\s*(https?://\S+)", recipe_text)
    if not repo_candidate:
        repo_candidate = find_first(r"^#\s*Quant mix recipe created.*(https?://\S+)", recipe_text, flags=re.MULTILINE)
    if repo_candidate:
        params['repo_url'] = repo_candidate.rstrip("/")

    # Parse GPU/CPU-friendly quants from the summary sections if present
    if "GPU-loaded quants" in recipe_text:
        m = re.search(r"##\s*GPU-loaded quants:([\s\S]*?)(?:\n##|\n#\s*CPU-[A-Za-z]+ quants|\n## Summary|\n$)", recipe_text)
        if m:
            lines = m.group(1).splitlines()
            gpu_qs = []
            for L in lines:
                Ls = L.strip()
                if Ls.startswith("#"):
                    t = Ls[1:].strip().split()
                    if t:
                        token = t[0].strip()
                        if re.match(r"[+A-Za-z0-9_]+", token):
                            gpu_qs.append(token)
            if gpu_qs:
                params['gpu_quants'] = gpu_qs

    if "CPU-" in recipe_text:
        m = re.search(r"##\s*CPU-[A-Za-z]+ quants:([\s\S]*?)(?:\n##|\n#\s*-Average BPW|\n## Summary|\n$)", recipe_text)
        if m:
            lines = m.group(1).splitlines()
            cpu_qs = []
            for L in lines:
                Ls = L.strip()
                if Ls.startswith("#"):
                    t = Ls[1:].strip().split()
                    if t:
                        token = t[0].strip()
                        if re.match(r"[+A-Za-z0-9_]+", token):
                            cpu_qs.append(token)
            if cpu_qs:
                params['cpu_quants'] = cpu_qs

    # Preferred: extract "Command used" block and parse flags from there
    cmd = extract_command_used_block(recipe_text)
    if cmd:
        tokens = tokenize_command(cmd)
        # Capture the first positional CSV filename (e.g. "ppl_results.csv" or "kld_results_partial_...csv")
        if 'csv_filename' not in params:
            for t in tokens:
                # accept .csv files (allow relative paths)
                if re.search(r"\.csv$", t, flags=re.IGNORECASE):
                    params['csv_filename'] = t
                    break
        multi_flags = {
            '--cpu-tensors': 'cpu_tensors',
            '--gpu-tensors': 'gpu_tensors',
            '--cpu-quants': 'cpu_quants',
            '--gpu-quants': 'gpu_quants',
            '--cpu-assign-tensors': 'cpu_assign_tensors',
            '--gpu-assign-tensors': 'gpu_assign_tensors',
        }
        for flag, outname in multi_flags.items():
            vals = collect_flag_values(tokens, flag)
            if vals:
                if outname in ('cpu_quants', 'gpu_quants'):
                    items = []
                    for v in vals:
                        for part in re.split(r"[,\s]+", v.strip()):
                            if part:
                                items.append(part)
                    params[outname] = items
                else:
                    items: List[str] = []
                    for v in vals:
                        # keep tokens like 'blk\...=q8_0' intact; if comma-separated, split
                        if "," in v and not v.startswith(("^blk\\.", "blk\\.")):
                            for part in v.split(","):
                                part = part.strip()
                                if part:
                                    items.append(part)
                        else:
                            items.append(v)
                    params[outname] = items

        single_flags = {
            '--cpu-assign-qtype': 'cpu_assign_qtype',
            '--gpu-assign-qtype': 'gpu_assign_qtype',
            '--cpu-tensors-max-size': 'cpu_tensors_max_size',
            '--gpu-tensors-max-size': 'gpu_tensors_max_size',
            '--tolerance': 'tolerance',
            '--exponential-factor': 'exponential_factor',
            '--harmonization-technique': 'harmonization_technique',
            '--cpu-irq-k': 'cpu_irq_k',
            '--gpu-irq-k': 'gpu_irq_k',
            '--qtype': 'qtype',
        }
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in single_flags:
                out = single_flags[t]
                i += 1
                if i < len(tokens):
                    val = tokens[i]
                    if re.fullmatch(r"-?\d+\.\d+", val):
                        try:
                            params[out] = float(val)
                        except Exception:
                            params[out] = val
                    elif re.fullmatch(r"-?\d+", val):
                        try:
                            params[out] = int(val)
                        except Exception:
                            params[out] = val
                    else:
                        params[out] = val
            i += 1

        # harmonize groups
        hvals = collect_flag_values(tokens, '--harmonize-tensors')
        if hvals:
            groups: List[List[str]] = []
            for hv in hvals:
                parts = [p.strip() for p in hv.split(",") if p.strip()]
                if parts:
                    groups.append(parts)
            if groups:
                params['harmonize_tensors'] = groups
            else:
                params['harmonize_tensors'] = ""

    # Fallback heuristics for gpu_quants if still missing
    if 'gpu_quants' not in params:
        qtypes = sorted(set(re.findall(r"\b(iq[0-9]_[a-z0-9_]+|q[0-9]_[a-z0-9_]+|[+]?f32|[+]?q8_0)\b", recipe_text)))
        if qtypes:
            params['gpu_quants'] = qtypes[:3]

    # Tidy numeric values
    if 'tolerance' in params:
        try:
            params['tolerance'] = float(params['tolerance'])
        except Exception:
            pass
    if 'exponential_factor' in params:
        try:
            params['exponential_factor'] = int(params['exponential_factor'])
        except Exception:
            pass
    if 'harmonization_technique' in params:
        try:
            params['harmonization_technique'] = int(params['harmonization_technique'])
        except Exception:
            pass

    # -----------------------------
    # Apply complete set of defaults for any missing keys (preserve any discovered values)
    # -----------------------------
    for k, v in DEFAULTS.items():
        params.setdefault(k, v)

    return params


def main():
    parser = argparse.ArgumentParser(description="Convert a .recipe file to Google Colab pipeline parameters")
    parser.add_argument("recipe", type=pathlib.Path, help=".recipe file to parse")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=None, help="Optional output file to write (defaults to stdout)")
    args = parser.parse_args()

    recipe_text = load_recipe(args.recipe)
    params = parse_recipe_to_params(recipe_text)
    if not params:
        error("No recognizable parameters found in recipe (nothing to translate).")

    snippet = emit_parameters(params)

    if args.out:
        args.out.write_text(snippet, encoding="utf-8")
        print(f"Wrote parameters snippet to {args.out}")
    else:
        print(snippet)


if __name__ == "__main__":
    main()
