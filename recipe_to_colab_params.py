#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_assign.py the recipe maker tool of choice! Use it   **#
#** to produce recipes that can be cooked and used by others. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-13-2025 -------------------- **#
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
If a parameter is not present in the recipe, it is omitted.
"""

from __future__ import annotations
import argparse
import pathlib
import re
import shlex
import json
from typing import List, Dict, Any, Optional

REPLACE_FROM = "https://gguf.thireus.com"
REPLACE_TO = "https://github.com/Thireus/GGUF-Tool-Suite/"

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
    Fetch the multi-line command used block that follows a recipe comment header like:
    # - Command used:
    # ../../quant_assign.py ... \
    # ... \
    This function is conservative: it only collects comment lines that look like shell command parts.
    """
    marker = re.search(r"^#\s*-?\s*Command used:\s*$", text, re.MULTILINE)
    if not marker:
        marker = re.search(r"^#\s*Command used:\s*$", text, re.MULTILINE)
    if not marker:
        return None
    start = marker.end()
    lines: List[str] = []
    # iterate over following comment lines
    for m in re.finditer(r"^#(.*)$", text[start:], re.MULTILINE):
        content = m.group(1).rstrip()
        stripped = content.strip()
        # stop on blank comment or explicit THE END marker
        if stripped == "" or stripped.upper().startswith("THE END"):
            break

        # Heuristic: accept this comment line only if it looks like part of a shell command:
        # contains a flag (--), an assignment (=), a python/script reference (.py), a path (./ or ../),
        # or ends with a backslash continuation "\".
        if not re.search(r'(--|=|\.py|(^\./)|(^\.\./)|\\\s*$)', content):
            # If we've already collected some command-lines, stop collecting (this is likely trailing commentary).
            if lines:
                break
            # if we haven't collected anything yet, skip this non-command-looking line and continue searching
            continue

        # remove trailing backslash for safe joining, but keep it if regex needs it escaped
        lines.append(content.rstrip("\\").strip())

    if not lines:
        return None
    joined = " ".join(lines)
    return joined.strip()

def tokenize_command(cmd: str) -> List[str]:
    """Use shlex to split the command string into tokens (handles quoted tokens)."""
    try:
        tokens = shlex.split(cmd)
    except Exception:
        tokens = cmd.split()

    # Filter out noisy tokens that might slip through:
    filtered = []
    for t in tokens:
        if t == "#":
            continue
        # remove tokens that are purely punctuation (e.g. '---', '***')
        if re.fullmatch(r'^[\W_]+$', t):
            continue
        # remove short all-caps tokens commonly found in narrative endings ("THE", "END", "END!")
        if re.fullmatch(r'^[A-Z]{2,10}\W*$', t):
            continue
        filtered.append(t)
    return filtered

def collect_flag_values(tokens: List[str], flag: str) -> List[str]:
    """
    Return all tokens that belong to the given flag.
    For flags that accept multiple values, the values appear after the flag until next --something or end.
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
            # Continue searching (there might be repeated flags)
        else:
            i += 1
    return vals

def pretty_py_list(items: List[str]) -> str:
    """Format a list of regex strings as Python raw-string list, preserving backslashes."""
    if not items:
        return "[]"
    elems = []
    for it in items:
        s = it
        s_escaped = s.replace('"', '\\"')
        elems.append(f'r"{s_escaped}"')
    return "[{}]".format(", ".join(elems))

def emit_parameters(params: Dict[str, Any]) -> str:
    """Generate the Python snippet string from extracted params."""
    lines: List[str] = []
    lines.append('# @title ⚙️ Pipeline Parameters')
    if 'repo_url' in params:
        lines.append(f'repo_url = "{params["repo_url"]}"         #@param {{type:"string"}}')
    if 'model_name' in params:
        lines.append(f'model_name = "{params["model_name"]}"                                     #@param {{type:"string"}}')
    if 'model_link' in params:
        lines.append(f'model_link = "{params["model_link"]}"  #@param {{type:"string"}}')

    if any(k in params for k in ['gpu_tensors', 'cpu_tensors']):
        lines.append("")
        lines.append("# regex lists as Python lists of strings - Tensor names can be found in *.recipe file of the model directory")
        if 'gpu_tensors' in params:
            lines.append(f'gpu_tensors = {pretty_py_list(params["gpu_tensors"])}    #@param {{type:"raw"}}')
        if 'cpu_tensors' in params:
            lines.append(f'cpu_tensors = {pretty_py_list(params["cpu_tensors"])}   #@param {{type:"raw"}}')

    if 'cpu_quants' in params or 'gpu_quants' in params:
        lines.append("")
        lines.append("# quant types")
        if 'cpu_quants' in params:
            qlist = ", ".join(f'"{q}"' for q in params['cpu_quants'])
            lines.append(f'cpu_quants = [{qlist}]   #@param {{type:"raw"}}')
        if 'gpu_quants' in params:
            qlist = ", ".join(f'"{q}"' for q in params['gpu_quants'])
            lines.append(f'gpu_quants = [{qlist}]              #@param {{type:"raw"}}')

    if any(k in params for k in ['cpu_tensors_max_size', 'gpu_tensors_max_size', 'tolerance', 'exponential_factor']):
        lines.append("")
        lines.append("# sizes & tuning")
        if 'cpu_tensors_max_size' in params:
            lines.append(f'cpu_tensors_max_size = "{params["cpu_tensors_max_size"]}"    #@param {{type:"string"}}')
        if 'gpu_tensors_max_size' in params:
            lines.append(f'gpu_tensors_max_size = "{params["gpu_tensors_max_size"]}"    #@param {{type:"string"}}')
        if 'tolerance' in params:
            lines.append(f'tolerance = {params["tolerance"]}                #@param {{type:"number"}}')
        if 'exponential_factor' in params:
            lines.append(f'exponential_factor = {params["exponential_factor"]}          #@param {{type:"integer"}}')

    if any(k in params for k in ['gpu_assign_qtype', 'gpu_assign_tensors', 'cpu_assign_qtype', 'cpu_assign_tensors']):
        lines.append("")
        lines.append("# assignment override")
        if 'gpu_assign_qtype' in params:
            lines.append(f'gpu_assign_qtype = "{params["gpu_assign_qtype"]}"    #@param {{type:"string"}}')
        if 'gpu_assign_tensors' in params:
            lines.append(f'gpu_assign_tensors = {pretty_py_list(params["gpu_assign_tensors"])} #@param {{type:"raw"}}')
        if 'cpu_assign_qtype' in params:
            lines.append(f'cpu_assign_qtype = "{params["cpu_assign_qtype"]}"        #@param {{type:"string"}}')
        if 'cpu_assign_tensors' in params:
            lines.append(f'cpu_assign_tensors = {pretty_py_list(params["cpu_assign_tensors"])}        #@param {{type:"raw"}}')

    if 'harmonize_tensors' in params or 'harmonization_technique' in params:
        lines.append("")
        lines.append("# harmonization options (optional)")
        if 'harmonize_tensors' in params:
            inner = []
            for group in params['harmonize_tensors']:
                inner.append(pretty_py_list(group))
            groups_str = "[" + ", ".join(group for group in inner) + "]"
            lines.append(f'harmonize_tensors = {groups_str}   #@param {{type:"raw"}}')
        if 'harmonization_technique' in params:
            lines.append(f'harmonization_technique = {params["harmonization_technique"]}    #@param {{type:"integer"}}')

    if any(k in params for k in ['qtype', 'debug', 'info', 'ignore_f32', 'tensors_from_csv', 'cpu_irq_k', 'gpu_irq_k', 'skip_gpg']):
        lines.append("")
        lines.append("# additional flags (advanced and optional)")
        if 'qtype' in params:
            lines.append(f'qtype = "{params["qtype"]}"                  #@param {{type:"string"}}')
        if 'debug' in params:
            lines.append(f'debug = {params["debug"]}               #@param {{type:"boolean"}}')
        if 'info' in params:
            lines.append(f'info = {params["info"]}                #@param {{type:"boolean"}}')
        if 'ignore_f32' in params:
            lines.append(f'ignore_f32 = {params["ignore_f32"]}          #@param {{type:"boolean"}}')
        if 'tensors_from_csv' in params:
            lines.append(f'tensors_from_csv = {params["tensors_from_csv"]}    #@param {{type:"boolean"}}')
        if 'cpu_irq_k' in params:
            lines.append(f'cpu_irq_k = {params["cpu_irq_k"]}             #@param {{type:"number"}}')
        if 'gpu_irq_k' in params:
            lines.append(f'gpu_irq_k = {params["gpu_irq_k"]}             #@param {{type:"number"}}')
        if 'skip_gpg' in params:
            lines.append(f'skip_gpg = {params["skip_gpg"]}            #@param {{type:"boolean"}}')

    if 'display_graphs' in params:
        lines.append("")
        lines.append("# other pipeline parameters (optional)")
        lines.append(f'display_graphs = {params["display_graphs"]}       #@param {{type:"boolean"}}')

    return "\n".join(lines)

def parse_recipe_to_params(recipe_text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

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

    # Try to extract summary "GPU-loaded quants" and "CPU-loaded quants" tables as fallback for quants
    if "GPU-loaded quants" in recipe_text:
        m = re.search(r"##\s*GPU-loaded quants:([\s\S]*?)(?:\n##|\n#\s*CPU-loaded quants|\n## Summary|\n$)", recipe_text)
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

    if "CPU-loaded quants" in recipe_text:
        m = re.search(r"##\s*CPU-loaded quants:([\s\S]*?)(?:\n##|\n#\s*-Average BPW|\n## Summary|\n$)", recipe_text)
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

    # Extract command-used block and parse flags (preferred)
    cmd = extract_command_used_block(recipe_text)
    if cmd:
        tokens = tokenize_command(cmd)
        multi_flags = {
            '--cpu-tensors': 'cpu_tensors',
            '--gpu-tensors': 'gpu_tensors',
            '--cpu-quants': 'cpu_quants',
            '--gpu-quants': 'gpu_quants',
            '--gpu-assign-tensors': 'gpu_assign_tensors',
            '--cpu-assign-tensors': 'cpu_assign_tensors',
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
                        if "," in v and not v.startswith("blk\\."):
                            for part in v.split(","):
                                part = part.strip()
                                if part:
                                    items.append(part)
                        else:
                            items.append(v)
                    params[outname] = items

        single_flags = {
            '--gpu-assign-qtype': 'gpu_assign_qtype',
            '--cpu-assign-qtype': 'cpu_assign_qtype',
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

        hvals = collect_flag_values(tokens, '--harmonize-tensors')
        if hvals:
            groups: List[List[str]] = []
            for hv in hvals:
                parts = [p.strip() for p in hv.split(",") if p.strip()]
                if parts:
                    groups.append(parts)
            if groups:
                params['harmonize_tensors'] = groups

    # Fallback heuristics if necessary
    if 'gpu_quants' not in params:
        qtypes = sorted(set(re.findall(r"\b(iq[0-9]_[a-z0-9_]+|q[0-9]_[a-z0-9_]+|[+]?f32|[+]?q8_0)\b", recipe_text)))
        if qtypes:
            params['gpu_quants'] = qtypes[:3]

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
