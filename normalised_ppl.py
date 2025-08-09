#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** normalised_ppl.py just a useful math model, too long to   **#
#** explain...                                                **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-10-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.  ₖᵤ𝒹ₒₛ ₜₒ ᵢₖₐ𝓌ᵣₐₖₒ𝓌 & ᵤᵦₑᵣ𝓰ₐᵣₘ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

"""
Quantized Model Compensation Curve: Bits-per-Weight vs. 100% - Normalized Loss Increase

This script computes and plots the number of bits-per-weight required
to compensate for the loss incurred by a low-bit quantized model.
Based on a second-order Taylor approximation of the cross-entropy loss.

Requires:
    pip install numpy matplotlib

Usage:
    python3 quant_ppl_curve.py [--bit-min 1.5] [--bit-max 8.25] [--steps 300]
                              [--swap-axes] [--bpw-list 2.0 4.0 8.0]

Arguments:
    --bit-min     : Minimum bits-per-weight to evaluate (default: 1)
    --bit-max     : Maximum bits-per-weight to evaluate (default: 32)
    --steps       : Number of points in bits-per-weight grid (default: 300)
    --swap-axes   : Swap x and y axes in the plot
    --bpw-list    : List of bits-per-weight floats; outputs compensation fractions
                     for each (0–1, 6-digit precision) and exits

Interpretation:
    The curve shows, for a given compensation percentage (100% - normalized loss),
    how many bits-per-weight are required to achieve it.
"""

# Examples:
# python normalised_ppl.py --bpw-list 1 1.75 2.375 3.06 3.4375 4.25 5.5 6.5 8.25 32
# python normalised_ppl.py --bit-min 1.75 --bit-max 4.25 --swap-axes

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Argument parsing
group = argparse.ArgumentParser(
    description="Plot or compute bits-per-weight vs. compensation for quantization loss"
)
group.add_argument('--bit-min', type=float, default=1.0,
                   help='Minimum bits-per-weight (default: 1)')
group.add_argument('--bit-max', type=float, default=32.0,
                   help='Maximum bits-per-weight (default: 32)')
group.add_argument('--steps', type=int, default=300,
                   help='Points in bit range (default: 300)')
group.add_argument('--swap-axes', action='store_true',
                   help='Swap x and y axes in the plot')
group.add_argument('--bpw-list', nargs='+', type=float,
                   help='List of bits-per-weight; outputs compensation fractions')
args = group.parse_args()

# -----------------------------------------------------------------------------
# Model sensitivity parameters (tune as needed)
R = 1.0               # Dynamic range of weights
trace_H = 100.0       # Hessian trace approximation

# Constant in ΔL formula: ΔL(b) = C * 2^(-2b)
C = (R**2 / 6) * trace_H

# -----------------------------------------------------------------------------
# Function: compute compensation fraction for a list of b values

def compute_compensation(bpw_list):
    """
    Given a list of bits-per-weight floats, return a list of compensation
    fractions (0–1), normalized over the min/max of the provided bpw_list.
    """
    # Compute raw loss increase values
    delta_vals = C * 2 ** (-2 * np.array(bpw_list))
    # Normalize delta to [0,1]
    dmin, dmax = delta_vals.min(), delta_vals.max()
    delta_norm = (delta_vals - dmin) / (dmax - dmin)
    # Compensation fraction = 1 - normalized loss
    comp_frac = 1 - delta_norm
    return comp_frac

# If bpw-list is provided, compute and print compensation fractions, then exit
def handle_list_mode():
    comp = compute_compensation(args.bpw_list)
    for b_val, c in zip(args.bpw_list, comp):
        print(f"{b_val}: {c:.6f}")
    sys.exit(0)

if args.bpw_list:
    handle_list_mode()

# -----------------------------------------------------------------------------
# Standard plotting mode: generate bit grid and compensation curve
b = np.linspace(args.bit_min, args.bit_max, args.steps)
# Compute and normalize loss increase
delta_L = C * 2 ** (-2 * b)
dmin, dmax = delta_L.min(), delta_L.max()
delta_norm = (delta_L - dmin) / (dmax - dmin) * 100
# Compensation percentage = 100 - normalized loss
comp = 100 - delta_norm

# Equations for user reference
eq_b_given_p = (
    "b(p) = -0.5 * log2(((100 - p)/100*(Δmax - Δmin) + Δmin) / C)"
)
eq_p_given_b = (
    "p(b) = 100 - (C*2^(-2b) - Δmin)/(Δmax - Δmin)*100"
)

# -----------------------------------------------------------------------------
# Plotting function
def main():
    # Print equation forms
    print("Equation: bits given compensation%:", eq_b_given_p)
    print("Equation: compensation% given bits:", eq_p_given_b)

    plt.figure()
    if args.swap_axes:
        plt.plot(comp, b)
        plt.xlabel('Compensation Achieved (%)')
        plt.ylabel('Bits per Weight (b)')
        plt.xlim(0, 100)
        plt.ylim(args.bit_min, args.bit_max)
    else:
        plt.plot(b, comp)
        plt.xlabel('Bits per Weight (b)')
        plt.ylabel('Compensation Achieved (%)')
        plt.xlim(args.bit_min, args.bit_max)
        plt.ylim(0, 100)
        
    plt.title('Quantization Compensation Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()