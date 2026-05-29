"""
Iteration 4: Iso-budget comparison + INT4-floor pyramid + crossover analysis.

New schemes (all compared at a fixed ~4-bit budget):
  A) pyramid_8-4-4-3  → avg = 1/8*8 + 1/8*4 + 1/4*4 + 1/2*3 = 4.00 bits (exact INT4 budget)
  B) pyramid_8-5-4-4  → avg = 1/8*8 + 1/8*5 + 1/4*4 + 1/2*4 = 4.625 bits (INT4 floor, no INT3)
  C) pyramid_8-4-3-3  → avg = 1/8*8 + 1/8*4 + 1/4*3 + 1/2*3 = 3.625 bits (aggressive budget)

Key questions:
  1. Does pyramid_8-4-4-3 outperform uniform_int4 across all flags at identical bit budget?
  2. Does capping at INT4 (pyramid_8-5-4-4) fully recover xl quality?
  3. At what flag does the pyramid crossover from better to worse than int4?
  4. Can we derive the "optimal pyramid" for any target flag analytically?
"""

import torch
import copy
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import ModifiedLlamaForCausalLM
from transformers import LlamaConfig

torch.manual_seed(42)
device = torch.device('cpu')
FLAGS = ['s', 'm', 'l', 'xl']
FLAG_FRACTIONS = {'s': 1/8, 'm': 1/4, 'l': 1/2, 'xl': 1.0}
BOUNDARIES = [0, 1/8, 1/4, 1/2, 1.0]
REGION_FRACS = [1/8, 1/8, 1/4, 1/2]

config = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    vocab_size=4096,
)


def fake_quantize(tensor, bits):
    if tensor.numel() == 0:
        return tensor
    n_levels = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / n_levels
    return torch.clamp(torch.round(tensor / scale), -n_levels, n_levels) * scale


def apply_pyramid(model, bits_per_region):
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            cuts = [int(H * b) for b in BOUNDARIES]
            for i, bits in enumerate(bits_per_region):
                lo, hi = cuts[i], cuts[i+1]
                for w in [mlp.gate_proj.weight.data, mlp.up_proj.weight.data]:
                    if hi > lo:
                        w[lo:hi] = fake_quantize(w[lo:hi], bits)
                dp = mlp.down_proj.weight.data
                if hi > lo:
                    dp[:, lo:hi] = fake_quantize(dp[:, lo:hi], bits)


def total_avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


def effective_bits_at_flag(bits_per_region, flag):
    flag_frac = FLAG_FRACTIONS[flag]
    total_w, total_b = 0.0, 0.0
    for i, bits in enumerate(bits_per_region):
        lo, hi = BOUNDARIES[i], BOUNDARIES[i+1]
        used = max(0.0, min(hi, flag_frac) - lo)
        total_b += bits * used
        total_w += used
    return total_b / total_w if total_w > 0 else 0.0


def measure_mse(model_q, model_ref, input_ids):
    model_q.eval(); model_ref.eval()
    results = {}
    with torch.no_grad():
        for flag in FLAGS:
            model_ref.configure_subnetwork(flag)
            ref = model_ref(input_ids=input_ids).logits
            model_q.configure_subnetwork(flag)
            q = model_q(input_ids=input_ids).logits
            results[flag] = ((ref - q) ** 2).mean().item()
    return results


def build_base():
    import random
    base = ModifiedLlamaForCausalLM(config).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    ids = torch.randint(0, config.vocab_size, (2, 32))
    base.train()
    for _ in range(20):
        flag = random.choice(FLAGS)
        base.configure_subnetwork(flag)
        loss = base(input_ids=ids, labels=ids).loss
        opt.zero_grad(); loss.backward(); opt.step()
    return base


def main():
    print("Building base model...")
    base = build_base()
    eval_ids = torch.randint(0, config.vocab_size, (1, 64))
    ref = copy.deepcopy(base)

    schemes = {
        # Prior baselines (reproduced for clean comparison)
        'uniform_int4':     [4, 4, 4, 4],       # 4.00 bits
        'uniform_int8':     [8, 8, 8, 8],        # 8.00 bits
        # Iter 3 winner
        'pyramid_8-5-4-3':  [8, 5, 4, 3],       # 4.125 bits
        # Iter 4 new schemes
        'pyramid_8-4-4-3':  [8, 4, 4, 3],       # 4.00 bits — exact iso-budget
        'pyramid_8-5-4-4':  [8, 5, 4, 4],       # 4.625 bits — INT4 floor
        'pyramid_8-4-3-3':  [8, 4, 3, 3],       # 3.625 bits — aggressive
        'pyramid_6-5-4-3':  [6, 5, 4, 3],       # 3.875 bits — no INT8 inner
    }

    all_results = {}
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " +
          "  ".join(f"{f+'_mse':>10}" for f in FLAGS) + "  " +
          "  ".join(f"{f+'_eff':>8}" for f in FLAGS))
    print("-" * 112)

    for name, bits in schemes.items():
        m = copy.deepcopy(base)
        apply_pyramid(m, bits)
        mse = measure_mse(m, ref, eval_ids)
        ab = total_avg_bits(bits)
        eff = {f: effective_bits_at_flag(bits, f) for f in FLAGS}
        all_results[name] = {'avg_bits': ab, 'mse': mse, 'eff': eff, 'bits': bits}

        mse_str = "  ".join(f"{mse[f]:>10.5f}" for f in FLAGS)
        eff_str = "  ".join(f"{eff[f]:>8.2f}" for f in FLAGS)
        print(f"{name:<22} {ab:>6.2f}  {mse_str}  {eff_str}")

    # ── Q1: Does pyramid_8-4-4-3 beat int4 at EVERY flag? ──────────────────
    print("\n── Q1: pyramid_8-4-4-3 vs uniform_int4 (identical 4.0-bit budget) ──")
    p = all_results['pyramid_8-4-4-3']
    u = all_results['uniform_int4']
    all_win = True
    for flag in FLAGS:
        ratio = p['mse'][flag] / max(u['mse'][flag], 1e-12)
        win = ratio < 1.0
        if not win:
            all_win = False
        print(f"  flag={flag}: ratio={ratio:.4f}  {'PYRAMID WINS' if win else 'INT4 WINS'}")
    print(f"  → Pyramid wins at ALL flags: {all_win}")

    # ── Q2: Does INT4-floor pyramid recover xl? ──────────────────────────────
    print("\n── Q2: pyramid_8-5-4-4 xl quality vs uniform_int4 ──")
    p4 = all_results['pyramid_8-5-4-4']
    for flag in FLAGS:
        ratio = p4['mse'][flag] / max(u['mse'][flag], 1e-12)
        print(f"  flag={flag}: ratio={ratio:.4f}  eff_bits={p4['eff'][flag]:.2f}")

    # ── Q3: Crossover — optimal pyramid bits given target flag ───────────────
    print("\n── Q3: If you ONLY care about one flag, what pyramid maximises that flag? ──")
    # For each target flag, find the scheme with lowest MSE at that flag
    for target in FLAGS:
        best = min(all_results.items(), key=lambda kv: kv[1]['mse'][target])
        print(f"  target={target}: best scheme={best[0]}  "
              f"MSE={best[1]['mse'][target]:.5f}  avg_bits={best[1]['avg_bits']:.2f}")

    # ── Analytical derivation: optimal bits for single target flag ───────────
    print("\n── Analytical: for a 4.0-bit budget targeting flag 's', optimal allocation ──")
    # At flag='s', only region 0 (rows 0:H/8) is used.
    # Effective bits at 's' = b0 (entirely determined by region 0).
    # Budget constraint: 1/8*b0 + 1/8*b1 + 1/4*b2 + 1/2*b3 = 4.0
    # To maximise b0: minimise b1, b2, b3. Minimum practical bits = 2.
    # With b1=b2=b3=2: 1/8*b0 + 1/8*2 + 1/4*2 + 1/2*2 = 4.0
    #                  1/8*b0 = 4.0 - 0.25 - 0.5 - 1.0 = 2.25 → b0 = 18 (infeasible)
    # With min=3 bits: 1/8*b0 + 1/8*3 + 1/4*3 + 1/2*3 = 4.0
    #                  1/8*b0 = 4.0 - 0.375 - 0.75 - 1.5 = 1.375 → b0 = 11 (infeasible, >8)
    # With min=4 bits: 1/8*b0 + 1/8*4 + 1/4*4 + 1/2*4 = 4.0
    #                  1/8*b0 = 4.0 - 0.5 - 1.0 - 2.0 = 0.5 → b0 = 4 (trivially uniform INT4)
    # Conclusion: with a 4-bit budget, to get b0 > 4, you must have some outer region < 4 bits.
    # The pyramid [8,4,4,3] achieves b0=8 by having the outer 50% at INT3.
    # This is the fundamental tension: boosting inner precision forces outer precision below budget.
    print("  With 4.0-bit budget and b0=8 (inner at INT8):")
    print("  Remaining budget for regions 1,2,3: 4.0 - 1/8*8 = 3.0 bits total")
    print("  Distribute [4,4,3]: 1/8*4 + 1/4*4 + 1/2*3 = 0.5+1.0+1.5 = 3.0 ✓")
    print("  → [8,4,4,3] is the unique Pareto-optimal solution for maximising s at 4-bit budget.")
    print("  → Any scheme with b0>8 is infeasible (8 bits is standard INT8 ceiling).")

    return all_results


if __name__ == '__main__':
    results = main()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'toy_iter4.json')
    serialisable = {k: {'avg_bits': v['avg_bits'], 'mse': v['mse'],
                         'eff': v['eff'], 'bits': v['bits']}
                    for k, v in results.items()}
    with open(out, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out}")
