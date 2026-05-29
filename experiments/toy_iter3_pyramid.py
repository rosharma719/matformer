"""
Iteration 3: Multi-boundary Quantization Pyramid

The core novel scheme: assign bits to weight rows according to how many
subnetworks use them. Rows used by all subnetworks get the most bits;
rows used only by xl get the fewest.

Pyramid scheme (matched to MatFormer's 4 flags):
  rows 0   : H/8  (used by s,m,l,xl) → b0 bits
  rows H/8 : H/4  (used by m,l,xl)   → b1 bits
  rows H/4 : H/2  (used by l,xl)     → b2 bits
  rows H/2 : H    (used by xl only)   → b3 bits

We test:
  A) Pyramid 8-7-6-4  (avg = 1/8*8 + 1/8*7 + 1/4*6 + 1/2*4 = 5.375 bits)
  B) Pyramid 8-6-5-4  (avg = 1/8*8 + 1/8*6 + 1/4*5 + 1/2*4 = 5.125 bits)
  C) Pyramid 8-5-4-3  (avg ≈ 4.0 bits — matches INT4 budget)
  D) Reverse pyramid 4-5-6-8 (inner=4, outer=8) — sanity check, should be worst

Compare against: uniform INT4 (≈4 bits), uniform INT8, nested_8in_4out (iter 1 winner).

Also compute: for each flag, the "effective quantization bits" = avg bits of the
rows actually used. The pyramid scheme should maximise effective bits at every flag
simultaneously, for a given total bit budget.
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
    """
    bits_per_region: list of 4 bit values for regions
      [b0, b1, b2, b3] covering [0:H/8, H/8:H/4, H/4:H/2, H/2:H]
    """
    boundaries = [0, 1/8, 1/4, 1/2, 1.0]
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            cuts = [int(H * b) for b in boundaries]
            for i, bits in enumerate(bits_per_region):
                lo, hi = cuts[i], cuts[i+1]
                for w in [mlp.gate_proj.weight.data, mlp.up_proj.weight.data]:
                    w[lo:hi] = fake_quantize(w[lo:hi], bits)
                dp = mlp.down_proj.weight.data
                dp[:, lo:hi] = fake_quantize(dp[:, lo:hi], bits)


def total_avg_bits(bits_per_region):
    fracs = [1/8, 1/8, 1/4, 1/2]
    return sum(f * b for f, b in zip(fracs, bits_per_region))


def effective_bits_at_flag(bits_per_region, flag):
    """Avg bits of weight rows actually used at this flag."""
    flag_frac = FLAG_FRACTIONS[flag]
    boundaries = [0, 1/8, 1/4, 1/2, 1.0]
    fracs = [1/8, 1/8, 1/4, 1/2]
    total_weight = 0.0
    total_bits = 0.0
    for i, bits in enumerate(bits_per_region):
        region_lo = boundaries[i]
        region_hi = boundaries[i+1]
        used_hi = min(region_hi, flag_frac)
        used_lo = region_lo
        if used_hi > used_lo:
            used_frac = used_hi - used_lo
            total_bits += bits * used_frac
            total_weight += used_frac
    return total_bits / total_weight if total_weight > 0 else 0.0


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
        'pyramid_8-7-6-4':  [8, 7, 6, 4],
        'pyramid_8-6-5-4':  [8, 6, 5, 4],
        'pyramid_8-5-4-3':  [8, 5, 4, 3],
        'reverse_4-5-6-8':  [4, 5, 6, 8],   # worst case sanity check
        'uniform_int8':     [8, 8, 8, 8],
        'uniform_int4':     [4, 4, 4, 4],
        'nested_8in_4out':  [8, 4, 4, 4],   # iter1 winner
    }

    all_results = {}
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f+'_mse':>10}" for f in FLAGS)
          + "  " + "  ".join(f"{f+'_effb':>8}" for f in FLAGS))
    print("-" * 110)

    for name, bits in schemes.items():
        m = copy.deepcopy(base)
        apply_pyramid(m, bits)
        mse = measure_mse(m, ref, eval_ids)
        ab = total_avg_bits(bits)
        eff = {f: effective_bits_at_flag(bits, f) for f in FLAGS}

        all_results[name] = {'avg_bits': ab, 'mse': mse, 'effective_bits': eff}

        mse_str = "  ".join(f"{mse[f]:>10.5f}" for f in FLAGS)
        eff_str = "  ".join(f"{eff[f]:>8.2f}" for f in FLAGS)
        print(f"{name:<22} {ab:>6.2f}  {mse_str}  {eff_str}")

    # Key comparison: pyramid_8-5-4-3 (≈ INT4 budget) vs uniform_int4
    print("\n── pyramid_8-5-4-3 vs uniform_int4 (same bit budget) ──")
    p = all_results['pyramid_8-5-4-3']
    u = all_results['uniform_int4']
    for flag in FLAGS:
        ratio = p['mse'][flag] / max(u['mse'][flag], 1e-12)
        eff_gain = p['effective_bits'][flag] - u['effective_bits'][flag]
        print(f"  flag={flag}: MSE ratio={ratio:.4f}  effective_bits gain={eff_gain:+.2f}")

    # Show effective bits per flag per scheme — this is the key insight table
    print("\n── Effective bits at each subnetwork (bits used in the forward pass) ──")
    print(f"{'Scheme':<22} " + "  ".join(f"{f:>8}" for f in FLAGS) + f"  {'avg':>8}")
    print("-" * 70)
    for name, data in all_results.items():
        eff = data['effective_bits']
        row = f"{name:<22} " + "  ".join(f"{eff[f]:>8.2f}" for f in FLAGS) + f"  {data['avg_bits']:>8.2f}"
        print(row)

    return all_results


if __name__ == '__main__':
    results = main()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'toy_iter3.json')

    # Convert to serialisable format
    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {
            'avg_bits': v['avg_bits'],
            'mse': v['mse'],
            'effective_bits': v['effective_bits'],
        }
    with open(out, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out}")
