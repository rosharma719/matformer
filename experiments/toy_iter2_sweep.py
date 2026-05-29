"""
Iteration 2: Sweep inner_fraction and build Pareto frontier.

Key question: For a fixed bit budget, what inner_fraction minimises MSE at each
subnetwork flag? We expect the optimal inner_fraction for flag X to approximately
equal that flag's scale (s=1/8, m=1/4, l=1/2, xl=1).
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
    n_levels = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / n_levels
    return torch.clamp(torch.round(tensor / scale), -n_levels, n_levels) * scale


def apply_nested(model, inner_bits, outer_bits, inner_fraction):
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            k = int(H * inner_fraction)
            for w in [mlp.gate_proj.weight.data, mlp.up_proj.weight.data]:
                if k > 0:
                    w[:k] = fake_quantize(w[:k], inner_bits)
                if k < H:
                    w[k:] = fake_quantize(w[k:], outer_bits)
            dp = mlp.down_proj.weight.data
            if k > 0:
                dp[:, :k] = fake_quantize(dp[:, :k], inner_bits)
            if k < H:
                dp[:, k:] = fake_quantize(dp[:, k:], outer_bits)


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


def avg_bits(inner_bits, outer_bits, inner_fraction):
    return inner_fraction * inner_bits + (1 - inner_fraction) * outer_bits


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

    # FP32 baseline
    fp32_mse = {f: 0.0 for f in FLAGS}

    # INT4 uniform baseline
    int4_model = copy.deepcopy(base)
    apply_nested(int4_model, 4, 4, inner_fraction=0.5)  # uniform INT4
    int4_mse = measure_mse(int4_model, ref, eval_ids)

    # Sweep inner_fraction × {inner_bits=8, outer_bits=4}
    inner_fractions = [1/8, 1/4, 1/2, 3/4, 1.0]
    results = {}

    print(f"\n{'inner_frac':>12} {'avg_bits':>9}  " + "  ".join(f"{f:>9}" for f in FLAGS))
    print("-" * 78)

    for frac in inner_fractions:
        m = copy.deepcopy(base)
        apply_nested(m, inner_bits=8, outer_bits=4, inner_fraction=frac)
        mse = measure_mse(m, ref, eval_ids)
        ab = avg_bits(8, 4, frac)
        label = f"{frac:.3f}"
        results[label] = {'avg_bits': ab, 'mse': mse}
        row = f"{label:>12} {ab:>9.2f}  " + "  ".join(f"{mse[f]:>9.5f}" for f in FLAGS)
        print(row)

    # Also test: uniform INT8 and INT4 for reference
    for bits, name in [(8, 'uniform_int8'), (4, 'uniform_int4')]:
        m = copy.deepcopy(base)
        apply_nested(m, bits, bits, inner_fraction=0.5)
        mse = measure_mse(m, ref, eval_ids)
        ab = float(bits)
        results[name] = {'avg_bits': ab, 'mse': mse}
        row = f"{name:>12} {ab:>9.2f}  " + "  ".join(f"{mse[f]:>9.5f}" for f in FLAGS)
        print(row)

    # Per-flag: which inner_fraction minimises MSE?
    print("\n── Optimal inner_fraction per flag (among nested_8in_4out variants) ──")
    frac_keys = [f"{f:.3f}" for f in inner_fractions]
    for flag in FLAGS:
        best_frac = min(frac_keys, key=lambda k: results[k]['mse'][flag])
        best_mse = results[best_frac]['mse'][flag]
        expected = f"{FLAG_FRACTIONS[flag]:.3f}"
        match = "✓" if best_frac == expected else f"(expected {expected})"
        print(f"  flag={flag}: best inner_frac={best_frac}  MSE={best_mse:.5f}  {match}")

    return results


if __name__ == '__main__':
    results = main()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'toy_iter2.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
