"""
Experiment: Nested Quantization for MatFormer

Hypothesis: In MatFormer's MLP, inner weights (rows 0:H/8, used by ALL subnetworks)
are more sensitivity-critical than outer weights (only used by larger subnetworks).
Assigning higher bit precision to inner weights should yield lower output error at
small subnetwork sizes compared to uniform low-bit quantization at the same avg bitrate.

Schemes tested:
  - FP32 (reference)
  - Uniform INT8
  - Uniform INT4
  - Nested INT8/INT4: inner 1/8 rows at INT8, outer 7/8 at INT4  (avg ~4.5 bits)
  - Nested INT4/INT8: inner 1/8 rows at INT4, outer 7/8 at INT8  (avg ~7.5 bits, sanity check)

Metric: MSE of logits vs FP32 reference at each subnetwork flag (s, m, l, xl).
"""

import torch
import copy
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import ModifiedLlamaForCausalLM
from transformers import LlamaConfig

# ── Small config for fast, CPU-tractable experimentation ─────────────────────
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    vocab_size=4096,
)

FLAGS = ['s', 'm', 'l', 'xl']
INNER_FRACTION = 1 / 8  # matches MatFormer's 's' subnetwork scale

torch.manual_seed(42)
device = torch.device('cpu')


# ── Fake quantization (symmetric, per-tensor) ────────────────────────────────
def fake_quantize(tensor, bits):
    n_levels = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / n_levels
    return torch.clamp(torch.round(tensor / scale), -n_levels, n_levels) * scale


# ── Apply quantization schemes ───────────────────────────────────────────────
def apply_uniform(model, bits):
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            mlp.gate_proj.weight.data = fake_quantize(mlp.gate_proj.weight.data, bits)
            mlp.up_proj.weight.data   = fake_quantize(mlp.up_proj.weight.data, bits)
            mlp.down_proj.weight.data = fake_quantize(mlp.down_proj.weight.data, bits)


def apply_nested(model, inner_bits, outer_bits, inner_fraction=INNER_FRACTION):
    """Quantize inner rows at inner_bits, outer rows at outer_bits."""
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            k = int(H * inner_fraction)

            for w in [mlp.gate_proj.weight.data, mlp.up_proj.weight.data]:
                w[:k]  = fake_quantize(w[:k],  inner_bits)
                w[k:]  = fake_quantize(w[k:],  outer_bits)

            dp = mlp.down_proj.weight.data
            dp[:, :k] = fake_quantize(dp[:, :k], inner_bits)
            dp[:, k:] = fake_quantize(dp[:, k:], outer_bits)


def avg_bits(inner_bits, outer_bits, inner_fraction=INNER_FRACTION):
    return inner_fraction * inner_bits + (1 - inner_fraction) * outer_bits


# ── Measure output MSE against FP32 reference ────────────────────────────────
def measure_mse(model_q, model_ref, input_ids):
    results = {}
    model_q.eval()
    model_ref.eval()
    with torch.no_grad():
        for flag in FLAGS:
            model_ref.configure_subnetwork(flag)
            ref_logits = model_ref(input_ids=input_ids).logits

            model_q.configure_subnetwork(flag)
            q_logits = model_q(input_ids=input_ids).logits

            results[flag] = ((ref_logits - q_logits) ** 2).mean().item()
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Initializing base model (random weights)...")
    base = ModifiedLlamaForCausalLM(config).to(device)

    # Simulate a few MatFormer training steps so weights are non-trivially structured
    import random
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-3)
    base.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    print("Warming up model with 20 MatFormer training steps...")
    for step in range(20):
        flag = random.choice(FLAGS)
        base.configure_subnetwork(flag)
        loss = base(input_ids=input_ids, labels=input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Final warmup loss: {loss.item():.4f}")

    # Build quantized variants from a frozen copy
    print("\nBuilding quantization variants...")
    schemes = {
        'fp32':              copy.deepcopy(base),
        'int8':              copy.deepcopy(base),
        'int4':              copy.deepcopy(base),
        'nested_8in_4out':  copy.deepcopy(base),  # inner=INT8, outer=INT4
        'nested_4in_8out':  copy.deepcopy(base),  # inner=INT4, outer=INT8 (sanity check)
    }

    apply_uniform(schemes['int8'],             bits=8)
    apply_uniform(schemes['int4'],             bits=4)
    apply_nested(schemes['nested_8in_4out'],   inner_bits=8, outer_bits=4)
    apply_nested(schemes['nested_4in_8out'],   inner_bits=4, outer_bits=8)

    avg_bits_nested_8_4 = avg_bits(8, 4)
    avg_bits_nested_4_8 = avg_bits(4, 8)
    print(f"  nested_8in_4out avg bits: {avg_bits_nested_8_4:.2f}")
    print(f"  nested_4in_8out avg bits: {avg_bits_nested_4_8:.2f}")

    # Evaluate — use a fresh batch
    eval_input = torch.randint(0, config.vocab_size, (1, 64))
    ref_model = schemes['fp32']

    print("\n── MSE vs FP32 at each subnetwork flag ──")
    print(f"{'Scheme':<22} {'avg_bits':>8}  " + "  ".join(f"{f:>8}" for f in FLAGS))
    print("-" * 70)

    all_results = {}
    bits_map = {
        'fp32': 32,
        'int8': 8,
        'int4': 4,
        'nested_8in_4out': avg_bits_nested_8_4,
        'nested_4in_8out': avg_bits_nested_4_8,
    }

    for name, model in schemes.items():
        if name == 'fp32':
            mses = {f: 0.0 for f in FLAGS}
        else:
            mses = measure_mse(model, ref_model, eval_input)
        all_results[name] = mses
        row = f"{name:<22} {bits_map[name]:>8.2f}  " + "  ".join(f"{mses[f]:>8.4f}" for f in FLAGS)
        print(row)

    # Key ratios: nested_8in_4out vs int4 at each flag
    print("\n── nested_8in_4out / int4 MSE ratio (< 1 = nested wins) ──")
    for flag in FLAGS:
        ratio = all_results['nested_8in_4out'][flag] / max(all_results['int4'][flag], 1e-12)
        print(f"  flag={flag}: {ratio:.4f}")

    return all_results


if __name__ == '__main__':
    results = main()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'toy_iter1.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
