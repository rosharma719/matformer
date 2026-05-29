"""
Shared quantization utilities for MatFormer pyramid quantization experiments.

Key concept: MatFormer MLP weight rows have usage frequency proportional to their
index. Row i is used by a subnetwork iff i < flag_fraction * intermediate_size.
The pyramid assigns higher bit precision to inner (high-frequency) rows.

Pyramid regions (matching MatFormer's 4 flags):
  rows 0   : H/8   (used by s, m, l, xl)  → highest bits
  rows H/8 : H/4   (used by m, l, xl)     → second
  rows H/4 : H/2   (used by l, xl)        → third
  rows H/2 : H     (used by xl only)      → lowest bits
"""

import math
import torch
import random
from transformers import LlamaConfig

# ── Constants ─────────────────────────────────────────────────────────────────
FLAGS        = ['s', 'm', 'l', 'xl']
FLAG_FRACS   = {'s': 1/8, 'm': 1/4, 'l': 1/2, 'xl': 1.0}
BOUNDARIES   = [0, 1/8, 1/4, 1/2, 1.0]
REGION_FRACS = [1/8, 1/8, 1/4, 1/2]   # fraction of H each region occupies

# Small config used in all toy experiments
TOY_CONFIG = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    vocab_size=128256,
)


# ── Quantization ──────────────────────────────────────────────────────────────
def fake_quantize_per_tensor(tensor, bits):
    """Symmetric per-tensor fake quantization. Fast but outlier-sensitive."""
    if tensor.numel() == 0:
        return tensor
    orig = tensor.dtype
    t = tensor.float()
    n = 2 ** (bits - 1) - 1
    scale = t.abs().max().clamp(min=1e-8) / n
    return torch.clamp(torch.round(t / scale), -n, n).mul(scale).to(orig)


def fake_quantize_per_channel(tensor, bits, channel_dim=0):
    """Per-channel fake quantization. Required for real pretrained LLM weights."""
    if tensor.numel() == 0:
        return tensor
    orig = tensor.dtype
    t = tensor.float()
    n = 2 ** (bits - 1) - 1
    reduce_dims = [d for d in range(tensor.ndim) if d != channel_dim]
    scale = t.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / n
    return torch.clamp(torch.round(t / scale), -n, n).mul(scale).to(orig)


def apply_pyramid(model, bits_per_region, per_channel=False):
    """
    Apply pyramid quantization to all MLP layers.

    bits_per_region: list of 4 ints [b0, b1, b2, b3] for regions
      [0:H/8, H/8:H/4, H/4:H/2, H/2:H]
    per_channel: use per-channel quantization (required for real pretrained models)
    """
    qfn = fake_quantize_per_channel if per_channel else fake_quantize_per_tensor
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            cuts = [int(H * b) for b in BOUNDARIES]
            for i, bits in enumerate(bits_per_region):
                lo, hi = cuts[i], cuts[i + 1]
                if hi <= lo:
                    continue
                mlp.gate_proj.weight.data[lo:hi] = qfn(mlp.gate_proj.weight.data[lo:hi], bits, 0)
                mlp.up_proj.weight.data[lo:hi]   = qfn(mlp.up_proj.weight.data[lo:hi],   bits, 0)
                mlp.down_proj.weight.data[:, lo:hi] = qfn(mlp.down_proj.weight.data[:, lo:hi], bits, 1)


def avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


def effective_bits_at_flag(bits_per_region, flag):
    """Average bit precision of weight rows actually used at this flag."""
    frac = FLAG_FRACS[flag]
    total_w, total_b = 0.0, 0.0
    for i, bits in enumerate(bits_per_region):
        lo, hi = BOUNDARIES[i], BOUNDARIES[i + 1]
        used = max(0.0, min(hi, frac) - lo)
        total_b += bits * used
        total_w += used
    return total_b / total_w if total_w > 0 else 0.0


# ── Standard schemes ──────────────────────────────────────────────────────────
SCHEMES = {
    'fp32':            None,
    'uniform_int8':    [8, 8, 8, 8],
    'uniform_int4':    [4, 4, 4, 4],
    'pyramid_8-7-6-4': [8, 7, 6, 4],   # 5.375 bits — high quality
    'pyramid_8-5-4-4': [8, 5, 4, 4],   # 4.625 bits — INT4 floor, wins at all flags
    'pyramid_8-4-4-3': [8, 4, 4, 3],   # 4.000 bits — iso-budget vs int4, wins at s/m/l
}


# ── Toy model training ────────────────────────────────────────────────────────
def build_toy_model(steps=20, device=None):
    """Build and warm-up a toy MatFormer model with random flag training."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model import ModifiedLlamaForCausalLM

    if device is None:
        device = torch.device('cpu')

    torch.manual_seed(42)
    random.seed(42)

    model = ModifiedLlamaForCausalLM(TOY_CONFIG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, TOY_CONFIG.vocab_size, (2, 32)).to(device)

    model.train()
    for _ in range(steps):
        flag = random.choice(FLAGS)
        model.configure_subnetwork(flag)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def measure_mse(model_q, model_ref, input_ids):
    """MSE of logits between quantized and FP32 model at each flag."""
    model_q.eval(); model_ref.eval()
    results = {}
    with torch.no_grad():
        for flag in FLAGS:
            model_ref.configure_subnetwork(flag)
            ref = model_ref(input_ids=input_ids).logits
            model_q.configure_subnetwork(flag)
            q   = model_q(input_ids=input_ids).logits
            results[flag] = ((ref - q) ** 2).mean().item()
    return results


def compute_perplexity(model, batches, flag, device=None):
    """Perplexity at a given MatFormer flag. Re-configures before each batch."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            model.configure_subnetwork(flag)
            ids = ids.to(device)
            out = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                return float('nan')
            total_loss  += out.loss.item() * ids[:, 1:].numel()
            total_tokens += ids[:, 1:].numel()
    return math.exp(min(total_loss / total_tokens, 20))


# ── Pretty printing ───────────────────────────────────────────────────────────
def print_mse_table(results: dict):
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>9}" for f in FLAGS))
    print("-" * 75)
    for name, data in results.items():
        ab   = data.get('avg_bits', 32.0)
        mses = data['mse']
        print(f"{name:<22} {ab:>6.2f}  " + "  ".join(f"{mses[f]:>9.5f}" for f in FLAGS))


def print_ppl_table(results: dict):
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print("-" * 78)
    for name, data in results.items():
        ab   = data.get('avg_bits', 32.0)
        ppls = data['ppl']
        print(f"{name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.1f}" for f in FLAGS))
