"""
Real-model experiment: LLaMA-3.2-1B + pyramid quantization, no fine-tuning.

Approach:
  1. Download pretrained NousResearch/Llama-3.2-1B (from cache if available)
  2. Patch each MLP layer to support MatFormer-style weight slicing
  3. Apply quantization schemes in-place, evaluate, restore weights
  4. Compute perplexity on WikiText-2 validation at each subnetwork flag

Note on flag='s'/'m': absolute perplexity will be high (model not MatFormer-trained),
but the RELATIVE gap between pyramid and uniform_int4 is the meaningful signal.
"""

import torch
import math
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
import torch.nn.functional as F
from datasets import load_dataset

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Device: MPS (Apple Metal)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Device: CUDA ({torch.cuda.get_device_name()})")
else:
    device = torch.device('cpu')
    print("Device: CPU (will be slow — ~30 min)")

MODEL_NAME  = "NousResearch/Llama-3.2-1B"
FLAGS       = ['s', 'm', 'l', 'xl']
FLAG_FRACS  = {'s': 1/8, 'm': 1/4, 'l': 1/2, 'xl': 1.0}
BOUNDARIES  = [0, 1/8, 1/4, 1/2, 1.0]
REGION_FRACS = [1/8, 1/8, 1/4, 1/2]
SEQ_LEN     = 64
BATCH       = 1
N_EVAL      = 15   # batches; enough for ±5% perplexity estimate


# ── Sliced MLP forward (no subclass needed) ───────────────────────────────────
def make_sliced_forward(mlp, k):
    """Return a forward fn that uses only the first k intermediate dims."""
    def sliced_forward(x):
        g = F.linear(x, mlp.gate_proj.weight[:k])
        u = F.linear(x, mlp.up_proj.weight[:k])
        d = F.linear(mlp.act_fn(g) * u, mlp.down_proj.weight[:, :k])
        return d
    return sliced_forward


def patch_model_for_flag(model, flag):
    """Monkey-patch every MLP to slice at flag's fraction."""
    frac = FLAG_FRACS[flag]
    for layer in model.model.layers:
        mlp = layer.mlp
        k = int(mlp.intermediate_size * frac)
        mlp._original_forward = mlp.forward
        mlp.forward = make_sliced_forward(mlp, k)


def unpatch_model(model):
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, '_original_forward'):
            mlp.forward = mlp._original_forward
            del mlp._original_forward


# ── Fake quantization (per-channel — one scale per output neuron) ─────────────
# Per-tensor is catastrophic for real LLM weights: a few outlier values set a
# coarse scale that zeros out 40%+ of weights. Per-channel fixes this.
def fake_quantize_per_channel(tensor, bits, channel_dim=0):
    """Quantize each row (output channel) with its own scale."""
    if tensor.numel() == 0:
        return tensor
    orig_dtype = tensor.dtype
    t = tensor.float()
    n_levels = 2 ** (bits - 1) - 1
    # abs max per row
    reduce_dims = list(range(tensor.ndim))
    reduce_dims.pop(channel_dim)
    abs_max = t.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = abs_max / n_levels
    q = torch.clamp(torch.round(t / scale), -n_levels, n_levels) * scale
    return q.to(orig_dtype)


def apply_pyramid(model, bits_per_region):
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H = mlp.intermediate_size
            cuts = [int(H * b) for b in BOUNDARIES]
            for i, bits in enumerate(bits_per_region):
                lo, hi = cuts[i], cuts[i+1]
                if hi > lo:
                    # gate/up: rows are output channels (dim=0)
                    mlp.gate_proj.weight.data[lo:hi] = fake_quantize_per_channel(mlp.gate_proj.weight.data[lo:hi], bits, 0)
                    mlp.up_proj.weight.data[lo:hi]   = fake_quantize_per_channel(mlp.up_proj.weight.data[lo:hi], bits, 0)
                    # down: columns are output channels for the down_proj input; quantize per input-column (dim=1)
                    mlp.down_proj.weight.data[:, lo:hi] = fake_quantize_per_channel(mlp.down_proj.weight.data[:, lo:hi], bits, 1)


def avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


# ── Data ──────────────────────────────────────────────────────────────────────
def load_eval_batches(tokenizer):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
    text = ' '.join(t for t in ds['text'] if t.strip())
    tokens = tokenizer.encode(text)
    batches = []
    for i in range(0, N_EVAL * SEQ_LEN, SEQ_LEN):
        chunk = tokens[i: i + SEQ_LEN + 1]
        if len(chunk) < SEQ_LEN + 1:
            break
        batches.append(torch.tensor(chunk).unsqueeze(0))  # (1, SEQ_LEN+1)
    return batches[:N_EVAL]


# ── Perplexity ────────────────────────────────────────────────────────────────
def compute_ppl(model, batches, flag):
    patch_model_for_flag(model, flag)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            ids = ids.to(device)
            out = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                unpatch_model(model)
                return float('nan')
            total_loss  += out.loss.item() * ids[:, 1:].numel()
            total_tokens += ids[:, 1:].numel()
    unpatch_model(model)
    return math.exp(min(total_loss / total_tokens, 20))  # cap at e^20 to avoid overflow


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_NAME} (bf16)...")
    t0 = time.time()
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s  "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return model


def save_mlp_weights(model):
    """Save a CPU copy of all MLP weights for restoration."""
    saved = {}
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        saved[i] = {
            'gate': mlp.gate_proj.weight.data.cpu().clone(),
            'up':   mlp.up_proj.weight.data.cpu().clone(),
            'down': mlp.down_proj.weight.data.cpu().clone(),
        }
    return saved


def restore_mlp_weights(model, saved):
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            mlp.gate_proj.weight.data.copy_(saved[i]['gate'].to(device))
            mlp.up_proj.weight.data.copy_(saved[i]['up'].to(device))
            mlp.down_proj.weight.data.copy_(saved[i]['down'].to(device))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading eval data (WikiText-2 validation)...")
    eval_batches = load_eval_batches(tokenizer)
    print(f"  {len(eval_batches)} batches × {SEQ_LEN} tokens = {len(eval_batches)*SEQ_LEN} tokens")

    model = load_model()
    print("Saving original MLP weights...")
    original_weights = save_mlp_weights(model)

    schemes = {
        'fp32':            None,
        'uniform_int8':    [8, 8, 8, 8],
        'uniform_int4':    [4, 4, 4, 4],
        'pyramid_8-4-4-3': [8, 4, 4, 3],   # exact 4.0-bit budget
        'pyramid_8-5-4-4': [8, 5, 4, 4],   # INT4 floor, 4.625-bit
    }

    all_results = {}
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print("-" * 75)

    for name, bits in schemes.items():
        restore_mlp_weights(model, original_weights)
        if bits is not None:
            apply_pyramid(model, bits)
        ab = avg_bits(bits) if bits else 32.0

        t0 = time.time()
        ppls = {}
        for flag in FLAGS:
            ppls[flag] = compute_ppl(model, eval_batches, flag)
        elapsed = time.time() - t0

        all_results[name] = {'avg_bits': ab, 'ppl': ppls}
        row = f"{name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.1f}" for f in FLAGS)
        print(f"{row}  ({elapsed:.0f}s)")

    # ── Key ratios ────────────────────────────────────────────────────────────
    print("\n── Perplexity ratio vs fp32 (1.00 = lossless; lower is better) ──")
    fp32 = all_results['fp32']['ppl']
    print(f"{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print("-" * 75)
    for name, data in all_results.items():
        if name == 'fp32':
            continue
        ratios = {f: data['ppl'][f] / max(fp32[f], 1e-6) for f in FLAGS}
        row = f"{name:<22} {data['avg_bits']:>6.2f}  " + \
              "  ".join(f"{ratios[f]:>10.4f}" for f in FLAGS)
        print(row)

    print("\n── Key: pyramid_8-4-4-3 vs uniform_int4 at each flag ──")
    p = all_results['pyramid_8-4-4-3']['ppl']
    u = all_results['uniform_int4']['ppl']
    for flag in FLAGS:
        ratio = p[flag] / max(u[flag], 1e-6)
        verdict = 'PYRAMID WINS' if ratio < 1 else 'int4 wins'
        print(f"  flag={flag}: pyramid/int4 = {ratio:.4f}  {verdict}")

    return all_results


if __name__ == '__main__':
    results = main()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'real_model.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")
