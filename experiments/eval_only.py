#!/usr/bin/env python3
"""
MatFormer Pyramid Quantization — Eval on Properly Trained MatFormer Model

Uses atrost/climbmix-matformer-353m-1p2b-h100: a real MatFormer checkpoint
trained from scratch with elastic width on ClimbMix data.

Model specs (MLP intermediate sizes):
  s=512, m=1024, l=2048, xl=3584

Pyramid regions (matching actual MatFormer boundaries):
  rows 0:512    used by s,m,l,xl  → highest bits
  rows 512:1024 used by m,l,xl    → second
  rows 1024:2048 used by l,xl     → third
  rows 2048:3584 used by xl only  → lowest bits

Region fracs: [512/3584, 512/3584, 1024/3584, 1536/3584]
            ≈ [0.143,    0.143,    0.286,     0.429]

Naive MatGPTQ baseline = uniform per-channel INT4, ignoring row reuse.
Real MatGPTQ would use GPTQ second-order optimization on top of this.

Run:
    python experiments/eval_only.py
    python experiments/eval_only.py --device mps   # Mac
    python experiments/eval_only.py --batches 20   # faster
"""

import os, sys, json, math, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ── MatFormer model config ─────────────────────────────────────────────────────
MODEL_NAME   = "atrost/climbmix-matformer-353m-1p2b-h100"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'eval_only.json')

FLAGS      = ['s', 'm', 'l', 'xl']

# Actual intermediate sizes from matformer_specs
INTERMEDIATE = {'s': 512, 'm': 1024, 'l': 2048, 'xl': 3584}
MAX_INTER    = 3584

# Pyramid regions: absolute row boundaries
BOUNDARIES   = [0, 512, 1024, 2048, 3584]
REGION_FRACS = [b / MAX_INTER for b in [512, 512, 1024, 1536]]  # fraction of max each region

EVAL_BATCHES = 40
SEQ_LEN      = 512
BATCH_SIZE   = 4

# Quantization schemes — bits per region [r0, r1, r2, r3]
SCHEMES = {
    'fp32':               None,
    'uniform_int8':       [8, 8, 8, 8],
    'naive_matgptq_int4': [4, 4, 4, 4],   # MatGPTQ naive: uniform INT4, ignores reuse
    'pyramid_8-7-6-4':   [8, 7, 6, 4],   # 5.57 avg bits — high quality reference
    'pyramid_8-5-4-4':   [8, 5, 4, 4],   # 4.86 avg bits — INT4 floor
    'pyramid_8-4-4-3':   [8, 4, 4, 3],   # 3.86 avg bits — aggressive
    'pyramid_6-5-4-3':   [6, 5, 4, 3],   # 3.71 avg bits — very aggressive
}


def avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


# ── Quantization ──────────────────────────────────────────────────────────────

def fake_quant_per_channel(tensor, bits, channel_dim=0):
    if tensor.numel() == 0:
        return tensor
    orig  = tensor.dtype
    t     = tensor.float()
    n     = 2 ** (bits - 1) - 1
    dims  = [d for d in range(t.ndim) if d != channel_dim]
    scale = t.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8) / n
    return (torch.clamp(torch.round(t / scale), -n, n) * scale).to(orig)


def apply_pyramid_mlp(model, bits_per_region):
    """Apply pyramid quantization to MLP rows only."""
    with torch.no_grad():
        for layer in model.layers:
            for i, bits in enumerate(bits_per_region):
                lo, hi = BOUNDARIES[i], BOUNDARIES[i + 1]
                if hi <= lo:
                    continue
                layer.gate_proj.weight.data[lo:hi] = fake_quant_per_channel(
                    layer.gate_proj.weight.data[lo:hi], bits, 0)
                layer.up_proj.weight.data[lo:hi] = fake_quant_per_channel(
                    layer.up_proj.weight.data[lo:hi], bits, 0)
                layer.down_proj.weight.data[:, lo:hi] = fake_quant_per_channel(
                    layer.down_proj.weight.data[:, lo:hi], bits, 1)


def save_mlp_weights(model):
    return {i: {
        'gate': layer.gate_proj.weight.data.cpu().clone(),
        'up':   layer.up_proj.weight.data.cpu().clone(),
        'down': layer.down_proj.weight.data.cpu().clone(),
    } for i, layer in enumerate(model.layers)}


def restore_mlp_weights(model, saved, device):
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.gate_proj.weight.data.copy_(saved[i]['gate'].to(device))
            layer.up_proj.weight.data.copy_(saved[i]['up'].to(device))
            layer.down_proj.weight.data.copy_(saved[i]['down'].to(device))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_eval_batches(tokenizer, n_batches):
    ds   = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
    text = ' '.join(t for t in ds['text'] if t.strip())
    ids  = tokenizer.encode(text)
    step = BATCH_SIZE * SEQ_LEN
    batches = []
    for i in range(0, len(ids) - step, step):
        chunk = ids[i: i + step]
        if len(chunk) < step:
            break
        batches.append(torch.tensor(chunk).view(BATCH_SIZE, SEQ_LEN))
        if len(batches) >= n_batches:
            break
    return batches


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_ppl(model, batches, flag, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            # Pass full sequence as both input and labels — model does the shift internally.
            # (Passing pre-shifted ids[:,:-1]/ids[:,1:] causes a double-shift bug.)
            ids  = batch.to(device)
            dtype = torch.bfloat16 if device.type in ('cuda', 'mps') else torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type in ('cuda', 'mps')):
                out = model(input_ids=ids, labels=ids, matformer_size=flag)
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                return float('nan')
            n_pred = (ids.shape[1] - 1) * ids.shape[0]   # model evaluates SEQ_LEN tokens
            total_loss   += out.loss.item() * n_pred
            total_tokens += n_pred
    return math.exp(min(total_loss / total_tokens, 20))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',  default=None)
    parser.add_argument('--batches', type=int, default=EVAL_BATCHES)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"[device] {device}")
    t_start = time.time()

    print(f"\n[1/3] Loading tokenizer and WikiText-2 ({args.batches} batches × {SEQ_LEN} tokens)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    batches = load_eval_batches(tokenizer, args.batches)
    print(f"  {len(batches)} batches loaded")

    print(f"\n[2/3] Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type in ('cuda', 'mps') else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.0f}M parameters")
    print(f"  Layers: {len(model.layers)}, hidden: {model.config.hidden_size}, "
          f"max_inter: {model.config.intermediate_size}")

    print(f"\n[3/3] Evaluating schemes...")
    saved = save_mlp_weights(model)
    results = {}

    header = f"  {'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS)
    sep    = "─" * 76
    print(f"\n{sep}")
    print(header)
    print(sep)

    for name, bits in SCHEMES.items():
        restore_mlp_weights(model, saved, device)
        if bits is not None:
            apply_pyramid_mlp(model, bits)
        ab = avg_bits(bits) if bits else 32.0

        t0   = time.time()
        ppls = {f: compute_ppl(model, batches, f, device) for f in FLAGS}
        elapsed = time.time() - t0

        results[name] = {'avg_bits': ab, 'ppl': ppls}
        row = f"  {name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.1f}" for f in FLAGS)
        print(f"{row}  ({elapsed:.0f}s)", flush=True)

    restore_mlp_weights(model, saved, device)

    # Summary
    print(f"\n{'═'*76}")
    print("  Pyramid vs naive_matgptq_int4")
    ref = results.get('naive_matgptq_int4', {}).get('ppl', {})
    for pname in ['pyramid_8-5-4-4', 'pyramid_8-4-4-3']:
        p  = results.get(pname, {}).get('ppl', {})
        ab = results.get(pname, {}).get('avg_bits', 0)
        print(f"\n  {pname}  ({ab:.2f} avg bits):")
        for flag in FLAGS:
            if p.get(flag) and ref.get(flag):
                ratio   = p[flag] / ref[flag]
                verdict = 'PYRAMID WINS' if ratio < 1.0 else 'int4 wins'
                print(f"    {flag}:  pyramid={p[flag]:>7.2f}  int4={ref[flag]:>7.2f}  "
                      f"ratio={ratio:.3f}  {verdict}")
    print(f"{'═'*76}")

    meta = {
        'model':        MODEL_NAME,
        'fine_tuned':   False,
        'eval_batches': len(batches),
        'seq_len':      SEQ_LEN,
        'device':       str(device),
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_min': round((time.time() - t_start) / 60, 1),
        'note': (
            'naive_matgptq_int4 = uniform per-channel INT4 (MatGPTQ naive baseline). '
            'Real MatGPTQ uses GPTQ second-order optimization on top of this.'
        ),
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f"\n  Saved → {RESULTS_FILE}")
    print(f"  Total time: {meta['total_time_min']} min\n")


if __name__ == '__main__':
    main()
