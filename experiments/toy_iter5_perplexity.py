"""
Iteration 5: Real perplexity on WikiText-2 with a MatFormer-trained toy model.

Approach:
  1. Train the toy model on a real text dataset (WikiText-2) for 200 steps with
     MatFormer's random flag selection — so the model genuinely learns elastic MLPs.
  2. Apply pyramid vs uniform quantization schemes.
  3. Compute perplexity at each subnetwork flag on a held-out eval set.

This replaces the MSE proxy with a real language modeling quality metric.
The model is small (hidden=256), so absolute perplexity will be high,
but the RELATIVE ordering between quantization schemes is what matters.
"""

import torch
import copy
import json
import sys
import os
import math
import random
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import ModifiedLlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
random.seed(42)
device = torch.device('cpu')
FLAGS = ['s', 'm', 'l', 'xl']
BOUNDARIES = [0, 1/8, 1/4, 1/2, 1.0]
REGION_FRACS = [1/8, 1/8, 1/4, 1/2]

config = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    vocab_size=128256,
)
SEQ_LEN = 64
BATCH = 4
TRAIN_STEPS = 1000


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


def compute_perplexity(model, token_batches, flag):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in token_batches:
            model.configure_subnetwork(flag)  # reset before every forward — resets to None after each
            ids = ids.to(device)
            out = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
            total_loss += out.loss.item() * ids[:, 1:].numel()
            total_tokens += ids[:, 1:].numel()
    return math.exp(total_loss / total_tokens)


def prepare_data(tokenizer, split, n_batches):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    text = ' '.join([t for t in ds['text'] if t.strip()])
    tokens = tokenizer.encode(text)
    batches = []
    step = SEQ_LEN * BATCH
    for i in range(0, min(len(tokens) - SEQ_LEN, n_batches * step), step):
        chunk = tokens[i:i + step]
        if len(chunk) < step:
            break
        ids = torch.tensor(chunk).view(BATCH, SEQ_LEN)
        batches.append(ids)
    return batches


def main():
    print("Loading tokenizer (NousResearch/Llama-3.2-1B)...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading WikiText-2...")
    train_batches = prepare_data(tokenizer, 'train', n_batches=300)
    eval_batches  = prepare_data(tokenizer, 'validation', n_batches=30)
    print(f"  Train batches: {len(train_batches)}, Eval batches: {len(eval_batches)}")

    print("Initialising MatFormer model...")
    base = ModifiedLlamaForCausalLM(config).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=2e-3)

    print(f"Training for {TRAIN_STEPS} steps with random flag selection...")
    base.train()
    step = 0
    while step < TRAIN_STEPS:
        for ids in train_batches:
            if step >= TRAIN_STEPS:
                break
            flag = random.choice(FLAGS)
            base.configure_subnetwork(flag)
            ids = ids.to(device)
            loss = base(input_ids=ids[:, :-1], labels=ids[:, 1:]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{TRAIN_STEPS}, loss={loss.item():.4f}")

    # Evaluate FP32 baseline perplexity
    print("\nComputing FP32 baseline perplexity...")
    base_ppl = {f: compute_perplexity(copy.deepcopy(base), eval_batches, f) for f in FLAGS}
    print("  FP32:", {f: f"{p:.2f}" for f, p in base_ppl.items()})

    schemes = {
        'fp32':            None,
        'uniform_int8':    [8, 8, 8, 8],
        'uniform_int4':    [4, 4, 4, 4],
        'pyramid_8-4-4-3': [8, 4, 4, 3],   # iso-budget 4.0 bits — iter4 winner
        'pyramid_8-5-4-4': [8, 5, 4, 4],   # INT4 floor, 4.625 bits
        'pyramid_8-7-6-4': [8, 7, 6, 4],   # 5.375 bits — high quality reference
    }

    results = {}
    print("\nEvaluating quantization schemes...")
    print(f"\n{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print("-" * 75)

    for name, bits in schemes.items():
        m = copy.deepcopy(base)
        if bits is not None:
            apply_pyramid(m, bits)
        ab = total_avg_bits(bits) if bits else 32.0
        ppls = {f: compute_perplexity(m, eval_batches, f) for f in FLAGS}
        results[name] = {'avg_bits': ab, 'ppl': ppls}
        row = f"{name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.2f}" for f in FLAGS)
        print(row)

    # Summary: perplexity ratio vs fp32
    print("\n── Perplexity increase vs FP32 (lower = less degradation) ──")
    fp32_ppl = results['fp32']['ppl']
    print(f"{'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print("-" * 75)
    for name, data in results.items():
        if name == 'fp32':
            continue
        ratios = {f: data['ppl'][f] / fp32_ppl[f] for f in FLAGS}
        row = f"{name:<22} {data['avg_bits']:>6.2f}  " + \
              "  ".join(f"{ratios[f]:>10.4f}" for f in FLAGS)
        print(row)

    # The key comparison
    print("\n── Key: pyramid_8-4-4-3 vs uniform_int4 perplexity ratio per flag ──")
    p = results['pyramid_8-4-4-3']['ppl']
    u = results['uniform_int4']['ppl']
    for flag in FLAGS:
        ratio = p[flag] / u[flag]
        print(f"  flag={flag}: pyramid/int4 = {ratio:.4f}  "
              f"({'pyramid wins' if ratio < 1 else 'int4 wins'})")

    return results


if __name__ == '__main__':
    results = main()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'toy_iter5.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
