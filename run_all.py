#!/usr/bin/env python3
"""
MatFormer Pyramid Quantization — Full Experiment
Self-contained script for a fresh GPU instance (RunPod / Vast.ai).

Phases:
  1. Load LLaMA-3.2-1B, replace MLPs with elastic MatFormer variant
  2. Fine-tune MLP weights with random flag selection on WikiText-2
     — checkpoints every SAVE_EVERY steps, resumes automatically
  3. Apply quantization schemes (pyramid vs uniform INT4/INT8)
  4. Evaluate perplexity at each subnetwork flag (s/m/l/xl)
  5. Save full results to results_full.json

Usage (inside tmux so laptop can close):
  python run_all.py 2>&1 | tee run.log
"""

import os, sys, json, math, random, time, copy, shutil
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.models.llama.modeling_llama import LlamaMLP
from datasets import load_dataset

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "NousResearch/Llama-3.2-1B"
CHECKPOINT_DIR = "checkpoints"
RESULTS_FILE   = "results_full.json"

TRAIN_STEPS    = 3000   # enough for elastic adaptation on 1B model
SAVE_EVERY     = 500    # checkpoint frequency
LR             = 2e-5   # conservative — pretrained weights
BATCH_SIZE     = 4
SEQ_LEN        = 256
WARMUP_STEPS   = 200
EVAL_BATCHES   = 80     # ~80k tokens — stable perplexity estimate

FLAGS        = ['s', 'm', 'l', 'xl']
FLAG_FRACS   = {'s': 1/8, 'm': 1/4, 'l': 1/2, 'xl': 1.0}
BOUNDARIES   = [0, 1/8, 1/4, 1/2, 1.0]
REGION_FRACS = [1/8, 1/8, 1/4, 1/2]

SCHEMES = {
    'fp32':            None,
    'uniform_int8':    [8, 8, 8, 8],
    'uniform_int4':    [4, 4, 4, 4],
    'pyramid_8-7-6-4': [8, 7, 6, 4],   # 5.375 bits — high quality reference
    'pyramid_8-5-4-4': [8, 5, 4, 4],   # 4.625 bits — INT4 floor, our main claim
    'pyramid_8-4-4-3': [8, 4, 4, 3],   # 4.000 bits — exact iso-budget vs int4
}

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {device}" + (f" — {torch.cuda.get_device_name()}" if device.type == 'cuda' else ""))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Elastic MLP (MatFormer-style weight slicing)
# ─────────────────────────────────────────────────────────────────────────────

class ElasticMLP(LlamaMLP):
    """
    Drop-in replacement for LlamaMLP that supports width slicing.
    At inference: call configure(flag) before each forward pass.
    During training: call configure(flag) before each forward pass.
    """
    SCALE = {'s': 1/8, 'm': 1/4, 'l': 1/2, 'xl': 1.0}

    def __init__(self, config):
        super().__init__(config)
        self._k = None  # active intermediate size; set by configure()

    def configure(self, flag: str):
        self._k = int(self.intermediate_size * self.SCALE[flag])

    def forward(self, x):
        if self._k is None:
            raise RuntimeError("Call configure(flag) before forward().")
        k = self._k
        self._k = None  # reset so stale config is caught immediately
        gate = F.linear(x, self.gate_proj.weight[:k])
        up   = F.linear(x, self.up_proj.weight[:k])
        down = F.linear(self.act_fn(gate) * up, self.down_proj.weight[:, :k])
        return down


def inject_elastic_mlp(model):
    """Replace every LlamaMLP in the model with ElasticMLP, copying weights."""
    config = model.config
    for layer in model.model.layers:
        orig = layer.mlp
        new  = ElasticMLP(config).to(orig.gate_proj.weight.device,
                                     dtype=orig.gate_proj.weight.dtype)
        new.gate_proj.weight.data.copy_(orig.gate_proj.weight.data)
        new.up_proj.weight.data.copy_(orig.up_proj.weight.data)
        new.down_proj.weight.data.copy_(orig.down_proj.weight.data)
        layer.mlp = new


def configure_all(model, flag):
    for layer in model.model.layers:
        layer.mlp.configure(flag)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantization
# ─────────────────────────────────────────────────────────────────────────────

def fake_quant_per_channel(tensor, bits, channel_dim=0):
    """Per-channel symmetric fake quantization. Required for real LLM weights."""
    if tensor.numel() == 0:
        return tensor
    orig = tensor.dtype
    t    = tensor.float()
    n    = 2 ** (bits - 1) - 1
    dims = [d for d in range(t.ndim) if d != channel_dim]
    scale = t.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8) / n
    q = torch.clamp(torch.round(t / scale), -n, n) * scale
    return q.to(orig)


def apply_pyramid(model, bits_per_region):
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp
            H   = mlp.intermediate_size
            cuts = [int(H * b) for b in BOUNDARIES]
            for i, bits in enumerate(bits_per_region):
                lo, hi = cuts[i], cuts[i + 1]
                if hi <= lo:
                    continue
                mlp.gate_proj.weight.data[lo:hi] = fake_quant_per_channel(
                    mlp.gate_proj.weight.data[lo:hi], bits, 0)
                mlp.up_proj.weight.data[lo:hi] = fake_quant_per_channel(
                    mlp.up_proj.weight.data[lo:hi], bits, 0)
                mlp.down_proj.weight.data[:, lo:hi] = fake_quant_per_channel(
                    mlp.down_proj.weight.data[:, lo:hi], bits, 1)


def avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


def save_mlp_weights(model):
    saved = {}
    for i, layer in enumerate(model.model.layers):
        m = layer.mlp
        saved[i] = {
            'gate': m.gate_proj.weight.data.cpu().clone(),
            'up':   m.up_proj.weight.data.cpu().clone(),
            'down': m.down_proj.weight.data.cpu().clone(),
        }
    return saved


def restore_mlp_weights(model, saved):
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            m = layer.mlp
            m.gate_proj.weight.data.copy_(saved[i]['gate'].to(device))
            m.up_proj.weight.data.copy_(saved[i]['up'].to(device))
            m.down_proj.weight.data.copy_(saved[i]['down'].to(device))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data
# ─────────────────────────────────────────────────────────────────────────────

def load_wikitext(tokenizer, split, n_batches):
    ds   = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    text = ' '.join(t for t in ds['text'] if t.strip())
    ids  = tokenizer.encode(text)
    batches = []
    step = SEQ_LEN * BATCH_SIZE
    for i in range(0, n_batches * step, step):
        chunk = ids[i: i + step + 1]
        if len(chunk) < step + 1:
            break
        batches.append(
            torch.tensor(chunk).view(BATCH_SIZE, SEQ_LEN + 1)
        )
        if len(batches) >= n_batches:
            break
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint():
    if not os.path.isdir(CHECKPOINT_DIR):
        return None, 0
    steps = []
    for name in os.listdir(CHECKPOINT_DIR):
        if name.startswith('step_'):
            try:
                steps.append(int(name.split('_')[1]))
            except ValueError:
                pass
    if not steps:
        return None, 0
    latest = max(steps)
    return os.path.join(CHECKPOINT_DIR, f'step_{latest}'), latest


def save_checkpoint(model, optimizer, scheduler, step):
    path = os.path.join(CHECKPOINT_DIR, f'step_{step}')
    os.makedirs(path, exist_ok=True)
    # Save MLP weights only (what we trained)
    mlp_state = {}
    for i, layer in enumerate(model.model.layers):
        m = layer.mlp
        mlp_state[i] = {
            'gate': m.gate_proj.weight.data.cpu(),
            'up':   m.up_proj.weight.data.cpu(),
            'down': m.down_proj.weight.data.cpu(),
        }
    torch.save(mlp_state,            os.path.join(path, 'mlp_weights.pt'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))
    torch.save({'step': step},          os.path.join(path, 'meta.pt'))
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    mlp_state = torch.load(os.path.join(path, 'mlp_weights.pt'), map_location='cpu')
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            m = layer.mlp
            m.gate_proj.weight.data.copy_(mlp_state[i]['gate'].to(device))
            m.up_proj.weight.data.copy_(mlp_state[i]['up'].to(device))
            m.down_proj.weight.data.copy_(mlp_state[i]['down'].to(device))
    optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=device))
    scheduler.load_state_dict(torch.load(os.path.join(path, 'scheduler.pt')))
    step = torch.load(os.path.join(path, 'meta.pt'))['step']
    print(f"  [ckpt] resumed from step {step}")
    return step


def train(model, train_batches):
    # Freeze everything except MLP weights
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ['gate_proj', 'up_proj', 'down_proj'])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_trainable/1e6:.1f}M (MLP weights only)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TRAIN_STEPS
    )

    # Resume from checkpoint if available
    ckpt_path, start_step = find_latest_checkpoint()
    if ckpt_path:
        print(f"  Found checkpoint at step {start_step} — resuming")
        start_step = load_checkpoint(model, optimizer, scheduler, ckpt_path)
    else:
        print(f"  No checkpoint found — training from scratch")
        start_step = 0

    if start_step >= TRAIN_STEPS:
        print(f"  Training already complete ({start_step} steps)")
        return

    model.train()
    step       = start_step
    loss_sum   = 0.0
    t0         = time.time()

    # Cycle through batches
    batch_iter = (b for _ in range(9999) for b in train_batches)

    print(f"\n{'─'*60}")
    print(f"  Training: steps {start_step} → {TRAIN_STEPS}")
    print(f"{'─'*60}")

    for batch in batch_iter:
        if step >= TRAIN_STEPS:
            break

        flag = random.choice(FLAGS)
        configure_all(model, flag)

        ids = batch.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(input_ids=ids[:, :-1], labels=ids[:, 1:]).loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()
        step += 1

        if step % 50 == 0:
            avg  = loss_sum / 50
            lr_  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{TRAIN_STEPS}  loss={avg:.4f}  lr={lr_:.2e}  "
                  f"elapsed={elapsed/60:.1f}m  flag={flag}", flush=True)
            loss_sum = 0.0

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, step)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, step)
    print(f"\n  Training complete — {step} steps in {(time.time()-t0)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppl(model, batches, flag):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            configure_all(model, flag)
            ids = batch.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                return float('nan')
            total_loss   += out.loss.item() * ids[:, 1:].numel()
            total_tokens += ids[:, 1:].numel()
    return math.exp(min(total_loss / total_tokens, 20))


def evaluate_all_schemes(model, eval_batches):
    original_weights = save_mlp_weights(model)
    results = {}

    print(f"\n{'─'*70}")
    print(f"  {'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS))
    print(f"{'─'*70}")

    for name, bits in SCHEMES.items():
        restore_mlp_weights(model, original_weights)
        if bits is not None:
            apply_pyramid(model, bits)
        ab = avg_bits(bits) if bits else 32.0

        t0   = time.time()
        ppls = {f: compute_ppl(model, eval_batches, f) for f in FLAGS}
        elapsed = time.time() - t0

        results[name] = {'avg_bits': ab, 'ppl': ppls}
        row = f"  {name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.2f}" for f in FLAGS)
        print(f"{row}  ({elapsed:.0f}s)", flush=True)

    restore_mlp_weights(model, original_weights)
    return results


def print_summary(results):
    print(f"\n{'═'*70}")
    print("  KEY RESULT: pyramid_8-4-4-3 vs uniform_int4 (same bit budget)")
    print(f"{'─'*70}")
    p = results.get('pyramid_8-4-4-3', {}).get('ppl', {})
    u = results.get('uniform_int4',    {}).get('ppl', {})
    for flag in FLAGS:
        if p.get(flag) and u.get(flag):
            ratio   = p[flag] / u[flag]
            verdict = 'PYRAMID WINS ✓' if ratio < 1 else 'int4 wins'
            print(f"  flag={flag}:  pyramid={p[flag]:.1f}  int4={u[flag]:.1f}  "
                  f"ratio={ratio:.4f}  {verdict}")

    print(f"\n  KEY RESULT: pyramid_8-5-4-4 vs uniform_int4 (INT4-floor pyramid)")
    print(f"{'─'*70}")
    p2 = results.get('pyramid_8-5-4-4', {}).get('ppl', {})
    for flag in FLAGS:
        if p2.get(flag) and u.get(flag):
            ratio   = p2[flag] / u[flag]
            verdict = 'PYRAMID WINS ✓' if ratio < 1 else 'int4 wins'
            print(f"  flag={flag}:  pyramid={p2[flag]:.1f}  int4={u[flag]:.1f}  "
                  f"ratio={ratio:.4f}  {verdict}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'═'*70}")
    print("  MatFormer Pyramid Quantization — Full Experiment")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}\n")

    # ── Load tokenizer and data ───────────────────────────────────────────────
    print("[1/4] Loading tokenizer and WikiText-2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_batches = load_wikitext(tokenizer, 'train',      n_batches=500)
    eval_batches  = load_wikitext(tokenizer, 'validation', n_batches=EVAL_BATCHES)
    print(f"  Train: {len(train_batches)} batches  "
          f"Eval: {len(eval_batches)} batches × {SEQ_LEN} tokens")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    print("  Injecting ElasticMLP layers...")
    inject_elastic_mlp(model)
    print("  Done.")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Fine-tuning ({TRAIN_STEPS} steps, checkpoint every {SAVE_EVERY})...")
    train(model, train_batches)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating quantization schemes on WikiText-2 validation...")
    model.eval()
    results = evaluate_all_schemes(model, eval_batches)

    # ── Save and summarise ────────────────────────────────────────────────────
    print_summary(results)

    meta = {
        'model':       MODEL_NAME,
        'train_steps': TRAIN_STEPS,
        'batch_size':  BATCH_SIZE,
        'seq_len':     SEQ_LEN,
        'eval_batches': len(eval_batches),
        'device':      str(device),
        'gpu':         torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
        'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_min': round((time.time() - t_start) / 60, 1),
    }

    output = {'meta': meta, 'results': results}
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {RESULTS_FILE}")
    print(f"  Total time: {meta['total_time_min']} min")
    print(f"\n  Paste {RESULTS_FILE} back to Claude for analysis.\n")


if __name__ == '__main__':
    main()
