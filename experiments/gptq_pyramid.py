#!/usr/bin/env python3
"""
GPTQ-style PTQ with pyramid bit allocation for MatFormer.
A
Unlike fake quantization (which just rounds weights), this uses calibration
data to compute the input Hessian H = X^T X and propagates quantization
error to unquantized columns — the core of the GPTQ algorithm.

Comparison:
  naive_int4_gptq  — uniform INT4 GPTQ across all MLP rows (MatGPTQ baseline)
  pyramid_gptq     — same algorithm, more bits for inner rows (our method)

Model: atrost/climbmix-matformer-353m-1p2b-h100 (properly trained MatFormer)

MLP weight shapes:
  gate_proj: [3584, 1280]  — pyramid applies to rows (intermediate neurons)
  up_proj:   [3584, 1280]  — same
  down_proj: [1280, 3584]  — pyramid applies to columns (intermediate neurons)

MatFormer pyramid boundaries (matching actual matformer_specs):
  rows 0:512    used by s,m,l,xl  → highest bits
  rows 512:1024 used by m,l,xl    → second
  rows 1024:2048 used by l,xl     → third
  rows 2048:3584 used by xl only  → lowest bits

Run:
  python experiments/gptq_pyramid.py
  python experiments/gptq_pyramid.py --device mps --calib 32
"""

import os, sys, json, math, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME   = "atrost/climbmix-matformer-353m-1p2b-h100"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'gptq_pyramid.json')

FLAGS      = ['s', 'm', 'l', 'xl']
BOUNDARIES = [0, 512, 1024, 2048, 3584]      # absolute intermediate-dim boundaries
REGION_FRACS = [512/3584, 512/3584, 1024/3584, 1536/3584]

EVAL_BATCHES  = 40
CALIB_BATCHES = 64    # calibration samples for Hessian estimation
SEQ_LEN       = 512
BATCH_SIZE    = 1     # keep small to avoid OOM during calibration

SCHEMES = {
    'fp32':              None,
    'naive_int4_gptq':   [4, 4, 4, 4],   # uniform INT4 — MatGPTQ naive baseline
    'pyramid_8-5-4-4':   [8, 5, 4, 4],   # our method (INT4 floor)
    'pyramid_8-4-4-3':   [8, 4, 4, 3],   # our method (iso-budget vs INT4)
}


def avg_bits(bits_per_region):
    return sum(f * b for f, b in zip(REGION_FRACS, bits_per_region))


# ── Per-channel fake quantization (used inside GPTQ blocks) ──────────────────

def quant_block(x, bits):
    """Per-row symmetric fake quantization on a block [out, in]."""
    n     = 2 ** (bits - 1) - 1
    scale = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / n
    return (torch.clamp(torch.round(x.float() / scale), -n, n) * scale).to(x.dtype)


# ── GPTQ core ─────────────────────────────────────────────────────────────────

def gptq_rows(W, H, bits, block_size=128):
    """
    GPTQ quantization of all rows of W with the given bit-width.

    W: [out, in]  weight matrix (rows = output neurons)
    H: [in, in]   Hessian X^T X over input dimensions
    bits: int     target bit-width for this slice

    Processes columns of W left-to-right, propagating quantization error
    to unquantized columns via H^{-1} — standard GPTQ update.
    Returns W_q of same shape and dtype as W.
    """
    device = W.device
    out_d, in_d = W.shape
    W_q = W.float().clone()

    # Stabilise and invert Hessian (shared across all rows)
    damp = 0.01 * H.diag().float().mean()
    Hf   = H.float() + damp * torch.eye(in_d, device=device)
    try:
        L     = torch.linalg.cholesky(Hf)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        H_inv = torch.linalg.pinv(Hf)

    H_inv_diag = H_inv.diag().clamp(min=1e-8)   # [in_d]

    for i in range(0, in_d, block_size):
        j    = min(i + block_size, in_d)
        w    = W_q[:, i:j].clone()               # [out, block]
        w_q  = quant_block(w, bits)              # quantise this block
        err  = w - w_q                           # [out, block]

        if j < in_d:
            # Propagate: W[:, j:] -= err @ (H_inv[i:j, j:] / diag[i:j])
            scale_inv = H_inv[i:j, j:] / H_inv_diag[i:j].unsqueeze(1)
            W_q[:, j:] -= err @ scale_inv.float()

        W_q[:, i:j] = w_q

    return W_q.to(W.dtype)


def gptq_cols(W, H, bits_per_region, block_size=128):
    """
    GPTQ quantization of columns of W with per-region bit allocation.

    W: [out, in]       weight matrix (cols = input neurons)
    H: [in, in]        Hessian over input (intermediate) dimensions
    bits_per_region:   list of 4 ints matching BOUNDARIES

    Used for down_proj where the pyramid maps to column regions.
    """
    device = W.device
    out_d, in_d = W.shape
    W_q = W.float().clone()

    damp = 0.01 * H.diag().float().mean()
    Hf   = H.float() + damp * torch.eye(in_d, device=device)
    try:
        L     = torch.linalg.cholesky(Hf)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        H_inv = torch.linalg.pinv(Hf)

    H_inv_diag = H_inv.diag().clamp(min=1e-8)

    def bits_for_col(c):
        for k in range(len(BOUNDARIES) - 1):
            if BOUNDARIES[k] <= c < BOUNDARIES[k + 1]:
                return bits_per_region[k]
        return bits_per_region[-1]

    for i in range(0, in_d, block_size):
        j    = min(i + block_size, in_d)
        bits = bits_for_col(i)
        w    = W_q[:, i:j].clone()
        w_q  = quant_block(w.T, bits).T          # quantise per output-col (= per-row of W^T)
        err  = w - w_q

        if j < in_d:
            scale_inv = H_inv[i:j, j:] / H_inv_diag[i:j].unsqueeze(1)
            W_q[:, j:] -= err @ scale_inv.float()

        W_q[:, i:j] = w_q

    return W_q.to(W.dtype)


# ── Calibration ───────────────────────────────────────────────────────────────

def collect_hessians(model, batches, device):
    """
    Forward all calibration batches through the model (at xl flag),
    accumulating H_in = X^T X for gate_proj/up_proj inputs
    and H_mid = act^T act for down_proj inputs per layer.

    Returns:
        hessians: list of {'H_in': tensor, 'H_mid': tensor} per layer
    """
    n_layers  = len(model.layers)
    hidden    = model.config.hidden_size
    max_inter = model.config.intermediate_size

    H_in  = [torch.zeros(hidden,    hidden,    device='cpu') for _ in range(n_layers)]
    H_mid = [torch.zeros(max_inter, max_inter, device='cpu') for _ in range(n_layers)]
    n_samples = [0] * n_layers

    hooks = []
    captured = [{'x': None, 'act': None} for _ in range(n_layers)]

    for li, layer in enumerate(model.layers):
        def make_pre_hook(idx):
            def pre_hook(module, args):
                x = args[0].detach().float()          # [B, T, hidden]
                captured[idx]['x'] = x.reshape(-1, hidden).cpu()
            return pre_hook

        def make_mid_hook(idx, gate, up):
            def hook(module, args, output):
                # down_proj input = silu(gate) * up
                # We capture it as the input to down_proj
                pass
            return hook

        h1 = layer.gate_proj.register_forward_pre_hook(make_pre_hook(li))
        hooks.append(h1)

        # Capture down_proj input via pre_hook
        def make_down_pre_hook(idx):
            def pre_hook(module, args):
                act = args[0].detach().float()         # [B, T, intermediate]
                captured[idx]['act'] = act.reshape(-1, max_inter).cpu()
            return pre_hook

        h2 = layer.down_proj.register_forward_pre_hook(make_down_pre_hook(li))
        hooks.append(h2)

    model.eval()
    with torch.no_grad():
        for batch in batches:
            ids = batch.to(device)
            model(input_ids=ids, matformer_size='xl')

            for li in range(n_layers):
                if captured[li]['x'] is not None:
                    x   = captured[li]['x'].float()    # [N, hidden]
                    act = captured[li]['act'].float()   # [N, intermediate]
                    H_in[li]  += x.T @ x
                    H_mid[li] += act.T @ act
                    n_samples[li] += x.shape[0]
                    captured[li] = {'x': None, 'act': None}

    for h in hooks:
        h.remove()

    # Normalise by sample count
    for li in range(n_layers):
        if n_samples[li] > 0:
            H_in[li]  /= n_samples[li]
            H_mid[li] /= n_samples[li]

    return [{'H_in': H_in[li], 'H_mid': H_mid[li]} for li in range(n_layers)]


# ── Apply pyramid GPTQ ────────────────────────────────────────────────────────

def apply_gptq_scheme(model, hessians, bits_per_region, device):
    """
    Apply GPTQ with the given per-region bit allocation to all MLP layers.
    If bits_per_region is [4,4,4,4] this is uniform INT4 (naive baseline).
    """
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            H_in  = hessians[li]['H_in'].to(device)
            H_mid = hessians[li]['H_mid'].to(device)

            # gate_proj / up_proj: pyramid by rows (output = intermediate neurons)
            for proj in [layer.gate_proj, layer.up_proj]:
                W = proj.weight.data                       # [3584, 1280]
                W_new = torch.empty_like(W)
                for k, bits in enumerate(bits_per_region):
                    lo, hi = BOUNDARIES[k], BOUNDARIES[k + 1]
                    if hi <= lo:
                        continue
                    W_new[lo:hi] = gptq_rows(W[lo:hi], H_in, bits)
                proj.weight.data.copy_(W_new)

            # down_proj: pyramid by columns (input = intermediate neurons)
            layer.down_proj.weight.data.copy_(
                gptq_cols(layer.down_proj.weight.data, H_mid, bits_per_region)
            )

    return model


def save_mlp_weights(model):
    return {li: {
        'gate': layer.gate_proj.weight.data.cpu().clone(),
        'up':   layer.up_proj.weight.data.cpu().clone(),
        'down': layer.down_proj.weight.data.cpu().clone(),
    } for li, layer in enumerate(model.layers)}


def restore_mlp_weights(model, saved, device):
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            layer.gate_proj.weight.data.copy_(saved[li]['gate'].to(device))
            layer.up_proj.weight.data.copy_(saved[li]['up'].to(device))
            layer.down_proj.weight.data.copy_(saved[li]['down'].to(device))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_batches(tokenizer, split, n_batches):
    ds   = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
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
            ids   = batch.to(device)
            dtype = torch.bfloat16 if device.type in ('cuda', 'mps') else torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type in ('cuda', 'mps')):
                out = model(input_ids=ids, labels=ids, matformer_size=flag)
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                return float('nan')
            n = (ids.shape[1] - 1) * ids.shape[0]
            total_loss   += out.loss.item() * n
            total_tokens += n
    return math.exp(min(total_loss / total_tokens, 20))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',  default=None)
    parser.add_argument('--calib',   type=int, default=CALIB_BATCHES)
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

    print(f"\n[1/4] Loading tokenizer + data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    calib_batches = load_batches(tokenizer, 'train',      args.calib)
    eval_batches  = load_batches(tokenizer, 'validation', args.batches)
    print(f"  Calib: {len(calib_batches)} batches  Eval: {len(eval_batches)} batches")

    print(f"\n[2/4] Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,    # float32 for stable Hessian computation
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    print(f"\n[3/4] Collecting calibration Hessians ({len(calib_batches)} batches at flag=xl)...")
    t0 = time.time()
    hessians = collect_hessians(model, calib_batches, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n[4/4] Evaluating schemes...")
    saved   = save_mlp_weights(model)
    results = {}

    header = f"  {'Scheme':<22} {'avg_b':>6}  " + "  ".join(f"{f:>10}" for f in FLAGS)
    sep    = "─" * 76
    print(f"\n{sep}\n{header}\n{sep}")

    for name, bits in SCHEMES.items():
        restore_mlp_weights(model, saved, device)
        if bits is not None:
            apply_gptq_scheme(model, hessians, bits, device)
        ab = avg_bits(bits) if bits else 32.0

        t0   = time.time()
        ppls = {f: compute_ppl(model, eval_batches, f, device) for f in FLAGS}
        elapsed = time.time() - t0

        results[name] = {'avg_bits': ab, 'ppl': ppls}
        row = f"  {name:<22} {ab:>6.2f}  " + "  ".join(f"{ppls[f]:>10.2f}" for f in FLAGS)
        print(f"{row}  ({elapsed:.0f}s)", flush=True)

    restore_mlp_weights(model, saved, device)

    # Summary
    print(f"\n{'═'*76}")
    print("  Pyramid GPTQ vs naive_int4_gptq")
    ref = results.get('naive_int4_gptq', {}).get('ppl', {})
    for pname in ['pyramid_8-5-4-4', 'pyramid_8-4-4-3']:
        p  = results.get(pname, {}).get('ppl', {})
        ab = results.get(pname, {}).get('avg_bits', 0)
        print(f"\n  {pname}  ({ab:.2f} avg bits):")
        for flag in FLAGS:
            if p.get(flag) and ref.get(flag):
                ratio   = p[flag] / ref[flag]
                verdict = 'PYRAMID WINS' if ratio < 1.0 else 'int4 wins'
                print(f"    {flag}:  pyramid={p[flag]:>8.2f}  int4={ref[flag]:>8.2f}  "
                      f"ratio={ratio:.4f}  {verdict}")
    print(f"{'═'*76}")

    meta = {
        'model':          MODEL_NAME,
        'method':         'gptq_style_ptq',
        'calib_batches':  len(calib_batches),
        'eval_batches':   len(eval_batches),
        'calib_flag':     'xl',
        'device':         str(device),
        'timestamp':      time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_min': round((time.time() - t_start) / 60, 1),
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f"\n  Saved → {RESULTS_FILE}  ({meta['total_time_min']} min total)")


if __name__ == '__main__':
    main()
