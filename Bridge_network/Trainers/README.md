# GPU Optimized Variants

This folder contains **clean, training-ready copies** of the off-policy DDMAC-CTDE implementation, based on the tested profiling variants, but **without** any profiling harness or artifact-specific wiring.

Each subfolder is self-contained and can be run directly.

## Folder Layout

- `gpu_optimized/torch_compiled/`
- `gpu_optimized/torch_compiled_fused_adam/`
- `gpu_optimized/torch_compiled_buffer_log_prob/`

## torch_compiled

**What changed (vs original):**

- Wraps `self.policy` with `torch.compile(self.policy, mode="reduce-overhead")` when available (best-effort; silently falls back if unsupported).

## torch_compiled_fused_adam

**What changed (vs original):**

- Same `torch.compile(..., mode="reduce-overhead")` as `torch_compiled/`.
- Uses `optim.Adam(..., fused=True)` when running on CUDA and the PyTorch build supports fused Adam; otherwise falls back to regular Adam.

## torch_compiled_buffer_log_prob

**What changed (vs original):**

- Same `torch.compile(..., mode="reduce-overhead")` as `torch_compiled/`.
- Replay buffer stores `behavior_logp_sum` (shape `(capacity,)`) instead of `behavior_probs` (shape `(capacity, TOT_COMP, NACOMP)`).
  - `behavior_logp_sum = sum_k log(mu(a_k|s))` for the executed actions (sum over components).
  - `update()` uses `sum_k log(pi(a_k|s)) - behavior_logp_sum` for the importance ratio.

This significantly reduces replay-buffer GPU memory and sampling bandwidth.

## How To Run

Run from inside a variant folder (recommended so relative checkpoint paths behave as expected):

```bash
cd gpu_optimized/torch_compiled
python3 gpu_offpolicy_training.py
```

Optional overrides (same CLI as the original script):

```bash
python3 gpu_offpolicy_training.py --checkpoint-dir checkpoints_offpolicy --wandb
```

