# Bridge Network RL — Multi-Algorithm Comparison

GPU-accelerated multi-agent reinforcement learning for optimizing maintenance decisions across a 96-component highway bridge network. Compares **Off-Policy Actor-Critic**, **PPO**, and **GRPO** under identical environments and constraints.

## Problem

A highway network of **85 pavement sections + 11 bridge decks** must be maintained over a planning horizon. Each component has 10 possible maintenance actions (do nothing → full replacement). The agent operates under:

- **Partial observability** (POMDP with Bayesian belief states)
- **Budget constraints** (1.3M units, resets every 5 steps)
- **6 network-level safety constraints** (handled via Lagrangian relaxation)

## Algorithms

| Algorithm | Advantage Estimation | Critic | Grad Steps/Episode | Key Property |
|-----------|---------------------|--------|-------------------|--------------|
| **Off-Policy** | TD(0) + importance sampling | Yes (~875K params) | ~1 (updates every 4 eps) | Sample-efficient via replay buffer |
| **PPO** | GAE (λ=0.95, per-step) | Yes (~875K params) | 16 (4 epochs × 4 minibatches) | Best per-step credit assignment |
| **GRPO** | Group-relative (episode-level) | No (critic-free) | 16 (4 epochs × 4 minibatches) | Simplest — no value function needed |

All share the same actor architecture (~70K params): shared encoders per component type → shared policy head → 10-action logits per component.

## Project Structure

```
Bridge_network/
├── Trainers/
│   ├── torch_compiled_fused_adam/         # Off-policy (recommended variant)
│   │   └── gpu_offpolicy_training.py
│   ├── ppo_torch_compiled_fused_adam/     # PPO
│   │   └── gpu_ppo_training.py
│   ├── grpo_torch_compiled_fused_adam/    # GRPO
│   │   └── gpu_grpo_training.py
│   ├── torch_compiled/                   # Off-policy (base torch.compile)
│   ├── torch_compiled_buffer_log_prob/   # Off-policy (compact buffer)
│   └── test_cost_parity.py              # Verify cost parity across variants
├── dependency_files/                     # Transition matrices, costs, initial states
├── DDMAC_CTDE_VDOT/                     # Original CPU baseline (Keras/TF)
├── run_all.sh                           # Launch all 3 algorithms in tmux
└── requirements.txt
```

Each trainer directory is self-contained with 6 modules:

| Module | Purpose |
|--------|---------|
| `gpu_*_training.py` | Training loop, buffer, config, CLI |
| `gpu_policy_network.py` | Actor (shared encoder) + Critic (MLP) |
| `gpu_environment_optimized.py` | Vectorized GPU environment (128 parallel envs) |
| `gpu_cost_module.py` | Action, observation, delay, risk costs + 6 constraints |
| `gpu_budget_constraints.py` | Budget allocation (greedy by cost) |
| `Network_data.py` | Network geometry, transition matrix loading |

## Quick Start

### Setup

```bash
cd Bridge_network
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "WANDB_API_KEY=your_key_here" > .env
```

### Run All 3 Algorithms (tmux)

```bash
./run_all.sh              # defaults to cuda:0
./run_all.sh cuda:1       # specify device
```

Opens a tmux session `rl_train` with 3 panes (off-policy | PPO | GRPO). All log to the same W&B project for direct comparison.

- `Ctrl+B` then `D` — detach (training continues)
- `tmux attach -t rl_train` — reattach

### Run Individually

```bash
# Off-policy
python3 Trainers/torch_compiled_fused_adam/gpu_offpolicy_training.py \
    --wandb --wandb-name offpolicy --device cuda:0

# PPO
python3 Trainers/ppo_torch_compiled_fused_adam/gpu_ppo_training.py \
    --wandb --wandb-name ppo --device cuda:0

# GRPO
python3 Trainers/grpo_torch_compiled_fused_adam/gpu_grpo_training.py \
    --wandb --wandb-name grpo --device cuda:0
```

### Resume from Checkpoint

```bash
python3 Trainers/torch_compiled_fused_adam/gpu_offpolicy_training.py \
    --resume Trainers/torch_compiled_fused_adam/checkpoints_offpolicy/checkpoint_latest.pt
```

## W&B Metrics

All algorithms log these shared metrics:

| Metric | Description |
|--------|-------------|
| `Total_cost` | Cumulative discounted cost per episode |
| `Lambda` | Lagrangian multiplier values (6 constraints) |
| `Episodic_Cost_ac_ob` | Action + observation costs |
| `Episodic Risk Cost` | Risk cost component |
| `Delay_Cost` | Traffic delay cost |
| `actor_lr` / `critic_lr` | Current learning rates |

Algorithm-specific metrics:

| Metric | PPO | GRPO |
|--------|-----|------|
| `*/pg_loss` | Clipped surrogate loss | Clipped surrogate loss |
| `*/v_loss` | Clipped value loss | — (no critic) |
| `*/entropy_loss` | Policy entropy | Policy entropy |
| `*/clip_frac` | Fraction of clipped ratios | Fraction of clipped ratios |
| `*/approx_kl` | KL divergence (old → new) | KL divergence (old → new) |
| `*/kl_loss` | — | KL to reference policy (optional) |

## Hyperparameters

### Shared

| Parameter | Value |
|-----------|-------|
| Parallel envs | 128 |
| Episode length | 20 steps |
| Discount (γ) | 0.97 |
| Actor LR | 1e-4 |
| Constraint caps | (5, 2, 18, 15, 35, 10) |
| Lambda LR | 0.001 |
| Total episodes | 2,000,000 |
| LR schedule | Cosine (100k–1M episodes) |

### Algorithm-Specific

| Parameter | Off-Policy | PPO | GRPO |
|-----------|-----------|-----|------|
| Critic LR | 1e-3 | 1e-3 | — |
| Replay buffer | 100k | — | — |
| GAE λ | — | 0.95 | — |
| Clip ε | — | 0.2 | 0.2 |
| Epochs/update | — | 4 | 4 |
| Minibatches | — | 4 | 4 |
| Entropy coef | 0.0 | 0.01 | 0.01 |
| Value coef | 1.0 | 0.5 | — |
| Update frequency | Every 4 eps | Every ep | Every ep |

## Hardware

- Tested on NVIDIA A100 / H100 (16+ GB VRAM)
- All 3 algorithms can run in parallel on a single GPU
- PyTorch 2.0+ with `torch.compile` and fused Adam

## References

- **DDMAC-CTDE**: Decentralized Multi-Agent Actor-Critic with Centralized Training, Decentralized Execution
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GRPO**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
- **Lagrangian Relaxation**: Constrained RL via dual optimization of safety constraints
