# Bridge Network Maintenance Optimization

GPU-accelerated multi-agent reinforcement learning for optimizing maintenance decisions across a 96-component highway bridge network.

## Overview

This project implements **Decentralized Multi-Agent Actor-Critic (DDMAC)** with **Centralized Training, Decentralized Execution (CTDE)** to learn cost-optimal maintenance policies for a network of 85 pavement sections and 11 bridge decks. The system operates under partial observability (POMDP), budget constraints, and 6 network-level safety constraints handled via Lagrangian relaxation.

### Key Features

- **Off-policy learning** with replay buffer and importance sampling for sample efficiency
- **Constrained RL** via Lagrangian multipliers with 6 constraint types
- **Fully vectorized GPU environment** — 128 parallel rollouts with no Python loops
- **Memory-efficient** belief state representation (~50-100x reduction vs naive replication)
- **torch.compile + fused Adam** optimizations for faster training
- **Weights & Biases** integration for experiment tracking

## Network Architecture

| Component Type | Count | Condition States | Notes |
|----------------|-------|------------------|-------|
| Interstate pavement | 12 | IRI (5) + CCI (6) | 8-lane highways |
| Primary pavement | 47 | IRI (5) + CCI (6) | 4-lane roads |
| Secondary pavement | 26 | IRI (5) + CCI (6) | 2-lane roads |
| Bridge deck | 11 | Deck condition (7) | Separate transition matrices |

Each of the 96 components selects from **10 discrete actions** ranging from "do nothing" to full replacement, with associated action costs, observation costs, delay costs, and risk costs.

## Project Structure

```
bridge-network-rl/
├── gpu_optimized/                        # GPU training implementations
│   ├── torch_compiled/                   # torch.compile optimization
│   ├── torch_compiled_fused_adam/         # torch.compile + fused Adam (recommended)
│   └── torch_compiled_buffer_log_prob/   # torch.compile + compact log-prob buffer
├── dependency_files/                     # Static data (transition matrices, costs, initial states)
├── .env                                  # WANDB_API_KEY (create locally, not committed)
├── train_gpu.sh                          # Shell script launcher
└── requirements.txt
```

Each variant in `gpu_optimized/` is self-contained with 6 modules:

| Module | Purpose |
|--------|---------|
| `gpu_offpolicy_training.py` | Training loop, replay buffer, config, CLI entrypoint |
| `gpu_policy_network.py` | Actor (shared encoder + per-component heads) and centralized critic |
| `gpu_environment_optimized.py` | Vectorized GPU environment with Bayesian belief updates |
| `gpu_cost_module.py` | Action, observation, delay, and risk cost computation |
| `gpu_budget_constraints.py` | Budget allocation strategies (5 options) |
| `Network_data.py` | Network constants and data loading |

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 16+ GB VRAM (tested on A100, H100)
- NVIDIA drivers compatible with PyTorch 2.0+

### Installation

```bash
git clone git@github.com:ullahsaif13407/bridge-network-rl.git
cd bridge-network-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root for Weights & Biases:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

### Training

Run from inside a variant folder:

```bash
cd gpu_optimized/torch_compiled_fused_adam
python3 gpu_offpolicy_training.py
```

With custom options:

```bash
python3 gpu_offpolicy_training.py \
    --checkpoint-dir checkpoints_offpolicy \
    --wandb \
    --wandb-project MyProject \
    --wandb-name run_01 \
    --lr-schedule cosine \
    --lr-min-ratio 0.01
```

Resume from a checkpoint:

```bash
python3 gpu_offpolicy_training.py --resume checkpoints_offpolicy/checkpoint_ep100000.pt
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | None | Path to checkpoint file to resume from |
| `--checkpoint-dir` | `./checkpoints_offpolicy` | Directory for saving checkpoints |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb-project` | `Bridge_network_rl_comparision` | W&B project name |
| `--wandb-entity` | None | W&B entity/team |
| `--wandb-name` | None | W&B run name |
| `--wandb-resume` | off | Allow resuming a W&B run |
| `--wandb-log-every` | 50 | Log to W&B every N episodes |
| `--lr-schedule` | `cosine` | LR schedule: `none`, `cosine`, or `linear` |
| `--lr-min-ratio` | 0.01 | Minimum LR ratio vs initial LR |
| `--lr-step-every` | 1000 | Step LR scheduler every N update() calls |

## GPU Optimization Variants

| Variant | What's Different | Trade-off |
|---------|------------------|-----------|
| **`torch_compiled/`** | `torch.compile(mode="reduce-overhead")` | Minimal overhead, baseline speedup |
| **`torch_compiled_fused_adam/`** | + `optim.Adam(fused=True)` on CUDA | Lower VRAM during updates (**recommended**) |
| **`torch_compiled_buffer_log_prob/`** | + stores scalar `log_prob_sum` per transition | Reduced buffer memory for long runs |

## Algorithm

The training loop follows an off-policy actor-critic approach:

1. **Data collection**: 128 parallel environments run 20-step episodes with epsilon-greedy exploration
2. **Replay buffer**: Stores up to 100k transitions on GPU
3. **Policy update** (every 4 episodes): Samples 1024-transition minibatches, performs 4 gradient steps per update using importance-weighted TD(0) advantages
4. **Constraint handling**: 6 Lagrangian multipliers updated per episode: `lambda <- max(0, lambda + lr * (violation - cap))`
5. **LR scheduling**: Cosine annealing from episode 100k to 1M (min ratio 0.01)

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Parallel envs | 128 | Batch environments per episode |
| Episode length | 20 | Steps per episode |
| Discount (gamma) | 0.97 | Cost discount factor |
| Buffer capacity | 100,000 | Replay buffer size |
| Mini-batch | 1,024 | Transitions per gradient step |
| Gradient steps | 4 | Updates per training call |
| Actor LR | 1e-4 | Actor learning rate |
| Critic LR | 1e-3 | Critic learning rate |
| Lambda LR | 0.001 | Lagrangian multiplier learning rate |
| Total episodes | 2,000,000 | Full training run |
| Constraint caps | (5, 2, 18, 15, 35, 10) | 6 constraint violation caps |

## Data Files

Static data in `dependency_files/`:

| File | Contents |
|------|----------|
| `Smoothed_TP_MSI_*.mat` | Pavement transition probability matrices (by road type) |
| `Deck_TP_for_DRL.mat` | Bridge deck transition matrices |
| `link_lengths.mat` | Network link lengths |
| `Dlay_Cost.npy` | Delay cost arrays |
| `Total_action_duration.npy` | Maintenance action durations |
| `current_*_excellent.txt` | Initial condition state definitions |
| `Current_age.txt` | Initial component ages |

## References

- **DDMAC**: Decentralized Multi-Agent Actor-Critic
- **CTDE**: Centralized Training, Decentralized Execution
- **POMDP**: Partially Observable Markov Decision Process
- **Lagrangian Relaxation**: Constrained RL via dual optimization
