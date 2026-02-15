# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated **Decentralized Multi-Agent Actor-Critic (DDMAC)** with **Centralized Training, Decentralized Execution (CTDE)** for bridge network maintenance optimization. The system optimizes maintenance decisions for a 96-component highway network (85 pavement sections + 11 bridge decks) using off-policy RL with importance sampling and constrained optimization via Lagrangian relaxation.

## Commands

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training (primary workflow)
```bash
# Run from inside a variant folder
cd gpu_optimized/torch_compiled_fused_adam
python3 gpu_offpolicy_training.py

# With options
python3 gpu_offpolicy_training.py --checkpoint-dir checkpoints_offpolicy --wandb --device cuda:0
```

### Training via shell script (legacy PPO variant)
```bash
./train_gpu.sh 128 300000 cuda:0   # batch_size episodes device
```

### Testing
```bash
pytest Test_folder/test_gpu_training.py -v
pytest Test_folder/test_legacy_vs_gpu.py -v    # baseline vs GPU comparison
pytest Test_folder/cost_comparison_test.py -v
```

### Formatting/Linting
```bash
black .
flake8 .
```

## Architecture

### Production code lives in `gpu_optimized/`

Three self-contained variants, each with the same 6-file layout:

| Variant | Optimization | Best for |
|---------|-------------|----------|
| `torch_compiled/` | `torch.compile(mode="reduce-overhead")` | Baseline GPU speedup |
| `torch_compiled_fused_adam/` | + `optim.Adam(fused=True)` | **Preferred (best balance)** |
| `torch_compiled_buffer_log_prob/` | + stores `log_prob_sum` instead of full probs | Long runs (lower memory) |

### Core modules (6 files per variant)

- **`gpu_offpolicy_training.py`** — `OffPolicyConfig` dataclass (all hyperparameters), `ReplayBuffer` (cyclic GPU buffer), `OffPolicyTrainer` (collect_rollout → update → train loop), CLI entrypoint
- **`gpu_policy_network.py`** — `BridgeNetworkActor` (shared encoders per component type → per-component policy heads → action logits `(batch, 96, 10)`), `BridgeNetworkCritic` (3-layer MLP on flattened 1194-dim global state → scalar value), `BridgeNetworkActorCritic` container
- **`gpu_environment_optimized.py`** — `OptimizedGPUEnvironment` with fully vectorized belief state updates (Bayesian POMDP), memory-efficient matrix storage (unique transition matrices only, indexed by component type), batched operations across 128 parallel environments
- **`gpu_cost_module.py`** — `GPUCostModule` computing action/observation/delay/risk costs and 6 constraint violation types via `CostBreakdown` dataclass
- **`gpu_budget_constraints.py`** — `GPUBudgetConstraint` with 5 allocation strategies (GREEDY_BY_COST is default); budget resets every 5 steps
- **`Network_data.py`** — Loads transition matrices from `dependency_files/*.mat`, network geometry, component type definitions

### Key data flow

1. `OffPolicyTrainer.collect_rollout()` runs 128 parallel envs for 20 steps each
2. Epsilon-greedy behavior policy samples actions; transitions stored in `ReplayBuffer`
3. Every 4 episodes, `update()` samples 1024-transition minibatches, computes importance ratios `ρ = π/μ`, TD advantages `A = cost + λ·constraint - V(s) + γ·V(s')`, updates actor/critic
4. Lagrangian multipliers updated: `λ ← max(0, λ + lr_λ·(constraint - cap))`
5. Cosine LR annealing from episode 50k–1M

### State representation (POMDP belief states)

- **Pavement input** `(batch, 85, 14)`: IRI belief(5) + CCI belief(6) + extra_state(1) + age(1) + time(1)
- **Deck input** `(batch, 11, 9)`: deck belief(7) + age(1) + time(1)
- **Critic input** `(batch, 1194)`: flattened concatenation of all beliefs + extra + age + time

### Component types

| Type | Count | Belief dims | Notes |
|------|-------|-------------|-------|
| Interstate pavement | 12 | IRI(5)+CCI(6) | 8-lane highways |
| Primary pavement | 47 | IRI(5)+CCI(6) | 4-lane roads |
| Secondary pavement | 26 | IRI(5)+CCI(6) | 2-lane roads |
| Bridge deck | 11 | Deck(7) | Different transition matrices |

### Other directories

- **`DDMAC_CTDE_VDOT/`** — Original CPU baseline (Keras/TF). Reference implementation for validation, not actively developed.
- **`Test_folder/`** — Tests and experimental training scripts
- **`GPU_optimization/`**, **`profiling/`** — Experimental/profiling variants, not production
- **`dependency_files/`** — Static data: MATLAB transition matrices, cost arrays, initial state files

## Key Constants & Hyperparameters

Defined in `OffPolicyConfig` dataclass in `gpu_offpolicy_training.py`:
- 96 total components, 10 actions each, 20-step episodes
- Replay buffer: 100k capacity, update after 2000 steps
- 6 constraint caps: `(5.0, 2.0, 18.0, 15.0, 35.0, 10.0)`
- Budget: 1.3M units, resets every 5 steps with gamma discounting
- Default 2M episodes, checkpoints every 10k episodes

## Critical Implementation Details

- **Budget reset schedule**: Budget resets at steps 0, 5, 10, 15 with `budget_lim = γ^step × 1300`
- **Age clamping**: Component age capped at 19 (not 20)
- **Belief normalization**: After Bayesian update, beliefs must sum to 1.0; renormalize after every transition
- **Action 9 = replacement**: Resets component to pristine state (belief = [1,0,...,0])
- **Constraint sign**: Stored as negative in buffer (`-constr`), see GPU_VALIDATION_CHECKLIST.md
- **Transition matrices are age-dependent**: Indexed by `[component_type][age]`, loaded from MATLAB files in `dependency_files/`

## Validation

See `GPU_VALIDATION_CHECKLIST.md` for the full protocol to verify GPU implementation matches the CPU baseline. Key tolerances: belief updates within 1e-5, costs within 1e-4, constraint lambdas within 1e-5.
