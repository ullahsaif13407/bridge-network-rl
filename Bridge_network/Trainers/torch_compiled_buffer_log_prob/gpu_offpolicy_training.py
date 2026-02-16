"""
Off-Policy DDMAC-CTDE Style Training (Replay Buffer, Importance Sampling)
==========================================================================

Implements a replay-buffer-based trainer that mirrors the legacy
`DRL_Network_model_bdgt_lim_excludes_proximity_discount.py` loop:
- Large replay buffer (~30k steps)
- Behaviour policy sampling with epsilon exploration
- Importance-sampling ratio between target and behaviour policies
- Advantage via TD(0): A = (cost + lambda*constraint - V + gamma * V')
- Lagrangian constraint handling
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time
import os
import argparse

from gpu_environment_optimized import OptimizedGPUEnvironment, BeliefStateBatch
from gpu_policy_network import BridgeNetworkActorCritic
from gpu_cost_module import GPUCostModule
from gpu_budget_constraints import GPUBudgetConstraint, BudgetStrategy

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wandb = None

# Constants
NCOMP_PAV = 85
NCOMP_DECK = 11
TOT_COMP = 96
NACOMP = 10


@dataclass
class OffPolicyConfig:
    # ========================================================================
    # EFFICIENCY OPTIMIZED CONFIG
    # ========================================================================
    # Data Collection
    batch_envs: int = 64           # Increased from 32: More parallel envs = faster data collection
    episode_length: int = 20
    gamma: float = 0.97
    epsilon: float = 1.0          # behaviour policy epsilon (prob of using policy vs uniform)
    
    # Replay Buffer
    buffer_capacity: int = 100000  # Increased: More diverse experience for off-policy learning
    update_after: int = 2000      # Start updates after this many steps in buffer
    
    # Training Efficiency
    update_every: int = 4         # OPTIMIZED: Update every 4 episodes (was 1)
                                   # - Reduces overhead: 4x fewer forward/backward passes
                                   # - Still frequent enough for stable learning
                                   # - Better GPU utilization (batch more updates)
    mini_batch: int = 512         # OPTIMIZED: Increased from 256
                                   # - Larger batches = better GPU utilization
                                   # - More stable gradients
                                   # - Faster training (fewer kernel launches)
    grad_steps: int = 4           # OPTIMIZED: Increased from 1
                                   # - Better sample efficiency (multiple passes over data)
                                   # - Standard for off-policy methods (DDPG, TD3 use 1-4)
                                   # - Still fast with GPU batching
    
    # Learning Rates
    actor_lr: float = 1e-4        # MATCH LEGACY: 0.0001
    critic_lr: float = 1e-3       # MATCH LEGACY: 0.001

    # Learning-rate scheduling (optional; stepped once per update() call)
    # Supported: 'none', 'cosine', 'linear'
    lr_schedule: str = 'cosine'
   
    lr_schedule_min_lr_ratio: float = 0.01      # final_lr = initial_lr * ratio (clamped)
    lr_schedule_step_every: int = 1000            # step scheduler every N update() calls (staircase if >1)
    lr_schedule_updates: Optional[int] = None   # if None, uses total_episodes // lr_schedule_step_every 

    # Loss Coefficients
    entropy_coef: float = 0.0     # MATCH LEGACY: no entropy term
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    # Budget: MATCH LEGACY: 1.3*10**3 = 1300, with per-step decay
    budget_limit_base: float = 1300000.0
    budget_reset_every: int = 5   # MATCH LEGACY: reset every 5 steps
    # Use GREEDY_BY_COST for deterministic, lower-variance budget allocation
    # STOCHASTIC_PARALLEL causes huge variance in costs
    budget_strategy: BudgetStrategy = BudgetStrategy.GREEDY_BY_COST
    use_budget: bool = True  # False = infinite budget, no yt_ratio input, no budget handler
    device: str = 'cuda:0'
    total_episodes: int = 2000000
    # Constraint limits (cap_alp in legacy)
    constraint_caps: tuple = (5.0, 2.0, 18.0, 15.0, 35.0, 10.0)
    lambda_lr: float = 0.001      # MATCH LEGACY: lr_lamda
    # Checkpointing
    checkpoint_dir: str = './checkpoints_offpolicy'  # Relative to script's working directory
    checkpoint_every: int = 10000  # Save checkpoint every N episodes
    save_best: bool = False        # Save best model based on average cost

    # Weights & Biases logging (optional)
    use_wandb: bool = True
    wandb_project: str = 'Bridge_network_rl_comparision'
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_resume: bool = False
    wandb_log_every: int = 50
    
    def print_efficiency_metrics(self):
        """Print training efficiency metrics for this config."""
        steps_per_episode = self.batch_envs * self.episode_length
        steps_per_update = self.update_every * steps_per_episode
        samples_per_update = self.mini_batch * self.grad_steps
        
        print("\n" + "="*70)
        print("TRAINING EFFICIENCY METRICS")
        print("="*70)
        print(f"Data Collection:")
        print(f"  - Parallel environments: {self.batch_envs}")
        print(f"  - Steps per episode: {steps_per_episode}")
        print(f"  - Steps per update: {steps_per_update}")
        print(f"\nTraining:")
        print(f"  - Update frequency: Every {self.update_every} episodes")
        print(f"  - Mini-batch size: {self.mini_batch}")
        print(f"  - Gradient steps per update: {self.grad_steps}")
        print(f"  - Total samples per update: {samples_per_update}")
        print(f"\nEfficiency:")
        print(f"  - Update overhead: {1/self.update_every:.1%} of episodes")
        print(f"  - Sample efficiency: {samples_per_update/steps_per_update:.2f}x reuse")
        print(f"  - Buffer utilization: {self.buffer_capacity/steps_per_episode:.1f} episodes")
        print("="*70 + "\n")


class ReplayBuffer:
    """Simple cyclic replay buffer on GPU."""

    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0
        self.full = False
        self.state_IRI = torch.zeros((self.capacity, NCOMP_PAV, 5), device=self.device)
        self.state_CCI = torch.zeros((self.capacity, NCOMP_PAV, 6), device=self.device)
        self.state_deck = torch.zeros((self.capacity, NCOMP_DECK, 7), device=self.device)
        self.extra_state = torch.zeros((self.capacity, NCOMP_PAV), device=self.device)
        self.age = torch.zeros((self.capacity, TOT_COMP), device=self.device, dtype=torch.long)
        self.time_ratio = torch.zeros((self.capacity, 1), device=self.device)
        self.yt_ratio = torch.zeros((self.capacity, 1), device=self.device)  # Budget spent ratio
        self.actions = torch.zeros((self.capacity, TOT_COMP), dtype=torch.long, device=self.device)
        # Store only behavior-policy log-prob sum for taken actions (saves GPU memory).
        self.behavior_logp_sum = torch.zeros((self.capacity,), device=self.device)
        self.costs = torch.zeros((self.capacity,), device=self.device)  # Cost (negative)
        self.constraints = torch.zeros((self.capacity, 6), device=self.device)  # Constraint values
        self.dones = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        self.next_state_IRI = torch.zeros((self.capacity, NCOMP_PAV, 5), device=self.device)
        self.next_state_CCI = torch.zeros((self.capacity, NCOMP_PAV, 6), device=self.device)
        self.next_state_deck = torch.zeros((self.capacity, NCOMP_DECK, 7), device=self.device)
        self.next_extra_state = torch.zeros((self.capacity, NCOMP_PAV), device=self.device)
        self.next_age = torch.zeros((self.capacity, TOT_COMP), device=self.device, dtype=torch.long)
        self.next_time_ratio = torch.zeros((self.capacity, 1), device=self.device)
        self.next_yt_ratio = torch.zeros((self.capacity, 1), device=self.device)

    def push(
        self,
        belief: BeliefStateBatch,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
        yt_ratio: torch.Tensor,
        actions: torch.Tensor,
        behavior_logp_sum: torch.Tensor,  # (batch,) sum over components of log mu(a|s)
        costs: torch.Tensor,
        constraints: torch.Tensor,
        done: torch.Tensor,
        next_belief: BeliefStateBatch,
        next_extra_state: torch.Tensor,
        next_age: torch.Tensor,
        next_time_ratio: torch.Tensor,
        next_yt_ratio: torch.Tensor,
    ):
        batch = actions.shape[0]
        idx = torch.arange(batch, device=self.device) + self.ptr
        idx = idx % self.capacity

        self.state_IRI[idx] = belief.IRI
        self.state_CCI[idx] = belief.CCI
        self.state_deck[idx] = belief.deck
        self.extra_state[idx] = extra_state
        self.age[idx] = age
        self.time_ratio[idx] = time_ratio
        self.yt_ratio[idx] = yt_ratio
        self.actions[idx] = actions
        self.behavior_logp_sum[idx] = behavior_logp_sum
        self.costs[idx] = costs
        self.constraints[idx] = constraints
        self.dones[idx] = done
        self.next_state_IRI[idx] = next_belief.IRI
        self.next_state_CCI[idx] = next_belief.CCI
        self.next_state_deck[idx] = next_belief.deck
        self.next_extra_state[idx] = next_extra_state
        self.next_age[idx] = next_age
        self.next_time_ratio[idx] = next_time_ratio
        self.next_yt_ratio[idx] = next_yt_ratio

        self.ptr = (self.ptr + batch) % self.capacity
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        max_idx = self.capacity if self.full else self.ptr
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return {
            'state_IRI': self.state_IRI[idx],
            'state_CCI': self.state_CCI[idx],
            'state_deck': self.state_deck[idx],
            'extra_state': self.extra_state[idx],
            'age': self.age[idx],
            'time_ratio': self.time_ratio[idx],
            'yt_ratio': self.yt_ratio[idx],
            'actions': self.actions[idx],
            'behavior_logp_sum': self.behavior_logp_sum[idx],
            'costs': self.costs[idx],
            'constraints': self.constraints[idx],
            'dones': self.dones[idx],
            'next_state_IRI': self.next_state_IRI[idx],
            'next_state_CCI': self.next_state_CCI[idx],
            'next_state_deck': self.next_state_deck[idx],
            'next_extra_state': self.next_extra_state[idx],
            'next_age': self.next_age[idx],
            'next_time_ratio': self.next_time_ratio[idx],
            'next_yt_ratio': self.next_yt_ratio[idx],
        }


class OffPolicyTrainer:
    """DDMAC-CTDE style off-policy trainer - matches legacy implementation."""

    def __init__(self, config: OffPolicyConfig):
        self.cfg = config
        self.device = config.device
        self._wandb_run = None

        # Resolve checkpoint directory BEFORE any modules change working directory
        # (gpu_environment_optimized.py calls os.chdir() which would affect relative paths)
        original_cwd = os.getcwd()
        self.checkpoint_dir_abs = os.path.abspath(config.checkpoint_dir)
        # Restore original directory (in case it was changed)
        os.chdir(original_cwd)
        
        self.env = OptimizedGPUEnvironment(device=self.device, batch_size=config.batch_envs)
        self.cost_module = GPUCostModule(device=self.device)
        self.policy = BridgeNetworkActorCritic(shared_actor=False, use_budget=config.use_budget).to(self.device)

        # Optional: torch.compile to reduce kernel-launch overhead and fuse pointwise ops.
        if hasattr(torch, "compile"):
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
            except Exception:
                pass

        self.actor_opt = optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr)
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.update_calls = 0
        self._init_lr_schedulers()

        self.buffer = ReplayBuffer(config.buffer_capacity, self.device)
        self.budget_handler = GPUBudgetConstraint(
            self.cost_module.cost_action,
            self.cost_module.cost_obsr,
            gamma=self.cost_module.gamma,
            device=self.device,
            strategy=config.budget_strategy,
        )

        # Lagrangian multipliers for constraints (MATCH LEGACY)
        self.lamda = torch.zeros(6, device=self.device)
        self.cap_alp = torch.tensor(config.constraint_caps, device=self.device)
        self.lr_lamda = config.lambda_lr

        self.global_steps = 0
        self.episodes_done = 0
        self.best_avg_cost = float('inf')  # Track best average cost
        
        # Create checkpoint directory (use pre-resolved absolute path)
        os.makedirs(self.checkpoint_dir_abs, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir_abs}")

    def _init_lr_schedulers(self) -> None:
        schedule = (self.cfg.lr_schedule or 'none').lower()
        if schedule == 'none':
            return

        total_updates = self.cfg.lr_schedule_updates
        if total_updates is None:
            total_updates = max(1, int(self.cfg.total_episodes // max(1, self.cfg.lr_schedule_step_every)))

        min_lr_ratio = float(self.cfg.lr_schedule_min_lr_ratio)
        if not np.isfinite(min_lr_ratio) or min_lr_ratio <= 0:
            raise ValueError(f"lr_schedule_min_lr_ratio must be > 0, got {self.cfg.lr_schedule_min_lr_ratio}")

        if schedule == 'cosine':
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt,
                T_max=total_updates,
                eta_min=self.cfg.actor_lr * min_lr_ratio,
            )
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_opt,
                T_max=total_updates,
                eta_min=self.cfg.critic_lr * min_lr_ratio,
            )
            return

        if schedule == 'linear':
            def linear_factor(step: int) -> float:
                if total_updates <= 1:
                    return 1.0
                progress = min(step, total_updates) / float(total_updates)
                return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

            self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_opt, lr_lambda=linear_factor)
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_opt, lr_lambda=linear_factor)
            return

        raise ValueError(f"Unsupported lr_schedule={self.cfg.lr_schedule!r}. Use 'none', 'cosine', or 'linear'.")

    def collect_rollout(self):
        """Collect one episode with budget tracking matching legacy.
        
        Returns:
            ep_cost_mean: mean discounted episode cost across envs
            ep_constraint_mean: mean constraint vector across envs
            ep_risk_mean: mean discounted risk cost across envs
            ep_action_obs_mean: mean discounted (action + observation) cost across envs
            ep_delay_mean: mean discounted delay cost across envs
        """
        batch = self.cfg.batch_envs
        ep_len = self.cfg.episode_length
        gamma = self.cfg.gamma
        
        # Load initial conditions matching legacy (from files)
        belief = self.env.create_initial_belief('legacy')
        ages = self.env.load_initial_ages()
        extra_state = torch.zeros((batch, NCOMP_PAV), device=self.device)
        
        ep_cost = torch.zeros(batch, device=self.device)
        ep_risk = torch.zeros(batch, device=self.device)
        ep_action_obs = torch.zeros(batch, device=self.device)
        ep_delay = torch.zeros(batch, device=self.device)
        ep_constraint_sum = torch.zeros(batch, 6, device=self.device)

        # Budget tracking (MATCH LEGACY: yt and budget_lim)
        yt = torch.zeros(batch, device=self.device)  # Cumulative spending
        budget_lim = torch.ones(batch, device=self.device) * self.cfg.budget_limit_base
        tau = 0  # Time since last budget reset

        for t in range(ep_len):
            time_ratio = torch.full((batch, 1), t / ep_len, device=self.device)
            
            # Budget reset every 5 steps (MATCH LEGACY)
            if t % self.cfg.budget_reset_every == 0:
                yt.zero_()
                budget_lim = (gamma ** t) * self.cfg.budget_limit_base * torch.ones(batch, device=self.device)
                tau = 0
            
            if self.cfg.use_budget:
                yt_ratio = (yt / budget_lim).unsqueeze(1)  # For state input
                budget_remaining = budget_lim - yt
            else:
                yt_ratio = None

            with torch.no_grad():
                action_logits, values = self.policy(
                    belief.IRI, belief.CCI, belief.deck,
                    extra_state, ages.float(), time_ratio, yt_ratio
                )
                # Behavior policy probabilities (used only to compute log mu(a|s) for replay).
                behavior_probs = F.softmax(action_logits, dim=-1)  # (batch, tot_comp, nacomp)

                behavior_dist = torch.distributions.Categorical(logits=action_logits)

                # Epsilon: with probability epsilon, use policy; else uniform
                if np.random.rand() < self.cfg.epsilon:
                    actions = behavior_dist.sample()
                else:
                    actions = torch.randint(0, NACOMP, (batch, TOT_COMP), device=self.device)

            # Budget constraint with remaining budget (MATCH LEGACY)
            if self.cfg.use_budget:
                can_repair = torch.ones(batch, TOT_COMP, dtype=torch.bool, device=self.device)
                comp_active, action_real = self.budget_handler.apply(
                    actions, budget_remaining, can_repair, temperature=1.0
                )
            else:
                comp_active = torch.ones(batch, TOT_COMP, dtype=torch.bool, device=self.device)
                action_real = actions

            # Sum of behavior-policy log-prob for executed actions (over components).
            mu_aa = torch.gather(behavior_probs, 2, action_real.unsqueeze(-1)).squeeze(-1)  # (batch, tot_comp)
            behavior_logp_sum = torch.log(mu_aa + 1e-10).sum(dim=1)  # (batch,)

            # Env step
            ages_prime = self.env.age_action(ages, action_real, comp_active)
            b_prime, b_a, extra_state_prime, _ = self.env.belief_update_vectorized(
                belief, action_real, ages_prime, extra_state, comp_active
            )

            # Costs (MATCH LEGACY: cost_action, cost_ob, cost_dlay, cost_risk)
            action_cost, obs_cost, _ = self.cost_module.compute_action_observation_costs(
                action_real, comp_active
            )
            delay_cost = self.cost_module.compute_delay_cost(action_real, comp_active)
            risk_cost = self.cost_module.compute_risk_cost(
                b_a.deck, ages_prime, action_real, comp_active, self.env.pcomp_deck
            )
            

            
            total_cost = action_cost + obs_cost + delay_cost + risk_cost  # Negative values
            
            # Accumulate episode cost with discount
            discount = (gamma ** t)
            ep_cost += discount * total_cost
            ep_risk += discount * risk_cost
            ep_action_obs += discount * (action_cost + obs_cost)
            ep_delay += discount * delay_cost

            # Update budget spent (MATCH LEGACY: yt += -gamma**tau*(cost_action + cost_ob))
            # Note: costs are negative, so -cost is positive spending
            if self.cfg.use_budget:
                yt += -(gamma ** tau) * (action_cost + obs_cost)
                tau += 1
            
            # MATCH LEGACY: age_prime += 1 after risk cost computation
            # Legacy does: age_prime += 1, then caps at 19 if it reaches 20
            ages_prime = ages_prime + 1
            ages_prime = torch.clamp(ages_prime, max=19)  # Cap at 19 like legacy

            done = torch.zeros(batch, dtype=torch.bool, device=self.device)
            if t == ep_len - 1:
                done[:] = True

            # Constraints BEFORE action (MATCH LEGACY: constr = Constraints(state))
            constraints = self.cost_module.compute_constraints(
                belief.IRI, belief.CCI, belief.deck
            )
            ep_constraint_sum += constraints

            # Next state budget ratio
            if self.cfg.use_budget:
                next_yt_ratio = (yt / budget_lim).unsqueeze(1)
                if (t + 1) % self.cfg.budget_reset_every == 0:
                    next_yt_ratio = torch.zeros(batch, 1, device=self.device)
            else:
                next_yt_ratio = torch.zeros(batch, 1, device=self.device)

            # Push to buffer
            # MATCH LEGACY: 
            # - Current state age: age (old, before increment) - stored in buffer_age[e]
            # - Next state age: age_prime (after increment) - stored in buffer_age[e_next] for next step
            # Legacy: buffer_age[buffer_count] = age (old), then age = age_prime (new for next step)
            buf_yt_ratio = yt_ratio if yt_ratio is not None else torch.zeros(batch, 1, device=self.device)
            self.buffer.push(
                belief, extra_state, ages, time_ratio, buf_yt_ratio,
                action_real, behavior_logp_sum,
                total_cost,  # Store cost (negative), not reward
                -constraints,  # Store negative constraints (MATCH LEGACY: buffer_constraint = -constr)
                done,
                b_prime, extra_state_prime, ages_prime,  # Next age is incremented age (for next step)
                torch.full_like(time_ratio, (t + 1) / ep_len),
                next_yt_ratio
            )

            belief = b_prime
            ages = ages_prime
            extra_state = extra_state_prime
            self.global_steps += batch
        
        # Update Lagrangian multipliers (MATCH LEGACY)
        avg_constr = (ep_constraint_sum / ep_len).mean(dim=0)  # (6,)
        self.lamda = self.lamda + self.lr_lamda * (avg_constr - self.cap_alp)
        self.lamda = torch.clamp(self.lamda, min=0)  # lamda = max(lamda, 0)
        
        ep_cost_mean = ep_cost.mean().item()
        ep_risk_mean = ep_risk.mean().item()
        ep_action_obs_mean = ep_action_obs.mean().item()
        ep_delay_mean = ep_delay.mean().item()
        ep_constraint_mean = (ep_constraint_sum / ep_len).mean(dim=0).detach().cpu().numpy()
        return ep_cost_mean, ep_constraint_mean, ep_risk_mean, ep_action_obs_mean, ep_delay_mean

    def update(self):
        """Off-policy update matching legacy implementation."""
        if len(self.buffer) < self.cfg.update_after:
            return
        self.update_calls += 1

        for _ in range(self.cfg.grad_steps):
            batch = self.buffer.sample(self.cfg.mini_batch)
            bs = self.cfg.mini_batch

            # Check for NaN in inputs
            if torch.isnan(batch['state_IRI']).any() or torch.isnan(batch['state_CCI']).any():
                print("WARNING: NaN detected in input states, skipping batch")
                continue

            # Current policy (MATCH LEGACY: pi_ac)
            yt = batch['yt_ratio'] if self.cfg.use_budget else None
            next_yt = batch['next_yt_ratio'] if self.cfg.use_budget else None
            action_logits, values = self.policy(
                batch['state_IRI'], batch['state_CCI'], batch['state_deck'],
                batch['extra_state'], batch['age'].float(), batch['time_ratio'],
                yt
            )

            if torch.isnan(action_logits).any():
                print("WARNING: NaN in action_logits, skipping batch")
                continue

            curr_probs = F.softmax(action_logits, dim=-1)  # (bs, tot_comp, nacomp)

            # Next state value (MATCH LEGACY: value_next)
            with torch.no_grad():
                _, next_values = self.policy(
                    batch['next_state_IRI'], batch['next_state_CCI'], batch['next_state_deck'],
                    batch['next_extra_state'], batch['next_age'].float(), batch['next_time_ratio'],
                    next_yt
                )

            # MATCH LEGACY ADVANTAGE:
            # Adv = cost + lamda.dot(constraints) - value + (1 - flag) * gamma * value_next
            # Note: constraints stored as negative in buffer
            constraint_penalty = torch.matmul(batch['constraints'], self.lamda)  # (bs,)
            
            adv = (batch['costs'] + constraint_penalty 
                   - values.squeeze(-1) 
                   + (~batch['dones']).float() * self.cfg.gamma * next_values.squeeze(-1))

            # Importance sampling ratio (MATCH LEGACY)
            # rho = prod(pi_aa / mu_aa) over all components
            actions = batch['actions']  # (bs, tot_comp)
            
            # Get pi(a|s) for taken actions
            pi_aa = torch.gather(curr_probs, 2, actions.unsqueeze(-1)).squeeze(-1)  # (bs, tot_comp)
            
            # Compute ratio with product (MATCH LEGACY), using log for stability
            pi_aa_log = torch.log(pi_aa + 1e-10)  # (bs, tot_comp)
            log_ratio = pi_aa_log.sum(dim=1) - batch['behavior_logp_sum']  # (bs,)
            # Clip to prevent explosion
            log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
            rho = torch.exp(log_ratio)
            # Clip ratio (MATCH LEGACY: rho = min(rho, 2.0))
            rho = torch.clamp(rho, max=2.0)
            
            # Weighted advantage
            weighted_adv = rho * adv

            # MATCH LEGACY VALUE TARGET: Vnet.train_on_batch(inp_vnet, Adv + value)
            value_target = adv + values.squeeze(-1)
            critic_loss = F.mse_loss(values.squeeze(-1), value_target.detach())

            # MATCH LEGACY POLICY LOSS: 
            # y_train_P[k][ind, action[k]] = Adv
            # This is cross-entropy with target = one-hot * Adv
            # Equivalent to: -sum(log(pi) * weighted_adv) for taken actions
            actor_loss = -(pi_aa_log * weighted_adv.unsqueeze(1).detach()).sum(dim=1).mean()

            loss = actor_loss + self.cfg.value_coef * critic_loss
            
            if torch.isnan(loss):
                print("WARNING: NaN loss, skipping update")
                continue

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()

        step_every = max(1, int(self.cfg.lr_schedule_step_every))
        if self.actor_scheduler is not None and (self.update_calls % step_every == 0):
            self.actor_scheduler.step()
        if self.critic_scheduler is not None and (self.update_calls % step_every == 0):
            self.critic_scheduler.step()

    def save_checkpoint(
        self,
        episode: int,
        ep_cost_log: list,
        ep_constr_log: list,
        ep_risk_log: list,
        ep_action_obs_log: list,
        ep_delay_log: list,
        is_best: bool = False,
    ):
        """Save training checkpoint.
        
        Args:
            episode: Current episode number
            ep_cost_log: List of episode costs
            ep_constr_log: List of episode constraints
            ep_risk_log: List of episode risk costs
            ep_action_obs_log: List of episode (action+observation) costs
            ep_delay_log: List of episode delay costs
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'episode': episode,
            'global_steps': self.global_steps,
            'episodes_done': self.episodes_done,
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict() if self.actor_scheduler is not None else None,
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict() if self.critic_scheduler is not None else None,
            'update_calls': self.update_calls,
            'lamda': self.lamda.cpu(),
            'best_avg_cost': self.best_avg_cost,
            'ep_cost_log': ep_cost_log,
            'ep_constr_log': ep_constr_log,
            'ep_risk_log': ep_risk_log,
            'ep_action_obs_log': ep_action_obs_log,
            'ep_delay_log': ep_delay_log,
            'config': self.cfg,
        }
        
        # Save latest checkpoint (use pre-resolved absolute path)
        latest_path = os.path.join(self.checkpoint_dir_abs, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint to {latest_path} (episode {episode})")
        
        # Save periodic checkpoint
        if episode % self.cfg.checkpoint_every == 0:
            periodic_path = os.path.join(self.checkpoint_dir_abs, f'checkpoint_ep{episode}.pt')
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint to {periodic_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir_abs, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved BEST checkpoint to {best_path} (avg_cost={self.best_avg_cost:.3f})")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[list, list, list, list, list, int]:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            ep_cost_log, ep_constr_log, ep_risk_log, ep_action_obs_log, ep_delay_log, start_episode: Training history
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        actor_sched_state = checkpoint.get('actor_scheduler_state_dict')
        critic_sched_state = checkpoint.get('critic_scheduler_state_dict')
        if self.actor_scheduler is not None and actor_sched_state is not None:
            self.actor_scheduler.load_state_dict(actor_sched_state)
        if self.critic_scheduler is not None and critic_sched_state is not None:
            self.critic_scheduler.load_state_dict(critic_sched_state)
        
        # Load training state
        self.global_steps = checkpoint['global_steps']
        self.episodes_done = checkpoint['episodes_done']
        self.lamda = checkpoint['lamda'].to(self.device)
        self.best_avg_cost = checkpoint['best_avg_cost']
        self.update_calls = int(checkpoint.get('update_calls', 0))
        
        # Load training history
        ep_cost_log = checkpoint.get('ep_cost_log', [])
        ep_constr_log = checkpoint.get('ep_constr_log', [])
        ep_risk_log = checkpoint.get('ep_risk_log', [])
        ep_action_obs_log = checkpoint.get('ep_action_obs_log', [])
        ep_delay_log = checkpoint.get('ep_delay_log', [])
        
        start_episode = checkpoint['episode']
        print(f"Resumed from episode {start_episode}, steps={self.global_steps}")
        print(f"Best avg cost so far: {self.best_avg_cost:.3f}")
        
        return ep_cost_log, ep_constr_log, ep_risk_log, ep_action_obs_log, ep_delay_log, start_episode

    def _maybe_init_wandb(self):
        if not self.cfg.use_wandb:
            return
        if wandb is None:
            print("WARNING: use_wandb=True but wandb is not installed; skipping W&B logging")
            return
        if self._wandb_run is not None:
            return

        wandb_config = {}
        for key, value in self.cfg.__dict__.items():
            if isinstance(value, BudgetStrategy):
                wandb_config[key] = value.name
            else:
                wandb_config[key] = value

        init_kwargs = {
            'project': self.cfg.wandb_project,
            'entity': self.cfg.wandb_entity,
            'name': self.cfg.wandb_run_name,
            'config': wandb_config,
        }
        if self.cfg.wandb_resume:
            init_kwargs['resume'] = 'allow'

        self._wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})

    def train(self, resume_from: Optional[str] = None):
        """Main training loop.
        
        Args:
            resume_from: Path to checkpoint file to resume from
        """
        start = time.time()
        ep_cost_log = []
        ep_constr_log = []
        ep_risk_log = []
        ep_action_obs_log = []
        ep_delay_log = []
        start_episode = 0

        self._maybe_init_wandb()
        
        # Load checkpoint if resuming
        if resume_from:
            if os.path.exists(resume_from):
                (
                    ep_cost_log,
                    ep_constr_log,
                    ep_risk_log,
                    ep_action_obs_log,
                    ep_delay_log,
                    start_episode,
                ) = self.load_checkpoint(resume_from)
            else:
                print(f"WARNING: Checkpoint {resume_from} not found, starting from scratch")
        
        for ep in range(start_episode, self.cfg.total_episodes):
            ep_cost, ep_constr, ep_risk, ep_action_obs, ep_delay = self.collect_rollout()
            ep_cost_log.append(ep_cost)
            ep_constr_log.append(ep_constr)
            ep_risk_log.append(ep_risk)
            ep_action_obs_log.append(ep_action_obs)
            ep_delay_log.append(ep_delay)
            self.episodes_done += 1
            
            # Update based on frequency (optimized for efficiency)
            if len(self.buffer) >= self.cfg.update_after:
                if (ep + 1) % self.cfg.update_every == 0:
                    self.update()

            if self._wandb_run is not None and (ep + 1) % self.cfg.wandb_log_every == 0:
                k = self.cfg.wandb_log_every
                if len(ep_cost_log) >= k:
                    lambda_norm = torch.norm(self.lamda).item()
                    wandb.log(
                        {
                            "Total_cost": float(np.mean(ep_cost_log[-k:]) * 1000.0),
                            "Episodic Risk Cost": float(np.mean(ep_risk_log[-k:]) * 1000.0),
                            "Episodic_Cost_ac_ob": float(np.mean(ep_action_obs_log[-k:]) * 1000.0),
                            "Delay_Cost": float(np.mean(ep_delay_log[-k:]) * 1000.0),
                            "Lambda": float(lambda_norm),
                            "episode": float((ep + 1) / k),
                            "episode_raw": int(ep + 1),
                            "global_steps": int(self.global_steps),
                            "buffer_size": int(len(self.buffer)),
                            "actor_lr": float(self.actor_opt.param_groups[0]["lr"]),
                            "critic_lr": float(self.critic_opt.param_groups[0]["lr"]),
                        },
                        step=int(ep + 1),
                    )

            if (ep + 1) % 100 == 0:
                elapsed = time.time() - start
                costs_100 = ep_cost_log[-100:]
                avg_cost = np.mean(costs_100) * 1000  # Scale like legacy
                std_cost = np.std(costs_100) * 1000
                min_cost = np.min(costs_100) * 1000
                max_cost = np.max(costs_100) * 1000
                last_constr = ep_constr_log[-1]
                print(f"time elapsed {elapsed:.1f}s")
                print(f"Average_costs {avg_cost:.3f} (std={std_cost:.1f}, min={min_cost:.1f}, max={max_cost:.1f})")
                print(f"Constraints: {last_constr}")
                print(f"Lambda: {self.lamda.cpu().numpy()}")
                
                # Check if this is best model
                is_best = False
                if self.cfg.save_best and avg_cost < self.best_avg_cost:
                    self.best_avg_cost = avg_cost
                    is_best = True
                
                # Save checkpoint
                self.save_checkpoint(
                    ep + 1,
                    ep_cost_log,
                    ep_constr_log,
                    ep_risk_log,
                    ep_action_obs_log,
                    ep_delay_log,
                    is_best=is_best,
                )
                
                start = time.time()

            if (ep + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"[OffPolicy] Episode {ep+1} | steps={self.global_steps} | buffer={len(self.buffer)} | time={elapsed:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Off-Policy DDMAC-CTDE Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb-resume', action='store_true', help='Resume/allow resuming W&B run')
    parser.add_argument('--wandb-log-every', type=int, default=None, help='Log to W&B every N episodes')
    parser.add_argument('--lr-schedule', type=str, default=None, choices=['none', 'cosine', 'linear'],
                        help='LR schedule for actor/critic (stepped per update)')
    parser.add_argument('--lr-schedule-updates', type=int, default=None,
                        help='Total scheduler steps (defaults to total_episodes // update_every)')
    parser.add_argument('--lr-min-ratio', type=float, default=None,
                        help='Min LR ratio vs initial LR (e.g. 0.1)')
    parser.add_argument('--lr-step-every', type=int, default=None,
                        help='Step LR scheduler every N update() calls (staircase if >1)')
    parser.add_argument('--no-budget', action='store_true',
                        help='Disable budget constraints (infinite budget, no yt_ratio input)')
    args = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = OffPolicyConfig(device=device)
    
    # Override checkpoint directory if provided
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir

    if args.wandb:
        cfg.use_wandb = True
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity:
        cfg.wandb_entity = args.wandb_entity
    if args.wandb_name:
        cfg.wandb_run_name = args.wandb_name
    if args.wandb_resume:
        cfg.wandb_resume = True
    if args.wandb_log_every is not None:
        cfg.wandb_log_every = args.wandb_log_every

    if args.lr_schedule is not None:
        cfg.lr_schedule = args.lr_schedule
    if args.lr_schedule_updates is not None:
        cfg.lr_schedule_updates = args.lr_schedule_updates
    if args.lr_min_ratio is not None:
        cfg.lr_schedule_min_lr_ratio = args.lr_min_ratio
    if args.lr_step_every is not None:
        cfg.lr_schedule_step_every = args.lr_step_every
    if args.no_budget:
        cfg.use_budget = False

    # Print efficiency metrics
    cfg.print_efficiency_metrics()
    
    trainer = OffPolicyTrainer(cfg)
    trainer.train(resume_from=args.resume)
