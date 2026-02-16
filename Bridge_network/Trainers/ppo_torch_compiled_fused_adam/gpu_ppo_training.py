"""
PPO DDMAC-CTDE Training (On-Policy, Clipped Surrogate + GAE)
=============================================================

Proximal Policy Optimization for bridge network maintenance.
On-policy: collect rollout -> compute GAE -> PPO update -> discard.
Uses same env/cost/budget/constraint infrastructure as off-policy baseline.
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
class PPOConfig:
    # Data Collection
    batch_envs: int = 128
    episode_length: int = 20
    gamma: float = 0.97

    # PPO-specific
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    num_minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    normalize_advantages: bool = True
    target_kl: Optional[float] = None  # Optional early stopping on KL

    # Learning Rates
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5

    # LR scheduling
    lr_schedule: str = 'cosine'
    lr_schedule_updates: Optional[int] = None
    lr_schedule_start_episode: int = 100000
    lr_schedule_end_episode: int = 1000000
    lr_schedule_min_lr_ratio: float = 0.01
    lr_schedule_step_every: int = 1000

    # Budget
    budget_limit_base: float = 1300000.0
    budget_reset_every: int = 5
    budget_strategy: BudgetStrategy = BudgetStrategy.GREEDY_BY_COST
    use_budget: bool = True  # False = infinite budget, no yt_ratio input, no budget handler

    device: str = 'cuda:0'
    total_episodes: int = 2000000

    # Constraints
    constraint_caps: tuple = (5.0, 2.0, 18.0, 15.0, 35.0, 10.0)
    lambda_lr: float = 0.001

    # Checkpointing
    checkpoint_dir: str = './checkpoints_ppo'
    checkpoint_every: int = 10000
    save_best: bool = True

    # W&B
    use_wandb: bool = True
    wandb_project: str = 'Bridge_network_rl_comparision'
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_resume: bool = False
    wandb_log_every: int = 50


class RolloutBuffer:
    """Temporary storage for one episode's rollout data (on-policy).

    Stores (episode_length * batch_envs) transitions, computes GAE,
    then yields minibatches. Discarded after each PPO update.
    """

    def __init__(self, episode_length: int, batch_envs: int, device: str):
        self.episode_length = episode_length
        self.batch_envs = batch_envs
        self.device = device
        self.total = episode_length * batch_envs
        self.reset()

    def reset(self):
        T, B = self.episode_length, self.batch_envs
        self.state_IRI = torch.zeros((T, B, NCOMP_PAV, 5), device=self.device)
        self.state_CCI = torch.zeros((T, B, NCOMP_PAV, 6), device=self.device)
        self.state_deck = torch.zeros((T, B, NCOMP_DECK, 7), device=self.device)
        self.extra_state = torch.zeros((T, B, NCOMP_PAV), device=self.device)
        self.age = torch.zeros((T, B, TOT_COMP), device=self.device)
        self.time_ratio = torch.zeros((T, B, 1), device=self.device)
        self.yt_ratio = torch.zeros((T, B, 1), device=self.device)
        self.actions = torch.zeros((T, B, TOT_COMP), dtype=torch.long, device=self.device)
        self.log_prob_sum = torch.zeros((T, B), device=self.device)  # sum over 96 components
        self.values = torch.zeros((T, B), device=self.device)
        self.costs = torch.zeros((T, B), device=self.device)
        self.constraint_penalties = torch.zeros((T, B), device=self.device)
        self.dones = torch.zeros((T, B), dtype=torch.bool, device=self.device)
        # Computed after rollout
        self.advantages = torch.zeros((T, B), device=self.device)
        self.returns = torch.zeros((T, B), device=self.device)
        self.ptr = 0

    def store(
        self,
        t: int,
        belief: BeliefStateBatch,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
        yt_ratio: torch.Tensor,
        actions: torch.Tensor,
        log_prob_sum: torch.Tensor,
        values: torch.Tensor,
        costs: torch.Tensor,
        constraint_penalty: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.state_IRI[t] = belief.IRI
        self.state_CCI[t] = belief.CCI
        self.state_deck[t] = belief.deck
        self.extra_state[t] = extra_state
        self.age[t] = age
        self.time_ratio[t] = time_ratio
        self.yt_ratio[t] = yt_ratio
        self.actions[t] = actions
        self.log_prob_sum[t] = log_prob_sum
        self.values[t] = values
        self.costs[t] = costs
        self.constraint_penalties[t] = constraint_penalty
        self.dones[t] = dones

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Backward scan GAE computation.

        delta_t = cost_t + constraint_penalty_t - V_t + gamma * V_{t+1}
        A_t = delta_t + gamma * lambda * A_{t+1}
        R_t = A_t + V_t  (returns for value target)
        """
        T, B = self.episode_length, self.batch_envs
        gae = torch.zeros(B, device=self.device)
        next_value = last_value  # V(s_{T})

        for t in reversed(range(T)):
            not_done = (~self.dones[t]).float()
            delta = (self.costs[t] + self.constraint_penalties[t]
                     - self.values[t]
                     + not_done * gamma * next_value)
            gae = delta + not_done * gamma * gae_lambda * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]

    def get_minibatch_generator(self, num_minibatches: int):
        """Flatten (T, B) -> (T*B,), shuffle, yield minibatches."""
        T, B = self.episode_length, self.batch_envs
        N = T * B

        # Flatten all tensors
        flat = {
            'state_IRI': self.state_IRI.reshape(N, NCOMP_PAV, 5),
            'state_CCI': self.state_CCI.reshape(N, NCOMP_PAV, 6),
            'state_deck': self.state_deck.reshape(N, NCOMP_DECK, 7),
            'extra_state': self.extra_state.reshape(N, NCOMP_PAV),
            'age': self.age.reshape(N, TOT_COMP),
            'time_ratio': self.time_ratio.reshape(N, 1),
            'yt_ratio': self.yt_ratio.reshape(N, 1),
            'actions': self.actions.reshape(N, TOT_COMP),
            'log_prob_sum': self.log_prob_sum.reshape(N),
            'values': self.values.reshape(N),
            'advantages': self.advantages.reshape(N),
            'returns': self.returns.reshape(N),
        }

        # Shuffle
        indices = torch.randperm(N, device=self.device)
        mb_size = N // num_minibatches

        for i in range(num_minibatches):
            idx = indices[i * mb_size: (i + 1) * mb_size]
            yield {k: v[idx] for k, v in flat.items()}


class PPOTrainer:
    """PPO trainer for DDMAC-CTDE bridge network maintenance."""

    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = config.device
        self._wandb_run = None

        original_cwd = os.getcwd()
        self.checkpoint_dir_abs = os.path.abspath(config.checkpoint_dir)
        os.chdir(original_cwd)

        self.env = OptimizedGPUEnvironment(device=self.device, batch_size=config.batch_envs)
        self.cost_module = GPUCostModule(device=self.device)
        self.policy = BridgeNetworkActorCritic(shared_actor=False, use_budget=config.use_budget).to(self.device)

        if hasattr(torch, "compile"):
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
            except Exception:
                pass

        if str(self.device).startswith("cuda"):
            try:
                self.actor_opt = optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr, fused=True)
                self.critic_opt = optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr, fused=True)
            except Exception:
                self.actor_opt = optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
                self.critic_opt = optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr)
        else:
            self.actor_opt = optim.Adam(self.policy.actor.parameters(), lr=config.actor_lr)
            self.critic_opt = optim.Adam(self.policy.critic.parameters(), lr=config.critic_lr)

        self.actor_scheduler = None
        self.critic_scheduler = None
        self.update_calls = 0
        self._init_lr_schedulers()

        self.rollout_buffer = RolloutBuffer(config.episode_length, config.batch_envs, self.device)

        self.budget_handler = GPUBudgetConstraint(
            self.cost_module.cost_action,
            self.cost_module.cost_obsr,
            gamma=self.cost_module.gamma,
            device=self.device,
            strategy=config.budget_strategy,
        )

        # Lagrangian multipliers
        self.lamda = torch.zeros(6, device=self.device)
        self.cap_alp = torch.tensor(config.constraint_caps, device=self.device)
        self.lr_lamda = config.lambda_lr

        self.global_steps = 0
        self.episodes_done = 0
        self.best_avg_cost = float('inf')

        os.makedirs(self.checkpoint_dir_abs, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir_abs}")

    def _init_lr_schedulers(self) -> None:
        schedule = (self.cfg.lr_schedule or 'none').lower()
        if schedule == 'none':
            return

        total_updates = self.cfg.lr_schedule_updates
        if total_updates is None:
            num_updates = (self.cfg.lr_schedule_end_episode - self.cfg.lr_schedule_start_episode) // self.cfg.lr_schedule_step_every
            total_updates = max(1, num_updates)

        min_lr_ratio = float(self.cfg.lr_schedule_min_lr_ratio)
        if not np.isfinite(min_lr_ratio) or min_lr_ratio <= 0:
            raise ValueError(f"lr_schedule_min_lr_ratio must be > 0, got {self.cfg.lr_schedule_min_lr_ratio}")

        if schedule == 'cosine':
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, T_max=total_updates,
                eta_min=self.cfg.actor_lr * min_lr_ratio,
            )
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_opt, T_max=total_updates,
                eta_min=self.cfg.critic_lr * min_lr_ratio,
            )
        elif schedule == 'linear':
            def linear_factor(step: int) -> float:
                if total_updates <= 1:
                    return 1.0
                progress = min(step, total_updates) / float(total_updates)
                return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

            self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_opt, lr_lambda=linear_factor)
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_opt, lr_lambda=linear_factor)
        else:
            raise ValueError(f"Unsupported lr_schedule={self.cfg.lr_schedule!r}. Use 'none', 'cosine', or 'linear'.")

    def collect_rollout(self):
        """Collect one episode across batch_envs, store in RolloutBuffer.

        Returns:
            ep_cost_mean, ep_constraint_mean, ep_risk_mean,
            ep_action_obs_mean, ep_delay_mean
        """
        batch = self.cfg.batch_envs
        ep_len = self.cfg.episode_length
        gamma = self.cfg.gamma

        self.rollout_buffer.reset()

        belief = self.env.create_initial_belief('legacy')
        ages = self.env.load_initial_ages()
        extra_state = torch.zeros((batch, NCOMP_PAV), device=self.device)

        ep_cost = torch.zeros(batch, device=self.device)
        ep_risk = torch.zeros(batch, device=self.device)
        ep_action_obs = torch.zeros(batch, device=self.device)
        ep_delay = torch.zeros(batch, device=self.device)
        ep_constraint_sum = torch.zeros(batch, 6, device=self.device)

        yt = torch.zeros(batch, device=self.device)
        budget_lim = torch.ones(batch, device=self.device) * self.cfg.budget_limit_base
        tau = 0

        for t in range(ep_len):
            time_ratio = torch.full((batch, 1), t / ep_len, device=self.device)

            if t % self.cfg.budget_reset_every == 0:
                yt.zero_()
                budget_lim = (gamma ** t) * self.cfg.budget_limit_base * torch.ones(batch, device=self.device)
                tau = 0

            if self.cfg.use_budget:
                yt_ratio = (yt / budget_lim).unsqueeze(1)
                budget_remaining = budget_lim - yt
            else:
                yt_ratio = None

            with torch.no_grad():
                action_logits, values = self.policy(
                    belief.IRI, belief.CCI, belief.deck,
                    extra_state, ages.float(), time_ratio, yt_ratio
                )
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)  # (batch, 96)
                log_prob_sum = log_probs.sum(dim=1)  # (batch,)

            if self.cfg.use_budget:
                can_repair = torch.ones(batch, TOT_COMP, dtype=torch.bool, device=self.device)
                comp_active, action_real = self.budget_handler.apply(
                    actions, budget_remaining, can_repair, temperature=1.0
                )
            else:
                comp_active = torch.ones(batch, TOT_COMP, dtype=torch.bool, device=self.device)
                action_real = actions

            ages_prime = self.env.age_action(ages, action_real, comp_active)
            b_prime, b_a, extra_state_prime, _ = self.env.belief_update_vectorized(
                belief, action_real, ages_prime, extra_state, comp_active
            )

            action_cost, obs_cost, _ = self.cost_module.compute_action_observation_costs(
                action_real, comp_active
            )
            delay_cost = self.cost_module.compute_delay_cost(action_real, comp_active)
            risk_cost = self.cost_module.compute_risk_cost(
                b_a.deck, ages_prime, action_real, comp_active, self.env.pcomp_deck
            )

            total_cost = action_cost + obs_cost + delay_cost + risk_cost

            discount = gamma ** t
            ep_cost += discount * total_cost
            ep_risk += discount * risk_cost
            ep_action_obs += discount * (action_cost + obs_cost)
            ep_delay += discount * delay_cost

            if self.cfg.use_budget:
                yt += -(gamma ** tau) * (action_cost + obs_cost)
                tau += 1

            ages_prime = ages_prime + 1
            ages_prime = torch.clamp(ages_prime, max=19)

            done = torch.zeros(batch, dtype=torch.bool, device=self.device)
            if t == ep_len - 1:
                done[:] = True

            constraints = self.cost_module.compute_constraints(
                belief.IRI, belief.CCI, belief.deck
            )
            ep_constraint_sum += constraints

            # Constraint penalty for GAE: lambda . (-constraints) matching off-policy sign convention
            constraint_penalty = torch.matmul(-constraints, self.lamda)  # (batch,)

            buf_yt_ratio = yt_ratio if yt_ratio is not None else torch.zeros(batch, 1, device=self.device)
            self.rollout_buffer.store(
                t, belief, extra_state, ages, time_ratio, buf_yt_ratio,
                actions, log_prob_sum, values.squeeze(-1),
                total_cost, constraint_penalty, done,
            )

            belief = b_prime
            ages = ages_prime
            extra_state = extra_state_prime
            self.global_steps += batch

        # Bootstrap last value
        if self.cfg.use_budget:
            terminal_yt_ratio = (yt / budget_lim).unsqueeze(1)
        else:
            terminal_yt_ratio = None
        with torch.no_grad():
            _, last_value = self.policy(
                belief.IRI, belief.CCI, belief.deck,
                extra_state, ages.float(),
                torch.full((batch, 1), 1.0, device=self.device),
                terminal_yt_ratio,
            )
            last_value = last_value.squeeze(-1)  # (batch,)

        self.rollout_buffer.compute_gae(last_value, gamma, self.cfg.gae_lambda)

        # Update Lagrangian multipliers
        avg_constr = (ep_constraint_sum / ep_len).mean(dim=0)
        self.lamda = torch.clamp(self.lamda + self.lr_lamda * (avg_constr - self.cap_alp), min=0)

        ep_cost_mean = ep_cost.mean().item()
        ep_risk_mean = ep_risk.mean().item()
        ep_action_obs_mean = ep_action_obs.mean().item()
        ep_delay_mean = ep_delay.mean().item()
        ep_constraint_mean = (ep_constraint_sum / ep_len).mean(dim=0).detach().cpu().numpy()
        return ep_cost_mean, ep_constraint_mean, ep_risk_mean, ep_action_obs_mean, ep_delay_mean

    def update(self, episode: int):
        """PPO clipped surrogate update over collected rollout."""
        self.update_calls += 1

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy_loss = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for epoch in range(self.cfg.ppo_epochs):
            for mb in self.rollout_buffer.get_minibatch_generator(self.cfg.num_minibatches):
                yt = mb['yt_ratio'] if self.cfg.use_budget else None
                # Re-evaluate actions under current policy
                log_probs, entropy = self.policy.actor.evaluate_actions(
                    mb['state_IRI'], mb['state_CCI'], mb['state_deck'],
                    mb['extra_state'], mb['age'].float(), mb['time_ratio'],
                    yt, mb['actions'],
                )
                new_log_prob_sum = log_probs.sum(dim=1)  # (mb_size,)

                # Also get new values
                _, new_values = self.policy(
                    mb['state_IRI'], mb['state_CCI'], mb['state_deck'],
                    mb['extra_state'], mb['age'].float(), mb['time_ratio'],
                    yt,
                )
                new_values = new_values.squeeze(-1)  # (mb_size,)

                # Ratio
                log_ratio = new_log_prob_sum - mb['log_prob_sum']
                ratio = torch.exp(log_ratio)

                # Advantages
                adv = mb['advantages']
                if self.cfg.normalize_advantages and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped surrogate loss (cost minimization: negate to do gradient ascent on negative cost)
                pg_loss1 = ratio * adv
                pg_loss2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * adv
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_clipped = mb['values'] + torch.clamp(
                    new_values - mb['values'],
                    -self.cfg.clip_epsilon, self.cfg.clip_epsilon,
                )
                v_loss1 = F.mse_loss(new_values, mb['returns'], reduction='none')
                v_loss2 = F.mse_loss(v_clipped, mb['returns'], reduction='none')
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.cfg.value_coef * v_loss + self.cfg.entropy_coef * entropy_loss

                if torch.isnan(loss):
                    print("WARNING: NaN loss in PPO update, skipping")
                    continue

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                # Metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean().item()
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_clip_frac += clip_frac
                total_approx_kl += approx_kl
                n_updates += 1

            # Early stopping on KL
            if self.cfg.target_kl is not None and n_updates > 0:
                if total_approx_kl / n_updates > self.cfg.target_kl:
                    break

        # LR scheduling
        step_every = max(1, int(self.cfg.lr_schedule_step_every))
        if (self.actor_scheduler is not None
                and self.update_calls % step_every == 0
                and self.cfg.lr_schedule_start_episode < episode < self.cfg.lr_schedule_end_episode):
            self.actor_scheduler.step()
        if (self.critic_scheduler is not None
                and self.update_calls % step_every == 0
                and self.cfg.lr_schedule_start_episode < episode < self.cfg.lr_schedule_end_episode):
            self.critic_scheduler.step()

        if n_updates > 0:
            return {
                'pg_loss': total_pg_loss / n_updates,
                'v_loss': total_v_loss / n_updates,
                'entropy_loss': total_entropy_loss / n_updates,
                'clip_frac': total_clip_frac / n_updates,
                'approx_kl': total_approx_kl / n_updates,
            }
        return {}

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
        checkpoint = {
            'episode': episode,
            'global_steps': self.global_steps,
            'episodes_done': self.episodes_done,
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict() if self.actor_scheduler else None,
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict() if self.critic_scheduler else None,
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

        latest_path = os.path.join(self.checkpoint_dir_abs, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint to {latest_path} (episode {episode})")

        if episode % self.cfg.checkpoint_every == 0:
            periodic_path = os.path.join(self.checkpoint_dir_abs, f'checkpoint_ep{episode}.pt')
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint to {periodic_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir_abs, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved BEST checkpoint to {best_path} (avg_cost={self.best_avg_cost:.3f})")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[list, list, list, list, list, int]:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        actor_sched_state = checkpoint.get('actor_scheduler_state_dict')
        critic_sched_state = checkpoint.get('critic_scheduler_state_dict')
        if self.actor_scheduler is not None and actor_sched_state is not None:
            self.actor_scheduler.load_state_dict(actor_sched_state)
        if self.critic_scheduler is not None and critic_sched_state is not None:
            self.critic_scheduler.load_state_dict(critic_sched_state)

        self.global_steps = checkpoint['global_steps']
        self.episodes_done = checkpoint['episodes_done']
        self.lamda = checkpoint['lamda'].to(self.device)
        self.best_avg_cost = checkpoint['best_avg_cost']
        self.update_calls = int(checkpoint.get('update_calls', 0))

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
        start = time.time()
        ep_cost_log = []
        ep_constr_log = []
        ep_risk_log = []
        ep_action_obs_log = []
        ep_delay_log = []
        start_episode = 0

        self._maybe_init_wandb()

        if resume_from:
            if os.path.exists(resume_from):
                (
                    ep_cost_log, ep_constr_log, ep_risk_log,
                    ep_action_obs_log, ep_delay_log, start_episode,
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

            # On-policy: update every episode
            ppo_metrics = self.update(episode=ep)

            if self._wandb_run is not None and (ep + 1) % self.cfg.wandb_log_every == 0:
                k = self.cfg.wandb_log_every
                if len(ep_cost_log) >= k:
                    lambda_norm = torch.norm(self.lamda).item()
                    log_dict = {
                        "Total_cost": float(np.mean(ep_cost_log[-k:]) * 1000.0),
                        "Episodic Risk Cost": float(np.mean(ep_risk_log[-k:]) * 1000.0),
                        "Episodic_Cost_ac_ob": float(np.mean(ep_action_obs_log[-k:]) * 1000.0),
                        "Delay_Cost": float(np.mean(ep_delay_log[-k:]) * 1000.0),
                        "Lambda": float(lambda_norm),
                        "episode": float((ep + 1) / k),
                        "episode_raw": int(ep + 1),
                        "global_steps": int(self.global_steps),
                        "actor_lr": float(self.actor_opt.param_groups[0]["lr"]),
                        "critic_lr": float(self.critic_opt.param_groups[0]["lr"]),
                    }
                    if ppo_metrics:
                        log_dict.update({
                            "ppo/pg_loss": ppo_metrics.get('pg_loss', 0),
                            "ppo/v_loss": ppo_metrics.get('v_loss', 0),
                            "ppo/entropy_loss": ppo_metrics.get('entropy_loss', 0),
                            "ppo/clip_frac": ppo_metrics.get('clip_frac', 0),
                            "ppo/approx_kl": ppo_metrics.get('approx_kl', 0),
                        })
                    wandb.log(log_dict, step=int(ep + 1))

            if (ep + 1) % 100 == 0:
                elapsed = time.time() - start
                costs_100 = ep_cost_log[-100:]
                avg_cost = np.mean(costs_100) * 1000
                std_cost = np.std(costs_100) * 1000
                min_cost = np.min(costs_100) * 1000
                max_cost = np.max(costs_100) * 1000
                last_constr = ep_constr_log[-1]
                print(f"time elapsed {elapsed:.1f}s")
                print(f"Average_costs {avg_cost:.3f} (std={std_cost:.1f}, min={min_cost:.1f}, max={max_cost:.1f})")
                print(f"Constraints: {last_constr}")
                print(f"Lambda: {self.lamda.cpu().numpy()}")
                if ppo_metrics:
                    print(f"PPO clip_frac={ppo_metrics.get('clip_frac',0):.3f} approx_kl={ppo_metrics.get('approx_kl',0):.5f}")

                is_best = False
                if self.cfg.save_best and avg_cost < self.best_avg_cost:
                    self.best_avg_cost = avg_cost
                    is_best = True

                self.save_checkpoint(
                    ep + 1, ep_cost_log, ep_constr_log,
                    ep_risk_log, ep_action_obs_log, ep_delay_log,
                    is_best=is_best,
                )
                start = time.time()

            if (ep + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"[PPO] Episode {ep+1} | steps={self.global_steps} | time={elapsed:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO DDMAC-CTDE Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb-resume', action='store_true', help='Resume W&B run')
    parser.add_argument('--wandb-log-every', type=int, default=None, help='Log to W&B every N episodes')
    parser.add_argument('--device', type=str, default=None, help='Device (e.g. cuda:0, cpu)')
    parser.add_argument('--lr-schedule', type=str, default=None, choices=['none', 'cosine', 'linear'],
                        help='LR schedule')
    parser.add_argument('--total-episodes', type=int, default=None, help='Total training episodes')
    parser.add_argument('--clip-epsilon', type=float, default=None, help='PPO clip epsilon')
    parser.add_argument('--ppo-epochs', type=int, default=None, help='PPO epochs per update')
    parser.add_argument('--entropy-coef', type=float, default=None, help='Entropy coefficient')
    parser.add_argument('--gae-lambda', type=float, default=None, help='GAE lambda')
    parser.add_argument('--no-budget', action='store_true',
                        help='Disable budget constraints (infinite budget, no yt_ratio input)')
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = PPOConfig(device=device)

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
    if args.total_episodes is not None:
        cfg.total_episodes = args.total_episodes
    if args.clip_epsilon is not None:
        cfg.clip_epsilon = args.clip_epsilon
    if args.ppo_epochs is not None:
        cfg.ppo_epochs = args.ppo_epochs
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef
    if args.gae_lambda is not None:
        cfg.gae_lambda = args.gae_lambda
    if args.no_budget:
        cfg.use_budget = False

    trainer = PPOTrainer(cfg)
    trainer.train(resume_from=args.resume)
