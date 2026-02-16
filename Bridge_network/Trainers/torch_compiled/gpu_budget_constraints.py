"""
GPU-Optimized Budget Constraint Strategies
============================================

Multiple approaches to handle budget constraints efficiently on GPU:

1. GREEDY_BY_COST: Sort by cost, select cheapest first (deterministic)
2. GREEDY_BY_PRIORITY: Sort by priority (e.g., urgency), select highest priority first
3. STOCHASTIC_PARALLEL: Parallel random selection using pre-generated priorities
4. SOFT_CONSTRAINT: No hard cutoff, add penalty to reward (most GPU-efficient)
5. DIFFERENTIABLE_SELECTION: Gumbel-softmax based selection (for end-to-end training)

All methods are fully vectorized for batch processing on GPU.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from enum import Enum


# Configuration
NCOMP_PAV = 85
NCOMP_DECK = 11
TOT_COMP = 96
NACOMP = 10


class BudgetStrategy(Enum):
    GREEDY_BY_COST = "greedy_by_cost"
    GREEDY_BY_PRIORITY = "greedy_by_priority"
    STOCHASTIC_PARALLEL = "stochastic_parallel"
    SOFT_CONSTRAINT = "soft_constraint"
    DIFFERENTIABLE = "differentiable"


class GPUBudgetConstraint:
    """
    GPU-optimized budget constraint handler.
    
    Supports multiple strategies with different trade-offs between:
    - Computational efficiency
    - Stochasticity (exploration)
    - Differentiability (for end-to-end training)
    """
    
    def __init__(
        self,
        action_costs: torch.Tensor,    # (tot_comp, nacomp)
        obs_costs: torch.Tensor,       # (tot_comp, nacomp)
        gamma: float = 0.97,
        device: str = 'cuda:0',
        strategy: BudgetStrategy = BudgetStrategy.GREEDY_BY_COST,
    ):
        self.action_costs = action_costs
        self.obs_costs = obs_costs
        self.gamma = gamma
        self.device = device
        self.strategy = strategy
        
        # Precompute total costs for all action combinations
        self._precompute_costs()
    
    def _precompute_costs(self):
        """Precompute cost lookup tables for efficiency"""
        # Total cost per (component, action) pair
        self.total_costs = -(self.action_costs + self.gamma * self.obs_costs)  # (96, 10) positive values
    
    def get_component_costs(
        self,
        actions: torch.Tensor,  # (batch, tot_comp)
    ) -> torch.Tensor:
        """Get cost for each component based on selected action"""
        batch = actions.shape[0]
        return torch.gather(
            self.total_costs.unsqueeze(0).expand(batch, -1, -1),
            2,
            actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch, tot_comp)
    
    # ========================================================================
    # STRATEGY 1: GREEDY BY COST (Deterministic, Fast)
    # ========================================================================
    
    def apply_greedy_by_cost(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        can_repair: torch.Tensor,        # (batch, tot_comp) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select components greedily by cost (cheapest first).
        
        Pros: Deterministic, very fast, fully vectorized
        Cons: No exploration, always selects same components
        
        Time complexity: O(batch * tot_comp * log(tot_comp)) for sort
        """
        batch = actions.shape[0]
        device = actions.device
        
        # Get per-component costs
        costs = self.get_component_costs(actions)  # (batch, tot_comp)
        
        # Set cost to infinity for components that can't be repaired
        costs = torch.where(can_repair, costs, torch.tensor(float('inf'), device=device))
        
        # Sort by cost (ascending - cheapest first)
        sorted_costs, sorted_indices = costs.sort(dim=1)
        
        # Cumulative sum
        cumsum = sorted_costs.cumsum(dim=1)
        
        # Find cutoff
        within_budget = cumsum <= budget_remaining.unsqueeze(1)
        
        # Scatter back to original indices
        comp_active = torch.zeros(batch, TOT_COMP, dtype=torch.bool, device=device)
        comp_active.scatter_(1, sorted_indices, within_budget)
        
        # Apply can_repair constraint (redundant but safe)
        comp_active = comp_active & can_repair
        
        return comp_active, actions * comp_active.long()
    
    # ========================================================================
    # STRATEGY 2: GREEDY BY PRIORITY (Policy-Guided)
    # ========================================================================
    
    def apply_greedy_by_priority(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        can_repair: torch.Tensor,        # (batch, tot_comp) bool
        priorities: torch.Tensor,        # (batch, tot_comp) - higher = more important
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select components by priority (highest priority first) until budget exhausted.
        
        Priorities can come from:
        - Policy network output (learned priorities)
        - State urgency (e.g., components in worst condition)
        - Cost-benefit ratio
        
        Pros: Can incorporate domain knowledge, policy can learn priorities
        Cons: Still deterministic given priorities
        """
        batch = actions.shape[0]
        device = actions.device
        
        # Get per-component costs
        costs = self.get_component_costs(actions)  # (batch, tot_comp)
        
        # Set priority to -inf for components that can't be repaired
        priorities = torch.where(can_repair, priorities, torch.tensor(float('-inf'), device=device))
        
        # Sort by priority (descending - highest priority first)
        sorted_priorities, sorted_indices = priorities.sort(dim=1, descending=True)
        
        # Gather costs in priority order
        sorted_costs = torch.gather(costs, 1, sorted_indices)
        
        # Cumulative sum of costs in priority order
        cumsum = sorted_costs.cumsum(dim=1)
        
        # Find cutoff
        within_budget = cumsum <= budget_remaining.unsqueeze(1)
        
        # Scatter back to original indices
        comp_active = torch.zeros(batch, TOT_COMP, dtype=torch.bool, device=device)
        comp_active.scatter_(1, sorted_indices, within_budget)
        
        comp_active = comp_active & can_repair
        
        return comp_active, actions * comp_active.long()
    
    # ========================================================================
    # STRATEGY 3: STOCHASTIC PARALLEL (Random, Vectorized)
    # ========================================================================
    
    def apply_stochastic_parallel(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        can_repair: torch.Tensor,        # (batch, tot_comp) bool
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic selection using random priorities (matches legacy behavior statistically).
        
        Each sample in batch gets different random ordering.
        
        Pros: Exploration, matches legacy statistical distribution
        Cons: Non-deterministic, slightly slower than greedy
        """
        batch = actions.shape[0]
        device = actions.device
        
        # Generate random priorities for this batch
        # Using Gumbel noise for proper random permutation
        noise = -torch.empty(batch, TOT_COMP, device=device).exponential_().log()
        random_priorities = noise / temperature
        
        # Use priority-based selection with random priorities
        return self.apply_greedy_by_priority(
            actions, budget_remaining, can_repair, random_priorities
        )
    
    # ========================================================================
    # STRATEGY 4: SOFT CONSTRAINT (Penalty-Based, Most Efficient)
    # ========================================================================
    
    def compute_soft_constraint_penalty(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        comp_active: torch.Tensor,       # (batch, tot_comp) bool - all active
        penalty_scale: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute penalty for budget violation instead of hard constraint.
        
        The policy learns to stay within budget to avoid penalty.
        
        Pros: Most GPU-efficient (no sorting), fully differentiable
        Cons: Soft constraint may not enforce exact budget
        
        Returns:
            penalty: (batch,) - penalty to add to cost/subtract from reward
        """
        # Total cost if all components are active
        costs = self.get_component_costs(actions)  # (batch, tot_comp)
        costs = costs * comp_active.float()
        total_cost = costs.sum(dim=1)  # (batch,)
        
        # Compute violation (positive if over budget)
        violation = torch.relu(total_cost - budget_remaining)
        
        # Quadratic penalty for smooth gradients
        penalty = penalty_scale * violation ** 2
        
        return penalty
    
    def apply_soft_constraint(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        can_repair: torch.Tensor,        # (batch, tot_comp) bool
        penalty_scale: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft constraint: all valid components active, return penalty.
        
        Returns:
            comp_active: (batch, tot_comp) - all valid components
            action_real: (batch, tot_comp) - actions for active components
            penalty: (batch,) - budget violation penalty
        """
        comp_active = can_repair.clone()
        action_real = actions * comp_active.long()
        penalty = self.compute_soft_constraint_penalty(
            actions, budget_remaining, comp_active, penalty_scale
        )
        return comp_active, action_real, penalty
    
    # ========================================================================
    # STRATEGY 5: DIFFERENTIABLE SELECTION (End-to-End Training)
    # ========================================================================
    
    def apply_differentiable_selection(
        self,
        actions: torch.Tensor,           # (batch, tot_comp)
        budget_remaining: torch.Tensor,  # (batch,)
        can_repair: torch.Tensor,        # (batch, tot_comp) bool
        logits: torch.Tensor,            # (batch, tot_comp) - selection logits from policy
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable component selection using Gumbel-Softmax.
        
        During training: soft selection for gradients
        During inference: hard selection (straight-through estimator)
        
        This allows end-to-end training where the policy learns which
        components to prioritize within budget.
        
        Pros: Fully differentiable, policy learns selection
        Cons: More complex, requires policy to output selection logits
        """
        batch = actions.shape[0]
        device = actions.device
        
        # Mask invalid components
        logits = torch.where(
            can_repair,
            logits,
            torch.tensor(float('-inf'), device=device)
        )
        
        # Gumbel-Softmax for differentiable sampling
        # This gives soft probabilities during training
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        gumbel_noise = -(-gumbel_noise).exponential_().log()
        
        soft_selection = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator: hard in forward, soft in backward
            hard_selection = (soft_selection == soft_selection.max(dim=-1, keepdim=True)[0]).float()
            selection = hard_selection - soft_selection.detach() + soft_selection
        else:
            selection = soft_selection
        
        # Now we need to select components within budget
        # Sort by selection probability, then apply budget cutoff
        costs = self.get_component_costs(actions)
        
        sorted_selection, sorted_indices = selection.sort(dim=1, descending=True)
        sorted_costs = torch.gather(costs, 1, sorted_indices)
        
        cumsum = sorted_costs.cumsum(dim=1)
        within_budget = (cumsum <= budget_remaining.unsqueeze(1)).float()
        
        # Weight selection by budget feasibility
        weighted_selection = sorted_selection * within_budget
        
        # Scatter back
        comp_active_soft = torch.zeros(batch, TOT_COMP, device=device)
        comp_active_soft.scatter_(1, sorted_indices, weighted_selection)
        
        # For hard selection
        comp_active = (comp_active_soft > 0.5) & can_repair
        
        return comp_active, actions * comp_active.long()
    
    # ========================================================================
    # UNIFIED INTERFACE
    # ========================================================================
    
    def apply(
        self,
        actions: torch.Tensor,
        budget_remaining: torch.Tensor,
        can_repair: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply budget constraint using configured strategy"""
        
        if self.strategy == BudgetStrategy.GREEDY_BY_COST:
            return self.apply_greedy_by_cost(actions, budget_remaining, can_repair)
        
        elif self.strategy == BudgetStrategy.GREEDY_BY_PRIORITY:
            priorities = kwargs.get('priorities')
            if priorities is None:
                raise ValueError("priorities required for GREEDY_BY_PRIORITY")
            return self.apply_greedy_by_priority(
                actions, budget_remaining, can_repair, priorities
            )
        
        elif self.strategy == BudgetStrategy.STOCHASTIC_PARALLEL:
            temperature = kwargs.get('temperature', 1.0)
            return self.apply_stochastic_parallel(
                actions, budget_remaining, can_repair, temperature
            )
        
        elif self.strategy == BudgetStrategy.SOFT_CONSTRAINT:
            penalty_scale = kwargs.get('penalty_scale', 10.0)
            comp_active, action_real, penalty = self.apply_soft_constraint(
                actions, budget_remaining, can_repair, penalty_scale
            )
            # Store penalty for later retrieval
            self._last_penalty = penalty
            return comp_active, action_real
        
        elif self.strategy == BudgetStrategy.DIFFERENTIABLE:
            logits = kwargs.get('logits')
            if logits is None:
                raise ValueError("logits required for DIFFERENTIABLE")
            return self.apply_differentiable_selection(
                actions, budget_remaining, can_repair, logits,
                kwargs.get('temperature', 1.0),
                kwargs.get('hard', True)
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# ============================================================================
# COMPARISON TEST
# ============================================================================

def benchmark_strategies():
    """Benchmark different budget constraint strategies"""
    import time
    
    print("=" * 70)
    print("BUDGET CONSTRAINT STRATEGY BENCHMARK")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1024
    n_iterations = 100
    
    # Create mock cost tensors
    action_costs = -torch.rand(TOT_COMP, NACOMP, device=device) * 20
    obs_costs = -torch.rand(TOT_COMP, NACOMP, device=device) * 0.1
    
    # Create test inputs
    actions = torch.randint(0, NACOMP, (batch_size, TOT_COMP), device=device)
    budget = torch.ones(batch_size, device=device) * 500  # Budget of 500
    can_repair = torch.ones(batch_size, TOT_COMP, dtype=torch.bool, device=device)
    
    strategies = [
        (BudgetStrategy.GREEDY_BY_COST, {}),
        (BudgetStrategy.STOCHASTIC_PARALLEL, {'temperature': 1.0}),
        (BudgetStrategy.SOFT_CONSTRAINT, {'penalty_scale': 10.0}),
    ]
    
    print(f"\nBatch size: {batch_size}, Iterations: {n_iterations}")
    print(f"Device: {device}\n")
    
    results = {}
    
    for strategy, kwargs in strategies:
        constraint = GPUBudgetConstraint(
            action_costs, obs_costs, device=device, strategy=strategy
        )
        
        # Warmup
        for _ in range(10):
            comp_active, action_real = constraint.apply(
                actions, budget, can_repair, **kwargs
            )
        
        torch.cuda.synchronize() if device.startswith('cuda') else None
        
        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            comp_active, action_real = constraint.apply(
                actions, budget, can_repair, **kwargs
            )
        torch.cuda.synchronize() if device.startswith('cuda') else None
        elapsed = time.time() - start
        
        # Stats
        active_count = comp_active.float().sum(dim=1).mean().item()
        
        results[strategy.value] = {
            'time_per_batch_ms': elapsed / n_iterations * 1000,
            'throughput': batch_size * n_iterations / elapsed,
            'avg_active_components': active_count,
        }
        
        print(f"Strategy: {strategy.value}")
        print(f"  Time per batch: {results[strategy.value]['time_per_batch_ms']:.3f} ms")
        print(f"  Throughput: {results[strategy.value]['throughput']:.0f} samples/sec")
        print(f"  Avg active components: {active_count:.1f} / {TOT_COMP}")
        print()
    
    return results


def compare_with_legacy():
    """Compare GPU strategies with actual legacy budget constraint implementation"""
    import time
    import scipy.io as spio
    import os
    from dataclasses import dataclass
    
    print("=" * 70)
    print("GPU vs ACTUAL LEGACY BUDGET CONSTRAINT COMPARISON")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load actual legacy cost data
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dependency_files')
    os.chdir(DATA_DIR)
    
    # Load link lengths and compute costs (same as legacy)
    mat1 = spio.loadmat('link_lengths.mat', squeeze_me=True)
    lenth_comp = mat1['length1'][:NCOMP_PAV]
    
    # Number of lanes
    n_lane_pav = np.zeros(NCOMP_PAV)
    for i in range(NCOMP_PAV):
        if i < 12:  # interstate
            n_lane_pav[i] = 8
        elif i < 12 + 47:  # primary
            n_lane_pav[i] = 4
        else:  # secondary
            n_lane_pav[i] = 2
    
    # Bridge properties
    brdg_len = np.array([7.07, 7.4, 5.6, 1.140, 1.13, 1.5, 0.647, 0.64, 1.01, 0.32, 0.90]) / 1.61
    n_lane = np.array([4, 4, 4, 4, 2, 4, 8, 4, 4, 4, 4])
    lane_wdth = np.array([5, 4, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7])
    Area_deck = brdg_len * n_lane * lane_wdth * 1.61 * 1000
    
    # Cost coefficients
    A_pav = 3.7 * np.array([[0, 20, 75, 350], [0, 16, 68, 330], [0, 10, 52, 250]])
    A_deck = np.array([0, 400, 1200, 2650.00])
    I_pav = 3.7 * np.array([0, 0.10, 0.20])
    I_deck = np.array([0, 0.48, 1.20])
    gamma = 0.97
    
    # Build cost matrices (same as Network_data.py)
    cost_comp_action = np.zeros((TOT_COMP, NACOMP))
    cost_comp_obsr = np.zeros((TOT_COMP, NACOMP))
    
    for i in range(NCOMP_PAV):
        for j in range(NACOMP):
            if i < 12:
                type_idx, n_lanes = 0, 8
            elif i < 12 + 47:
                type_idx, n_lanes = 1, 4
            else:
                type_idx, n_lanes = 2, 2
            
            length_m = lenth_comp[i] * 1.61 * 1000
            maint_type = j % 3
            insp_type = j // 3
            
            if j == 9:
                cost_comp_action[i, j] = -length_m * A_pav[type_idx, 3] * n_lanes / 1e6
            else:
                cost_comp_action[i, j] = -length_m * A_pav[type_idx, maint_type] * n_lanes / 1e6
                if insp_type == 1:
                    cost_comp_obsr[i, j] = -length_m * I_pav[1] * n_lanes / 1e6
                elif insp_type == 2:
                    cost_comp_obsr[i, j] = -length_m * I_pav[2] * n_lanes / 1e6
    
    for i in range(NCOMP_DECK):
        area = Area_deck[i]
        for j in range(NACOMP):
            maint_type = j % 3
            insp_type = j // 3
            if j == 9:
                cost_comp_action[NCOMP_PAV + i, j] = -area * A_deck[3] / 1e6
            else:
                cost_comp_action[NCOMP_PAV + i, j] = -area * A_deck[maint_type] / 1e6
                if insp_type == 1:
                    cost_comp_obsr[NCOMP_PAV + i, j] = -area * I_deck[1] / 1e6
                elif insp_type == 2:
                    cost_comp_obsr[NCOMP_PAV + i, j] = -area * I_deck[2] / 1e6
    
    # Load delay costs
    Dlay_cost = np.load('Dlay_Cost.npy')
    intdx = list(range(NCOMP_PAV)) + list(range(NCOMP_PAV, NCOMP_PAV + 11))
    Dlay_cost = Dlay_cost[intdx, :]
    
    # Convert to GPU tensors
    action_costs_gpu = torch.tensor(cost_comp_action, dtype=torch.float32, device=device)
    obs_costs_gpu = torch.tensor(cost_comp_obsr, dtype=torch.float32, device=device)
    delay_costs_gpu = torch.tensor(Dlay_cost, dtype=torch.float32, device=device)
    
    # ========================================================================
    # LEGACY BUDGET CONSTRAINT FUNCTION (exact copy from DRL code)
    # ========================================================================
    
    def legacy_budget_constraint(action, extra_state, budget_lim, y=0):
        """
        Legacy budget constraint - exact implementation from DRL code
        """
        cost_action = 0
        cost_obs = 0
        cost_delay = 0
        
        if budget_lim - y < 0:
            comp_active = []
        else:
            # Find components that cannot be repaired
            indxx = list(np.where(extra_state[:NCOMP_PAV] == 1)[0])
            ind_xr = list(set(range(NCOMP_PAV)) - set(indxx))
            
            comp_active_repair = ind_xr[:]
            cost_action = sum(cost_comp_action[ind_xr, action[0, ind_xr]])
            cost_obs = gamma * sum(cost_comp_obsr[ind_xr, action[0, ind_xr]])
            
            # Add deck components
            for j in range(NCOMP_DECK):
                comp_active_repair.append(j + NCOMP_PAV)
                cost_action += cost_comp_action[j + NCOMP_PAV, action[0, j + NCOMP_PAV]]
                cost_obs += gamma * cost_comp_obsr[j + NCOMP_PAV, action[0, j + NCOMP_PAV]]
            
            cost_ac_ob2 = -cost_action - cost_obs
            comp_active = comp_active_repair[:]
            
            # Apply budget constraint with random ordering
            if cost_ac_ob2 > budget_lim - y:
                comp_order = np.random.permutation(comp_active_repair)
                cost_ac_ob1 = 0
                comp_active = []
                
                for j in comp_order:
                    cost_ac_ob1 -= cost_comp_action[j, action[0, j]]
                    if action[0, j] in [3, 4, 5, 6, 7, 8]:
                        cost_ac_ob1 -= gamma * cost_comp_obsr[j, action[0, j]]
                    
                    if cost_ac_ob1 <= budget_lim - y:
                        comp_active.append(j)
                    else:
                        cost_ac_ob1 += cost_comp_action[j, action[0, j]]
                        if action[0, j] in [3, 4, 5, 6, 7, 8]:
                            cost_ac_ob1 += gamma * cost_comp_obsr[j, action[0, j]]
        
        # Calculate final costs
        cost_obs = gamma * sum(cost_comp_obsr[comp_active, action[0, comp_active]])
        cost_action = sum(cost_comp_action[comp_active, action[0, comp_active]])
        cost_delay = sum(Dlay_cost[comp_active, action[0, comp_active]])
        
        action_real = np.zeros((1, TOT_COMP), dtype=int)
        action_real[0, comp_active] = action[0, comp_active]
        
        return cost_action, cost_obs, cost_delay, comp_active, action_real
    
    # ========================================================================
    # TEST COMPARISON
    # ========================================================================
    
    print(f"\nDevice: {device}")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test parameters
    n_samples = 20000
    budget_lim = 200.0
    
    # Generate random actions
    actions_np = np.random.randint(0, NACOMP, (n_samples, TOT_COMP))
    actions_gpu = torch.tensor(actions_np, dtype=torch.long, device=device)
    extra_state = np.zeros(NCOMP_PAV)
    
    # Create GPU budget constraint
    constraint_stochastic = GPUBudgetConstraint(
        action_costs_gpu, obs_costs_gpu, device=device,
        strategy=BudgetStrategy.STOCHASTIC_PARALLEL
    )
    constraint_greedy = GPUBudgetConstraint(
        action_costs_gpu, obs_costs_gpu, device=device,
        strategy=BudgetStrategy.GREEDY_BY_COST
    )
    
    budget_gpu = torch.ones(n_samples, device=device) * budget_lim
    can_repair = torch.ones(n_samples, TOT_COMP, dtype=torch.bool, device=device)
    
    # ========================================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================================
    
    print(f"\n--- Side-by-Side Budget Constraint Comparison (budget={budget_lim}) ---")
    print(f"{'Sample':<8} {'Legacy Active':<14} {'GPU Stoch':<12} {'GPU Greedy':<12} {'Legacy Cost':<14} {'GPU Stoch Cost':<16} {'GPU Greedy Cost':<16}")
    print("-" * 110)
    
    legacy_costs = []
    gpu_stoch_costs = []
    gpu_greedy_costs = []
    legacy_actives = []
    gpu_stoch_actives = []
    gpu_greedy_actives = []
    
    for i in range(n_samples):
        action_single = actions_np[i:i+1]
        
        # Legacy
        leg_action, leg_obs, leg_delay, leg_active, _ = legacy_budget_constraint(
            action_single, extra_state, budget_lim
        )
        leg_total = leg_action + leg_obs + leg_delay
        
        # GPU Stochastic
        comp_active_stoch, action_real_stoch = constraint_stochastic.apply(
            actions_gpu[i:i+1], budget_gpu[i:i+1], can_repair[i:i+1], temperature=1.0
        )
        
        # GPU Greedy
        comp_active_greedy, action_real_greedy = constraint_greedy.apply(
            actions_gpu[i:i+1], budget_gpu[i:i+1], can_repair[i:i+1]
        )
        
        # Calculate GPU costs
        def calc_gpu_cost(comp_active, actions):
            active_mask = comp_active.float()
            action_cost = torch.gather(action_costs_gpu.unsqueeze(0), 2, actions.unsqueeze(-1)).squeeze(-1)
            obs_cost = torch.gather(obs_costs_gpu.unsqueeze(0), 2, actions.unsqueeze(-1)).squeeze(-1) * gamma
            delay_cost = torch.gather(delay_costs_gpu.unsqueeze(0), 2, actions.unsqueeze(-1)).squeeze(-1)
            total = ((action_cost + obs_cost + delay_cost) * active_mask).sum().item()
            return total
        
        gpu_stoch_total = calc_gpu_cost(comp_active_stoch, actions_gpu[i:i+1])
        gpu_greedy_total = calc_gpu_cost(comp_active_greedy, actions_gpu[i:i+1])
        
        n_active_legacy = len(leg_active)
        n_active_stoch = comp_active_stoch.sum().item()
        n_active_greedy = comp_active_greedy.sum().item()
        
        legacy_costs.append(leg_total)
        gpu_stoch_costs.append(gpu_stoch_total)
        gpu_greedy_costs.append(gpu_greedy_total)
        legacy_actives.append(n_active_legacy)
        gpu_stoch_actives.append(n_active_stoch)
        gpu_greedy_actives.append(n_active_greedy)
        
        print(f"{i:<8} {n_active_legacy:<14} {n_active_stoch:<12.0f} {n_active_greedy:<12.0f} {leg_total:<14.4f} {gpu_stoch_total:<16.4f} {gpu_greedy_total:<16.4f}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print(f"\n--- Summary Statistics ---")
    print(f"{'Metric':<25} {'Legacy':<15} {'GPU Stochastic':<18} {'GPU Greedy':<15}")
    print("-" * 75)
    print(f"{'Avg Active Components':<25} {np.mean(legacy_actives):<15.2f} {np.mean(gpu_stoch_actives):<18.2f} {np.mean(gpu_greedy_actives):<15.2f}")
    print(f"{'Std Active Components':<25} {np.std(legacy_actives):<15.2f} {np.std(gpu_stoch_actives):<18.2f} {np.std(gpu_greedy_actives):<15.2f}")
    print(f"{'Avg Total Cost':<25} {np.mean(legacy_costs):<15.4f} {np.mean(gpu_stoch_costs):<18.4f} {np.mean(gpu_greedy_costs):<15.4f}")
    print(f"{'Std Total Cost':<25} {np.std(legacy_costs):<15.4f} {np.std(gpu_stoch_costs):<18.4f} {np.std(gpu_greedy_costs):<15.4f}")
    print(f"{'Min Total Cost':<25} {np.min(legacy_costs):<15.4f} {np.min(gpu_stoch_costs):<18.4f} {np.min(gpu_greedy_costs):<15.4f}")
    print(f"{'Max Total Cost':<25} {np.max(legacy_costs):<15.4f} {np.max(gpu_stoch_costs):<18.4f} {np.max(gpu_greedy_costs):<15.4f}")
    
    # ========================================================================
    # BUDGET COMPLIANCE CHECK
    # ========================================================================
    
    print(f"\n--- Budget Compliance Check ---")
    
    def check_budget_compliance(actions_np, comp_active_list, name):
        violations = 0
        for i, (action, active) in enumerate(zip(actions_np, comp_active_list)):
            if isinstance(active, (list, np.ndarray)):
                active_mask = np.zeros(TOT_COMP)
                active_mask[active] = 1
            else:
                active_mask = active.cpu().numpy().flatten()
            
            action_cost = sum(cost_comp_action[j, action[j]] for j in range(TOT_COMP) if active_mask[j] > 0.5)
            obs_cost = gamma * sum(cost_comp_obsr[j, action[j]] for j in range(TOT_COMP) if active_mask[j] > 0.5 and action[j] in [3,4,5,6,7,8])
            total = -(action_cost + obs_cost)
            if total > budget_lim + 0.01:  # Small tolerance
                violations += 1
        return violations
    
    # Collect comp_active for each method
    legacy_comp_actives = []
    gpu_stoch_comp_actives = []
    gpu_greedy_comp_actives = []
    
    for i in range(n_samples):
        action_single = actions_np[i:i+1]
        _, _, _, leg_active, _ = legacy_budget_constraint(action_single, extra_state, budget_lim)
        legacy_comp_actives.append(leg_active)
        
        comp_active_stoch, _ = constraint_stochastic.apply(
            actions_gpu[i:i+1], budget_gpu[i:i+1], can_repair[i:i+1], temperature=1.0
        )
        gpu_stoch_comp_actives.append(comp_active_stoch)
        
        comp_active_greedy, _ = constraint_greedy.apply(
            actions_gpu[i:i+1], budget_gpu[i:i+1], can_repair[i:i+1]
        )
        gpu_greedy_comp_actives.append(comp_active_greedy)
    
    print(f"Legacy violations: {check_budget_compliance(actions_np, legacy_comp_actives, 'Legacy')}/{n_samples}")
    print(f"GPU Stochastic violations: {check_budget_compliance(actions_np, gpu_stoch_comp_actives, 'GPU Stoch')}/{n_samples}")
    print(f"GPU Greedy violations: {check_budget_compliance(actions_np, gpu_greedy_comp_actives, 'GPU Greedy')}/{n_samples}")
    
    # ========================================================================
    # PERFORMANCE COMPARISON
    # ========================================================================
    
    print(f"\n--- Performance Comparison ---")
    
    batch_size = 1024
    n_iter = 100
    
    actions_batch = torch.randint(0, NACOMP, (batch_size, TOT_COMP), device=device)
    budget_batch = torch.ones(batch_size, device=device) * budget_lim
    can_repair_batch = torch.ones(batch_size, TOT_COMP, dtype=torch.bool, device=device)
    
    # GPU Stochastic timing
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start = time.time()
    for _ in range(n_iter):
        constraint_stochastic.apply(actions_batch, budget_batch, can_repair_batch, temperature=1.0)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    gpu_stoch_time = (time.time() - start) / n_iter * 1000
    
    # GPU Greedy timing
    torch.cuda.synchronize() if device.startswith('cuda') else None
    start = time.time()
    for _ in range(n_iter):
        constraint_greedy.apply(actions_batch, budget_batch, can_repair_batch)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    gpu_greedy_time = (time.time() - start) / n_iter * 1000
    
    # Legacy timing (single sample, extrapolated)
    start = time.time()
    for _ in range(100):
        legacy_budget_constraint(actions_np[0:1], extra_state, budget_lim)
    legacy_time_single = (time.time() - start) / 100 * 1000
    legacy_time_batch = legacy_time_single * batch_size  # Extrapolated
    
    print(f"{'Method':<20} {'Time/Batch (ms)':<18} {'Throughput (samples/sec)':<25}")
    print("-" * 65)
    print(f"{'Legacy (sequential)':<20} {legacy_time_batch:<18.2f} {batch_size / (legacy_time_batch/1000):<25.0f}")
    print(f"{'GPU Stochastic':<20} {gpu_stoch_time:<18.3f} {batch_size / (gpu_stoch_time/1000):<25.0f}")
    print(f"{'GPU Greedy':<20} {gpu_greedy_time:<18.3f} {batch_size / (gpu_greedy_time/1000):<25.0f}")
    print(f"\nSpeedup vs Legacy: GPU Stochastic = {legacy_time_batch/gpu_stoch_time:.0f}x, GPU Greedy = {legacy_time_batch/gpu_greedy_time:.0f}x")
    
    print("\n✓ Comparison completed")


if __name__ == '__main__':
    benchmark_strategies()
    compare_with_legacy()

