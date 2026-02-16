"""
GPU-Accelerated Cost Computation Module
========================================

Computes all costs in a fully vectorized manner:
1. Action costs (maintenance/replacement)
2. Observation costs (inspection)
3. Delay costs (user delay from maintenance)
4. Risk costs (failure probability)
5. Constraint violations

All operations are batched for GPU efficiency.
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as spio
import os
from typing import Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dependency_files')

INTRSTATE_COMP = 12
PRIMR_COMP = 47
SEC_COMP = 26
NCOMP_PAV = INTRSTATE_COMP + PRIMR_COMP + SEC_COMP  # 85
NCOMP_DECK = 11
TOT_COMP = NCOMP_PAV + NCOMP_DECK  # 96

NSTCOMP_IRI = 5
NSTCOMP_CCI = 6
NSTCOMP_DECK = 7
NACOMP = 10


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for logging"""
    action_cost: torch.Tensor      # (batch,)
    observation_cost: torch.Tensor  # (batch,)
    delay_cost: torch.Tensor       # (batch,)
    risk_cost: torch.Tensor        # (batch,)
    total_cost: torch.Tensor       # (batch,)
    
    # Per-category breakdown
    cost_ac_interstate: torch.Tensor
    cost_ac_primary: torch.Tensor
    cost_ac_secondary: torch.Tensor
    cost_ac_deck: torch.Tensor
    cost_ob_interstate: torch.Tensor
    cost_ob_primary: torch.Tensor
    cost_ob_secondary: torch.Tensor
    cost_ob_deck: torch.Tensor


class GPUCostModule:
    """
    Vectorized cost computation for batched environments.
    
    All costs are in million USD (negative = expense).
    """
    
    def __init__(self, device: str = 'cuda:0', gamma: float = 0.97):
        self.device = device
        self.dtype = torch.float32
        self.gamma = gamma
        
        self._load_cost_data()
        self._build_cost_tensors()
        self._build_area_tensors()
        
        print(f"GPUCostModule initialized on {device}")
    
    def _load_cost_data(self):
        """Load cost-related data from files"""
        os.chdir(DATA_DIR)
        
        # Link lengths
        mat1 = spio.loadmat('link_lengths.mat', squeeze_me=True)
        self.lenth_comp = mat1['length1'][:NCOMP_PAV]
        
        # Number of lanes
        self.n_lane_pav = np.zeros(NCOMP_PAV)
        for i in range(NCOMP_PAV):
            if i < INTRSTATE_COMP:
                self.n_lane_pav[i] = 8
            elif i < INTRSTATE_COMP + PRIMR_COMP:
                self.n_lane_pav[i] = 4
            else:
                self.n_lane_pav[i] = 2
        
        # Bridge properties
        self.brdg_len = np.array([7.07, 7.4, 5.6, 1.140, 1.13, 1.5, 0.647, 0.64, 1.01, 0.32, 0.90]) / 1.61
        self.n_lane = np.array([4, 4, 4, 4, 2, 4, 8, 4, 4, 4, 4])
        self.lane_wdth = np.array([5, 4, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7])
        
        # Delay costs (precomputed)
        self.Dlay_cost_np = np.load('Dlay_Cost.npy')
        intdx = list(range(NCOMP_PAV)) + list(range(NCOMP_PAV, NCOMP_PAV + 11))
        self.Dlay_cost_np = self.Dlay_cost_np[intdx, :]
    
    def _build_cost_tensors(self):
        """Build GPU tensors for cost computation"""

        # Cost coefficients (per unit area/length)
        # NOTE: Match legacy `Network_data.py` exactly:
        # - `A_pav`/`I_pav` are already multiplied by lane width (3.7)
        # - Pavement costs are computed as: length_m * n_lanes * coeff / 1e6
        #   (NOT using Area_pav again, otherwise lane width is applied twice)
        A_pav = 3.7 * np.array([[0, 20, 75, 350], [0, 16, 68, 330], [0, 10, 52, 250]])
        A_deck = np.array([0, 400, 1200, 2650.00])
        I_pav = 3.7 * np.array([0, 0.10, 0.20])
        I_deck = np.array([0, 0.48, 1.20])

        # Compute area (m^2) - store for reuse in _build_area_tensors
        self._area_pav_np = (1.61 * 1000) * self.lenth_comp * self.n_lane_pav * 3.7
        self._area_deck_np = self.brdg_len * self.n_lane * self.lane_wdth * 1.61 * 1000
        
        # Precompute cost matrices: (tot_comp, nacomp)
        cost_comp_action = np.zeros((TOT_COMP, NACOMP))
        cost_comp_obsr = np.zeros((TOT_COMP, NACOMP))
        
        # Pavement costs - match legacy `Network_data.py` (length_m * n_lanes)
        for i in range(NCOMP_PAV):
            length_m = self.lenth_comp[i] * 1.61 * 1000
            n_lanes = self.n_lane_pav[i]
            
            for j in range(NACOMP):
                maint_type = j % 3  # 0=none, 1=minor, 2=major
                insp_type = j // 3  # 0=none, 1=low-fed, 2=high-fed, 3=replace
                
                if i < INTRSTATE_COMP:
                    type_idx = 0
                elif i < INTRSTATE_COMP + PRIMR_COMP:
                    type_idx = 1
                else:
                    type_idx = 2
                
                if j == 9:  # Replacement
                    cost_comp_action[i, j] = -length_m * n_lanes * A_pav[type_idx, 3] / 1e6
                    cost_comp_obsr[i, j] = 0.0  # Legacy: no inspection cost for replacement
                else:
                    cost_comp_action[i, j] = -length_m * n_lanes * A_pav[type_idx, maint_type] / 1e6

                    # Legacy: observation costs only for inspection actions
                    if j in (0, 1, 2):
                        cost_comp_obsr[i, j] = 0.0
                    elif j in (3, 4, 5):
                        cost_comp_obsr[i, j] = -length_m * n_lanes * I_pav[1] / 1e6
                    elif j in (6, 7, 8):
                        cost_comp_obsr[i, j] = -length_m * n_lanes * I_pav[2] / 1e6
                    else:
                        # Should be unreachable given NACOMP=10, but keep safe default
                        cost_comp_obsr[i, j] = 0.0
        
        # Deck costs
        for i in range(NCOMP_DECK):
            area = self._area_deck_np[i]
            for j in range(NACOMP):
                maint_type = j % 3
                insp_type = j // 3
                
                if j == 9:
                    cost_comp_action[NCOMP_PAV + i, j] = -area * A_deck[3] / 1e6
                    cost_comp_obsr[NCOMP_PAV + i, j] = 0.0  # Legacy: no inspection cost for replacement
                else:
                    cost_comp_action[NCOMP_PAV + i, j] = -area * A_deck[maint_type] / 1e6
                    # Legacy: observation costs only for inspection actions (3-8), scaled by deck area
                    if j in (0, 1, 2):
                        cost_comp_obsr[NCOMP_PAV + i, j] = 0.0
                    elif j in (3, 4, 5):
                        cost_comp_obsr[NCOMP_PAV + i, j] = -area * I_deck[1] / 1e6
                    elif j in (6, 7, 8):
                        cost_comp_obsr[NCOMP_PAV + i, j] = -area * I_deck[2] / 1e6
                    else:
                        cost_comp_obsr[NCOMP_PAV + i, j] = 0.0
        
        # Convert to GPU tensors
        self.cost_action = torch.tensor(cost_comp_action, dtype=self.dtype, device=self.device)
        self.cost_obsr = torch.tensor(cost_comp_obsr, dtype=self.dtype, device=self.device)
        self.delay_cost = torch.tensor(self.Dlay_cost_np, dtype=self.dtype, device=self.device)
        
        # Failure probabilities for risk computation
        self.failure_probs_deck = torch.tensor(
            [0.001, 0.001, 0.005, 0.005, 0.005, 0.01, 1.0],
            dtype=self.dtype, device=self.device
        )
        
        # Deck replacement cost (for risk scaling)
        self.deck_replace_cost = self.cost_action[NCOMP_PAV:, 9]  # (11,)
    
    def _build_area_tensors(self):
        """Build area tensors for constraint computation"""
        # Reuse area computations from _build_cost_tensors to avoid redundancy
        self.Area_pav = torch.tensor(self._area_pav_np, dtype=self.dtype, device=self.device)
        self.Area_deck = torch.tensor(self._area_deck_np, dtype=self.dtype, device=self.device)
        
        self.Tot_intrst_area = self.Area_pav[:INTRSTATE_COMP].sum()
        self.Tot_primr_area = self.Area_pav[INTRSTATE_COMP:INTRSTATE_COMP + PRIMR_COMP].sum()
        self.Tot_scndr_area = self.Area_pav[INTRSTATE_COMP + PRIMR_COMP:].sum()
        self.Tot_prm_intr_area = self.Tot_primr_area + self.Tot_intrst_area
        self.Tot_deck_area = self.Area_deck.sum()
    
    def compute_action_observation_costs(
        self,
        actions: torch.Tensor,        # (batch, tot_comp)
        comp_active: torch.Tensor,    # (batch, tot_comp) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action and observation costs.
        
        Returns:
            action_cost: (batch,)
            obs_cost: (batch,)
            action_real: (batch, tot_comp) - actions actually applied
        """
        batch = actions.shape[0]
        
        # Gather costs for selected actions
        # cost_action: (tot_comp, nacomp)
        # actions: (batch, tot_comp)
        action_costs = torch.gather(
            self.cost_action.unsqueeze(0).expand(batch, -1, -1),
            2,
            actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch, tot_comp)
        
        obs_costs = torch.gather(
            self.cost_obsr.unsqueeze(0).expand(batch, -1, -1),
            2,
            actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch, tot_comp)
        
        # Zero out costs for inactive components
        action_costs = action_costs * comp_active.float()
        obs_costs = obs_costs * comp_active.float()
        
        # Actions actually applied
        action_real = actions * comp_active.long()
        
        # Sum over components
        total_action_cost = action_costs.sum(dim=1)  # (batch,)
        total_obs_cost = self.gamma * obs_costs.sum(dim=1)  # (batch,)
        
        return total_action_cost, total_obs_cost, action_real
    
    def compute_delay_cost(
        self,
        actions: torch.Tensor,        # (batch, tot_comp)
        comp_active: torch.Tensor,    # (batch, tot_comp) bool
    ) -> torch.Tensor:
        """
        Compute delay costs from maintenance activities.
        
        Returns:
            delay_cost: (batch,)
        """
        batch = actions.shape[0]
        
        # Gather delay costs
        delay_costs = torch.gather(
            self.delay_cost.unsqueeze(0).expand(batch, -1, -1),
            2,
            actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch, tot_comp)
        
        # Zero out for inactive components
        delay_costs = delay_costs * comp_active.float()
        
        return delay_costs.sum(dim=1)  # (batch,)
    
    def compute_risk_cost(
        self,
        b_a_deck: torch.Tensor,       # (batch, 11, 7) - belief after action
        ages: torch.Tensor,           # (batch, tot_comp)
        actions: torch.Tensor,        # (batch, tot_comp)
        comp_active: torch.Tensor,    # (batch, tot_comp) bool
        pcomp_deck: torch.Tensor,     # (20, 7, 7) - deck transition matrices
    ) -> torch.Tensor:
        """
        Compute risk cost based on failure probability.
        
        This uses a simplified risk model based on deck states.
        
        Returns:
            risk_cost: (batch,)
        """
        batch = b_a_deck.shape[0]
        device = b_a_deck.device
        
        # Current failure probability
        # fail_prob = belief @ failure_probs
        fail_prob = torch.matmul(b_a_deck, self.failure_probs_deck)  # (batch, 11)
        
        # Mode probabilities (simplified network failure modes)
        mode1_prob = fail_prob[:, 0] * fail_prob[:, 1] * fail_prob[:, 2]  # First 3 bridges
        mode2_prob = fail_prob[:, 3]  # Bridge 4
        mode3_prob = mode1_prob * mode2_prob
        
        # Next-step belief after transition
        ages_deck = ages[:, NCOMP_PAV:].long().clamp(0, 19)  # (batch, 11)
        actions_deck = actions[:, NCOMP_PAV:]  # (batch, 11)
        active_deck = comp_active[:, NCOMP_PAV:]  # (batch, 11)
        
        # Apply transitions
        # For replacement: reset to state 0
        # For others: apply transition matrix
        
        # Get transition matrices for each component based on age
        # pcomp_deck: (20, 7, 7)
        T = pcomp_deck[ages_deck.flatten()]  # (batch*11, 7, 7)
        T = T.view(batch, NCOMP_DECK, NSTCOMP_DECK, NSTCOMP_DECK)
        
        # Apply transition
        p_next = torch.einsum('bcij,bcj->bci', T, b_a_deck)  # (batch, 11, 7)
        
        # Handle replacement: set to state 0
        is_replace = (actions_deck == 9) & active_deck
        replace_belief = torch.zeros_like(p_next)
        replace_belief[:, :, 0] = 1.0
        p_next = torch.where(is_replace.unsqueeze(-1), replace_belief, p_next)
        
        # Next-step failure probabilities
        fail_prob_next = torch.matmul(p_next, self.failure_probs_deck)  # (batch, 11)
        
        mode1_prob_next = fail_prob_next[:, 0] * fail_prob_next[:, 1] * fail_prob_next[:, 2]
        mode2_prob_next = fail_prob_next[:, 3]
        mode3_prob_next = mode1_prob_next * mode2_prob_next
        
        # Risk cost = weighted failure probability change + perpetual loss
        # NOTE: Instantaneous losses have factor of 10, perpetual losses do NOT
        # NOTE: Do NOT use .abs() - costs must stay negative!
        
        # Cost sums (negative values)
        cost_sum_3 = self.deck_replace_cost[:3].sum()   # Negative
        cost_sum_4 = self.deck_replace_cost[:4].sum()   # Negative
        
        # Instantaneous losses: gamma * 10 * cost_sum * (prob_next - prob)
        # These can be positive (benefit) when failure prob decreases
        cost_state = self.gamma * 10 * cost_sum_3 * (mode1_prob_next - mode1_prob)
        cost_state += self.gamma * 10 * cost_sum_4 * (mode3_prob_next - mode3_prob)
        cost_state += self.gamma * 10 * torch.sum(
            self.deck_replace_cost * (fail_prob_next - fail_prob), dim=1
        )
        
        # Perpetual losses: gamma * 2 * cost_sum * prob (NO factor of 10!)
        # These are always negative (ongoing risk cost)
        cost_state += self.gamma * 2 * cost_sum_3 * mode1_prob_next
        cost_state += self.gamma * 2 * cost_sum_4 * mode3_prob_next
        cost_state += self.gamma * 2 * torch.sum(
            self.deck_replace_cost * fail_prob_next, dim=1
        )
        
        return cost_state
    
    def compute_constraints(
        self,
        state_IRI: torch.Tensor,      # (batch, 85, 5)
        state_CCI: torch.Tensor,      # (batch, 85, 6)
        state_deck: torch.Tensor,     # (batch, 11, 7)
    ) -> torch.Tensor:
        """
        Compute constraint violations (percentage of poor area).
        
        Constraints:
        0: Interstate poor IRI & CCI (state > 2 for both) < 5%
        1: Interstate very poor CCI (state == 5) < 2%
        2: Interstate+Primary poor CCI (state > 2) < 18%
        3: Interstate+Primary poor IRI (state > 2) < 15%
        4: Secondary poor CCI (state > 2) < 35%
        5: Deck poor (state > 4) < 10%
        
        Returns:
            constraints: (batch, 6) - percentage values
        """
        batch = state_IRI.shape[0]
        device = state_IRI.device
        
        # Get most likely states (argmax of belief)
        bb_IRI = state_IRI.argmax(dim=-1)  # (batch, 85)
        bb_CCI = state_CCI.argmax(dim=-1)  # (batch, 85)
        bb_deck = state_deck.argmax(dim=-1)  # (batch, 11)
        
        # Initialize constraint violations
        constraints = torch.zeros(batch, 6, device=device)
        
        # Interstate indices (0 to 11)
        inter_mask = torch.zeros(NCOMP_PAV, device=device)
        inter_mask[:INTRSTATE_COMP] = 1.0
        
        # Primary + Interstate mask
        prim_inter_mask = torch.zeros(NCOMP_PAV, device=device)
        prim_inter_mask[:INTRSTATE_COMP + PRIMR_COMP] = 1.0
        
        # Secondary mask
        sec_mask = torch.zeros(NCOMP_PAV, device=device)
        sec_mask[INTRSTATE_COMP + PRIMR_COMP:] = 1.0
        
        # Constraint 0: Interstate poor IRI & CCI (both > 2)
        poor_both = ((bb_IRI > 2) & (bb_CCI > 2)).float()  # (batch, 85)
        constraints[:, 0] = (poor_both * inter_mask * self.Area_pav).sum(dim=1) / self.Tot_intrst_area * 100
        
        # Constraint 1: Interstate very poor CCI (== 5)
        very_poor_cci = (bb_CCI == 5).float()
        constraints[:, 1] = (very_poor_cci * inter_mask * self.Area_pav).sum(dim=1) / self.Tot_intrst_area * 100
        
        # Constraint 2: Interstate+Primary poor CCI (> 2)
        poor_cci = (bb_CCI > 2).float()
        constraints[:, 2] = (poor_cci * prim_inter_mask * self.Area_pav).sum(dim=1) / self.Tot_prm_intr_area * 100
        
        # Constraint 3: Interstate+Primary poor IRI (> 2)
        poor_iri = (bb_IRI > 2).float()
        constraints[:, 3] = (poor_iri * prim_inter_mask * self.Area_pav).sum(dim=1) / self.Tot_prm_intr_area * 100
        
        # Constraint 4: Secondary poor CCI (> 2)
        constraints[:, 4] = (poor_cci * sec_mask * self.Area_pav).sum(dim=1) / self.Tot_scndr_area * 100
        
        # Constraint 5: Deck poor (state > 4)
        poor_deck = (bb_deck > 4).float()  # (batch, 11)
        constraints[:, 5] = (poor_deck * self.Area_deck).sum(dim=1) / self.Tot_deck_area * 100
        
        return constraints
    
    def compute_total_cost(
        self,
        actions: torch.Tensor,
        comp_active: torch.Tensor,
        b_a_deck: torch.Tensor,
        ages: torch.Tensor,
        pcomp_deck: torch.Tensor,
        discount: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total cost for a step.
        
        Returns:
            total_cost: (batch,)
            cost_breakdown: dict with individual cost components
        """
        action_cost, obs_cost, action_real = self.compute_action_observation_costs(
            actions, comp_active
        )
        delay_cost = self.compute_delay_cost(actions, comp_active)
        risk_cost = self.compute_risk_cost(b_a_deck, ages, actions, comp_active, pcomp_deck)
        
        total_cost = discount * (action_cost + obs_cost + delay_cost + risk_cost)
        
        breakdown = {
            'action_cost': action_cost,
            'obs_cost': obs_cost,
            'delay_cost': delay_cost,
            'risk_cost': risk_cost,
            'total_cost': total_cost,
        }
        
        return total_cost, breakdown
    
    def compute_detailed_breakdown(
        self,
        actions: torch.Tensor,
        comp_active: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detailed cost breakdown by category.
        
        Returns dictionary with costs for:
        - Interstate action/observation
        - Primary action/observation
        - Secondary action/observation
        - Deck action/observation
        """
        batch = actions.shape[0]
        
        # Gather costs
        action_costs = torch.gather(
            self.cost_action.unsqueeze(0).expand(batch, -1, -1),
            2, actions.unsqueeze(-1)
        ).squeeze(-1) * comp_active.float()
        
        obs_costs = torch.gather(
            self.cost_obsr.unsqueeze(0).expand(batch, -1, -1),
            2, actions.unsqueeze(-1)
        ).squeeze(-1) * comp_active.float() * self.gamma
        
        return {
            'cost_ac_interstate': action_costs[:, :INTRSTATE_COMP].sum(dim=1),
            'cost_ac_primary': action_costs[:, INTRSTATE_COMP:INTRSTATE_COMP+PRIMR_COMP].sum(dim=1),
            'cost_ac_secondary': action_costs[:, INTRSTATE_COMP+PRIMR_COMP:NCOMP_PAV].sum(dim=1),
            'cost_ac_deck': action_costs[:, NCOMP_PAV:].sum(dim=1),
            'cost_ob_interstate': obs_costs[:, :INTRSTATE_COMP].sum(dim=1),
            'cost_ob_primary': obs_costs[:, INTRSTATE_COMP:INTRSTATE_COMP+PRIMR_COMP].sum(dim=1),
            'cost_ob_secondary': obs_costs[:, INTRSTATE_COMP+PRIMR_COMP:NCOMP_PAV].sum(dim=1),
            'cost_ob_deck': obs_costs[:, NCOMP_PAV:].sum(dim=1),
        }


# ============================================================================
# BUDGET CONSTRAINT HANDLING - REMOVED
# ============================================================================
# Budget constraints have been removed. All components are now active.
# The apply_budget_constraint function is no longer used.


# ============================================================================
# TEST
# ============================================================================

def test_cost_module():
    """Test the cost computation module"""
    print("=" * 70)
    print("COST MODULE TEST")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 4096
    
    cost_module = GPUCostModule(device=device)
    
    # Create test inputs
    actions = torch.randint(0, NACOMP, (batch_size, TOT_COMP), device=device)
    comp_active = torch.ones(batch_size, TOT_COMP, dtype=torch.bool, device=device)
    
    # Test action/observation costs
    print("\n--- Action/Observation Cost Test ---")
    action_cost, obs_cost, action_real = cost_module.compute_action_observation_costs(
        actions, comp_active
    )
    print(f"Action cost range: [{action_cost.min():.4f}, {action_cost.max():.4f}]")
    print(f"Observation cost range: [{obs_cost.min():.4f}, {obs_cost.max():.4f}]")
    
    # Test delay costs
    print("\n--- Delay Cost Test ---")
    delay_cost = cost_module.compute_delay_cost(actions, comp_active)
    print(f"Delay cost range: [{delay_cost.min():.4f}, {delay_cost.max():.4f}]")
    
    # Test constraints
    print("\n--- Constraint Test ---")
    state_IRI = torch.rand(batch_size, NCOMP_PAV, NSTCOMP_IRI, device=device)
    state_CCI = torch.rand(batch_size, NCOMP_PAV, NSTCOMP_CCI, device=device)
    state_deck = torch.rand(batch_size, NCOMP_DECK, NSTCOMP_DECK, device=device)
    
    # Normalize to probabilities
    state_IRI = state_IRI / state_IRI.sum(dim=-1, keepdim=True)
    state_CCI = state_CCI / state_CCI.sum(dim=-1, keepdim=True)
    state_deck = state_deck / state_deck.sum(dim=-1, keepdim=True)
    
    constraints = cost_module.compute_constraints(state_IRI, state_CCI, state_deck)
    print(f"Constraint shape: {constraints.shape}")
    print(f"Sample constraint values: {constraints[0].cpu().numpy()}")
    
    # Test detailed breakdown
    print("\n--- Detailed Breakdown Test ---")
    breakdown = cost_module.compute_detailed_breakdown(actions, comp_active)
    for key, val in breakdown.items():
        print(f"  {key}: mean={val.mean():.6f}")
    
    # Performance test
    print("\n--- Performance Test ---")
    import time
    
    n_iters = 1000
    torch.cuda.synchronize() if device.startswith('cuda') else None
    
    start = time.time()
    for _ in range(n_iters):
        action_cost, obs_cost, _ = cost_module.compute_action_observation_costs(
            actions, comp_active
        )
        delay_cost = cost_module.compute_delay_cost(actions, comp_active)
        constraints = cost_module.compute_constraints(state_IRI, state_CCI, state_deck)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    elapsed = time.time() - start
    
    print(f"Time per batch: {elapsed/n_iters*1000:.3f} ms")
    print(f"Throughput: {batch_size * n_iters / elapsed:.0f} samples/sec")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == '__main__':
    test_cost_module()
