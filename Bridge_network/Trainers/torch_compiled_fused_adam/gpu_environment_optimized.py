"""
Optimized GPU Environment for Bridge Network
=============================================

Key optimizations:
1. NO matrix replication - store unique matrices only
2. Component type indexing - select matrices via indices
3. Fully vectorized operations - no Python loops
4. Batch processing - multiple environments in parallel
5. Memory-efficient layout - contiguous GPU tensors

Memory comparison:
- Original: pcomp_CCI (6,6,20,85) = 612K floats replicated
- Optimized: 3 unique (6,6,20) = 2.16K floats + type indices

Speedup: ~50-100x through vectorization + batching
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as spio
import os
import time
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dependency_files')

# Network dimensions
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
MAX_AGE = 20


@dataclass
class BeliefStateBatch:
    """Batched belief state for multiple environments"""
    IRI: torch.Tensor    # (batch, ncomp_pav, nstcomp_IRI)
    CCI: torch.Tensor    # (batch, ncomp_pav, nstcomp_CCI)
    deck: torch.Tensor   # (batch, ncomp_deck, nstcomp_deck)
    
    def clone(self):
        return BeliefStateBatch(
            self.IRI.clone(),
            self.CCI.clone(),
            self.deck.clone()
        )


class OptimizedGPUEnvironment:
    """
    Memory-efficient, fully vectorized GPU environment.
    
    Key design principles:
    1. Store only UNIQUE matrices (not replicated per component)
    2. Use component-type indices to select the right matrix
    3. All operations are batched and vectorized
    4. No Python loops in the hot path
    
    NOTE: BUDGET REMOVED FOR INFINITE BUDGET MODE
    =============================================
    Budget constraints have been removed. All components are always active.
    Budget-related code has been commented out (not deleted) for easy restoration.
    
    To restore budget functionality:
    1. Uncomment comp_active parameter in belief_update_vectorized()
    2. Uncomment active_pav and active_deck definitions
    3. Comment out the "All components active" definitions
    4. Update all callers to pass comp_active argument
    5. Re-enable budget constraint logic in cost module and training code
    """
    
    def __init__(self, device: str = 'cuda:0', batch_size: int = 1):
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.float32
        
        # Load and process data
        self._load_data()
        self._build_optimized_matrices()
        self._build_component_indices()
        
        print(f"OptimizedGPUEnvironment initialized on {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Memory reduction: ~{self._compute_memory_savings():.1f}x")
    
    def _load_data(self):
        """Load raw data from .mat files"""
        os.chdir(DATA_DIR)
        
        self.mat_pav06 = spio.loadmat('Smoothed_TP_MSI_06.mat', squeeze_me=True)
        self.mat_pav08 = spio.loadmat('Smoothed_TP_MSI_08.mat', squeeze_me=True)
        self.mat_pav20 = spio.loadmat('Smoothed_TP_MSI_20.mat', squeeze_me=True)
        self.mat_deck = spio.loadmat('Deck_TP_for_DRL.mat', squeeze_me=True)
        
        # Load Tag (components needing 2-year replacement)
        Ta = np.load('Total_action_duration.npy')
        intdx = list(range(NCOMP_PAV)) + list(range(NCOMP_PAV, NCOMP_PAV + 11))
        Ta = Ta[intdx, :]
        self.Tag = torch.tensor((Ta[:, -1] > 365).astype(np.float32), device=self.device)
    
    def _build_optimized_matrices(self):
        """Build memory-efficient matrix storage"""
        
        # ================================================================
        # TRANSITION MATRICES - Only store unique ones
        # ================================================================
        
        # IRI: Single (5,5) matrix for all components
        pcomp_IRI = np.array([
            [0.839, 0.121, 0.039, 0., 0.],
            [0., 0.787, 0.142, 0.07, 0.],
            [0., 0., 0.708, 0.192, 0.099],
            [0., 0., 0., 0.578, 0.421],
            [0., 0., 0., 0., 1.]
        ])
        self.pcomp_IRI = torch.tensor(pcomp_IRI.T, dtype=self.dtype, device=self.device)
        
        # CCI: 3 unique matrices (interstate, primary, secondary) x 20 ages
        # prob2 shape is (6, 6, 20) - [from_state, to_state, age]
        # We want: (3, 20, to_state, from_state) for efficient matmul
        pcomp_CCI_unique = np.zeros((3, MAX_AGE, NSTCOMP_CCI, NSTCOMP_CCI))
        prob2_06 = self.mat_pav06['prob2'][:NSTCOMP_CCI, :NSTCOMP_CCI, :]  # (6, 6, 20)
        prob2_08 = self.mat_pav08['prob2'][:NSTCOMP_CCI, :NSTCOMP_CCI, :]
        prob2_20 = self.mat_pav20['prob2'][:NSTCOMP_CCI, :NSTCOMP_CCI, :]
        for age in range(MAX_AGE):
            pcomp_CCI_unique[0, age, :, :] = prob2_06[:, :, age].T  # Transpose for T @ b
            pcomp_CCI_unique[1, age, :, :] = prob2_08[:, :, age].T
            pcomp_CCI_unique[2, age, :, :] = prob2_20[:, :, age].T  
        
        self.pcomp_CCI = torch.tensor(
            pcomp_CCI_unique, 
            dtype=self.dtype, device=self.device
        )
        
        # Deck: Single matrix for all deck components x 20 ages
        # Tp_1 shape is (7, 7, 20) - [from_state, to_state, age]
        # We want: (20, to_state, from_state) for efficient matmul
        Tp_1 = self.mat_deck['Tp_1'][:NSTCOMP_DECK, :NSTCOMP_DECK, :]  # (7, 7, 20)
        pcomp_deck_unique = np.zeros((MAX_AGE, NSTCOMP_DECK, NSTCOMP_DECK))
        for age in range(MAX_AGE):
            pcomp_deck_unique[age, :, :] = Tp_1[:, :, age].T  # Transpose for T @ b
        
        self.pcomp_deck = torch.tensor(
            pcomp_deck_unique,
            dtype=self.dtype, device=self.device
        )
        
        # ================================================================
        # ACTION EFFECT MATRICES - Store once
        # ================================================================
        
        # Minor repair matrices
        self.pobs_minor_IRI = torch.tensor(np.array([
            [0.97, 0.03, 0, 0, 0],
            [0.85, 0.12, 0.03, 0, 0],
            [0.45, 0.4, 0.12, 0.03, 0],
            [0, 0.45, 0.4, 0.12, 0.03],
            [0, 0, 0.45, 0.4, 0.15]
        ]).T, dtype=self.dtype, device=self.device)
        
        self.pobs_minor_CCI = torch.tensor(np.array([
            [0.97, 0.03, 0, 0, 0, 0],
            [0.87, 0.1, 0.03, 0, 0, 0],
            [0.4, 0.47, 0.1, 0.03, 0, 0],
            [0, 0.4, 0.47, 0.1, 0.03, 0],
            [0, 0, 0.4, 0.47, 0.1, 0.03],
            [0, 0, 0, 0.4, 0.47, 0.13]
        ]).T, dtype=self.dtype, device=self.device)
        
        self.pobs_minor_deck = torch.tensor(np.array([
            [0.97, 0.03, 0, 0, 0, 0, 0],
            [0.85, 0.12, 0.03, 0, 0, 0, 0],
            [0.4, 0.45, 0.12, 0.03, 0, 0, 0],
            [0, 0.4, 0.45, 0.12, 0.03, 0, 0],
            [0, 0, 0.4, 0.45, 0.12, 0.03, 0],
            [0, 0, 0, 0.40, 0.45, 0.15, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]).T, dtype=self.dtype, device=self.device)
        
        # Major repair matrices
        self.pobs_major_IRI = torch.tensor(np.array([
            [1, 0, 0, 0, 0],
            [0.95, 0.05, 0, 0, 0],
            [0.80, 0.20, 0, 0, 0],
            [0.7, 0.25, 0.05, 0, 0],
            [0.45, 0.35, 0.2, 0, 0]
        ]).T, dtype=self.dtype, device=self.device)
        
        self.pobs_major_CCI = torch.tensor(np.array([
            [1, 0, 0, 0, 0, 0],
            [0.96, 0.04, 0, 0, 0, 0],
            [0.8, 0.2, 0, 0, 0, 0],
            [0.65, 0.25, 0.1, 0, 0, 0],
            [0.5, 0.3, 0.2, 0, 0, 0],
            [0.4, 0.3, 0.3, 0, 0, 0]
        ]).T, dtype=self.dtype, device=self.device)
        
        self.pobs_major_deck = torch.tensor(np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0.95, 0.05, 0, 0, 0, 0, 0],
            [0.8, 0.2, 0, 0, 0, 0, 0],
            [0.6, 0.3, 0.1, 0, 0, 0, 0],
            [0.4, 0.4, 0.2, 0, 0, 0, 0],
            [0.3, 0.4, 0.3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]).T, dtype=self.dtype, device=self.device)
        
        # ================================================================
        # OBSERVATION MATRICES - Indexed by action type (0-2: none, 3-5: low, 6-8: high, 9: replace)
        # ================================================================
        
        # IRI observation: (3, 5, 5) - [obs_type, state, obs]
        pobs_IRI_types = np.zeros((3, NSTCOMP_IRI, NSTCOMP_IRI))
        pobs_IRI_types[0] = 1.0 / NSTCOMP_IRI  # No inspection
        pobs_IRI_types[1] = np.array([  # Low-fed inspection
            [0.80, 0.20, 0, 0, 0],
            [0.20, 0.60, 0.20, 0, 0],
            [0, 0.20, 0.60, 0.20, 0],
            [0, 0, 0.20, 0.60, 0.20],
            [0, 0, 0, 0.20, 0.80]
        ])
        pobs_IRI_types[2] = np.array([  # High-fed inspection
            [0.90, 0.10, 0, 0, 0],
            [0.05, 0.90, 0.05, 0, 0],
            [0, 0.05, 0.90, 0.05, 0],
            [0, 0, 0.05, 0.90, 0.05],
            [0, 0, 0, 0.10, 0.90]
        ])
        self.pobs_IRI_types = torch.tensor(pobs_IRI_types, dtype=self.dtype, device=self.device)
        
        # CCI observation: (3, 6, 6)
        pobs_CCI_types = np.zeros((3, NSTCOMP_CCI, NSTCOMP_CCI))
        pobs_CCI_types[0] = 1.0 / NSTCOMP_CCI
        pobs_CCI_types[1] = np.array([
            [0.688, 0.258, 0.054, 0, 0, 0],
            [0.277, 0.422, 0.297, 0.004, 0, 0],
            [0.024, 0.14, 0.648, 0.166, 0.022, 0.001],
            [0, 0.003, 0.266, 0.455, 0.249, 0.027],
            [0, 0, 0.031, 0.223, 0.486, 0.26],
            [0, 0, 0, 0.006, 0.061, 0.936]
        ])
        pobs_CCI_types[2] = np.array([
            [0.803, 0.195, 0.002, 0, 0, 0],
            [0.152, 0.664, 0.183, 0, 0, 0],
            [0.001, 0.078, 0.822, 0.1, 0, 0],
            [0, 0, 0.149, 0.693, 0.158, 0],
            [0, 0, 0.001, 0.137, 0.718, 0.144],
            [0, 0, 0, 0, 0.045, 0.97]
        ])
        self.pobs_CCI_types = torch.tensor(pobs_CCI_types, dtype=self.dtype, device=self.device)
        
        # Deck observation: (3, 7, 7)
        pobs_deck_types = np.zeros((3, NSTCOMP_DECK, NSTCOMP_DECK))
        pobs_deck_types[0, :6, :6] = 1.0 / 6
        pobs_deck_types[0, 6, 6] = 1.0
        pobs_deck_types[1] = np.array([
            [0.8, 0.15, 0.05, 0, 0, 0, 0],
            [0.15, 0.65, 0.15, 0.05, 0, 0, 0],
            [0.05, 0.15, 0.6, 0.15, 0.05, 0, 0],
            [0, 0.05, 0.15, 0.6, 0.15, 0.05, 0],
            [0, 0, 0.05, 0.15, 0.65, 0.15, 0],
            [0, 0, 0, 0.05, 0.15, 0.8, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        pobs_deck_types[2] = np.array([
            [0.9, 0.1, 0, 0, 0, 0, 0],
            [0.1, 0.8, 0.1, 0, 0, 0, 0],
            [0, 0.1, 0.8, 0.1, 0, 0, 0],
            [0, 0, 0.1, 0.8, 0.1, 0, 0],
            [0, 0, 0, 0.1, 0.8, 0.1, 0],
            [0, 0, 0, 0, 0.1, 0.9, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.pobs_deck_types = torch.tensor(pobs_deck_types, dtype=self.dtype, device=self.device)
    
    def _build_component_indices(self):
        """Build index tensors for component type lookup"""
        
        # Pavement component types: 0=interstate, 1=primary, 2=secondary
        pav_types = torch.zeros(NCOMP_PAV, dtype=torch.long, device=self.device)
        pav_types[INTRSTATE_COMP:INTRSTATE_COMP + PRIMR_COMP] = 1
        pav_types[INTRSTATE_COMP + PRIMR_COMP:] = 2
        self.pav_type_idx = pav_types
        
        # Action to observation type: 0-2->0, 3-5->1, 6-8->2
        action_to_obs = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0], 
                                      dtype=torch.long, device=self.device)
        self.action_to_obs_type = action_to_obs
        
        # Action mod 3 for maintenance type
        self.action_mod3 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                                         dtype=torch.long, device=self.device)
    
    def _compute_memory_savings(self):
        """Compute approximate memory reduction factor"""
        # Original: replicated matrices
        original = (6*6*20*85 + 7*7*20*11 + 85*10*5*5 + 85*10*6*6 + 11*10*7*7)
        # Optimized: unique matrices only
        optimized = (5*5 + 3*20*6*6 + 20*7*7 + 3*5*5 + 3*6*6 + 3*7*7 + 
                     2*5*5 + 2*6*6 + 2*7*7)
        return original / optimized
    
    # ========================================================================
    # VECTORIZED BELIEF UPDATE
    # ========================================================================
    
    def belief_update_vectorized(
        self,
        belief: BeliefStateBatch,
        actions: torch.Tensor,      # (batch, tot_comp)
        ages: torch.Tensor,         # (batch, tot_comp)
        extra_state: torch.Tensor,  # (batch, ncomp_pav)
        comp_active: Optional[torch.Tensor] = None,  # (batch, tot_comp) bool
        legacy_sampling: bool = False,
    ) -> Tuple[BeliefStateBatch, BeliefStateBatch, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fully vectorized belief update - NO PYTHON LOOPS.
        
        Processes all components in all batch environments simultaneously.
        
        NOTE: BUDGET REMOVED FOR INFINITE BUDGET MODE
        =============================================
        This function has been modified to work without budget constraints.
        Budget-related code has been commented out (not deleted) for easy restoration.
        
        To restore budget functionality:
        1. Uncomment comp_active parameter in function signature
        2. Uncomment active_pav and active_deck definitions (lines ~335, ~478)
        3. Comment out the "All components active" definitions (lines ~336, ~479)
        4. Update all callers to pass comp_active argument
        """
        batch = actions.shape[0]
        device = self.device
        
        # Clamp ages to valid range
        ages = ages.clamp(0, MAX_AGE - 1)
        
        # ================================================================
        # PAVEMENT COMPONENTS (IRI + CCI)
        # ================================================================
        
        actions_pav = actions[:, :NCOMP_PAV]  # (batch, 85)
        ages_pav = ages[:, :NCOMP_PAV]        # (batch, 85)
        # Active mask (budget-aware if provided)
        if comp_active is None:
            active_pav = torch.ones(batch, NCOMP_PAV, dtype=torch.bool, device=device)
        else:
            active_pav = comp_active[:, :NCOMP_PAV]
        
        # Get action properties
        action_mod3 = actions_pav % 3         # (batch, 85)
        is_replace = (actions_pav == 9)       # (batch, 85)
        is_minor = (action_mod3 == 1)         # (batch, 85)
        is_major = (action_mod3 == 2)         # (batch, 85)
        is_donothing = (action_mod3 == 0) & ~is_replace
        
        # Observation types for each component
        obs_type_idx = self.action_to_obs_type[actions_pav]  # (batch, 85)
        
        # Get component type for CCI transitions
        comp_types = self.pav_type_idx.unsqueeze(0).expand(batch, -1)  # (batch, 85)
        
        # ----------------------------------------------------------------
        # Step 1: Compute transition probabilities p1 (IRI), p2 (CCI)
        # ----------------------------------------------------------------

        # Legacy semantics for extra_state==1 (pavement only):
        # Reset belief to state 0 BEFORE the normal update (and still do sampling/update if action != 9).
        reset_mask = (extra_state == 1)
        reset_mask_bool = reset_mask.to(dtype=torch.bool)
        reset_belief_IRI = torch.zeros_like(belief.IRI)
        reset_belief_IRI[:, :, 0] = 1.0
        reset_belief_CCI = torch.zeros_like(belief.CCI)
        reset_belief_CCI[:, :, 0] = 1.0
        prior_IRI = torch.where(reset_mask_bool.unsqueeze(-1), reset_belief_IRI, belief.IRI)
        prior_CCI = torch.where(reset_mask_bool.unsqueeze(-1), reset_belief_CCI, belief.CCI)
        
        # IRI transition: legacy semantics are (to,from) @ belief(from)
        # self.pcomp_IRI is stored as (to,from) (we pre-transpose when building it)
        p1_donothing = torch.einsum('ij,bcj->bci', self.pcomp_IRI, prior_IRI)  # (batch, 85, 5)
        
        # Minor repair: legacy uses pobs_minor_IRI.T @ belief
        p1_minor = torch.einsum('ij,bcj->bci', self.pobs_minor_IRI, prior_IRI)  # (batch, 85, 5)
        
        # Major repair: legacy uses pobs_major_IRI.T @ belief
        p1_major = torch.einsum('ij,bcj->bci', self.pobs_major_IRI, prior_IRI)  # (batch, 85, 5)
        
        # Combine based on action type
        p1 = torch.where(is_minor.unsqueeze(-1), p1_minor,
             torch.where(is_major.unsqueeze(-1), p1_major, p1_donothing))
        
        # CCI transition: indexed by component type and age
        # pcomp_CCI: (3, 20, 6, 6) - [type, age, to, from]
        # Need to gather the right matrix for each (batch, comp) pair
        
        # Flatten indices for gather
        type_flat = comp_types.flatten()           # (batch*85,)
        age_flat = ages_pav.flatten().long()       # (batch*85,)
        
        # Get transition matrices: (batch*85, 6, 6)
        T_CCI = self.pcomp_CCI[type_flat, age_flat]  # (batch*85, 6, 6)
        T_CCI = T_CCI.view(batch, NCOMP_PAV, NSTCOMP_CCI, NSTCOMP_CCI)
        
        # Apply transition: einsum for batched matmul
        p2_donothing = torch.einsum('bcij,bcj->bci', T_CCI, prior_CCI)  # (batch, 85, 6)
        
        # Minor/major for CCI: legacy uses pobs_*_CCI.T @ belief
        p2_minor = torch.einsum('ij,bcj->bci', self.pobs_minor_CCI, prior_CCI)
        p2_major = torch.einsum('ij,bcj->bci', self.pobs_major_CCI, prior_CCI)
        
        p2 = torch.where(is_minor.unsqueeze(-1), p2_minor,
             torch.where(is_major.unsqueeze(-1), p2_major, p2_donothing))
        
        # ----------------------------------------------------------------
        # Step 2: Sample observations
        # ----------------------------------------------------------------
        
        # Observation distributions: p @ pobs
        # pobs_*_types: (3, nstate, nobs)
        
        # IRI observation dist
        pobs_IRI_selected = self.pobs_IRI_types[obs_type_idx]  # (batch, 85, 5, 5)
        obs_dist_IRI = torch.einsum('bcs,bcso->bco', p1, pobs_IRI_selected)  # (batch, 85, 5)
        obs_dist_IRI = obs_dist_IRI.clamp(min=1e-10)
        obs_dist_IRI = obs_dist_IRI / obs_dist_IRI.sum(dim=-1, keepdim=True)
        
        # CCI observation dist
        pobs_CCI_selected = self.pobs_CCI_types[obs_type_idx]  # (batch, 85, 6, 6)
        obs_dist_CCI = torch.einsum('bcs,bcso->bco', p2, pobs_CCI_selected)
        obs_dist_CCI = obs_dist_CCI.clamp(min=1e-10)
        obs_dist_CCI = obs_dist_CCI / obs_dist_CCI.sum(dim=-1, keepdim=True)

        # Sample observations (Torch-fast default; NumPy legacy-compatible optional)
        if legacy_sampling:
            obs_IRI_np = np.zeros((batch, NCOMP_PAV), dtype=np.int64)
            obs_CCI_np = np.zeros((batch, NCOMP_PAV), dtype=np.int64)
            obs_dist_IRI_np = obs_dist_IRI.detach().cpu().double().numpy()
            obs_dist_CCI_np = obs_dist_CCI.detach().cpu().double().numpy()
            # Re-normalize in numpy to ensure sum == 1.0 exactly (float precision)
            obs_dist_IRI_np = obs_dist_IRI_np / obs_dist_IRI_np.sum(axis=-1, keepdims=True)
            obs_dist_CCI_np = obs_dist_CCI_np / obs_dist_CCI_np.sum(axis=-1, keepdims=True)
            actions_pav_np = actions_pav.detach().cpu().numpy()
            for b_ix in range(batch):
                for i in range(NCOMP_PAV):
                    if actions_pav_np[b_ix, i] == 9:
                        obs_IRI_np[b_ix, i] = 0
                        obs_CCI_np[b_ix, i] = 0
                        continue
                    obs_IRI_np[b_ix, i] = np.random.choice(NSTCOMP_IRI, p=obs_dist_IRI_np[b_ix, i])
                    obs_CCI_np[b_ix, i] = np.random.choice(NSTCOMP_CCI, p=obs_dist_CCI_np[b_ix, i])
            obs_IRI = torch.from_numpy(obs_IRI_np).to(device=device, dtype=torch.long)
            obs_CCI = torch.from_numpy(obs_CCI_np).to(device=device, dtype=torch.long)
        else:
            obs_IRI = torch.multinomial(obs_dist_IRI.view(-1, NSTCOMP_IRI), 1).view(batch, NCOMP_PAV)
            obs_CCI = torch.multinomial(obs_dist_CCI.view(-1, NSTCOMP_CCI), 1).view(batch, NCOMP_PAV)
        
        # ----------------------------------------------------------------
        # Step 3: Bayesian belief update
        # ----------------------------------------------------------------
        
        # Gather likelihood: pobs[state, obs]
        # One-hot encode observations for gathering
        obs_IRI_onehot = F.one_hot(obs_IRI, NSTCOMP_IRI).float()  # (batch, 85, 5)
        obs_CCI_onehot = F.one_hot(obs_CCI, NSTCOMP_CCI).float()  # (batch, 85, 6)
        
        # Likelihood = pobs[:, :, obs] for each component
        likelihood_IRI = torch.einsum('bcso,bco->bcs', pobs_IRI_selected, obs_IRI_onehot)
        likelihood_CCI = torch.einsum('bcso,bco->bcs', pobs_CCI_selected, obs_CCI_onehot)
        
        # Posterior: p * likelihood / Z
        posterior_IRI = p1 * likelihood_IRI
        posterior_IRI = posterior_IRI.clamp(min=1e-10)
        posterior_IRI = posterior_IRI / posterior_IRI.sum(dim=-1, keepdim=True)
        
        posterior_CCI = p2 * likelihood_CCI
        posterior_CCI = posterior_CCI.clamp(min=1e-10)
        posterior_CCI = posterior_CCI / posterior_CCI.sum(dim=-1, keepdim=True)
        
        # ----------------------------------------------------------------
        # Step 4: Handle replacements and extra_state
        # ----------------------------------------------------------------
        
        # Replacement: belief goes to state 0
        replace_belief_IRI = torch.zeros_like(posterior_IRI)
        replace_belief_IRI[:, :, 0] = 1.0
        replace_belief_CCI = torch.zeros_like(posterior_CCI)
        replace_belief_CCI[:, :, 0] = 1.0
        
        # Apply replacement mask
        replace_mask = (is_replace & active_pav)
        b_prime_IRI = torch.where(replace_mask.unsqueeze(-1), replace_belief_IRI, posterior_IRI)
        b_prime_CCI = torch.where(replace_mask.unsqueeze(-1), replace_belief_CCI, posterior_CCI)
        
        # Handle inactive components (just transition, no observation update)
        inactive_pav = ~active_pav & (extra_state == 0)
        b_prime_IRI = torch.where(inactive_pav.unsqueeze(-1), posterior_IRI, b_prime_IRI)
        b_prime_CCI = torch.where(inactive_pav.unsqueeze(-1), posterior_CCI, b_prime_CCI)
        
        # Update observations for replacements
        obs_IRI = torch.where(replace_mask, torch.zeros_like(obs_IRI), obs_IRI)
        obs_CCI = torch.where(replace_mask, torch.zeros_like(obs_CCI), obs_CCI)
        
        # Update extra_state
        extra_state_prime = extra_state.clone()
        was_replaced = is_replace & active_pav
        extra_state_prime = torch.where(extra_state == 1, torch.zeros_like(extra_state), extra_state_prime)
        # Tag components get extra_state=1 after replacement
        tag_mask = self.Tag[:NCOMP_PAV].unsqueeze(0).expand(batch, -1)
        extra_state_prime = torch.where(was_replaced & (tag_mask == 1) & (extra_state == 0),
                                        torch.ones_like(extra_state), extra_state_prime)
        
        # ================================================================
        # DECK COMPONENTS
        # ================================================================
        
        actions_deck = actions[:, NCOMP_PAV:]  # (batch, 11)
        ages_deck = ages[:, NCOMP_PAV:]        # (batch, 11)
        if comp_active is None:
            active_deck = torch.ones(batch, NCOMP_DECK, dtype=torch.bool, device=device)
        else:
            active_deck = comp_active[:, NCOMP_PAV:]
        
        action_mod3_deck = actions_deck % 3
        is_replace_deck = (actions_deck == 9)
        is_minor_deck = (action_mod3_deck == 1)
        is_major_deck = (action_mod3_deck == 2)
        
        obs_type_deck = self.action_to_obs_type[actions_deck]  # (batch, 11)
        
        # Deck transitions - same matrix for all decks, indexed by age
        T_deck = self.pcomp_deck[ages_deck.flatten().long()]  # (batch*11, 7, 7)
        T_deck = T_deck.view(batch, NCOMP_DECK, NSTCOMP_DECK, NSTCOMP_DECK)
        
        # For minor/major, first apply action effect, then transition
        p_deck_direct = torch.einsum('bcij,bcj->bci', T_deck, belief.deck)
        
        # Minor repair path: legacy uses pobs_minor_deck.T @ belief
        p_deck_minor_pre = torch.einsum('ij,bcj->bci', self.pobs_minor_deck, belief.deck)
        p_deck_minor = torch.einsum('bcij,bcj->bci', T_deck, p_deck_minor_pre)
        
        # Major repair path: legacy uses pobs_major_deck.T @ belief
        p_deck_major_pre = torch.einsum('ij,bcj->bci', self.pobs_major_deck, belief.deck)
        p_deck_major = torch.einsum('bcij,bcj->bci', T_deck, p_deck_major_pre)
        
        p_deck = torch.where(is_minor_deck.unsqueeze(-1), p_deck_minor,
                 torch.where(is_major_deck.unsqueeze(-1), p_deck_major, p_deck_direct))
        
        # Sample deck observations
        pobs_deck_selected = self.pobs_deck_types[obs_type_deck]  # (batch, 11, 7, 7)
        obs_dist_deck = torch.einsum('bcs,bcso->bco', p_deck, pobs_deck_selected)
        obs_dist_deck = obs_dist_deck.clamp(min=1e-10)
        obs_dist_deck = obs_dist_deck / obs_dist_deck.sum(dim=-1, keepdim=True)

        if legacy_sampling:
            obs_deck_np = np.zeros((batch, NCOMP_DECK), dtype=np.int64)
            obs_dist_deck_np = obs_dist_deck.detach().cpu().double().numpy()
            # Re-normalize in numpy to ensure sum == 1.0 exactly (float precision)
            obs_dist_deck_np = obs_dist_deck_np / obs_dist_deck_np.sum(axis=-1, keepdims=True)
            actions_deck_np = actions_deck.detach().cpu().numpy()
            for b_ix in range(batch):
                for j in range(NCOMP_DECK):
                    if actions_deck_np[b_ix, j] == 9:
                        obs_deck_np[b_ix, j] = 0
                        continue
                    obs_deck_np[b_ix, j] = np.random.choice(NSTCOMP_DECK, p=obs_dist_deck_np[b_ix, j])
            obs_deck = torch.from_numpy(obs_deck_np).to(device=device, dtype=torch.long)
        else:
            obs_deck = torch.multinomial(obs_dist_deck.view(-1, NSTCOMP_DECK), 1).view(batch, NCOMP_DECK)
        
        # Deck belief update
        obs_deck_onehot = F.one_hot(obs_deck, NSTCOMP_DECK).float()
        likelihood_deck = torch.einsum('bcso,bco->bcs', pobs_deck_selected, obs_deck_onehot)
        
        posterior_deck = p_deck * likelihood_deck
        posterior_deck = posterior_deck.clamp(min=1e-10)
        posterior_deck = posterior_deck / posterior_deck.sum(dim=-1, keepdim=True)
        
        # Handle replacements
        replace_belief_deck = torch.zeros_like(posterior_deck)
        replace_belief_deck[:, :, 0] = 1.0
        
        replace_mask_deck = is_replace_deck & active_deck
        b_prime_deck = torch.where(replace_mask_deck.unsqueeze(-1), replace_belief_deck, posterior_deck)
        
        obs_deck = torch.where(replace_mask_deck, torch.zeros_like(obs_deck), obs_deck)
        
        # ================================================================
        # BUILD OUTPUT
        # ================================================================
        
        b_prime = BeliefStateBatch(b_prime_IRI, b_prime_CCI, b_prime_deck)
        
        # b_a (belief after action effect, before observation) - for compatibility
        b_a_IRI = torch.where(is_minor.unsqueeze(-1), p1_minor,
                  torch.where(is_major.unsqueeze(-1), p1_major, prior_IRI))
        b_a_CCI = torch.where(is_minor.unsqueeze(-1), p2_minor,
                  torch.where(is_major.unsqueeze(-1), p2_major, prior_CCI))
        b_a_deck = torch.where(is_minor_deck.unsqueeze(-1), p_deck_minor_pre,
                   torch.where(is_major_deck.unsqueeze(-1), p_deck_major_pre, belief.deck))
        b_a = BeliefStateBatch(b_a_IRI, b_a_CCI, b_a_deck)
        
        obs = {
            'IRI': obs_IRI,
            'CCI': obs_CCI,
            'deck': obs_deck
        }
        
        return b_prime, b_a, extra_state_prime, obs
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def create_initial_belief(self, condition: str = 'excellent') -> BeliefStateBatch:
        """Create initial belief state for batch
        
        Args:
            condition: 'excellent' (all state 0), 'uniform' (uniform distribution), 
                      or 'legacy' (load from files matching legacy implementation)
        """
        if condition == 'excellent':
            IRI = torch.zeros(self.batch_size, NCOMP_PAV, NSTCOMP_IRI, 
                             dtype=self.dtype, device=self.device)
            IRI[:, :, 0] = 1.0
            CCI = torch.zeros(self.batch_size, NCOMP_PAV, NSTCOMP_CCI,
                             dtype=self.dtype, device=self.device)
            CCI[:, :, 0] = 1.0
            deck = torch.zeros(self.batch_size, NCOMP_DECK, NSTCOMP_DECK,
                              dtype=self.dtype, device=self.device)
            deck[:, :, 0] = 1.0
        elif condition == 'uniform':
            IRI = torch.ones(self.batch_size, NCOMP_PAV, NSTCOMP_IRI,
                            dtype=self.dtype, device=self.device) / NSTCOMP_IRI
            CCI = torch.ones(self.batch_size, NCOMP_PAV, NSTCOMP_CCI,
                            dtype=self.dtype, device=self.device) / NSTCOMP_CCI
            deck = torch.ones(self.batch_size, NCOMP_DECK, NSTCOMP_DECK,
                             dtype=self.dtype, device=self.device) / NSTCOMP_DECK
        elif condition == 'legacy':
            # Load initial states from files (matching legacy implementation)
            IRI, CCI, deck = self._load_initial_states_from_files()
        else:
            raise ValueError(f"Unknown condition: {condition}. Use 'excellent', 'uniform', or 'legacy'")
        
        return BeliefStateBatch(IRI, CCI, deck)
    
    def _load_initial_states_from_files(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load initial states from files matching legacy implementation.
        
        Returns:
            IRI: (batch_size, NCOMP_PAV, NSTCOMP_IRI) one-hot belief
            CCI: (batch_size, NCOMP_PAV, NSTCOMP_CCI) one-hot belief
            deck: (batch_size, NCOMP_DECK, NSTCOMP_DECK) one-hot belief
        """
        # Load state files (try Test_folder first, then DATA_DIR)
        import os as os_module
        
        # Try Test_folder first (where training runs)
        test_folder = DATA_DIR
        data_dir = DATA_DIR
        
        def load_state_file(filename):
            for base_dir in [test_folder, data_dir]:
                filepath = os_module.path.join(base_dir, filename)
                if os_module.path.exists(filepath):
                    return np.loadtxt(filepath).astype(int)
            raise FileNotFoundError(f"Could not find {filename} in {test_folder} or {data_dir}")
        
        # Load initial states
        cci_st = load_state_file('current_CCI_0_excellent.txt')
        iri_st = load_state_file('current_IRI_0_excellent.txt')
        deck_st = load_state_file('current_Deck_0_excellent.txt')
        
        # Convert to one-hot (matching legacy one_hot function)
        def one_hot(x, c):
            ncomp = len(x)
            x_hot = np.zeros((ncomp, c))
            for i in range(ncomp):
                x_hot[i][x[i]] = 1
            return x_hot
        
        IRI_np = one_hot(iri_st, NSTCOMP_IRI)  # (85, 5)
        CCI_np = one_hot(cci_st, NSTCOMP_CCI)  # (85, 6)
        deck_np = one_hot(deck_st, NSTCOMP_DECK)  # (11, 7)
        
        # Convert to tensors and expand for batch
        IRI = torch.tensor(IRI_np, dtype=self.dtype, device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1)
        CCI = torch.tensor(CCI_np, dtype=self.dtype, device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1)
        deck = torch.tensor(deck_np, dtype=self.dtype, device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1)
        
        return IRI, CCI, deck
    
    def load_initial_ages(self) -> torch.Tensor:
        """Load initial ages from file matching legacy implementation.
        
        Returns:
            ages: (batch_size, TOT_COMP) initial ages
        """
        import os as os_module
        
        # Try Test_folder first, then DATA_DIR
        test_folder = DATA_DIR
        data_dir = DATA_DIR
        
        for base_dir in [test_folder, data_dir]:
            filepath = os_module.path.join(base_dir, 'Current_age.txt')
            if os_module.path.exists(filepath):
                ages_np = np.loadtxt(filepath).astype(int)
                # Expand for batch
                ages = torch.tensor(ages_np, dtype=torch.long, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
                return ages
        
        raise FileNotFoundError(f"Could not find Current_age.txt in {test_folder} or {data_dir}")
    
    def age_action(self, ages: torch.Tensor, actions: torch.Tensor, 
                   comp_active: torch.Tensor) -> torch.Tensor:
        """Vectorized age update"""
        ages_prime = ages.clone()
        
        # Major repair: reduce age by 5
        major_mask = ((actions % 3) == 2) & comp_active
        ages_prime = torch.where(major_mask, (ages_prime - 5).clamp(min=0), ages_prime)
        
        # Replacement: set age to -1 (will be reset to 0 next step)
        replace_mask = (actions == 9) & comp_active
        ages_prime = torch.where(replace_mask, torch.full_like(ages_prime, -1), ages_prime)
        
        return ages_prime


# ============================================================================
# TESTING
# ============================================================================

def test_optimized_environment():
    """Test the optimized GPU environment"""
    print("\n" + "=" * 70)
    print("OPTIMIZED GPU ENVIRONMENT TEST")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    
    env = OptimizedGPUEnvironment(device=device, batch_size=batch_size)
    
    # Test 1: Replacement (deterministic)
    print("\n--- Test 1: Replacement Action ---")
    belief = env.create_initial_belief('uniform')
    actions = torch.full((batch_size, TOT_COMP), 9, dtype=torch.long, device=device)
    ages = torch.randint(0, MAX_AGE, (batch_size, TOT_COMP), device=device)
    extra_state = torch.zeros(batch_size, NCOMP_PAV, device=device)
    comp_active = torch.ones(batch_size, TOT_COMP, dtype=torch.bool, device=device)
    
    ages_prime = env.age_action(ages, actions, comp_active)
    # NOTE: comp_active removed from belief_update_vectorized call (infinite budget mode)
    # b_prime, b_a, extra_prime, obs = env.belief_update_vectorized(
    #     belief, actions, ages_prime, extra_state, comp_active
    # )
    b_prime, b_a, extra_prime, obs = env.belief_update_vectorized(
        belief, actions, ages_prime, extra_state
    )
    
    # Verify all in state 0
    iri_state0 = (b_prime.IRI[:, :, 0] == 1.0).all()
    cci_state0 = (b_prime.CCI[:, :, 0] == 1.0).all()
    deck_state0 = (b_prime.deck[:, :, 0] == 1.0).all()
    
    print(f"  IRI all state 0: {iri_state0.item()}")
    print(f"  CCI all state 0: {cci_state0.item()}")
    print(f"  Deck all state 0: {deck_state0.item()}")
    
    # Test 2: Performance benchmark with different batch sizes
    print("\n--- Test 2: Scalability Benchmark ---")
    batch_sizes = [1, 8, 32, 64, 128, 256, 512,1024]
    n_iterations = 100
    legacy_steps_per_sec = 650  # Baseline from legacy test
    
    results = []
    
    print(f"\n{'Batch':<8} {'Time/batch':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for bs in batch_sizes:
        env_test = OptimizedGPUEnvironment(device=device, batch_size=bs)
        
        belief = env_test.create_initial_belief('uniform')
        ages = torch.zeros(bs, TOT_COMP, dtype=torch.long, device=device)
        extra_state = torch.zeros(bs, NCOMP_PAV, device=device)
        
        # Warmup
        for _ in range(10):
            actions = torch.randint(0, NACOMP, (bs, TOT_COMP), device=device)
            comp_active = actions > 0  # Still needed for age_action
            ages_prime = env_test.age_action(ages, actions, comp_active)
            # NOTE: comp_active removed from belief_update_vectorized call (infinite budget mode)
            # b_prime, _, _, _ = env_test.belief_update_vectorized(
            #     belief, actions, ages_prime, extra_state, comp_active
            # )
            b_prime, _, _, _ = env_test.belief_update_vectorized(
                belief, actions, ages_prime, extra_state
            )
            belief = b_prime
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Timed run
        times = []
        for _ in range(n_iterations):
            actions = torch.randint(0, NACOMP, (bs, TOT_COMP), device=device)
            comp_active = actions > 0
            ages_prime = env_test.age_action(ages, actions, comp_active)
            
            t0 = time.time()
            # NOTE: comp_active removed from belief_update_vectorized call (infinite budget mode)
            # b_prime, _, _, _ = env_test.belief_update_vectorized(
            #     belief, actions, ages_prime, extra_state, comp_active
            # )
            b_prime, _, _, _ = env_test.belief_update_vectorized(
                belief, actions, ages_prime, extra_state
            )
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
            
            belief = b_prime
            ages = (ages + 1).clamp(max=MAX_AGE - 1)
        
        times = np.array(times)
        throughput = bs * 1000 / times.mean()
        speedup = throughput / legacy_steps_per_sec
        
        results.append({
            'batch': bs,
            'time_ms': times.mean(),
            'throughput': throughput,
            'speedup': speedup
        })
        
        print(f"{bs:<8} {times.mean():.3f} ms      {throughput:>10.0f}/s    {speedup:>6.1f}x")
    
    # Memory usage comparison
    print("\n--- Memory Comparison ---")
    print(f"  Memory reduction factor: ~{env._compute_memory_savings():.1f}x")
    
    if device.startswith('cuda'):
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU memory allocated: {mem_allocated:.1f} MB")
        print(f"  GPU memory reserved: {mem_reserved:.1f} MB")
    
    # Summary
    print("\n--- Summary ---")
    best = max(results, key=lambda x: x['throughput'])
    print(f"  Peak throughput: {best['throughput']:.0f} env-steps/sec (batch={best['batch']})")
    print(f"  Peak speedup vs legacy: {best['speedup']:.1f}x")
    
    return True


def compare_legacy_vs_optimized():
    """Detailed comparison between legacy and optimized implementations"""
    from test_legacy_vs_gpu import belief_update_legacy, age_action_legacy, create_initial_belief as create_legacy_belief
    
    print("\n" + "=" * 70)
    print("LEGACY vs OPTIMIZED GPU COMPARISON")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    env = OptimizedGPUEnvironment(device=device, batch_size=1)
    
    # Prepare identical inputs
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create initial beliefs
    b_legacy = create_legacy_belief('uniform')
    b_gpu = env.create_initial_belief('uniform')
    
    # All replacement test
    print("\n--- Replacement Test (Deterministic) ---")
    action = np.full((1, TOT_COMP), 9, dtype=int)
    action_gpu = torch.full((1, TOT_COMP), 9, dtype=torch.long, device=device)
    
    age = np.zeros((1, TOT_COMP), dtype=int)
    age_gpu = torch.zeros(1, TOT_COMP, dtype=torch.long, device=device)
    
    extra_state = np.zeros(NCOMP_PAV)
    extra_gpu = torch.zeros(1, NCOMP_PAV, device=device)
    
    comp_active = list(range(TOT_COMP))
    comp_active_gpu = torch.ones(1, TOT_COMP, dtype=torch.bool, device=device)
    
    age_prime = age_action_legacy(age, action, comp_active)
    age_prime_gpu = env.age_action(age_gpu, action_gpu, comp_active_gpu)
    
    # Run both
    b_prime_legacy, _, _, _ = belief_update_legacy(
        b_legacy, action, age_prime, extra_state, comp_active
    )
    
    # NOTE: comp_active removed from belief_update_vectorized call (infinite budget mode)
    # b_prime_gpu, _, _, _ = env.belief_update_vectorized(
    #     b_gpu, action_gpu, age_prime_gpu, extra_gpu, comp_active_gpu
    # )
    b_prime_gpu, _, _, _ = env.belief_update_vectorized(
        b_gpu, action_gpu, age_prime_gpu, extra_gpu
    )
    
    # Compare
    iri_diff = np.abs(b_prime_legacy.IRI - b_prime_gpu.IRI[0].cpu().numpy()).max()
    cci_diff = np.abs(b_prime_legacy.CCI - b_prime_gpu.CCI[0].cpu().numpy()).max()
    deck_diff = np.abs(b_prime_legacy.deck - b_prime_gpu.deck[0].cpu().numpy()).max()
    
    print(f"  IRI max diff:  {iri_diff:.2e}")
    print(f"  CCI max diff:  {cci_diff:.2e}")
    print(f"  Deck max diff: {deck_diff:.2e}")
    
    passed = max(iri_diff, cci_diff, deck_diff) < 1e-5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_single_component_all_actions():
    """
    Test component 0 with each action type (0-8), then test all components.
    
    Progressive testing:
    1. Component 0 with each action type individually
    2. All components with random actions
    """
    from test_legacy_vs_gpu import belief_update_legacy, age_action_legacy, belief
    
    print("\n" + "=" * 70)
    print("PROGRESSIVE DEBUG TEST")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    env = OptimizedGPUEnvironment(device=device, batch_size=1)
    
    tolerance = 1e-3
    comp_active = list(range(TOT_COMP))
    comp_active_gpu = torch.ones(1, TOT_COMP, dtype=torch.bool, device=device)
    
    # ================================================================
    # PHASE 1: Component 0 with each action type
    # ================================================================
    print("\n" + "-" * 50)
    print("PHASE 1: Component 0 with each action type (0-8)")
    print("-" * 50)
    
    action_names = {
        0: "Do-nothing (no insp)",
        1: "Minor repair (no insp)",
        2: "Major repair (no insp)",
        3: "Do-nothing + low insp",
        4: "Minor + low insp",
        5: "Major + low insp",
        6: "Do-nothing + high insp",
        7: "Minor + high insp",
        8: "Major + high insp",
    }
    
    phase1_passed = True
    
    for test_action in range(9):
        seed = 100 + test_action
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Uniform initial belief
        iri_legacy = np.ones((NCOMP_PAV, NSTCOMP_IRI)) / NSTCOMP_IRI
        cci_legacy = np.ones((NCOMP_PAV, NSTCOMP_CCI)) / NSTCOMP_CCI
        deck_legacy = np.ones((NCOMP_DECK, NSTCOMP_DECK)) / NSTCOMP_DECK
        
        b_legacy = belief()
        b_legacy.IRI = iri_legacy.copy()
        b_legacy.CCI = cci_legacy.copy()
        b_legacy.deck = deck_legacy.copy()
        
        b_gpu = BeliefStateBatch(
            IRI=torch.tensor(iri_legacy, dtype=torch.float32, device=device).unsqueeze(0),
            CCI=torch.tensor(cci_legacy, dtype=torch.float32, device=device).unsqueeze(0),
            deck=torch.tensor(deck_legacy, dtype=torch.float32, device=device).unsqueeze(0)
        )
        
        # All components get same action
        action_legacy = np.full((1, TOT_COMP), test_action, dtype=int)
        action_gpu = torch.tensor(action_legacy, dtype=torch.long, device=device)
        
        age_legacy = np.full((1, TOT_COMP), 5, dtype=int)
        age_gpu = torch.tensor(age_legacy, dtype=torch.long, device=device)
        
        extra_state_legacy = np.zeros(NCOMP_PAV, dtype=int)
        extra_gpu = torch.zeros(1, NCOMP_PAV, device=device)
        
        # Age update
        age_prime_legacy = age_action_legacy(age_legacy.copy(), action_legacy.copy(), comp_active)
        age_prime_gpu = env.age_action(age_gpu.clone(), action_gpu.clone(), comp_active_gpu)
        
        # Run both with same seed
        np.random.seed(seed + 1000)
        b_prime_legacy, _, _, obs_legacy = belief_update_legacy(
            b_legacy, action_legacy, age_prime_legacy, extra_state_legacy, comp_active
        )
        
        np.random.seed(seed + 1000)
        b_prime_gpu, _, _, obs_gpu = env.belief_update_vectorized(
            b_gpu, action_gpu, age_prime_gpu, extra_gpu, legacy_sampling=True
        )
        
        # Compare component 0
        iri_diff = np.abs(b_prime_legacy.IRI[0] - b_prime_gpu.IRI[0, 0].cpu().numpy()).max()
        cci_diff = np.abs(b_prime_legacy.CCI[0] - b_prime_gpu.CCI[0, 0].cpu().numpy()).max()
        obs_iri_match = obs_legacy.IRI[0] == obs_gpu['IRI'][0, 0].item()
        obs_cci_match = obs_legacy.CCI[0] == obs_gpu['CCI'][0, 0].item()
        
        passed = iri_diff < tolerance and cci_diff < tolerance
        status = "✓" if passed else "✗"
        
        print(f"\n  Action {test_action} ({action_names[test_action]}):")
        print(f"    IRI diff: {iri_diff:.2e}, CCI diff: {cci_diff:.2e} [{status}]")
        print(f"    obs.IRI: legacy={obs_legacy.IRI[0]}, gpu={obs_gpu['IRI'][0,0].item()} {'✓' if obs_iri_match else '✗'}")
        print(f"    obs.CCI: legacy={obs_legacy.CCI[0]}, gpu={obs_gpu['CCI'][0,0].item()} {'✓' if obs_cci_match else '✗'}")
        
        if not passed:
            phase1_passed = False
            print(f"    Legacy IRI[0]: {b_prime_legacy.IRI[0]}")
            print(f"    GPU IRI[0]:    {b_prime_gpu.IRI[0, 0].cpu().numpy()}")
            print(f"    Legacy CCI[0]: {b_prime_legacy.CCI[0]}")
            print(f"    GPU CCI[0]:    {b_prime_gpu.CCI[0, 0].cpu().numpy()}")
    
    print(f"\n  Phase 1 Result: {'PASS' if phase1_passed else 'FAIL'}")
    
    if not phase1_passed:
        print("\n  ✗ Phase 1 failed - fix component 0 before testing all components")
        return False
    
    # ================================================================
    # PHASE 2: All components with random actions
    # ================================================================
    print("\n" + "-" * 50)
    print("PHASE 2: All components with random actions")
    print("-" * 50)
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Random initial beliefs
    iri_legacy = np.random.rand(NCOMP_PAV, NSTCOMP_IRI)
    iri_legacy = iri_legacy / iri_legacy.sum(axis=1, keepdims=True)
    cci_legacy = np.random.rand(NCOMP_PAV, NSTCOMP_CCI)
    cci_legacy = cci_legacy / cci_legacy.sum(axis=1, keepdims=True)
    deck_legacy = np.random.rand(NCOMP_DECK, NSTCOMP_DECK)
    deck_legacy = deck_legacy / deck_legacy.sum(axis=1, keepdims=True)
    
    b_legacy = belief()
    b_legacy.IRI = iri_legacy.copy()
    b_legacy.CCI = cci_legacy.copy()
    b_legacy.deck = deck_legacy.copy()
    
    b_gpu = BeliefStateBatch(
        IRI=torch.tensor(iri_legacy, dtype=torch.float32, device=device).unsqueeze(0),
        CCI=torch.tensor(cci_legacy, dtype=torch.float32, device=device).unsqueeze(0),
        deck=torch.tensor(deck_legacy, dtype=torch.float32, device=device).unsqueeze(0)
    )
    
    # Random actions (0-8, no replacement)
    action_legacy = np.random.randint(0, 9, size=(1, TOT_COMP), dtype=int)
    action_gpu = torch.tensor(action_legacy, dtype=torch.long, device=device)
    
    # Random ages
    age_legacy = np.random.randint(0, MAX_AGE, size=(1, TOT_COMP), dtype=int)
    age_gpu = torch.tensor(age_legacy, dtype=torch.long, device=device)
    
    extra_state_legacy = np.zeros(NCOMP_PAV, dtype=int)
    extra_gpu = torch.zeros(1, NCOMP_PAV, device=device)
    
    print(f"\n  Actions (first 10): {action_legacy[0, :10]}")
    print(f"  Ages (first 10):    {age_legacy[0, :10]}")
    
    # Age update
    age_prime_legacy = age_action_legacy(age_legacy.copy(), action_legacy.copy(), comp_active)
    age_prime_gpu = env.age_action(age_gpu.clone(), action_gpu.clone(), comp_active_gpu)
    
    # Run both with same seed
    np.random.seed(seed + 180)
    b_prime_legacy, _, _, obs_legacy = belief_update_legacy(
        b_legacy, action_legacy, age_prime_legacy, extra_state_legacy, comp_active
    )
    
    np.random.seed(seed + 180)
    b_prime_gpu, _, _, obs_gpu = env.belief_update_vectorized(
        b_gpu, action_gpu, age_prime_gpu, extra_gpu, legacy_sampling=True
    )
    comp = 66
    print("beleif_gpu: ", b_gpu.CCI[0].cpu().numpy()[comp])
    print("beleif_legacy: ", b_legacy.CCI[comp])


    print(f"b_prime_gpu.CCI: {b_prime_gpu.CCI[0].cpu().numpy()[comp]}")
    print(f"b_prime_legacy.CCI: {b_prime_legacy.CCI[comp]}")
    print("observation_gpu: ", obs_gpu['CCI'][0, comp].item())
    print("observation_legacy: ", obs_legacy.CCI[comp])
    # Compare all pavement components
    iri_diffs = np.abs(b_prime_legacy.IRI - b_prime_gpu.IRI[0].cpu().numpy())
    cci_diffs = np.abs(b_prime_legacy.CCI - b_prime_gpu.CCI[0].cpu().numpy())
    deck_diffs = np.abs(b_prime_legacy.deck - b_prime_gpu.deck[0].cpu().numpy())
    print(f"cci_diffs: {cci_diffs[comp]}")
    max_iri_diff = iri_diffs.max()
    max_cci_diff = cci_diffs.max()
    max_deck_diff = deck_diffs.max()
    
    # Find worst components
    worst_iri_comp = np.unravel_index(iri_diffs.argmax(), iri_diffs.shape)[0]
    worst_cci_comp = np.unravel_index(cci_diffs.argmax(), cci_diffs.shape)[0]
    worst_deck_comp = np.unravel_index(deck_diffs.argmax(), deck_diffs.shape)[0]
    
    print(f"\n  Max IRI diff:  {max_iri_diff:.2e} (comp {worst_iri_comp}, action {action_legacy[0, worst_iri_comp]})")
    print(f"  Max CCI diff:  {max_cci_diff:.2e} (comp {worst_cci_comp}, action {action_legacy[0, worst_cci_comp]})")
    print(f"  Max Deck diff: {max_deck_diff:.2e} (comp {worst_deck_comp}, action {action_legacy[0, NCOMP_PAV + worst_deck_comp]})")
    
    phase2_passed = max_iri_diff < tolerance and max_cci_diff < tolerance and max_deck_diff < tolerance
    
    if not phase2_passed:
        # Show details for worst component
        print(f"\n  Worst IRI component {worst_iri_comp}:")
        print(f"    Action: {action_legacy[0, worst_iri_comp]}, Age: {age_legacy[0, worst_iri_comp]}")
        print(f"    Legacy: {b_prime_legacy.IRI[worst_iri_comp]}")
        print(f"    GPU:    {b_prime_gpu.IRI[0, worst_iri_comp].cpu().numpy()}")
        print(f"    obs: legacy={obs_legacy.IRI[worst_iri_comp]}, gpu={obs_gpu['IRI'][0, worst_iri_comp].item()}")
    
    print(f"\n  Phase 2 Result: {'PASS' if phase2_passed else 'FAIL'}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Phase 1 (comp 0, all actions): {'PASS' if phase1_passed else 'FAIL'}")
    print(f"  Phase 2 (all comps, random):   {'PASS' if phase2_passed else 'FAIL'}")
    
    all_passed = phase1_passed and phase2_passed
    if all_passed:
        print("\n✓ All tests PASSED!")
    else:
        print("\n✗ Some tests FAILED")
    
    return all_passed


if __name__ == '__main__':
    # Skip the scalability benchmark, just run the debug test
    print("Skipping scalability benchmark, running debug test only...")
    
    # Optional: detailed comparison with legacy
    try:
        compare_legacy_vs_optimized()
        
        # Progressive debug test
        print("\n" + "=" * 70)
        test_single_component_all_actions()
    except ImportError as e:
        print(f"\n(Skipping legacy comparison - test_legacy_vs_gpu.py not properly configured)")
        print(f"Error: {e}")
