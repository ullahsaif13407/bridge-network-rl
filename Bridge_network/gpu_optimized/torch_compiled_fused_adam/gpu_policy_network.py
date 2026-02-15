"""
GPU-Accelerated Policy Network for Bridge Network
==================================================

Training Paradigm:
- CTDE (Centralized Training, Decentralized Execution)
- On-Policy (PPO with GAE)
- Online Learning
- Constrained RL via Lagrangian relaxation

Architecture:
- Actor: 96 policy heads (one per component), parameter sharing within types
- Critic: Centralized value function on global state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
NCOMP_PAV = 85
NCOMP_DECK = 11
TOT_COMP = 96
NSTCOMP_IRI = 5
NSTCOMP_CCI = 6
NSTCOMP_DECK = 7
NACOMP = 10

INTRSTATE_COMP = 12
PRIMR_COMP = 47
SEC_COMP = 26


class ComponentPolicyHead(nn.Module):
    """
    Decentralized policy head for a single component.
    
    Each component observes:
    - Local belief state (IRI or deck)
    - Local CCI belief (for pavement)
    - Extra state (for pavement)
    - Age
    - Budget remaining (global)
    - Time step (global)
    
    Outputs action probabilities over 10 actions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NACOMP)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits"""
        return self.net(x)


class SharedPolicyEncoder(nn.Module):
    """
    Shared encoder for components of the same type.
    
    Component types:
    - Pavement Interstate (0-11): IRI(5) + CCI(6) + extra(1) + age(1) + time(1) = 14
    - Pavement Primary (12-58): Same as interstate
    - Pavement Secondary (59-84): Same as interstate
    - Deck (85-95): Deck(7) + age(1) + time(1) = 9
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BridgeNetworkActor(nn.Module):
    """
    CTDE Actor Network for the Bridge Network.
    
    Decentralized execution: Each component has local policy
    Centralized training: Shared parameters within component types
    
    Architecture:
    - Shared encoder per component type (4 types: Interstate, Primary, Secondary, Deck)
    - Individual policy heads (can be shared across types for efficiency)
    """
    
    def __init__(self, hidden_dim: int = 128, shared: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared = shared
        
        # Input dimensions (budget removed)
        self.pav_input_dim = NSTCOMP_IRI + NSTCOMP_CCI + 3  # 5+6+3 = 14 (IRI + CCI + extra + age + time)
        self.deck_input_dim = NSTCOMP_DECK + 2  # 7+2 = 9 (deck + age + time)
        
        if shared:
            # Shared encoders per type
            self.pav_encoder = SharedPolicyEncoder(self.pav_input_dim, hidden_dim, hidden_dim)
            self.deck_encoder = SharedPolicyEncoder(self.deck_input_dim, hidden_dim, hidden_dim)
            
            # Shared policy head
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, NACOMP)
            )
        else:
            # Separate networks for each component (original architecture)
            self.pav_nets = nn.ModuleList([
                ComponentPolicyHead(self.pav_input_dim, hidden_dim)
                for _ in range(NCOMP_PAV)
            ])
            self.deck_nets = nn.ModuleList([
                ComponentPolicyHead(self.deck_input_dim, hidden_dim)
                for _ in range(NCOMP_DECK)
            ])
    
    def forward(
        self,
        state_IRI: torch.Tensor,      # (batch, 85, 5)
        state_CCI: torch.Tensor,      # (batch, 85, 6)
        state_deck: torch.Tensor,     # (batch, 11, 7)
        extra_state: torch.Tensor,    # (batch, 85)
        age: torch.Tensor,            # (batch, 96)
        time_ratio: torch.Tensor,     # (batch, 1)
    ) -> torch.Tensor:
        """
        Forward pass returning action logits for all components.
        
        Returns:
            action_logits: (batch, 96, 10)
        """
        batch = state_IRI.shape[0]
        device = state_IRI.device
        
        # Prepare pavement inputs: (batch, 85, 14)
        pav_input = torch.cat([
            state_IRI,  # (batch, 85, 5)
            state_CCI,  # (batch, 85, 6)
            extra_state.unsqueeze(-1),  # (batch, 85, 1)
            age[:, :NCOMP_PAV].unsqueeze(-1) / 20.0,  # (batch, 85, 1)
            time_ratio.unsqueeze(1).expand(-1, NCOMP_PAV, -1),  # (batch, 85, 1)
        ], dim=-1)
        
        # Prepare deck inputs: (batch, 11, 9)
        deck_input = torch.cat([
            state_deck,  # (batch, 11, 7)
            age[:, NCOMP_PAV:].unsqueeze(-1) / 20.0,  # (batch, 11, 1)
            time_ratio.unsqueeze(1).expand(-1, NCOMP_DECK, -1),  # (batch, 11, 1)
        ], dim=-1)
        
        if self.shared:
            # Shared processing
            pav_encoded = self.pav_encoder(pav_input)  # (batch, 85, hidden)
            deck_encoded = self.deck_encoder(deck_input)  # (batch, 11, hidden)
            
            # Apply shared policy head
            pav_logits = self.policy_head(pav_encoded)  # (batch, 85, 10)
            deck_logits = self.policy_head(deck_encoded)  # (batch, 11, 10)
            
            action_logits = torch.cat([pav_logits, deck_logits], dim=1)  # (batch, 96, 10)
        else:
            # Per-component processing (slower but matches original)
            all_logits = []
            for i in range(NCOMP_PAV):
                logits = self.pav_nets[i](pav_input[:, i])  # (batch, 10)
                all_logits.append(logits)
            for i in range(NCOMP_DECK):
                logits = self.deck_nets[i](deck_input[:, i])  # (batch, 10)
                all_logits.append(logits)
            
            action_logits = torch.stack(all_logits, dim=1)  # (batch, 96, 10)
        
        return action_logits
    
    def get_action(
        self,
        state_IRI: torch.Tensor,
        state_CCI: torch.Tensor,
        state_deck: torch.Tensor,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy.
        
        Returns:
            actions: (batch, 96) - sampled actions
            log_probs: (batch, 96) - log probability of sampled actions
        """
        logits = self.forward(state_IRI, state_CCI, state_deck, extra_state, 
                             age, time_ratio)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def evaluate_actions(
        self,
        state_IRI: torch.Tensor,
        state_CCI: torch.Tensor,
        state_deck: torch.Tensor,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.
        
        Returns:
            log_probs: (batch, 96)
            entropy: (batch, 96)
        """
        logits = self.forward(state_IRI, state_CCI, state_deck, extra_state,
                             age, time_ratio)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class BridgeNetworkCritic(nn.Module):
    """
    Centralized Critic (Value Network) for the Bridge Network.
    
    Takes global state and estimates state value V(s).
    
    Input: Concatenated state
    - IRI beliefs: (batch, 85, 5) -> flattened 425
    - CCI beliefs: (batch, 85, 6) -> flattened 510
    - Deck beliefs: (batch, 11, 7) -> flattened 77
    - Extra state: (batch, 85) -> 85
    - Age: (batch, 96) -> 96
    - Time ratio: (batch, 1) -> 1
    Total: 1194
    """
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        
        # Calculate input dimension
        self.input_dim = (
            NCOMP_PAV * NSTCOMP_IRI +  # 425
            NCOMP_PAV * NSTCOMP_CCI +  # 510
            NCOMP_DECK * NSTCOMP_DECK +  # 77
            NCOMP_PAV +  # 85
            TOT_COMP +  # 96
            1  # time
        )  # Total: 1194
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        state_IRI: torch.Tensor,
        state_CCI: torch.Tensor,
        state_deck: torch.Tensor,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning state value.
        
        Returns:
            value: (batch, 1)
        """
        batch = state_IRI.shape[0]
        
        # Flatten and concatenate
        x = torch.cat([
            state_IRI.view(batch, -1),
            state_CCI.view(batch, -1),
            state_deck.view(batch, -1),
            extra_state,
            age / 20.0,
            time_ratio,
        ], dim=-1)
        
        return self.net(x)


class BridgeNetworkActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO training.
    """
    
    def __init__(self, actor_hidden: int = 128, critic_hidden: int = 512, shared_actor: bool = True):
        super().__init__()
        self.actor = BridgeNetworkActor(hidden_dim=actor_hidden, shared=shared_actor)
        self.critic = BridgeNetworkCritic(hidden_dim=critic_hidden)
    
    def forward(
        self,
        state_IRI: torch.Tensor,
        state_CCI: torch.Tensor,
        state_deck: torch.Tensor,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (batch, 96, 10)
            values: (batch, 1)
        """
        action_logits = self.actor(state_IRI, state_CCI, state_deck, extra_state,
                                   age, time_ratio)
        values = self.critic(state_IRI, state_CCI, state_deck, extra_state,
                            age, time_ratio)
        return action_logits, values
    
    def get_action_and_value(
        self,
        state_IRI: torch.Tensor,
        state_CCI: torch.Tensor,
        state_deck: torch.Tensor,
        extra_state: torch.Tensor,
        age: torch.Tensor,
        time_ratio: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions, log probs, entropy, and value in one forward pass.
        
        Returns:
            actions: (batch, 96)
            log_probs: (batch, 96)
            entropy: (batch, 96)
            values: (batch, 1)
        """
        action_logits, values = self.forward(
            state_IRI, state_CCI, state_deck, extra_state,
            age, time_ratio
        )
        
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy, values


# ============================================================================
# LAGRANGIAN CONSTRAINT HANDLING
# ============================================================================

class LagrangianMultipliers:
    """
    Manages Lagrangian multipliers for constrained optimization.
    
    Constraints:
    0: Interstate poor IRI & CCI < 5%
    1: Interstate very poor CCI < 2%
    2: Interstate+Primary poor CCI < 18%
    3: Interstate+Primary poor IRI < 15%
    4: Secondary poor CCI < 35%
    5: Deck poor (state > 4) < 10%
    """
    
    def __init__(
        self,
        n_constraints: int = 6,
        cap_alp: np.ndarray = None,
        lr_lambda: float = 0.001,
        device: str = 'cuda:0'
    ):
        self.n_constraints = n_constraints
        self.device = device
        
        if cap_alp is None:
            self.cap_alp = torch.tensor([5, 2, 18, 15, 35, 10], 
                                        dtype=torch.float32, device=device)
        else:
            self.cap_alp = torch.tensor(cap_alp, dtype=torch.float32, device=device)
        
        self.lr_lambda = lr_lambda
        self.lambdas = torch.zeros(n_constraints, dtype=torch.float32, device=device)
    
    def update(self, constraint_violations: torch.Tensor):
        """
        Update multipliers based on constraint violations.
        
        Args:
            constraint_violations: (n_constraints,) - percentage values
        """
        self.lambdas = torch.clamp(
            self.lambdas + self.lr_lambda * (constraint_violations - self.cap_alp),
            min=0.0
        )
    
    def get_penalty(self, constraint_violations: torch.Tensor) -> torch.Tensor:
        """
        Compute Lagrangian penalty term.
        
        Returns:
            penalty: scalar
        """
        return torch.sum(self.lambdas * (constraint_violations - self.cap_alp))
    
    def state_dict(self) -> Dict:
        return {
            'lambdas': self.lambdas.cpu().numpy(),
            'cap_alp': self.cap_alp.cpu().numpy(),
            'lr_lambda': self.lr_lambda
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.lambdas = torch.tensor(state_dict['lambdas'], 
                                   dtype=torch.float32, device=self.device)


# ============================================================================
# TEST
# ============================================================================

def test_policy_network():
    """Test the policy network architecture"""
    print("=" * 70)
    print("POLICY NETWORK TEST")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    
    # Create networks
    actor_critic = BridgeNetworkActorCritic(
        actor_hidden=128,
        critic_hidden=512,
        shared_actor=True
    ).to(device)
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor_critic.actor.parameters())
    critic_params = sum(p.numel() for p in actor_critic.critic.parameters())
    print(f"\nActor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Total parameters: {actor_params + critic_params:,}")
    
    # Create dummy inputs
    state_IRI = torch.rand(batch_size, NCOMP_PAV, NSTCOMP_IRI, device=device)
    state_CCI = torch.rand(batch_size, NCOMP_PAV, NSTCOMP_CCI, device=device)
    state_deck = torch.rand(batch_size, NCOMP_DECK, NSTCOMP_DECK, device=device)
    extra_state = torch.zeros(batch_size, NCOMP_PAV, device=device)
    age = torch.randint(0, 20, (batch_size, TOT_COMP), device=device).float()
    budget_ratio = torch.rand(batch_size, 1, device=device)
    time_ratio = torch.rand(batch_size, 1, device=device)
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    actions, log_probs, entropy, values = actor_critic.get_action_and_value(
        state_IRI, state_CCI, state_deck, extra_state,
        age, budget_ratio, time_ratio
    )
    
    print(f"Actions shape: {actions.shape}")  # (32, 96)
    print(f"Log probs shape: {log_probs.shape}")  # (32, 96)
    print(f"Entropy shape: {entropy.shape}")  # (32, 96)
    print(f"Values shape: {values.shape}")  # (32, 1)
    
    # Test inference speed
    print("\n--- Inference Speed Test ---")
    import time
    
    n_iters = 100
    torch.cuda.synchronize() if device.startswith('cuda') else None
    
    start = time.time()
    for _ in range(n_iters):
        actions, log_probs, entropy, values = actor_critic.get_action_and_value(
            state_IRI, state_CCI, state_deck, extra_state,
            age, budget_ratio, time_ratio
        )
    torch.cuda.synchronize() if device.startswith('cuda') else None
    elapsed = time.time() - start
    
    print(f"Time per batch: {elapsed/n_iters*1000:.2f} ms")
    print(f"Throughput: {batch_size * n_iters / elapsed:.0f} samples/sec")
    
    # Test Lagrangian multipliers
    print("\n--- Lagrangian Multipliers Test ---")
    lagrangian = LagrangianMultipliers(device=device)
    
    # Simulate constraint violations
    violations = torch.tensor([6, 1, 20, 12, 30, 8], dtype=torch.float32, device=device)
    
    penalty_before = lagrangian.get_penalty(violations)
    print(f"Initial penalty: {penalty_before.item():.4f}")
    
    for _ in range(100):
        lagrangian.update(violations)
    
    penalty_after = lagrangian.get_penalty(violations)
    print(f"Penalty after 100 updates: {penalty_after.item():.4f}")
    print(f"Lambda values: {lagrangian.lambdas.cpu().numpy()}")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == '__main__':
    test_policy_network()

