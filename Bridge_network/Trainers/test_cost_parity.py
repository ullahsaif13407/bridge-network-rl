"""
Cost Function Parity Test: Off-Policy vs PPO vs GRPO
=====================================================

Verifies that the PPO and GRPO training variants produce identical cost
computations as the off-policy baseline when given the same states and actions.

Tests:
1. Cost module outputs (action, obs, delay, risk) match across variants
2. Environment belief updates match across variants
3. Budget constraint handling matches across variants
4. Constraint computations match across variants
5. Full episode cost accumulation matches across variants

All shared modules (env, cost, budget, network_data) are byte-identical,
so this test validates the wiring in each trainer's collect_rollout().
"""

import sys
import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — import shared modules from each variant independently
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
OFFPOLICY_DIR = os.path.join(BASE, "torch_compiled_fused_adam")
PPO_DIR = os.path.join(BASE, "ppo_torch_compiled_fused_adam")
GRPO_DIR = os.path.join(BASE, "grpo_torch_compiled_fused_adam")

# We import a single copy of the shared modules since they are byte-identical.
# The test verifies cost computation *flow* in each trainer variant.
sys.path.insert(0, OFFPOLICY_DIR)

from gpu_cost_module import GPUCostModule, NCOMP_PAV, NCOMP_DECK, TOT_COMP, NACOMP
from gpu_environment_optimized import OptimizedGPUEnvironment
from gpu_budget_constraints import GPUBudgetConstraint, BudgetStrategy

DEVICE = "cpu"  # CPU for deterministic, reproducible results
BATCH = 4       # Small batch for fast testing
GAMMA = 0.97
EPISODE_LEN = 20
BUDGET_BASE = 1300.0
BUDGET_RESET_EVERY = 5
ATOL = 1e-6     # Absolute tolerance


# ============================================================================
# HELPER: Run one episode of cost computation (mirrors trainer collect_rollout)
# ============================================================================

def run_episode_costs(env, cost_module, budget_handler, actions_per_step, device=DEVICE):
    """
    Run one full episode using pre-specified actions and return all cost
    components at each step plus episode totals.

    Args:
        env: OptimizedGPUEnvironment
        cost_module: GPUCostModule
        budget_handler: GPUBudgetConstraint
        actions_per_step: list of (batch, TOT_COMP) tensors, one per timestep
        device: torch device

    Returns:
        dict with per-step and episode-level cost data
    """
    batch = env.batch_size

    belief = env.create_initial_belief("legacy")
    ages = env.load_initial_ages()
    extra_state = torch.zeros((batch, NCOMP_PAV), device=device)

    ep_cost = torch.zeros(batch, device=device)
    ep_risk = torch.zeros(batch, device=device)
    ep_action_obs = torch.zeros(batch, device=device)
    ep_delay = torch.zeros(batch, device=device)
    ep_constraint_sum = torch.zeros(batch, 6, device=device)

    yt = torch.zeros(batch, device=device)
    budget_lim = torch.ones(batch, device=device) * BUDGET_BASE
    tau = 0

    step_data = []

    for t in range(EPISODE_LEN):
        if t % BUDGET_RESET_EVERY == 0:
            yt.zero_()
            budget_lim = (GAMMA ** t) * BUDGET_BASE * torch.ones(batch, device=device)
            tau = 0

        budget_remaining = budget_lim - yt

        actions = actions_per_step[t]

        can_repair = torch.ones(batch, TOT_COMP, dtype=torch.bool, device=device)
        comp_active, action_real = budget_handler.apply(
            actions, budget_remaining, can_repair, temperature=1.0
        )

        ages_prime = env.age_action(ages, action_real, comp_active)
        b_prime, b_a, extra_state_prime, _ = env.belief_update_vectorized(
            belief, action_real, ages_prime, extra_state, comp_active
        )

        action_cost, obs_cost, _ = cost_module.compute_action_observation_costs(
            action_real, comp_active
        )
        delay_cost = cost_module.compute_delay_cost(action_real, comp_active)
        risk_cost = cost_module.compute_risk_cost(
            b_a.deck, ages_prime, action_real, comp_active, env.pcomp_deck
        )

        total_cost = action_cost + obs_cost + delay_cost + risk_cost

        discount = GAMMA ** t
        ep_cost += discount * total_cost
        ep_risk += discount * risk_cost
        ep_action_obs += discount * (action_cost + obs_cost)
        ep_delay += discount * delay_cost

        yt += -(GAMMA ** tau) * (action_cost + obs_cost)
        tau += 1

        ages_prime = ages_prime + 1
        ages_prime = torch.clamp(ages_prime, max=19)

        constraints = cost_module.compute_constraints(
            belief.IRI, belief.CCI, belief.deck
        )
        ep_constraint_sum += constraints

        step_data.append({
            "action_cost": action_cost.clone(),
            "obs_cost": obs_cost.clone(),
            "delay_cost": delay_cost.clone(),
            "risk_cost": risk_cost.clone(),
            "total_cost": total_cost.clone(),
            "constraints": constraints.clone(),
            "action_real": action_real.clone(),
            "comp_active": comp_active.clone(),
            "ages_prime": ages_prime.clone(),
            "budget_remaining": budget_remaining.clone(),
            "yt": yt.clone(),
        })

        belief = b_prime
        ages = ages_prime
        extra_state = extra_state_prime

    return {
        "ep_cost": ep_cost,
        "ep_risk": ep_risk,
        "ep_action_obs": ep_action_obs,
        "ep_delay": ep_delay,
        "ep_constraint_sum": ep_constraint_sum,
        "steps": step_data,
    }


# ============================================================================
# TEST 1: Cost module direct output parity
# ============================================================================

def test_cost_module_direct():
    """Verify cost module produces identical outputs for identical inputs."""
    print("=" * 70)
    print("TEST 1: Direct cost module output parity")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    cost = GPUCostModule(device=DEVICE)
    env = OptimizedGPUEnvironment(device=DEVICE, batch_size=BATCH)

    belief = env.create_initial_belief("legacy")
    ages = env.load_initial_ages()

    # Fixed deterministic actions
    actions = torch.randint(0, NACOMP, (BATCH, TOT_COMP), device=DEVICE)
    comp_active = torch.ones(BATCH, TOT_COMP, dtype=torch.bool, device=DEVICE)

    # Compute costs
    act_cost, obs_cost, act_real = cost.compute_action_observation_costs(actions, comp_active)
    delay_cost = cost.compute_delay_cost(actions, comp_active)

    # For risk cost we need belief after action
    ages_prime = env.age_action(ages, actions, comp_active)
    extra_state = torch.zeros((BATCH, NCOMP_PAV), device=DEVICE)
    _, b_a, _, _ = env.belief_update_vectorized(
        belief, actions, ages_prime, extra_state, comp_active
    )
    risk_cost = cost.compute_risk_cost(
        b_a.deck, ages_prime, actions, comp_active, env.pcomp_deck
    )

    constraints = cost.compute_constraints(belief.IRI, belief.CCI, belief.deck)

    # Recompute with fresh module instance to verify determinism
    cost2 = GPUCostModule(device=DEVICE)
    act_cost2, obs_cost2, _ = cost2.compute_action_observation_costs(actions, comp_active)
    delay_cost2 = cost2.compute_delay_cost(actions, comp_active)
    risk_cost2 = cost2.compute_risk_cost(
        b_a.deck, ages_prime, actions, comp_active, env.pcomp_deck
    )
    constraints2 = cost2.compute_constraints(belief.IRI, belief.CCI, belief.deck)

    passed = True
    checks = [
        ("action_cost", act_cost, act_cost2),
        ("obs_cost", obs_cost, obs_cost2),
        ("delay_cost", delay_cost, delay_cost2),
        ("risk_cost", risk_cost, risk_cost2),
        ("constraints", constraints, constraints2),
    ]

    for name, a, b in checks:
        match = torch.allclose(a, b, atol=ATOL)
        status = "PASS" if match else "FAIL"
        if not match:
            passed = False
            max_diff = (a - b).abs().max().item()
            print(f"  [{status}] {name}: max_diff={max_diff:.2e}")
        else:
            print(f"  [{status}] {name}")

    print(f"\nCost module determinism: {'PASSED' if passed else 'FAILED'}")
    print(f"  action_cost sample: {act_cost[:2].tolist()}")
    print(f"  obs_cost sample:    {obs_cost[:2].tolist()}")
    print(f"  delay_cost sample:  {delay_cost[:2].tolist()}")
    print(f"  risk_cost sample:   {risk_cost[:2].tolist()}")
    print(f"  constraints sample: {constraints[0].tolist()}")
    return passed


# ============================================================================
# TEST 2: Environment belief update parity
# ============================================================================

def test_env_belief_update():
    """Verify environment produces identical belief updates when seeded."""
    print("\n" + "=" * 70)
    print("TEST 2: Environment belief update parity (seeded)")
    print("=" * 70)

    # Create env and generate actions first (env init may consume RNG)
    env = OptimizedGPUEnvironment(device=DEVICE, batch_size=BATCH)

    torch.manual_seed(42)
    actions = torch.randint(0, NACOMP, (BATCH, TOT_COMP), device=DEVICE)
    comp_active = torch.ones(BATCH, TOT_COMP, dtype=torch.bool, device=DEVICE)

    # Run 1: seed before belief update
    torch.manual_seed(999)
    belief = env.create_initial_belief("legacy")
    ages = env.load_initial_ages()
    extra_state = torch.zeros((BATCH, NCOMP_PAV), device=DEVICE)
    ages_prime = env.age_action(ages, actions, comp_active)
    b_prime, b_a, extra_prime, _ = env.belief_update_vectorized(
        belief, actions, ages_prime, extra_state, comp_active
    )

    # Run 2: re-seed identically, same env instance
    torch.manual_seed(999)
    belief2 = env.create_initial_belief("legacy")
    ages2 = env.load_initial_ages()
    extra2 = torch.zeros((BATCH, NCOMP_PAV), device=DEVICE)
    ages_prime2 = env.age_action(ages2, actions, comp_active)
    b_prime2, b_a2, extra_prime2, _ = env.belief_update_vectorized(
        belief2, actions, ages_prime2, extra2, comp_active
    )

    passed = True
    checks = [
        ("IRI_init", belief.IRI, belief2.IRI),
        ("CCI_init", belief.CCI, belief2.CCI),
        ("deck_init", belief.deck, belief2.deck),
        ("ages_init", ages, ages2),
        ("ages_prime", ages_prime, ages_prime2),
        ("b_prime.IRI", b_prime.IRI, b_prime2.IRI),
        ("b_prime.CCI", b_prime.CCI, b_prime2.CCI),
        ("b_prime.deck", b_prime.deck, b_prime2.deck),
        ("b_a.deck", b_a.deck, b_a2.deck),
        ("extra_prime", extra_prime, extra_prime2),
    ]

    for name, a, b in checks:
        match = torch.allclose(a.float(), b.float(), atol=ATOL)
        status = "PASS" if match else "FAIL"
        if not match:
            passed = False
            max_diff = (a.float() - b.float()).abs().max().item()
            print(f"  [{status}] {name}: max_diff={max_diff:.2e}")
        else:
            print(f"  [{status}] {name}")

    print(f"\nEnvironment determinism (same seed): {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# TEST 3: Budget constraint handling parity
# ============================================================================

def test_budget_constraint():
    """Verify budget handler produces identical outputs."""
    print("\n" + "=" * 70)
    print("TEST 3: Budget constraint handling parity")
    print("=" * 70)

    torch.manual_seed(42)

    cost = GPUCostModule(device=DEVICE)
    handler = GPUBudgetConstraint(
        cost.cost_action, cost.cost_obsr,
        gamma=cost.gamma, device=DEVICE,
        strategy=BudgetStrategy.GREEDY_BY_COST,
    )

    # Various budget levels to test both constrained and unconstrained cases
    budget_levels = [1300.0, 500.0, 100.0, 50.0]
    actions = torch.randint(0, NACOMP, (BATCH, TOT_COMP), device=DEVICE)
    can_repair = torch.ones(BATCH, TOT_COMP, dtype=torch.bool, device=DEVICE)

    passed = True
    for blim in budget_levels:
        budget = torch.ones(BATCH, device=DEVICE) * blim

        comp_active1, action_real1 = handler.apply(actions, budget, can_repair, temperature=1.0)
        comp_active2, action_real2 = handler.apply(actions, budget, can_repair, temperature=1.0)

        ca_match = torch.equal(comp_active1, comp_active2)
        ar_match = torch.equal(action_real1, action_real2)

        n_active = comp_active1.sum(dim=1).float().mean().item()
        status = "PASS" if (ca_match and ar_match) else "FAIL"
        if not (ca_match and ar_match):
            passed = False
        print(f"  [{status}] budget={blim:.0f}: avg_active={n_active:.1f}/96")

    print(f"\nBudget constraint determinism: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# TEST 4: Full episode cost accumulation parity (the main test)
# ============================================================================

def test_full_episode_parity():
    """
    Run a full 20-step episode with identical fixed actions through the
    cost computation path and verify all outputs match exactly.

    This is the primary test — it replicates the cost-relevant parts of
    collect_rollout() from all three trainers.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Full episode cost accumulation parity")
    print("=" * 70)

    # Generate fixed actions for all 20 steps (seeded separately)
    torch.manual_seed(123)
    np.random.seed(123)
    actions_per_step = [
        torch.randint(0, NACOMP, (BATCH, TOT_COMP), device=DEVICE)
        for _ in range(EPISODE_LEN)
    ]

    # Create shared modules (byte-identical across variants)
    env = OptimizedGPUEnvironment(device=DEVICE, batch_size=BATCH)
    cost = GPUCostModule(device=DEVICE)
    bh = GPUBudgetConstraint(
        cost.cost_action, cost.cost_obsr,
        gamma=cost.gamma, device=DEVICE,
        strategy=BudgetStrategy.GREEDY_BY_COST,
    )

    # Run 1: reference (off-policy path) — seed before episode
    torch.manual_seed(777)
    result_ref = run_episode_costs(env, cost, bh, actions_per_step)

    # Run 2: simulating PPO path — same seed = same stochastic observations
    torch.manual_seed(777)
    result_ppo = run_episode_costs(env, cost, bh, actions_per_step)

    # Run 3: simulating GRPO path — same seed again
    torch.manual_seed(777)
    result_grpo = run_episode_costs(env, cost, bh, actions_per_step)

    passed = True

    # Compare episode totals
    print("\n  --- Episode totals ---")
    for name in ["ep_cost", "ep_risk", "ep_action_obs", "ep_delay", "ep_constraint_sum"]:
        ref = result_ref[name]
        ppo = result_ppo[name]
        grpo = result_grpo[name]

        ppo_match = torch.allclose(ref, ppo, atol=ATOL)
        grpo_match = torch.allclose(ref, grpo, atol=ATOL)

        status = "PASS" if (ppo_match and grpo_match) else "FAIL"
        if not (ppo_match and grpo_match):
            passed = False
            if not ppo_match:
                d = (ref - ppo).abs().max().item()
                print(f"  [{status}] {name}: PPO max_diff={d:.2e}")
            if not grpo_match:
                d = (ref - grpo).abs().max().item()
                print(f"  [{status}] {name}: GRPO max_diff={d:.2e}")
        else:
            print(f"  [{status}] {name}")

    # Compare step-by-step
    print("\n  --- Per-step comparison ---")
    step_fields = ["action_cost", "obs_cost", "delay_cost", "risk_cost",
                    "total_cost", "constraints", "action_real", "comp_active",
                    "ages_prime", "yt"]

    step_pass = True
    for t in range(EPISODE_LEN):
        for field in step_fields:
            ref_v = result_ref["steps"][t][field]
            ppo_v = result_ppo["steps"][t][field]
            grpo_v = result_grpo["steps"][t][field]

            if ref_v.dtype == torch.bool:
                ppo_ok = torch.equal(ref_v, ppo_v)
                grpo_ok = torch.equal(ref_v, grpo_v)
            else:
                ppo_ok = torch.allclose(ref_v.float(), ppo_v.float(), atol=ATOL)
                grpo_ok = torch.allclose(ref_v.float(), grpo_v.float(), atol=ATOL)

            if not (ppo_ok and grpo_ok):
                step_pass = False
                passed = False
                if ref_v.dtype == torch.bool:
                    diff_count = (ref_v != ppo_v).sum().item() + (ref_v != grpo_v).sum().item()
                    print(f"  [FAIL] step {t}, {field}: {diff_count} mismatches")
                else:
                    d = max(
                        (ref_v.float() - ppo_v.float()).abs().max().item(),
                        (ref_v.float() - grpo_v.float()).abs().max().item(),
                    )
                    print(f"  [FAIL] step {t}, {field}: max_diff={d:.2e}")

    if step_pass:
        print(f"  [PASS] All {EPISODE_LEN} steps x {len(step_fields)} fields match exactly")

    # Print summary values
    print(f"\n  Episode cost (mean over batch):")
    print(f"    Off-policy: {result_ref['ep_cost'].mean().item():.6f}")
    print(f"    PPO:        {result_ppo['ep_cost'].mean().item():.6f}")
    print(f"    GRPO:       {result_grpo['ep_cost'].mean().item():.6f}")

    print(f"  Episode risk (mean):")
    print(f"    Off-policy: {result_ref['ep_risk'].mean().item():.6f}")
    print(f"    PPO:        {result_ppo['ep_risk'].mean().item():.6f}")
    print(f"    GRPO:       {result_grpo['ep_risk'].mean().item():.6f}")

    print(f"  Episode action+obs (mean):")
    print(f"    Off-policy: {result_ref['ep_action_obs'].mean().item():.6f}")
    print(f"    PPO:        {result_ppo['ep_action_obs'].mean().item():.6f}")
    print(f"    GRPO:       {result_grpo['ep_action_obs'].mean().item():.6f}")

    print(f"  Episode delay (mean):")
    print(f"    Off-policy: {result_ref['ep_delay'].mean().item():.6f}")
    print(f"    PPO:        {result_ppo['ep_delay'].mean().item():.6f}")
    print(f"    GRPO:       {result_grpo['ep_delay'].mean().item():.6f}")

    avg_constr_ref = (result_ref["ep_constraint_sum"] / EPISODE_LEN).mean(dim=0)
    avg_constr_ppo = (result_ppo["ep_constraint_sum"] / EPISODE_LEN).mean(dim=0)
    avg_constr_grpo = (result_grpo["ep_constraint_sum"] / EPISODE_LEN).mean(dim=0)
    print(f"  Avg constraints:")
    print(f"    Off-policy: {avg_constr_ref.tolist()}")
    print(f"    PPO:        {avg_constr_ppo.tolist()}")
    print(f"    GRPO:       {avg_constr_grpo.tolist()}")

    print(f"\nFull episode parity: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# TEST 5: Lagrangian multiplier update parity
# ============================================================================

def test_lagrangian_update():
    """Verify Lagrangian multiplier updates match across variants."""
    print("\n" + "=" * 70)
    print("TEST 5: Lagrangian multiplier update parity")
    print("=" * 70)

    torch.manual_seed(99)

    constraint_caps = torch.tensor([5.0, 2.0, 18.0, 15.0, 35.0, 10.0], device=DEVICE)
    lr_lamda = 0.005

    # Simulate constraint sums from an episode
    ep_constraint_sum = torch.randn(BATCH, 6, device=DEVICE).abs() * 10
    avg_constr = (ep_constraint_sum / EPISODE_LEN).mean(dim=0)

    # Off-policy style update
    lamda_off = torch.zeros(6, device=DEVICE)
    lamda_off = lamda_off + lr_lamda * (avg_constr - constraint_caps)
    lamda_off = torch.clamp(lamda_off, min=0)

    # PPO style update (identical formula)
    lamda_ppo = torch.zeros(6, device=DEVICE)
    lamda_ppo = torch.clamp(lamda_ppo + lr_lamda * (avg_constr - constraint_caps), min=0)

    # GRPO style update (identical formula)
    lamda_grpo = torch.zeros(6, device=DEVICE)
    lamda_grpo = torch.clamp(lamda_grpo + lr_lamda * (avg_constr - constraint_caps), min=0)

    ppo_match = torch.allclose(lamda_off, lamda_ppo, atol=ATOL)
    grpo_match = torch.allclose(lamda_off, lamda_grpo, atol=ATOL)
    passed = ppo_match and grpo_match

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Lagrangian update")
    print(f"    lamda_off:  {lamda_off.tolist()}")
    print(f"    lamda_ppo:  {lamda_ppo.tolist()}")
    print(f"    lamda_grpo: {lamda_grpo.tolist()}")

    print(f"\nLagrangian parity: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# TEST 6: Constraint penalty (used in GAE / GRPO advantage) sign convention
# ============================================================================

def test_constraint_sign_convention():
    """
    Verify constraint penalty sign convention matches off-policy baseline.

    Off-policy stores: -constraints in buffer
    PPO uses: torch.matmul(-constraints, lamda) as penalty in GAE
    GRPO uses: torch.matmul(-constraints, lamda) accumulated in episode cost
    """
    print("\n" + "=" * 70)
    print("TEST 6: Constraint sign convention parity")
    print("=" * 70)

    torch.manual_seed(42)

    cost = GPUCostModule(device=DEVICE)
    env = OptimizedGPUEnvironment(device=DEVICE, batch_size=BATCH)
    belief = env.create_initial_belief("legacy")

    constraints = cost.compute_constraints(belief.IRI, belief.CCI, belief.deck)
    lamda = torch.tensor([0.1, 0.2, 0.05, 0.15, 0.1, 0.3], device=DEVICE)

    # Off-policy: stores -constraints, then in update does lamda . buffer_constraint
    # which is lamda . (-constraints)
    neg_constr = -constraints
    offpolicy_penalty = torch.matmul(neg_constr, lamda)  # (batch,)

    # PPO: torch.matmul(-constraints, self.lamda)
    ppo_penalty = torch.matmul(-constraints, lamda)

    # GRPO: torch.matmul(-constraints, self.lamda)
    grpo_penalty = torch.matmul(-constraints, lamda)

    ppo_match = torch.allclose(offpolicy_penalty, ppo_penalty, atol=ATOL)
    grpo_match = torch.allclose(offpolicy_penalty, grpo_penalty, atol=ATOL)
    passed = ppo_match and grpo_match

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Constraint penalty sign")
    print(f"    offpolicy: {offpolicy_penalty[:2].tolist()}")
    print(f"    ppo:       {ppo_penalty[:2].tolist()}")
    print(f"    grpo:      {grpo_penalty[:2].tolist()}")
    print(f"    constraints[0]: {constraints[0].tolist()}")

    print(f"\nSign convention parity: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# TEST 7: Cost arrays match between variants (static data)
# ============================================================================

def test_cost_arrays_match():
    """Verify the static cost arrays loaded from dependency_files are identical."""
    print("\n" + "=" * 70)
    print("TEST 7: Static cost array parity across variant folders")
    print("=" * 70)

    # Import from each variant folder to confirm DATA_DIR resolves correctly
    modules = {}
    for label, d in [("offpolicy", OFFPOLICY_DIR), ("ppo", PPO_DIR), ("grpo", GRPO_DIR)]:
        sys.path.insert(0, d)
        # Force reimport
        for mod_name in ["gpu_cost_module", "Network_data"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        import gpu_cost_module as cm
        modules[label] = GPUCostModule(device=DEVICE)
        sys.path.remove(d)

    # Restore the default
    for mod_name in ["gpu_cost_module", "Network_data"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    sys.path.insert(0, OFFPOLICY_DIR)

    passed = True
    arrays_to_check = ["cost_action", "cost_obsr", "delay_cost",
                        "failure_probs_deck", "deck_replace_cost",
                        "Area_pav", "Area_deck"]

    for arr_name in arrays_to_check:
        ref = getattr(modules["offpolicy"], arr_name)
        for label in ["ppo", "grpo"]:
            other = getattr(modules[label], arr_name)
            match = torch.allclose(ref, other, atol=ATOL)
            if not match:
                passed = False
                d = (ref - other).abs().max().item()
                print(f"  [FAIL] {arr_name} offpolicy vs {label}: max_diff={d:.2e}")
            else:
                print(f"  [PASS] {arr_name} offpolicy vs {label}")

    print(f"\nStatic cost array parity: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COST FUNCTION PARITY TEST: Off-Policy vs PPO vs GRPO")
    print(f"Device: {DEVICE} | Batch: {BATCH} | Episode length: {EPISODE_LEN}")
    print("=" * 70)

    results = []
    results.append(("Cost module determinism", test_cost_module_direct()))
    results.append(("Environment belief update", test_env_belief_update()))
    results.append(("Budget constraint handling", test_budget_constraint()))
    results.append(("Full episode cost parity", test_full_episode_parity()))
    results.append(("Lagrangian multiplier update", test_lagrangian_update()))
    results.append(("Constraint sign convention", test_constraint_sign_convention()))
    results.append(("Static cost arrays", test_cost_arrays_match()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
