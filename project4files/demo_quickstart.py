#!/usr/bin/env python3
"""
Self-Contained Demo: Multi-Algorithm Quantile Bandit Comparison
================================================================

This demo shows how to use the Project 4 framework to compare different
bandit algorithms with quantile regression updates.

Runtime: ~5-10 minutes
Requirements: See ../requirements.txt

Usage:
    cd project4files
    python demo_standalone.py

What you'll learn:
    1. How to set up a simulation with multiple algorithms
    2. How to configure different beta/error distributions
    3. How to run the simulation and analyze results
    4. How to create comparison visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from project4_simulation import Project4Simulation
from generators import NormalGenerator, TGenerator, TruncatedNormalGenerator

# Configuration
np.random.seed(42)  # For reproducibility
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

print("=" * 80)
print(" PROJECT 4 DEMO: Multi-Algorithm Quantile Bandit Comparison")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Understanding the Problem Setup
# ============================================================================
print("STEP 1: Understanding the Problem Setup")
print("-" * 80)
print("""
We have a contextual bandit problem where:
- At each time t, we observe context X_t (d-dimensional vector)
- We choose arm k from K available arms
- We receive reward: Y_t = X_t^T * β_k + ε_t
- Goal: Maximize cumulative reward (minimize regret)

Key Challenge: Errors ε_t are heavy-tailed (Student-t with df=2.25)
Solution: Use quantile regression instead of OLS for robust estimation
""")

input("Press Enter to continue...")
print()

# ============================================================================
# STEP 2: Setting Up the Simulation
# ============================================================================
print("STEP 2: Setting Up the Simulation")
print("-" * 80)
print("""
Configuration:
- Number of arms (K): 2
- Context dimension (d): 10
- Time horizon (T): 100
- Number of replications (n_sim): 15
- Quantile level (τ): 0.5 (median regression)

Algorithms to compare:
1. ForcedSampling - Baseline from Projects 2-3
2. LinUCB - UCB-style exploration with quantile regression
3. EpsilonGreedy - Simple ε-greedy with quantile regression
4. ThompsonSampling - Bayesian posterior sampling with quantile regression
""")

# Create the simulation
print("\nCreating simulation study...")

study = Project4Simulation(
    n_sim=15,                    # 15 simulation replications
    K=2,                         # 2 arms
    d=10,                        # 10-dimensional context
    T=100,                       # 100 time steps
    tau=0.5,                     # Median regression
    algorithms='all',            # Compare all 4 algorithms
    beta_generator=NormalGenerator(mean=0, std=1),  # Gaussian coefficients
    err_generator=TGenerator(df=2.25, scale=0.7),   # Heavy-tailed errors
    context_generator=TruncatedNormalGenerator(mean=0, std=1),
    random_seed=42              # For reproducibility
)

print("✓ Simulation created successfully!")
print()

input("Press Enter to continue...")
print()

# ============================================================================
# STEP 3: Running the Simulation
# ============================================================================
print("STEP 3: Running the Simulation")
print("-" * 80)
print("""
The simulation will run 15 replications for each of the 4 algorithms.
Each replication simulates 100 time steps of:
1. Observe context X_t
2. Choose arm based on algorithm strategy
3. Receive reward Y_t
4. Update β estimates using quantile regression
5. Track regret and estimation error

This should take ~5-10 minutes...
""")

input("Press Enter to start simulation...")

start_time = time.time()

print("\nRunning simulation...")
results = study.run_simulation()

elapsed = time.time() - start_time
print(f"✓ Simulation complete in {elapsed:.1f} seconds")
print()

input("Press Enter to continue...")
print()

# ============================================================================
# STEP 4: Analyzing Results
# ============================================================================
print("STEP 4: Analyzing Results")
print("-" * 80)
print("""
We track two key metrics:

1. Cumulative Regret: Sum of (optimal reward - actual reward)
   - Measures decision quality
   - Lower is better

2. Beta Estimation Error: ||β̂ - β_true||
   - Measures learning quality
   - Lower is better
""")

print("\nResults Summary:")
print("-" * 80)
print(f"{'Algorithm':<20} {'Final Regret':<18} {'Beta Error':<15} {'Runtime (s)':<12}")
print("-" * 80)

# Compute and display results
algorithm_results = {}
for alg in results['algorithms']:
    regret = results['regret'][alg][:, -1]  # Final regret
    beta_err = results['beta_errors'][alg][:, -1, :]  # Final beta error
    runtime = results['computation_time'][alg]

    mean_regret = np.mean(regret)
    std_regret = np.std(regret)
    mean_beta_err = np.mean(beta_err)

    algorithm_results[alg] = {
        'regret': mean_regret,
        'regret_std': std_regret,
        'beta_error': mean_beta_err,
        'runtime': runtime
    }

    print(f"{alg:<20} {mean_regret:<7.2f} ± {std_regret:<7.2f}  "
          f"{mean_beta_err:<15.4f} {runtime:<12.3f}")

print("-" * 80)

# Find winner
winner = min(algorithm_results.items(), key=lambda x: x[1]['regret'])
baseline_regret = algorithm_results['ForcedSampling']['regret']

print(f"\n🏆 Winner: {winner[0]}")
print(f"   Final Regret: {winner[1]['regret']:.2f} ± {winner[1]['regret_std']:.2f}")
print(f"   Improvement over baseline: {(1 - winner[1]['regret']/baseline_regret)*100:.1f}%")
print()

input("Press Enter to see visualizations...")
print()

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("STEP 5: Visualizations")
print("-" * 80)
print("""
Creating two plots:

1. Cumulative Regret over Time
   - Shows how regret accumulates
   - Steeper slope = worse performance
   - All algorithms eventually learn

2. Beta Estimation Error over Time
   - Shows convergence of parameter estimates
   - Decreases as we see more data
""")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Colors for each algorithm
colors = {
    'ForcedSampling': '#e74c3c',      # Red
    'LinUCB': '#3498db',              # Blue
    'EpsilonGreedy': '#f39c12',       # Orange
    'ThompsonSampling': '#2ecc71'     # Green
}

# Labels for legend
labels = {
    'ForcedSampling': 'Forced Sampling',
    'LinUCB': 'LinUCB',
    'EpsilonGreedy': 'Epsilon-Greedy',
    'ThompsonSampling': 'Thompson Sampling'
}

steps = np.arange(1, 101)

# Plot 1: Cumulative Regret
ax1 = axes[0]
for alg in results['algorithms']:
    regret = results['regret'][alg]
    mean_regret = np.mean(regret, axis=0)
    std_regret = np.std(regret, axis=0)

    color = colors[alg]
    label = labels[alg]

    ax1.plot(steps, mean_regret, label=label, color=color, linewidth=2.5)
    ax1.fill_between(steps,
                     mean_regret - std_regret,
                     mean_regret + std_regret,
                     color=color, alpha=0.2)

ax1.set_xlabel('Time Step', fontweight='bold', fontsize=12)
ax1.set_ylabel('Cumulative Regret', fontweight='bold', fontsize=12)
ax1.set_title('Algorithm Comparison: Regret Over Time',
              fontweight='bold', fontsize=13)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Beta Estimation Error
ax2 = axes[1]
for alg in results['algorithms']:
    beta_err = results['beta_errors'][alg]
    # Average across arms and dimensions
    beta_err_avg = np.mean(beta_err, axis=2)  # Average over dimensions
    mean_err = np.mean(beta_err_avg, axis=0)
    std_err = np.std(beta_err_avg, axis=0)

    color = colors[alg]
    label = labels[alg]

    ax2.plot(steps, mean_err, label=label, color=color, linewidth=2.5)
    ax2.fill_between(steps,
                     mean_err - std_err,
                     mean_err + std_err,
                     color=color, alpha=0.2)

ax2.set_xlabel('Time Step', fontweight='bold', fontsize=12)
ax2.set_ylabel('Beta Estimation Error (L2 norm)', fontweight='bold', fontsize=12)
ax2.set_title('Algorithm Comparison: Estimation Error Over Time',
              fontweight='bold', fontsize=13)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
save_path = 'demo_results.pdf'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved as: {save_path}")

# Show plot
plt.show(block=False)
plt.pause(0.1)

print()

# ============================================================================
# STEP 6: How to Use This Code for Your Own Research
# ============================================================================
print("STEP 6: How to Use This Code for Your Own Research")
print("-" * 80)
print("""
You can easily adapt this code for your own experiments:

1. Change the beta distribution:

   from generators import UniformGenerator, TGenerator

   # Heavy-tailed coefficients
   beta_gen = TGenerator(df=5, scale=1)

   # Sparse coefficients
   beta_gen = UniformGenerator(low=-0.3, high=0.3)

   # Heterogeneous (different per arm)
   beta_gen = [
       NormalGenerator(mean=0, std=1),
       UniformGenerator(low=0.5, high=1.5)
   ]

2. Compare specific algorithms only:

   algorithms=['LinUCB', 'ThompsonSampling']  # Just these two

3. Run more extensive simulations:

   study = Project4Simulation(
       n_sim=100,    # More replications
       T=500,        # Longer horizon
       K=5,          # More arms
       d=20,         # Higher dimension
       ...
   )

4. Vary quantile level τ:

   tau=0.75   # 75th percentile instead of median

5. Use the complete workflow:

   cd project4files
   make scenarios          # Run all 6 scenarios
   make experiments        # Run parameter sweeps
   make analyze-all        # Generate all figures

See ../README.md for complete documentation and examples.
""")

# ============================================================================
# CONCLUSION
# ============================================================================
print("=" * 80)
print(" DEMO COMPLETE")
print("=" * 80)
print("""
What you've learned:

✓ How to set up a multi-algorithm bandit simulation
✓ How to configure beta/error distributions
✓ How to run the simulation and track metrics
✓ How to create comparison visualizations
✓ How to adapt the code for your own research

Next Steps:

1. Explore different scenarios:
   make workflow-scenarios

2. Run parameter sweeps:
   make workflow-experiments

3. Read the full report:
   ../report-Xie.md (concise version)
   ../report-Xie-full.md (detailed version)

4. Check the documentation:
   ../README.md

Happy experimenting!
""")
print("=" * 80)

input("\nPress Enter to exit...")
