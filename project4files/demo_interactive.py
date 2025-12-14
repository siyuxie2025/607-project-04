"""
Project 4 Interactive Demonstration
====================================

This script provides an interactive menu-driven demonstration for exploring
the multi-algorithm quantile bandit comparison framework with different
configurations.

For a step-by-step tutorial, use demo_quickstart.py instead.

Usage:
    python demo_interactive.py                 # Interactive mode
    python demo_interactive.py --quick         # Quick demo
    python demo_interactive.py --all           # All algorithms demo
    python demo_interactive.py --scenarios     # Scenario comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from project4_simulation import Project4Simulation
from generators import (
    NormalGenerator,
    UniformGenerator,
    TGenerator,
    TruncatedNormalGenerator
)

# Styling
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['font.size'] = 10

DIVIDER = "=" * 70


def print_header(text):
    """Print formatted header."""
    print(f"\n{DIVIDER}")
    print(f"  {text}")
    print(f"{DIVIDER}\n")


def print_section(text):
    """Print formatted section."""
    print(f"\n{'-' * 70}")
    print(f"  {text}")
    print(f"{'-' * 70}")


def demo_quick():
    """Quick demonstration (5 sims, 50 rounds)."""
    print_header("PROJECT 4: QUICK DEMONSTRATION")
    
    print("This demo runs a minimal simulation to show how the framework works.")
    print("Configuration: n_sim=5, T=50, K=2, d=10")
    print("\nAlgorithms:")
    print("  • Forced Sampling (baseline from Projects 2-3)")
    print("  • Thompson Sampling (the practical winner)")
    print("")
    
    input("Press Enter to start demo...")
    
    # Create simulation
    print_section("Step 1: Creating Simulation")
    
    print("Setting up simulation with Gaussian beta generation...")
    study = Project4Simulation(
        n_sim=5,
        K=2,
        d=10,
        T=50,
        tau=0.5,
        algorithms=['ForcedSampling', 'ThompsonSampling'],
        beta_generator=NormalGenerator(mean=0, std=1),
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(0, 1),
        random_seed=42
    )
    print("✓ Simulation created")
    
    # Run simulation
    print_section("Step 2: Running Simulation")
    
    print("Running 5 simulation replications...")
    start_time = time.time()
    results = study.run_simulation()
    elapsed = time.time() - start_time
    
    print(f"✓ Simulation complete in {elapsed:.2f}s")
    
    # Show results
    print_section("Step 3: Results Summary")
    
    for alg in results['algorithms']:
        regret = results['regret'][alg][:, -1]
        beta_err = results['beta_errors'][alg][:, -1, :]
        runtime = results['computation_time'][alg]
        
        print(f"\n{alg}:")
        print(f"  Final Regret:     {np.mean(regret):.2f} ± {np.std(regret):.2f}")
        print(f"  Final Beta Error: {np.mean(beta_err):.4f} ± {np.std(beta_err):.4f}")
        print(f"  Runtime:          {runtime:.3f}s")
    
    # Create plot
    print_section("Step 4: Visualization")
    
    print("Generating comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    T = 50
    steps = np.arange(1, T + 1)
    colors = {'ForcedSampling': 'red', 'ThompsonSampling': 'green'}
    
    # Regret plot
    ax1 = axes[0]
    for alg in results['algorithms']:
        regret = results['regret'][alg]
        mean_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        
        color = colors[alg]
        ax1.plot(steps, mean_regret, label=alg, color=color, linewidth=2)
        ax1.fill_between(steps, mean_regret - std_regret, mean_regret + std_regret,
                         color=color, alpha=0.2)
    
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Cumulative Regret', fontweight='bold')
    ax1.set_title('Regret Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Beta error plot
    ax2 = axes[1]
    for alg in results['algorithms']:
        beta_err = results['beta_errors'][alg]
        # Average across arms
        beta_err_avg = np.mean(beta_err, axis=2)
        mean_err = np.mean(beta_err_avg, axis=0)
        std_err = np.std(beta_err_avg, axis=0)
        
        color = colors[alg]
        ax2.plot(steps, mean_err, label=alg, color=color, linewidth=2)
        ax2.fill_between(steps, mean_err - std_err, mean_err + std_err,
                         color=color, alpha=0.2)
    
    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_ylabel('Beta Estimation Error', fontweight='bold')
    ax2.set_title('Beta Error Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show
    save_path = 'demo_quick_comparison.pdf'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {save_path}")
    
    plt.show(block=False)
    plt.pause(0.1)
    
    print("\nKey Observations:")
    print("  • Thompson Sampling (green) converges faster")
    print("  • Thompson has lower final regret")
    print("  • Thompson achieves better beta estimation")
    print("  • Both use quantile regression updates")
    
    print_section("Demo Complete")
    print("\nNext steps:")
    print("  • Run full scenarios: make scenario-gaussian")
    print("  • Compare all 4 algorithms: make scenarios")
    print("  • Run experimental sweeps: make experiments")
    print("\nSee PROJECT4_README.md for complete documentation.")


def demo_all_algorithms():
    """Demonstrate all 4 algorithms."""
    print_header("ALL ALGORITHMS DEMONSTRATION")
    
    print("This demo compares all 4 algorithms:")
    print("  1. Forced Sampling (baseline)")
    print("  2. LinUCB (quantile version)")
    print("  3. Epsilon-Greedy (quantile version)")
    print("  4. Thompson Sampling (quantile version)")
    print("\nConfiguration: n_sim=10, T=100")
    print("")
    
    input("Press Enter to continue...")
    
    # Create simulation
    print("\nRunning simulation with all 4 algorithms...")
    study = Project4Simulation(
        n_sim=10,
        K=2,
        d=10,
        T=100,
        tau=0.5,
        algorithms='all',  # All 4 algorithms
        beta_generator=NormalGenerator(mean=0, std=1),
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(0, 1),
        random_seed=42
    )
    
    start_time = time.time()
    results = study.run_simulation()
    elapsed = time.time() - start_time
    
    print(f"✓ Simulation complete in {elapsed:.2f}s")
    
    # Results table
    print("\n" + "-" * 80)
    print(f"{'Algorithm':<20} {'Mean Regret':<15} {'Beta Error':<15} {'Runtime (s)':<12}")
    print("-" * 80)
    
    for alg in results['algorithms']:
        regret = results['regret'][alg][:, -1]
        beta_err = results['beta_errors'][alg][:, -1, :]
        runtime = results['computation_time'][alg]
        
        print(f"{alg:<20} {np.mean(regret):<15.2f} {np.mean(beta_err):<15.4f} {runtime:<12.3f}")
    
    print("-" * 80)
    
    # Create comprehensive plot
    print("\nGenerating comparison plot...")
    study.plot_comparison(save_path='demo_all_algorithms.pdf')
    print("✓ Plot saved: demo_all_algorithms.pdf")
    
    # Show winner
    print("\n" + "=" * 70)
    print("  WINNER ANALYSIS")
    print("=" * 70)
    
    regrets = {alg: np.mean(results['regret'][alg][:, -1]) for alg in results['algorithms']}
    winner = min(regrets, key=regrets.get)
    
    print(f"\n🏆 Best Algorithm: {winner}")
    print(f"   Final Regret: {regrets[winner]:.2f}")
    print(f"   Performance vs Forced Sampling: {(1 - regrets[winner]/regrets['ForcedSampling'])*100:.1f}% better")


def demo_scenario_comparison():
    """Demonstrate different scenarios."""
    print_header("SCENARIO COMPARISON DEMONSTRATION")
    
    print("This demo shows how algorithms perform under different beta distributions:")
    print("  1. Uniform Beta (baseline)")
    print("  2. Gaussian Beta (zero-mean)")
    print("  3. Sparse Beta (weak signals)")
    print("\nWe'll run Thompson Sampling on each and compare.")
    print("Configuration: n_sim=10, T=100 per scenario")
    print("")
    
    input("Press Enter to continue...")
    
    scenarios = {
        'Uniform': UniformGenerator(low=0.5, high=1.5),
        'Gaussian': NormalGenerator(mean=0, std=1),
        'Sparse': UniformGenerator(low=-0.3, high=0.3)
    }
    
    results_by_scenario = {}
    
    for scenario_name, beta_gen in scenarios.items():
        print(f"\nRunning scenario: {scenario_name}")
        
        study = Project4Simulation(
            n_sim=10,
            K=2,
            d=10,
            T=100,
            tau=0.5,
            algorithms=['ThompsonSampling'],
            beta_generator=beta_gen,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        results_by_scenario[scenario_name] = results
        
        regret = results['regret']['ThompsonSampling'][:, -1]
        print(f"  Final Regret: {np.mean(regret):.2f} ± {np.std(regret):.2f}")
    
    # Comparison plot
    print("\nGenerating scenario comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Uniform': 'blue', 'Gaussian': 'green', 'Sparse': 'orange'}
    
    for scenario_name, results in results_by_scenario.items():
        regret = results['regret']['ThompsonSampling']
        mean_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0)
        
        steps = np.arange(1, 101)
        color = colors[scenario_name]
        
        ax.plot(steps, mean_regret, label=f'{scenario_name} Beta', 
                color=color, linewidth=2)
        ax.fill_between(steps, mean_regret - std_regret, mean_regret + std_regret,
                         color=color, alpha=0.2)
    
    ax.set_xlabel('Time Step', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Regret (Thompson Sampling)', fontweight='bold', fontsize=12)
    ax.set_title('Thompson Sampling Performance Across Scenarios', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_scenario_comparison.pdf', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: demo_scenario_comparison.pdf")
    
    plt.show(block=False)
    plt.pause(0.1)
    
    print("\nKey Observations:")
    print("  • Gaussian beta: Best performance (symmetric around zero)")
    print("  • Uniform beta: Standard baseline")
    print("  • Sparse beta: Most challenging (weak signals)")


def demo_interactive():
    """Interactive menu-driven demo."""
    print_header("PROJECT 4: INTERACTIVE DEMONSTRATION")
    
    while True:
        print("\nAvailable Demonstrations:")
        print("  1. Quick Demo (5 sims, 2 algorithms, ~10 seconds)")
        print("  2. All Algorithms Demo (10 sims, 4 algorithms, ~30 seconds)")
        print("  3. Scenario Comparison Demo (3 scenarios, ~45 seconds)")
        print("  4. Custom Demo (specify your own parameters)")
        print("  5. Exit")
        print("")
        
        choice = input("Select demo (1-5): ").strip()
        
        if choice == '1':
            demo_quick()
        elif choice == '2':
            demo_all_algorithms()
        elif choice == '3':
            demo_scenario_comparison()
        elif choice == '4':
            demo_custom()
        elif choice == '5':
            print("\nThank you for exploring Project 4!")
            print("For full documentation, see PROJECT4_README.md")
            break
        else:
            print("Invalid choice. Please select 1-5.")
        
        if choice in ['1', '2', '3', '4']:
            cont = input("\nReturn to main menu? (y/n): ").strip().lower()
            if cont != 'y':
                print("\nThank you for exploring Project 4!")
                break


def demo_custom():
    """Custom demo with user-specified parameters."""
    print_section("CUSTOM DEMO")
    
    print("Specify your demo parameters:")
    
    try:
        n_sim = int(input("  Number of simulations (default 10): ") or "10")
        T = int(input("  Number of rounds (default 100): ") or "100")
        K = int(input("  Number of arms (default 2): ") or "2")
        d = int(input("  Context dimension (default 10): ") or "10")
        
        print("\nBeta distribution:")
        print("  1. Uniform")
        print("  2. Gaussian")
        print("  3. Sparse")
        beta_choice = input("  Select (1-3, default 2): ") or "2"
        
        if beta_choice == '1':
            beta_gen = UniformGenerator(low=0.5, high=1.5)
            beta_name = "Uniform"
        elif beta_choice == '3':
            beta_gen = UniformGenerator(low=-0.3, high=0.3)
            beta_name = "Sparse"
        else:
            beta_gen = NormalGenerator(mean=0, std=1)
            beta_name = "Gaussian"
        
        print(f"\nRunning custom demo: n_sim={n_sim}, T={T}, K={K}, d={d}, beta={beta_name}")
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=K,
            d=d,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=beta_gen,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        start_time = time.time()
        results = study.run_simulation()
        elapsed = time.time() - start_time
        
        print(f"✓ Simulation complete in {elapsed:.2f}s")
        
        # Show results
        print("\n" + "-" * 70)
        print(f"{'Algorithm':<20} {'Mean Regret':<15} {'Std Regret':<15}")
        print("-" * 70)
        
        for alg in results['algorithms']:
            regret = results['regret'][alg][:, -1]
            print(f"{alg:<20} {np.mean(regret):<15.2f} {np.std(regret):<15.2f}")
        
        print("-" * 70)
        
        # Plot
        save_path = f'demo_custom_{beta_name}.pdf'
        study.plot_comparison(save_path=save_path)
        print(f"✓ Plot saved: {save_path}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please enter valid numeric values.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Project 4 Interactive Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python PROJECT4_DEMO.py              # Interactive menu
  python PROJECT4_DEMO.py --quick      # Quick demo
  python PROJECT4_DEMO.py --all        # All algorithms demo
  python PROJECT4_DEMO.py --scenarios  # Scenario comparison
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo')
    parser.add_argument('--all', action='store_true',
                       help='Run all algorithms demo')
    parser.add_argument('--scenarios', action='store_true',
                       help='Run scenario comparison demo')
    
    args = parser.parse_args()
    
    if args.quick:
        demo_quick()
    elif args.all:
        demo_all_algorithms()
    elif args.scenarios:
        demo_scenario_comparison()
    else:
        # Interactive mode
        demo_interactive()


if __name__ == "__main__":
    main()