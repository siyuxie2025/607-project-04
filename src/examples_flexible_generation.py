"""
Comprehensive Examples: Flexible Beta/Alpha Generation
======================================================

This script demonstrates the new flexible parameter generation features,
allowing you to explore different scenarios:

1. Default uniform beta (backward compatible)
2. Gaussian beta with zero mean (robust regression scenarios)
3. Heterogeneous arms (different distributions per arm)
4. Sparse coefficients (small magnitude beta)
5. Different context generators

Run individual examples or all at once.

Usage:
    python examples_flexible_generation.py --all
    python examples_flexible_generation.py --example 1
    python examples_flexible_generation.py --gaussian
"""

import argparse
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from src.simulation import SimulationStudy
from src.parallel_simulation import ParallelSimulationStudy
from src.generators import (
    NormalGenerator, 
    TGenerator, 
    UniformGenerator, 
    TruncatedNormalGenerator
)
import os

# Create results directory
os.makedirs('results/examples', exist_ok=True)

def example_1_default_uniform():
    """Example 1: Default Uniform Beta (Backward Compatible)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Default Uniform Beta Distribution")
    print("="*80)
    print("\nThis replicates the original behavior from Project 2.")
    print("Beta coefficients drawn from Uniform([0, 0.5], [1, 1.5])")
    print()
    
    study = SimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=200,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        random_seed=1010
    )
    
    print("Generated Beta Values:")
    print(f"  Shape: {study.beta_real_value.shape}")
    print(f"  Arm 0 mean: {np.mean(study.beta_real_value[0]):.3f}")
    print(f"  Arm 1 mean: {np.mean(study.beta_real_value[1]):.3f}")
    print(f"\nGenerated Alpha Values:")
    print(f"  Shape: {study.alpha_real_value.shape}")
    print(f"  Values: {study.alpha_real_value}")
    
    print("\nRunning simulation...")
    results = study.run_simulation()
    
    print(f"\nFinal Regret (mean ± std):")
    print(f"  RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
    print(f"  OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_OLS'][:, -1]):.2f}")
    
    study.save_results('results/examples/example1_default_uniform.pkl')
    print("\n✓ Example 1 complete!")


def example_2_gaussian_beta():
    """Example 2: Gaussian Beta (Zero Mean, Unit Variance)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Gaussian Beta Distribution")
    print("="*80)
    print("\nBeta coefficients drawn from Normal(0, 1)")
    print("Useful for studying robust regression properties")
    print()
    
    beta_gen = NormalGenerator(mean=0, std=1)
    
    study = SimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=200,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        beta_generator=beta_gen,  # <-- New parameter!
        random_seed=1010
    )
    
    print("Generated Beta Values:")
    print(f"  Shape: {study.beta_real_value.shape}")
    print(f"  Arm 0 mean: {np.mean(study.beta_real_value[0]):.3f}")
    print(f"  Arm 1 mean: {np.mean(study.beta_real_value[1]):.3f}")
    print(f"  Overall std: {np.std(study.beta_real_value):.3f}")
    
    print("\nRunning simulation...")
    results = study.run_simulation()
    
    print(f"\nFinal Regret (mean ± std):")
    print(f"  RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
    print(f"  OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_OLS'][:, -1]):.2f}")
    
    study.save_results('results/examples/example2_gaussian_beta.pkl')
    print("\n✓ Example 2 complete!")


def example_3_heterogeneous_arms():
    """Example 3: Heterogeneous Arms (Different Distribution Per Arm)"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Heterogeneous Arms")
    print("="*80)
    print("\nArm 0: Gaussian N(0, 1)")
    print("Arm 1: Uniform U(0.5, 1.5)")
    print("\nUseful for studying algorithm robustness to arm heterogeneity")
    print()
    
    beta_gens = [
        NormalGenerator(mean=0, std=1),      # Arm 0
        UniformGenerator(low=0.5, high=1.5)  # Arm 1
    ]
    
    study = SimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=200,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        beta_generator=beta_gens,  # <-- List of generators!
        random_seed=1010
    )
    
    print("Generated Beta Values:")
    print(f"  Arm 0 (Gaussian) mean: {np.mean(study.beta_real_value[0]):.3f}")
    print(f"  Arm 0 (Gaussian) std:  {np.std(study.beta_real_value[0]):.3f}")
    print(f"  Arm 1 (Uniform) mean:  {np.mean(study.beta_real_value[1]):.3f}")
    print(f"  Arm 1 (Uniform) std:   {np.std(study.beta_real_value[1]):.3f}")
    
    print("\nRunning simulation...")
    results = study.run_simulation()
    
    print(f"\nFinal Regret (mean ± std):")
    print(f"  RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
    print(f"  OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_OLS'][:, -1]):.2f}")
    
    study.save_results('results/examples/example3_heterogeneous_arms.pkl')
    print("\n✓ Example 3 complete!")


def example_4_sparse_coefficients():
    """Example 4: Sparse Coefficients (Small Magnitude)"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Sparse Coefficients")
    print("="*80)
    print("\nBeta coefficients drawn from Uniform(-0.5, 0.5)")
    print("Simulates sparse/weak signal scenarios")
    print()
    
    beta_gen = UniformGenerator(low=-0.5, high=0.5)
    
    study = SimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=200,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        beta_generator=beta_gen,
        random_seed=1010
    )
    
    print("Generated Beta Values:")
    print(f"  Shape: {study.beta_real_value.shape}")
    print(f"  Mean magnitude: {np.mean(np.abs(study.beta_real_value)):.3f}")
    print(f"  Max magnitude:  {np.max(np.abs(study.beta_real_value)):.3f}")
    print(f"  Proportion near zero (<0.1): {np.mean(np.abs(study.beta_real_value) < 0.1)*100:.1f}%")
    
    print("\nRunning simulation...")
    results = study.run_simulation()
    
    print(f"\nFinal Regret (mean ± std):")
    print(f"  RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
    print(f"  OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_OLS'][:, -1]):.2f}")
    
    study.save_results('results/examples/example4_sparse_coefficients.pkl')
    print("\n✓ Example 4 complete!")


def example_5_different_contexts():
    """Example 5: Different Context Generators"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Different Context Generators")
    print("="*80)
    print("\nComparing performance with different context distributions:")
    print("  A. Truncated Normal (default)")
    print("  B. Standard Normal")
    print("  C. Uniform")
    print()
    
    context_gens = {
        'TruncNormal': TruncatedNormalGenerator(mean=0.0, std=1.0, low=-3, high=3),
        'Normal': NormalGenerator(mean=0.0, std=1.0),
        'Uniform': UniformGenerator(low=-1, high=1)
    }
    
    beta_gen = NormalGenerator(mean=0, std=1)
    
    results_by_context = {}
    
    for name, context_gen in context_gens.items():
        print(f"\n  Testing with {name} context generator...")
        
        study = SimulationStudy(
            n_sim=20,
            K=2,
            d=10,
            T=200,
            q=2,
            h=0.5,
            tau=0.5,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=context_gen,  # <-- Different contexts!
            beta_generator=beta_gen,
            random_seed=1010
        )
        
        results = study.run_simulation()
        results_by_context[name] = results
        
        rab_regret = np.mean(results['cumulated_regret_RiskAware'][:, -1])
        ols_regret = np.mean(results['cumulated_regret_OLS'][:, -1])
        
        print(f"    RAB regret: {rab_regret:.2f}")
        print(f"    OLS regret: {ols_regret:.2f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(context_gens))
    width = 0.35
    
    rab_means = [np.mean(results_by_context[name]['cumulated_regret_RiskAware'][:, -1]) 
                 for name in context_gens.keys()]
    ols_means = [np.mean(results_by_context[name]['cumulated_regret_OLS'][:, -1]) 
                 for name in context_gens.keys()]
    
    ax.bar(x - width/2, rab_means, width, label='RAB', color='red', alpha=0.7)
    ax.bar(x + width/2, ols_means, width, label='OLS', color='blue', alpha=0.7)
    
    ax.set_xlabel('Context Generator', fontsize=12)
    ax.set_ylabel('Final Regret', fontsize=12)
    ax.set_title('Performance with Different Context Distributions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(context_gens.keys())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/examples/example5_different_contexts.pdf', dpi=300)
    print("\n  ✓ Plot saved: results/examples/example5_different_contexts.pdf")
    
    print("\n✓ Example 5 complete!")


def example_6_combined_alpha_beta():
    """Example 6: Custom Beta AND Alpha Generators"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Beta AND Alpha Generators")
    print("="*80)
    print("\nBeta: Gaussian N(0, 1)")
    print("Alpha: Uniform U(0, 2)")
    print()
    
    beta_gen = NormalGenerator(mean=0, std=1)
    alpha_gen = UniformGenerator(low=0, high=2)
    
    study = SimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=200,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        beta_generator=beta_gen,   # Custom beta
        alpha_generator=alpha_gen,  # Custom alpha
        random_seed=1010
    )
    
    print("Generated Parameters:")
    print(f"  Beta mean: {np.mean(study.beta_real_value):.3f}")
    print(f"  Beta std:  {np.std(study.beta_real_value):.3f}")
    print(f"  Alpha values: {study.alpha_real_value}")
    print(f"  Alpha mean: {np.mean(study.alpha_real_value):.3f}")
    
    print("\nRunning simulation...")
    results = study.run_simulation()
    
    print(f"\nFinal Regret (mean ± std):")
    print(f"  RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
    print(f"  OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f} ± "
          f"{np.std(results['cumulated_regret_OLS'][:, -1]):.2f}")
    
    study.save_results('results/examples/example6_combined_generators.pkl')
    print("\n✓ Example 6 complete!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Flexible Beta/Alpha Generation Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5,6],
                       help='Run specific example')
    parser.add_argument('--gaussian', action='store_true',
                       help='Run Gaussian beta example (alias for --example 2)')
    parser.add_argument('--heterogeneous', action='store_true',
                       help='Run heterogeneous arms example (alias for --example 3)')
    parser.add_argument('--contexts', action='store_true',
                       help='Run different contexts example (alias for --example 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FLEXIBLE BETA/ALPHA GENERATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate the new flexible parameter generation")
    print("features added in Project 3.")
    print("="*80)
    
    examples = {
        1: example_1_default_uniform,
        2: example_2_gaussian_beta,
        3: example_3_heterogeneous_arms,
        4: example_4_sparse_coefficients,
        5: example_5_different_contexts,
        6: example_6_combined_alpha_beta
    }
    
    if args.all:
        print("\nRunning all examples...\n")
        for ex_num, ex_func in examples.items():
            ex_func()
    elif args.example:
        examples[args.example]()
    elif args.gaussian:
        example_2_gaussian_beta()
    elif args.heterogeneous:
        example_3_heterogeneous_arms()
    elif args.contexts:
        example_5_different_contexts()
    else:
        parser.print_help()
        print("\n" + "="*80)
        print("QUICK START")
        print("="*80)
        print("\nRun all examples:")
        print("  python examples_flexible_generation.py --all")
        print("\nRun specific example:")
        print("  python examples_flexible_generation.py --example 2")
        print("\nQuick shortcuts:")
        print("  python examples_flexible_generation.py --gaussian")
        print("  python examples_flexible_generation.py --heterogeneous")
        print("  python examples_flexible_generation.py --contexts")


if __name__ == "__main__":
    main()