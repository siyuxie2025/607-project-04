"""
Project 4 Main Execution Script
================================

One-line execution for different beta generation scenarios:
    python project4_main.py --scenario default
    python project4_main.py --scenario gaussian
    python project4_main.py --scenario heterogeneous
    python project4_main.py --scenario sparse
    python project4_main.py --scenario all

This script runs complete simulations comparing all 4 algorithms
(LinUCB, Thompson Sampling, Epsilon-Greedy, Forced Sampling) with
different beta generation strategies.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from project4_simulation import Project4Simulation
from generators import (
    NormalGenerator, 
    UniformGenerator, 
    TGenerator, 
    TruncatedNormalGenerator
)

# Create results directory
Path('results/project4').mkdir(parents=True, exist_ok=True)


class Project4Scenarios:
    """
    Pre-defined scenarios for Project 4 experiments.
    
    Each scenario tests a different beta generation strategy to answer
    research questions about algorithm robustness and performance.
    """
    
    @staticmethod
    def default(n_sim=50, T=200):
        """
        Scenario 1: Default (Baseline)
        
        Uses uniform beta distribution - this is the baseline from Projects 2-3.
        Answers: "How do algorithms perform under standard conditions?"
        """
        print("\n" + "="*80)
        print("SCENARIO 1: Default (Uniform Beta)")
        print("="*80)
        print("Research Question: How do algorithms perform under standard conditions?")
        print("Beta Generation: Uniform([0.5, 1], [1, 1.5]) - standard baseline")
        print("="*80 + "\n")
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=2,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=None,  # Default uniform
            alpha_generator=None,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_default.pkl')
        study.plot_comparison(save_path='results/project4/scenario_default.pdf')
        
        return results
    
    @staticmethod
    def gaussian(n_sim=50, T=200):
        """
        Scenario 2: Gaussian Beta (Zero-Mean)
        
        Uses Gaussian beta with mean 0 - symmetric around zero.
        Answers: "How do algorithms handle symmetric, zero-mean coefficients?"
        """
        print("\n" + "="*80)
        print("SCENARIO 2: Gaussian Beta (Zero-Mean)")
        print("="*80)
        print("Research Question: How do algorithms handle zero-mean coefficients?")
        print("Beta Generation: Normal(μ=0, σ=1) - symmetric, unbiased")
        print("="*80 + "\n")
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=2,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=NormalGenerator(mean=0, std=1),
            alpha_generator=NormalGenerator(mean=0, std=0.5),
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_gaussian.pkl')
        study.plot_comparison(save_path='results/project4/scenario_gaussian.pdf')
        
        return results
    
    @staticmethod
    def heterogeneous(n_sim=50, T=200):
        """
        Scenario 3: Heterogeneous Arms
        
        Each arm has different beta distribution.
        Answers: "How robust are algorithms to arm heterogeneity?"
        """
        print("\n" + "="*80)
        print("SCENARIO 3: Heterogeneous Arms")
        print("="*80)
        print("Research Question: How robust are algorithms to arm heterogeneity?")
        print("Beta Generation:")
        print("  - Arm 0: Normal(0, 1) - Gaussian")
        print("  - Arm 1: Uniform(0.5, 1.5) - Uniform")
        print("="*80 + "\n")
        
        beta_gens = [
            NormalGenerator(mean=0, std=1),      # Arm 0: Gaussian
            UniformGenerator(low=0.5, high=1.5)  # Arm 1: Uniform
        ]
        
        alpha_gens = [
            NormalGenerator(mean=0, std=0.5),
            UniformGenerator(low=0.5, high=1.0)
        ]
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=2,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=beta_gens,
            alpha_generator=alpha_gens,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_heterogeneous.pkl')
        study.plot_comparison(save_path='results/project4/scenario_heterogeneous.pdf')
        
        return results
    
    @staticmethod
    def sparse(n_sim=50, T=200):
        """
        Scenario 4: Sparse Coefficients
        
        Uses small uniform range - tests weak signal detection.
        Answers: "How do algorithms perform with weak/sparse signals?"
        """
        print("\n" + "="*80)
        print("SCENARIO 4: Sparse Coefficients (Weak Signals)")
        print("="*80)
        print("Research Question: How do algorithms perform with weak signals?")
        print("Beta Generation: Uniform(-0.3, 0.3) - sparse/weak coefficients")
        print("="*80 + "\n")
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=2,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=UniformGenerator(low=-0.3, high=0.3),
            alpha_generator=UniformGenerator(low=-0.2, high=0.2),
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_sparse.pkl')
        study.plot_comparison(save_path='results/project4/scenario_sparse.pdf')
        
        return results
    
    @staticmethod
    def heavy_tailed_beta(n_sim=50, T=200):
        """
        Scenario 5: Heavy-Tailed Beta
        
        Uses t-distribution for beta - extreme coefficient values.
        Answers: "How do algorithms handle outlier coefficients?"
        """
        print("\n" + "="*80)
        print("SCENARIO 5: Heavy-Tailed Beta")
        print("="*80)
        print("Research Question: How do algorithms handle outlier coefficients?")
        print("Beta Generation: t(df=3) - heavy tails, extreme values")
        print("="*80 + "\n")
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=2,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=TGenerator(df=3, scale=1),
            alpha_generator=TGenerator(df=3, scale=0.5),
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_heavy_tailed.pkl')
        study.plot_comparison(save_path='results/project4/scenario_heavy_tailed.pdf')
        
        return results
    
    @staticmethod
    def multi_arm(n_sim=30, T=150):
        """
        Scenario 6: Multiple Arms (K=5)
        
        Tests scalability to more arms with heterogeneous generation.
        Answers: "How do algorithms scale to more arms?"
        """
        print("\n" + "="*80)
        print("SCENARIO 6: Multiple Arms (K=5)")
        print("="*80)
        print("Research Question: How do algorithms scale to more arms?")
        print("Beta Generation: Different distribution per arm")
        print("="*80 + "\n")
        
        beta_gens = [
            NormalGenerator(mean=0, std=1),
            UniformGenerator(low=0.5, high=1.5),
            TGenerator(df=5, scale=1),
            NormalGenerator(mean=0.5, std=0.8),
            UniformGenerator(low=0, high=1)
        ]
        
        study = Project4Simulation(
            n_sim=n_sim,
            K=5,
            d=10,
            T=T,
            tau=0.5,
            algorithms='all',
            beta_generator=beta_gens,
            err_generator=TGenerator(df=2.25, scale=0.7),
            context_generator=TruncatedNormalGenerator(0, 1),
            random_seed=42
        )
        
        results = study.run_simulation()
        study.save_results('results/project4/scenario_multi_arm.pkl')
        study.plot_comparison(save_path='results/project4/scenario_multi_arm.pdf')
        
        return results


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Project 4: Multi-Algorithm Bandit Comparison with Flexible Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project4_main.py --scenario default       # Uniform beta (baseline)
  python project4_main.py --scenario gaussian      # Gaussian beta (zero-mean)
  python project4_main.py --scenario heterogeneous # Different beta per arm
  python project4_main.py --scenario sparse        # Weak signals
  python project4_main.py --scenario heavy         # Heavy-tailed beta
  python project4_main.py --scenario multi         # K=5 arms
  python project4_main.py --scenario all           # Run all scenarios
  python project4_main.py --scenario quick         # Quick test (n_sim=5, T=50)

Scenarios test different research questions:
  - default: Standard baseline performance
  - gaussian: Zero-mean symmetric coefficients
  - heterogeneous: Robustness to arm differences
  - sparse: Weak signal detection
  - heavy: Outlier coefficient handling
  - multi: Scalability to more arms
        """
    )
    
    parser.add_argument(
        '--scenario', 
        type=str,
        default='default',
        choices=['default', 'gaussian', 'heterogeneous', 'sparse', 
                 'heavy', 'multi', 'all', 'quick'],
        help='Scenario to run (default: default)'
    )
    
    parser.add_argument(
        '--n_sim',
        type=int,
        default=None,
        help='Number of simulation replications (overrides default)'
    )
    
    parser.add_argument(
        '--T',
        type=int,
        default=None,
        help='Time horizon (overrides default)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (n_sim=5, T=50)'
    )
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick or args.scenario == 'quick':
        n_sim = 5
        T = 50
        print("\n⚡ QUICK TEST MODE (n_sim=5, T=50)")
    else:
        n_sim = args.n_sim if args.n_sim is not None else 50
        T = args.T if args.T is not None else 200
    
    print("\n" + "="*80)
    print("PROJECT 4: MULTI-ALGORITHM BANDIT COMPARISON")
    print("Flexible Beta/Alpha Generation Framework")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Simulations: {n_sim}")
    print(f"  - Time horizon: {T}")
    print(f"  - Scenario: {args.scenario}")
    print("="*80)
    
    start_time = time.time()
    
    # Run scenarios
    scenarios = Project4Scenarios()
    
    if args.scenario == 'all':
        print("\nRunning ALL scenarios (this will take a while)...\n")
        results_all = {}
        
        for scenario_name in ['default', 'gaussian', 'heterogeneous', 
                             'sparse', 'heavy', 'multi']:
            print(f"\n{'='*80}")
            print(f"Running scenario: {scenario_name}")
            print(f"{'='*80}")
            
            scenario_func = getattr(scenarios, scenario_name.replace('-', '_'))
            results_all[scenario_name] = scenario_func(n_sim=n_sim, T=T)
        
        print("\n" + "="*80)
        print("✓ ALL SCENARIOS COMPLETE")
        print("="*80)
        print("\nResults saved to results/project4/")
        print("Run 'python project4_analysis.py' to analyze all results.")
        
    else:
        # Run single scenario
        scenario_func = getattr(scenarios, args.scenario.replace('-', '_'))
        results = scenario_func(n_sim=n_sim, T=T)
        
        print("\n" + "="*80)
        print(f"✓ SCENARIO '{args.scenario}' COMPLETE")
        print("="*80)
        print(f"\nResults saved to results/project4/scenario_{args.scenario}.pkl")
        print(f"Plot saved to results/project4/scenario_{args.scenario}.pdf")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. View results: open results/project4/scenario_*.pdf")
    print("2. Run analysis: python project4_analysis.py")
    print("3. Try another scenario: python project4_main.py --scenario <name>")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()