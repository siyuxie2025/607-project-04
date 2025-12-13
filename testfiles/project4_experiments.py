"""
Project 4 Experiments Script
=============================

Batch experiments and parameter sweeps for research questions.

Usage:
    python project4_experiments.py --experiment df_sweep
    python project4_experiments.py --experiment tau_sweep
    python project4_experiments.py --experiment dimension_sweep
    python project4_experiments.py --experiment all
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json

sys.path.insert(0, 'src')

from project4_simulation import Project4Simulation
from generators import (
    NormalGenerator,
    UniformGenerator,
    TGenerator,
    TruncatedNormalGenerator
)

Path('results/project4/experiments').mkdir(parents=True, exist_ok=True)


class Project4Experiments:
    """
    Research experiments for Project 4.
    
    These experiments systematically vary parameters to answer specific
    research questions about algorithm performance and robustness.
    """
    
    @staticmethod
    def df_sweep(n_sim=30, T=150):
        """
        Experiment 1: Tail Heaviness Sweep
        
        Research Question: How do algorithms perform as error distribution
        gets heavier tails (lower df)?
        
        Tests df = [1.5, 2.25, 3, 5, 10] with Gaussian beta
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: Tail Heaviness Sweep (df)")
        print("="*80)
        print("Research Question: How do algorithms handle heavy-tailed errors?")
        print("Parameter: df = [1.5, 2.25, 3, 5, 10]")
        print("Beta: Gaussian(0, 1)")
        print("="*80 + "\n")
        
        df_values = [1.5, 2.25, 3.0, 5.0, 10.0]
        results_by_df = {}
        
        for df in df_values:
            print(f"\n{'='*60}")
            print(f"Running df = {df}")
            print('='*60)
            
            study = Project4Simulation(
                n_sim=n_sim,
                K=2,
                d=10,
                T=T,
                tau=0.5,
                algorithms='all',
                beta_generator=NormalGenerator(mean=0, std=1),
                err_generator=TGenerator(df=df, scale=0.7),
                context_generator=TruncatedNormalGenerator(0, 1),
                random_seed=42
            )
            
            results = study.run_simulation()
            results_by_df[df] = results
            
            # Save individual result
            study.save_results(f'results/project4/experiments/df_sweep_df{df}.pkl')
        
        # Save summary
        summary = _create_df_summary(results_by_df, df_values)
        summary.to_csv('results/project4/experiments/df_sweep_summary.csv', index=False)
        
        print("\n✓ DF sweep complete")
        print(f"Summary saved to: results/project4/experiments/df_sweep_summary.csv")
        
        return results_by_df
    
    @staticmethod
    def tau_sweep(n_sim=30, T=150):
        """
        Experiment 2: Quantile Level Sweep
        
        Research Question: How does quantile level τ affect performance?
        
        Tests τ = [0.25, 0.5, 0.75] with Gaussian beta
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: Quantile Level Sweep (τ)")
        print("="*80)
        print("Research Question: How does target quantile affect performance?")
        print("Parameter: τ = [0.25, 0.5, 0.75]")
        print("Beta: Gaussian(0, 1)")
        print("="*80 + "\n")
        
        tau_values = [0.25, 0.5, 0.75]
        results_by_tau = {}
        
        for tau in tau_values:
            print(f"\n{'='*60}")
            print(f"Running τ = {tau}")
            print('='*60)
            
            study = Project4Simulation(
                n_sim=n_sim,
                K=2,
                d=10,
                T=T,
                tau=tau,
                algorithms='all',
                beta_generator=NormalGenerator(mean=0, std=1),
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(0, 1),
                random_seed=42
            )
            
            results = study.run_simulation()
            results_by_tau[tau] = results
            
            study.save_results(f'results/project4/experiments/tau_sweep_tau{tau}.pkl')
        
        # Save summary
        summary = _create_tau_summary(results_by_tau, tau_values)
        summary.to_csv('results/project4/experiments/tau_sweep_summary.csv', index=False)
        
        print("\n✓ τ sweep complete")
        print(f"Summary saved to: results/project4/experiments/tau_sweep_summary.csv")
        
        return results_by_tau
    
    @staticmethod
    def dimension_sweep(n_sim=20, T=100):
        """
        Experiment 3: Context Dimension Sweep
        
        Research Question: How do algorithms scale with dimension?
        
        Tests d = [5, 10, 20, 30] with Gaussian beta
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: Context Dimension Sweep (d)")
        print("="*80)
        print("Research Question: How do algorithms scale with dimension?")
        print("Parameter: d = [5, 10, 20, 30]")
        print("Beta: Gaussian(0, 1)")
        print("="*80 + "\n")
        
        d_values = [5, 10, 20, 30]
        results_by_d = {}
        
        for d in d_values:
            print(f"\n{'='*60}")
            print(f"Running d = {d}")
            print('='*60)
            
            study = Project4Simulation(
                n_sim=n_sim,
                K=2,
                d=d,
                T=T,
                tau=0.5,
                algorithms='all',
                beta_generator=NormalGenerator(mean=0, std=1),
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(0, 1),
                random_seed=42
            )
            
            results = study.run_simulation()
            results_by_d[d] = results
            
            study.save_results(f'results/project4/experiments/dim_sweep_d{d}.pkl')
        
        # Save summary
        summary = _create_dim_summary(results_by_d, d_values)
        summary.to_csv('results/project4/experiments/dim_sweep_summary.csv', index=False)
        
        print("\n✓ Dimension sweep complete")
        print(f"Summary saved to: results/project4/experiments/dim_sweep_summary.csv")
        
        return results_by_d
    
    @staticmethod
    def arm_count_sweep(n_sim=20, T=100):
        """
        Experiment 4: Number of Arms Sweep
        
        Research Question: How do algorithms scale with number of arms?
        
        Tests K = [2, 3, 5, 8] with heterogeneous beta
        """
        print("\n" + "="*80)
        print("EXPERIMENT 4: Number of Arms Sweep (K)")
        print("="*80)
        print("Research Question: How do algorithms scale with more arms?")
        print("Parameter: K = [2, 3, 5, 8]")
        print("Beta: Heterogeneous (different per arm)")
        print("="*80 + "\n")
        
        K_values = [2, 3, 5, 8]
        results_by_K = {}
        
        for K in K_values:
            print(f"\n{'='*60}")
            print(f"Running K = {K}")
            print('='*60)
            
            # Create heterogeneous beta generators
            beta_gens = []
            for k in range(K):
                if k % 2 == 0:
                    beta_gens.append(NormalGenerator(mean=0, std=1))
                else:
                    beta_gens.append(UniformGenerator(low=0.5, high=1.5))
            
            study = Project4Simulation(
                n_sim=n_sim,
                K=K,
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
            results_by_K[K] = results
            
            study.save_results(f'results/project4/experiments/arms_sweep_K{K}.pkl')
        
        # Save summary
        summary = _create_K_summary(results_by_K, K_values)
        summary.to_csv('results/project4/experiments/arms_sweep_summary.csv', index=False)
        
        print("\n✓ Arms sweep complete")
        print(f"Summary saved to: results/project4/experiments/arms_sweep_summary.csv")
        
        return results_by_K
    
    @staticmethod
    def beta_generation_comparison(n_sim=30, T=150):
        """
        Experiment 5: Beta Generation Strategy Comparison
        
        Research Question: How do different beta distributions affect
        algorithm performance?
        
        Tests: Uniform, Gaussian, Heavy-tailed, Sparse
        """
        print("\n" + "="*80)
        print("EXPERIMENT 5: Beta Generation Strategy Comparison")
        print("="*80)
        print("Research Question: How do beta distributions affect performance?")
        print("Strategies: Uniform, Gaussian, Heavy-tailed, Sparse")
        print("="*80 + "\n")
        
        strategies = {
            'uniform': UniformGenerator(low=0.5, high=1.5),
            'gaussian': NormalGenerator(mean=0, std=1),
            'heavy_tailed': TGenerator(df=3, scale=1),
            'sparse': UniformGenerator(low=-0.3, high=0.3)
        }
        
        results_by_strategy = {}
        
        for name, beta_gen in strategies.items():
            print(f"\n{'='*60}")
            print(f"Running strategy: {name}")
            print('='*60)
            
            study = Project4Simulation(
                n_sim=n_sim,
                K=2,
                d=10,
                T=T,
                tau=0.5,
                algorithms='all',
                beta_generator=beta_gen,
                err_generator=TGenerator(df=2.25, scale=0.7),
                context_generator=TruncatedNormalGenerator(0, 1),
                random_seed=42
            )
            
            results = study.run_simulation()
            results_by_strategy[name] = results
            
            study.save_results(f'results/project4/experiments/beta_strategy_{name}.pkl')
        
        # Save summary
        summary = _create_strategy_summary(results_by_strategy)
        summary.to_csv('results/project4/experiments/beta_strategy_summary.csv', index=False)
        
        print("\n✓ Beta generation comparison complete")
        print(f"Summary saved to: results/project4/experiments/beta_strategy_summary.csv")
        
        return results_by_strategy


def _create_df_summary(results_by_df, df_values):
    """Create summary table for df sweep."""
    summary_data = []
    
    for df in df_values:
        results = results_by_df[df]
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]  # Final regret
            beta_err = results['beta_errors'][alg_name][:, -1, :]  # Final beta error
            
            summary_data.append({
                'df': df,
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'runtime': results['computation_time'][alg_name]
            })
    
    return pd.DataFrame(summary_data)


def _create_tau_summary(results_by_tau, tau_values):
    """Create summary table for tau sweep."""
    summary_data = []
    
    for tau in tau_values:
        results = results_by_tau[tau]
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]
            beta_err = results['beta_errors'][alg_name][:, -1, :]
            
            summary_data.append({
                'tau': tau,
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'runtime': results['computation_time'][alg_name]
            })
    
    return pd.DataFrame(summary_data)


def _create_dim_summary(results_by_d, d_values):
    """Create summary table for dimension sweep."""
    summary_data = []
    
    for d in d_values:
        results = results_by_d[d]
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]
            beta_err = results['beta_errors'][alg_name][:, -1, :]
            
            summary_data.append({
                'd': d,
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'runtime': results['computation_time'][alg_name]
            })
    
    return pd.DataFrame(summary_data)


def _create_K_summary(results_by_K, K_values):
    """Create summary table for K sweep."""
    summary_data = []
    
    for K in K_values:
        results = results_by_K[K]
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]
            beta_err = results['beta_errors'][alg_name][:, -1, :]
            
            summary_data.append({
                'K': K,
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'runtime': results['computation_time'][alg_name]
            })
    
    return pd.DataFrame(summary_data)


def _create_strategy_summary(results_by_strategy):
    """Create summary table for beta strategy comparison."""
    summary_data = []
    
    for strategy, results in results_by_strategy.items():
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]
            beta_err = results['beta_errors'][alg_name][:, -1, :]
            
            summary_data.append({
                'beta_strategy': strategy,
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'runtime': results['computation_time'][alg_name]
            })
    
    return pd.DataFrame(summary_data)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Project 4 Batch Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project4_experiments.py --experiment df_sweep         # Vary df
  python project4_experiments.py --experiment tau_sweep        # Vary τ
  python project4_experiments.py --experiment dimension_sweep  # Vary d
  python project4_experiments.py --experiment arms_sweep       # Vary K
  python project4_experiments.py --experiment beta_comparison  # Compare beta strategies
  python project4_experiments.py --experiment all              # Run all experiments
  python project4_experiments.py --experiment quick            # Quick test
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='df_sweep',
        choices=['df_sweep', 'tau_sweep', 'dimension_sweep', 'arms_sweep',
                 'beta_comparison', 'all', 'quick'],
        help='Experiment to run'
    )
    
    parser.add_argument('--n_sim', type=int, default=None)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    
    # Set parameters
    if args.quick or args.experiment == 'quick':
        n_sim = 5
        T = 50
        print("\n⚡ QUICK TEST MODE")
    else:
        n_sim = args.n_sim if args.n_sim is not None else 30
        T = args.T if args.T is not None else 150
    
    print("\n" + "="*80)
    print("PROJECT 4 EXPERIMENTS")
    print("="*80)
    print(f"Configuration: n_sim={n_sim}, T={T}")
    print(f"Experiment: {args.experiment}")
    print("="*80)
    
    start_time = time.time()
    experiments = Project4Experiments()
    
    if args.experiment == 'all':
        print("\nRunning ALL experiments (this will take a LONG time)...\n")
        
        experiments.df_sweep(n_sim=n_sim, T=T)
        experiments.tau_sweep(n_sim=n_sim, T=T)
        experiments.dimension_sweep(n_sim=20, T=100)
        experiments.arm_count_sweep(n_sim=20, T=100)
        experiments.beta_generation_comparison(n_sim=n_sim, T=T)
        
        print("\n✓ ALL EXPERIMENTS COMPLETE")
    else:
        exp_map = {
            'df_sweep': experiments.df_sweep,
            'tau_sweep': experiments.tau_sweep,
            'dimension_sweep': experiments.dimension_sweep,
            'arms_sweep': experiments.arm_count_sweep,
            'beta_comparison': experiments.beta_generation_comparison
        }
        
        exp_map[args.experiment](n_sim=n_sim, T=T)
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print("\nResults saved to: results/project4/experiments/")
    print("Run 'python project4_analysis.py' to analyze results.")


if __name__ == "__main__":
    main()