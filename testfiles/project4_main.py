"""
Project 4: Multi-Algorithm Quantile Bandit Comparison
======================================================

Compares four bandit algorithms with quantile regression updates:
1. Forced Sampling (from Project 3)
2. LinUCB with quantile regression
3. Epsilon-greedy with quantile regression  
4. Thompson Sampling with quantile regression

All algorithms tested in both low-dimensional and high-dimensional settings.

Usage:
    python project4_main.py --config configs/default.json
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import sys

# Import existing infrastructure
sys.path.insert(0, 'src')
from generators import TGenerator, TruncatedNormalGenerator

# Import new algorithms (to be created)
from project4_methods import (
    ForcedSamplingQuantile,
    LinUCBQuantile, 
    EpsilonGreedyQuantile,
    ThompsonSamplingQuantile
)


class Project4Simulation:
    """
    Multi-algorithm comparison framework for quantile bandits.
    
    Extends Project 3 infrastructure to compare multiple algorithms
    systematically across different error distributions and dimensionalities.
    """
    
    def __init__(
        self,
        n_sim: int,
        K: int,
        d: int,
        T: int,
        tau: float,
        err_generator,
        context_generator,
        algorithms: List[str],
        high_dim: bool = False,
        random_seed: int = None
    ):
        """
        Initialize multi-algorithm simulation.
        
        Parameters
        ----------
        n_sim : int
            Number of simulation replications
        K : int
            Number of arms
        d : int
            Context dimension
        T : int
            Time horizon
        tau : float
            Target quantile level
        err_generator : DataGenerator
            Error distribution generator
        context_generator : DataGenerator
            Context vector generator
        algorithms : list of str
            Algorithm names to compare
        high_dim : bool
            Whether to use high-dimensional methods (with Lasso)
        random_seed : int
            Random seed for reproducibility
        """
        self.n_sim = n_sim
        self.K = K
        self.d = d
        self.T = T
        self.tau = tau
        self.err_generator = err_generator
        self.context_generator = context_generator
        self.algorithms = algorithms
        self.high_dim = high_dim
        self.random_seed = random_seed
        
        # Set random seed
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
            np.random.seed(random_seed)
        else:
            self.rng = np.random.default_rng()
        
        # Generate true parameters (same for all algorithms)
        self._generate_true_parameters()
        
        # Precompute quantile error
        self.q_err = np.quantile(
            self.err_generator.generate(2000, rng=self.rng), 
            self.tau
        )
        
        # Results storage
        self.results = None
        
    def _generate_true_parameters(self):
        """Generate true beta and alpha parameters."""
        # For low-dimensional: use dense parameters
        # For high-dimensional: use sparse parameters
        
        if not self.high_dim:
            # Dense parameters (Project 3 style)
            beta1 = self.rng.uniform(0.5, 1.5, (self.K//2, self.d))
            beta2 = self.rng.uniform(1.0, 2.0, (self.K - self.K//2, self.d))
            self.beta_real = np.vstack([beta1, beta2])
            
            self.alpha_real = self.rng.uniform(0.5, 1.0, self.K)
        else:
            # Sparse parameters for high-dimensional setting
            # Only s << d features are truly relevant
            s = min(10, self.d // 5)  # Sparsity level
            
            self.beta_real = np.zeros((self.K, self.d))
            for k in range(self.K):
                # Randomly select s features
                active_features = self.rng.choice(self.d, size=s, replace=False)
                # Assign non-zero coefficients
                self.beta_real[k, active_features] = self.rng.uniform(0.5, 2.0, s)
            
            self.alpha_real = self.rng.uniform(0.5, 1.0, self.K)
    
    def _initialize_algorithm(self, algo_name: str):
        """Initialize a specific algorithm."""
        common_params = {
            'K': self.K,
            'd': self.d,
            'tau': self.tau,
            'beta_real': self.beta_real,
            'alpha_real': self.alpha_real,
            'high_dim': self.high_dim
        }
        
        if algo_name == 'ForcedSampling':
            return ForcedSamplingQuantile(
                q=2, h=0.5, **common_params
            )
        elif algo_name == 'LinUCB':
            return LinUCBQuantile(
                alpha=1.0, **common_params
            )
        elif algo_name == 'EpsilonGreedy':
            return EpsilonGreedyQuantile(
                epsilon=0.1, decay=True, **common_params
            )
        elif algo_name == 'ThompsonSampling':
            return ThompsonSamplingQuantile(
                **common_params
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    def run_one_replication(self, algo_name: str, verbose: bool = False):
        """
        Run one simulation replication for a specific algorithm.
        
        Returns
        -------
        dict
            Contains: cumulative_regret, beta_errors, actions
        """
        # Initialize algorithm
        algorithm = self._initialize_algorithm(algo_name)
        
        # Storage for this replication
        cumulative_regret = np.zeros(self.T)
        beta_errors = np.zeros((self.T, self.K))
        actions = np.zeros(self.T, dtype=int)
        
        # Run simulation
        iterator = tqdm(range(1, self.T + 1), desc=f"{algo_name}", leave=False) if verbose else range(1, self.T + 1)
        
        for t in iterator:
            # Generate context
            x = self.context_generator.generate(self.d, rng=self.rng)
            if x.ndim > 1:
                x = x.ravel()
            
            # Choose action
            action = algorithm.choose_action(t, x)
            actions[t-1] = action
            
            # Generate reward
            true_reward = np.dot(self.beta_real[action], x) + self.alpha_real[action]
            heteroskedastic_noise = (0.5 * x[-1] + 1) * (
                self.err_generator.generate(1, rng=self.rng)[0] - self.q_err
            )
            noisy_reward = true_reward + heteroskedastic_noise
            
            # Update algorithm
            algorithm.update(x, action, noisy_reward, t)
            
            # Compute regret
            optimal_reward = np.max(
                np.dot(self.beta_real, x) + self.alpha_real
            )
            instant_regret = optimal_reward - true_reward
            cumulative_regret[t-1] = instant_regret if t == 1 else cumulative_regret[t-2] + instant_regret
            
            # Record beta errors
            beta_errors[t-1, :] = algorithm.get_beta_errors()
        
        return {
            'cumulative_regret': cumulative_regret,
            'beta_errors': beta_errors,
            'actions': actions
        }
    
    def run_simulation(self, n_jobs: int = 1, verbose: bool = True):
        """
        Run full simulation comparing all algorithms.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (not implemented yet, kept for compatibility)
        verbose : bool
            Print progress information
        
        Returns
        -------
        dict
            Results for all algorithms
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"PROJECT 4: MULTI-ALGORITHM COMPARISON")
            print(f"{'='*80}")
            print(f"Algorithms: {', '.join(self.algorithms)}")
            print(f"Setting: {'High-dimensional' if self.high_dim else 'Low-dimensional'}")
            print(f"n_sim={self.n_sim}, K={self.K}, d={self.d}, T={self.T}, tau={self.tau}")
            print(f"Error distribution: {self.err_generator.name}")
            print(f"{'='*80}\n")
        
        # Initialize results storage
        results = {}
        
        for algo_name in self.algorithms:
            if verbose:
                print(f"\nRunning {algo_name}...")
            
            # Storage for all replications
            all_regrets = []
            all_beta_errors = []
            all_actions = []
            
            # Run replications
            for sim in tqdm(range(self.n_sim), desc=f"{algo_name} replications"):
                replication_results = self.run_one_replication(
                    algo_name, verbose=False
                )
                
                all_regrets.append(replication_results['cumulative_regret'])
                all_beta_errors.append(replication_results['beta_errors'])
                all_actions.append(replication_results['actions'])
            
            # Store results
            results[algo_name] = {
                'cumulative_regret': np.array(all_regrets),  # (n_sim, T)
                'beta_errors': np.array(all_beta_errors),    # (n_sim, T, K)
                'actions': np.array(all_actions)             # (n_sim, T)
            }
            
            if verbose:
                final_regret = np.mean(all_regrets, axis=0)[-1]
                print(f"  Final regret: {final_regret:.2f}")
        
        self.results = results
        return results
    
    def save_results(self, filepath: str = None):
        """Save results to disk."""
        if self.results is None:
            raise ValueError("No results to save. Run simulation first.")
        
        if filepath is None:
            filepath = f'results/project4_results_d{self.d}_K{self.K}_T{self.T}.pkl'
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"\nResults saved to: {filepath}")
        
        # Save metadata
        metadata = {
            'n_sim': self.n_sim,
            'K': self.K,
            'd': self.d,
            'T': self.T,
            'tau': self.tau,
            'algorithms': self.algorithms,
            'high_dim': self.high_dim,
            'err_generator': self.err_generator.name,
            'random_seed': self.random_seed
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")


def create_comparison_plots(results: Dict, save_dir: str = 'results/figures'):
    """
    Create comparison plots for all algorithms.
    
    Parameters
    ----------
    results : dict
        Results from run_simulation()
    save_dir : str
        Directory to save figures
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    algorithms = list(results.keys())
    n_algos = len(algorithms)
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, n_algos))
    
    # Plot 1: Cumulative Regret Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, algo in enumerate(algorithms):
        regret = results[algo]['cumulative_regret']  # (n_sim, T)
        mean_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0) / np.sqrt(regret.shape[0])
        
        T = regret.shape[1]
        steps = np.arange(1, T + 1)
        
        ax.plot(steps, mean_regret, label=algo, color=colors[i], linewidth=2.5)
        ax.fill_between(
            steps, 
            mean_regret - 1.96 * std_regret,
            mean_regret + 1.96 * std_regret,
            color=colors[i], alpha=0.2
        )
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison: Cumulative Regret', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/regret_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/regret_comparison.pdf")
    plt.close()
    
    # Plot 2: Beta Estimation Error Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, algo in enumerate(algorithms):
        beta_errors = results[algo]['beta_errors']  # (n_sim, T, K)
        # Average across arms
        avg_errors = np.mean(beta_errors, axis=2)  # (n_sim, T)
        mean_error = np.mean(avg_errors, axis=0)
        std_error = np.std(avg_errors, axis=0) / np.sqrt(avg_errors.shape[0])
        
        T = beta_errors.shape[1]
        steps = np.arange(1, T + 1)
        
        ax.plot(steps, mean_error, label=algo, color=colors[i], linewidth=2.5)
        ax.fill_between(
            steps,
            mean_error - 1.96 * std_error,
            mean_error + 1.96 * std_error,
            color=colors[i], alpha=0.2
        )
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Beta Estimation Error', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison: Beta Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/beta_error_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/beta_error_comparison.pdf")
    plt.close()
    
    # Plot 3: Summary Table
    create_summary_table(results, save_dir)


def create_summary_table(results: Dict, save_dir: str):
    """Create summary statistics table."""
    summary_data = []
    
    for algo in results.keys():
        regret = results[algo]['cumulative_regret']
        beta_errors = results[algo]['beta_errors']
        
        final_regret = regret[:, -1]
        final_beta = np.mean(beta_errors[:, -1, :], axis=1)
        
        summary_data.append({
            'Algorithm': algo,
            'Mean_Regret': np.mean(final_regret),
            'Std_Regret': np.std(final_regret),
            'Mean_Beta_Error': np.mean(final_beta),
            'Std_Beta_Error': np.std(final_beta),
            'Min_Regret': np.min(final_regret),
            'Max_Regret': np.max(final_regret)
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Mean_Regret')
    
    # Save to CSV
    csv_path = f'{save_dir}/summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df.to_string(index=False, float_format='%.2f'))
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Project 4: Multi-algorithm comparison')
    parser.add_argument('--n_sim', type=int, default=50, help='Number of simulations')
    parser.add_argument('--K', type=int, default=2, help='Number of arms')
    parser.add_argument('--d', type=int, default=10, help='Context dimension')
    parser.add_argument('--T', type=int, default=500, help='Time horizon')
    parser.add_argument('--df', type=float, default=2.25, help='t-distribution df')
    parser.add_argument('--tau', type=float, default=0.5, help='Quantile level')
    parser.add_argument('--high_dim', action='store_true', help='Use high-dimensional methods')
    parser.add_argument('--seed', type=int, default=1010, help='Random seed')
    
    args = parser.parse_args()
    
    # Create generators
    err_generator = TGenerator(df=args.df, scale=0.7)
    context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
    
    # Define algorithms to compare
    algorithms = ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    
    # Run simulation
    study = Project4Simulation(
        n_sim=args.n_sim,
        K=args.K,
        d=args.d,
        T=args.T,
        tau=args.tau,
        err_generator=err_generator,
        context_generator=context_generator,
        algorithms=algorithms,
        high_dim=args.high_dim,
        random_seed=args.seed
    )
    
    results = study.run_simulation(verbose=True)
    
    # Save results
    study.save_results()
    
    # Create plots
    create_comparison_plots(results)
    
    print("\n" + "="*80)
    print("✓ PROJECT 4 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()