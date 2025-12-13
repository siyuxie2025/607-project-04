"""
Project 4 Simulation Framework
==============================

Multi-algorithm bandit simulation with flexible generation support.
Compares 4 algorithms:
1. LinUCB with quantile regression
2. Thompson Sampling with quantile regression
3. Epsilon-Greedy with quantile regression
4. Forced Sampling (baseline from Projects 2-3)

Usage:
    from project4_simulation import Project4Simulation

    study = Project4Simulation(
        n_sim=50, K=2, d=10, T=200, tau=0.5,
        algorithms='all',
        beta_generator=NormalGenerator(mean=0, std=1),
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(0, 1),
        random_seed=42
    )
    results = study.run_simulation()
    study.plot_comparison(save_path='results/comparison.pdf')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import time
from pathlib import Path

# Import algorithm implementations
from project4_methods import (
    ForcedSamplingQuantile,
    LinUCBQuantile,
    EpsilonGreedyQuantile,
    ThompsonSamplingQuantile
)

# Import generators
from generators import (
    NormalGenerator,
    UniformGenerator,
    TGenerator,
    TruncatedNormalGenerator
)


class Project4Simulation:
    """
    Multi-algorithm bandit simulation framework for Project 4.

    Compares performance of 4 algorithms with quantile regression:
    - LinUCB: Upper confidence bound with exploration bonus
    - ThompsonSampling: Posterior sampling for exploration
    - EpsilonGreedy: Random exploration with probability epsilon
    - ForcedSampling: Periodic forced exploration (baseline)

    Supports flexible beta/alpha generation for different scenarios.
    """

    def __init__(
        self,
        n_sim: int,
        K: int,
        d: int,
        T: int,
        tau: float,
        algorithms: str = 'all',
        beta_generator=None,
        alpha_generator=None,
        err_generator=None,
        context_generator=None,
        # Algorithm-specific parameters
        linucb_alpha: float = 1.0,
        epsilon: float = 0.1,
        epsilon_decay: bool = True,
        forced_q: int = 2,
        forced_h: float = 0.5,
        random_seed: int = None
    ):
        """
        Initialize Project 4 simulation study.

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
            Target quantile level (0.5 for median)
        algorithms : str or list
            Which algorithms to run. Options:
            - 'all': Run all 4 algorithms
            - List of names: ['LinUCB', 'ThompsonSampling', 'EpsilonGreedy', 'ForcedSampling']
        beta_generator : DataGenerator or list, optional
            Generator(s) for beta coefficients
        alpha_generator : DataGenerator or list, optional
            Generator(s) for alpha intercepts
        err_generator : DataGenerator
            Generator for error terms
        context_generator : DataGenerator
            Generator for context vectors
        linucb_alpha : float
            Exploration parameter for LinUCB
        epsilon : float
            Initial exploration probability for epsilon-greedy
        epsilon_decay : bool
            Whether to decay epsilon over time
        forced_q : int
            Number of forced samples per arm per round
        forced_h : float
            Threshold for action selection in forced sampling
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.n_sim = n_sim
        self.K = K
        self.d = d
        self.T = T
        self.tau = tau
        self.random_seed = random_seed

        # Algorithm parameters
        self.linucb_alpha = linucb_alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.forced_q = forced_q
        self.forced_h = forced_h

        # Set up random state
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
            np.random.seed(random_seed)
        else:
            self.rng = np.random.default_rng()

        # Set up generators
        self.beta_generator = beta_generator
        self.alpha_generator = alpha_generator

        if err_generator is None:
            self.err_generator = TGenerator(df=2.25, scale=0.7)
        else:
            self.err_generator = err_generator

        if context_generator is None:
            self.context_generator = TruncatedNormalGenerator(mean=0, std=1)
        else:
            self.context_generator = context_generator

        # Compute error quantile for reward generation
        self.q_err = np.quantile(
            self.err_generator.generate(2000, rng=self.rng),
            self.tau
        )

        # Generate true parameters
        self.beta_real = self._generate_beta_values()
        self.alpha_real = self._generate_alpha_values()

        # Set up algorithms to run
        if algorithms == 'all':
            self.algorithm_names = ['LinUCB', 'ThompsonSampling', 'EpsilonGreedy', 'ForcedSampling']
        else:
            self.algorithm_names = algorithms

        # Results storage
        self.results = None

    def _generate_beta_values(self):
        """Generate true beta coefficient values for all arms."""
        if self.beta_generator is None:
            # Default: Uniform distribution
            beta1 = UniformGenerator(low=0.5, high=1.0).generate(
                self.K // 2 * self.d, rng=self.rng
            ).reshape(self.K // 2, self.d)
            beta2 = UniformGenerator(low=1.0, high=1.5).generate(
                (self.K - self.K // 2) * self.d, rng=self.rng
            ).reshape(self.K - self.K // 2, self.d)
            return np.vstack([beta1, beta2])

        elif isinstance(self.beta_generator, list):
            # List of generators - one per arm
            if len(self.beta_generator) != self.K:
                raise ValueError(f"beta_generator list must have length K={self.K}")

            beta_values = []
            for k in range(self.K):
                beta_k = self.beta_generator[k].generate(self.d, rng=self.rng)
                if beta_k.ndim > 1:
                    beta_k = beta_k.ravel()
                beta_values.append(beta_k)
            return np.array(beta_values)

        else:
            # Single generator for all arms
            beta_values = []
            for k in range(self.K):
                beta_k = self.beta_generator.generate(self.d, rng=self.rng)
                if beta_k.ndim > 1:
                    beta_k = beta_k.ravel()
                beta_values.append(beta_k)
            return np.array(beta_values)

    def _generate_alpha_values(self):
        """Generate true alpha intercept values for all arms."""
        if self.alpha_generator is None:
            # Default: Uniform distribution
            alpha1 = UniformGenerator(low=0.5, high=1.0).generate(
                self.K // 2, rng=self.rng
            )
            alpha2 = UniformGenerator(low=1.0, high=1.5).generate(
                self.K - self.K // 2, rng=self.rng
            )
            return np.concatenate([alpha1, alpha2])

        elif isinstance(self.alpha_generator, list):
            # List of generators - one per arm
            if len(self.alpha_generator) != self.K:
                raise ValueError(f"alpha_generator list must have length K={self.K}")

            alpha_values = []
            for k in range(self.K):
                alpha_k = self.alpha_generator[k].generate(1, rng=self.rng)
                if hasattr(alpha_k, '__iter__'):
                    alpha_k = alpha_k[0]
                alpha_values.append(alpha_k)
            return np.array(alpha_values)

        else:
            # Single generator for all arms
            alpha_values = []
            for k in range(self.K):
                alpha_k = self.alpha_generator.generate(1, rng=self.rng)
                if hasattr(alpha_k, '__iter__'):
                    alpha_k = alpha_k[0]
                alpha_values.append(alpha_k)
            return np.array(alpha_values)

    def _create_algorithm(self, name):
        """Create an algorithm instance by name."""
        if name == 'LinUCB':
            return LinUCBQuantile(
                K=self.K, d=self.d, alpha=self.linucb_alpha, tau=self.tau,
                beta_real=self.beta_real, alpha_real=self.alpha_real
            )
        elif name == 'ThompsonSampling':
            return ThompsonSamplingQuantile(
                K=self.K, d=self.d, tau=self.tau,
                beta_real=self.beta_real, alpha_real=self.alpha_real
            )
        elif name == 'EpsilonGreedy':
            return EpsilonGreedyQuantile(
                K=self.K, d=self.d, epsilon=self.epsilon, tau=self.tau,
                beta_real=self.beta_real, alpha_real=self.alpha_real,
                decay=self.epsilon_decay
            )
        elif name == 'ForcedSampling':
            return ForcedSamplingQuantile(
                K=self.K, d=self.d, q=self.forced_q, h=self.forced_h, tau=self.tau,
                beta_real=self.beta_real, alpha_real=self.alpha_real
            )
        else:
            raise ValueError(f"Unknown algorithm: {name}")

    def _run_one_scenario(self, alg_name):
        """Run one simulation scenario for a single algorithm."""
        alg = self._create_algorithm(alg_name)

        rewards = []
        optimal_rewards = []
        beta_errors = []

        for t in range(1, self.T + 1):
            # Generate context
            x = self.context_generator.generate(self.d, rng=self.rng)
            if x.ndim > 1:
                x = x.ravel()

            # Choose action
            action = alg.choose_action(t, x)

            # Generate reward
            true_reward = np.dot(self.beta_real[action], x) + self.alpha_real[action]
            noise = self.err_generator.generate(1, rng=self.rng)[0] - self.q_err
            noisy_reward = true_reward + (0.5 * x[-1] + 1) * noise

            # Update algorithm
            alg.update(x, action, noisy_reward, t)

            # Track metrics
            rewards.append(true_reward)
            opt_reward = np.max(np.dot(self.beta_real, x) + self.alpha_real)
            optimal_rewards.append(opt_reward)
            beta_errors.append(alg.get_beta_errors())

        # Compute cumulative regret
        regret = np.cumsum(optimal_rewards) - np.cumsum(rewards)

        return regret, np.array(beta_errors)

    def run_simulation(self):
        """
        Run the full simulation study.

        Returns
        -------
        dict
            Results dictionary containing:
            - 'algorithms': list of algorithm names
            - 'regret': dict mapping algorithm name to regret array (n_sim, T)
            - 'beta_errors': dict mapping algorithm name to error array (n_sim, T, K)
            - 'computation_time': dict mapping algorithm name to runtime
        """
        results = {
            'algorithms': self.algorithm_names,
            'regret': {},
            'beta_errors': {},
            'computation_time': {},
            'config': {
                'n_sim': self.n_sim,
                'K': self.K,
                'd': self.d,
                'T': self.T,
                'tau': self.tau,
                'random_seed': self.random_seed
            }
        }

        for alg_name in self.algorithm_names:
            print(f"\nRunning {alg_name}...")
            start_time = time.time()

            all_regret = []
            all_beta_errors = []

            for sim in tqdm(range(self.n_sim), desc=f"{alg_name}"):
                regret, beta_errors = self._run_one_scenario(alg_name)
                all_regret.append(regret)
                all_beta_errors.append(beta_errors)

            elapsed = time.time() - start_time

            results['regret'][alg_name] = np.array(all_regret)
            results['beta_errors'][alg_name] = np.array(all_beta_errors)
            results['computation_time'][alg_name] = elapsed

            print(f"  {alg_name} completed in {elapsed:.1f}s")
            print(f"  Final mean regret: {np.mean(all_regret, axis=0)[-1]:.2f}")

        self.results = results
        return results

    def save_results(self, filepath):
        """Save results to pickle file."""
        if self.results is None:
            raise ValueError("No results to save. Run simulation first.")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)

        print(f"Results saved to: {filepath}")

    @staticmethod
    def load_results(filepath):
        """Load results from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def plot_comparison(self, results=None, save_path=None, figsize=(14, 10)):
        """
        Plot comparison of all algorithms.

        Parameters
        ----------
        results : dict, optional
            Results to plot. If None, uses self.results
        save_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if results is None:
            if self.results is None:
                raise ValueError("No results to plot. Run simulation first.")
            results = self.results

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        colors = {
            'LinUCB': 'blue',
            'ThompsonSampling': 'green',
            'EpsilonGreedy': 'orange',
            'ForcedSampling': 'red'
        }

        steps = np.arange(1, self.T + 1)

        # Plot 1: Cumulative Regret
        ax1 = axes[0, 0]
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name]
            mean_regret = np.mean(regret, axis=0)
            std_regret = np.std(regret, axis=0)

            color = colors.get(alg_name, 'black')
            ax1.plot(steps, mean_regret, label=alg_name, color=color, linewidth=2)
            ax1.fill_between(steps, mean_regret - std_regret, mean_regret + std_regret,
                            color=color, alpha=0.2)

        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Cumulative Regret', fontsize=12)
        ax1.set_title('Cumulative Regret Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Beta Estimation Error
        ax2 = axes[0, 1]
        for alg_name in results['algorithms']:
            beta_err = results['beta_errors'][alg_name]
            # Average across arms
            beta_err_avg = np.mean(beta_err, axis=2)
            mean_err = np.mean(beta_err_avg, axis=0)
            std_err = np.std(beta_err_avg, axis=0)

            color = colors.get(alg_name, 'black')
            ax2.plot(steps, mean_err, label=alg_name, color=color, linewidth=2)
            ax2.fill_between(steps, mean_err - std_err, mean_err + std_err,
                            color=color, alpha=0.2)

        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Beta Estimation Error', fontsize=12)
        ax2.set_title('Beta Error Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Final Regret Box Plot
        ax3 = axes[1, 0]
        final_regrets = [results['regret'][alg][:, -1] for alg in results['algorithms']]
        bp = ax3.boxplot(final_regrets, labels=results['algorithms'], patch_artist=True)

        for i, alg_name in enumerate(results['algorithms']):
            bp['boxes'][i].set_facecolor(colors.get(alg_name, 'gray'))
            bp['boxes'][i].set_alpha(0.6)

        ax3.set_ylabel('Final Regret', fontsize=12)
        ax3.set_title('Final Regret Distribution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Runtime Comparison
        ax4 = axes[1, 1]
        runtimes = [results['computation_time'][alg] for alg in results['algorithms']]
        bars = ax4.bar(results['algorithms'], runtimes,
                      color=[colors.get(alg, 'gray') for alg in results['algorithms']],
                      alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

        ax4.set_ylabel('Runtime (seconds)', fontsize=12)
        ax4.set_title('Computation Time', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Project 4: Algorithm Comparison (K={self.K}, d={self.d}, T={self.T}, n_sim={self.n_sim})',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.close()

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        for alg_name in results['algorithms']:
            final_regret = results['regret'][alg_name][:, -1]
            print(f"\n{alg_name}:")
            print(f"  Mean Final Regret: {np.mean(final_regret):.2f} +/- {np.std(final_regret):.2f}")
            print(f"  Median Final Regret: {np.median(final_regret):.2f}")
            print(f"  Runtime: {results['computation_time'][alg_name]:.1f}s")


if __name__ == "__main__":
    # Quick test
    print("Testing Project4Simulation...")

    study = Project4Simulation(
        n_sim=3,
        K=2,
        d=5,
        T=50,
        tau=0.5,
        algorithms='all',
        beta_generator=NormalGenerator(mean=0, std=1),
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(0, 1),
        random_seed=42
    )

    print(f"\nBeta real values:\n{study.beta_real}")
    print(f"Alpha real values: {study.alpha_real}")

    results = study.run_simulation()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
