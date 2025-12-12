"""
Main script for running bandit algorithm simulations.

This script compares Risk-Aware Bandit vs OLS Bandit across different
error distributions (t-distributions with varying degrees of freedom).

Generates two main result plots:
1. Cumulative regret comparison across different df values
2. Beta estimation error comparison across different df values
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os

from simulation import SimulationStudy
from parallel_simulation import ParallelSimulationStudy
from generators import NormalGenerator, TGenerator, TruncatedNormalGenerator


def run_experiments(df_values, n_sim=50, K=2, d=10, T=1000, q=5, h=0.5, tau=0.5, random_seed=1010, parallel=True):
    """
    Run simulations across different t-distribution degrees of freedom.
    
    Parameters
    ----------
    df_values : list
        List of degrees of freedom values for t-distribution
    n_sim : int
        Number of simulation replications
    K : int
        Number of arms
    d : int
        Dimension of context vectors
    T : int
        Time horizon
    q : int
        Number of forced samples per arm
    h : float
        Difference threshold
    tau : float
        Quantile level
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Results for each df value
    """
    results_by_df = {}
    
    context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)

    if parallel:

        for df in tqdm(df_values, desc="Running experiments across df values"):
            print(f"\n{'='*60}")
            print(f"Running simulation with t-distribution df={df}")
            print(f"{'='*60}")
            
            err_generator = TGenerator(df=df, scale=0.7)
            
            study = ParallelSimulationStudy(
                n_sim=n_sim, 
                K=K, 
                d=d, 
                T=T, 
                q=q, 
                h=h, 
                tau=tau,
                random_seed=random_seed,
                err_generator=err_generator,
                context_generator=context_generator
            )
            
            results = study.run_simulation(n_jobs=os.cpu_count()-1)
            results_by_df[df] = results
            
            print(f"\nCompleted df={df}")
            print(f"Final regret - RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
            print(f"Final regret - OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f}")
        
        return results_by_df, study.T, study.n_sim, study.K, study.d, study.tau
    
    else:
        for df in tqdm(df_values, desc="Running experiments across df values"):
            print(f"\n{'='*60}")
            print(f"Running simulation with t-distribution df={df}")
            print(f"{'='*60}")
            
            err_generator = TGenerator(df=df, scale=0.7)
            
            study = SimulationStudy(
                n_sim=n_sim, 
                K=K, 
                d=d, 
                T=T, 
                q=q, 
                h=h, 
                tau=tau,
                random_seed=random_seed,
                err_generator=err_generator,
                context_generator=context_generator
            )
            
            results = study.run_simulation()
            results_by_df[df] = results
            
            print(f"\nCompleted df={df}")
            print(f"Final regret - RAB: {np.mean(results['cumulated_regret_RiskAware'][:, -1]):.2f}")
            print(f"Final regret - OLS: {np.mean(results['cumulated_regret_OLS'][:, -1]):.2f}")
        
        return results_by_df, study.T, study.n_sim, study.K, study.d, study.tau


def plot_combined_regret(results_by_df, df_values, T, n_sim, K, d, tau, use_ci=True, ci_level=0.95):
    """
    Plot cumulative regret for all df values in a single figure.
    
    Parameters
    ----------
    results_by_df : dict
        Results dictionary for each df value
    df_values : list
        List of df values
    T : int
        Time horizon
    n_sim : int
        Number of simulations
    K, d, tau : int/float
        Other simulation parameters
    use_ci : bool
        Whether to show confidence intervals
    ci_level : float
        Confidence level
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    steps = np.arange(1, T + 1)
    
    for idx, df in enumerate(df_values):
        ax = axes[idx]
        results = results_by_df[df]
        
        regret_rab = results['cumulated_regret_RiskAware']
        regret_ols = results['cumulated_regret_OLS']
        
        # Compute means
        mean_rab = np.mean(regret_rab, axis=0)
        mean_ols = np.mean(regret_ols, axis=0)
        
        if use_ci:
            se_rab = np.std(regret_rab, axis=0, ddof=1) / np.sqrt(n_sim+0.01)
            se_ols = np.std(regret_ols, axis=0, ddof=1) / np.sqrt(n_sim+0.01)
            
            t_crit = stats.t.ppf((1 + ci_level) / 2., n_sim - 1)
            
            lower_rab = mean_rab - t_crit * se_rab
            upper_rab = mean_rab + t_crit * se_rab
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols
            
            ax.fill_between(steps, lower_rab, upper_rab, color='red', alpha=0.2)
            ax.fill_between(steps, lower_ols, upper_ols, color='blue', alpha=0.2)
        
        ax.plot(steps, mean_rab, 'r-', label='Risk Aware', linewidth=2.5)
        ax.plot(steps, mean_ols, 'b--', label='OLS', linewidth=2.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Cumulative Regret', fontsize=11)
        ax.set_title(f't-distribution (df={df})', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final regret values as text
        final_rab = mean_rab[-1]
        final_ols = mean_ols[-1]
        ax.text(0.95, 0.05, f'Final:\nRAB: {final_rab:.1f}\nOLS: {final_ols:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the 6th subplot if only 5 df values
    if len(df_values) < 6:
        axes[-1].axis('off')
    
    fig.suptitle(f'Cumulative Regret Comparison Across Error Distributions\n(K={K}, d={d}, T={T}, τ={tau})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/main_regret_comparison_K{K}_d{d}_T{T}.pdf', 
                bbox_inches='tight', dpi=300)
    print(f"\nSaved: results/main_regret_comparison_K{K}_d{d}_T{T}.pdf")
    plt.show()


def plot_combined_beta_error(results_by_df, df_values, T, n_sim, K, d, tau, use_ci=True, ci_level=0.95):
    """
    Plot beta estimation error for all df values in a single figure.
    
    Parameters
    ----------
    results_by_df : dict
        Results dictionary for each df value
    df_values : list
        List of df values
    T : int
        Time horizon
    n_sim : int
        Number of simulations
    K, d, tau : int/float
        Other simulation parameters
    use_ci : bool
        Whether to show confidence intervals
    ci_level : float
        Confidence level
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    steps = np.arange(1, T + 1)
    
    for idx, df in enumerate(df_values):
        ax = axes[idx]
        results = results_by_df[df]
        
        beta_errors_rab = results['beta_errors_rab']  # Shape: (n_sim, T, K)
        beta_errors_ols = results['beta_errors_ols']  # Shape: (n_sim, T, K)
        
        # Average across arms
        rab_avg = np.mean(beta_errors_rab, axis=2)  # Shape: (n_sim, T)
        ols_avg = np.mean(beta_errors_ols, axis=2)  # Shape: (n_sim, T)
        
        # Mean across simulations
        mean_rab = np.mean(rab_avg, axis=0)
        mean_ols = np.mean(ols_avg, axis=0)
        
        if use_ci:
            se_rab = np.std(rab_avg, axis=0, ddof=1) / np.sqrt(n_sim+0.01)
            se_ols = np.std(ols_avg, axis=0, ddof=1) / np.sqrt(n_sim+0.01)
            
            t_crit = stats.t.ppf((1 + ci_level) / 2, n_sim - 1)
            
            lower_rab = mean_rab - t_crit * se_rab
            upper_rab = mean_rab + t_crit * se_rab
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols
            
            ax.fill_between(steps, lower_rab, upper_rab, color='red', alpha=0.2)
            ax.fill_between(steps, lower_ols, upper_ols, color='blue', alpha=0.2)
        
        ax.plot(steps, mean_rab, 'r-', label='Risk Aware', linewidth=2.5)
        ax.plot(steps, mean_ols, 'b--', label='OLS', linewidth=2.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Beta Estimation Error', fontsize=11)
        ax.set_title(f't-distribution (df={df})', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final error values as text
        final_rab = mean_rab[-1]
        final_ols = mean_ols[-1]
        improvement = ((final_ols - final_rab) / final_ols * 100) if final_ols > final_rab else 0
        ax.text(0.95, 0.95, f'Final Error:\nRAB: {final_rab:.4f}\nOLS: {final_ols:.4f}\nΔ: {improvement:.1f}%',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Hide the 6th subplot if only 5 df values
    if len(df_values) < 6:
        axes[-1].axis('off')
    
    fig.suptitle(f'Beta Estimation Error Comparison Across Error Distributions\n(K={K}, d={d}, T={T}, τ={tau})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/main_beta_error_comparison_K{K}_d{d}_T{T}.pdf', 
                bbox_inches='tight', dpi=300)
    print(f"\nSaved: results/main_beta_error_comparison_K{K}_d{d}_T{T}.pdf")
    plt.show()


def print_summary_table(results_by_df, df_values, T):
    """Print a summary table comparing performance across df values."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Final Performance Comparison")
    print("="*80)
    print(f"{'df':<8} {'RAB Regret':<15} {'OLS Regret':<15} {'RAB Beta Err':<15} {'OLS Beta Err':<15} {'Winner':<10}")
    print("-"*80)
    
    for df in df_values:
        results = results_by_df[df]
        
        # Regret
        final_regret_rab = np.mean(results['cumulated_regret_RiskAware'][:, -1])
        final_regret_ols = np.mean(results['cumulated_regret_OLS'][:, -1])
        
        # Beta error (averaged across arms)
        beta_rab = np.mean(results['beta_errors_rab'][:, -1, :])
        beta_ols = np.mean(results['beta_errors_ols'][:, -1, :])
        
        # Determine winner
        regret_winner = "RAB" if final_regret_rab < final_regret_ols else "OLS"
        beta_winner = "RAB" if beta_rab < beta_ols else "OLS"
        overall_winner = f"{regret_winner}(R), {beta_winner}(B)"
        
        print(f"{df:<8} {final_regret_rab:<15.2f} {final_regret_ols:<15.2f} "
              f"{beta_rab:<15.4f} {beta_ols:<15.4f} {overall_winner:<10}")
    
    print("="*80)
    print("Note: (R) = Regret winner, (B) = Beta error winner")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # ==================== CONFIGURATION ====================
    # Degrees of freedom for t-distribution (controls tail heaviness)
    df_values = [1.5, 2.25, 3, 5, 10]  # Lower df = heavier tails
    
    # Simulation parameters
    n_sim = 50      # Number of simulation replications
    K = 2          # Number of arms
    d = 10         # Dimension of context vectors
    T = 1000        # Time horizon
    q = 2          # Number of forced samples per arm per round
    h = 0.5        # Threshold for action selection
    tau = 0.5      # Quantile level for risk-aware approach
    random_seed = 1010  # For reproducibility
    
    print("="*80)
    print("BANDIT ALGORITHM COMPARISON: Risk-Aware vs OLS")
    print("="*80)
    print(f"Configuration:")
    print(f"  - t-distribution df values: {df_values}")
    print(f"  - Number of simulations: {n_sim}")
    print(f"  - Number of arms (K): {K}")
    print(f"  - Context dimension (d): {d}")
    print(f"  - Time horizon (T): {T}")
    print(f"  - Forced samples per arm (q): {q}")
    print(f"  - Threshold (h): {h}")
    print(f"  - Quantile level (τ): {tau}")
    print(f"  - Random seed: {random_seed}")
    print("="*80 + "\n")
    
    # ==================== RUN EXPERIMENTS ====================
    results_by_df, T, n_sim, K, d, tau = run_experiments(
        df_values=df_values,
        n_sim=n_sim,
        K=K,
        d=d,
        T=T,
        q=q,
        h=h,
        tau=tau,
        random_seed=random_seed
    )
    
    # ==================== GENERATE PLOTS ====================
    print("\n" + "="*80)
    print("GENERATING MAIN RESULT PLOTS")
    print("="*80 + "\n")
    
    # Plot 1: Cumulative Regret Comparison
    print("Creating Plot 1: Cumulative Regret Comparison...")
    plot_combined_regret(results_by_df, df_values, T, n_sim, K, d, tau, use_ci=True, ci_level=0.95)
    
    # Plot 2: Beta Estimation Error Comparison
    print("Creating Plot 2: Beta Estimation Error Comparison...")
    plot_combined_beta_error(results_by_df, df_values, T, n_sim, K, d, tau, use_ci=True, ci_level=0.95)
    
    # ==================== SUMMARY TABLE ====================
    print_summary_table(results_by_df, df_values, T)
    
    print("="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)



if __name__ == "__main__":
    main()