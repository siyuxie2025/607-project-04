"""
Analyze and visualize results from bandit algorithm simulations.

This script processes saved simulation results and generates:
- Summary statistics (CSV)
- Regret comparison plots
- Beta estimation error plots
- Combined comparison figures
"""

import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import pickle


def load_all_results(data_dir):
    """
    Load all simulation results from data directory.
    
    Parameters
    ----------
    data_dir : str
        Directory containing .pkl result files
    
    Returns
    -------
    list of tuples
        List of (results_dict, metadata_dict, filepath) tuples
    """
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")
    
    all_results = []
    
    for pkl_file in sorted(pkl_files):
        # Skip if it's a metadata file
        if '_metadata' in pkl_file:
            continue
            
        print(f"Loading: {pkl_file}")
        
        # Load results
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        
        # Try to load metadata
        metadata_file = pkl_file.replace('.pkl', '_metadata.json')
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        all_results.append((results, metadata, pkl_file))
    
    print(f"\nLoaded {len(all_results)} result files")
    return all_results


def generate_summary_statistics(all_results, output_dir):
    """
    Generate summary statistics CSV from all results.
    
    Parameters
    ----------
    all_results : list
        List of (results, metadata, filepath) tuples
    output_dir : str
        Directory to save output CSV
    
    Returns
    -------
    str
        Path to generated CSV file
    """
    summary_data = []
    
    for results, metadata, filepath in all_results:
        filename = os.path.basename(filepath)
        
        # Extract basic info
        if metadata:
            err_gen = metadata.get('err_generator', 'Unknown')
            n_sim = metadata.get('n_sim', 'N/A')
            T = metadata.get('T', 'N/A')
            K = metadata.get('K', 'N/A')
            d = metadata.get('d', 'N/A')
            tau = metadata.get('tau', 'N/A')
        else:
            err_gen = 'Unknown'
            n_sim = results['cumulated_regret_RiskAware'].shape[0]
            T = results['cumulated_regret_RiskAware'].shape[1]
            K = results['beta_errors_rab'].shape[2]
            d = 'N/A'
            tau = 'N/A'
        
        # Calculate statistics for RiskAware
        regret_rab = results['cumulated_regret_RiskAware']
        final_regret_rab = regret_rab[:, -1]
        
        beta_errors_rab = results['beta_errors_rab'][:, -1, :]
        avg_beta_error_rab = np.mean(beta_errors_rab, axis=1)
        
        summary_data.append({
            'File': filename,
            'Error_Generator': err_gen,
            'Method': 'RiskAware',
            'N_Sim': n_sim,
            'T': T,
            'K': K,
            'D': d,
            'Tau': tau,
            'Mean_Final_Regret': np.mean(final_regret_rab),
            'Median_Final_Regret': np.median(final_regret_rab),
            'Std_Final_Regret': np.std(final_regret_rab),
            'Min_Final_Regret': np.min(final_regret_rab),
            'Max_Final_Regret': np.max(final_regret_rab),
            'Q25_Final_Regret': np.percentile(final_regret_rab, 25),
            'Q75_Final_Regret': np.percentile(final_regret_rab, 75),
            'Mean_Beta_Error': np.mean(avg_beta_error_rab),
            'Median_Beta_Error': np.median(avg_beta_error_rab),
            'Std_Beta_Error': np.std(avg_beta_error_rab),
        })
        
        # Calculate statistics for OLS
        regret_ols = results['cumulated_regret_OLS']
        final_regret_ols = regret_ols[:, -1]
        
        beta_errors_ols = results['beta_errors_ols'][:, -1, :]
        avg_beta_error_ols = np.mean(beta_errors_ols, axis=1)
        
        summary_data.append({
            'File': filename,
            'Error_Generator': err_gen,
            'Method': 'OLS',
            'N_Sim': n_sim,
            'T': T,
            'K': K,
            'D': d,
            'Tau': tau,
            'Mean_Final_Regret': np.mean(final_regret_ols),
            'Median_Final_Regret': np.median(final_regret_ols),
            'Std_Final_Regret': np.std(final_regret_ols),
            'Min_Final_Regret': np.min(final_regret_ols),
            'Max_Final_Regret': np.max(final_regret_ols),
            'Q25_Final_Regret': np.percentile(final_regret_ols, 25),
            'Q75_Final_Regret': np.percentile(final_regret_ols, 75),
            'Mean_Beta_Error': np.mean(avg_beta_error_ols),
            'Median_Beta_Error': np.median(avg_beta_error_ols),
            'Std_Beta_Error': np.std(avg_beta_error_ols),
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'summary_statistics.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)
    print(df.to_string(index=False))
    print('='*80)
    print(f"\nSaved to: {output_file}")
    
    return output_file


def plot_regret_comparison(all_results, output_dir, use_ci=True, ci_level=0.95):
    """
    Create regret comparison plots for all results.
    
    Parameters
    ----------
    all_results : list
        List of (results, metadata, filepath) tuples
    output_dir : str
        Directory to save figures
    use_ci : bool
        Whether to show confidence intervals
    ci_level : float
        Confidence level for intervals
    """
    n_results = len(all_results)
    
    if n_results == 0:
        return
    
    # Determine subplot layout
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_results == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (results, metadata, filepath) in enumerate(all_results):
        ax = axes[idx]
        
        regret_rab = results['cumulated_regret_RiskAware']
        regret_ols = results['cumulated_regret_OLS']
        
        n_sim = regret_rab.shape[0]
        T = regret_rab.shape[1]
        steps = np.arange(1, T + 1)
        
        # Compute means
        mean_rab = np.mean(regret_rab, axis=0)
        mean_ols = np.mean(regret_ols, axis=0)
        
        if use_ci and n_sim > 1:
            se_rab = np.std(regret_rab, axis=0, ddof=1) / np.sqrt(n_sim)
            se_ols = np.std(regret_ols, axis=0, ddof=1) / np.sqrt(n_sim)
            
            t_crit = stats.t.ppf((1 + ci_level) / 2., n_sim - 1)
            
            lower_rab = mean_rab - t_crit * se_rab
            upper_rab = mean_rab + t_crit * se_rab
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols
            
            ax.fill_between(steps, lower_rab, upper_rab, color='red', alpha=0.2)
            ax.fill_between(steps, lower_ols, upper_ols, color='blue', alpha=0.2)
        
        ax.plot(steps, mean_rab, 'r-', label='Risk Aware', linewidth=2.5)
        ax.plot(steps, mean_ols, 'b--', label='OLS', linewidth=2.5)
        
        # Get title from metadata
        if metadata:
            title = f"{metadata.get('err_generator', 'Unknown')}\n(T={metadata.get('T', 'N/A')})"
        else:
            title = f"Simulation {idx+1}"
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Cumulative Regret', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final values
        final_rab = mean_rab[-1]
        final_ols = mean_ols[-1]
        ax.text(0.95, 0.05, f'Final:\nRAB: {final_rab:.1f}\nOLS: {final_ols:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_results, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cumulative Regret Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'regret_comparison.pdf')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_beta_error_comparison(all_results, output_dir, use_ci=True, ci_level=0.95):
    """
    Create beta estimation error comparison plots.
    
    Parameters
    ----------
    all_results : list
        List of (results, metadata, filepath) tuples
    output_dir : str
        Directory to save figures
    use_ci : bool
        Whether to show confidence intervals
    ci_level : float
        Confidence level for intervals
    """
    n_results = len(all_results)
    
    if n_results == 0:
        return
    
    # Determine subplot layout
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_results == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (results, metadata, filepath) in enumerate(all_results):
        ax = axes[idx]
        
        beta_errors_rab = results['beta_errors_rab']  # Shape: (n_sim, T, K)
        beta_errors_ols = results['beta_errors_ols']
        
        n_sim = beta_errors_rab.shape[0]
        T = beta_errors_rab.shape[1]
        K = beta_errors_rab.shape[2]
        steps = np.arange(1, T + 1)
        
        # Average across arms
        rab_avg = np.mean(beta_errors_rab, axis=2)  # Shape: (n_sim, T)
        ols_avg = np.mean(beta_errors_ols, axis=2)
        
        # Mean across simulations
        mean_rab = np.mean(rab_avg, axis=0)
        mean_ols = np.mean(ols_avg, axis=0)
        
        if use_ci and n_sim > 1:
            se_rab = np.std(rab_avg, axis=0, ddof=1) / np.sqrt(n_sim)
            se_ols = np.std(ols_avg, axis=0, ddof=1) / np.sqrt(n_sim)
            
            t_crit = stats.t.ppf((1 + ci_level) / 2., n_sim - 1)
            
            lower_rab = mean_rab - t_crit * se_rab
            upper_rab = mean_rab + t_crit * se_rab
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols
            
            ax.fill_between(steps, lower_rab, upper_rab, color='red', alpha=0.2)
            ax.fill_between(steps, lower_ols, upper_ols, color='blue', alpha=0.2)
        
        ax.plot(steps, mean_rab, 'r-', label='Risk Aware', linewidth=2.5)
        ax.plot(steps, mean_ols, 'b--', label='OLS', linewidth=2.5)
        
        # Get title from metadata
        if metadata:
            title = f"{metadata.get('err_generator', 'Unknown')}\n(K={K})"
        else:
            title = f"Simulation {idx+1}"
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Beta Estimation Error', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final values
        final_rab = mean_rab[-1]
        final_ols = mean_ols[-1]
        improvement = ((final_ols - final_rab) / (final_ols * 100)* 10000) if final_ols > final_rab else 0
        ax.text(0.95, 0.95, f'Final Error:\nRAB: {final_rab:.4f}\nOLS: {final_ols:.4f}\nΔ: {improvement:.1f}%',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_results, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Beta Estimation Error Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'beta_error_comparison.pdf')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze bandit simulation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='results/data',
                       help='Directory containing simulation results (.pkl files)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save analysis outputs')
    parser.add_argument('--figures_only', action='store_true',
                       help='Only generate figures, skip summary statistics')
    parser.add_argument('--no_ci', action='store_true',
                       help='Do not show confidence intervals in plots')
    parser.add_argument('--ci_level', type=float, default=0.95,
                       help='Confidence level for intervals (0-1)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BANDIT ALGORITHM RESULTS ANALYSIS")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all results
    print("\nLoading results...")
    all_results = load_all_results(args.data_dir)
    
    # Generate summary statistics
    if not args.figures_only:
        print("\nGenerating summary statistics...")
        generate_summary_statistics(all_results, args.output_dir)
    
    # Generate figures
    print("\nGenerating figures...")
    figures_dir = os.path.join(args.output_dir, 'figures') if args.figures_only else args.output_dir
    os.makedirs(figures_dir, exist_ok=True)
    
    use_ci = not args.no_ci
    
    plot_regret_comparison(all_results, figures_dir, use_ci=use_ci, ci_level=args.ci_level)
    plot_beta_error_comparison(all_results, figures_dir, use_ci=use_ci, ci_level=args.ci_level)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()