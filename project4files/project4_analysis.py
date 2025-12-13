"""
Project 4 Analysis Script
==========================

Analyze and visualize results from Project 4 experiments.

Usage:
    python project4_analysis.py                    # Analyze all scenarios
    python project4_analysis.py --scenario default # Analyze single scenario
    python project4_analysis.py --experiment df_sweep # Analyze experiment
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy import stats

sys.path.insert(0, 'src')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

Path('results/project4/analysis').mkdir(parents=True, exist_ok=True)


class Project4Analysis:
    """Analysis and visualization for Project 4 results."""
    
    @staticmethod
    def load_results(filepath):
        """Load results from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def analyze_scenario(scenario_name):
        """Analyze single scenario results."""
        print(f"\n{'='*80}")
        print(f"Analyzing Scenario: {scenario_name}")
        print('='*80)
        
        filepath = f'results/project4/scenario_{scenario_name}.pkl'
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found")
            print("Run: python project4_main.py --scenario {scenario_name} first")
            return
        
        results = Project4Analysis.load_results(filepath)
        
        # Create summary table
        summary = Project4Analysis._create_summary_table(results)
        print("\nSummary Statistics:")
        print(summary.to_string(index=False))
        
        # Save summary
        summary.to_csv(
            f'results/project4/analysis/summary_{scenario_name}.csv',
            index=False
        )
        
        # Create visualizations
        Project4Analysis._plot_regret_comparison(
            results, 
            title=f'Regret Comparison: {scenario_name.title()}',
            save_path=f'results/project4/analysis/regret_{scenario_name}.pdf'
        )
        
        Project4Analysis._plot_beta_error_comparison(
            results,
            title=f'Beta Error: {scenario_name.title()}',
            save_path=f'results/project4/analysis/beta_error_{scenario_name}.pdf'
        )
        
        Project4Analysis._plot_performance_summary(
            results,
            save_path=f'results/project4/analysis/performance_{scenario_name}.pdf'
        )
        
        print(f"\n✓ Analysis complete for {scenario_name}")
        print(f"  Summary: results/project4/analysis/summary_{scenario_name}.csv")
        print(f"  Plots: results/project4/analysis/*_{scenario_name}.pdf")
    
    @staticmethod
    def analyze_all_scenarios():
        """Analyze all scenario results."""
        print("\n" + "="*80)
        print("Analyzing All Scenarios")
        print("="*80)
        
        scenario_files = glob.glob('results/project4/scenario_*.pkl')
        
        if not scenario_files:
            print("No scenario results found!")
            print("Run: python project4_main.py --scenario all")
            return
        
        # Collect all summaries
        all_summaries = []
        
        for filepath in sorted(scenario_files):
            scenario_name = os.path.basename(filepath).replace('scenario_', '').replace('.pkl', '')
            print(f"\nProcessing: {scenario_name}")
            
            results = Project4Analysis.load_results(filepath)
            summary = Project4Analysis._create_summary_table(results)
            summary['scenario'] = scenario_name
            all_summaries.append(summary)
        
        # Combine all summaries
        combined = pd.concat(all_summaries, ignore_index=True)
        
        # Reorder columns
        cols = ['scenario', 'algorithm'] + [c for c in combined.columns if c not in ['scenario', 'algorithm']]
        combined = combined[cols]
        
        print("\n" + "="*80)
        print("COMBINED SUMMARY - ALL SCENARIOS")
        print("="*80)
        print(combined.to_string(index=False))
        
        # Save combined summary
        combined.to_csv('results/project4/analysis/summary_all_scenarios.csv', index=False)
        
        # Create comparison plots
        Project4Analysis._plot_cross_scenario_comparison(
            combined,
            save_path='results/project4/analysis/cross_scenario_comparison.pdf'
        )
        
        print("\n✓ Cross-scenario analysis complete")
        print("  Combined summary: results/project4/analysis/summary_all_scenarios.csv")
        print("  Comparison plot: results/project4/analysis/cross_scenario_comparison.pdf")
    
    @staticmethod
    def analyze_experiment(experiment_name):
        """Analyze experimental sweep results."""
        print(f"\n{'='*80}")
        print(f"Analyzing Experiment: {experiment_name}")
        print('='*80)
        
        summary_file = f'results/project4/experiments/{experiment_name}_summary.csv'
        
        if not os.path.exists(summary_file):
            print(f"Error: {summary_file} not found")
            print(f"Run: python project4_experiments.py --experiment {experiment_name}")
            return
        
        df = pd.read_csv(summary_file)
        
        print("\nExperiment Summary:")
        print(df.to_string(index=False))
        
        # Create visualizations based on experiment type
        if 'df' in experiment_name:
            Project4Analysis._plot_df_sweep(df, save_path='results/project4/analysis/df_sweep.pdf')
        elif 'tau' in experiment_name:
            Project4Analysis._plot_tau_sweep(df, save_path='results/project4/analysis/tau_sweep.pdf')
        elif 'dim' in experiment_name:
            Project4Analysis._plot_dim_sweep(df, save_path='results/project4/analysis/dim_sweep.pdf')
        elif 'arms' in experiment_name:
            Project4Analysis._plot_arms_sweep(df, save_path='results/project4/analysis/arms_sweep.pdf')
        elif 'beta' in experiment_name:
            Project4Analysis._plot_beta_comparison(df, save_path='results/project4/analysis/beta_strategy.pdf')
        
        print(f"\n✓ Experiment analysis complete")
    
    @staticmethod
    def _create_summary_table(results):
        """Create summary statistics table from results."""
        summary_data = []
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name][:, -1]  # Final regret
            beta_err = results['beta_errors'][alg_name][:, -1, :]  # Final beta error
            
            summary_data.append({
                'algorithm': alg_name,
                'mean_regret': np.mean(regret),
                'std_regret': np.std(regret),
                'median_regret': np.median(regret),
                'mean_beta_error': np.mean(beta_err),
                'std_beta_error': np.std(beta_err),
                'median_beta_error': np.median(beta_err),
                'runtime_sec': results['computation_time'][alg_name]
            })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def _plot_regret_comparison(results, title='Regret Comparison', save_path=None):
        """Plot cumulative regret for all algorithms."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        T = results['regret'][results['algorithms'][0]].shape[1]
        steps = np.arange(1, T + 1)
        
        colors = {'LinUCB': 'blue', 'ThompsonSampling': 'green',
                  'EpsilonGreedy': 'orange', 'ForcedSampling': 'red'}
        
        for alg_name in results['algorithms']:
            regret = results['regret'][alg_name]
            mean_regret = np.mean(regret, axis=0)
            std_regret = np.std(regret, axis=0)
            
            color = colors.get(alg_name, 'black')
            
            ax.plot(steps, mean_regret, label=alg_name, color=color, linewidth=2.5)
            ax.fill_between(steps, 
                           mean_regret - std_regret, 
                           mean_regret + std_regret,
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_beta_error_comparison(results, title='Beta Error Comparison', save_path=None):
        """Plot beta estimation error for all algorithms."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        T = results['beta_errors'][results['algorithms'][0]].shape[1]
        steps = np.arange(1, T + 1)
        
        colors = {'LinUCB': 'blue', 'ThompsonSampling': 'green',
                  'EpsilonGreedy': 'orange', 'ForcedSampling': 'red'}
        
        for alg_name in results['algorithms']:
            beta_err = results['beta_errors'][alg_name]
            # Average across arms
            beta_err_avg = np.mean(beta_err, axis=2)
            mean_err = np.mean(beta_err_avg, axis=0)
            std_err = np.std(beta_err_avg, axis=0)
            
            color = colors.get(alg_name, 'black')
            
            ax.plot(steps, mean_err, label=alg_name, color=color, linewidth=2.5)
            ax.fill_between(steps, 
                           mean_err - std_err,
                           mean_err + std_err,
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
        ax.set_ylabel('Beta Estimation Error', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_performance_summary(results, save_path=None):
        """Create comprehensive performance summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = results['algorithms']
        
        # 1. Final Regret Box Plot
        ax1 = axes[0, 0]
        final_regrets = [results['regret'][alg][:, -1] for alg in algorithms]
        ax1.boxplot(final_regrets, labels=algorithms)
        ax1.set_ylabel('Final Regret', fontweight='bold')
        ax1.set_title('Final Regret Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Final Beta Error Box Plot
        ax2 = axes[0, 1]
        final_beta_errs = [np.mean(results['beta_errors'][alg][:, -1, :], axis=1) 
                           for alg in algorithms]
        ax2.boxplot(final_beta_errs, labels=algorithms)
        ax2.set_ylabel('Final Beta Error', fontweight='bold')
        ax2.set_title('Beta Error Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Computation Time Bar Chart
        ax3 = axes[1, 0]
        runtimes = [results['computation_time'][alg] for alg in algorithms]
        bars = ax3.bar(algorithms, runtimes, color=['blue', 'green', 'orange', 'red'])
        ax3.set_ylabel('Runtime (seconds)', fontweight='bold')
        ax3.set_title('Computation Time', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Speed Trade-off
        ax4 = axes[1, 1]
        mean_regrets = [np.mean(results['regret'][alg][:, -1]) for alg in algorithms]
        colors_list = ['blue', 'green', 'orange', 'red']
        
        ax4.scatter(runtimes, mean_regrets, s=200, c=colors_list, alpha=0.7)
        
        for i, alg in enumerate(algorithms):
            ax4.annotate(alg, (runtimes[i], mean_regrets[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('Runtime (seconds)', fontweight='bold')
        ax4.set_ylabel('Mean Final Regret', fontweight='bold')
        ax4.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_cross_scenario_comparison(df, save_path=None):
        """Plot cross-scenario comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Regret comparison
        ax1 = axes[0]
        pivot_regret = df.pivot_table(values='mean_regret', 
                                       index='algorithm', 
                                       columns='scenario')
        pivot_regret.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        ax1.set_title('Regret Across Scenarios', fontweight='bold', fontsize=13)
        ax1.legend(title='Scenario', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Beta error comparison
        ax2 = axes[1]
        pivot_beta = df.pivot_table(values='mean_beta_error',
                                     index='algorithm',
                                     columns='scenario')
        pivot_beta.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_ylabel('Mean Beta Error', fontweight='bold', fontsize=12)
        ax2.set_title('Beta Error Across Scenarios', fontweight='bold', fontsize=13)
        ax2.legend(title='Scenario', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_df_sweep(df, save_path=None):
        """Plot df sweep results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            
            axes[0].plot(alg_data['df'], alg_data['mean_regret'], 
                        marker='o', label=alg, linewidth=2, markersize=8)
            axes[1].plot(alg_data['df'], alg_data['mean_beta_error'],
                        marker='o', label=alg, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Degrees of Freedom (df)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        axes[0].set_title('Regret vs Tail Heaviness', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Degrees of Freedom (df)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Mean Beta Error', fontweight='bold', fontsize=12)
        axes[1].set_title('Beta Error vs Tail Heaviness', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_tau_sweep(df, save_path=None):
        """Plot tau sweep results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            
            axes[0].plot(alg_data['tau'], alg_data['mean_regret'],
                        marker='o', label=alg, linewidth=2, markersize=8)
            axes[1].plot(alg_data['tau'], alg_data['mean_beta_error'],
                        marker='o', label=alg, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Quantile Level (τ)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        axes[0].set_title('Regret vs Quantile Level', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Quantile Level (τ)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Mean Beta Error', fontweight='bold', fontsize=12)
        axes[1].set_title('Beta Error vs Quantile Level', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_dim_sweep(df, save_path=None):
        """Plot dimension sweep results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            
            axes[0].plot(alg_data['d'], alg_data['mean_regret'],
                        marker='o', label=alg, linewidth=2, markersize=8)
            axes[1].plot(alg_data['d'], alg_data['mean_beta_error'],
                        marker='o', label=alg, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Context Dimension (d)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        axes[0].set_title('Regret vs Dimension', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Context Dimension (d)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Mean Beta Error', fontweight='bold', fontsize=12)
        axes[1].set_title('Beta Error vs Dimension', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_arms_sweep(df, save_path=None):
        """Plot number of arms sweep results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            
            axes[0].plot(alg_data['K'], alg_data['mean_regret'],
                        marker='o', label=alg, linewidth=2, markersize=8)
            axes[1].plot(alg_data['K'], alg_data['runtime'],
                        marker='o', label=alg, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Number of Arms (K)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        axes[0].set_title('Regret vs Number of Arms', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Number of Arms (K)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Runtime (seconds)', fontweight='bold', fontsize=12)
        axes[1].set_title('Computation Time vs Number of Arms', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def _plot_beta_comparison(df, save_path=None):
        """Plot beta strategy comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Regret comparison
        ax1 = axes[0]
        pivot_regret = df.pivot_table(values='mean_regret',
                                       index='algorithm',
                                       columns='beta_strategy')
        pivot_regret.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_ylabel('Mean Final Regret', fontweight='bold', fontsize=12)
        ax1.set_title('Regret by Beta Strategy', fontweight='bold', fontsize=13)
        ax1.legend(title='Strategy', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Beta error comparison
        ax2 = axes[1]
        pivot_beta = df.pivot_table(values='mean_beta_error',
                                     index='algorithm',
                                     columns='beta_strategy')
        pivot_beta.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_ylabel('Mean Beta Error', fontweight='bold', fontsize=12)
        ax2.set_title('Beta Error by Strategy', fontweight='bold', fontsize=13)
        ax2.legend(title='Strategy', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Project 4 Results Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project4_analysis.py                           # Analyze all scenarios
  python project4_analysis.py --scenario default        # Analyze one scenario
  python project4_analysis.py --experiment df_sweep     # Analyze experiment
        """
    )
    
    parser.add_argument('--scenario', type=str, default=None,
                       help='Analyze specific scenario')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Analyze specific experiment')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all scenarios')
    
    args = parser.parse_args()
    
    analyzer = Project4Analysis()
    
    if args.scenario:
        analyzer.analyze_scenario(args.scenario)
    elif args.experiment:
        analyzer.analyze_experiment(args.experiment)
    else:
        # Default: analyze all scenarios
        analyzer.analyze_all_scenarios()
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults in: results/project4/analysis/")


if __name__ == "__main__":
    main()