"""
Project 4 Results Analysis
===========================

Comprehensive analysis of multi-algorithm comparison results.

Features:
- Load and aggregate results from multiple experiments
- Statistical comparison between algorithms
- Regret convergence analysis
- Beta error analysis
- Computational cost comparison
- LaTeX table generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
from scipy import stats


class Project4Analyzer:
    """
    Analyzer for Project 4 results.
    
    Loads results from multiple experiments and provides
    comprehensive statistical analysis and visualization.
    """
    
    def __init__(self, results_dir: str = 'results/project4'):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        results_dir : str
            Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.summary_stats = None
    
    def load_experiments(self, pattern: str = '*.pkl'):
        """
        Load all experiment results matching pattern.
        
        Parameters
        ----------
        pattern : str
            Glob pattern for result files
        """
        data_dir = self.results_dir / 'data'
        result_files = list(data_dir.glob(pattern))
        
        print(f"Loading {len(result_files)} experiment files...")
        
        for filepath in result_files:
            exp_name = filepath.stem
            
            # Load results
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            # Load metadata if available
            metadata_path = filepath.parent / f"{exp_name}_metadata.json"
            metadata = None
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            self.experiments[exp_name] = {
                'results': results,
                'metadata': metadata
            }
        
        print(f"✓ Loaded {len(self.experiments)} experiments")
    
    def compute_summary_statistics(self):
        """
        Compute summary statistics for all experiments.
        
        Returns
        -------
        DataFrame
            Summary statistics for each experiment and algorithm
        """
        summary_data = []
        
        for exp_name, exp_data in self.experiments.items():
            results = exp_data['results']
            metadata = exp_data['metadata'] or {}
            
            for algo_name, algo_results in results.items():
                regret = algo_results['cumulative_regret']  # (n_sim, T)
                beta_errors = algo_results['beta_errors']   # (n_sim, T, K)
                
                # Final values
                final_regret = regret[:, -1]
                final_beta = np.mean(beta_errors[:, -1, :], axis=1)
                
                # Compute statistics
                summary_data.append({
                    'Experiment': exp_name,
                    'Algorithm': algo_name,
                    'df': metadata.get('err_generator', 'Unknown'),
                    'd': metadata.get('d', 'Unknown'),
                    'T': metadata.get('T', 'Unknown'),
                    'High_Dim': metadata.get('high_dim', False),
                    'Mean_Regret': np.mean(final_regret),
                    'Std_Regret': np.std(final_regret),
                    'Median_Regret': np.median(final_regret),
                    'Q25_Regret': np.percentile(final_regret, 25),
                    'Q75_Regret': np.percentile(final_regret, 75),
                    'Mean_Beta_Error': np.mean(final_beta),
                    'Std_Beta_Error': np.std(final_beta),
                    'Min_Regret': np.min(final_regret),
                    'Max_Regret': np.max(final_regret)
                })
        
        self.summary_stats = pd.DataFrame(summary_data)
        return self.summary_stats
    
    def statistical_tests(self):
        """
        Perform statistical tests comparing algorithms.
        
        Returns
        -------
        DataFrame
            Pairwise comparison results
        """
        if self.summary_stats is None:
            self.compute_summary_statistics()
        
        test_results = []
        
        # Group by experiment
        for exp_name in self.summary_stats['Experiment'].unique():
            exp_data = self.summary_stats[
                self.summary_stats['Experiment'] == exp_name
            ]
            
            algorithms = exp_data['Algorithm'].unique()
            
            # Pairwise comparisons
            for i, algo1 in enumerate(algorithms):
                for algo2 in algorithms[i+1:]:
                    # Get regrets for both algorithms
                    regret1 = self.experiments[exp_name]['results'][algo1]['cumulative_regret'][:, -1]
                    regret2 = self.experiments[exp_name]['results'][algo2]['cumulative_regret'][:, -1]
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(regret1, regret2)
                    
                    # Effect size (Cohen's d)
                    diff = regret1 - regret2
                    cohens_d = np.mean(diff) / np.std(diff)
                    
                    test_results.append({
                        'Experiment': exp_name,
                        'Algorithm_1': algo1,
                        'Algorithm_2': algo2,
                        'Mean_Diff': np.mean(diff),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'Cohens_d': cohens_d,
                        'Significant': p_value < 0.05,
                        'Winner': algo1 if np.mean(diff) < 0 else algo2
                    })
        
        return pd.DataFrame(test_results)
    
    def plot_algorithm_comparison(self, experiment_name: str, save_path: str = None):
        """
        Create comprehensive comparison plot for one experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of experiment to plot
        save_path : str, optional
            Path to save figure
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        results = self.experiments[experiment_name]['results']
        algorithms = list(results.keys())
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cumulative Regret
        ax = axes[0, 0]
        for algo in algorithms:
            regret = results[algo]['cumulative_regret']
            mean_regret = np.mean(regret, axis=0)
            std_regret = np.std(regret, axis=0) / np.sqrt(regret.shape[0])
            
            T = regret.shape[1]
            steps = np.arange(1, T + 1)
            
            ax.plot(steps, mean_regret, label=algo, linewidth=2.5)
            ax.fill_between(steps,
                           mean_regret - 1.96 * std_regret,
                           mean_regret + 1.96 * std_regret,
                           alpha=0.2)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
        ax.set_title('(A) Cumulative Regret', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Beta Errors
        ax = axes[0, 1]
        for algo in algorithms:
            beta_errors = results[algo]['beta_errors']
            avg_errors = np.mean(beta_errors, axis=2)  # Average over arms
            mean_error = np.mean(avg_errors, axis=0)
            std_error = np.std(avg_errors, axis=0) / np.sqrt(avg_errors.shape[0])
            
            T = beta_errors.shape[1]
            steps = np.arange(1, T + 1)
            
            ax.plot(steps, mean_error, label=algo, linewidth=2.5)
            ax.fill_between(steps,
                           mean_error - 1.96 * std_error,
                           mean_error + 1.96 * std_error,
                           alpha=0.2)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Beta Estimation Error', fontsize=12, fontweight='bold')
        ax.set_title('(B) Parameter Estimation', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Final Regret Distribution
        ax = axes[1, 0]
        final_regrets = [results[algo]['cumulative_regret'][:, -1] for algo in algorithms]
        
        positions = np.arange(len(algorithms))
        bp = ax.boxplot(final_regrets, labels=algorithms, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Final Regret', fontsize=12, fontweight='bold')
        ax.set_title('(C) Final Regret Distribution', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 4: Regret Reduction (vs baseline)
        ax = axes[1, 1]
        baseline_algo = algorithms[0]  # Use first as baseline
        baseline_regret = results[baseline_algo]['cumulative_regret'][:, -1]
        
        improvements = []
        for algo in algorithms:
            algo_regret = results[algo]['cumulative_regret'][:, -1]
            improvement = (baseline_regret - algo_regret) / baseline_regret * 100
            improvements.append(np.mean(improvement))
        
        bars = ax.bar(algorithms, improvements, color=colors)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Regret Reduction vs Baseline (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'(D) Improvement over {baseline_algo}', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{improvement:.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')
        
        plt.suptitle(f'Algorithm Comparison: {experiment_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_latex_table(self, save_path: str = None) -> str:
        """
        Generate LaTeX table of summary statistics.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save LaTeX file
        
        Returns
        -------
        str
            LaTeX table code
        """
        if self.summary_stats is None:
            self.compute_summary_statistics()
        
        # Select key columns
        table_df = self.summary_stats[[
            'Experiment', 'Algorithm', 'Mean_Regret', 'Std_Regret',
            'Mean_Beta_Error', 'Std_Beta_Error'
        ]].copy()
        
        # Round values
        table_df['Mean_Regret'] = table_df['Mean_Regret'].round(2)
        table_df['Std_Regret'] = table_df['Std_Regret'].round(2)
        table_df['Mean_Beta_Error'] = table_df['Mean_Beta_Error'].round(4)
        table_df['Std_Beta_Error'] = table_df['Std_Beta_Error'].round(4)
        
        # Generate LaTeX
        latex_str = table_df.to_latex(
            index=False,
            column_format='lcccc',
            caption='Algorithm Performance Comparison',
            label='tab:algorithm_comparison'
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex_str)
            print(f"Saved LaTeX table to: {save_path}")
        
        return latex_str
    
    def create_summary_report(self, output_dir: str = None):
        """
        Create comprehensive summary report with all analyses.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save report files
        """
        if output_dir is None:
            output_dir = self.results_dir / 'summary'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("="*80 + "\n")
        
        # 1. Summary statistics
        print("Computing summary statistics...")
        summary = self.compute_summary_statistics()
        summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
        print(f"  Saved: {output_dir / 'summary_statistics.csv'}")
        
        # 2. Statistical tests
        print("Performing statistical tests...")
        tests = self.statistical_tests()
        tests.to_csv(output_dir / 'statistical_tests.csv', index=False)
        print(f"  Saved: {output_dir / 'statistical_tests.csv'}")
        
        # 3. LaTeX table
        print("Generating LaTeX table...")
        latex_path = output_dir / 'summary_table.tex'
        self.generate_latex_table(save_path=str(latex_path))
        
        # 4. Plots for each experiment
        print("Creating comparison plots...")
        for exp_name in self.experiments.keys():
            save_path = output_dir / f'comparison_{exp_name}.pdf'
            self.plot_algorithm_comparison(exp_name, save_path=str(save_path))
        
        print("\n" + "="*80)
        print("✓ SUMMARY REPORT COMPLETE")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print("\nGenerated files:")
        print("  - summary_statistics.csv")
        print("  - statistical_tests.csv")
        print("  - summary_table.tex")
        print("  - comparison_*.pdf (one per experiment)")
        print("="*80 + "\n")


def main():
    """Main entry point for analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Project 4 results'
    )
    parser.add_argument(
        '--results_dir',
        default='results/project4',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--pattern',
        default='*.pkl',
        help='Pattern for result files'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Directory to save analysis outputs'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = Project4Analyzer(results_dir=args.results_dir)
    
    # Load experiments
    analyzer.load_experiments(pattern=args.pattern)
    
    if len(analyzer.experiments) == 0:
        print("No experiments found!")
        return
    
    # Generate summary report
    analyzer.create_summary_report(output_dir=args.output_dir)
    
    print("\n✓ Analysis complete")


if __name__ == "__main__":
    main()