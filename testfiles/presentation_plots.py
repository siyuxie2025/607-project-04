"""
Generate Presentation-Ready Plots for Project 4
================================================

Creates publication-quality plots optimized for:
- Projector visibility (large fonts, thick lines)
- Print quality (300 DPI)
- Color accessibility

Usage:
    python create_presentation_plots.py --results_dir results/project4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from pathlib import Path
import argparse

# ========================================
# PRESENTATION STYLE SETTINGS
# ========================================

# Color scheme (colorblind-friendly)
COLORS = {
    'ForcedSampling': '#2E86AB',      # Blue
    'LinUCB': '#06A77D',               # Green
    'EpsilonGreedy': '#D36135',        # Red/Orange
    'ThompsonSampling': '#7B2D26'      # Purple/Brown
}

# Plotting parameters for presentations
PRESENTATION_STYLE = {
    'figure.figsize': (12, 7),         # Wide format for slides
    'figure.dpi': 300,                 # High resolution
    'font.size': 18,                   # Large base font
    'axes.labelsize': 22,              # Axis labels
    'axes.titlesize': 24,              # Title
    'xtick.labelsize': 18,             # X tick labels
    'ytick.labelsize': 18,             # Y tick labels
    'legend.fontsize': 16,             # Legend
    'lines.linewidth': 3.5,            # Thick lines
    'lines.markersize': 10,            # Large markers
    'axes.linewidth': 2,               # Thick axes
    'grid.linewidth': 1.5,             # Visible grid
    'figure.autolayout': True,         # Auto tight layout
}

# Apply style
mpl.rcParams.update(PRESENTATION_STYLE)


def load_baseline_results(results_dir: str):
    """
    Load baseline experiment results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing result files
    
    Returns
    -------
    dict
        Results for all algorithms
    """
    results_dir = Path(results_dir)
    data_dir = results_dir / 'data'
    
    # Find baseline result file
    baseline_files = list(data_dir.glob('*baseline*df2.25*.pkl'))
    
    if not baseline_files:
        raise FileNotFoundError(f"No baseline results found in {data_dir}")
    
    # Load first match
    with open(baseline_files[0], 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded: {baseline_files[0].name}")
    return results


def create_main_result_plot(results: dict, save_path: str = None):
    """
    Create the main result plot for Slide 4.
    
    This is THE money plot - make it perfect!
    
    Parameters
    ----------
    results : dict
        Results from simulation
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots()
    
    # Sort algorithms for consistent ordering
    algorithm_order = ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    algorithms = [a for a in algorithm_order if a in results]
    
    # Plot each algorithm
    for algo in algorithms:
        if algo not in results:
            continue
            
        regret = results[algo]['cumulative_regret']  # (n_sim, T)
        
        # Compute statistics
        mean_regret = np.mean(regret, axis=0)
        std_regret = np.std(regret, axis=0) / np.sqrt(regret.shape[0])
        
        T = regret.shape[1]
        steps = np.arange(1, T + 1)
        
        # Get color
        color = COLORS.get(algo, 'black')
        
        # Format algorithm name for legend
        name_map = {
            'ForcedSampling': 'Forced Sampling',
            'LinUCB': 'LinUCB',
            'EpsilonGreedy': 'ε-Greedy',
            'ThompsonSampling': 'Thompson Sampling'
        }
        display_name = name_map.get(algo, algo)
        
        # Plot mean line with confidence band
        ax.plot(steps, mean_regret, 
                color=color, 
                linewidth=4.0,  # Extra thick for visibility
                label=display_name,
                alpha=0.9,
                zorder=10)
        
        # Add 95% confidence interval
        ax.fill_between(steps,
                       mean_regret - 1.96 * std_regret,
                       mean_regret + 1.96 * std_regret,
                       color=color,
                       alpha=0.15,
                       zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('Time (rounds)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=24, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', 
             fontsize=18,
             frameon=True,
             fancybox=True,
             shadow=True,
             framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    
    # Make sure everything is visible
    ax.tick_params(labelsize=20, width=2, length=6)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = 'presentation_main_result.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    
    # Also save as PDF (vector graphics)
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def create_robustness_plot(results_list: list, df_values: list, save_path: str = None):
    """
    Create robustness plot showing performance across tail heaviness.
    
    For Slide 5.
    
    Parameters
    ----------
    results_list : list of dict
        Results for each df value
    df_values : list
        Corresponding df values
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots()
    
    algorithm_order = ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    
    # Collect final regrets for each algorithm across df values
    algorithm_data = {algo: [] for algo in algorithm_order}
    
    for results in results_list:
        for algo in algorithm_order:
            if algo in results:
                regret = results[algo]['cumulative_regret']
                final_regret = np.mean(regret[:, -1])
                algorithm_data[algo].append(final_regret)
    
    # Plot
    for algo in algorithm_order:
        if not algorithm_data[algo]:
            continue
        
        color = COLORS.get(algo, 'black')
        name_map = {
            'ForcedSampling': 'Forced Sampling',
            'LinUCB': 'LinUCB',
            'EpsilonGreedy': 'ε-Greedy',
            'ThompsonSampling': 'Thompson Sampling'
        }
        display_name = name_map.get(algo, algo)
        
        ax.plot(df_values, algorithm_data[algo],
               marker='o',
               markersize=12,
               color=color,
               linewidth=3.5,
               label=display_name,
               alpha=0.9)
    
    # Labels
    ax.set_xlabel('Tail Heaviness (df)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Final Regret', fontsize=24, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=18, frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Make x-axis show all df values
    ax.set_xticks(df_values)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = 'presentation_robustness.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def create_performance_table(results: dict, save_path: str = None):
    """
    Create a simple performance comparison table.
    
    For use in slides or report.
    
    Parameters
    ----------
    results : dict
        Results from simulation
    save_path : str, optional
        Path to save table
    """
    algorithm_order = ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    
    # Collect statistics
    stats = []
    for algo in algorithm_order:
        if algo not in results:
            continue
        
        regret = results[algo]['cumulative_regret']
        final_regret = regret[:, -1]
        
        stats.append({
            'Algorithm': algo.replace('ForcedSampling', 'Forced Sampling')
                            .replace('EpsilonGreedy', 'ε-Greedy')
                            .replace('ThompsonSampling', 'Thompson Sampling'),
            'Mean': np.mean(final_regret),
            'Std': np.std(final_regret),
            'Median': np.median(final_regret),
            'Min': np.min(final_regret),
            'Max': np.max(final_regret)
        })
    
    # Print as formatted table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<20} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-"*70)
    
    for s in stats:
        print(f"{s['Algorithm']:<20} {s['Mean']:>10.2f}  {s['Std']:>10.2f}  {s['Median']:>10.2f}")
    
    print("="*70 + "\n")
    
    # Save to CSV if requested
    if save_path:
        import pandas as pd
        df = pd.DataFrame(stats)
        df.to_csv(save_path, index=False, float_format='%.2f')
        print(f"✓ Saved table: {save_path}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Generate presentation-ready plots for Project 4'
    )
    parser.add_argument(
        '--results_dir',
        default='results/project4',
        help='Directory containing simulation results'
    )
    parser.add_argument(
        '--output_dir',
        default='presentation_plots',
        help='Directory to save presentation plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING PRESENTATION PLOTS")
    print("="*70 + "\n")
    
    # Load baseline results
    print("Loading baseline results...")
    try:
        results = load_baseline_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run: make baseline")
        return
    
    # Create main result plot (CRITICAL for Slide 4)
    print("\nCreating main result plot (Slide 4)...")
    create_main_result_plot(
        results,
        save_path=str(output_dir / 'slide4_main_result.png')
    )
    
    # Create performance table
    print("\nGenerating performance table...")
    create_performance_table(
        results,
        save_path=str(output_dir / 'performance_table.csv')
    )
    
    # Try to create robustness plot if multiple df results available
    print("\nLooking for robustness data (Slide 5)...")
    data_dir = Path(args.results_dir) / 'data'
    df_files = sorted(data_dir.glob('*df*.pkl'))
    
    if len(df_files) > 1:
        print(f"Found {len(df_files)} df configurations, creating robustness plot...")
        
        # Load all results
        results_by_df = []
        df_values_found = []
        
        for f in df_files:
            # Extract df from filename
            import re
            match = re.search(r'df([0-9.]+)', f.name)
            if match:
                df_val = float(match.group(1))
                with open(f, 'rb') as file:
                    res = pickle.load(file)
                results_by_df.append(res)
                df_values_found.append(df_val)
        
        if results_by_df:
            # Sort by df value
            sorted_pairs = sorted(zip(df_values_found, results_by_df))
            df_values_found = [p[0] for p in sorted_pairs]
            results_by_df = [p[1] for p in sorted_pairs]
            
            create_robustness_plot(
                results_by_df,
                df_values_found,
                save_path=str(output_dir / 'slide5_robustness.png')
            )
    else:
        print("  Not enough df configurations for robustness plot")
        print("  Run: make custom DF=1.5 && make custom DF=3.0")
    
    print("\n" + "="*70)
    print("✓ PRESENTATION PLOTS COMPLETE")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print("\nKey files for slides:")
    print(f"  - slide4_main_result.png     (Main result - CRITICAL)")
    print(f"  - slide4_main_result.pdf     (Vector version)")
    print(f"  - performance_table.csv      (Summary statistics)")
    
    if (output_dir / 'slide5_robustness.png').exists():
        print(f"  - slide5_robustness.png      (Robustness plot)")
    
    print("\nImport these into your PowerPoint/Google Slides!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()