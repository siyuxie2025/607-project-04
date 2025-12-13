"""
Generate Results for Report Part C
===================================

This script runs key experiments and generates summary statistics
for the final project report.

Usage:
    python generate_report_results.py --mode quick    # Fast version (~15 min)
    python generate_report_results.py --mode full     # Full version (~2 hours)
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_summary_table(results):
    """Generate summary statistics table."""
    summary = []

    for alg in results['algorithms']:
        regret = results['regret'][alg][:, -1]
        beta_err = results['beta_errors'][alg][:, -1, :]
        runtime = results['computation_time'][alg]

        summary.append({
            'Algorithm': alg,
            'Mean Final Regret': f"{np.mean(regret):.2f}",
            'Std Final Regret': f"{np.std(regret):.2f}",
            'Mean Beta Error': f"{np.mean(beta_err):.4f}",
            'Std Beta Error': f"{np.std(beta_err):.4f}",
            'Runtime (s)': f"{runtime:.2f}"
        })

    return pd.DataFrame(summary)

def print_report_section(scenario_name, results_file):
    """Print formatted report section for a scenario."""
    if not Path(results_file).exists():
        print(f"\n⚠️  {scenario_name}: Results not found at {results_file}")
        print(f"   Run: make scenario-{scenario_name.lower().replace(' ', '-')}")
        return None

    results = load_results(results_file)

    print(f"\n{'='*80}")
    print(f"  {scenario_name.upper()}")
    print(f"{'='*80}")

    # Summary table
    df = generate_summary_table(results)
    print("\nPerformance Summary:")
    print(df.to_string(index=False))

    # Winner analysis
    regrets = {alg: np.mean(results['regret'][alg][:, -1]) for alg in results['algorithms']}
    winner = min(regrets, key=regrets.get)
    baseline_regret = regrets.get('ForcedSampling', regrets[winner])

    print(f"\n🏆 Best Algorithm: {winner}")
    print(f"   Final Regret: {regrets[winner]:.2f}")
    if 'ForcedSampling' in regrets and winner != 'ForcedSampling':
        improvement = (1 - regrets[winner]/baseline_regret) * 100
        print(f"   Improvement vs Forced Sampling: {improvement:.1f}% better")

    return results

def main():
    parser = argparse.ArgumentParser(description='Generate report results')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full'],
                       help='Quick test or full results')
    args = parser.parse_args()

    print("="*80)
    print("  GENERATING REPORT RESULTS")
    print("="*80)

    # Define scenarios
    scenarios = {
        'Default (Baseline)': 'results/project4/scenario_default.pkl',
        'Gaussian Beta': 'results/project4/scenario_gaussian.pkl',
        'Sparse Coefficients': 'results/project4/scenario_sparse.pkl',
        'Heavy-Tailed Beta': 'results/project4/scenario_heavy_tailed.pkl',
    }

    # Check what exists
    print("\nChecking available results...")
    available = {}
    missing = []

    for name, path in scenarios.items():
        if Path(path).exists():
            print(f"  ✓ {name}: Found")
            available[name] = path
        else:
            print(f"  ✗ {name}: Missing")
            missing.append(name)

    if not available:
        print("\n⚠️  No results found!")
        print("\nTo generate results, run:")
        print("  cd project4files")
        if args.mode == 'quick':
            print("  make scenario-default N_SIM=10 T=100")
            print("  make scenario-gaussian N_SIM=10 T=100")
            print("  make scenario-sparse N_SIM=10 T=100")
        else:
            print("  make workflow-scenarios")
        return

    # Generate summaries for available results
    print("\n" + "="*80)
    print("  SUMMARY STATISTICS FOR REPORT")
    print("="*80)

    all_results = {}
    for name, path in available.items():
        result = print_report_section(name, path)
        if result:
            all_results[name] = result

    # Cross-scenario comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("  CROSS-SCENARIO COMPARISON")
        print("="*80)

        comparison_data = []
        for scenario_name, results in all_results.items():
            for alg in results['algorithms']:
                regret = np.mean(results['regret'][alg][:, -1])
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Algorithm': alg,
                    'Final Regret': f"{regret:.2f}"
                })

        df_comparison = pd.DataFrame(comparison_data)
        pivot = df_comparison.pivot(index='Algorithm', columns='Scenario', values='Final Regret')
        print("\nFinal Regret Across Scenarios:")
        print(pivot.to_string())

    # Save summary for report
    output_file = 'report_summary.txt'
    print(f"\n💾 Saving summary to: {output_file}")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("  PROJECT 4 REPORT RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for name, path in available.items():
            results = load_results(path)
            f.write(f"\n{name.upper()}\n")
            f.write("-"*80 + "\n")
            df = generate_summary_table(results)
            f.write(df.to_string(index=False))
            f.write("\n\n")

    print(f"✓ Summary saved to: {output_file}")

    # Instructions for missing scenarios
    if missing:
        print("\n" + "="*80)
        print("  TO COMPLETE ALL RESULTS")
        print("="*80)
        print("\nMissing scenarios:")
        for name in missing:
            scenario_key = name.lower().replace(' (', '').replace(')', '').replace(' ', '-')
            if 'heavy' in name:
                scenario_key = 'heavy'
            print(f"  • {name}")
            if args.mode == 'quick':
                print(f"    make scenario-{scenario_key} N_SIM=10 T=100")
            else:
                print(f"    make scenario-{scenario_key}")

    print("\n" + "="*80)
    print("  DONE")
    print("="*80)

if __name__ == "__main__":
    main()
