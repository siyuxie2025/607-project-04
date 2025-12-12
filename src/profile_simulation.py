"""
Profiling Wrapper for Bandit Simulation (src/simulation.py)
=============================================================

This module profiles your SimulationStudy class with numerical tracking.

Usage:
    python profile_simulation.py --full
    python profile_simulation.py --quick --n_sim=10 --T=100 --df=2.0
"""

import time
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import your actual simulation code
from simulation import SimulationStudy
from generators import TGenerator, TruncatedNormalGenerator, UniformGenerator

# ==============================================
# SETUP LOGGING FOR NUMERICAL ISSUES
# ==============================================
Path('results').mkdir(exist_ok=True)

logging.basicConfig(
    filename='results/simulation_warnings.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)


class NumericalIssueTracker:
    """Track numerical warnings and convergence issues during simulation"""
    
    def __init__(self):
        self.warnings = []
        self.convergence_failures = []
        self.exceptions = []
        self.large_values = []
    
    def log_warning(self, message: str, context: Dict):
        """Log a numerical warning"""
        entry = {
            'message': message,
            'context': context,
            'timestamp': time.time()
        }
        self.warnings.append(entry)
        logger.warning(f"{message} | {context}")
    
    def log_convergence_failure(self, method: str, iteration: int, context: Dict):
        """Log convergence failure"""
        entry = {
            'method': method,
            'iteration': iteration,
            'context': context,
            'timestamp': time.time()
        }
        self.convergence_failures.append(entry)
        logger.error(f"Convergence failure in {method} at iteration {iteration} | {context}")
    
    def log_exception(self, exception: Exception, context: Dict):
        """Log exception"""
        entry = {
            'exception': str(exception),
            'type': type(exception).__name__,
            'context': context,
            'timestamp': time.time()
        }
        self.exceptions.append(entry)
        logger.error(f"Exception: {exception} | {context}")
    
    def log_large_value(self, value_type: str, value: float, context: Dict):
        """Log extremely large values"""
        entry = {
            'value_type': value_type,
            'value': value,
            'context': context,
            'timestamp': time.time()
        }
        self.large_values.append(entry)
        logger.warning(f"Large {value_type}: {value:.2e} | {context}")
    
    def summary(self) -> Dict:
        """Return summary of all tracked issues"""
        return {
            'total_warnings': len(self.warnings),
            'total_convergence_failures': len(self.convergence_failures),
            'total_exceptions': len(self.exceptions),
            'total_large_values': len(self.large_values),
            'warnings': self.warnings,
            'convergence_failures': self.convergence_failures,
            'exceptions': self.exceptions,
            'large_values': self.large_values
        }
    
    def save_summary(self, filepath: str = 'results/numerical_issues_summary.json'):
        """Save summary to JSON"""
        summary = self.summary()
        
        # Convert to JSON-serializable format
        json_summary = {
            'total_warnings': summary['total_warnings'],
            'total_convergence_failures': summary['total_convergence_failures'],
            'total_exceptions': summary['total_exceptions'],
            'total_large_values': summary['total_large_values'],
            'warnings_sample': summary['warnings'][:10],  # First 10
            'convergence_failures_sample': summary['convergence_failures'][:10],
            'exceptions_sample': summary['exceptions'][:10],
            'large_values_sample': summary['large_values'][:10]
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        logger.info(f"Numerical issues summary saved to {filepath}")


# Global tracker
issue_tracker = NumericalIssueTracker()


# ==============================================
# WRAPPER FOR YOUR SIMULATION WITH TRACKING
# ==============================================

def run_simulation_with_tracking(
    n_sim: int,
    K: int,
    d: int,
    T: int,
    df: float,
    q: int = 2,
    h: float = 0.5,
    tau: float = 0.5,
    random_seed: int = None,
    verbose: bool = False
):
    """
    Wrapper around SimulationStudy that adds numerical tracking.
    
    Parameters
    ----------
    n_sim : int
        Number of simulation replications
    K : int
        Number of arms
    d : int
        Dimension of context vectors
    T : int
        Number of time steps (rounds)
    df : float
        Degrees of freedom for t-distribution (controls tail heaviness)
    q : int
        Parameter for forced sampling (initial exploration)
    h : float
        Hyperparameter for the bandit algorithms
    tau : float
        Quantile level for risk-aware bandit
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress information
    
    Returns
    -------
    dict
        Results dictionary from SimulationStudy
    """
    
    context = {
        'n_sim': n_sim,
        'K': K,
        'd': d,
        'T': T,
        'df': df,
        'q': q,
        'h': h,
        'tau': tau
    }
    
    # Catch warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Create error generator (t-distribution with specified df)
            err_generator = TGenerator(df=df, scale=0.7)
            
            # Create context generator
            context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
            
            # Create simulation study
            study = SimulationStudy(
                n_sim=n_sim,
                K=K,
                d=d,
                T=T,
                q=q,
                h=h,
                tau=tau,
                err_generator=err_generator,
                context_generator=context_generator,
                random_seed=random_seed
            )
            
            # Run simulation
            results = study.run_simulation()
            
            # Check results for numerical issues
            if results is not None:
                # Check for NaN or Inf in regrets
                for method_name, key in [('RiskAware', 'cumulated_regret_RiskAware'),
                                         ('OLS', 'cumulated_regret_OLS')]:
                    regret = results[key]
                    
                    if np.any(np.isnan(regret)):
                        issue_tracker.log_warning(
                            f"NaN detected in {method_name} regret",
                            context
                        )
                    
                    if np.any(np.isinf(regret)):
                        issue_tracker.log_warning(
                            f"Inf detected in {method_name} regret",
                            context
                        )
                    
                    # Check for extremely large values
                    max_regret = np.max(np.abs(regret))
                    if max_regret > 1e6:
                        issue_tracker.log_large_value(
                            f'{method_name}_regret',
                            max_regret,
                            context
                        )
                
                # Check beta errors
                for method_name, key in [('RiskAware', 'beta_errors_rab'),
                                         ('OLS', 'beta_errors_ols')]:
                    beta_errors = results[key]
                    
                    if np.any(np.isnan(beta_errors)):
                        issue_tracker.log_warning(
                            f"NaN detected in {method_name} beta errors",
                            context
                        )
                    
                    max_error = np.max(np.abs(beta_errors))
                    if max_error > 1e3:
                        issue_tracker.log_large_value(
                            f'{method_name}_beta_error',
                            max_error,
                            context
                        )
            
            # Log any Python warnings
            if len(w) > 0:
                for warning in w:
                    issue_tracker.log_warning(
                        f"{warning.category.__name__}: {warning.message}",
                        context
                    )
            
            return results
        
        except Exception as e:
            issue_tracker.log_exception(e, context)
            raise


# ==============================================
# PROFILING FUNCTIONS
# ==============================================

def profile_with_cprofile(
    output_file: str = 'results/simulation.prof',
    n_sim: int = 10,
    K: int = 5,
    d: int = 10,
    T: int = 100,
    df: float = 2.0,
    **kwargs
):
    """
    Profile SimulationStudy using cProfile.
    
    This shows which functions consume the most time.
    """
    print("\n" + "="*80)
    print("PROFILING WITH cProfile")
    print("="*80)
    print(f"Configuration: n_sim={n_sim}, K={K}, d={d}, T={T}, df={df}")
    print("This may take a few minutes...")
    print("-"*80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = run_simulation_with_tracking(
        n_sim=n_sim,
        K=K,
        d=d,
        T=T,
        df=df,
        **kwargs
    )
    
    profiler.disable()
    
    # Save stats
    profiler.dump_stats(output_file)
    
    # Print to console
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions
    
    print("\nTOP FUNCTIONS BY CUMULATIVE TIME:")
    print("="*80)
    print(s.getvalue())
    
    print(f"\nDetailed profile saved to: {output_file}")
    print(f"View interactively with: python -m pstats {output_file}")
    
    return results


def simple_timer(n_sim, K, d, T, df, **kwargs) -> Tuple[float, dict]:
    """Time a single simulation run"""
    start = time.perf_counter()
    results = run_simulation_with_tracking(
        n_sim=n_sim,
        K=K,
        d=d,
        T=T,
        df=df,
        **kwargs
    )
    elapsed = time.perf_counter() - start
    return elapsed, results


def empirical_complexity_analysis(
    param_name: str,
    param_values: List,
    fixed_params: Dict = None,
    n_repeats: int = 3
) -> Dict:
    """
    Analyze empirical computational complexity.
    
    Parameters
    ----------
    param_name : str
        Parameter to vary (e.g., 'T', 'K', 'df', 'n_sim')
    param_values : list
        List of values to test
    fixed_params : dict
        Fixed parameters
    n_repeats : int
        Number of repetitions per configuration
    """
    if fixed_params is None:
        fixed_params = {
            'n_sim': 10,
            'K': 5,
            'd': 10,
            'T': 100,
            'df': 2.0
        }
    
    timings = []
    std_devs = []
    
    print(f"\n{'='*80}")
    print(f"EMPIRICAL COMPLEXITY ANALYSIS: varying {param_name}")
    print(f"{'='*80}")
    print(f"Fixed parameters: {fixed_params}")
    print(f"\n{'Value':<15} {'Mean Time':<15} {'Std Dev':<15} {'Samples'}")
    print("-" * 80)
    
    for value in param_values:
        params = fixed_params.copy()
        params[param_name] = value
        
        run_times = []
        for repeat in range(n_repeats):
            print(f"  Testing {param_name}={value}, run {repeat+1}/{n_repeats}...", end='\r')
            elapsed, _ = simple_timer(**params)
            run_times.append(elapsed)
        
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        timings.append(mean_time)
        std_devs.append(std_time)
        
        print(f"{value:<15} {mean_time:>10.3f}s     ±{std_time:>8.3f}s     n={n_repeats}")
    
    # Fit power law: time = a * n^b
    log_params = np.log(param_values)
    log_times = np.log(timings)
    
    coeffs = np.polyfit(log_params, log_times, 1)
    b = coeffs[0]  # Exponent
    a = np.exp(coeffs[1])  # Coefficient
    
    # Calculate R²
    predicted = a * np.array(param_values) ** b
    ss_res = np.sum((np.array(timings) - predicted) ** 2)
    ss_tot = np.sum((np.array(timings) - np.mean(timings)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\n{'='*80}")
    print(f"COMPLEXITY ESTIMATE FOR {param_name}")
    print(f"{'='*80}")
    print(f"Fitted model: time ≈ {a:.6f} × {param_name}^{b:.2f}")
    print(f"Time complexity: O({param_name}^{b:.2f})")
    print(f"R² = {r_squared:.4f}")
    
    # Interpret complexity
    if b < 0.8:
        interpretation = "sublinear (very efficient!)"
    elif 0.8 <= b < 1.2:
        interpretation = "approximately linear O(n)"
    elif 1.2 <= b < 1.8:
        interpretation = "superlinear (between O(n) and O(n²))"
    elif 1.8 <= b < 2.2:
        interpretation = "approximately quadratic O(n²)"
    elif 2.2 <= b < 2.8:
        interpretation = "between O(n²) and O(n³)"
    else:
        interpretation = "polynomial (high degree)"
    
    print(f"Interpretation: {interpretation}")
    
    if param_name == 'df':
        print("\nNote: df (tail heaviness) typically doesn't affect algorithmic complexity,")
        print("but may impact convergence and numerical stability.")
    
    return {
        'param_name': param_name,
        'param_values': param_values,
        'timings': timings,
        'std_devs': std_devs,
        'complexity_exponent': b,
        'coefficient': a,
        'r_squared': r_squared,
        'interpretation': interpretation
    }


def plot_complexity(results: Dict, save_path: str = None):
    """Plot empirical complexity analysis"""
    if save_path is None:
        save_path = f'results/complexity_{results["param_name"]}.pdf'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    param_values = np.array(results['param_values'])
    timings = np.array(results['timings'])
    std_devs = np.array(results['std_devs'])
    
    # Linear scale
    ax1.errorbar(param_values, timings, yerr=std_devs,
                 marker='o', capsize=5, markersize=8,
                 linewidth=2, label='Measured', color='blue')
    
    predicted = results['coefficient'] * param_values ** results['complexity_exponent']
    ax1.plot(param_values, predicted, 'r--', linewidth=2,
             label=f"O({results['param_name']}^{results['complexity_exponent']:.2f})")
    
    ax1.set_xlabel(results['param_name'], fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Runtime vs {results["param_name"]} (Linear Scale)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale
    ax2.errorbar(param_values, timings, yerr=std_devs,
                 marker='o', capsize=5, markersize=8,
                 linewidth=2, label='Measured', color='blue')
    ax2.plot(param_values, predicted, 'r--', linewidth=2,
             label=f"Slope = {results['complexity_exponent']:.2f}\nR² = {results['r_squared']:.3f}")
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(f"{results['param_name']} (log scale)", fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Log-Log Plot (Slope = Complexity Exponent)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Complexity plot saved to: {save_path}")
    plt.close()


def create_complexity_table(results_list: List[Dict]) -> pd.DataFrame:
    """Create a summary table of complexity analyses"""
    data = []
    for result in results_list:
        data.append({
            'Parameter': result['param_name'],
            'Exponent (b)': f"{result['complexity_exponent']:.2f}",
            'Complexity': f"O({result['param_name']}^{result['complexity_exponent']:.2f})",
            'R²': f"{result['r_squared']:.3f}",
            'Interpretation': result['interpretation'],
            'Min→Max Time': f"{min(result['timings']):.2f}s → {max(result['timings']):.2f}s"
        })
    
    df = pd.DataFrame(data)
    return df


# ==============================================
# MAIN ANALYSIS FUNCTIONS
# ==============================================

def run_full_profiling_suite():
    """Run complete profiling analysis"""
    
    print("\n" + "="*80)
    print("  BANDIT SIMULATION PROFILING SUITE")
    print("  Risk-Aware Bandit vs OLS Bandit with Forced Sampling")
    print("="*80 + "\n")
    
    results_list = []
    
    # 1. Profile single run with cProfile
    print("\n[1/5] Profiling single simulation run with cProfile...")
    print("-" * 80)
    profile_with_cprofile(
        output_file='results/simulation.prof',
        n_sim=10,
        K=5,
        d=10,
        T=100,
        df=2.0
    )
    
    # 2. Complexity vs T (number of rounds) - MOST IMPORTANT
    print("\n[2/5] Analyzing complexity: varying T (number of rounds)...")
    print("-" * 80)
    results_T = empirical_complexity_analysis(
        param_name='T',
        param_values=[50, 100, 200, 500],  # Reasonable range
        fixed_params={'n_sim': 5, 'K': 5, 'd': 10, 'df': 2.0},
        n_repeats=3
    )
    plot_complexity(results_T)
    results_list.append(results_T)
    
    # 3. Complexity vs K (number of arms)
    print("\n[3/5] Analyzing complexity: varying K (number of arms)...")
    print("-" * 80)
    results_K = empirical_complexity_analysis(
        param_name='K',
        param_values=[2, 5, 10, 20],
        fixed_params={'n_sim': 5, 'K': 5, 'd': 10, 'T': 100, 'df': 2.0},
        n_repeats=3
    )
    plot_complexity(results_K)
    results_list.append(results_K)
    
    # 4. Complexity vs d (context dimension)
    print("\n[4/5] Analyzing complexity: varying d (context dimension)...")
    print("-" * 80)
    results_d = empirical_complexity_analysis(
        param_name='d',
        param_values=[5, 10, 20, 50],
        fixed_params={'n_sim': 5, 'K': 5, 'd': 10, 'T': 100, 'df': 2.0},
        n_repeats=3
    )
    plot_complexity(results_d)
    results_list.append(results_d)
    
    # 5. Test numerical stability across df values (heavy tails)
    print("\n[5/5] Testing numerical stability: varying df (tail heaviness)...")
    print("-" * 80)
    df_values = [1.5, 2.0, 2.25, 3.0, 5.0, 10.0]
    
    print("\nTesting different tail heaviness (df) values...")
    print("Lower df = heavier tails, more extreme values\n")
    
    for df in df_values:
        print(f"Testing df={df}...", end=' ')
        start_issues = len(issue_tracker.warnings)
        _, _ = simple_timer(n_sim=5, K=5, d=10, T=100, df=df)
        new_issues = len(issue_tracker.warnings) - start_issues
        if new_issues > 0:
            print(f"⚠ {new_issues} warnings")
        else:
            print("✓ No issues")
    
    # Analyze complexity with df
    results_df = empirical_complexity_analysis(
        param_name='df',
        param_values=df_values,
        fixed_params={'n_sim': 5, 'K': 5, 'd': 10, 'T': 100},
        n_repeats=3
    )
    plot_complexity(results_df)
    results_list.append(results_df)
    
    # Generate summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80 + "\n")
    
    # Complexity table
    df_complexity = create_complexity_table(results_list)
    print("\nComplexity Analysis Summary:")
    print(df_complexity.to_string(index=False))
    df_complexity.to_csv('results/complexity_summary.csv', index=False)
    print("\nTable saved to: results/complexity_summary.csv")
    
    # Numerical issues summary
    print("\n" + "-"*80)
    print("Numerical Stability Report:")
    print("-"*80)
    summary = issue_tracker.summary()
    print(f"Total warnings: {summary['total_warnings']}")
    print(f"Convergence failures: {summary['total_convergence_failures']}")
    print(f"Exceptions: {summary['total_exceptions']}")
    print(f"Large values detected: {summary['total_large_values']}")
    
    if summary['total_warnings'] > 0:
        print(f"\nMost recent warnings (up to 5):")
        for i, warn in enumerate(summary['warnings'][-5:], 1):
            print(f"  {i}. {warn['message']}")
            print(f"     Context: df={warn['context'].get('df', 'N/A')}, "
                  f"T={warn['context'].get('T', 'N/A')}")
    
    # Group warnings by df
    warnings_by_df = {}
    for warn in summary['warnings']:
        df = warn['context'].get('df', 'unknown')
        warnings_by_df[df] = warnings_by_df.get(df, 0) + 1
    
    if warnings_by_df:
        print("\nWarnings by df (tail heaviness):")
        for df in sorted(warnings_by_df.keys()):
            print(f"  df={df}: {warnings_by_df[df]} warnings")
    
    issue_tracker.save_summary()
    
    print("\n" + "="*80)
    print("  FILES GENERATED")
    print("="*80)
    print("  - results/simulation.prof            (cProfile detailed output)")
    print("  - results/complexity_T.pdf           (runtime vs T)")
    print("  - results/complexity_K.pdf           (runtime vs K)")
    print("  - results/complexity_d.pdf           (runtime vs d)")
    print("  - results/complexity_df.pdf          (runtime vs df)")
    print("  - results/complexity_summary.csv     (summary table)")
    print("  - results/simulation_warnings.log    (detailed log)")
    print("  - results/numerical_issues_summary.json")
    print("\n" + "="*80 + "\n")


def quick_profile(n_sim=10, K=5, d=10, T=100, df=2.0):
    """Quick profile of a single configuration"""
    print(f"\n{'='*80}")
    print(f"  QUICK PROFILE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  n_sim = {n_sim} (simulation replications)")
    print(f"  K     = {K} (number of arms)")
    print(f"  d     = {d} (context dimension)")
    print(f"  T     = {T} (time steps/rounds)")
    print(f"  df    = {df} (t-distribution degrees of freedom)")
    print("-" * 80)
    
    start_issues = len(issue_tracker.warnings)
    
    elapsed, result = simple_timer(
        n_sim=n_sim,
        K=K,
        d=d,
        T=T,
        df=df
    )
    
    new_issues = len(issue_tracker.warnings) - start_issues
    
    print(f"\nRuntime: {elapsed:.3f} seconds")
    print(f"Average time per simulation: {elapsed/n_sim:.3f} seconds")
    
    # Check for issues
    if new_issues > 0:
        print(f"⚠ {new_issues} numerical warnings detected (see log)")
    else:
        print("✓ No numerical issues detected")
    
    # Show final regret if available
    if result is not None:
        rab_regret = result['cumulated_regret_RiskAware'][:, -1]
        ols_regret = result['cumulated_regret_OLS'][:, -1]
        
        print(f"\nFinal Regret Summary:")
        print(f"  Risk-Aware: {np.mean(rab_regret):.2f} ± {np.std(rab_regret):.2f}")
        print(f"  OLS:        {np.mean(ols_regret):.2f} ± {np.std(ols_regret):.2f}")
    
    print("="*80 + "\n")
    
    return elapsed, result


# ==============================================
# COMMAND LINE INTERFACE
# ==============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Profile bandit simulation (SimulationStudy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_simulation.py --full
  python profile_simulation.py --quick
  python profile_simulation.py --quick --T=200 --df=1.5
  python profile_simulation.py --quick --n_sim=20 --K=10
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full profiling suite (takes 10-20 minutes)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick profile of single configuration')
    
    # Parameters for quick profile
    parser.add_argument('--n_sim', type=int, default=10,
                       help='Number of simulation replications (default: 10)')
    parser.add_argument('--K', type=int, default=5,
                       help='Number of arms (default: 5)')
    parser.add_argument('--d', type=int, default=10,
                       help='Context dimension (default: 10)')
    parser.add_argument('--T', type=int, default=100,
                       help='Number of time steps/rounds (default: 100)')
    parser.add_argument('--df', type=float, default=2.0,
                       help='Degrees of freedom for t-distribution (default: 2.0)')
    
    args = parser.parse_args()
    
    if args.full:
        print("Running full profiling suite...")
        print("This will take approximately 10-20 minutes depending on your machine.")
        print("Press Ctrl+C to cancel.\n")
        time.sleep(2)
        run_full_profiling_suite()
    elif args.quick:
        quick_profile(
            n_sim=args.n_sim,
            K=args.K,
            d=args.d,
            T=args.T,
            df=args.df
        )
    else:
        # Default: show help
        parser.print_help()
        print("\nNo action specified. Use --full or --quick")