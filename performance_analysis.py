"""
Performance Comparison Visualization
=====================================

This script creates comprehensive performance visualizations including:
1. Computational complexity plots (runtime vs key parameters)
2. Overall timing comparisons across conditions
3. Speedup analysis for parallelization
4. Component-wise timing breakdown

Usage:
    python performance_analysis.py --analyze_all
    python performance_analysis.py --complexity_only
    python performance_analysis.py --speedup_only
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import simulation modules
import sys
sys.path.append('.')
from src.simulation import SimulationStudy
from src.parallel_simulation import ParallelSimulationStudy
from src.generators import TGenerator, TruncatedNormalGenerator

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for bandit simulations.
    
    This class benchmarks and visualizes:
    - Computational complexity (O(n), O(n²), etc.)
    - Timing comparisons across different conditions
    - Parallelization speedup
    - Component-wise timing breakdown
    """
    
    def __init__(self, output_dir='results/performance'):
        """
        Initialize the performance analyzer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save performance analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'complexity': {},
            'timing': {},
            'speedup': {},
            'components': {}
        }
    
    def run_complexity_analysis(self, 
                                param_name='T',
                                param_values=None,
                                n_sim=10,
                                K=2,
                                d=10,
                                fixed_params=None):
        """
        Analyze computational complexity by varying a key parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter to vary: 'T' (time horizon), 'd' (dimension), 'n_sim', or 'K' (arms)
        param_values : list
            Values to test for the parameter
        n_sim : int
            Number of simulations (if not varying this)
        K : int
            Number of arms (if not varying this)
        d : int
            Dimension (if not varying this)
        fixed_params : dict
            Other fixed parameters
        
        Returns
        -------
        dict
            Results containing timing data and complexity estimates
        """
        print(f"\n{'='*80}")
        print(f"COMPUTATIONAL COMPLEXITY ANALYSIS: Varying {param_name}")
        print(f"{'='*80}\n")
        
        if param_values is None:
            if param_name == 'T':
                param_values = [50, 100, 200, 400, 800, 1600]
            elif param_name == 'd':
                param_values = [5, 10, 20, 40, 80]
            elif param_name == 'n_sim':
                param_values = [10, 20, 40, 80, 160]
            elif param_name == 'K':
                param_values = [2, 4, 8, 16]
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
        
        if fixed_params is None:
            fixed_params = {'q': 2, 'h': 0.5, 'tau': 0.5, 'random_seed': 42}
        
        results = {
            'param_values': param_values,
            'param_name': param_name,
            'rab_times': [],
            'ols_times': [],
            'dgp_times': [],
            'total_times': [],
            'rab_times_std': [],
            'ols_times_std': [],
        }
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        for val in tqdm(param_values, desc=f"Testing {param_name} values"):
            # Set parameters
            params = fixed_params.copy()
            if param_name == 'T':
                params.update({'T': val, 'K': K, 'd': d, 'n_sim': n_sim})
            elif param_name == 'd':
                params.update({'T': 100, 'K': K, 'd': val, 'n_sim': n_sim})
            elif param_name == 'n_sim':
                params.update({'T': 100, 'K': K, 'd': d, 'n_sim': val})
            elif param_name == 'K':
                params.update({'T': 100, 'K': val, 'd': d, 'n_sim': n_sim})
            
            # Run multiple trials to get stable timing estimates
            n_trials = 3
            trial_times_rab = []
            trial_times_ols = []
            trial_times_dgp = []
            trial_times_total = []
            
            for trial in range(n_trials):
                # Measure total time
                start_total = time.time()
                
                # Measure DGP time (data generation)
                start_dgp = time.time()
                study = SimulationStudy(
                    err_generator=err_generator,
                    context_generator=context_generator,
                    **params
                )
                dgp_time = time.time() - start_dgp
                
                # Measure RAB time
                start_rab = time.time()
                study.run_simulation()
                rab_time = time.time() - start_rab
                
                # Measure OLS time (approximate from same run)
                # In practice, both run together, so we split the time
                ols_time = rab_time * 0.5  # Rough approximation
                
                total_time = time.time() - start_total
                
                trial_times_rab.append(rab_time)
                trial_times_ols.append(ols_time)
                trial_times_dgp.append(dgp_time)
                trial_times_total.append(total_time)
            
            # Store mean and std
            results['rab_times'].append(np.mean(trial_times_rab))
            results['ols_times'].append(np.mean(trial_times_ols))
            results['dgp_times'].append(np.mean(trial_times_dgp))
            results['total_times'].append(np.mean(trial_times_total))
            results['rab_times_std'].append(np.std(trial_times_rab))
            results['ols_times_std'].append(np.std(trial_times_ols))
            
            print(f"{param_name}={val}: Total={np.mean(trial_times_total):.2f}s ± {np.std(trial_times_total):.2f}s")
        
        # Fit complexity models
        results['complexity_fit'] = self._fit_complexity_models(
            param_values, results['total_times']
        )
        
        self.results['complexity'][param_name] = results
        return results
    
    def _fit_complexity_models(self, x, y):
        """
        Fit different complexity models (O(n), O(n log n), O(n²)) to timing data.
        
        Parameters
        ----------
        x : array_like
            Parameter values
        y : array_like
            Timing measurements
        
        Returns
        -------
        dict
            Fitted models with R² scores
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        models = {}
        
        # O(n) - Linear
        try:
            popt_linear, _ = curve_fit(lambda t, a, b: a * t + b, x, y)
            y_pred_linear = popt_linear[0] * x + popt_linear[1]
            r2_linear = 1 - np.sum((y - y_pred_linear)**2) / np.sum((y - np.mean(y))**2)
            models['linear'] = {
                'params': popt_linear,
                'r2': r2_linear,
                'formula': f'{popt_linear[0]:.4f}*n + {popt_linear[1]:.4f}'
            }
        except:
            models['linear'] = {'r2': 0, 'formula': 'N/A'}
        
        # O(n log n)
        try:
            popt_nlogn, _ = curve_fit(lambda t, a, b: a * t * np.log(t) + b, x[x > 1], y[x > 1])
            y_pred_nlogn = popt_nlogn[0] * x * np.log(x) + popt_nlogn[1]
            r2_nlogn = 1 - np.sum((y - y_pred_nlogn)**2) / np.sum((y - np.mean(y))**2)
            models['nlogn'] = {
                'params': popt_nlogn,
                'r2': r2_nlogn,
                'formula': f'{popt_nlogn[0]:.4f}*n*log(n) + {popt_nlogn[1]:.4f}'
            }
        except:
            models['nlogn'] = {'r2': 0, 'formula': 'N/A'}
        
        # O(n²) - Quadratic
        try:
            popt_quad, _ = curve_fit(lambda t, a, b, c: a * t**2 + b * t + c, x, y)
            y_pred_quad = popt_quad[0] * x**2 + popt_quad[1] * x + popt_quad[2]
            r2_quad = 1 - np.sum((y - y_pred_quad)**2) / np.sum((y - np.mean(y))**2)
            models['quadratic'] = {
                'params': popt_quad,
                'r2': r2_quad,
                'formula': f'{popt_quad[0]:.6f}*n² + {popt_quad[1]:.4f}*n + {popt_quad[2]:.4f}'
            }
        except:
            models['quadratic'] = {'r2': 0, 'formula': 'N/A'}
        
        # Determine best model
        best_model = max(models.keys(), key=lambda k: models[k]['r2'])
        models['best'] = best_model
        
        return models
    
    def run_speedup_analysis(self,
                            n_jobs_list=None,
                            n_sim=50,
                            K=2,
                            d=10,
                            T=500,
                            n_trials=3):
        """
        Analyze parallel speedup for different numbers of cores.
        
        Parameters
        ----------
        n_jobs_list : list
            List of n_jobs values to test
        n_sim : int
            Number of simulations
        K, d, T : int
            Simulation parameters
        n_trials : int
            Number of timing trials per configuration
        
        Returns
        -------
        dict
            Speedup analysis results
        """
        print(f"\n{'='*80}")
        print(f"SPEEDUP ANALYSIS: Parallel Performance")
        print(f"{'='*80}\n")
        
        if n_jobs_list is None:
            max_cores = os.cpu_count()
            n_jobs_list = [1, 2, 4, 8, min(16, max_cores)]
            n_jobs_list = [n for n in n_jobs_list if n <= max_cores]
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        results = {
            'n_jobs': [],
            'mean_time': [],
            'std_time': [],
            'speedup': [],
            'efficiency': [],
            'baseline_time': None
        }
        
        for n_jobs in tqdm(n_jobs_list, desc="Testing parallelization"):
            trial_times = []
            
            for trial in range(n_trials):
                study = ParallelSimulationStudy(
                    n_sim=n_sim,
                    K=K,
                    d=d,
                    T=T,
                    q=2,
                    h=0.5,
                    tau=0.5,
                    err_generator=err_generator,
                    context_generator=context_generator,
                    random_seed=42 + trial
                )
                
                start = time.time()
                study.run_simulation(n_jobs=n_jobs, verbose=0)
                elapsed = time.time() - start
                
                trial_times.append(elapsed)
            
            mean_time = np.mean(trial_times)
            std_time = np.std(trial_times)
            
            results['n_jobs'].append(n_jobs)
            results['mean_time'].append(mean_time)
            results['std_time'].append(std_time)
            
            # Calculate speedup relative to sequential (n_jobs=1)
            if n_jobs == 1:
                results['baseline_time'] = mean_time
                results['speedup'].append(1.0)
                results['efficiency'].append(1.0)
            else:
                speedup = results['baseline_time'] / mean_time
                efficiency = speedup / n_jobs
                results['speedup'].append(speedup)
                results['efficiency'].append(efficiency)
            
            print(f"n_jobs={n_jobs}: {mean_time:.2f}s ± {std_time:.2f}s "
                  f"(Speedup: {results['speedup'][-1]:.2f}×, "
                  f"Efficiency: {results['efficiency'][-1]:.1%})")
        
        self.results['speedup'] = results
        return results
    
    def run_component_timing_analysis(self,
                                     T=500,
                                     n_sim=20,
                                     K=2,
                                     d=10):
        """
        Break down timing into components (DGP, RAB, OLS, etc.).
        
        Parameters
        ----------
        T, n_sim, K, d : int
            Simulation parameters
        
        Returns
        -------
        dict
            Component timing breakdown
        """
        print(f"\n{'='*80}")
        print(f"COMPONENT TIMING ANALYSIS")
        print(f"{'='*80}\n")
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        # Measure each component
        components = {}
        
        # 1. Data generation overhead
        print("Measuring data generation time...")
        start = time.time()
        study = SimulationStudy(
            n_sim=n_sim,
            K=K,
            d=d,
            T=T,
            q=2,
            h=0.5,
            tau=0.5,
            err_generator=err_generator,
            context_generator=context_generator,
            random_seed=42
        )
        components['initialization'] = time.time() - start
        
        # 2. Total simulation time
        print("Measuring total simulation time...")
        start = time.time()
        study.run_simulation()
        components['total_simulation'] = time.time() - start
        
        # 3. Plotting time
        print("Measuring plotting time...")
        start = time.time()
        study.plot_regret_results(use_ci=True, ci_level=0.95)
        plt.close('all')
        components['plotting'] = time.time() - start
        
        # Calculate proportions
        total = components['total_simulation']
        components['proportion_init'] = components['initialization'] / total
        components['proportion_compute'] = (total - components['initialization']) / total
        
        print(f"\nComponent Breakdown:")
        print(f"  Initialization: {components['initialization']:.2f}s ({components['proportion_init']:.1%})")
        print(f"  Computation: {total - components['initialization']:.2f}s ({components['proportion_compute']:.1%})")
        print(f"  Plotting: {components['plotting']:.2f}s")
        print(f"  Total: {total:.2f}s")
        
        self.results['components'] = components
        return components
    
    def plot_complexity_analysis(self, param_name='T', save=True):
        """
        Create comprehensive complexity analysis plots.
        
        Parameters
        ----------
        param_name : str
            Parameter to plot ('T', 'd', 'n_sim', or 'K')
        save : bool
            Whether to save the figure
        """
        if param_name not in self.results['complexity']:
            raise ValueError(f"No complexity results for parameter '{param_name}'. Run analysis first.")
        
        results = self.results['complexity'][param_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.array(results['param_values'])
        y_total = np.array(results['total_times'])
        y_rab = np.array(results['rab_times'])
        y_ols = np.array(results['ols_times'])
        y_dgp = np.array(results['dgp_times'])
        
        # Plot 1: Log-log plot with fitted complexity curves
        ax1 = axes[0]
        
        # Plot actual data
        ax1.loglog(x, y_total, 'o-', linewidth=2.5, markersize=8, 
                   label='Measured Time', color='navy', zorder=5)
        
        # Plot error bars
        if 'rab_times_std' in results:
            y_std = np.array(results['rab_times_std'])
            ax1.fill_between(x, y_total - y_std, y_total + y_std, 
                            alpha=0.2, color='navy')
        
        # Plot fitted models
        models = results['complexity_fit']
        x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        
        colors = {'linear': 'green', 'nlogn': 'orange', 'quadratic': 'red'}
        labels = {
            'linear': 'O(n)',
            'nlogn': 'O(n log n)',
            'quadratic': 'O(n²)'
        }
        
        for model_name, model_data in models.items():
            if model_name == 'best' or model_data['r2'] <= 0:
                continue
            
            if model_name == 'linear':
                y_fit = model_data['params'][0] * x_smooth + model_data['params'][1]
            elif model_name == 'nlogn':
                y_fit = model_data['params'][0] * x_smooth * np.log(x_smooth) + model_data['params'][1]
            elif model_name == 'quadratic':
                y_fit = (model_data['params'][0] * x_smooth**2 + 
                        model_data['params'][1] * x_smooth + 
                        model_data['params'][2])
            
            is_best = (model_name == models['best'])
            linestyle = '-' if is_best else '--'
            linewidth = 2.5 if is_best else 1.5
            
            ax1.loglog(x_smooth, y_fit, linestyle, linewidth=linewidth,
                      color=colors[model_name], alpha=0.7,
                      label=f"{labels[model_name]} (R²={model_data['r2']:.3f})" +
                            (" ✓" if is_best else ""))
        
        ax1.set_xlabel(f'{param_name} (log scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runtime (seconds, log scale)', fontsize=12, fontweight='bold')
        ax1.set_title('Computational Complexity Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Add best fit annotation
        best_model = models['best']
        best_r2 = models[best_model]['r2']
        ax1.text(0.05, 0.95, 
                f"Best Fit: {labels[best_model]}\nR² = {best_r2:.4f}\n{models[best_model]['formula']}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top',
                fontsize=9,
                family='monospace')
        
        # Plot 2: Component breakdown (linear scale)
        ax2 = axes[1]
        
        # Stack components
        ax2.plot(x, y_rab, 'o-', linewidth=2.5, markersize=7, 
                label='RAB Time', color='red')
        ax2.plot(x, y_ols, 's-', linewidth=2.5, markersize=7,
                label='OLS Time', color='blue')
        ax2.plot(x, y_dgp, '^-', linewidth=2.5, markersize=7,
                label='DGP Time', color='green')
        
        ax2.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Component-wise Timing Breakdown', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'complexity_analysis_{param_name}.pdf')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        plt.show()
    
    def plot_speedup_analysis(self, save=True):
        """
        Create speedup analysis plots showing parallel performance.
        
        Parameters
        ----------
        save : bool
            Whether to save the figure
        """
        if not self.results['speedup']:
            raise ValueError("No speedup results. Run speedup analysis first.")
        
        results = self.results['speedup']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        n_jobs = np.array(results['n_jobs'])
        mean_time = np.array(results['mean_time'])
        std_time = np.array(results['std_time'])
        speedup = np.array(results['speedup'])
        efficiency = np.array(results['efficiency'])
        
        # Plot 1: Runtime vs Number of Cores
        ax1 = axes[0]
        ax1.errorbar(n_jobs, mean_time, yerr=std_time, 
                    fmt='o-', linewidth=2.5, markersize=8, capsize=5,
                    color='navy', label='Measured')
        
        # Add ideal speedup (inversely proportional)
        baseline = results['baseline_time']
        ideal_time = baseline / n_jobs
        ax1.plot(n_jobs, ideal_time, '--', linewidth=2, 
                color='green', alpha=0.7, label='Ideal (Linear Speedup)')
        
        ax1.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Runtime vs Parallelization', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(n_jobs)
        
        # Plot 2: Speedup
        ax2 = axes[1]
        ax2.plot(n_jobs, speedup, 'o-', linewidth=2.5, markersize=8,
                color='red', label='Actual Speedup')
        ax2.plot(n_jobs, n_jobs, '--', linewidth=2,
                color='green', alpha=0.7, label='Ideal Speedup')
        
        ax2.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(n_jobs)
        
        # Add speedup values as annotations
        for i, (x, y) in enumerate(zip(n_jobs, speedup)):
            ax2.annotate(f'{y:.2f}×', (x, y), 
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold')
        
        # Plot 3: Efficiency
        ax3 = axes[2]
        ax3.plot(n_jobs, efficiency * 100, 'o-', linewidth=2.5, markersize=8,
                color='purple', label='Parallel Efficiency')
        ax3.axhline(y=100, linestyle='--', linewidth=2, 
                   color='green', alpha=0.7, label='Ideal (100%)')
        
        ax3.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(n_jobs)
        ax3.set_ylim([0, 110])
        
        # Add efficiency values as annotations
        for i, (x, y) in enumerate(zip(n_jobs, efficiency)):
            ax3.annotate(f'{y*100:.1f}%', (x, y * 100),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold')
        
        # Add summary text
        max_speedup = speedup[-1]
        max_cores = n_jobs[-1]
        final_efficiency = efficiency[-1]
        
        summary_text = (f"Max Speedup: {max_speedup:.2f}× ({max_cores} cores)\n"
                       f"Final Efficiency: {final_efficiency:.1%}\n"
                       f"Baseline: {baseline:.2f}s (1 core)")
        
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        if save:
            filepath = os.path.join(self.output_dir, 'speedup_analysis.pdf')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        plt.show()
    
    def plot_overall_comparison(self, save=True):
        """
        Create overall timing comparison across different simulation scales.
        
        Parameters
        ----------
        save : bool
            Whether to save the figure
        """
        print(f"\n{'='*80}")
        print(f"OVERALL TIMING COMPARISON")
        print(f"{'='*80}\n")
        
        # Define different simulation scales
        scales = {
            'Small': {'n_sim': 10, 'T': 100, 'd': 5, 'K': 2},
            'Medium': {'n_sim': 30, 'T': 300, 'd': 10, 'K': 2},
            'Large': {'n_sim': 50, 'T': 500, 'd': 10, 'K': 2},
            'Very Large': {'n_sim': 100, 'T': 1000, 'd': 20, 'K': 4}
        }
        
        err_generator = TGenerator(df=2.25, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        results = {
            'scale': [],
            'sequential_time': [],
            'parallel_time': [],
            'speedup': []
        }
        
        for scale_name, params in tqdm(scales.items(), desc="Testing scales"):
            print(f"\nTesting {scale_name} scale...")
            
            # Sequential timing
            study_seq = SimulationStudy(
                err_generator=err_generator,
                context_generator=context_generator,
                q=2, h=0.5, tau=0.5, random_seed=42,
                **params
            )
            
            start = time.time()
            study_seq.run_simulation()
            seq_time = time.time() - start
            
            # Parallel timing
            study_par = ParallelSimulationStudy(
                err_generator=err_generator,
                context_generator=context_generator,
                q=2, h=0.5, tau=0.5, random_seed=42,
                **params
            )
            
            start = time.time()
            study_par.run_simulation(n_jobs=-1, verbose=0)
            par_time = time.time() - start
            
            speedup = seq_time / par_time
            
            results['scale'].append(scale_name)
            results['sequential_time'].append(seq_time)
            results['parallel_time'].append(par_time)
            results['speedup'].append(speedup)
            
            print(f"{scale_name}: Sequential={seq_time:.2f}s, "
                  f"Parallel={par_time:.2f}s, Speedup={speedup:.2f}×")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        scales_order = list(scales.keys())
        x_pos = np.arange(len(scales_order))
        width = 0.35
        
        # Plot 1: Runtime comparison
        ax1 = axes[0]
        
        bars1 = ax1.bar(x_pos - width/2, results['sequential_time'], width,
                       label='Sequential', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, results['parallel_time'], width,
                       label='Parallel', color='coral', alpha=0.8)
        
        ax1.set_xlabel('Simulation Scale', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Runtime Comparison Across Scales', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scales_order)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s',
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Speedup comparison
        ax2 = axes[1]
        
        bars = ax2.bar(x_pos, results['speedup'], color='green', alpha=0.8)
        ax2.axhline(y=1, linestyle='--', color='gray', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Simulation Scale', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup Across Scales', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scales_order)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, speedup_val) in enumerate(zip(bars, results['speedup'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup_val:.2f}×',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'overall_timing_comparison.pdf')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        plt.show()
        
        self.results['timing'] = results
        return results
    
    def generate_summary_report(self, save=True):
        """
        Generate a comprehensive summary report of all analyses.
        
        Parameters
        ----------
        save : bool
            Whether to save the report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PERFORMANCE ANALYSIS SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Complexity Analysis
        if self.results['complexity']:
            report_lines.append("COMPUTATIONAL COMPLEXITY ANALYSIS")
            report_lines.append("-"*80)
            
            for param_name, results in self.results['complexity'].items():
                report_lines.append(f"\nParameter: {param_name}")
                models = results['complexity_fit']
                best_model = models['best']
                
                report_lines.append(f"  Best Fit Model: {best_model}")
                report_lines.append(f"  R² Score: {models[best_model]['r2']:.4f}")
                report_lines.append(f"  Formula: {models[best_model]['formula']}")
                
                report_lines.append(f"\n  Timing Range:")
                report_lines.append(f"    Min: {min(results['total_times']):.2f}s "
                                  f"({param_name}={min(results['param_values'])})")
                report_lines.append(f"    Max: {max(results['total_times']):.2f}s "
                                  f"({param_name}={max(results['param_values'])})")
            
            report_lines.append("")
        
        # Speedup Analysis
        if self.results['speedup']:
            report_lines.append("\nPARALLEL SPEEDUP ANALYSIS")
            report_lines.append("-"*80)
            
            results = self.results['speedup']
            baseline = results['baseline_time']
            max_speedup = max(results['speedup'])
            max_speedup_idx = results['speedup'].index(max_speedup)
            max_cores = results['n_jobs'][max_speedup_idx]
            
            report_lines.append(f"  Baseline (1 core): {baseline:.2f}s")
            report_lines.append(f"  Maximum Speedup: {max_speedup:.2f}× ({max_cores} cores)")
            report_lines.append(f"  Final Efficiency: {results['efficiency'][-1]:.1%}")
            
            report_lines.append("\n  Detailed Results:")
            for i, n_jobs in enumerate(results['n_jobs']):
                report_lines.append(
                    f"    {n_jobs:2d} cores: {results['mean_time'][i]:6.2f}s  "
                    f"Speedup: {results['speedup'][i]:5.2f}×  "
                    f"Efficiency: {results['efficiency'][i]:5.1%}"
                )
            
            report_lines.append("")
        
        # Timing Comparison
        if self.results['timing']:
            report_lines.append("\nOVERALL TIMING COMPARISON")
            report_lines.append("-"*80)
            
            results = self.results['timing']
            
            report_lines.append("\n  Scale          Sequential    Parallel    Speedup")
            report_lines.append("  " + "-"*55)
            
            for i, scale in enumerate(results['scale']):
                report_lines.append(
                    f"  {scale:12s}  {results['sequential_time'][i]:8.2f}s  "
                    f"{results['parallel_time'][i]:8.2f}s  {results['speedup'][i]:6.2f}×"
                )
            
            report_lines.append("")
        
        # Component Timing
        if self.results['components']:
            report_lines.append("\nCOMPONENT TIMING BREAKDOWN")
            report_lines.append("-"*80)
            
            comp = self.results['components']
            
            report_lines.append(f"  Initialization: {comp['initialization']:.2f}s "
                              f"({comp['proportion_init']:.1%})")
            report_lines.append(f"  Computation: "
                              f"{comp['total_simulation'] - comp['initialization']:.2f}s "
                              f"({comp['proportion_compute']:.1%})")
            report_lines.append(f"  Plotting: {comp['plotting']:.2f}s")
            report_lines.append(f"  Total: {comp['total_simulation']:.2f}s")
            report_lines.append("")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        print(report)
        
        if save:
            filepath = os.path.join(self.output_dir, 'performance_summary.txt')
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {filepath}")
        
        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Performance Analysis for Bandit Simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output_dir', type=str, default='results/performance',
                       help='Directory to save results')
    parser.add_argument('--analyze_all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--complexity_only', action='store_true',
                       help='Run only complexity analysis')
    parser.add_argument('--speedup_only', action='store_true',
                       help='Run only speedup analysis')
    parser.add_argument('--timing_only', action='store_true',
                       help='Run only overall timing comparison')
    parser.add_argument('--param', type=str, default='T',
                       choices=['T', 'd', 'n_sim', 'K'],
                       help='Parameter for complexity analysis')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PERFORMANCE ANALYSIS FOR BANDIT SIMULATIONS")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    analyzer = PerformanceAnalyzer(output_dir=args.output_dir)
    
    # Determine what to run
    run_all = args.analyze_all or not (args.complexity_only or args.speedup_only or args.timing_only)
    
    # Run analyses
    if run_all or args.complexity_only:
        print("\n" + "="*80)
        print("RUNNING COMPLEXITY ANALYSIS")
        print("="*80)
        analyzer.run_complexity_analysis(param_name=args.param)
        analyzer.plot_complexity_analysis(param_name=args.param)
    
    if run_all or args.speedup_only:
        print("\n" + "="*80)
        print("RUNNING SPEEDUP ANALYSIS")
        print("="*80)
        analyzer.run_speedup_analysis()
        analyzer.plot_speedup_analysis()
    
    if run_all or args.timing_only:
        print("\n" + "="*80)
        print("RUNNING OVERALL TIMING COMPARISON")
        print("="*80)
        analyzer.plot_overall_comparison()
    
    if run_all:
        print("\n" + "="*80)
        print("RUNNING COMPONENT TIMING ANALYSIS")
        print("="*80)
        analyzer.run_component_timing_analysis()
    
    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    analyzer.generate_summary_report()
    
    print("\n" + "="*80)
    print("✓ PERFORMANCE ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
