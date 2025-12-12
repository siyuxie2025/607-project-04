"""
Parallel Simulation Implementation
===================================

This module provides parallelized versions of the SimulationStudy class
to speed up Monte Carlo simulations by running replications in parallel.

Key improvements:
- Parallelize across simulation replications (n_sim)
- Parallelize across scenarios (different df values)
- Support for both multiprocessing and joblib backends

Usage:
    from parallel_simulation import ParallelSimulationStudy
    
    study = ParallelSimulationStudy(n_sim=50, K=2, d=10, T=150, ...)
    results = study.run_simulation(n_jobs=8)  # Use 8 cores
"""

import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings

# Import the original SimulationStudy
from simulation import SimulationStudy


class ParallelSimulationStudy(SimulationStudy):
    """
    Parallel version of SimulationStudy that runs replications in parallel.
    
    This class maintains the same interface as SimulationStudy but runs
    multiple simulation replications in parallel across CPU cores.
    
    Expected speedup: 4-8× on typical machines with 8+ cores
    
    Parameters
    ----------
    Same as SimulationStudy
    
    Additional Methods
    ------------------
    run_simulation(n_jobs=-1) : Run in parallel with specified number of jobs
    run_simulation_sequential() : Run sequentially (same as parent class)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as SimulationStudy"""
        super().__init__(*args, **kwargs)
        self._parallel_mode = True
    
    def run_simulation(self, n_jobs: int = -1, backend: str = 'loky', verbose: int = 1):
        """
        Run multiple simulation replications in parallel.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs to run
            - If -1: use all available CPU cores
            - If 1: run sequentially (no parallelization)
            - If > 1: use specified number of cores
        backend : str, optional
            Joblib backend: 'loky' (default), 'multiprocessing', or 'threading'
            - 'loky': robust, works with complex objects (recommended)
            - 'multiprocessing': faster but may have pickling issues
            - 'threading': for I/O bound tasks (not recommended here)
        verbose : int, optional
            Verbosity level (0=silent, 1=progress bar, 10+=debug)
            
        Returns
        -------
        dict
            Results dictionary with cumulative regret, beta errors, etc.
            
        Notes
        -----
        Each replication is independent, making this "embarrassingly parallel".
        Expected speedup with n_jobs=8: ~6-7× (accounting for overhead).
        """
        
        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1 or -1, got {n_jobs}")
        
        # If n_jobs=1, run sequentially
        if n_jobs == 1:
            if verbose:
                print("Running sequentially (n_jobs=1)")
            return self.run_simulation_sequential()
        
        # Run in parallel
        if verbose:
            print(f"Running {self.n_sim} simulations in parallel using {n_jobs} cores")
            print(f"Backend: {backend}")
        
        start_time = time.time()
        
        # Create a worker function that captures the random seed properly
        def run_single_replication(seed_offset):
            """Run one simulation replication with a unique seed"""
            # Create a new random state for this worker
            if self.random_seed is not None:
                worker_seed = self.random_seed + seed_offset
                worker_rng = np.random.default_rng(worker_seed)
                np.random.seed(worker_seed)
            else:
                worker_rng = np.random.default_rng()
            
            # Temporarily replace the RNG
            original_rng = self.rng
            self.rng = worker_rng
            
            try:
                # Run one scenario
                result = self.run_one_scenario()
                return result
            finally:
                # Restore original RNG
                self.rng = original_rng
        
        # Run simulations in parallel
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
                delayed(run_single_replication)(i) 
                for i in range(self.n_sim)
            )
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nParallel execution completed in {elapsed_time:.2f}s")
            print(f"Average time per simulation: {elapsed_time/self.n_sim:.3f}s")
            print(f"Speedup estimate: {n_jobs * 0.8:.1f}× (with overhead)")
        
        # Aggregate results (same as original implementation)
        cumulated_regret_RiskAware = []
        cumulated_regret_OLS = []
        num_diff = []
        all_beta_errors_rab = []
        all_beta_errors_ols = []
        all_beta_estimates_rab = []
        all_beta_estimates_ols = []
        
        for result in results:
            (regret_RAB, regret_OLSB, diff, 
             beta_errors_rab, beta_errors_ols,
             beta_estimates_rab, beta_estimates_ols) = result
            
            cumulated_regret_RiskAware.append(regret_RAB)
            cumulated_regret_OLS.append(regret_OLSB)
            num_diff.append(diff / self.T)
            all_beta_errors_rab.append(np.array(beta_errors_rab))
            all_beta_errors_ols.append(np.array(beta_errors_ols))
            all_beta_estimates_rab.append(np.array(beta_estimates_rab))
            all_beta_estimates_ols.append(np.array(beta_estimates_ols))
        
        # Store results
        self.results = {
            "cumulated_regret_RiskAware": np.array(cumulated_regret_RiskAware),
            "cumulated_regret_OLS": np.array(cumulated_regret_OLS),
            "beta_errors_rab": np.array(all_beta_errors_rab),
            "beta_errors_ols": np.array(all_beta_errors_ols),
            "beta_estimates_rab": np.array(all_beta_estimates_rab),
            "beta_estimates_ols": np.array(all_beta_estimates_ols),
            "num_diff": num_diff,
            "parallel_info": {
                "n_jobs": n_jobs,
                "backend": backend,
                "elapsed_time": elapsed_time,
                "time_per_sim": elapsed_time / self.n_sim
            }
        }
        
        return self.results
    
    def run_simulation_sequential(self):
        """
        Run simulations sequentially (same as original SimulationStudy).
        
        This is useful for comparison or when parallelization is not desired.
        """
        return super().run_simulation()


def run_parallel_scenarios(
    scenarios: List[Dict],
    n_jobs: int = -1,
    verbose: int = 1
) -> List[Dict]:
    """
    Run multiple simulation scenarios in parallel.
    
    This is useful when you want to test different parameter configurations
    (e.g., different df values) in parallel.
    
    Parameters
    ----------
    scenarios : list of dict
        List of scenario configurations. Each dict should contain
        all parameters needed for SimulationStudy initialization.
    n_jobs : int, optional
        Number of parallel jobs
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    list of dict
        Results for each scenario
        
    Examples
    --------
    >>> scenarios = [
    ...     {'n_sim': 50, 'K': 2, 'd': 10, 'T': 150, 'df': 1.5, ...},
    ...     {'n_sim': 50, 'K': 2, 'd': 10, 'T': 150, 'df': 2.25, ...},
    ...     {'n_sim': 50, 'K': 2, 'd': 10, 'T': 150, 'df': 3.0, ...},
    ... ]
    >>> results = run_parallel_scenarios(scenarios, n_jobs=3)
    """
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    if verbose:
        print(f"Running {len(scenarios)} scenarios in parallel")
        print(f"Using {n_jobs} cores")
    
    def run_one_scenario(scenario_params):
        """Run one complete scenario"""
        # Import generators here to avoid pickling issues
        from generators import TGenerator, TruncatedNormalGenerator
        
        # Extract df and create generators
        df = scenario_params.pop('df')
        err_generator = TGenerator(df=df, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        # Create and run study
        study = ParallelSimulationStudy(
            err_generator=err_generator,
            context_generator=context_generator,
            **scenario_params
        )
        
        # Run with parallel replications
        results = study.run_simulation(n_jobs=1, verbose=0)  # Sequential within scenario
        
        # Add scenario info to results
        results['scenario_params'] = {**scenario_params, 'df': df}
        
        return results
    
    # Run scenarios in parallel
    start_time = time.time()
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_one_scenario)(scenario.copy())
        for scenario in scenarios
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\nAll scenarios completed in {elapsed_time:.2f}s")
        print(f"Average time per scenario: {elapsed_time/len(scenarios):.2f}s")
    
    return results


class BatchSimulationRunner:
    """
    Utility class for running batches of simulations with different configurations.
    
    This is useful for systematic parameter sweeps or reproducibility studies.
    
    Examples
    --------
    >>> runner = BatchSimulationRunner()
    >>> runner.add_scenario(n_sim=50, K=2, d=10, T=150, df=1.5, name="heavy_tails")
    >>> runner.add_scenario(n_sim=50, K=2, d=10, T=150, df=2.25, name="medium_tails")
    >>> runner.add_scenario(n_sim=50, K=2, d=10, T=150, df=5.0, name="light_tails")
    >>> results = runner.run_all(n_jobs=8)
    """
    
    def __init__(self):
        self.scenarios = []
        
    def add_scenario(
        self,
        name: str,
        n_sim: int,
        K: int,
        d: int,
        T: int,
        df: float,
        q: int = 2,
        h: float = 0.5,
        tau: float = 0.5,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Add a scenario to the batch.
        
        Parameters
        ----------
        name : str
            Descriptive name for this scenario
        n_sim, K, d, T, df : various
            Simulation parameters
        **kwargs
            Additional parameters
        """
        scenario = {
            'name': name,
            'n_sim': n_sim,
            'K': K,
            'd': d,
            'T': T,
            'df': df,
            'q': q,
            'h': h,
            'tau': tau,
            'random_seed': random_seed,
            **kwargs
        }
        self.scenarios.append(scenario)
        
    def run_all(
        self,
        n_jobs: int = -1,
        parallel_scenarios: bool = True,
        save_results: bool = True,
        results_dir: str = 'results'
    ) -> Dict[str, Dict]:
        """
        Run all scenarios.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs
        parallel_scenarios : bool
            If True, run scenarios in parallel
            If False, run scenarios sequentially but parallelize within each
        save_results : bool
            Whether to save results to disk
        results_dir : str
            Directory for saving results
            
        Returns
        -------
        dict
            Dictionary mapping scenario names to their results
        """
        print(f"\n{'='*80}")
        print(f"BATCH SIMULATION RUNNER")
        print(f"{'='*80}")
        print(f"Number of scenarios: {len(self.scenarios)}")
        print(f"Parallel mode: {'scenarios' if parallel_scenarios else 'replications'}")
        print(f"Number of jobs: {n_jobs if n_jobs != -1 else mp.cpu_count()}")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        if parallel_scenarios and len(self.scenarios) > 1:
            # Run scenarios in parallel
            scenario_params = [
                {k: v for k, v in s.items() if k != 'name'}
                for s in self.scenarios
            ]
            
            results_list = run_parallel_scenarios(
                scenario_params,
                n_jobs=min(n_jobs, len(self.scenarios)),
                verbose=1
            )
            
            for scenario, results in zip(self.scenarios, results_list):
                all_results[scenario['name']] = results
                
        else:
            # Run scenarios sequentially, but parallelize replications within each
            from generators import TGenerator, TruncatedNormalGenerator
            
            for i, scenario in enumerate(self.scenarios, 1):
                print(f"\nScenario {i}/{len(self.scenarios)}: {scenario['name']}")
                print("-" * 80)
                
                # Create generators
                err_generator = TGenerator(df=scenario['df'], scale=0.7)
                context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
                
                # Create study
                study = ParallelSimulationStudy(
                    n_sim=scenario['n_sim'],
                    K=scenario['K'],
                    d=scenario['d'],
                    T=scenario['T'],
                    q=scenario['q'],
                    h=scenario['h'],
                    tau=scenario['tau'],
                    err_generator=err_generator,
                    context_generator=context_generator,
                    random_seed=scenario['random_seed']
                )
                
                # Run with parallelization
                results = study.run_simulation(n_jobs=n_jobs, verbose=1)
                all_results[scenario['name']] = results
                
                # Save if requested
                if save_results:
                    import os
                    os.makedirs(results_dir, exist_ok=True)
                    filepath = os.path.join(
                        results_dir,
                        f"simulation_{scenario['name']}_df{scenario['df']}.pkl"
                    )
                    study.save_results(filepath)
        
        print(f"\n{'='*80}")
        print(f"BATCH COMPLETE")
        print(f"{'='*80}\n")
        
        return all_results
    
    def compare_results(self, results: Dict[str, Dict]):
        """
        Print comparison of results across scenarios.
        
        Parameters
        ----------
        results : dict
            Results dictionary from run_all()
        """
        import pandas as pd
        
        comparison_data = []
        
        for name, result in results.items():
            rab_regret = result['cumulated_regret_RiskAware'][:, -1]
            ols_regret = result['cumulated_regret_OLS'][:, -1]
            
            comparison_data.append({
                'Scenario': name,
                'RAB Mean Regret': np.mean(rab_regret),
                'RAB Std Regret': np.std(rab_regret),
                'OLS Mean Regret': np.mean(ols_regret),
                'OLS Std Regret': np.std(ols_regret),
                'RAB Better': np.mean(rab_regret < ols_regret) * 100
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nScenario Comparison:")
        print(df.to_string(index=False))
        
        return df


# Example usage and testing
if __name__ == "__main__":
    print("Parallel Simulation Implementation - Examples\n")
    
    # Example 1: Basic parallel simulation
    print("Example 1: Basic parallel simulation")
    print("-" * 80)
    
    from generators import TGenerator, TruncatedNormalGenerator
    
    study = ParallelSimulationStudy(
        n_sim=20,
        K=2,
        d=10,
        T=100,
        q=2,
        h=0.5,
        tau=0.5,
        err_generator=TGenerator(df=2.25, scale=0.7),
        context_generator=TruncatedNormalGenerator(mean=0.0, std=1.0),
        random_seed=42
    )
    
    print("\nRunning sequentially...")
    start = time.time()
    study.run_simulation_sequential()
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.2f}s")
    
    print("\nRunning in parallel (4 cores)...")
    start = time.time()
    study.run_simulation(n_jobs=4)
    par_time = time.time() - start
    print(f"Parallel time: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}×")
    
    # Example 2: Batch runner
    print("\n\nExample 2: Batch simulation runner")
    print("-" * 80)
    
    runner = BatchSimulationRunner()
    runner.add_scenario("df_1.5", n_sim=10, K=2, d=10, T=50, df=1.5)
    runner.add_scenario("df_2.25", n_sim=10, K=2, d=10, T=50, df=2.25)
    runner.add_scenario("df_5.0", n_sim=10, K=2, d=10, T=50, df=5.0)
    
    results = runner.run_all(n_jobs=4, parallel_scenarios=True)
    runner.compare_results(results)