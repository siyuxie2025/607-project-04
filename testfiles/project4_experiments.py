"""
Project 4 Experimental Design Matrix
=====================================

This script runs systematic experiments across the design matrix:
- Multiple error distributions (varying tail heaviness)
- Low-dimensional and high-dimensional settings
- All four algorithms

Generates comprehensive results for final report.
"""

import numpy as np
import json
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.insert(0, 'src')
from generators import TGenerator, TruncatedNormalGenerator
from project4_main import Project4Simulation, create_comparison_plots


# =====================================================
# EXPERIMENTAL DESIGN MATRIX
# =====================================================

LOW_DIM_EXPERIMENTS = {
    'baseline': {
        'n_sim': 50,
        'K': 2,
        'd': 10,
        'T': 500,
        'tau': 0.5,
        'df_values': [2.25],  # Baseline
        'high_dim': False,
        'algorithms': ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    },
    
    'tail_heaviness': {
        'n_sim': 50,
        'K': 2,
        'd': 10,
        'T': 500,
        'tau': 0.5,
        'df_values': [1.5, 2.25, 3.0, 5.0, 10.0],  # Vary tail heaviness
        'high_dim': False,
        'algorithms': ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    },
    
    'multiple_arms': {
        'n_sim': 50,
        'K': 5,  # More arms
        'd': 10,
        'T': 1000,  # Longer horizon
        'tau': 0.5,
        'df_values': [2.25, 3.0],
        'high_dim': False,
        'algorithms': ['ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling']
    },
    
    'different_quantiles': {
        'n_sim': 50,
        'K': 2,
        'd': 10,
        'T': 500,
        'tau_values': [0.25, 0.5, 0.75],  # Vary quantile level
        'df': 2.25,
        'high_dim': False,
        'algorithms': ['ForcedSampling', 'EpsilonGreedy', 'ThompsonSampling']  # Skip LinUCB (slow)
    }
}

HIGH_DIM_EXPERIMENTS = {
    'sparse_baseline': {
        'n_sim': 30,  # Fewer replications (slow)
        'K': 2,
        'd': 100,  # High-dimensional
        'T': 500,
        'tau': 0.5,
        'df_values': [2.25],
        'high_dim': True,
        'algorithms': ['ForcedSampling', 'EpsilonGreedy', 'ThompsonSampling']  # Skip LinUCB (very slow)
    },
    
    'sparse_comparison': {
        'n_sim': 30,
        'K': 2,
        'd': 100,
        'T': 500,
        'tau': 0.5,
        'df_values': [2.25, 3.0],
        'high_dim': True,
        'algorithms': ['ForcedSampling', 'EpsilonGreedy', 'ThompsonSampling']
    },
    
    'very_high_dim': {
        'n_sim': 20,  # Even fewer replications
        'K': 2,
        'd': 200,  # Very high-dimensional
        'T': 500,
        'tau': 0.5,
        'df_values': [2.25],
        'high_dim': True,
        'algorithms': ['ForcedSampling', 'ThompsonSampling']  # Only fastest algorithms
    }
}


# =====================================================
# EXPERIMENT RUNNER
# =====================================================

class ExperimentRunner:
    """
    Systematic experiment runner for Project 4.
    
    Runs all experiments in design matrix and saves results.
    """
    
    def __init__(self, output_dir: str = 'results/project4'):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save all results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        
        # Experiment log
        self.log = []
    
    def run_experiment(
        self,
        experiment_name: str,
        config: dict,
        random_seed: int = 1010
    ):
        """
        Run a single experiment configuration.
        
        Parameters
        ----------
        experiment_name : str
            Name for this experiment
        config : dict
            Experiment configuration
        random_seed : int
            Random seed
        """
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*80}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Handle different experimental variations
        if 'df_values' in config:
            # Vary error distribution
            for df in config['df_values']:
                self._run_single_config(
                    experiment_name, config, df=df,
                    tau=config.get('tau', 0.5),
                    random_seed=random_seed
                )
        elif 'tau_values' in config:
            # Vary quantile level
            df = config['df']
            for tau in config['tau_values']:
                self._run_single_config(
                    experiment_name, config, df=df,
                    tau=tau, random_seed=random_seed
                )
        else:
            # Single configuration
            self._run_single_config(
                experiment_name, config,
                df=config.get('df', 2.25),
                tau=config.get('tau', 0.5),
                random_seed=random_seed
            )
        
        elapsed = time.time() - start_time
        
        # Log experiment
        log_entry = {
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed,
            'config': config
        }
        self.log.append(log_entry)
        
        print(f"\n✓ Experiment '{experiment_name}' complete ({elapsed/60:.1f} min)")
    
    def _run_single_config(
        self,
        experiment_name: str,
        config: dict,
        df: float,
        tau: float,
        random_seed: int
    ):
        """Run single configuration with specific df and tau."""
        # Create generators
        err_generator = TGenerator(df=df, scale=0.7)
        context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)
        
        # Create simulation
        study = Project4Simulation(
            n_sim=config['n_sim'],
            K=config['K'],
            d=config['d'],
            T=config['T'],
            tau=tau,
            err_generator=err_generator,
            context_generator=context_generator,
            algorithms=config['algorithms'],
            high_dim=config.get('high_dim', False),
            random_seed=random_seed
        )
        
        # Run simulation
        results = study.run_simulation(verbose=True)
        
        # Save results
        filename = f"{experiment_name}_df{df}_tau{tau}_d{config['d']}_T{config['T']}"
        filepath = self.output_dir / 'data' / f"{filename}.pkl"
        study.save_results(str(filepath))
        
        # Create plots
        figures_dir = self.output_dir / 'figures' / experiment_name
        figures_dir.mkdir(exist_ok=True)
        
        create_comparison_plots(
            results,
            save_dir=str(figures_dir)
        )
    
    def run_all_low_dim(self):
        """Run all low-dimensional experiments."""
        print("\n" + "="*80)
        print("RUNNING ALL LOW-DIMENSIONAL EXPERIMENTS")
        print("="*80 + "\n")
        
        for exp_name, config in LOW_DIM_EXPERIMENTS.items():
            self.run_experiment(f"lowdim_{exp_name}", config)
    
    def run_all_high_dim(self):
        """Run all high-dimensional experiments."""
        print("\n" + "="*80)
        print("RUNNING ALL HIGH-DIMENSIONAL EXPERIMENTS")
        print("="*80 + "\n")
        
        for exp_name, config in HIGH_DIM_EXPERIMENTS.items():
            self.run_experiment(f"highdim_{exp_name}", config)
    
    def run_all(self):
        """Run all experiments (low-dim + high-dim)."""
        total_start = time.time()
        
        print("\n" + "="*80)
        print("PROJECT 4: FULL EXPERIMENTAL SUITE")
        print("="*80)
        print("This will take several hours!")
        print("="*80 + "\n")
        
        # Run low-dimensional experiments
        self.run_all_low_dim()
        
        # Run high-dimensional experiments
        self.run_all_high_dim()
        
        total_elapsed = time.time() - total_start
        
        # Save experiment log
        log_path = self.output_dir / 'experiment_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        print("\n" + "="*80)
        print("✓ ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"Total time: {total_elapsed/3600:.2f} hours")
        print(f"Experiments run: {len(self.log)}")
        print(f"Log saved to: {log_path}")
        print("="*80 + "\n")
    
    def generate_report(self):
        """Generate comprehensive report from all experiments."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80 + "\n")
        
        # TODO: Implement comprehensive analysis
        # This would:
        # 1. Load all results
        # 2. Create comparison tables
        # 3. Generate summary figures
        # 4. Write LaTeX report
        
        print("Report generation not yet implemented")
        print("Use project4_analysis.py for detailed analysis")


# =====================================================
# QUICK EXPERIMENTS (for testing)
# =====================================================

QUICK_EXPERIMENTS = {
    'quick_test': {
        'n_sim': 5,  # Very few replications
        'K': 2,
        'd': 10,
        'T': 100,  # Short horizon
        'tau': 0.5,
        'df_values': [2.25],
        'high_dim': False,
        'algorithms': ['ForcedSampling', 'EpsilonGreedy']  # Skip slow algorithms
    }
}


def run_quick_test():
    """Run quick test to verify everything works."""
    print("\n" + "="*80)
    print("QUICK TEST - Verify Project 4 Setup")
    print("="*80 + "\n")
    
    runner = ExperimentRunner(output_dir='results/project4_test')
    
    for exp_name, config in QUICK_EXPERIMENTS.items():
        runner.run_experiment(exp_name, config)
    
    print("\n✓ Quick test complete!")
    print("If this ran successfully, you're ready for full experiments\n")


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main():
    """Main entry point for experimental design."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Project 4 experimental design matrix'
    )
    parser.add_argument(
        '--mode',
        choices=['quick', 'lowdim', 'highdim', 'all'],
        default='quick',
        help='Which experiments to run'
    )
    parser.add_argument(
        '--output_dir',
        default='results/project4',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    else:
        runner = ExperimentRunner(output_dir=args.output_dir)
        
        if args.mode == 'lowdim':
            runner.run_all_low_dim()
        elif args.mode == 'highdim':
            runner.run_all_high_dim()
        elif args.mode == 'all':
            runner.run_all()
    
    print("\n✓ Experimental design execution complete")


if __name__ == "__main__":
    main()