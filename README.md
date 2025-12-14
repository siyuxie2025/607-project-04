# Final Project: Multi-Algorithm Quantile Bandit Comparison

---

## Overview

This project extends Projects 2-3 by comparing **four different bandit algorithms** (Forced Sampling, LinUCB, Epsilon-Greedy, Thompson Sampling) with **quantile regression updates** under various beta/alpha generation strategies. The goal is to understand how different algorithms perform with different coefficient distributions and identify winners for heavy-tailed environments.

### Key Features

- **4 Algorithm Implementations**: Forced Sampling (baseline), LinUCB, Epsilon-Greedy, Thompson Sampling
- **All Use Quantile Regression**: Robust to heavy-tailed errors
- **Flexible Generation Framework**: Test different beta/alpha distributions
- **Comprehensive Analysis**: Tracks both regret AND beta estimation error
- **6+ Scenarios**: Different coefficient generation strategies
- **5 Experimental Sweeps**: Systematic parameter studies

---

## Quick Start

### Installation
```bash
# Install dependencies
make install

# Verify setup
make check-setup
```

### Run Quick Test 
```bash
make quick-test
```

This runs a minimal simulation (5 replications, 50 rounds) and generates a sample figure.

### Run All Scenarios 
```bash
make workflow-scenarios
```

This runs all 6 scenarios and generates comprehensive analysis.

### View Results
```bash
# List generated figures
make list-figures

# Show summary statistics
make show-results

# Open specific figure
open results/project4/scenario_default.pdf
```

---

## Project Structure

```
607project4/
├── README.md                      # This documentation file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── report-Xie                     # Project report
│
├── src/                           # Core modules from Projects 2-3
│   ├── __init__.py
│   ├── generators.py              # Data generators (Normal, T, Uniform, etc.)
│   ├── methods.py                 # Base bandit algorithms (RiskAware, OLS)
│   ├── simulation.py              # SimulationStudy base class
│   ├── numerical_stability.py     # Numerical stability utilities
│   ├── main.py                    # Legacy main from Projects 2-3
│   ├── analyze_results.py
│   ├── create_performance_plots.py
│   ├── examples_flexible_generation.py
│   ├── parallel_simulation.py
│   ├── profile_simulation.py
│   └── tests.py
│
├── project4files/                 # Project 4 implementation
│   ├── Makefile                   # Build automation & workflows
│   ├── project4_main.py           # Main execution (scenarios)
│   ├── project4_simulation.py     # Project4Simulation class (4 algorithms)
│   ├── project4_methods.py        # Algorithm implementations
│   ├── project4_experiments.py    # Experimental sweeps (df, tau, d, K)
│   ├── project4_analysis.py       # Analysis and visualization
│   ├── project4_demo.py           # Interactive demonstration script
│   ├── generate_report_results.py # Report results extraction script
│   ├── generators.py              # Data generators (local copy)
│   ├── numerical_stability.py     # Stability utilities (local copy)
│   ├── presentation_plots.py      # Presentation figure generation
│   ├── report_summary.txt         # Auto-generated summary
│   │
│   └── results/                   # Output directory
│       ├── project4/              # Main project 4 results
│       │   ├── scenario_*.pkl     # Scenario raw data (6 scenarios)
│       │   ├── scenario_*.pdf     # Scenario comparison plots
│       │   ├── data/              # Experimental data files
│       │   ├── figures/           # Per-scenario/experiment figures
│       │   ├── experiments/       # Experimental sweep results
│       │   │   ├── df_sweep_*.pkl
│       │   │   ├── tau_sweep_*.pkl
│       │   │   ├── dim_sweep_*.pkl
│       │   │   ├── arms_sweep_*.pkl
│       │   │   └── *_summary.csv
│       │   └── analysis/          # Cross-scenario analysis
│       │       ├── summary_*.csv
│       │       └── *.pdf
│       ├── project4_test/         # Quick test results
│       ├── figures/               # Legacy figures
│       └── *.pkl, *.json          # Legacy result files
│
└── venv/                          # Python virtual environment
```

### Running the Project

All commands should be run from the `project4files/` directory:

```bash
cd project4files
source ../venv/bin/activate  # Activate virtual environment
make quick-test              # Run quick test
```

---

## Interactive Demo

For a hands-on introduction to the project, run the interactive demonstration:

```bash
cd project4files
python project4_demo.py              # Interactive menu
```

The demo provides several options:

### 1. Quick Demo (5 sims, 2 algorithms, ~10 seconds)
```bash
python project4_demo.py --quick
```
Compares Forced Sampling vs Thompson Sampling with minimal configuration.

### 2. All Algorithms Demo (10 sims, 4 algorithms, ~30 seconds)
```bash
python project4_demo.py --all
```
Compares all 4 algorithms and shows a winner analysis.

### 3. Scenario Comparison Demo (~45 seconds)
```bash
python project4_demo.py --scenarios
```
Shows Thompson Sampling performance across Uniform, Gaussian, and Sparse beta distributions.

### 4. Custom Demo
Select option 4 in interactive mode to specify your own:
- Number of simulations
- Time horizon
- Number of arms
- Context dimension
- Beta distribution type

**Demo Output:**
- Summary statistics table
- Comparison plots (saved as PDF)
- Key observations and insights

---

## Algorithms Implemented

### 1. Forced Sampling (Baseline)
- **Source:** Projects 2-3, Bastani & Bayati (2020)
- **Update:** Quantile regression
- **Exploration:** Forced sampling every $2^n$ rounds
- **Pros:** Theoretically grounded, proven regret bounds
- **Cons:** Slow convergence, inefficient exploration

### 2. LinUCB (Quantile Version)
- **Source:** Li et al. (2010), adapted for quantile regression
- **Update:** Quantile regression with UCB-style confidence bounds
- **Exploration:** Upper confidence bound on quantile estimates
- **Pros:** Principled exploration-exploitation trade-off
- **Cons:** Computationally expensive, requires matrix operations

### 3. Epsilon-Greedy (Quantile Version)
- **Source:** Classic ε-greedy adapted for quantile regression
- **Update:** Quantile regression
- **Exploration:** ε-probability of random action, (1-ε) greedy
- **Pros:** Simple, fast, easy to implement
- **Cons:** Suboptimal exploration, constant ε may be wasteful

### 4. Thompson Sampling (Quantile Version)
- **Source:** Agrawal & Goyal (2013), adapted for quantile regression
- **Update:** Quantile regression with Bayesian posterior sampling
- **Exploration:** Probability matching via posterior sampling
- **Pros:** Theoretically optimal, efficient exploration
- **Cons:** Requires distributional assumptions

--- 


## Usage Examples

### Basic Usage
```python
# Run single scenario
make scenario-gaussian

# Analyze results
make analyze-gaussian

# View figures
open results/project4/analysis/regret_gaussian.pdf
```

### Custom Parameters
```python
# Custom scenario with specific parameters
make custom-scenario BETA=gaussian N_SIM=100 T=500

# Quick test with different settings
make custom-quick BETA=sparse N_SIM=10 T=100
```

### Complete Workflows
```python
# Quick test workflow (5 minutes)
make workflow-quick

# All scenarios workflow (30 minutes)
make workflow-scenarios

# All experiments workflow (2-3 hours)
make workflow-experiments

# Everything (3-4 hours)
make workflow-full
```

### Comparison Studies
```python
# Compare beta distributions
make compare-beta

# Compare context generators
make compare-contexts

# Compare error distributions
make compare-errors
```

### Generate Publication Figures
```python
# High-quality figures (100 sims, 500 rounds)
make paper-figures

# Presentation figures (50 sims, 200 rounds)
make presentation-figures
```

---

## Output Files

### Scenario Results
- `results/project4/scenario_*.pkl` - Raw simulation data
- `results/project4/scenario_*.pdf` - Quick comparison plots

### Analysis Files
- `results/project4/analysis/summary_*.csv` - Summary statistics
- `results/project4/analysis/regret_*.pdf` - Regret comparison plots
- `results/project4/analysis/beta_error_*.pdf` - Beta error plots
- `results/project4/analysis/performance_*.pdf` - Comprehensive summaries

### Experimental Sweeps
- `results/project4/experiments/*_sweep_*.pkl` - Raw sweep data
- `results/project4/experiments/*_sweep_summary.csv` - Sweep summaries
- `results/project4/analysis/*_sweep.pdf` - Sweep visualization

---

## Key Results (Preview)

Based on preliminary runs with default settings (K=2, d=10, T=200, df=2.25):

### Algorithm Rankings

**By Final Regret (Lower is Better):**
1. **Thompson Sampling** - Best overall performance
2. LinUCB - Close second, more stable
3. Epsilon-Greedy - Simple but effective
4. Forced Sampling - Baseline, slowest convergence

**By Beta Estimation (Lower is Better):**
1. **Thompson Sampling** - Best parameter learning
2. LinUCB - Good estimation quality
3. Epsilon-Greedy - Moderate estimation
4. Forced Sampling - Adequate but slow

**By Computation Time (Lower is Better):**
1. **Epsilon-Greedy** - Fastest (~1.2s)
2. Thompson Sampling - Fast (~1.5s)
3. Forced Sampling - Moderate (~2.0s)
4. LinUCB - Slowest (~2.8s)

### Key Findings

1. **Thompson Sampling is the practical winner** - Best regret, best estimation, fast
2. **LinUCB provides stability** - More consistent across scenarios
3. **Epsilon-Greedy is surprisingly competitive** - Simple but effective
4. **Forced Sampling serves as baseline** - Theoretical guarantees but slower

---

## Dependencies

### Required Packages
```
numpy >= 2.3.4
scipy >= 1.16.2
pandas >= 2.3.3
matplotlib >= 3.10.7
scikit-learn >= 1.7.2
quantes >= 2.0.8
tqdm >= 4.67.1
joblib >= 1.5.2
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Reproducibility

All experiments use **fixed random seeds** (default: 42) for reproducibility.
```bash
# Reproduce all results from scratch
make reproduce-all
```

This will:
1. Clean all previous results
2. Run all scenarios with fixed seed
3. Run all experiments with fixed seed
4. Generate all figures
5. Create comprehensive report


---

## Testing

### Quick Test
```bash
make quick-test
```
Runs minimal simulation (5 sims, 50 rounds) to verify everything works.

### Development Test
```bash
make dev-test
```
Even faster test for development (3 sims, 50 rounds).

### Full Verification
```bash
make verify
```
Comprehensive verification of all components.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'quantes'"
**Solution:**
```bash
pip install quantes
```

### Issue: "No results found"
**Solution:** Run experiments first:
```bash
make scenarios
```

### Issue: Figures look different than expected
**Solution:** Ensure you're using the same random seed:
```bash
make reproduce-all
```

### Issue: Out of memory
**Solution:** Reduce N_SIM or T:
```bash
make custom-scenario BETA=default N_SIM=20 T=100
```

### Issue: Takes too long
**Solution:** Use quick mode:
```bash
make quick-test
# or
make scenario-default --quick
```

---

## Advanced Usage

### Custom Algorithm Comparison
```python
from project4_simulation import Project4Simulation
from generators import NormalGenerator, TGenerator

study = Project4Simulation(
    n_sim=50,
    K=3,
    d=15,
    T=300,
    tau=0.5,
    algorithms=['LinUCB', 'ThompsonSampling'],  # Only these two
    beta_generator=NormalGenerator(mean=0, std=1),
    err_generator=TGenerator(df=2.25, scale=0.7),
    random_seed=42
)

results = study.run_simulation()
study.plot_comparison(save_path='my_custom_plot.pdf')
```

### Heterogeneous Arm Setup
```python
# Different beta distribution per arm
beta_gens = [
    NormalGenerator(mean=0, std=1),      # Arm 0
    UniformGenerator(low=0.5, high=1.5), # Arm 1
    TGenerator(df=5, scale=1)            # Arm 2
]

study = Project4Simulation(
    K=3,
    beta_generator=beta_gens,
    ...
)
```

---


## References

1. Bastani, H., & Bayati, M. (2020). "Online Decision Making with High-Dimensional Covariates." *Operations Research*
2. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW*
3. Agrawal, S., & Goyal, N. (2013). "Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML*
4. Koenker, R. (2005). "Quantile Regression." *Cambridge University Press*

--- 

## Appendix: Complete Command Reference

### Setup Commands
```bash
make install          # Install dependencies
make check-setup      # Verify installation
make info             # Show project information
```

### Testing Commands
```bash
make quick-test       # Quick test (2 min)
make dev-test         # Development test
make verify           # Full verification
```

### Scenario Commands
```bash
make scenario-default       # Uniform beta
make scenario-gaussian      # Gaussian beta
make scenario-heterogeneous # Heterogeneous arms
make scenario-sparse        # Sparse coefficients
make scenario-heavy         # Heavy-tailed beta
make scenario-multi         # Multiple arms
make scenarios              # All scenarios
```

### Experiment Commands
```bash
make exp-df-sweep          # Vary df
make exp-tau-sweep         # Vary τ
make exp-dim-sweep         # Vary d
make exp-arms-sweep        # Vary K
make exp-beta-comparison   # Compare beta strategies
make experiments           # All experiments
```

### Analysis Commands
```bash
make analyze-all          # Analyze everything
make analyze-default      # Analyze default scenario
make all-figures          # Generate all figures
make summary-report       # Generate summary report
make show-results         # Show summary stats
make list-figures         # List generated figures
```

### Workflow Commands
```bash
make workflow-quick        # Quick workflow
make workflow-scenarios    # Scenarios workflow
make workflow-experiments  # Experiments workflow
make workflow-full         # Full workflow
```

### Comparison Commands
```bash
make compare-beta         # Compare beta distributions
make compare-contexts     # Compare context generators
make compare-errors       # Compare error distributions
```

### Publication Commands
```bash
make paper-figures        # High-quality figures
make presentation-figures # Presentation figures
make reproduce-all        # Reproduce everything
```

### Utility Commands
```bash
make tree                 # Show directory structure
make clean-results        # Clean result files
make clean-all            # Clean everything
make backup               # Backup results
```

---

**Last Updated:** December 2025