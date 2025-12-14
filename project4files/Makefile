# Makefile for Project 4: Multi-Algorithm Bandit Comparison
# =============================================================
# Quick figure generation for different context/error scenarios

.PHONY: help install test clean info

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
N_SIM := 50
T := 200
QUICK_N_SIM := 5
QUICK_T := 50

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m

# ======================================================================
# HELP
# ======================================================================

help:  ## Show this help message
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Project 4: Multi-Algorithm Bandit Comparison$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  $(YELLOW)make quick-test$(NC)      # Test everything works (2 min)"
	@echo "  $(YELLOW)make scenarios$(NC)       # Run all scenarios (30 min)"
	@echo "  $(YELLOW)make experiments$(NC)     # Run all experiments (2 hours)"
	@echo "  $(YELLOW)make all-figures$(NC)     # Generate all figures"
	@echo ""
	@echo "$(GREEN)Individual Scenarios:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## Scenario:' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Experimental Sweeps:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## Experiment:' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Analysis & Figures:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## Analysis:' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## Util:' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ======================================================================
# SETUP
# ======================================================================

install:  ## Util: Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install -r ../requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

check-setup:  ## Util: Verify setup
	@echo "$(GREEN)Checking setup...$(NC)"
	@$(PYTHON) -c "import numpy, scipy, pandas, matplotlib, quantes; print('✓ OK')"

verify:  ## Util: Full verification of all components
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Full Verification$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) quick-test
	@echo ""
	@echo "$(GREEN)✓ All components verified$(NC)"

# ======================================================================
# QUICK TESTING
# ======================================================================

quick-test:  ## Util: Quick test (5 sims, 50 rounds)
	@echo "$(BLUE)=== Quick Test ===$(NC)"
	@$(PYTHON) project4_main.py --scenario quick
	@$(PYTHON) project4_analysis.py --scenario default
	@echo "$(GREEN)✓ Quick test complete$(NC)"
	@echo "$(YELLOW)Check: results/project4/scenario_default.pdf$(NC)"

# ======================================================================
# SCENARIOS (Different Beta/Context Generation)
# ======================================================================

scenario-default:  ## Scenario: Uniform beta (baseline)
	@echo "$(GREEN)Running: Default (Uniform Beta)$(NC)"
	@$(PYTHON) project4_main.py --scenario default \
		--n_sim $(N_SIM) --T $(T)

scenario-gaussian:  ## Scenario: Gaussian beta (zero-mean)
	@echo "$(GREEN)Running: Gaussian Beta$(NC)"
	@$(PYTHON) project4_main.py --scenario gaussian \
		--n_sim $(N_SIM) --T $(T)

scenario-heterogeneous:  ## Scenario: Different beta per arm
	@echo "$(GREEN)Running: Heterogeneous Arms$(NC)"
	@$(PYTHON) project4_main.py --scenario heterogeneous \
		--n_sim $(N_SIM) --T $(T)

scenario-sparse:  ## Scenario: Sparse/weak signals
	@echo "$(GREEN)Running: Sparse Coefficients$(NC)"
	@$(PYTHON) project4_main.py --scenario sparse \
		--n_sim $(N_SIM) --T $(T)

scenario-heavy:  ## Scenario: Heavy-tailed beta
	@echo "$(GREEN)Running: Heavy-Tailed Beta$(NC)"
	@$(PYTHON) project4_main.py --scenario heavy \
		--n_sim $(N_SIM) --T $(T)

scenario-multi:  ## Scenario: Multiple arms (K=5)
	@echo "$(GREEN)Running: Multi-Arm$(NC)"
	@$(PYTHON) project4_main.py --scenario multi \
		--n_sim 30 --T 150

scenarios:  ## Run all scenarios
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Running All Scenarios$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(YELLOW)Estimated time: 30-40 minutes$(NC)"
	@echo ""
	@$(MAKE) scenario-default
	@$(MAKE) scenario-gaussian
	@$(MAKE) scenario-heterogeneous
	@$(MAKE) scenario-sparse
	@$(MAKE) scenario-heavy
	@$(MAKE) scenario-multi
	@echo ""
	@echo "$(GREEN)✓ All scenarios complete$(NC)"
	@echo "$(YELLOW)Results in: results/project4/scenario_*.pkl$(NC)"

# ======================================================================
# EXPERIMENTAL SWEEPS
# ======================================================================

exp-df-sweep:  ## Experiment: Vary tail heaviness (df)
	@echo "$(GREEN)Running: DF Sweep$(NC)"
	@$(PYTHON) project4_experiments.py --experiment df_sweep \
		--n_sim $(N_SIM) --T $(T)

exp-tau-sweep:  ## Experiment: Vary quantile level (τ)
	@echo "$(GREEN)Running: Tau Sweep$(NC)"
	@$(PYTHON) project4_experiments.py --experiment tau_sweep \
		--n_sim $(N_SIM) --T $(T)

exp-dim-sweep:  ## Experiment: Vary dimension (d)
	@echo "$(GREEN)Running: Dimension Sweep$(NC)"
	@$(PYTHON) project4_experiments.py --experiment dimension_sweep \
		--n_sim 20 --T 100

exp-arms-sweep:  ## Experiment: Vary number of arms (K)
	@echo "$(GREEN)Running: Arms Sweep$(NC)"
	@$(PYTHON) project4_experiments.py --experiment arms_sweep \
		--n_sim 20 --T 100

exp-beta-comparison:  ## Experiment: Compare beta strategies
	@echo "$(GREEN)Running: Beta Strategy Comparison$(NC)"
	@$(PYTHON) project4_experiments.py --experiment beta_comparison \
		--n_sim $(N_SIM) --T $(T)

experiments:  ## Run all experimental sweeps
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Running All Experiments$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(RED)WARNING: This will take 2-3 hours!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	@$(MAKE) exp-df-sweep
	@$(MAKE) exp-tau-sweep
	@$(MAKE) exp-dim-sweep
	@$(MAKE) exp-arms-sweep
	@$(MAKE) exp-beta-comparison
	@echo ""
	@echo "$(GREEN)✓ All experiments complete$(NC)"

# ======================================================================
# ANALYSIS & FIGURES
# ======================================================================

analyze-scenario:  ## Analysis: Analyze one scenario (use SCENARIO=name)
	@echo "$(GREEN)Analyzing: $(SCENARIO)$(NC)"
	@$(PYTHON) project4_analysis.py --scenario $(SCENARIO)
	@echo "$(GREEN)✓ Analysis complete$(NC)"
	@echo "$(YELLOW)Figures: results/project4/analysis/*_$(SCENARIO).pdf$(NC)"

analyze-experiment:  ## Analysis: Analyze one experiment (use EXP=name)
	@echo "$(GREEN)Analyzing: $(EXP)$(NC)"
	@$(PYTHON) project4_analysis.py --experiment $(EXP)
	@echo "$(GREEN)✓ Analysis complete$(NC)"

analyze-all:  ## Analysis: Analyze all results
	@echo "$(GREEN)Analyzing all results...$(NC)"
	@$(PYTHON) project4_analysis.py
	@echo "$(GREEN)✓ Analysis complete$(NC)"

# Individual scenario analysis targets
analyze-default:  ## Analysis: Analyze default scenario
	@$(MAKE) analyze-scenario SCENARIO=default

analyze-gaussian:  ## Analysis: Analyze gaussian scenario
	@$(MAKE) analyze-scenario SCENARIO=gaussian

analyze-heterogeneous:  ## Analysis: Analyze heterogeneous scenario
	@$(MAKE) analyze-scenario SCENARIO=heterogeneous

analyze-sparse:  ## Analysis: Analyze sparse scenario
	@$(MAKE) analyze-scenario SCENARIO=sparse

analyze-heavy:  ## Analysis: Analyze heavy-tailed scenario
	@$(MAKE) analyze-scenario SCENARIO=heavy

analyze-multi:  ## Analysis: Analyze multi-arm scenario
	@$(MAKE) analyze-scenario SCENARIO=multi

# Individual experiment analysis targets
analyze-df:  ## Analysis: Analyze df sweep
	@$(MAKE) analyze-experiment EXP=df_sweep

analyze-tau:  ## Analysis: Analyze tau sweep
	@$(MAKE) analyze-experiment EXP=tau_sweep

analyze-dim:  ## Analysis: Analyze dimension sweep
	@$(MAKE) analyze-experiment EXP=dim_sweep

analyze-arms:  ## Analysis: Analyze arms sweep
	@$(MAKE) analyze-experiment EXP=arms_sweep

analyze-beta:  ## Analysis: Analyze beta comparison
	@$(MAKE) analyze-experiment EXP=beta_strategy

all-figures:  ## Analysis: Generate all figures
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Generating All Figures$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(MAKE) analyze-all
	@echo ""
	@echo "$(GREEN)✓ All figures generated$(NC)"
	@echo ""
	@echo "$(BLUE)Available figures:$(NC)"
	@ls -lh results/project4/analysis/*.pdf 2>/dev/null || echo "No figures yet"

# ======================================================================
# COMPLETE WORKFLOWS
# ======================================================================

workflow-quick:  ## Complete quick workflow (test)
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Quick Workflow (Test)$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) quick-test
	@echo "$(GREEN)✓ Quick workflow complete$(NC)"

workflow-scenarios:  ## Complete workflow: all scenarios
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Scenarios Workflow$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) scenarios
	@$(MAKE) analyze-all
	@echo ""
	@echo "$(GREEN)✓ Scenarios workflow complete$(NC)"
	@echo "$(YELLOW)Results in: results/project4/$(NC)"

workflow-experiments:  ## Complete workflow: all experiments
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Experiments Workflow$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(RED)WARNING: 2-3 hours runtime!$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) experiments
	@$(MAKE) analyze-all
	@echo ""
	@echo "$(GREEN)✓ Experiments workflow complete$(NC)"

workflow-full:  ## Complete workflow: everything
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Full Workflow (Scenarios + Experiments)$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(RED)WARNING: 3-4 hours runtime!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 10 seconds to cancel...$(NC)"
	@sleep 10
	@$(MAKE) scenarios
	@$(MAKE) experiments
	@$(MAKE) analyze-all
	@$(MAKE) summary-report
	@echo ""
	@echo "$(GREEN)✓ Full workflow complete$(NC)"

# ======================================================================
# REPORTING
# ======================================================================

summary-report:  ## Generate summary report
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Summary Report$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Scenario Results:$(NC)"
	@for f in results/project4/analysis/summary_*.csv; do \
		if [ -f "$$f" ]; then \
			echo ""; \
			echo "$(YELLOW)$$(basename $$f)$(NC)"; \
			head -5 "$$f" | column -t -s, ; \
		fi \
	done
	@echo ""
	@echo "$(GREEN)Generated Figures:$(NC)"
	@ls -1 results/project4/analysis/*.pdf 2>/dev/null | wc -l | \
		xargs echo "  Total PDF files:"
	@echo ""
	@echo "$(GREEN)Experimental Data:$(NC)"
	@ls -1 results/project4/*.pkl 2>/dev/null | wc -l | \
		xargs echo "  Total result files:"
	@echo ""

show-results:  ## Show summary of results
	@echo "$(GREEN)Summary Statistics:$(NC)"
	@if [ -f results/project4/analysis/summary_all_scenarios.csv ]; then \
		head -20 results/project4/analysis/summary_all_scenarios.csv | column -t -s,; \
	else \
		echo "$(YELLOW)No combined results yet. Run 'make analyze-all'$(NC)"; \
	fi

list-figures:  ## List all generated figures
	@echo "$(GREEN)Generated Figures:$(NC)"
	@ls -lh results/project4/analysis/*.pdf 2>/dev/null || \
		echo "$(YELLOW)No figures yet. Run 'make all-figures'$(NC)"
	@ls -lh results/project4/scenario_*.pdf 2>/dev/null || true

# ======================================================================
# CUSTOM RUNS
# ======================================================================

custom-scenario:  ## Run custom scenario (set BETA=gaussian/uniform/heavy/sparse)
	@echo "$(GREEN)Running custom scenario: BETA=$(BETA)$(NC)"
	@$(PYTHON) project4_main.py --scenario $(BETA) \
		--n_sim $(N_SIM) --T $(T)
	@$(PYTHON) project4_analysis.py --scenario $(BETA)
	@echo "$(GREEN)✓ Custom scenario complete$(NC)"

custom-quick:  ## Quick custom run (use: make custom-quick BETA=gaussian N_SIM=10 T=100)
	@$(PYTHON) project4_main.py --scenario $(BETA) \
		--n_sim $(or $(N_SIM),10) --T $(or $(T),100)

# ======================================================================
# UTILITIES
# ======================================================================

info:  ## Util: Show project information
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Project 4 Information$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Algorithms Implemented:$(NC)"
	@echo "  • Forced Sampling (baseline from Projects 2-3)"
	@echo "  • LinUCB (with quantile regression)"
	@echo "  • Epsilon-Greedy (with quantile regression)"
	@echo "  • Thompson Sampling (with quantile regression)"
	@echo ""
	@echo "$(GREEN)Scenarios Available:$(NC)"
	@echo "  • default         - Uniform beta (baseline)"
	@echo "  • gaussian        - Gaussian beta (zero-mean)"
	@echo "  • heterogeneous   - Different beta per arm"
	@echo "  • sparse          - Weak/sparse signals"
	@echo "  • heavy           - Heavy-tailed beta"
	@echo "  • multi           - Multiple arms (K=5)"
	@echo ""
	@echo "$(GREEN)Experimental Sweeps:$(NC)"
	@echo "  • df_sweep        - Vary tail heaviness"
	@echo "  • tau_sweep       - Vary quantile level"
	@echo "  • dimension_sweep - Vary context dimension"
	@echo "  • arms_sweep      - Vary number of arms"
	@echo "  • beta_comparison - Compare beta strategies"
	@echo ""
	@echo "$(GREEN)Current Configuration:$(NC)"
	@echo "  N_SIM = $(N_SIM)"
	@echo "  T = $(T)"
	@echo ""

tree:  ## Util: Show directory structure
	@echo "$(GREEN)Directory Structure:$(NC)"
	@tree -L 2 results/project4 2>/dev/null || ls -R results/project4 2>/dev/null || \
		echo "$(YELLOW)No results directory yet$(NC)"

clean-results:  ## Util: Clean result files
	@echo "$(YELLOW)Cleaning results...$(NC)"
	@rm -rf results/project4/*.pkl
	@rm -rf results/project4/analysis/*.pdf
	@rm -rf results/project4/analysis/*.csv
	@rm -rf results/project4/experiments/*.pkl
	@rm -rf results/project4/experiments/*.csv
	@echo "$(GREEN)✓ Results cleaned$(NC)"

clean-all:  ## Util: Clean everything (results + cache)
	@$(MAKE) clean-results
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

backup:  ## Util: Backup results
	@echo "$(GREEN)Creating backup...$(NC)"
	@mkdir -p backups
	@tar -czf backups/project4_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		results/project4 \
		project4_*.py
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"
	@ls -lh backups/ | tail -1

# ======================================================================
# DEVELOPMENT
# ======================================================================

dev-test:  ## Development: Minimal test
	@$(PYTHON) project4_main.py --scenario quick

dev-profile:  ## Development: Profile performance
	@$(PYTHON) -m cProfile -o results/project4_profile.prof \
		project4_main.py --scenario default --n_sim 5 --T 50
	@echo "$(GREEN)Profile saved: results/project4_profile.prof$(NC)"
	@echo "$(YELLOW)View: python -m pstats results/project4_profile.prof$(NC)"

# ======================================================================
# PAPER/PRESENTATION FIGURES
# ======================================================================

paper-figures:  ## Generate high-quality figures for paper
	@echo "$(GREEN)Generating paper-quality figures...$(NC)"
	@$(MAKE) scenarios N_SIM=100 T=500
	@$(MAKE) analyze-all
	@echo "$(GREEN)✓ Paper figures ready$(NC)"
	@echo "$(YELLOW)High-res PDFs in: results/project4/analysis/$(NC)"

presentation-figures:  ## Generate figures for presentation
	@echo "$(GREEN)Generating presentation figures...$(NC)"
	@$(MAKE) scenario-default N_SIM=50 T=200
	@$(MAKE) scenario-gaussian N_SIM=50 T=200
	@$(MAKE) scenario-heterogeneous N_SIM=50 T=200
	@$(MAKE) analyze-default
	@$(MAKE) analyze-gaussian
	@$(MAKE) analyze-heterogeneous
	@echo "$(GREEN)✓ Presentation figures ready$(NC)"

# ======================================================================
# SPECIAL COMBINATIONS
# ======================================================================

compare-contexts:  ## Compare different context generators
	@echo "$(GREEN)Comparing context generators...$(NC)"
	@$(MAKE) scenario-default
	@$(MAKE) scenario-gaussian
	@$(MAKE) analyze-default
	@$(MAKE) analyze-gaussian
	@echo "$(GREEN)✓ Context comparison complete$(NC)"

compare-errors:  ## Compare different error distributions
	@echo "$(GREEN)Comparing error distributions...$(NC)"
	@$(MAKE) exp-df-sweep
	@$(MAKE) analyze-df
	@echo "$(GREEN)✓ Error comparison complete$(NC)"

compare-beta:  ## Compare different beta distributions
	@echo "$(GREEN)Comparing beta distributions...$(NC)"
	@$(MAKE) scenario-default
	@$(MAKE) scenario-gaussian
	@$(MAKE) scenario-sparse
	@$(MAKE) scenario-heavy
	@$(MAKE) analyze-all
	@echo "$(GREEN)✓ Beta comparison complete$(NC)"

# ======================================================================
# REPRODUCIBILITY
# ======================================================================

reproduce-all:  ## Reproduce all results (with fixed seed)
	@echo "$(BLUE)================================================$(NC)"
	@echo "$(BLUE) Reproducing All Results (Fixed Seed=42)$(NC)"
	@echo "$(BLUE)================================================$(NC)"
	@$(MAKE) clean-results
	@$(MAKE) workflow-full
	@echo "$(GREEN)✓ Reproducible results generated$(NC)"

# ======================================================================
# End
# ======================================================================

.PHONY: all help install check-setup verify quick-test \
	scenario-default scenario-gaussian scenario-heterogeneous \
	scenario-sparse scenario-heavy scenario-multi scenarios \
	exp-df-sweep exp-tau-sweep exp-dim-sweep exp-arms-sweep \
	exp-beta-comparison experiments \
	analyze-scenario analyze-experiment analyze-all all-figures \
	analyze-default analyze-gaussian analyze-heterogeneous \
	analyze-sparse analyze-heavy analyze-multi \
	analyze-df analyze-tau analyze-dim analyze-arms analyze-beta \
	workflow-quick workflow-scenarios workflow-experiments workflow-full \
	summary-report show-results list-figures \
	custom-scenario custom-quick \
	info tree clean-results clean-all backup \
	dev-test dev-profile \
	paper-figures presentation-figures \
	compare-contexts compare-errors compare-beta \
	reproduce-all