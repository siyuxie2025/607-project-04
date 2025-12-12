# Makefile for Project 4: Multi-Algorithm Quantile Bandit Comparison
# ======================================================================

.PHONY: help install test quick lowdim highdim all analyze clean info

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python
N_SIM := 50
K := 2
D := 10
T := 500
DF := 2.25
TAU := 0.5
SEED := 1010

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

# ======================================================================
# HELP
# ======================================================================

help:  ## Show this help message
	@echo "$(BLUE)================================$(NC)"
	@echo "$(BLUE) Project 4: Makefile Commands$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Setup:/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Testing:/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Experiments:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Experiments:/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Analysis:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Analysis:/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Utilities:/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "Examples:"
	@echo "  make quick              # Quick test (5 sims, 2 algorithms)"
	@echo "  make lowdim             # Run all low-dim experiments"
	@echo "  make highdim            # Run all high-dim experiments"
	@echo "  make all                # Run complete experimental suite"
	@echo "  make analyze            # Analyze results and generate report"
	@echo ""

# ======================================================================
# SETUP
# ======================================================================

install:  ## Setup: Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

check-setup:  ## Setup: Verify installation
	@echo "$(GREEN)Checking setup...$(NC)"
	@$(PYTHON) -c "import numpy, scipy, pandas, matplotlib, quantes; print('✓ Core dependencies OK')"
	@$(PYTHON) -c "from quantes.linear import high_dim; print('✓ High-dimensional quantes available')" || \
		echo "$(YELLOW)⚠ High-dimensional quantes not available$(NC)"
	@echo "$(GREEN)✓ Setup verified$(NC)"

# ======================================================================
# TESTING
# ======================================================================

test:  ## Testing: Test algorithm implementations
	@echo "$(GREEN)Testing algorithm implementations...$(NC)"
	@$(PYTHON) project4_methods.py
	@echo "$(GREEN)✓ Tests passed$(NC)"

quick:  ## Testing: Quick test run (5 sims, 100 rounds)
	@echo "$(GREEN)Running quick test...$(NC)"
	@$(PYTHON) project4_experiments.py --mode quick
	@echo "$(GREEN)✓ Quick test complete$(NC)"

verify:  ## Testing: Verify all components work
	@echo "$(GREEN)Verifying all components...$(NC)"
	@$(MAKE) test
	@$(MAKE) quick
	@echo "$(GREEN)✓ All components verified$(NC)"

# ======================================================================
# EXPERIMENTS
# ======================================================================

baseline:  ## Experiments: Run baseline comparison (default settings)
	@echo "$(GREEN)Running baseline experiment...$(NC)"
	@$(PYTHON) project4_main.py \
		--n_sim $(N_SIM) \
		--K $(K) \
		--d $(D) \
		--T $(T) \
		--df $(DF) \
		--tau $(TAU) \
		--seed $(SEED)
	@echo "$(GREEN)✓ Baseline complete$(NC)"

lowdim:  ## Experiments: Run all low-dimensional experiments
	@echo "$(GREEN)Running low-dimensional experiments...$(NC)"
	@echo "$(YELLOW)This will take ~1-2 hours$(NC)"
	@$(PYTHON) project4_experiments.py --mode lowdim
	@echo "$(GREEN)✓ Low-dimensional experiments complete$(NC)"

highdim:  ## Experiments: Run all high-dimensional experiments
	@echo "$(GREEN)Running high-dimensional experiments...$(NC)"
	@echo "$(YELLOW)This will take ~2-3 hours$(NC)"
	@$(PYTHON) project4_experiments.py --mode highdim
	@echo "$(GREEN)✓ High-dimensional experiments complete$(NC)"

all-experiments:  ## Experiments: Run complete experimental suite
	@echo "$(GREEN)Running complete experimental suite...$(NC)"
	@echo "$(RED)WARNING: This will take several hours!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 5 seconds to cancel...$(NC)"
	@sleep 5
	@$(PYTHON) project4_experiments.py --mode all
	@echo "$(GREEN)✓ All experiments complete$(NC)"

# ======================================================================
# CUSTOM EXPERIMENTS
# ======================================================================

custom:  ## Experiments: Run custom experiment (use make custom N_SIM=... K=... etc)
	@echo "$(GREEN)Running custom experiment...$(NC)"
	@$(PYTHON) project4_main.py \
		--n_sim $(N_SIM) \
		--K $(K) \
		--d $(D) \
		--T $(T) \
		--df $(DF) \
		--tau $(TAU) \
		--seed $(SEED) \
		$(if $(HIGH_DIM),--high_dim)
	@echo "$(GREEN)✓ Custom experiment complete$(NC)"

heavy-tails:  ## Experiments: Compare with very heavy tails (df=1.5)
	@$(MAKE) custom DF=1.5 T=500

multiple-arms:  ## Experiments: Test with more arms (K=5)
	@$(MAKE) custom K=5 T=1000

high-dimensional:  ## Experiments: Test high-dimensional (d=100)
	@$(MAKE) custom D=100 HIGH_DIM=1 T=500

# ======================================================================
# ANALYSIS
# ======================================================================

analyze:  ## Analysis: Analyze all results and generate report
	@echo "$(GREEN)Analyzing results...$(NC)"
	@$(PYTHON) project4_analysis.py
	@echo "$(GREEN)✓ Analysis complete$(NC)"
	@echo ""
	@echo "$(BLUE)Results available in:$(NC)"
	@echo "  - results/project4/summary/"
	@echo "  - results/project4/figures/"
	@echo ""

stats:  ## Analysis: Show summary statistics
	@echo "$(GREEN)Summary Statistics:$(NC)"
	@if [ -f results/project4/summary/summary_statistics.csv ]; then \
		head -20 results/project4/summary/summary_statistics.csv | column -t -s,; \
	else \
		echo "$(YELLOW)No results found. Run experiments first.$(NC)"; \
	fi

figures:  ## Analysis: Regenerate all figures
	@echo "$(GREEN)Generating figures...$(NC)"
	@$(PYTHON) project4_analysis.py --results_dir results/project4
	@echo "$(GREEN)✓ Figures generated$(NC)"

report:  ## Analysis: Generate comprehensive report
	@echo "$(GREEN)Generating comprehensive report...$(NC)"
	@$(MAKE) analyze
	@echo "$(GREEN)✓ Report generated$(NC)"
	@echo ""
	@echo "$(BLUE)Report components:$(NC)"
	@ls -lh results/project4/summary/
	@echo ""

# ======================================================================
# WORKFLOW TARGETS
# ======================================================================

workflow-test:  ## Complete workflow for testing
	@echo "$(BLUE)================================$(NC)"
	@echo "$(BLUE) Test Workflow$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) test
	@$(MAKE) quick
	@$(MAKE) analyze
	@echo "$(GREEN)✓ Test workflow complete$(NC)"

workflow-basic:  ## Complete workflow with baseline experiments
	@echo "$(BLUE)================================$(NC)"
	@echo "$(BLUE) Basic Workflow$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@$(MAKE) check-setup
	@$(MAKE) test
	@$(MAKE) baseline
	@$(MAKE) analyze
	@echo "$(GREEN)✓ Basic workflow complete$(NC)"

workflow-full:  ## Complete workflow with all experiments
	@echo "$(BLUE)================================$(NC)"
	@echo "$(BLUE) Full Workflow$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo "$(RED)WARNING: This will take several hours!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C within 10 seconds to cancel...$(NC)"
	@sleep 10
	@$(MAKE) check-setup
	@$(MAKE) test
	@$(MAKE) lowdim
	@$(MAKE) highdim
	@$(MAKE) analyze
	@$(MAKE) report
	@echo "$(GREEN)✓ Full workflow complete$(NC)"

# ======================================================================
# UTILITIES
# ======================================================================

info:  ## Utilities: Show project information
	@echo "$(BLUE)================================$(NC)"
	@echo "$(BLUE) Project 4 Information$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(GREEN)Algorithms:$(NC)"
	@echo "  - Forced Sampling"
	@echo "  - LinUCB (quantile)"
	@echo "  - Epsilon-Greedy (quantile)"
	@echo "  - Thompson Sampling (quantile)"
	@echo ""
	@echo "$(GREEN)Current Configuration:$(NC)"
	@echo "  N_SIM = $(N_SIM)"
	@echo "  K = $(K) arms"
	@echo "  D = $(D) dimensions"
	@echo "  T = $(T) rounds"
	@echo "  DF = $(DF) (tail heaviness)"
	@echo "  TAU = $(TAU) (quantile level)"
	@echo "  SEED = $(SEED)"
	@echo ""
	@echo "$(GREEN)Directory Structure:$(NC)"
	@tree -L 2 results/project4 2>/dev/null || ls -R results/project4
	@echo ""

list-results:  ## Utilities: List available results
	@echo "$(GREEN)Available Results:$(NC)"
	@ls -lh results/project4/data/*.pkl 2>/dev/null | wc -l | xargs echo "  Experiment files:"
	@ls -lh results/project4/figures/*.pdf 2>/dev/null | wc -l | xargs echo "  Figure files:"
	@echo ""

clean-results:  ## Utilities: Clean result files (keeps code)
	@echo "$(YELLOW)Cleaning results...$(NC)"
	@rm -rf results/project4/data/*.pkl
	@rm -rf results/project4/figures/*.pdf
	@rm -rf results/project4/summary/*
	@echo "$(GREEN)✓ Results cleaned$(NC)"

clean-all:  ## Utilities: Clean everything (results + cache)
	@echo "$(YELLOW)Cleaning everything...$(NC)"
	@$(MAKE) clean-results
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

backup:  ## Utilities: Backup results to timestamped folder
	@echo "$(GREEN)Creating backup...$(NC)"
	@mkdir -p backups
	@tar -czf backups/project4_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		results/project4 \
		project4_*.py \
		PROJECT4_README.md
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"
	@ls -lh backups/ | tail -1

# ======================================================================
# DEVELOPMENT
# ======================================================================

dev-test:  ## Development: Quick test for development
	@$(MAKE) quick N_SIM=3 T=50

dev-profile:  ## Development: Profile algorithm performance
	@echo "$(GREEN)Profiling algorithms...$(NC)"
	@$(PYTHON) -m cProfile -o results/project4_profile.prof \
		project4_main.py --n_sim 5 --T 100
	@echo "$(GREEN)Profile saved to: results/project4_profile.prof$(NC)"
	@echo "$(YELLOW)View with: python -m pstats results/project4_profile.prof$(NC)"

dev-debug:  ## Development: Run with detailed debug output
	@$(PYTHON) -u project4_main.py \
		--n_sim 2 \
		--T 50 \
		--K 2 \
		--d 5

# ======================================================================
# DOCUMENTATION
# ======================================================================

docs:  ## Documentation: Generate documentation
	@echo "$(GREEN)Documentation available in:$(NC)"
	@echo "  - PROJECT4_README.md"
	@echo "  - docs/PROJECT4_DESIGN.md"
	@echo "  - docs/PROJECT4_THEORY.md"
	@echo ""

view-readme:  ## Documentation: View README
	@cat PROJECT4_README.md | less

# ======================================================================
# END
# ======================================================================

.PHONY: help install check-setup test quick verify baseline lowdim highdim \
        all-experiments custom heavy-tails multiple-arms high-dimensional \
        analyze stats figures report workflow-test workflow-basic workflow-full \
        info list-results clean-results clean-all backup dev-test dev-profile \
        dev-debug docs view-readme
