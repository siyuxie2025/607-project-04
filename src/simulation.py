import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import seaborn as sns
from src.methods import RiskAwareBandit, OLSBandit
from tqdm import tqdm
from src.generators import NormalGenerator, TGenerator, UniformGenerator, TruncatedNormalGenerator
import pickle
import os
from datetime import datetime

class SimulationStudy:
    """
    Orchestrate simulation studies for assessing bandit algorithms.
    This class manages running multiple simulation replications to estimate
    statistical properties. It handles:
    - Running individual scenarios (one generator + one bandit algorithm)
    - Running factorial designs (all combinations of generators and bandit algorithms)
    - Computing Monte Carlo confidence intervals for estimates
    - Storing and reporting results
    """
    def __init__(self, n_sim, K, d, T, q, h, tau, err_generator, context_generator, 
                 beta_low=[0.0, 0.5], beta_high=[1, 1.5], random_seed=None):
        self.n_sim = n_sim
        self.K = K
        self.d = d
        self.T = T
        self.q = q
        self.h = h
        self.tau = tau
        self.err_generator = err_generator
        self.context_generator = context_generator
        self.random_seed = random_seed

        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
            np.random.seed(random_seed)
        else:
            self.rng = np.random.default_rng()

        self.q_err = np.quantile(self.err_generator.generate(2000, rng=self.rng), self.tau)

        # Generate real beta and alpha values once for all simulations
        beta1 = UniformGenerator(low=beta_low[0], high=beta_high[0]).generate((self.K//2, self.d), rng=self.rng)
        beta2 = UniformGenerator(low=beta_low[1], high=beta_high[1]).generate((self.K-self.K//2, self.d), rng=self.rng)
        self.beta_real_value = np.vstack([beta1, beta2])
        
        alpha1 = UniformGenerator(low=beta_low[0], high=beta_high[0]).generate(self.K//2, rng=self.rng)
        alpha2 = UniformGenerator(low=beta_low[0], high=beta_high[0]).generate(self.K-self.K//2, rng=self.rng)
        self.alpha_real_value = np.concatenate([alpha1, alpha2])

        # Store results
        self.results = None

    def run_one_scenario(self):
        """Run one scenario to compute cumulative regret for both bandit algorithms."""
        RAB = RiskAwareBandit(
            q=self.q, h=self.h, tau=self.tau, 
            d=self.d, K=self.K, 
            beta_real_value=self.beta_real_value, 
            alpha_real_value=self.alpha_real_value
        )
        OLSB = OLSBandit(
            q=self.q, h=self.h, d=self.d, K=self.K,
            beta_real_value=self.beta_real_value
        )
        
        diff = 0
        RWD = []
        RWD_OLS = []
        opt_RWD = []
        opt_RWD_OLS = []
        
        # Track beta errors and estimates for this scenario
        beta_errors_rab = []
        beta_errors_ols = []
        beta_estimates_rab = []
        beta_estimates_ols = []

        for t in tqdm(range(1, self.T + 1), desc="Running one scenario", leave=False):
            rwd, RAB, rwd_OLS, OLSB, opt_rwd, opt_rwd_OLS, diff = self._run_one_timestep(
                RAB, OLSB, t, diff
            )
            RWD.append(rwd)
            RWD_OLS.append(rwd_OLS)
            opt_RWD.append(opt_rwd)
            opt_RWD_OLS.append(opt_rwd_OLS)
            
            # Record beta errors and estimates at each timestep
            beta_errors_rab.append(RAB.beta_error_a.copy())
            beta_errors_ols.append(OLSB.beta_error_a.copy())
            beta_estimates_rab.append(RAB.beta_a.copy())  # Shape: (K, d)
            beta_estimates_ols.append(OLSB.beta_a.copy())  # Shape: (K, d)
        
        regret_RAB = np.cumsum(opt_RWD) - np.cumsum(RWD)
        regret_OLSB = np.cumsum(opt_RWD_OLS) - np.cumsum(RWD_OLS)

        return (regret_RAB, regret_OLSB, diff, 
                beta_errors_rab, beta_errors_ols,
                beta_estimates_rab, beta_estimates_ols)

    def _run_one_timestep(self, RAB, OLSB, t, diff, err_generator=None):
        """Run one timestep of the simulation for both bandit algorithms."""
        if err_generator is None:
            err_generator = self.err_generator

        # Generate d-dimensional context vector
        x = self.context_generator.generate(self.d, rng=self.rng)
        if x.ndim > 1:
            x = x.ravel()
        
        a = RAB.choose_a(t, x)
        a_OLS = OLSB.choose_a(t, x)

        diff += (a != a_OLS)

        # Risk Aware Bandit update
        rwd = np.dot(self.beta_real_value[a], x) + self.alpha_real_value[a]
        rwd_noisy = rwd + (0.5 * x[-1] + 1) * (err_generator.generate(1, rng=self.rng)[0] - self.q_err)
        RAB.update_beta(rwd_noisy, t)

        # OLS Bandit update
        rwd_OLS = np.dot(self.beta_real_value[a_OLS], x) + self.alpha_real_value[a_OLS]
        rwd_OLS_noisy = rwd_OLS + (0.5 * x[-1] + 1) * (err_generator.generate(1, rng=self.rng)[0] - self.q_err)
        OLSB.update_beta(rwd_OLS_noisy, t)

        # Optimal rewards (same for both)
        opt_rwd = np.amax(np.dot(self.beta_real_value, x) + self.alpha_real_value)

        return rwd, RAB, rwd_OLS, OLSB, opt_rwd, opt_rwd, diff

    def run_simulation(self):
        """Run multiple simulation replications and store results.
        
        Each simulation uses the SAME true parameters (beta, alpha) but different
        random realizations of contexts and errors. This estimates the expected
        performance of each algorithm on this specific problem instance.
        """
        cumulated_regret_RiskAware = []
        cumulated_regret_OLS = []
        num_diff = []
        
        # Store beta errors and estimates
        all_beta_errors_rab = []
        all_beta_errors_ols = []
        all_beta_estimates_rab = []
        all_beta_estimates_ols = []

        for sim in tqdm(range(self.n_sim), desc="Running simulations"):
            # Unpack all return values including beta errors and estimates
            (regret_RAB, regret_OLSB, diff, 
             beta_errors_rab, beta_errors_ols,
             beta_estimates_rab, beta_estimates_ols) = self.run_one_scenario()

            cumulated_regret_RiskAware.append(regret_RAB)
            cumulated_regret_OLS.append(regret_OLSB)
            num_diff.append(diff / self.T)
            
            # Store errors and estimates as numpy arrays
            all_beta_errors_rab.append(np.array(beta_errors_rab))  # Shape: (T, K)
            all_beta_errors_ols.append(np.array(beta_errors_ols))  # Shape: (T, K)
            all_beta_estimates_rab.append(np.array(beta_estimates_rab))  # Shape: (T, K, d)
            all_beta_estimates_ols.append(np.array(beta_estimates_ols))  # Shape: (T, K, d)
        
        # Store results for later access
        self.results = {
            "cumulated_regret_RiskAware": np.array(cumulated_regret_RiskAware), 
            "cumulated_regret_OLS": np.array(cumulated_regret_OLS),
            "beta_errors_rab": np.array(all_beta_errors_rab),  # Shape: (n_sim, T, K)
            "beta_errors_ols": np.array(all_beta_errors_ols),  # Shape: (n_sim, T, K)
            "beta_estimates_rab": np.array(all_beta_estimates_rab),  # Shape: (n_sim, T, K, d)
            "beta_estimates_ols": np.array(all_beta_estimates_ols),  # Shape: (n_sim, T, K, d)
            "num_diff": num_diff
        }
        
        return self.results
    
    def save_results(self, filepath=None, save_metadata=True):
        """
        Save simulation results to disk.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save results. If None, generates automatic filename.
        save_metadata : bool, optional
            Whether to save metadata alongside results
        
        Returns
        -------
        str
            Path where results were saved
        """
        if self.results is None:
            raise ValueError("No results to save. Run simulation first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate automatic filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            err_name = self.err_generator.name.replace('(', '').replace(')', '').replace('=', '').replace(',', '_').replace(' ', '')
            filepath = f'results/simulation_{err_name}_K{self.K}_d{self.d}_T{self.T}_{timestamp}.pkl'
        
        # Save results using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to: {filepath}")
        
        # Save metadata if requested
        if save_metadata:
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'n_sim': self.n_sim,
                'K': self.K,
                'd': self.d,
                'T': self.T,
                'q': self.q,
                'h': self.h,
                'tau': self.tau,
                'random_seed': self.random_seed,
                'err_generator': self.err_generator.name,
                'context_generator': self.context_generator.name,
                'timestamp': datetime.now().isoformat(),
                'results_shapes': {
                    'cumulated_regret_RiskAware': self.results['cumulated_regret_RiskAware'].shape,
                    'cumulated_regret_OLS': self.results['cumulated_regret_OLS'].shape,
                    'beta_errors_rab': self.results['beta_errors_rab'].shape,
                    'beta_errors_ols': self.results['beta_errors_ols'].shape,
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to: {metadata_path}")
        
        return filepath
    
    @classmethod
    def load_results(cls, filepath):
        """
        Load previously saved simulation results.
        
        Parameters
        ----------
        filepath : str
            Path to the saved results file
        
        Returns
        -------
        dict
            Dictionary containing simulation results
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Results loaded from: {filepath}")
        
        # Try to load metadata if it exists
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Metadata loaded from: {metadata_path}")
            return results, metadata
        
        return results

    def save_summary_statistics(self, filepath=None):
        """
        Save summary statistics to CSV file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save CSV. If None, generates automatic filename.
        
        Returns
        -------
        str
            Path where summary was saved
        """
        if self.results is None:
            raise ValueError("No results to summarize. Run simulation first.")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'results/summary_statistics_{timestamp}.csv'
        
        # Calculate summary statistics
        summary_data = []
        
        for method, regret_key in [('RiskAware', 'cumulated_regret_RiskAware'), 
                                     ('OLS', 'cumulated_regret_OLS')]:
            regret = self.results[regret_key]
            final_regret = regret[:, -1]
            
            summary_data.append({
                'Method': method,
                'Mean_Final_Regret': np.mean(final_regret),
                'Median_Final_Regret': np.median(final_regret),
                'Std_Final_Regret': np.std(final_regret),
                'Min_Final_Regret': np.min(final_regret),
                'Max_Final_Regret': np.max(final_regret),
                'Q25_Final_Regret': np.percentile(final_regret, 25),
                'Q75_Final_Regret': np.percentile(final_regret, 75),
            })
        
        # Add beta error statistics
        for method, error_key in [('RiskAware', 'beta_errors_rab'), 
                                   ('OLS', 'beta_errors_ols')]:
            errors = self.results[error_key][:, -1, :]  # Final timestep, all arms
            avg_error = np.mean(errors, axis=1)  # Average across arms
            
            summary_data[-1 if method == 'OLS' else -2].update({
                'Mean_Final_Beta_Error': np.mean(avg_error),
                'Median_Final_Beta_Error': np.median(avg_error),
                'Std_Final_Beta_Error': np.std(avg_error),
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)
        
        print(f"Summary statistics saved to: {filepath}")
        print("\nSummary Statistics:")
        print(df.to_string(index=False))
        
        return filepath

    def plot_regret_results(self, results=None, figsize=(10, 6), use_ci=True, ci_level=0.95):
        """Plot simulation results with confidence intervals or min/max ranges."""
        steps = np.arange(1, self.T + 1)

        if results is None:
            if self.results is None:
                raise ValueError("No results to plot. Please run the simulation first.")
            results = self.results

        cumulated_regret_RiskAware = results["cumulated_regret_RiskAware"]
        cumulated_regret_OLS = results["cumulated_regret_OLS"]
        num_diff = results["num_diff"]

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Left plot: Cumulative Regret
        ax1 = axes[0]
        mean_risk_aware = np.mean(cumulated_regret_RiskAware, axis=0)
        mean_ols = np.mean(cumulated_regret_OLS, axis=0)

        if use_ci:
            # Compute standard errors
            se_risk_aware = np.std(cumulated_regret_RiskAware, axis=0, ddof=1) / np.sqrt(self.n_sim+0.01)
            se_ols = np.std(cumulated_regret_OLS, axis=0, ddof=1) / np.sqrt(self.n_sim)

            t_crit = stats.t.ppf((1 + ci_level) / 2, df=self.n_sim - 1)

            # Confidence intervals
            lower_risk_aware = mean_risk_aware - t_crit * se_risk_aware
            upper_risk_aware = mean_risk_aware + t_crit * se_risk_aware
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols

            ci_label = f"{int(ci_level * 100)}% CI"
        else:
            # Min/max ranges
            lower_risk_aware = np.min(cumulated_regret_RiskAware, axis=0)
            upper_risk_aware = np.max(cumulated_regret_RiskAware, axis=0)
            lower_ols = np.min(cumulated_regret_OLS, axis=0)
            upper_ols = np.max(cumulated_regret_OLS, axis=0)

            ci_label = "Min/Max Range"

        # Plot Risk Aware Bandit
        ax1.fill_between(steps, lower_risk_aware, upper_risk_aware, 
                         color='red', alpha=0.2, label=f'Risk Aware {ci_label}')
        ax1.plot(steps, mean_risk_aware, 'r-', 
                label='Risk Aware Bandit (Mean)', linewidth=2)

        # Plot OLS Bandit
        ax1.fill_between(steps, lower_ols, upper_ols, 
                         color='blue', alpha=0.2, label=f'OLS {ci_label}')
        ax1.plot(steps, mean_ols, 'b-', 
                label='OLS Bandit (Mean)', linewidth=2)

        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Cumulative Regret', fontsize=12)
        ax1.legend(loc='best')
        ax1.set_title(f'Cumulative Regret, d={self.d}, K={self.K}, tau={self.tau}', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Right plot: Number of different actions
        ax2 = axes[1]
        ax2.plot(num_diff, 'g-', linewidth=1.5)
        ax2.axhline(y=np.mean(num_diff), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(num_diff):.3f}')
        ax2.set_xlabel('Simulation', fontsize=12)
        ax2.set_ylabel('Proportion of Different Actions', fontsize=12)
        ax2.set_title('Action Disagreement between Algorithms', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/simulation_results_d{self.d}_K{self.K}_tau{self.tau}.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.show()

        # Print summary statistics
        print(f"\n=== Simulation Summary (n_sim={self.n_sim}) ===")
        print(f"Final Median Regret - Risk Aware: {np.median(cumulated_regret_RiskAware[:, -1]):.2f}")
        print(f"Final Median Regret - OLS: {np.median(cumulated_regret_OLS[:, -1]):.2f}")
        print(f"Mean Action Disagreement: {np.mean(num_diff):.2%}")
        print(f"Std Action Disagreement: {np.std(num_diff):.2%}")

    def plot_beta_error_results(self, results=None, figsize=(10, 6), use_ci=True, ci_level=0.95, separate_arms=False):
        """Plot beta estimation error results over time for both bandit algorithms.
        
        Parameters
        ----------
        results : dict, optional
            Results dictionary
        figsize : tuple
            Figure size
        use_ci : bool
            If True, show confidence intervals
        ci_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        separate_arms : bool
            If True, plot each arm in a separate subplot. If False, plot all arms together.
        """
        if results is None:
            if self.results is None:
                raise ValueError("No results to plot. Please run the simulation first.")
            results = self.results

        beta_errors_rab = results["beta_errors_rab"]  # Shape: (n_sim, T, K)
        beta_errors_ols = results["beta_errors_ols"]  # Shape: (n_sim, T, K)

        print(f"Debug: beta_errors_rab shape: {beta_errors_rab.shape}")
        print(f"Debug: beta_errors_ols shape: {beta_errors_ols.shape}")

        steps = np.arange(1, self.T + 1)

        if separate_arms:
            # Plot each arm in separate subplot
            fig, axes = plt.subplots(1, self.K, figsize=(figsize[0] * self.K / 2, figsize[1]))
            if self.K == 1:
                axes = [axes]
            
            for k in range(self.K):
                ax = axes[k]
                
                # Extract errors for this arm: (n_sim, T)
                rab_k = beta_errors_rab[:, :, k]
                ols_k = beta_errors_ols[:, :, k]
                
                # Mean across simulations
                mean_rab = np.mean(rab_k, axis=0)
                mean_ols = np.mean(ols_k, axis=0)
                
                if use_ci:
                    se_rab = np.std(rab_k, axis=0, ddof=1) / np.sqrt(self.n_sim)
                    se_ols = np.std(ols_k, axis=0, ddof=1) / np.sqrt(self.n_sim)
                    
                    t_crit = stats.t.ppf((1 + ci_level) / 2, self.n_sim - 1)
                    
                    lower_rab = mean_rab - t_crit * se_rab
                    upper_rab = mean_rab + t_crit * se_rab
                    lower_ols = mean_ols - t_crit * se_ols
                    upper_ols = mean_ols + t_crit * se_ols
                    
                    ax.fill_between(steps, lower_rab, upper_rab, color='red', alpha=0.2, label=f'{int(ci_level*100)}% CI RAB')
                    ax.fill_between(steps, lower_ols, upper_ols, color='blue', alpha=0.2, label=f'{int(ci_level*100)}% CI OLS')

                ax.plot(steps, mean_rab, 'r-', label='Risk Aware Bandit', linewidth=2)
                ax.plot(steps, mean_ols, 'b--', label='OLS Bandit', linewidth=2)
                
                ax.set_xlabel('Time', fontsize=11)
                ax.set_ylabel('Beta Estimation Error', fontsize=11)
                ax.set_title(f'Arm {k}', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Print summary for this arm
                print(f"\n--- Arm {k} Summary ---")
                print(f"Final Beta Error - Risk Aware: {mean_rab[-1]:.4f}")
                print(f"Final Beta Error - OLS: {mean_ols[-1]:.4f}")
            
            plt.suptitle(f'Beta Estimation Error by Arm (d={self.d}, K={self.K}, τ={self.tau})', 
                        fontsize=14, y=1.02)
        else:
            # Plot all arms together in one plot
            fig, ax = plt.subplots(figsize=figsize)

            colors_rab = plt.cm.Reds(np.linspace(0.5, 0.9, self.K))
            colors_ols = plt.cm.Blues(np.linspace(0.5, 0.9, self.K))
            
            for k in range(self.K):
                # Extract errors for this arm: (n_sim, T)
                rab_k = beta_errors_rab[:, :, k]
                ols_k = beta_errors_ols[:, :, k]
                
                # Mean across simulations
                mean_rab = np.mean(rab_k, axis=0)
                mean_ols = np.mean(ols_k, axis=0)
                
                if use_ci:
                    se_rab = np.std(rab_k, axis=0, ddof=1) / np.sqrt(self.n_sim)
                    se_ols = np.std(ols_k, axis=0, ddof=1) / np.sqrt(self.n_sim)
                    
                    t_crit = stats.t.ppf((1 + ci_level) / 2, self.n_sim - 1)
                    
                    lower_rab = mean_rab - t_crit * se_rab
                    upper_rab = mean_rab + t_crit * se_rab
                    lower_ols = mean_ols - t_crit * se_ols
                    upper_ols = mean_ols + t_crit * se_ols
                    
                    ax.fill_between(steps, lower_rab, upper_rab, color=colors_rab[k], alpha=0.2)
                    ax.fill_between(steps, lower_ols, upper_ols, color=colors_ols[k], alpha=0.2)

                ax.plot(steps, mean_rab, color=colors_rab[k], label=f'Risk Aware - Arm {k}', 
                       linestyle='-', linewidth=2)
                ax.plot(steps, mean_ols, color=colors_ols[k], label=f'OLS - Arm {k}', 
                       linestyle='--', linewidth=2)

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Mean Beta Estimation Error', fontsize=12)
            ax.set_title(f'Beta Estimation Error over Time (d={self.d}, K={self.K}, τ={self.tau})', 
                        fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        suffix = '_separate' if separate_arms else '_combined'
        plt.savefig(f'results/beta_error{suffix}_d{self.d}_K{self.K}_tau{self.tau}.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.show()


# Usage example:
if __name__ == "__main__":
    n_sim = 50
    K = 2
    d = 10
    T = 150
    q = 2
    h = 0.5
    tau = 0.5

    RANDOM_SEED = 1010

    err_generator = TGenerator(df=2.25, scale=0.7)
    context_generator = TruncatedNormalGenerator(mean=0.0, std=1.0)

    study = SimulationStudy(
        n_sim=n_sim, K=K, d=d, T=T, q=q, h=h, tau=tau, 
        random_seed=RANDOM_SEED,
        err_generator=err_generator,
        context_generator=context_generator
    )

    results = study.run_simulation()
    study.plot_regret_results(use_ci=True, ci_level=0.95)
    
    results_path = study.save_results()
    study.save_summary_statistics()

    # Option 1: Plot all arms together (default)
    study.plot_beta_error_results(use_ci=True, ci_level=0.95, separate_arms=False)

    # Option 2: Plot each arm in separate subplot (what you want!)
    study.plot_beta_error_results(use_ci=True, ci_level=0.95, separate_arms=True)