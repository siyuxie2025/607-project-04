"""
Project 4: Quantile Bandit Algorithm Implementations
=====================================================

Implements four bandit algorithms with quantile regression updates:
1. Forced Sampling (from Project 3, adapted)
2. LinUCB with quantile regression
3. Epsilon-greedy with quantile regression
4. Thompson Sampling with quantile regression

Each algorithm supports both low-dimensional and high-dimensional variants.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import warnings

# Low-dimensional quantile regression
from quantes.linear import low_dim

# High-dimensional quantile regression
try:
    from quantes.linear import high_dim
    HIGH_DIM_AVAILABLE = True
except ImportError:
    HIGH_DIM_AVAILABLE = False
    warnings.warn("High-dimensional quantes not available")

# Numerical stability utilities from Project 3
import sys
sys.path.insert(0, 'src')
from numerical_stability import (
    clip_extreme_values,
    handle_heavy_tails,
    check_matrix_condition,
    stable_variance,
    EPSILON
)


class QuantileBanditBase(ABC):
    """
    Base class for all quantile bandit algorithms.
    
    Provides common functionality:
    - Beta error tracking
    - Numerical stability utilities
    - High-dimensional vs low-dimensional selection
    """
    
    def __init__(
        self,
        K: int,
        d: int,
        tau: float,
        beta_real: np.ndarray,
        alpha_real: np.ndarray,
        high_dim: bool = False
    ):
        """
        Initialize base quantile bandit.
        
        Parameters
        ----------
        K : int
            Number of arms
        d : int
            Context dimension
        tau : float
            Target quantile level (e.g., 0.5 for median)
        beta_real : ndarray
            True beta coefficients (for error tracking)
        alpha_real : ndarray
            True intercepts (for error tracking)
        high_dim : bool
            Whether to use high-dimensional methods (Lasso)
        """
        self.K = K
        self.d = d
        self.tau = tau
        self.beta_real = beta_real
        self.alpha_real = alpha_real
        self.high_dim = high_dim
        
        # Check if high-dimensional is requested but not available
        if high_dim and not HIGH_DIM_AVAILABLE:
            warnings.warn("High-dimensional quantes not available, falling back to low-dim")
            self.high_dim = False
        
        # Initialize parameter estimates
        self.beta = np.random.uniform(0.0, 2.0, (K, d))
        self.alpha = np.random.uniform(0.0, 2.0, K)
        
        # Data storage for each arm
        self.X = [[] for _ in range(K)]  # Contexts
        self.y = [[] for _ in range(K)]  # Rewards
    
    @abstractmethod
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action at time t given context x."""
        pass
    
    def update(self, x: np.ndarray, action: int, reward: float, t: int):
        """
        Update algorithm with observed (context, action, reward).
        
        Parameters
        ----------
        x : ndarray
            Context vector
        action : int
            Chosen action
        reward : float
            Observed reward
        t : int
            Current timestep
        """
        # Clip extreme rewards (numerical stability)
        reward = clip_extreme_values(reward, max_val=1e6, warn=False)
        
        # Store data
        self.X[action].append(x)
        self.y[action].append(reward)
        
        # Update parameters for this arm if enough data
        if len(self.X[action]) > max(2, self.d // 2):
            self._fit_quantile_regression(action)
    
    def _fit_quantile_regression(self, action: int):
        """
        Fit quantile regression for a specific arm.
        
        Uses appropriate method based on dimensionality:
        - Low-dim: quantes.linear.low_dim
        - High-dim: quantes.linear.high_dim (with Lasso)
        """
        X_arm = np.array(self.X[action])
        y_arm = np.array(self.y[action])
        
        # Handle extreme values (heavy tails)
        y_arm = handle_heavy_tails(
            y_arm, method='winsorize',
            lower_percentile=1, upper_percentile=99
        )
        
        try:
            if not self.high_dim:
                # Low-dimensional quantile regression
                qr = low_dim(X_arm, y_arm, intercept=True).fit(tau=self.tau)
                self.beta[action] = qr['beta'][1:]
                self.alpha[action] = qr['beta'][0]
            else:
                # High-dimensional quantile regression with Lasso
                # Use cross-validation to select regularization parameter
                qr = high_dim(X_arm, y_arm, intercept=True).fit(
                    tau=self.tau,
                    method='lasso'  # Sparse estimation
                )
                self.beta[action] = qr['beta'][1:]
                self.alpha[action] = qr['beta'][0]
                
        except Exception as e:
            # Keep previous estimates if fitting fails
            warnings.warn(f"Quantile regression failed for arm {action}: {e}")
            pass
    
    def get_beta_errors(self) -> np.ndarray:
        """
        Compute L2 error between estimated and true beta for each arm.
        
        Returns
        -------
        ndarray
            L2 errors for each arm (length K)
        """
        errors = np.zeros(self.K)
        for k in range(self.K):
            errors[k] = np.linalg.norm(self.beta[k] - self.beta_real[k])
        return errors


class ForcedSamplingQuantile(QuantileBanditBase):
    """
    Forced sampling algorithm with quantile regression.
    
    This is the algorithm from Project 3, adapted for the new framework.
    Uses exponentially growing exploration rounds.
    """
    
    def __init__(self, q: int, h: float, **kwargs):
        """
        Initialize forced sampling.
        
        Parameters
        ----------
        q : int
            Number of forced samples per arm per round
        h : float
            Threshold for arm selection
        **kwargs
            Passed to QuantileBanditBase
        """
        super().__init__(**kwargs)
        self.q = q
        self.h = h
        self.n = 0  # Current round
        self.forced_set = np.array([])
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using forced sampling strategy."""
        # Check if we're starting a new exploration round
        if t == ((2**self.n - 1) * self.K * self.q + 1):
            self.forced_set = np.arange(t, t + self.q * self.K)
            self.n += 1
        
        # Forced sampling phase
        if t in self.forced_set:
            ind = list(self.forced_set).index(t)
            return int(ind // self.q)
        
        # Exploitation phase
        # Estimate rewards for all arms
        estimated_rewards = np.dot(self.beta, x) + self.alpha
        max_reward = np.amax(estimated_rewards)
        
        # Find arms within h/2 of maximum
        K_hat = np.where(estimated_rewards > max_reward - self.h / 2.0)[0]
        
        # Among candidate arms, choose the one with highest estimate
        if len(K_hat) > 0:
            best_in_Khat = K_hat[np.argmax(estimated_rewards[K_hat])]
            return int(best_in_Khat)
        else:
            # Fallback: choose best arm
            return int(np.argmax(estimated_rewards))


class LinUCBQuantile(QuantileBanditBase):
    """
    LinUCB algorithm with quantile regression.
    
    Uses upper confidence bounds based on quantile regression estimates.
    This is computationally expensive as it requires fitting at each timestep.
    """
    
    def __init__(self, alpha: float, **kwargs):
        """
        Initialize LinUCB with quantile regression.
        
        Parameters
        ----------
        alpha : float
            Exploration parameter (controls confidence bound width)
        **kwargs
            Passed to QuantileBanditBase
        """
        super().__init__(**kwargs)
        self.alpha_param = alpha
        
        # Initialize covariance matrices for each arm
        self.A = [np.eye(self.d) * 0.1 for _ in range(self.K)]
        self.b = [np.zeros(self.d) for _ in range(self.K)]
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using UCB strategy."""
        ucb_values = np.zeros(self.K)
        
        for k in range(self.K):
            # Refit quantile regression at each timestep (expensive!)
            if len(self.X[k]) > max(2, self.d // 2):
                self._fit_quantile_regression(k)
            
            # Compute estimated reward
            estimated_reward = np.dot(self.beta[k], x) + self.alpha[k]
            
            # Compute confidence bonus
            # Use quantile-based variance estimate
            if len(self.y[k]) > 2:
                # Estimate variance from residuals
                y_k = np.array(self.y[k])
                X_k = np.array(self.X[k])
                pred_k = np.dot(X_k, self.beta[k]) + self.alpha[k]
                residuals = y_k - pred_k
                
                # Robust variance estimate
                variance = stable_variance(residuals)
            else:
                variance = 1.0  # Default
            
            # Compute UCB
            A_inv = np.linalg.inv(self.A[k] + EPSILON * np.eye(self.d))
            confidence_bonus = self.alpha_param * np.sqrt(
                variance * x.T @ A_inv @ x
            )
            
            ucb_values[k] = estimated_reward + confidence_bonus
        
        return int(np.argmax(ucb_values))
    
    def update(self, x: np.ndarray, action: int, reward: float, t: int):
        """Update with additional covariance matrix updates."""
        # Standard update
        super().update(x, action, reward, t)
        
        # Update covariance matrix (for confidence bounds)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


class EpsilonGreedyQuantile(QuantileBanditBase):
    """
    Epsilon-greedy algorithm with quantile regression.
    
    Simple exploration-exploitation strategy: with probability epsilon,
    explore uniformly; otherwise exploit current estimates.
    """
    
    def __init__(self, epsilon: float, decay: bool = True, **kwargs):
        """
        Initialize epsilon-greedy.
        
        Parameters
        ----------
        epsilon : float
            Exploration probability (0 < epsilon < 1)
        decay : bool
            Whether to decay epsilon over time
        **kwargs
            Passed to QuantileBanditBase
        """
        super().__init__(**kwargs)
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.decay = decay
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using epsilon-greedy strategy."""
        # Decay epsilon if requested
        if self.decay:
            # Epsilon = epsilon_0 / sqrt(t)
            self.epsilon = self.epsilon_initial / np.sqrt(t)
        
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.K)
        
        # Exploit: choose arm with highest estimated reward
        estimated_rewards = np.dot(self.beta, x) + self.alpha
        return int(np.argmax(estimated_rewards))


class ThompsonSamplingQuantile(QuantileBanditBase):
    """
    Thompson Sampling with quantile regression.
    
    Samples parameters from posterior distribution and chooses arm
    with highest sampled reward.
    
    For quantile regression, we use empirical Bayesian approach:
    - Estimate beta via quantile regression
    - Compute empirical covariance from data
    - Sample from Gaussian with these parameters
    """
    
    def __init__(self, **kwargs):
        """Initialize Thompson Sampling."""
        super().__init__(**kwargs)
        
        # Prior parameters (will be updated based on data)
        self.prior_mean = [np.zeros(self.d) for _ in range(self.K)]
        self.prior_cov = [np.eye(self.d) for _ in range(self.K)]
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using Thompson Sampling."""
        sampled_rewards = np.zeros(self.K)
        
        for k in range(self.K):
            # Update posterior if enough data
            if len(self.X[k]) > max(2, self.d):
                self._update_posterior(k)
            
            # Sample beta from posterior
            try:
                beta_sample = np.random.multivariate_normal(
                    self.prior_mean[k],
                    self.prior_cov[k]
                )
            except np.linalg.LinAlgError:
                # If covariance is singular, add regularization
                beta_sample = np.random.multivariate_normal(
                    self.prior_mean[k],
                    self.prior_cov[k] + 1e-6 * np.eye(self.d)
                )
            
            # Compute sampled reward
            sampled_rewards[k] = np.dot(beta_sample, x) + self.alpha[k]
        
        return int(np.argmax(sampled_rewards))
    
    def _update_posterior(self, action: int):
        """
        Update posterior distribution for an arm.
        
        Uses empirical Bayes: estimate parameters from data
        """
        X_arm = np.array(self.X[action])
        y_arm = np.array(self.y[action])
        
        # Fit quantile regression to get point estimate
        y_arm = handle_heavy_tails(y_arm, method='winsorize')
        
        try:
            if not self.high_dim:
                qr = low_dim(X_arm, y_arm, intercept=True).fit(tau=self.tau)
                beta_est = qr['beta'][1:]
                alpha_est = qr['beta'][0]
            else:
                qr = high_dim(X_arm, y_arm, intercept=True).fit(
                    tau=self.tau, method='lasso'
                )
                beta_est = qr['beta'][1:]
                alpha_est = qr['beta'][0]
            
            # Update point estimates
            self.beta[action] = beta_est
            self.alpha[action] = alpha_est
            
            # Estimate covariance from residuals
            predictions = np.dot(X_arm, beta_est) + alpha_est
            residuals = y_arm - predictions
            
            # Robust covariance estimate
            # Use residual variance × (X^T X)^{-1}
            XtX = X_arm.T @ X_arm
            
            if check_matrix_condition(XtX):
                residual_var = stable_variance(residuals)
                cov_estimate = residual_var * np.linalg.inv(XtX + 1e-6 * np.eye(self.d))
            else:
                # Fallback: use prior
                cov_estimate = np.eye(self.d)
            
            # Update posterior parameters
            self.prior_mean[action] = beta_est
            self.prior_cov[action] = cov_estimate
            
        except Exception as e:
            # Keep previous posterior if update fails
            warnings.warn(f"Posterior update failed for arm {action}: {e}")
            pass


# Convenience function for creating algorithms
def create_algorithm(
    algo_name: str,
    K: int,
    d: int,
    tau: float,
    beta_real: np.ndarray,
    alpha_real: np.ndarray,
    high_dim: bool = False,
    **kwargs
) -> QuantileBanditBase:
    """
    Factory function to create algorithm instances.
    
    Parameters
    ----------
    algo_name : str
        Algorithm name: 'ForcedSampling', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling'
    K, d, tau, beta_real, alpha_real, high_dim
        Standard parameters
    **kwargs
        Algorithm-specific parameters
    
    Returns
    -------
    QuantileBanditBase
        Initialized algorithm instance
    """
    common_params = {
        'K': K, 'd': d, 'tau': tau,
        'beta_real': beta_real,
        'alpha_real': alpha_real,
        'high_dim': high_dim
    }
    
    if algo_name == 'ForcedSampling':
        q = kwargs.get('q', 2)
        h = kwargs.get('h', 0.5)
        return ForcedSamplingQuantile(q=q, h=h, **common_params)
    
    elif algo_name == 'LinUCB':
        alpha = kwargs.get('alpha', 1.0)
        return LinUCBQuantile(alpha=alpha, **common_params)
    
    elif algo_name == 'EpsilonGreedy':
        epsilon = kwargs.get('epsilon', 0.1)
        decay = kwargs.get('decay', True)
        return EpsilonGreedyQuantile(epsilon=epsilon, decay=decay, **common_params)
    
    elif algo_name == 'ThompsonSampling':
        return ThompsonSamplingQuantile(**common_params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


if __name__ == "__main__":
    # Test algorithm implementations
    print("Testing Project 4 Algorithm Implementations\n")
    
    K, d = 3, 5
    tau = 0.5
    beta_real = np.random.randn(K, d)
    alpha_real = np.random.randn(K)
    
    algorithms = [
        ('ForcedSampling', {'q': 2, 'h': 0.5}),
        ('LinUCB', {'alpha': 1.0}),
        ('EpsilonGreedy', {'epsilon': 0.1}),
        ('ThompsonSampling', {})
    ]
    
    print("Testing algorithm initialization and basic operations:\n")
    
    for algo_name, params in algorithms:
        print(f"Testing {algo_name}...")
        
        algo = create_algorithm(
            algo_name, K, d, tau, beta_real, alpha_real,
            high_dim=False, **params
        )
        
        # Test action selection
        x = np.random.randn(d)
        action = algo.choose_action(1, x)
        print(f"  Chose action: {action}")
        
        # Test update
        reward = np.random.randn()
        algo.update(x, action, reward, 1)
        print(f"  Updated successfully")
        
        # Test error computation
        errors = algo.get_beta_errors()
        print(f"  Beta errors: {errors}")
        print()
    
    print("✓ All algorithms initialized and tested successfully")