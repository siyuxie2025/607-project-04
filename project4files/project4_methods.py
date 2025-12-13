"""
Project 4: Bandit Algorithm Implementations with Flexible Generation Support
=============================================================================

Four algorithms with quantile regression updates:
1. Forced Sampling (from Project 3)
2. LinUCB with quantile regression
3. Epsilon-Greedy with quantile regression
4. Thompson Sampling with quantile regression

All algorithms support flexible beta/alpha generation and track estimation errors.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import warnings

# Low-dimensional quantile regression
try:
    from quantes.linear import low_dim
    QUANTES_AVAILABLE = True
except ImportError:
    QUANTES_AVAILABLE = False
    warnings.warn("quantes not available - using fallback")

# High-dimensional quantile regression
try:
    from quantes.linear import high_dim as high_dim_qr
    HIGH_DIM_AVAILABLE = True
except ImportError:
    HIGH_DIM_AVAILABLE = False

# Numerical stability utilities
import sys
sys.path.insert(0, 'src')
try:
    from numerical_stability import (
        clip_extreme_values,
        handle_heavy_tails,
        check_matrix_condition,
        stable_variance,
        EPSILON
    )
except ImportError:
    # Fallback implementations
    EPSILON = 1e-10
    def clip_extreme_values(x, max_val=1e6, warn=False):
        return np.clip(x, -max_val, max_val)
    def handle_heavy_tails(x, method='winsorize', **kwargs):
        lower = kwargs.get('lower_percentile', 1)
        upper = kwargs.get('upper_percentile', 99)
        return np.clip(x, np.percentile(x, lower), np.percentile(x, upper))
    def check_matrix_condition(X, threshold=1e10):
        try:
            cond = np.linalg.cond(X)
            return cond < threshold
        except:
            return False
    def stable_variance(x, ddof=1, min_var=1e-8, axis=None):
        var = np.var(x, ddof=ddof, axis=axis)
        return np.maximum(var, min_var)


class QuantileBanditBase(ABC):
    """
    Base class for all quantile bandit algorithms.
    
    Provides common functionality for tracking beta errors and estimates.
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
            Target quantile level (0.5 for median)
        beta_real : ndarray of shape (K, d)
            True beta coefficients (for error tracking)
        alpha_real : ndarray of shape (K,)
            True intercepts (for error tracking)
        high_dim : bool
            Whether to use high-dimensional methods
        """
        self.K = K
        self.d = d
        self.tau = tau
        self.beta_real = beta_real
        self.alpha_real = alpha_real
        self.high_dim = high_dim
        
        # Check high-dim availability
        if high_dim and not HIGH_DIM_AVAILABLE:
            warnings.warn("High-dim quantile regression not available, using low-dim")
            self.high_dim = False
        
        # Initialize estimates
        self.beta = np.random.uniform(0.0, 2.0, (K, d))
        self.alpha = np.random.uniform(0.0, 2.0, K)
        
        # Data storage
        self.X = [[] for _ in range(K)]  # Contexts per arm
        self.y = [[] for _ in range(K)]  # Rewards per arm
    
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
        # Clip extreme rewards
        reward = clip_extreme_values(reward, max_val=1e6, warn=False)
        
        # Store data
        self.X[action].append(x)
        self.y[action].append(reward)
        
        # Update estimates if enough data
        if len(self.X[action]) > self.d:
            self._fit_quantile_regression(action)
    
    def _fit_quantile_regression(self, action: int):
        """Fit quantile regression for a specific arm."""
        X_arm = np.array(self.X[action])
        y_arm = np.array(self.y[action])
        
        # Handle extreme values
        y_arm = handle_heavy_tails(y_arm, method='winsorize',
                                   lower_percentile=1, upper_percentile=99)
        
        try:
            if QUANTES_AVAILABLE:
                if self.high_dim and HIGH_DIM_AVAILABLE:
                    # High-dimensional with Lasso
                    qr = high_dim_qr(X_arm, y_arm, intercept=True).fit(
                        tau=self.tau,
                        method='lasso'
                    )
                else:
                    # Low-dimensional
                    qr = low_dim(X_arm, y_arm, intercept=True).fit(tau=self.tau)
                
                self.beta[action] = qr['beta'][1:]
                self.alpha[action] = qr['beta'][0]
            else:
                # Fallback to OLS
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression(fit_intercept=True)
                lr.fit(X_arm, y_arm)
                self.beta[action] = lr.coef_
                self.alpha[action] = lr.intercept_
        
        except Exception as e:
            # Keep previous estimates if fitting fails
            pass
    
    def get_beta_errors(self) -> np.ndarray:
        """Get L2 norm of beta estimation error for all arms."""
        errors = np.zeros(self.K)
        for k in range(self.K):
            beta_diff = self.beta[k] - self.beta_real[k]
            errors[k] = np.linalg.norm(beta_diff)
        return errors
    
    def get_beta_estimates(self) -> np.ndarray:
        """Get current beta estimates for all arms."""
        return self.beta.copy()


class ForcedSamplingQuantile(QuantileBanditBase):
    """
    Forced Sampling algorithm with quantile regression.
    
    Based on Bastani & Bayati (2020).
    Uses periodic forced exploration with quantile regression updates.
    """
    
    def __init__(
        self,
        K: int,
        d: int,
        q: int,
        h: float,
        tau: float,
        beta_real: np.ndarray,
        alpha_real: np.ndarray,
        high_dim: bool = False
    ):
        """
        Initialize Forced Sampling bandit.
        
        Parameters
        ----------
        q : int
            Number of forced samples per arm per round
        h : float
            Threshold for action selection
        """
        super().__init__(K, d, tau, beta_real, alpha_real, high_dim)
        self.q = q
        self.h = h
        self.n = 0  # Current round
        self.set = np.array([])  # Forced sampling set
        
        # Separate storage for forced samples
        self.Tx = [[] for _ in range(K)]
        self.Tr = [[] for _ in range(K)]
        
        # Forced sampling estimates
        self.beta_t = np.random.uniform(0.0, 2.0, (K, d))
        self.alpha_t = np.random.uniform(0.0, 2.0, K)
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using forced sampling strategy."""
        # Check if new round starts
        if t == ((2**self.n - 1) * self.K * self.q + 1):
            self.set = np.arange(t, t + self.q * self.K)
            self.n += 1
        
        if t in self.set:
            # Forced sampling phase
            ind = list(self.set).index(t)
            action = int(ind // self.q)
            self.Tx[action].append(x)
        else:
            # Selection phase
            forced_est = np.dot(self.beta_t, x) + self.alpha_t
            max_forced_est = np.max(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.0)[0]
            
            all_est = [np.dot(self.beta[k_hat], x) + self.alpha[k_hat] 
                      for k_hat in K_hat]
            action = K_hat[np.argmax(all_est)]
        
        return action
    
    def update(self, x: np.ndarray, action: int, reward: float, t: int):
        """Update with forced sampling logic."""
        reward = clip_extreme_values(reward, max_val=1e6, warn=False)
        
        # Update forced sampling estimates
        if t in self.set:
            self.Tr[action].append(reward)
            if len(self.Tr[action]) > self.d:
                X_t = np.array(self.Tx[action])
                y_t = np.array(self.Tr[action])
                y_t = handle_heavy_tails(y_t, method='winsorize')
                
                try:
                    if QUANTES_AVAILABLE:
                        qr = low_dim(X_t, y_t, intercept=True).fit(tau=self.tau)
                        self.beta_t[action] = qr['beta'][1:]
                        self.alpha_t[action] = qr['beta'][0]
                except:
                    pass
        
        # Update all-sample estimates
        self.X[action].append(x)
        self.y[action].append(reward)
        
        if len(self.X[action]) > self.d:
            self._fit_quantile_regression(action)


class LinUCBQuantile(QuantileBanditBase):
    """
    LinUCB algorithm with quantile regression.
    
    Uses upper confidence bounds based on quantile regression estimates.
    """
    
    def __init__(
        self,
        K: int,
        d: int,
        alpha: float,
        tau: float,
        beta_real: np.ndarray,
        alpha_real: np.ndarray,
        high_dim: bool = False
    ):
        """
        Initialize LinUCB with quantile regression.
        
        Parameters
        ----------
        alpha : float
            Exploration parameter (confidence width)
        """
        super().__init__(K, d, tau, beta_real, alpha_real, high_dim)
        self.alpha_param = alpha
        
        # Covariance matrices (for confidence bounds)
        self.A = [np.eye(d) for _ in range(K)]
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using UCB strategy."""
        ucb_values = np.zeros(self.K)
        
        for k in range(self.K):
            # Predicted value
            pred = np.dot(self.beta[k], x) + self.alpha[k]
            
            # Confidence bonus
            try:
                A_inv = np.linalg.inv(self.A[k] + 1e-6 * np.eye(self.d))
                bonus = self.alpha_param * np.sqrt(x @ A_inv @ x)
            except:
                bonus = self.alpha_param
            
            ucb_values[k] = pred + bonus
        
        return np.argmax(ucb_values)
    
    def update(self, x: np.ndarray, action: int, reward: float, t: int):
        """Update with covariance matrix update."""
        # Update covariance matrix
        self.A[action] += np.outer(x, x)
        
        # Standard update
        super().update(x, action, reward, t)


class EpsilonGreedyQuantile(QuantileBanditBase):
    """
    Epsilon-Greedy algorithm with quantile regression.
    
    With probability epsilon, explores uniformly; otherwise exploits.
    """
    
    def __init__(
        self,
        K: int,
        d: int,
        epsilon: float,
        tau: float,
        beta_real: np.ndarray,
        alpha_real: np.ndarray,
        high_dim: bool = False,
        decay: bool = True
    ):
        """
        Initialize Epsilon-Greedy with quantile regression.
        
        Parameters
        ----------
        epsilon : float
            Initial exploration probability
        decay : bool
            Whether to decay epsilon over time (epsilon_t = epsilon / sqrt(t))
        """
        super().__init__(K, d, tau, beta_real, alpha_real, high_dim)
        self.epsilon_init = epsilon
        self.decay = decay
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action using epsilon-greedy strategy."""
        # Compute epsilon for this timestep
        if self.decay:
            epsilon_t = self.epsilon_init / np.sqrt(max(1, t))
        else:
            epsilon_t = self.epsilon_init
        
        # Explore or exploit
        if np.random.rand() < epsilon_t:
            # Explore: uniform random
            return np.random.randint(0, self.K)
        else:
            # Exploit: choose best arm
            values = np.array([
                np.dot(self.beta[k], x) + self.alpha[k]
                for k in range(self.K)
            ])
            return np.argmax(values)


class ThompsonSamplingQuantile(QuantileBanditBase):
    """
    Thompson Sampling algorithm with quantile regression.
    
    Samples from empirical posterior based on quantile regression estimates.
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
        """Initialize Thompson Sampling with quantile regression."""
        super().__init__(K, d, tau, beta_real, alpha_real, high_dim)
    
    def choose_action(self, t: int, x: np.ndarray) -> int:
        """Choose action by Thompson sampling."""
        sampled_values = np.zeros(self.K)
        
        for k in range(self.K):
            if len(self.y[k]) > self.d:
                # Estimate uncertainty from residuals
                X_k = np.array(self.X[k])
                y_k = np.array(self.y[k])
                
                # Predicted values
                y_pred = X_k @ self.beta[k] + self.alpha[k]
                residuals = y_k - y_pred
                
                # Variance estimate
                sigma_k = stable_variance(residuals, ddof=1, min_var=1e-4)
                
                # Sample from posterior
                beta_sample = np.random.normal(self.beta[k], np.sqrt(sigma_k / len(y_k)))
                alpha_sample = np.random.normal(self.alpha[k], np.sqrt(sigma_k))
                
                sampled_values[k] = np.dot(beta_sample, x) + alpha_sample
            else:
                # Not enough data - optimistic initialization
                sampled_values[k] = np.dot(self.beta[k], x) + self.alpha[k] + np.random.randn()
        
        return np.argmax(sampled_values)


if __name__ == "__main__":
    # Quick test of all algorithms
    print("Testing all algorithms...\n")
    
    K, d, tau = 2, 5, 0.5
    beta_real = np.random.randn(K, d)
    alpha_real = np.random.randn(K)
    
    algorithms = [
        ("ForcedSampling", ForcedSamplingQuantile(K, d, q=2, h=0.5, tau=tau, 
                                                   beta_real=beta_real, alpha_real=alpha_real)),
        ("LinUCB", LinUCBQuantile(K, d, alpha=1.0, tau=tau,
                                  beta_real=beta_real, alpha_real=alpha_real)),
        ("EpsilonGreedy", EpsilonGreedyQuantile(K, d, epsilon=0.1, tau=tau,
                                                beta_real=beta_real, alpha_real=alpha_real)),
        ("ThompsonSampling", ThompsonSamplingQuantile(K, d, tau=tau,
                                                      beta_real=beta_real, alpha_real=alpha_real))
    ]
    
    # Test each algorithm
    for name, alg in algorithms:
        print(f"Testing {name}...")
        for t in range(1, 21):
            x = np.random.randn(d)
            action = alg.choose_action(t, x)
            reward = np.random.randn()
            alg.update(x, action, reward, t)
            
            if t % 10 == 0:
                errors = alg.get_beta_errors()
                print(f"  t={t}: action={action}, errors={errors.mean():.4f}")
        
        print(f"✓ {name} passed\n")
    
    print("All algorithms working!")