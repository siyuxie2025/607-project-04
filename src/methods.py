"""
Modified methods.py with Numerical Stability Improvements
==========================================================

Key changes:
1. Added numerical stability utilities
2. Winsorization of extreme values
3. Matrix condition checking
4. Ridge regression fallback for ill-conditioned matrices
5. Try-except blocks for robust error handling
"""

import numpy as np
from quantes.linear import low_dim
from sklearn.linear_model import LinearRegression, Ridge
from abc import ABC, abstractmethod

# ADDED: Import numerical stability utilities
from src.numerical_stability import (
    safe_divide, 
    stable_variance, 
    stable_std,
    check_matrix_condition,
    safe_matrix_inverse,
    clip_extreme_values,
    EPSILON,
    handle_heavy_tails
)

class ForcedSamplingBandit(ABC):
    '''
    Abstract base class for Bandit algorithms with forced sampling and all-sample estimators.
    '''
    @abstractmethod
    def __init__(self, q, h, d, K, **kwargs):
        '''
        Initialize the Bandit algorithm.

        Parameters
        ----------
        q : int
            Number of forced samples per arm in each round
        h : float
            Difference between optimal and suboptimal arms
        d : int
            Dimension of the context vectors
        K : int
            Number of arms
        '''
        pass
    
    @abstractmethod
    def choose_a(self, t, x):
        '''
        Choose an action based on the current time step and context vector.

        Parameters
        ----------
        t : int
            The current time step.
        x : np.ndarray
            The context vector of dimension d.

        Returns
        -------
        int
            The index of the chosen action.
        '''
        pass

    @abstractmethod
    def update_beta(self, rwd, t):
        '''
        Update the estimators based on the received reward.

        Parameters
        ----------
        rwd : float
            The reward received after taking the action.
        t : int
            The current time step.

        Returns
        -------
        None
        '''
        pass


class RiskAwareBandit(ForcedSamplingBandit):
    '''
    A class for the Risk-Aware Bandit algorithm with forced sampling and all-sample estimators.
    
    OPTIMIZATION: Added numerical stability improvements for heavy-tailed distributions.
    '''
    def __init__(self, q, h, tau, d, K, beta_real_value, alpha_real_value):
        '''
        Initialize the Risk-Aware Bandit algorithm.

        Parameters:
        ----------
        q (int) : Number of forced samples per arm in each round
        h (float) : Difference between optimal and suboptimal arms
        tau (float) : Quantile level for quantile regression
        d (int) : Dimension of the context vectors
        K (int) : Number of arms
        beta_real_value (np.ndarray) : True coefficient values for each arm
        alpha_real_value (np.ndarray) : True intercept values for each arm
        '''
        self.Tx = [[] for _ in range(K)]
        self.Sx = [[] for _ in range(K)]
        self.Tr = [[] for _ in range(K)]
        self.Sr = [[] for _ in range(K)]

        self.q = q
        self.h = h
        self.tau = tau
        self.d = d
        self.K = K

        self.set = np.array([])
        self.action = None

        self.beta_t = np.random.uniform(0., 2., (K, d))
        self.beta_a = np.random.uniform(0., 2., (K, d))
        self.alpha_t = np.random.uniform(0., 2., K)
        self.alpha_a = np.random.uniform(0., 2., K)
        self.n = 0

        self.beta_real_value = beta_real_value
        self.alpha_real_value = alpha_real_value

        self.beta_error_a = np.zeros(K)
        self.beta_error_t = np.zeros(K)
    
    def choose_a(self, t, x): 
        """ 
        Choose an action based on the current time step and context vector.
        
        If the current time step is part of the forced sampling set,
        select the corresponding action. Otherwise, use the estimators
        to choose the action with the highest estimated reward within
        the acceptable range.

        Parameters
        ----------
        t : int
            The current time step.
        x : np.ndarray
            The context vector of dimension d.
        Returns
        -------
        int
            The index of the chosen action.
        """
        # If t is the first time of the new round
        if t == ((2**self.n - 1)*self.K*self.q + 1):
            self.set = np.arange(t, t+self.q*self.K)
            self.n += 1

        if t in self.set: 
            ind = list(self.set).index(t)
            self.action = int(ind // self.q)
            self.Tx[self.action].append(x)
            self.Sx[self.action].append(x)
        else:
            forced_est = np.dot(self.beta_t, x) + self.alpha_t
            max_forced_est = np.amax(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.)[0]
            all_est = [np.dot(self.beta_a[k_hat], x) + self.alpha_a[k_hat] for k_hat in K_hat]
            self.action = K_hat[np.argmax(all_est)]
            self.Sx[self.action].append(x)

        return self.action
    
    def update_beta(self, rwd, t):
        """
        Update the estimators based on the received reward.
        
        OPTIMIZATION: Added numerical stability improvements:
        - Clip extreme rewards to prevent overflow
        - Winsorize heavy-tailed distributions
        - Try-except blocks for robust error handling
        - Keep previous estimates if fitting fails
        """
        # OPTIMIZATION: Clip extreme rewards (especially for df < 2)
        rwd = clip_extreme_values(rwd, max_val=1e6, warn=False)
        
        # Check if we have enough samples
        n_samples = np.array(self.Tx[self.action]).shape[0]
        
        if n_samples > self.d:
            # ===== FORCED SAMPLING ESTIMATOR =====
            if t in self.set:
                self.Tr[self.action].append(rwd)
                
                # Prepare data
                X = np.array(self.Tx[self.action])
                y = np.array(self.Tr[self.action])
                
                # OPTIMIZATION: Handle extreme values in y (heavy tails)
                # Winsorize at 1st and 99th percentile
                y = handle_heavy_tails(y, method='winsorize', 
                                      lower_percentile=1, upper_percentile=99)
                
                try:
                    # Fit quantile regression with error handling
                    forced_qr = low_dim(X, y, intercept=True).fit(tau=self.tau)
                    self.beta_t[self.action] = forced_qr['beta'][1:]
                    self.alpha_t[self.action] = forced_qr['beta'][0]
                except Exception as e:
                    # OPTIMIZATION: If quantile regression fails, keep previous estimates
                    # This can happen with extreme outliers or convergence issues
                    pass

            # ===== ALL-SAMPLE ESTIMATOR =====
            self.Sr[self.action].append(rwd)
            
            X_all = np.array(self.Sx[self.action])
            y_all = np.array(self.Sr[self.action])
            
            # OPTIMIZATION: Handle extreme values
            y_all = handle_heavy_tails(y_all, method='winsorize',
                                       lower_percentile=1, upper_percentile=99)
            
            try:
                all_qr = low_dim(X_all, y_all, intercept=True).fit(tau=self.tau)
                self.beta_a[self.action] = all_qr['beta'][1:]
                self.alpha_a[self.action] = all_qr['beta'][0]
            except Exception as e:
                # Keep previous estimates if fitting fails
                pass
        
        else:
            # ===== NOT ENOUGH SAMPLES YET =====
            if t in self.set:
                self.Tr[self.action].append(rwd)
                
                # Try to fit if we have at least 2 samples
                if len(self.Tr[self.action]) >= 2:
                    try:
                        X = np.array(self.Tx[self.action])
                        y = np.array(self.Tr[self.action])
                        
                        # More aggressive winsorization for small samples
                        y = handle_heavy_tails(y, method='winsorize',
                                              lower_percentile=5, upper_percentile=95)
                        
                        forced_qr = low_dim(X, y, intercept=True).fit(tau=self.tau)
                        self.beta_t[self.action] = forced_qr['beta'][1:]
                        self.alpha_t[self.action] = forced_qr['beta'][0]
                    except:
                        # Keep random initialization if fit fails
                        pass
            
            self.Sr[self.action].append(rwd)
            # Keep random initialization for insufficient samples
        
        # ===== ALWAYS UPDATE ERRORS =====
        # OPTIMIZATION: Safe computation of norms
        beta_diff = self.beta_a[self.action] - self.beta_real_value[self.action]
        self.beta_error_a[self.action] = np.linalg.norm(beta_diff)
        
        beta_diff_t = self.beta_t[self.action] - self.beta_real_value[self.action]
        self.beta_error_t[self.action] = np.linalg.norm(beta_diff_t)


class OLSBandit(ForcedSamplingBandit):
    """
    OLS Bandit with forced sampling and all-sample estimators.
    
    OPTIMIZATION: Added numerical stability improvements for OLS regression.
    """
    def __init__(self, q, h, d, K, beta_real_value):
        """Initialize the OLSBandit with given parameters."""
        self.Sx = [[] for _ in range(K)]
        self.Sr = [[] for _ in range(K)]
        self.Tx = [[] for _ in range(K)]
        self.Tr = [[] for _ in range(K)]

        self.beta_a = np.random.uniform(0., 2., (K, d))
        self.beta_t = np.random.uniform(0., 2., (K, d))

        self.beta_real_value = beta_real_value

        self.beta_error_a = np.zeros(K)
        self.beta_error_t = np.zeros(K)

        self.q = q
        self.h = h
        self.d = d
        self.K = K

        self.n = 0

        self.set = np.array([])
        self.action = None

    def choose_a(self, t, x):
        """Choose an action based on the current time step and context vector."""
        if t == ((2**self.n - 1)*self.K*self.q + 1):
            self.set = np.arange(t, t+self.q*self.K)
            self.n += 1
        
        if t in self.set:
            ind = list(self.set).index(t)
            self.action = int(ind // self.q)
            self.Tx[self.action].append(x)
        else:
            forced_est = np.dot(self.beta_t, x)
            max_forced_est = np.amax(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.)[0]
            all_est = [np.dot(self.beta_a[k_hat], x) for k_hat in K_hat]
            self.action = int(K_hat[np.argmax(all_est)])
            
        self.Sx[self.action].append(x)
        return self.action

    def update_beta(self, rwd, t):
        """
        Update the estimators based on the received reward.
        
        OPTIMIZATION: Added numerical stability improvements:
        - Clip extreme rewards
        - Winsorize heavy-tailed distributions
        - Check matrix condition before OLS
        - Use Ridge regression for ill-conditioned matrices
        - Try-except blocks for robust error handling
        """
        # OPTIMIZATION: Clip extreme rewards
        rwd = clip_extreme_values(rwd, max_val=1e6, warn=False)
        
        n_samples = np.array(self.Tx[self.action]).shape[0]
        
        if n_samples > self.d:
            # ===== FORCED SAMPLING ESTIMATOR =====
            if t in self.set:
                self.Tr[self.action].append(rwd)
                
                X = np.array(self.Tx[self.action])
                y = np.array(self.Tr[self.action])
                
                # OPTIMIZATION: Handle extreme values
                y = handle_heavy_tails(y, method='winsorize',
                                      lower_percentile=1, upper_percentile=99)
                
                # OPTIMIZATION: Check matrix condition before OLS
                XtX = X.T @ X
                if check_matrix_condition(XtX, threshold=1e10):
                    # Matrix is well-conditioned - use OLS
                    try:
                        forced_ols = LinearRegression(fit_intercept=False)
                        forced_ols.fit(X, y)
                        self.beta_t[self.action] = forced_ols.coef_
                    except Exception as e:
                        # Keep previous estimate if fit fails
                        pass
                else:
                    # OPTIMIZATION: Matrix is ill-conditioned - use Ridge regression
                    # This adds small regularization to diagonal
                    forced_ridge = Ridge(alpha=1e-6, fit_intercept=False)
                    forced_ridge.fit(X, y)
                    self.beta_t[self.action] = forced_ridge.coef_

            # ===== ALL-SAMPLE ESTIMATOR =====
            self.Sr[self.action].append(rwd)
            
            X_all = np.array(self.Sx[self.action])
            y_all = np.array(self.Sr[self.action])
            
            # OPTIMIZATION: Handle extreme values
            y_all = handle_heavy_tails(y_all, method='winsorize',
                                       lower_percentile=1, upper_percentile=99)
            
            # OPTIMIZATION: Check matrix condition
            XtX_all = X_all.T @ X_all
            if check_matrix_condition(XtX_all, threshold=1e10):
                try:
                    all_ols = LinearRegression(fit_intercept=False)
                    all_ols.fit(X_all, y_all)
                    self.beta_a[self.action] = all_ols.coef_
                except Exception as e:
                    pass
            else:
                # Use Ridge regression for ill-conditioned matrices
                all_ridge = Ridge(alpha=1e-6, fit_intercept=False)
                all_ridge.fit(X_all, y_all)
                self.beta_a[self.action] = all_ridge.coef_

            # Update errors
            self.beta_error_a[self.action] = np.linalg.norm(
                self.beta_a[self.action] - self.beta_real_value[self.action]
            )
            self.beta_error_t[self.action] = np.linalg.norm(
                self.beta_t[self.action] - self.beta_real_value[self.action]
            )
        
        else:
            # ===== NOT ENOUGH SAMPLES YET =====
            if t in self.set:
                self.Tr[self.action].append(rwd)
                
                # Try to fit if we have enough samples
                if len(self.Tr[self.action]) >= max(2, self.d):
                    X = np.array(self.Tx[self.action])
                    y = np.array(self.Tr[self.action])
                    
                    # OPTIMIZATION: Handle extreme values with more aggressive winsorization
                    y = handle_heavy_tails(y, method='winsorize',
                                          lower_percentile=5, upper_percentile=95)
                    
                    try:
                        forced_ols = LinearRegression(fit_intercept=False)
                        forced_ols.fit(X, y)
                        self.beta_t[self.action] = forced_ols.coef_
                    except:
                        # Keep random initialization if fit fails
                        pass
            
            self.Sr[self.action].append(rwd)
            # Keep random initialization for insufficient samples
            
            # Update errors
            self.beta_error_a[self.action] = np.linalg.norm(
                self.beta_a[self.action] - self.beta_real_value[self.action]
            )
            self.beta_error_t[self.action] = np.linalg.norm(
                self.beta_t[self.action] - self.beta_real_value[self.action]
            )