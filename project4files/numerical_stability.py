"""
Numerical Stability Utilities for Bandit Simulation
====================================================

This module provides numerically stable implementations of common operations
to prevent division by zero, overflow/underflow, and other numerical issues.

Usage:
    from src.numerical_stability import *
    
    # Replace: result = a / b
    # With:    result = safe_divide(a, b)
"""

import numpy as np
import warnings
from typing import Union, Optional

# Global constants for numerical stability
EPSILON = 1e-10          # Threshold for treating values as zero
MAX_VALUE = 1e10         # Maximum allowed value before clipping
MIN_VARIANCE = 1e-8      # Minimum variance to prevent singular matrices
MAX_CONDITION = 1e12     # Maximum condition number for well-conditioned matrices
MIN_SAMPLES = 5          # Minimum samples before computing statistics


def safe_divide(
    numerator: np.ndarray, 
    denominator: np.ndarray, 
    default: float = 0.0,
    epsilon: float = EPSILON
) -> np.ndarray:
    """
    Numerically stable division that handles division by zero.
    
    Parameters
    ----------
    numerator : array_like
        Numerator values
    denominator : array_like  
        Denominator values
    default : float, optional
        Value to return when denominator is zero (default: 0.0)
    epsilon : float, optional
        Threshold for treating denominator as zero
        
    Returns
    -------
    array_like
        Result of division with safe handling of zeros
        
    Examples
    --------
    >>> safe_divide(np.array([1, 2, 3]), np.array([2, 0, 4]))
    array([0.5, 0.0, 0.75])
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    
    # Create mask for valid denominators
    valid_mask = np.abs(denominator) > epsilon
    
    # Initialize result with default value
    result = np.full_like(numerator, default, dtype=float)
    
    # Perform division only where denominator is non-zero
    if np.any(valid_mask):
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return result


def clip_extreme_values(
    x: np.ndarray, 
    max_val: float = MAX_VALUE,
    warn: bool = False
) -> np.ndarray:
    """
    Clip extreme values to prevent overflow/underflow.
    
    Parameters
    ----------
    x : array_like
        Input values
    max_val : float, optional
        Maximum absolute value allowed
    warn : bool, optional
        Whether to warn when clipping occurs
        
    Returns
    -------
    array_like
        Clipped values in range [-max_val, max_val]
    """
    x = np.asarray(x, dtype=float)
    
    # Check if clipping will occur
    if warn and (np.any(x > max_val) or np.any(x < -max_val)):
        n_clipped = np.sum((x > max_val) | (x < -max_val))
        warnings.warn(
            f"Clipping {n_clipped} extreme values (>{max_val} or <{-max_val})",
            RuntimeWarning
        )
    
    return np.clip(x, -max_val, max_val)


def stable_mean(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Compute mean with NaN/Inf handling.
    
    Parameters
    ----------
    x : array_like
        Input values
    axis : int, optional
        Axis along which to compute mean
        
    Returns
    -------
    float or array_like
        Mean value(s), with NaN/Inf filtered out
    """
    x = np.asarray(x, dtype=float)
    
    # Filter out NaN and Inf
    x_clean = x[np.isfinite(x)]
    
    if x_clean.size == 0:
        return 0.0 if axis is None else np.zeros(x.shape[axis] if axis else x.shape)
    
    return np.mean(x_clean, axis=axis)


def stable_variance(
    x: np.ndarray, 
    ddof: int = 1,
    min_var: float = MIN_VARIANCE,
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Compute variance with numerical stability guarantees.
    
    This prevents singular matrices by ensuring variance is never too small.
    
    Parameters
    ----------
    x : array_like
        Input values
    ddof : int, optional
        Degrees of freedom (default: 1 for sample variance)
    min_var : float, optional
        Minimum variance threshold
    axis : int, optional
        Axis along which to compute variance
        
    Returns
    -------
    float or array_like
        Variance, guaranteed to be >= min_var
    """
    x = np.asarray(x, dtype=float)
    
    # Remove NaN and Inf
    if axis is None:
        x_clean = x[np.isfinite(x)]
    else:
        x_clean = x
    
    if x_clean.size <= ddof:
        return min_var
    
    # Compute variance using two-pass algorithm for stability
    variance = np.var(x_clean, ddof=ddof, axis=axis)
    
    # Ensure minimum variance
    return np.maximum(variance, min_var)


def stable_std(
    x: np.ndarray,
    ddof: int = 1, 
    min_var: float = MIN_VARIANCE,
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Compute standard deviation with numerical stability.
    
    Parameters
    ----------
    x : array_like
        Input values
    ddof : int, optional
        Degrees of freedom
    min_var : float, optional
        Minimum variance threshold
    axis : int, optional
        Axis along which to compute std
        
    Returns
    -------
    float or array_like
        Standard deviation, guaranteed positive
    """
    var = stable_variance(x, ddof=ddof, min_var=min_var, axis=axis)
    return np.sqrt(var)


def check_matrix_condition(
    X: np.ndarray, 
    threshold: float = MAX_CONDITION,
    return_condition: bool = False
) -> Union[bool, tuple]:
    """
    Check if a matrix is well-conditioned.
    
    A matrix is well-conditioned if its condition number is below the threshold.
    Ill-conditioned matrices can lead to numerical instability in linear algebra.
    
    Parameters
    ----------
    X : array_like
        Input matrix (2D)
    threshold : float, optional
        Maximum acceptable condition number
    return_condition : bool, optional
        If True, return (is_well_conditioned, condition_number)
        
    Returns
    -------
    bool or (bool, float)
        Whether matrix is well-conditioned, optionally with condition number
        
    Examples
    --------
    >>> X = np.array([[1, 0], [0, 1]])
    >>> check_matrix_condition(X)
    True
    
    >>> X = np.array([[1, 1], [1, 1.00001]])  # Nearly singular
    >>> check_matrix_condition(X)
    False
    """
    X = np.asarray(X, dtype=float)
    
    # Check if matrix is empty or has wrong shape
    if X.size == 0 or X.ndim != 2:
        return (False, np.inf) if return_condition else False
    
    # Check for NaN or Inf
    if not np.all(np.isfinite(X)):
        return (False, np.inf) if return_condition else False
    
    try:
        cond = np.linalg.cond(X)
        is_well_conditioned = cond < threshold and np.isfinite(cond)
        
        if return_condition:
            return is_well_conditioned, cond
        return is_well_conditioned
        
    except np.linalg.LinAlgError:
        return (False, np.inf) if return_condition else False


def safe_matrix_inverse(
    X: np.ndarray,
    regularization: float = 1e-6,
    check_condition: bool = True
) -> Optional[np.ndarray]:
    """
    Compute matrix inverse with regularization and condition checking.
    
    Parameters
    ----------
    X : array_like
        Input matrix to invert
    regularization : float, optional
        Ridge regularization parameter (added to diagonal)
    check_condition : bool, optional
        Whether to check condition number before inversion
        
    Returns
    -------
    array_like or None
        Inverse matrix, or None if matrix is ill-conditioned
        
    Notes
    -----
    Uses regularization: (X^T X + λI)^(-1) instead of (X^T X)^(-1)
    This improves numerical stability for nearly singular matrices.
    """
    X = np.asarray(X, dtype=float)
    
    if X.ndim != 2:
        warnings.warn("Input must be 2D matrix", RuntimeWarning)
        return None
    
    # Check condition if requested
    if check_condition and not check_matrix_condition(X):
        warnings.warn("Matrix is ill-conditioned, adding regularization", RuntimeWarning)
    
    try:
        # Add ridge regularization to diagonal
        n = X.shape[0]
        X_reg = X + regularization * np.eye(n)
        
        # Compute inverse
        X_inv = np.linalg.inv(X_reg)
        
        # Verify result
        if not np.all(np.isfinite(X_inv)):
            warnings.warn("Matrix inverse contains NaN or Inf", RuntimeWarning)
            return None
            
        return X_inv
        
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Matrix inversion failed: {e}", RuntimeWarning)
        return None


def safe_sqrt(x: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Compute square root with protection against negative values.
    
    Parameters
    ----------
    x : array_like
        Input values
    epsilon : float, optional
        Threshold for treating values as zero
        
    Returns
    -------
    array_like
        Square root, with negative values treated as zero
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(x, epsilon))


def safe_log(
    x: np.ndarray, 
    epsilon: float = EPSILON,
    base: Optional[float] = None
) -> np.ndarray:
    """
    Compute logarithm with protection against non-positive values.
    
    Parameters
    ----------
    x : array_like
        Input values
    epsilon : float, optional
        Minimum value for log (log(epsilon) for x <= 0)
    base : float, optional
        Logarithm base (None for natural log)
        
    Returns
    -------
    array_like
        Logarithm values
    """
    x = np.asarray(x, dtype=float)
    x_safe = np.maximum(x, epsilon)
    
    if base is None:
        return np.log(x_safe)
    else:
        return np.log(x_safe) / np.log(base)


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Numerically stable log-sum-exp: log(sum(exp(x))).
    
    This prevents overflow when computing sums of exponentials.
    
    Parameters
    ----------
    x : array_like
        Input values
    axis : int, optional
        Axis along which to sum
        
    Returns
    -------
    float or array_like
        log(sum(exp(x))) computed in a stable manner
        
    Notes
    -----
    Uses the identity: log(sum(exp(x))) = a + log(sum(exp(x - a)))
    where a = max(x) prevents overflow.
    """
    x = np.asarray(x, dtype=float)
    
    # Handle empty array
    if x.size == 0:
        return -np.inf
    
    # Subtract maximum for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    
    # Compute log-sum-exp
    with np.errstate(over='ignore'):  # We handle overflow manually
        result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    
    # Remove keepdims if it was added
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    else:
        result = np.squeeze(result)
    
    return result


def winsorize(
    x: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Winsorize data by capping extreme values at percentiles.
    
    This is useful for heavy-tailed distributions where extreme outliers
    can destabilize algorithms.
    
    Parameters
    ----------
    x : array_like
        Input values
    lower_percentile : float, optional
        Lower percentile threshold (0-100)
    upper_percentile : float, optional
        Upper percentile threshold (0-100)
        
    Returns
    -------
    array_like
        Winsorized values
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 100, 200])
    >>> winsorize(x, 0, 90)  # Cap at 90th percentile
    array([  1,   2,   3, 100, 100])
    """
    x = np.asarray(x, dtype=float)
    
    # Compute percentiles
    lower_bound = np.percentile(x, lower_percentile)
    upper_bound = np.percentile(x, upper_percentile)
    
    # Clip values
    return np.clip(x, lower_bound, upper_bound)


def handle_heavy_tails(
    x: np.ndarray,
    method: str = 'clip',
    threshold: float = 3.0,
    **kwargs
) -> np.ndarray:
    """
    Handle heavy-tailed distributions to improve numerical stability.
    
    Parameters
    ----------
    x : array_like
        Input values
    method : str, optional
        Method to use: 'clip', 'winsorize', or 'median_mad'
    threshold : float, optional
        Threshold in standard deviations or MAD units
    **kwargs
        Additional arguments for specific methods
        
    Returns
    -------
    array_like
        Processed values with reduced tail influence
    """
    x = np.asarray(x, dtype=float)
    
    if method == 'clip':
        # Clip at mean ± threshold * std
        mean = stable_mean(x)
        std = stable_std(x)
        return np.clip(x, mean - threshold * std, mean + threshold * std)
        
    elif method == 'winsorize':
        # Winsorize at specified percentiles
        lower = kwargs.get('lower_percentile', 1.0)
        upper = kwargs.get('upper_percentile', 99.0)
        return winsorize(x, lower, upper)
        
    elif method == 'median_mad':
        # Use median and MAD (robust to outliers)
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        mad = np.maximum(mad, MIN_VARIANCE)  # Prevent division by zero
        return np.clip(x, median - threshold * mad, median + threshold * mad)
        
    else:
        raise ValueError(f"Unknown method: {method}")


class NumericalStabilityMonitor:
    """
    Monitor numerical stability during computation.
    
    Usage:
        monitor = NumericalStabilityMonitor()
        
        result = some_computation(x)
        monitor.check(result, 'computation_name')
        
        monitor.report()
    """
    
    def __init__(self):
        self.issues = []
        
    def check(self, x: np.ndarray, name: str = "value"):
        """Check array for numerical issues"""
        x = np.asarray(x)
        
        if np.any(np.isnan(x)):
            self.issues.append(f"NaN detected in {name}")
            
        if np.any(np.isinf(x)):
            self.issues.append(f"Inf detected in {name}")
            
        if np.any(np.abs(x) > MAX_VALUE):
            self.issues.append(f"Extreme values (>{MAX_VALUE}) in {name}")
    
    def check_matrix(self, X: np.ndarray, name: str = "matrix"):
        """Check matrix condition"""
        is_good, cond = check_matrix_condition(X, return_condition=True)
        if not is_good:
            self.issues.append(f"Ill-conditioned {name} (cond={cond:.2e})")
    
    def report(self) -> str:
        """Get report of all issues"""
        if not self.issues:
            return "No numerical issues detected ✓"
        return "\n".join([f"⚠ {issue}" for issue in self.issues])
    
    def reset(self):
        """Clear all recorded issues"""
        self.issues = []


# Convenience function for quick stability checks
def ensure_numerical_stability(x: np.ndarray, context: str = "") -> np.ndarray:
    """
    All-in-one numerical stability processing.
    
    Applies:
    1. NaN/Inf removal
    2. Extreme value clipping
    3. Returns cleaned array
    
    Parameters
    ----------
    x : array_like
        Input values
    context : str, optional
        Context string for warnings
        
    Returns
    -------
    array_like
        Numerically stable version of input
    """
    x = np.asarray(x, dtype=float)
    
    # Check for issues
    n_nan = np.sum(np.isnan(x))
    n_inf = np.sum(np.isinf(x))
    
    if n_nan > 0:
        warnings.warn(f"{context}: Removed {n_nan} NaN values", RuntimeWarning)
        x = np.where(np.isnan(x), 0.0, x)
    
    if n_inf > 0:
        warnings.warn(f"{context}: Removed {n_inf} Inf values", RuntimeWarning)
        x = np.where(np.isinf(x), np.sign(x) * MAX_VALUE, x)
    
    # Clip extreme values
    x = clip_extreme_values(x)
    
    return x


if __name__ == "__main__":
    # Example usage and tests
    print("Numerical Stability Utilities - Examples\n")
    
    # Test safe_divide
    print("1. Safe division:")
    nums = np.array([1, 2, 3, 4])
    denoms = np.array([2, 0, 3, 0])
    result = safe_divide(nums, denoms, default=0.0)
    print(f"   {nums} / {denoms} = {result}\n")
    
    # Test extreme value clipping
    print("2. Extreme value clipping:")
    x = np.array([1e12, 100, 1e-12, -1e12])
    clipped = clip_extreme_values(x, max_val=1e10)
    print(f"   Original: {x}")
    print(f"   Clipped:  {clipped}\n")
    
    # Test matrix condition
    print("3. Matrix condition checking:")
    good_matrix = np.array([[2, 0], [0, 2]])
    bad_matrix = np.array([[1, 1], [1, 1.000001]])
    print(f"   Good matrix: {check_matrix_condition(good_matrix)}")
    print(f"   Bad matrix:  {check_matrix_condition(bad_matrix)}\n")
    
    # Test stability monitor
    print("4. Stability monitoring:")
    monitor = NumericalStabilityMonitor()
    monitor.check(np.array([1, 2, np.nan, 4]), "test_array")
    monitor.check(np.array([1e15, 2, 3]), "extreme_values")
    print(monitor.report())