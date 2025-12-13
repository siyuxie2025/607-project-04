import numpy as np
from abc import ABC, abstractmethod

class DataGenerator(ABC):
    '''
    Abstract base class for data generation.
    
    All data generators must implement three methods:
    - generate(n): Create n samples from the distribution
    - name(): Return a descriptive name for reporting
    - null_value(): Return the true parameter value under H0

    This ensures all generators can be used interchangeably in simulation
    '''
    @abstractmethod
    def generate(self, n, rng=None):
        '''
        Generate n samples from the distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng (no seed) 
        
        Returns
        -------
        np.ndarray
            Array of n samples from the distribution
        '''
        
        pass
    
    @property
    @abstractmethod
    def name(self):
        '''
        Return a descriptive name for reporting.
        
        Returns
        -------
        str
            Descriptive name of the data generator
        '''
        
        pass

    @property
    @abstractmethod
    def null_value(self):
        '''
        Return the true parameter value under H0.

        Returns
        -------
        float
            True parameter value under H0
        '''

        pass

class NormalGenerator(DataGenerator):
    """ Generate data from a normal distribution.
    
    The normal distribution is symmetric and serves as the baseline case
    for most parametric tests. It is characterized by its mean (loc) and
    standard deviation (scale).
    
    Parameters
    ----------
    loc : float, default=0
        Mean of the distribution (location parameter)
    scale : float, default=1
        Standard deviation of the distribution (scale parameter)
    
    Examples
    --------
    >>> gen = NormalGenerator(loc=5, scale=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'Normal(μ=5, σ=2)'
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=self.mean, scale=self.std, size=n)

    @property
    def name(self):
        return f'Normal(mean={self.mean}, std={self.std})'

    @property
    def null_value(self):
        return self.mean
    

class TruncatedNormalGenerator(DataGenerator):
    """Generate data from a truncated normal distribution.
    
    The truncated normal distribution is a normal distribution bounded
    within a specified range. It is useful for modeling data that cannot
    exceed certain limits.
    
    Parameters
    ----------
    mean : float, default=0
        Mean of the underlying normal distribution
    std : float, default=1
        Standard deviation of the underlying normal distribution
    low : float, default=-5
        Lower bound of the truncation
    high : float, default=5
        Upper bound of the truncation
    
    Examples
    --------
    >>> gen = TruncatedNormalGenerator(mean=0, std=1, low=-2, high=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'TruncatedNormal(μ=0, σ=1, low=-5, high=5)'
    """

    def __init__(self, mean=0.0, std=1.0, low=-5.0, high=5.0):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        samples = []
        while len(samples) < n:
            sample = rng.normal(loc=self.mean, scale=self.std)
            if self.low <= sample <= self.high:
                samples.append(sample)
        return np.array(samples)
    
    @property
    def name(self):
        return f'TruncatedNormal(mean={self.mean}, std={self.std}, low={self.low}, high={self.high})'
    
    @property
    def null_value(self):
        return self.mean

class UniformGenerator(DataGenerator):
    """Generate data from a uniform distribution.
    
    The uniform distribution has finite support and zero skewness but heavy
    tails relative to the normal distribution. It is useful for studying
    how bounded distributions affect test performance.
    
    Parameters
    ----------
    low : float, default=0
        Lower bound of the distribution
    high : float, default=2
        Upper bound of the distribution
    
    Examples
    --------
    >>> gen = UniformGenerator(low=0, high=10)
    >>> data = gen.generate(100)
    >>> gen.name
    'Uniform(0, 10)'
    """
    def __init__(self, low=0.0, high=2.0):
        self.low = low
        self.high = high

    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(low=self.low, high=self.high, size=n)

    @property
    def name(self):
        return f'Uniform(low={self.low}, high={self.high})'

    @property
    def null_value(self):
        return (self.low + self.high) / 2.
    

class TGenerator(DataGenerator):
    """Generate data from a t-distribution.
    
    The t-distribution is useful for modeling data with heavier tails
    than the normal distribution. It is characterized by its degrees of
    freedom (df).
    
    Parameters
    ----------
    df : int
        Degrees of freedom of the t-distribution
    
    Examples
    --------
    >>> gen = TGenerator(df=10)
    >>> data = gen.generate(100)
    >>> gen.name()
    't(df=10)'
    """
    
    def __init__(self, df, scale=1):
        self.df = df
        self.scale = scale
    
    def generate(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return self.scale * rng.standard_t(self.df, n)
    
    @property
    def name(self):
        return f"t(df={self.df})"
    
    @property
    def null_value(self):
        return 0.0