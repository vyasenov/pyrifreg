"""
Collection of Recentered Influence Function (RIF) generators for various distributional statistics.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, Callable


class RIFGenerator:
    """Base class for RIF generators."""
    
    def __init__(self):
        self._kde = None
    
    def _estimate_density(self, y: np.ndarray) -> None:
        """Estimate the density function using kernel density estimation."""
        self._kde = stats.gaussian_kde(y)
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the RIF for the given statistic.
        
        Parameters
        ----------
        y : array-like
            Input data
            
        Returns
        -------
        rif : array-like
            Recentered Influence Function values
        """
        raise NotImplementedError("Subclasses must implement compute method")


class MeanRIF(RIFGenerator):
    """RIF generator for the mean."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        return y


class QuantileRIF(RIFGenerator):
    """RIF generator for quantiles."""
    
    def __init__(self, q: float):
        """
        Parameters
        ----------
        q : float
            Quantile level (between 0 and 1)
        """
        super().__init__()
        if not 0 < q < 1:
            raise ValueError("Quantile level must be between 0 and 1")
        self.q = q
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        self._estimate_density(y)
        q_val = np.quantile(y, self.q)
        f_q = self._kde(q_val)
        return q_val + (self.q - (y <= q_val)) / f_q


class VarianceRIF(RIFGenerator):
    """RIF generator for variance."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        mean_y = np.mean(y)
        var_y = np.var(y)
        return (y - mean_y)**2


class GiniRIF(RIFGenerator):
    """RIF generator for Gini coefficient."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        y_sorted = np.sort(y)
        ranks = np.arange(1, n + 1)
        gini = 2 * np.sum(ranks * y_sorted) / (n * np.sum(y_sorted)) - (n + 1) / n
        
        # Compute RIF for Gini
        rif = np.zeros_like(y)
        for i in range(n):
            y_temp = y.copy()
            y_temp[i] = y[i] + 1e-6  # Small perturbation
            y_temp_sorted = np.sort(y_temp)
            gini_temp = 2 * np.sum(ranks * y_temp_sorted) / (n * np.sum(y_temp_sorted)) - (n + 1) / n
            rif[i] = (gini_temp - gini) / 1e-6
        
        return rif


class IQRRIF(RIFGenerator):
    """RIF generator for Interquartile Range (IQR)."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        self._estimate_density(y)
        q75 = np.quantile(y, 0.75)
        q25 = np.quantile(y, 0.25)
        f_q75 = self._kde(q75)
        f_q25 = self._kde(q25)
        
        # RIF for IQR is the difference of RIFs for 75th and 25th quantiles
        rif_q75 = q75 + (0.75 - (y <= q75)) / f_q75
        rif_q25 = q25 + (0.25 - (y <= q25)) / f_q25
        
        return rif_q75 - rif_q25


class EntropyRIF(RIFGenerator):
    """RIF generator for entropy."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        self._estimate_density(y)
        n = len(y)
        h = 1.06 * np.std(y) * n**(-1/5)  # Silverman's rule of thumb
        entropy = -np.sum(self._kde(y) * np.log(self._kde(y) + 1e-10)) * h
        
        # Compute RIF for entropy
        rif = np.zeros_like(y)
        for i in range(n):
            y_temp = y.copy()
            y_temp[i] = y[i] + 1e-6
            kde_temp = stats.gaussian_kde(y_temp)
            entropy_temp = -np.sum(kde_temp(y_temp) * np.log(kde_temp(y_temp) + 1e-10)) * h
            rif[i] = (entropy_temp - entropy) / 1e-6
        
        return rif


def get_rif_generator(statistic: str, **kwargs) -> RIFGenerator:
    """
    Factory function to get the appropriate RIF generator.
    
    Parameters
    ----------
    statistic : str
        Name of the statistic. Options are:
        - 'mean': Mean
        - 'quantile': Quantile (requires q parameter)
        - 'variance': Variance
        - 'gini': Gini coefficient
        - 'iqr': Interquartile Range
        - 'entropy': Entropy
    **kwargs : dict
        Additional parameters for specific generators (e.g., q for quantile)
        
    Returns
    -------
    generator : RIFGenerator
        The appropriate RIF generator instance
    """
    generators = {
        'mean': MeanRIF,
        'quantile': QuantileRIF,
        'variance': VarianceRIF,
        'gini': GiniRIF,
        'iqr': IQRRIF,
        'entropy': EntropyRIF
    }
    
    if statistic not in generators:
        raise ValueError(f"Unknown statistic: {statistic}. "
                        f"Available options are: {list(generators.keys())}")
    
    return generators[statistic](**kwargs) 