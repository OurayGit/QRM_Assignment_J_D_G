"""
Copula Module

Implementation of copulas for modeling default dependency in the credit portfolio.
Includes Gaussian and Student-t copulas.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, t as t_dist
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class Copula(ABC):
    """Abstract base class for copulas."""
    
    @abstractmethod
    def generate_uniforms(self, n_simulations: int, n_dimensions: int) -> np.ndarray:
        """Generate correlated uniform random variables."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the copula."""
        pass


class GaussianCopula(Copula):
    """
    Gaussian (Normal) Copula implementation.
    
    The Gaussian copula models dependence using the multivariate normal distribution.
    It's widely used in credit risk modeling but has limitations in capturing
    tail dependence.
    """
    
    def __init__(self, correlation: float = 0.3, seed: Optional[int] = None):
        """
        Initialize Gaussian Copula.
        
        Args:
            correlation: Pairwise correlation coefficient (default: 0.3)
            seed: Random seed for reproducibility
        """
        self.correlation = correlation
        self.seed = seed
        
    def _build_correlation_matrix(self, n: int) -> np.ndarray:
        """Build equicorrelation matrix."""
        corr_matrix = np.full((n, n), self.correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix
    
    def generate_uniforms(self, n_simulations: int, n_dimensions: int) -> np.ndarray:
        """
        Generate correlated uniform random variables using Gaussian copula.
        
        Args:
            n_simulations: Number of simulations (scenarios)
            n_dimensions: Number of dimensions (counterparties)
            
        Returns:
            Array of shape (n_simulations, n_dimensions) with correlated uniforms
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Build correlation matrix
        corr_matrix = self._build_correlation_matrix(n_dimensions)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate independent standard normals
        Z = np.random.standard_normal((n_simulations, n_dimensions))
        
        # Apply Cholesky factor to induce correlation
        X = Z @ L.T
        
        # Transform to uniforms using normal CDF
        U = norm.cdf(X)
        
        return U
    
    def name(self) -> str:
        return f"Gaussian Copula (ρ={self.correlation})"


class TCopula(Copula):
    """
    Student-t Copula implementation.
    
    The t-copula captures tail dependence, making it more suitable for
    modeling extreme joint events (simultaneous defaults) than the Gaussian copula.
    """
    
    def __init__(self, correlation: float = 0.3, df: int = 4, seed: Optional[int] = None):
        """
        Initialize Student-t Copula.
        
        Args:
            correlation: Pairwise correlation coefficient (default: 0.3)
            df: Degrees of freedom (default: 4, lower = heavier tails)
            seed: Random seed for reproducibility
        """
        self.correlation = correlation
        self.df = df
        self.seed = seed
        
    def _build_correlation_matrix(self, n: int) -> np.ndarray:
        """Build equicorrelation matrix."""
        corr_matrix = np.full((n, n), self.correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix
    
    def generate_uniforms(self, n_simulations: int, n_dimensions: int) -> np.ndarray:
        """
        Generate correlated uniform random variables using t-copula.
        
        Args:
            n_simulations: Number of simulations (scenarios)
            n_dimensions: Number of dimensions (counterparties)
            
        Returns:
            Array of shape (n_simulations, n_dimensions) with correlated uniforms
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Build correlation matrix
        corr_matrix = self._build_correlation_matrix(n_dimensions)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate independent standard normals
        Z = np.random.standard_normal((n_simulations, n_dimensions))
        
        # Apply Cholesky factor to induce correlation
        X = Z @ L.T
        
        # Generate chi-squared for t-distribution
        chi2 = np.random.chisquare(self.df, size=n_simulations)
        
        # Scale by chi-squared to get multivariate t
        T = X / np.sqrt(chi2[:, np.newaxis] / self.df)
        
        # Transform to uniforms using t CDF
        U = t_dist.cdf(T, self.df)
        
        return U
    
    def name(self) -> str:
        return f"t-Copula (ρ={self.correlation}, ν={self.df})"


class FactorCopula(Copula):
    """
    One-Factor Gaussian Copula (Vasicek model).
    
    This is the standard model used in Basel II/III for regulatory capital.
    Each counterparty's creditworthiness is driven by a common systematic
    factor and an idiosyncratic factor.
    """
    
    def __init__(self, asset_correlation: float = 0.2, seed: Optional[int] = None):
        """
        Initialize One-Factor Copula.
        
        Args:
            asset_correlation: Asset correlation parameter (ρ in Basel)
            seed: Random seed for reproducibility
        """
        self.asset_correlation = asset_correlation
        self.seed = seed
        
    def generate_uniforms(self, n_simulations: int, n_dimensions: int) -> np.ndarray:
        """
        Generate correlated uniform random variables using one-factor model.
        
        Asset value: A_i = sqrt(ρ) * M + sqrt(1-ρ) * ε_i
        where M is the systematic factor and ε_i is idiosyncratic.
        
        Args:
            n_simulations: Number of simulations (scenarios)
            n_dimensions: Number of dimensions (counterparties)
            
        Returns:
            Array of shape (n_simulations, n_dimensions) with correlated uniforms
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        rho = self.asset_correlation
        sqrt_rho = np.sqrt(rho)
        sqrt_1_minus_rho = np.sqrt(1 - rho)
        
        # Systematic factor (common to all counterparties)
        M = np.random.standard_normal(n_simulations)
        
        # Idiosyncratic factors
        epsilon = np.random.standard_normal((n_simulations, n_dimensions))
        
        # Asset values
        A = sqrt_rho * M[:, np.newaxis] + sqrt_1_minus_rho * epsilon
        
        # Transform to uniforms
        U = norm.cdf(A)
        
        return U
    
    def name(self) -> str:
        return f"One-Factor Copula (ρ={self.asset_correlation})"


def compute_tail_dependence(copula: Copula, n_simulations: int = 100000) -> Tuple[float, float]:
    """
    Estimate upper and lower tail dependence coefficients.
    
    Args:
        copula: Copula instance
        n_simulations: Number of simulations for estimation
        
    Returns:
        Tuple of (lower_tail_dependence, upper_tail_dependence)
    """
    # Generate bivariate sample
    U = copula.generate_uniforms(n_simulations, 2)
    
    # Estimate tail dependence at different quantile levels
    quantiles = [0.01, 0.02, 0.05]
    
    lower_deps = []
    upper_deps = []
    
    for q in quantiles:
        # Lower tail: P(U2 <= q | U1 <= q)
        mask_lower = U[:, 0] <= q
        if mask_lower.sum() > 0:
            lower_deps.append(np.mean(U[mask_lower, 1] <= q) * q / q)
        
        # Upper tail: P(U2 > 1-q | U1 > 1-q)
        mask_upper = U[:, 0] > (1 - q)
        if mask_upper.sum() > 0:
            upper_deps.append(np.mean(U[mask_upper, 1] > (1 - q)))
    
    lower_tail = np.mean(lower_deps) if lower_deps else 0
    upper_tail = np.mean(upper_deps) if upper_deps else 0
    
    return lower_tail, upper_tail
