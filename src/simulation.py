"""
Monte Carlo Simulation Module

Implementation of Monte Carlo simulation for credit portfolio losses.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats

from .portfolio import CreditPortfolio
from .copulas import Copula, GaussianCopula
from .risk_measures import calculate_risk_measures


class MonteCarloSimulator:
    """
    Monte Carlo Simulator for Credit Portfolio Losses.
    
    This simulator generates loss scenarios for a credit portfolio using
    copula-based dependency modeling.
    """
    
    def __init__(self, portfolio: CreditPortfolio, copula: Copula,
                 n_simulations: int = 100000, seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            portfolio: CreditPortfolio instance
            copula: Copula instance for dependency modeling
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.portfolio = portfolio
        self.copula = copula
        self.n_simulations = n_simulations
        self.seed = seed
        
        # Pre-compute portfolio parameters
        self.exposures = portfolio.get_exposures()
        self.pds = portfolio.get_pds()
        self.lgds = portfolio.get_lgds()
        self.n_counterparties = portfolio.n_counterparties
        
        # Results storage
        self.losses = None
        self.defaults = None
        
    def simulate_defaults(self) -> np.ndarray:
        """
        Simulate default events using copula dependency.
        
        Returns:
            Boolean array of shape (n_simulations, n_counterparties)
            True indicates default, False indicates no default
        """
        # Generate correlated uniforms from copula
        U = self.copula.generate_uniforms(self.n_simulations, self.n_counterparties)
        
        # Transform PDs to default thresholds
        # A counterparty defaults if U <= PD
        # This is equivalent to comparing normalized asset values to default barrier
        defaults = U <= self.pds
        
        return defaults
    
    def simulate_losses(self) -> np.ndarray:
        """
        Simulate portfolio losses.
        
        Returns:
            Array of shape (n_simulations,) containing total portfolio loss
            for each simulation
        """
        # Simulate defaults
        self.defaults = self.simulate_defaults()
        
        # Calculate individual losses: Loss = Default * EAD * LGD
        individual_losses = self.defaults * self.exposures * self.lgds
        
        # Sum across counterparties to get portfolio loss
        self.losses = np.sum(individual_losses, axis=1)
        
        return self.losses
    
    def run(self) -> np.ndarray:
        """
        Run the full Monte Carlo simulation.
        
        Returns:
            Array of simulated portfolio losses
        """
        return self.simulate_losses()
    
    def get_default_statistics(self) -> Dict:
        """
        Get statistics on default events.
        
        Returns:
            Dictionary with default statistics
        """
        if self.defaults is None:
            self.simulate_defaults()
        
        n_defaults_per_sim = np.sum(self.defaults, axis=1)
        
        return {
            'mean_defaults': np.mean(n_defaults_per_sim),
            'std_defaults': np.std(n_defaults_per_sim),
            'max_defaults': np.max(n_defaults_per_sim),
            'min_defaults': np.min(n_defaults_per_sim),
            'prob_zero_defaults': np.mean(n_defaults_per_sim == 0),
            'prob_multiple_defaults': np.mean(n_defaults_per_sim > 1),
            'default_rate_by_counterparty': np.mean(self.defaults, axis=0)
        }
    
    def get_loss_distribution(self) -> Dict:
        """
        Get statistics on the loss distribution.
        
        Returns:
            Dictionary with loss distribution statistics
        """
        if self.losses is None:
            self.simulate_losses()
        
        return {
            'mean': np.mean(self.losses),
            'std': np.std(self.losses),
            'median': np.median(self.losses),
            'skewness': stats.skew(self.losses),
            'kurtosis': stats.kurtosis(self.losses),
            'min': np.min(self.losses),
            'max': np.max(self.losses),
            'percentiles': {
                1: np.percentile(self.losses, 1),
                5: np.percentile(self.losses, 5),
                25: np.percentile(self.losses, 25),
                50: np.percentile(self.losses, 50),
                75: np.percentile(self.losses, 75),
                95: np.percentile(self.losses, 95),
                99: np.percentile(self.losses, 99)
            }
        }


class StressTestSimulator(MonteCarloSimulator):
    """
    Extended simulator with stress testing capabilities.
    """
    
    def __init__(self, portfolio: CreditPortfolio, copula: Copula,
                 n_simulations: int = 100000, seed: Optional[int] = None):
        super().__init__(portfolio, copula, n_simulations, seed)
        
    def stress_test_pd_multiplier(self, multiplier: float) -> np.ndarray:
        """
        Run stress test with increased default probabilities.
        
        Args:
            multiplier: Factor to multiply PDs by (e.g., 2.0 doubles PDs)
            
        Returns:
            Array of stressed portfolio losses
        """
        # Store original PDs
        original_pds = self.pds.copy()
        
        # Apply stress
        self.pds = np.minimum(original_pds * multiplier, 0.999)
        
        # Run simulation
        stressed_losses = self.simulate_losses()
        
        # Restore original PDs
        self.pds = original_pds
        
        return stressed_losses
    
    def stress_test_lgd_increase(self, increase: float) -> np.ndarray:
        """
        Run stress test with increased LGD values.
        
        Args:
            increase: Amount to add to LGDs (e.g., 0.1 adds 10%)
            
        Returns:
            Array of stressed portfolio losses
        """
        # Store original LGDs
        original_lgds = self.lgds.copy()
        
        # Apply stress
        self.lgds = np.minimum(original_lgds + increase, 1.0)
        
        # Run simulation
        stressed_losses = self.simulate_losses()
        
        # Restore original LGDs
        self.lgds = original_lgds
        
        return stressed_losses
    
    def stress_test_correlation_increase(self, correlation: float) -> np.ndarray:
        """
        Run stress test with increased correlation.
        
        Args:
            correlation: New correlation value
            
        Returns:
            Array of stressed portfolio losses
        """
        # Create new copula with increased correlation
        if hasattr(self.copula, 'correlation'):
            original_correlation = self.copula.correlation
            self.copula.correlation = correlation
            
            # Run simulation
            stressed_losses = self.simulate_losses()
            
            # Restore original correlation
            self.copula.correlation = original_correlation
        else:
            # For factor copulas
            original_correlation = self.copula.asset_correlation
            self.copula.asset_correlation = correlation
            
            # Run simulation
            stressed_losses = self.simulate_losses()
            
            # Restore
            self.copula.asset_correlation = original_correlation
        
        return stressed_losses


def compare_copulas(portfolio: CreditPortfolio, copulas: list,
                    n_simulations: int = 100000) -> Dict:
    """
    Compare risk measures across different copulas.
    
    Args:
        portfolio: CreditPortfolio instance
        copulas: List of Copula instances to compare
        n_simulations: Number of simulations per copula
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for copula in copulas:
        simulator = MonteCarloSimulator(portfolio, copula, n_simulations)
        losses = simulator.run()
        risk_measures = calculate_risk_measures(losses)
        
        results[copula.name()] = {
            'losses': losses,
            'risk_measures': risk_measures,
            'default_stats': simulator.get_default_statistics()
        }
    
    return results
