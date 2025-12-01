"""
Risk Measures Module

Implementation of Value-at-Risk (VaR) and Expected Shortfall (ES)
for credit portfolio loss distributions.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict, List


class VaRCalculator:
    """
    Value-at-Risk (VaR) Calculator.
    
    VaR represents the maximum loss that will not be exceeded with a given
    probability (confidence level) over a specified time horizon.
    """
    
    @staticmethod
    def historical_var(losses: np.ndarray, confidence_level: float = 0.99) -> float:
        """
        Calculate VaR using historical simulation method.
        
        Args:
            losses: Array of simulated losses (positive values = losses)
            confidence_level: Confidence level (e.g., 0.99 for 99% VaR)
            
        Returns:
            VaR at the specified confidence level
        """
        return np.percentile(losses, confidence_level * 100)
    
    @staticmethod
    def parametric_var_normal(mean: float, std: float, 
                               confidence_level: float = 0.99) -> float:
        """
        Calculate VaR assuming normal distribution.
        
        Args:
            mean: Mean of the loss distribution
            std: Standard deviation of the loss distribution
            confidence_level: Confidence level
            
        Returns:
            VaR at the specified confidence level
        """
        z_score = stats.norm.ppf(confidence_level)
        return mean + z_score * std
    
    @staticmethod
    def parametric_var_t(mean: float, std: float, df: int,
                         confidence_level: float = 0.99) -> float:
        """
        Calculate VaR assuming Student-t distribution.
        
        Args:
            mean: Mean of the loss distribution
            std: Standard deviation of the loss distribution
            df: Degrees of freedom
            confidence_level: Confidence level
            
        Returns:
            VaR at the specified confidence level
        """
        t_score = stats.t.ppf(confidence_level, df)
        # Adjust for t-distribution scaling
        scale = std * np.sqrt((df - 2) / df) if df > 2 else std
        return mean + t_score * scale
    
    @staticmethod
    def calculate_multiple_vars(losses: np.ndarray, 
                                 confidence_levels: List[float] = None) -> Dict[float, float]:
        """
        Calculate VaR at multiple confidence levels.
        
        Args:
            losses: Array of simulated losses
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence levels to VaR values
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
        
        return {cl: VaRCalculator.historical_var(losses, cl) 
                for cl in confidence_levels}


class ESCalculator:
    """
    Expected Shortfall (ES) Calculator.
    
    ES (also known as Conditional VaR or CVaR) represents the expected loss
    given that the loss exceeds the VaR threshold. It is a coherent risk
    measure and provides information about tail risk beyond VaR.
    """
    
    @staticmethod
    def historical_es(losses: np.ndarray, confidence_level: float = 0.99) -> float:
        """
        Calculate ES using historical simulation method.
        
        ES = E[Loss | Loss > VaR]
        
        Args:
            losses: Array of simulated losses (positive values = losses)
            confidence_level: Confidence level (e.g., 0.99 for 99% ES)
            
        Returns:
            ES at the specified confidence level
        """
        var = VaRCalculator.historical_var(losses, confidence_level)
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) == 0:
            return var
        
        return np.mean(tail_losses)
    
    @staticmethod
    def parametric_es_normal(mean: float, std: float,
                             confidence_level: float = 0.99) -> float:
        """
        Calculate ES assuming normal distribution.
        
        For normal distribution: ES = μ + σ * φ(z) / (1 - α)
        where φ is the standard normal pdf and z is the α-quantile.
        
        Args:
            mean: Mean of the loss distribution
            std: Standard deviation of the loss distribution
            confidence_level: Confidence level
            
        Returns:
            ES at the specified confidence level
        """
        z = stats.norm.ppf(confidence_level)
        phi_z = stats.norm.pdf(z)
        return mean + std * phi_z / (1 - confidence_level)
    
    @staticmethod
    def parametric_es_t(mean: float, std: float, df: int,
                        confidence_level: float = 0.99) -> float:
        """
        Calculate ES assuming Student-t distribution.
        
        Args:
            mean: Mean of the loss distribution
            std: Standard deviation of the loss distribution
            df: Degrees of freedom
            confidence_level: Confidence level
            
        Returns:
            ES at the specified confidence level
        """
        t_quantile = stats.t.ppf(confidence_level, df)
        t_pdf = stats.t.pdf(t_quantile, df)
        
        # Scale factor for t-distribution
        scale = std * np.sqrt((df - 2) / df) if df > 2 else std
        
        # ES formula for t-distribution
        factor = (df + t_quantile**2) / (df - 1)
        es = mean + scale * t_pdf / (1 - confidence_level) * factor
        
        return es
    
    @staticmethod
    def calculate_multiple_es(losses: np.ndarray,
                               confidence_levels: List[float] = None) -> Dict[float, float]:
        """
        Calculate ES at multiple confidence levels.
        
        Args:
            losses: Array of simulated losses
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence levels to ES values
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
        
        return {cl: ESCalculator.historical_es(losses, cl) 
                for cl in confidence_levels}


def calculate_risk_measures(losses: np.ndarray, 
                            confidence_levels: List[float] = None) -> Dict:
    """
    Calculate comprehensive risk measures for a loss distribution.
    
    Args:
        losses: Array of simulated losses
        confidence_levels: List of confidence levels to calculate
        
    Returns:
        Dictionary containing VaR, ES, and summary statistics
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
    
    # Basic statistics
    summary = {
        'mean': np.mean(losses),
        'std': np.std(losses),
        'median': np.median(losses),
        'min': np.min(losses),
        'max': np.max(losses),
        'skewness': stats.skew(losses),
        'kurtosis': stats.kurtosis(losses)
    }
    
    # VaR at multiple levels
    var_results = VaRCalculator.calculate_multiple_vars(losses, confidence_levels)
    
    # ES at multiple levels
    es_results = ESCalculator.calculate_multiple_es(losses, confidence_levels)
    
    # ES/VaR ratio (measure of tail heaviness)
    es_var_ratios = {cl: es_results[cl] / var_results[cl] if var_results[cl] != 0 else np.inf
                    for cl in confidence_levels}
    
    return {
        'summary': summary,
        'VaR': var_results,
        'ES': es_results,
        'ES_VaR_ratio': es_var_ratios,
        'n_simulations': len(losses)
    }


def print_risk_report(results: Dict, title: str = "Risk Measures Report") -> None:
    """
    Print a formatted risk measures report.
    
    Args:
        results: Dictionary from calculate_risk_measures
        title: Report title
    """
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    print("\n--- Summary Statistics ---")
    summary = results['summary']
    print(f"Mean Loss:      ${summary['mean']:,.2f}")
    print(f"Std Dev:        ${summary['std']:,.2f}")
    print(f"Median Loss:    ${summary['median']:,.2f}")
    print(f"Min Loss:       ${summary['min']:,.2f}")
    print(f"Max Loss:       ${summary['max']:,.2f}")
    print(f"Skewness:       {summary['skewness']:.3f}")
    print(f"Excess Kurtosis: {summary['kurtosis']:.3f}")
    
    print("\n--- Value-at-Risk (VaR) ---")
    for cl, var in sorted(results['VaR'].items()):
        print(f"VaR {cl*100:.1f}%:      ${var:,.2f}")
    
    print("\n--- Expected Shortfall (ES) ---")
    for cl, es in sorted(results['ES'].items()):
        print(f"ES {cl*100:.1f}%:       ${es:,.2f}")
    
    print("\n--- ES/VaR Ratios ---")
    for cl, ratio in sorted(results['ES_VaR_ratio'].items()):
        print(f"ES/VaR {cl*100:.1f}%:   {ratio:.3f}")
    
    print(f"\nBased on {results['n_simulations']:,} simulations")
    print("=" * 60)
