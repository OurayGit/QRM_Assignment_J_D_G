"""
QRM Assignment - Credit Portfolio Risk Analysis
Value-at-Risk (VaR) and Expected Shortfall (ES) Estimation

Authors: Baruth Jasper Matteo, Mühlemann Gian Ouray, Zen Davide
University of St. Gallen - Master's Program in Quantitative Economics and Finance
Autumn Semester 2025
"""

from .portfolio import CreditPortfolio
from .risk_measures import VaRCalculator, ESCalculator
from .copulas import GaussianCopula, TCopula
from .simulation import MonteCarloSimulator

__all__ = [
    'CreditPortfolio',
    'VaRCalculator',
    'ESCalculator',
    'GaussianCopula',
    'TCopula',
    'MonteCarloSimulator'
]

__version__ = '1.0.0'
