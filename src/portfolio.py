"""
Credit Portfolio Module

This module defines the credit portfolio structure with 100 counterparties
representing corporate bonds from Swiss and US firms.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Counterparty:
    """
    Represents a single counterparty in the credit portfolio.
    
    Attributes:
        id: Unique identifier for the counterparty
        name: Name of the counterparty
        country: Country of operations (Switzerland or United States)
        exposure: Exposure at default (EAD) in USD
        probability_of_default: Probability of default (PD)
        loss_given_default: Loss given default (LGD)
        rating: Credit rating
    """
    id: int
    name: str
    country: str
    exposure: float
    probability_of_default: float
    loss_given_default: float
    rating: str


class CreditPortfolio:
    """
    Credit Portfolio consisting of 100 counterparties.
    
    The portfolio contains corporate bonds of firms with operations in 
    Switzerland and the United States.
    """
    
    # Credit rating categories and their default probabilities (based on historical data)
    RATING_PD = {
        'AAA': 0.0001,
        'AA': 0.0002,
        'A': 0.0005,
        'BBB': 0.002,
        'BB': 0.01,
        'B': 0.04,
        'CCC': 0.15
    }
    
    # Average LGD by rating (senior unsecured bonds)
    RATING_LGD = {
        'AAA': 0.35,
        'AA': 0.38,
        'A': 0.40,
        'BBB': 0.42,
        'BB': 0.45,
        'B': 0.50,
        'CCC': 0.60
    }
    
    def __init__(self, n_counterparties: int = 100, seed: Optional[int] = 42):
        """
        Initialize the credit portfolio.
        
        Args:
            n_counterparties: Number of counterparties (default: 100)
            seed: Random seed for reproducibility
        """
        self.n_counterparties = n_counterparties
        self.seed = seed
        self.counterparties: List[Counterparty] = []
        self._generate_portfolio()
        
    def _generate_portfolio(self) -> None:
        """Generate the portfolio with counterparties from Switzerland and US."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Distribution of counterparties by country (60% US, 40% Switzerland)
        n_us = int(self.n_counterparties * 0.6)
        n_ch = self.n_counterparties - n_us
        
        # Rating distribution (investment grade heavy)
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        rating_probs = [0.05, 0.10, 0.25, 0.30, 0.15, 0.10, 0.05]
        
        # Generate counterparties
        for i in range(self.n_counterparties):
            country = 'United States' if i < n_us else 'Switzerland'
            rating = np.random.choice(ratings, p=rating_probs)
            
            # Exposure between 1M and 50M USD (log-normal distribution)
            exposure = np.exp(np.random.normal(16, 1.2))  # ~10M average
            exposure = np.clip(exposure, 1e6, 5e7)
            
            counterparty = Counterparty(
                id=i + 1,
                name=f"Corp_{country[:2]}_{i+1:03d}",
                country=country,
                exposure=exposure,
                probability_of_default=self.RATING_PD[rating],
                loss_given_default=self.RATING_LGD[rating],
                rating=rating
            )
            self.counterparties.append(counterparty)
    
    def get_exposures(self) -> np.ndarray:
        """Get array of exposures for all counterparties."""
        return np.array([c.exposure for c in self.counterparties])
    
    def get_pds(self) -> np.ndarray:
        """Get array of default probabilities for all counterparties."""
        return np.array([c.probability_of_default for c in self.counterparties])
    
    def get_lgds(self) -> np.ndarray:
        """Get array of loss given defaults for all counterparties."""
        return np.array([c.loss_given_default for c in self.counterparties])
    
    def get_expected_losses(self) -> np.ndarray:
        """Calculate expected loss for each counterparty: EL = EAD * PD * LGD."""
        return self.get_exposures() * self.get_pds() * self.get_lgds()
    
    def get_total_expected_loss(self) -> float:
        """Calculate total expected loss for the portfolio."""
        return np.sum(self.get_expected_losses())
    
    def get_total_exposure(self) -> float:
        """Calculate total exposure of the portfolio."""
        return np.sum(self.get_exposures())
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to pandas DataFrame."""
        data = {
            'ID': [c.id for c in self.counterparties],
            'Name': [c.name for c in self.counterparties],
            'Country': [c.country for c in self.counterparties],
            'Exposure_USD': [c.exposure for c in self.counterparties],
            'PD': [c.probability_of_default for c in self.counterparties],
            'LGD': [c.loss_given_default for c in self.counterparties],
            'Rating': [c.rating for c in self.counterparties],
            'Expected_Loss': self.get_expected_losses()
        }
        return pd.DataFrame(data)
    
    def get_country_breakdown(self) -> pd.DataFrame:
        """Get portfolio breakdown by country."""
        df = self.to_dataframe()
        return df.groupby('Country').agg({
            'ID': 'count',
            'Exposure_USD': 'sum',
            'Expected_Loss': 'sum'
        }).rename(columns={'ID': 'Count'})
    
    def get_rating_breakdown(self) -> pd.DataFrame:
        """Get portfolio breakdown by rating."""
        df = self.to_dataframe()
        return df.groupby('Rating').agg({
            'ID': 'count',
            'Exposure_USD': 'sum',
            'Expected_Loss': 'sum',
            'PD': 'mean',
            'LGD': 'mean'
        }).rename(columns={'ID': 'Count'})
    
    def summary(self) -> dict:
        """Get portfolio summary statistics."""
        return {
            'n_counterparties': self.n_counterparties,
            'total_exposure': self.get_total_exposure(),
            'total_expected_loss': self.get_total_expected_loss(),
            'avg_pd': np.mean(self.get_pds()),
            'avg_lgd': np.mean(self.get_lgds()),
            'exposure_range': (np.min(self.get_exposures()), np.max(self.get_exposures()))
        }
    
    def __repr__(self) -> str:
        summary = self.summary()
        return (f"CreditPortfolio(n={summary['n_counterparties']}, "
                f"total_exposure=${summary['total_exposure']/1e6:.1f}M, "
                f"expected_loss=${summary['total_expected_loss']/1e6:.3f}M)")
