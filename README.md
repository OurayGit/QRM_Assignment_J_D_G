# Quantitative Risk Management Assignment

**Master's Program in Quantitative Economics and Finance**  
**University of St. Gallen - Autumn Semester 2025**

**Authors:** Baruth Jasper Matteo, Mühlemann Gian Ouray, Zen Davide

## Overview

This project implements Value-at-Risk (VaR) and Expected Shortfall (ES) estimation for a credit portfolio of 100 corporate bonds from firms with operations in Switzerland and the United States.

The goal is to apply theoretical results on copulas and risk measures to assess the risk of the investment portfolio.

## Portfolio Description

The credit portfolio consists of:
- **100 counterparties** representing corporate bonds
- **Geographic distribution:** 60% United States, 40% Switzerland  
- **Credit ratings:** AAA through CCC (investment grade heavy)
- **Exposures:** $1M - $50M per counterparty (log-normal distribution)

## Methodology

### Risk Measures

1. **Value-at-Risk (VaR):** Maximum loss that will not be exceeded with a given probability
2. **Expected Shortfall (ES):** Expected loss given that the loss exceeds VaR (tail risk measure)

### Dependency Modeling (Copulas)

The project implements three copula models for capturing default dependency:

1. **Gaussian Copula:** Standard model with no tail dependence
2. **Student-t Copula:** Captures tail dependence for joint extreme events
3. **One-Factor (Vasicek) Copula:** Basel II/III regulatory model

### Monte Carlo Simulation

- 100,000 simulations per copula model
- Correlation parameter: ρ = 0.3
- t-copula degrees of freedom: ν = 4

## Project Structure

```
QRM_Assignment_J_D_G/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Main analysis script
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── portfolio.py          # Credit portfolio definition
│   ├── copulas.py            # Copula implementations
│   ├── risk_measures.py      # VaR and ES calculators
│   └── simulation.py         # Monte Carlo simulation
├── data/                     # Input data (if any)
└── results/                  # Output results
    ├── portfolio_composition.csv
    ├── portfolio_analysis.png
    ├── loss_distribution_comparison.png
    ├── tail_comparison.png
    ├── stress_test_results.png
    ├── risk_measures.csv
    └── risk_report.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OurayGit/QRM_Assignment_J_D_G.git
cd QRM_Assignment_J_D_G
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis:
```bash
python main.py
```

This will:
1. Generate the credit portfolio
2. Run Monte Carlo simulations with different copulas
3. Calculate VaR and ES at multiple confidence levels
4. Perform stress tests
5. Generate visualizations and reports

## Results

The analysis produces:

### Risk Measures at 99% Confidence Level

| Copula | VaR | ES |
|--------|-----|-----|
| Gaussian | See results | See results |
| t-Copula | See results | See results |
| Factor | See results | See results |

### Key Findings

1. **Tail Dependence:** The t-copula produces higher ES values due to its ability to capture simultaneous defaults
2. **Stress Testing:** Correlation stress shows the largest impact on tail risk
3. **Model Comparison:** Gaussian copula underestimates tail risk compared to t-copula

## Stress Test Scenarios

1. **PD Stress (x2):** Doubling default probabilities (economic downturn)
2. **LGD Stress (+15%):** Increased loss severity (collateral devaluation)
3. **Correlation Stress (ρ=0.5):** Increased dependence (systemic risk)

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## References

- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press.
- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards*.

## License

This project is for academic purposes as part of the QRM course at University of St. Gallen.
