#!/usr/bin/env python3
"""
QRM Assignment - Main Analysis Script

Value-at-Risk (VaR) and Expected Shortfall (ES) Estimation
for a Credit Portfolio of 100 Corporate Bonds

Authors: Baruth Jasper Matteo, Mühlemann Gian Ouray, Zen Davide
University of St. Gallen - Master's Program in Quantitative Economics and Finance
Autumn Semester 2025 - Quantitative Risk Management
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.portfolio import CreditPortfolio
from src.copulas import GaussianCopula, TCopula, FactorCopula
from src.simulation import MonteCarloSimulator, StressTestSimulator, compare_copulas
from src.risk_measures import (
    VaRCalculator, ESCalculator, 
    calculate_risk_measures, print_risk_report
)


def create_results_directory():
    """Create results directory if it doesn't exist."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def analyze_portfolio(portfolio: CreditPortfolio, results_dir: str):
    """Analyze and visualize the portfolio composition."""
    print("\n" + "="*60)
    print("PORTFOLIO ANALYSIS")
    print("="*60)
    
    # Portfolio summary
    print(f"\n{portfolio}")
    summary = portfolio.summary()
    print(f"\nTotal Exposure: ${summary['total_exposure']/1e6:.2f} Million")
    print(f"Expected Loss: ${summary['total_expected_loss']/1e6:.4f} Million")
    print(f"Average PD: {summary['avg_pd']*100:.3f}%")
    print(f"Average LGD: {summary['avg_lgd']*100:.2f}%")
    
    # Country breakdown
    print("\n--- Country Breakdown ---")
    country_df = portfolio.get_country_breakdown()
    print(country_df)
    
    # Rating breakdown
    print("\n--- Rating Breakdown ---")
    rating_df = portfolio.get_rating_breakdown()
    print(rating_df)
    
    # Save portfolio data
    portfolio_df = portfolio.to_dataframe()
    portfolio_df.to_csv(os.path.join(results_dir, 'portfolio_composition.csv'), index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Exposure by country
    ax1 = axes[0, 0]
    country_exposure = portfolio_df.groupby('Country')['Exposure_USD'].sum() / 1e6
    country_exposure.plot(kind='bar', ax=ax1, color=['#2ecc71', '#3498db'])
    ax1.set_title('Total Exposure by Country')
    ax1.set_ylabel('Exposure (Million USD)')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. Distribution by rating
    ax2 = axes[0, 1]
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    rating_counts = portfolio_df['Rating'].value_counts().reindex(rating_order)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rating_order)))
    rating_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Number of Counterparties by Rating')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Credit Rating')
    ax2.tick_params(axis='x', rotation=0)
    
    # 3. Exposure distribution
    ax3 = axes[1, 0]
    portfolio_df['Exposure_USD'].hist(bins=30, ax=ax3, color='#3498db', edgecolor='white')
    ax3.set_title('Exposure Distribution')
    ax3.set_xlabel('Exposure (USD)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(portfolio_df['Exposure_USD'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${portfolio_df["Exposure_USD"].mean()/1e6:.1f}M')
    ax3.legend()
    
    # 4. Expected loss by rating
    ax4 = axes[1, 1]
    el_by_rating = portfolio_df.groupby('Rating')['Expected_Loss'].sum().reindex(rating_order) / 1e6
    el_by_rating.plot(kind='bar', ax=ax4, color=colors)
    ax4.set_title('Expected Loss by Rating')
    ax4.set_ylabel('Expected Loss (Million USD)')
    ax4.set_xlabel('Credit Rating')
    ax4.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'portfolio_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return portfolio_df


def run_monte_carlo_simulation(portfolio: CreditPortfolio, copula, 
                                n_simulations: int, results_dir: str,
                                copula_name: str):
    """Run Monte Carlo simulation and calculate risk measures."""
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION - {copula.name()}")
    print(f"{'='*60}")
    print(f"Running {n_simulations:,} simulations...")
    
    # Run simulation
    simulator = MonteCarloSimulator(portfolio, copula, n_simulations)
    losses = simulator.run()
    
    # Calculate risk measures
    confidence_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
    risk_results = calculate_risk_measures(losses, confidence_levels)
    
    # Print report
    print_risk_report(risk_results, f"Risk Measures - {copula.name()}")
    
    # Get default statistics
    default_stats = simulator.get_default_statistics()
    print("\n--- Default Statistics ---")
    print(f"Mean number of defaults: {default_stats['mean_defaults']:.3f}")
    print(f"Std dev of defaults: {default_stats['std_defaults']:.3f}")
    print(f"Max defaults in single simulation: {default_stats['max_defaults']}")
    print(f"Probability of zero defaults: {default_stats['prob_zero_defaults']*100:.2f}%")
    print(f"Probability of multiple defaults: {default_stats['prob_multiple_defaults']*100:.2f}%")
    
    return losses, risk_results, default_stats


def plot_loss_distribution(losses_dict: dict, risk_results_dict: dict, 
                          results_dir: str):
    """Plot and compare loss distributions across copulas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'Gaussian': '#3498db', 't-Copula': '#e74c3c', 'Factor': '#2ecc71'}
    
    # 1. Loss distribution comparison
    ax1 = axes[0, 0]
    for name, losses in losses_dict.items():
        # Filter out zero losses for better visualization
        non_zero_losses = losses[losses > 0]
        if len(non_zero_losses) > 0:
            ax1.hist(non_zero_losses / 1e6, bins=50, alpha=0.5, 
                    label=name, color=colors.get(name, 'gray'), density=True)
    ax1.set_title('Loss Distribution Comparison (Non-Zero Losses)')
    ax1.set_xlabel('Loss (Million USD)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.set_xlim(left=0)
    
    # 2. Full distribution including zeros
    ax2 = axes[0, 1]
    for name, losses in losses_dict.items():
        percentiles = np.percentile(losses, np.arange(0, 101, 1))
        ax2.plot(np.arange(0, 101, 1), percentiles / 1e6, 
                label=name, color=colors.get(name, 'gray'), linewidth=2)
    ax2.set_title('Loss Distribution Percentiles')
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Loss (Million USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. VaR comparison
    ax3 = axes[1, 0]
    conf_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
    x = np.arange(len(conf_levels))
    width = 0.25
    
    for i, (name, results) in enumerate(risk_results_dict.items()):
        vars_values = [results['VaR'][cl] / 1e6 for cl in conf_levels]
        ax3.bar(x + i * width, vars_values, width, label=name, 
               color=colors.get(name, 'gray'))
    
    ax3.set_title('Value-at-Risk Comparison')
    ax3.set_ylabel('VaR (Million USD)')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'{cl*100:.1f}%' for cl in conf_levels])
    ax3.legend()
    ax3.set_xlabel('Confidence Level')
    
    # 4. ES comparison
    ax4 = axes[1, 1]
    for i, (name, results) in enumerate(risk_results_dict.items()):
        es_values = [results['ES'][cl] / 1e6 for cl in conf_levels]
        ax4.bar(x + i * width, es_values, width, label=name,
               color=colors.get(name, 'gray'))
    
    ax4.set_title('Expected Shortfall Comparison')
    ax4.set_ylabel('ES (Million USD)')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([f'{cl*100:.1f}%' for cl in conf_levels])
    ax4.legend()
    ax4.set_xlabel('Confidence Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_distribution_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_tail_comparison(losses_dict: dict, results_dir: str):
    """Plot tail behavior comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'Gaussian': '#3498db', 't-Copula': '#e74c3c', 'Factor': '#2ecc71'}
    
    # 1. Q-Q plot against normal distribution (using Gaussian as reference)
    ax1 = axes[0]
    for name, losses in losses_dict.items():
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        theoretical_quantiles = np.array([
            np.percentile(losses_dict['Gaussian'], 100 * (i + 0.5) / n) 
            for i in range(n)
        ])
        # Sample for plotting
        sample_idx = np.linspace(0, n-1, 1000).astype(int)
        ax1.scatter(theoretical_quantiles[sample_idx] / 1e6, 
                   sorted_losses[sample_idx] / 1e6,
                   alpha=0.5, label=name, color=colors.get(name, 'gray'), s=10)
    
    # Add 45-degree line
    max_val = max([l.max() for l in losses_dict.values()]) / 1e6
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='45° line')
    ax1.set_title('Q-Q Plot (vs Gaussian Copula)')
    ax1.set_xlabel('Gaussian Copula Quantiles (Million USD)')
    ax1.set_ylabel('Sample Quantiles (Million USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Exceedance probability (survival function)
    ax2 = axes[1]
    for name, losses in losses_dict.items():
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)
        exceedance_prob = 1 - np.arange(1, n + 1) / n
        
        # Sample for plotting
        sample_idx = np.linspace(0, n-1, 1000).astype(int)
        ax2.semilogy(sorted_losses[sample_idx] / 1e6, exceedance_prob[sample_idx],
                    label=name, color=colors.get(name, 'gray'), linewidth=2)
    
    ax2.set_title('Exceedance Probability (Log Scale)')
    ax2.set_xlabel('Loss (Million USD)')
    ax2.set_ylabel('P(Loss > x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-5, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tail_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def run_stress_tests(portfolio: CreditPortfolio, copula, 
                     n_simulations: int, results_dir: str):
    """Run stress test scenarios."""
    print("\n" + "="*60)
    print("STRESS TESTING")
    print("="*60)
    
    stress_simulator = StressTestSimulator(portfolio, copula, n_simulations)
    
    # Baseline
    baseline_losses = stress_simulator.run()
    baseline_var99 = VaRCalculator.historical_var(baseline_losses, 0.99)
    baseline_es99 = ESCalculator.historical_es(baseline_losses, 0.99)
    
    # Stress scenarios
    scenarios = {
        'Baseline': (baseline_losses, baseline_var99, baseline_es99),
    }
    
    # 1. PD stress (double PDs - economic downturn)
    print("\nRunning PD stress test (2x PDs)...")
    pd_stress_losses = stress_simulator.stress_test_pd_multiplier(2.0)
    pd_stress_var = VaRCalculator.historical_var(pd_stress_losses, 0.99)
    pd_stress_es = ESCalculator.historical_es(pd_stress_losses, 0.99)
    scenarios['PD x2'] = (pd_stress_losses, pd_stress_var, pd_stress_es)
    
    # 2. LGD stress (+15% LGD - collateral devaluation)
    print("Running LGD stress test (+15% LGD)...")
    lgd_stress_losses = stress_simulator.stress_test_lgd_increase(0.15)
    lgd_stress_var = VaRCalculator.historical_var(lgd_stress_losses, 0.99)
    lgd_stress_es = ESCalculator.historical_es(lgd_stress_losses, 0.99)
    scenarios['LGD +15%'] = (lgd_stress_losses, lgd_stress_var, lgd_stress_es)
    
    # 3. Correlation stress (increase correlation - systemic risk)
    print("Running correlation stress test (ρ=0.5)...")
    corr_stress_losses = stress_simulator.stress_test_correlation_increase(0.5)
    corr_stress_var = VaRCalculator.historical_var(corr_stress_losses, 0.99)
    corr_stress_es = ESCalculator.historical_es(corr_stress_losses, 0.99)
    scenarios['ρ = 0.5'] = (corr_stress_losses, corr_stress_var, corr_stress_es)
    
    # Print stress test results
    print("\n--- Stress Test Results (99% Confidence) ---")
    print(f"{'Scenario':<15} {'VaR (M USD)':<15} {'ES (M USD)':<15} {'VaR Change':<12} {'ES Change':<12}")
    print("-" * 70)
    
    for name, (losses, var99, es99) in scenarios.items():
        var_change = (var99 / baseline_var99 - 1) * 100 if name != 'Baseline' else 0
        es_change = (es99 / baseline_es99 - 1) * 100 if name != 'Baseline' else 0
        print(f"{name:<15} {var99/1e6:<15.2f} {es99/1e6:<15.2f} {var_change:>+10.1f}% {es_change:>+10.1f}%")
    
    # Plot stress test results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenario_names = list(scenarios.keys())
    var_values = [scenarios[s][1] / 1e6 for s in scenario_names]
    es_values = [scenarios[s][2] / 1e6 for s in scenario_names]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    # VaR comparison
    ax1 = axes[0]
    bars1 = ax1.bar(scenario_names, var_values, color=colors)
    ax1.set_title('99% VaR Under Stress Scenarios')
    ax1.set_ylabel('VaR (Million USD)')
    ax1.axhline(y=var_values[0], color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, var_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val:.1f}M', ha='center', va='bottom')
    
    # ES comparison
    ax2 = axes[1]
    bars2 = ax2.bar(scenario_names, es_values, color=colors)
    ax2.set_title('99% ES Under Stress Scenarios')
    ax2.set_ylabel('ES (Million USD)')
    ax2.axhline(y=es_values[0], color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars2, es_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stress_test_results.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return scenarios


def generate_summary_report(portfolio: CreditPortfolio, 
                           risk_results_dict: dict,
                           results_dir: str):
    """Generate a summary report."""
    report_path = os.path.join(results_dir, 'risk_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTITATIVE RISK MANAGEMENT ASSIGNMENT\n")
        f.write("Value-at-Risk and Expected Shortfall Estimation\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Authors: Baruth Jasper Matteo, Mühlemann Gian Ouray, Zen Davide\n")
        f.write("University of St. Gallen - Autumn Semester 2025\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PORTFOLIO SUMMARY\n")
        f.write("-"*70 + "\n")
        summary = portfolio.summary()
        f.write(f"Number of Counterparties: {summary['n_counterparties']}\n")
        f.write(f"Total Exposure: ${summary['total_exposure']/1e6:.2f} Million\n")
        f.write(f"Expected Loss: ${summary['total_expected_loss']/1e6:.4f} Million\n")
        f.write(f"Average PD: {summary['avg_pd']*100:.3f}%\n")
        f.write(f"Average LGD: {summary['avg_lgd']*100:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RISK MEASURES COMPARISON\n")
        f.write("-"*70 + "\n\n")
        
        for copula_name, results in risk_results_dict.items():
            f.write(f"=== {copula_name} ===\n")
            f.write(f"Mean Loss: ${results['summary']['mean']/1e6:.4f} Million\n")
            f.write(f"Std Dev: ${results['summary']['std']/1e6:.4f} Million\n\n")
            
            f.write("Value-at-Risk:\n")
            for cl, var in sorted(results['VaR'].items()):
                f.write(f"  VaR {cl*100:.1f}%: ${var/1e6:.4f} Million\n")
            
            f.write("\nExpected Shortfall:\n")
            for cl, es in sorted(results['ES'].items()):
                f.write(f"  ES {cl*100:.1f}%: ${es/1e6:.4f} Million\n")
            f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n")
        
        # Compare VaR 99% across copulas
        f.write("\n99% VaR Comparison:\n")
        for name, results in risk_results_dict.items():
            f.write(f"  {name}: ${results['VaR'][0.99]/1e6:.4f} Million\n")
        
        f.write("\n99% ES Comparison:\n")
        for name, results in risk_results_dict.items():
            f.write(f"  {name}: ${results['ES'][0.99]/1e6:.4f} Million\n")
        
        # Analysis of tail risk
        gaussian_es = risk_results_dict['Gaussian']['ES'][0.99]
        t_es = risk_results_dict['t-Copula']['ES'][0.99]
        tail_diff = (t_es - gaussian_es) / gaussian_es * 100
        
        f.write(f"\nTail Risk Analysis:\n")
        f.write(f"  The t-copula shows {tail_diff:.1f}% higher 99% ES compared to Gaussian,\n")
        f.write(f"  reflecting its ability to capture tail dependence.\n")
        
    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    """Main function to run the complete analysis."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#    QUANTITATIVE RISK MANAGEMENT ASSIGNMENT" + " "*25 + "#")
    print("#    VaR and ES Estimation for Credit Portfolio" + " "*21 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Configuration
    N_SIMULATIONS = 100000
    N_COUNTERPARTIES = 100
    CORRELATION = 0.3  # Asset correlation
    T_DF = 4  # Degrees of freedom for t-copula
    RANDOM_SEED = 42  # Seed passed to individual components for reproducibility
    
    # Create results directory
    results_dir = create_results_directory()
    print(f"\nResults will be saved to: {results_dir}")
    
    # Step 1: Create and analyze portfolio
    print("\n" + "="*60)
    print("STEP 1: PORTFOLIO CREATION")
    print("="*60)
    portfolio = CreditPortfolio(n_counterparties=N_COUNTERPARTIES, seed=RANDOM_SEED)
    portfolio_df = analyze_portfolio(portfolio, results_dir)
    
    # Step 2: Define copulas for comparison
    copulas = {
        'Gaussian': GaussianCopula(correlation=CORRELATION, seed=RANDOM_SEED),
        't-Copula': TCopula(correlation=CORRELATION, df=T_DF, seed=RANDOM_SEED + 1),
        'Factor': FactorCopula(asset_correlation=CORRELATION, seed=RANDOM_SEED + 2)
    }
    
    # Step 3: Run Monte Carlo simulations for each copula
    losses_dict = {}
    risk_results_dict = {}
    default_stats_dict = {}
    
    for name, copula in copulas.items():
        losses, risk_results, default_stats = run_monte_carlo_simulation(
            portfolio, copula, N_SIMULATIONS, results_dir, name
        )
        losses_dict[name] = losses
        risk_results_dict[name] = risk_results
        default_stats_dict[name] = default_stats
    
    # Step 4: Create comparison visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_loss_distribution(losses_dict, risk_results_dict, results_dir)
    print("Created: loss_distribution_comparison.png")
    
    plot_tail_comparison(losses_dict, results_dir)
    print("Created: tail_comparison.png")
    
    # Step 5: Run stress tests
    run_stress_tests(portfolio, GaussianCopula(correlation=CORRELATION, seed=RANDOM_SEED + 3),
                     N_SIMULATIONS, results_dir)
    print("Created: stress_test_results.png")
    
    # Step 6: Generate summary report
    generate_summary_report(portfolio, risk_results_dict, results_dir)
    
    # Save risk measures to CSV
    risk_df = pd.DataFrame({
        'Copula': [],
        'Confidence_Level': [],
        'VaR': [],
        'ES': [],
        'ES_VaR_Ratio': []
    })
    
    rows = []
    for copula_name, results in risk_results_dict.items():
        for cl in [0.90, 0.95, 0.99, 0.995, 0.999]:
            rows.append({
                'Copula': copula_name,
                'Confidence_Level': f'{cl*100:.1f}%',
                'VaR': results['VaR'][cl],
                'ES': results['ES'][cl],
                'ES_VaR_Ratio': results['ES_VaR_ratio'][cl]
            })
    
    risk_df = pd.DataFrame(rows)
    risk_df.to_csv(os.path.join(results_dir, 'risk_measures.csv'), index=False)
    print(f"Saved: risk_measures.csv")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {results_dir}/")
    print("\nGenerated files:")
    print("  - portfolio_composition.csv")
    print("  - portfolio_analysis.png")
    print("  - loss_distribution_comparison.png")
    print("  - tail_comparison.png")
    print("  - stress_test_results.png")
    print("  - risk_measures.csv")
    print("  - risk_report.txt")
    
    print("\n" + "="*60)
    print("KEY RESULTS SUMMARY")
    print("="*60)
    print(f"\nPortfolio: {N_COUNTERPARTIES} counterparties")
    print(f"Total Exposure: ${portfolio.get_total_exposure()/1e6:.2f} Million")
    print(f"Simulations per copula: {N_SIMULATIONS:,}")
    
    print("\n99% VaR (Million USD):")
    for name, results in risk_results_dict.items():
        print(f"  {name}: ${results['VaR'][0.99]/1e6:.4f}M")
    
    print("\n99% ES (Million USD):")
    for name, results in risk_results_dict.items():
        print(f"  {name}: ${results['ES'][0.99]/1e6:.4f}M")


if __name__ == "__main__":
    main()
