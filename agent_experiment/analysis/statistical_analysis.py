import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SpaceSettlementAnalyzer:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def analyze_solution_impact(self, with_solution_data: pd.DataFrame, without_solution_data: pd.DataFrame):
        """Analyze solution impact with improved statistical metrics"""
        results = {}
        
        stress_before = without_solution_data['stress_level'].mean()
        stress_after = with_solution_data['stress_level'].mean()
        stress_improvement = ((stress_before - stress_after) / stress_before * 100) if stress_before > 0 else 0
        
        if len(with_solution_data) >= 3 and len(without_solution_data) >= 3:
            t_stat, p_val = stats.ttest_ind(
                with_solution_data['stress_level'],
                without_solution_data['stress_level'],
                equal_var=False
            )
            effect_size = (stress_before - stress_after) / (without_solution_data['stress_level'].std() or 1)
        else:
            t_stat, p_val, effect_size = np.nan, np.nan, np.nan
        
        results['stress_analysis'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'improvement_percent': stress_improvement
        }
        
        social_before = without_solution_data['social_connections'].mean()
        social_after = with_solution_data['social_connections'].mean()
        
        max_possible_connections = len(with_solution_data) - 1
        density_before = social_before / max_possible_connections if max_possible_connections > 0 else 0
        density_after = social_after / max_possible_connections if max_possible_connections > 0 else 0
        density_improvement = ((density_after - density_before) / (density_before or 1)) * 100
        
        results['social_network'] = {
            'avg_connections_with': social_after,
            'avg_connections_without': social_before,
            'density_improvement': density_improvement,
            'network_density_before': density_before,
            'network_density_after': density_after
        }
        
        adaptation_before = without_solution_data['adaptation_score'].mean()
        adaptation_after = with_solution_data['adaptation_score'].mean()
        
        adaptation_improvement = (
            ((adaptation_after - adaptation_before) / (adaptation_before or 1)) * 100
        )
        
        if len(with_solution_data) >= 3 and len(without_solution_data) >= 3:
            t_stat, p_val = stats.ttest_ind(
                with_solution_data['adaptation_score'],
                without_solution_data['adaptation_score'],
                equal_var=False
            )
        else:
            t_stat, p_val = np.nan, np.nan
        
        results['adaptation_analysis'] = {
            'improvement_percent': adaptation_improvement,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_before': adaptation_before,
            'mean_after': adaptation_after
        }
        
        return results

    def generate_solution_report(self, analysis_results: dict) -> str:
        """Generate a detailed report of solution effectiveness"""
        report = []
        report.append("Space Settlement Solution Impact Analysis")
        report.append("=======================================\n")
        
        report.append("1. Stress Level Impact")
        report.append("-----------------------")
        stress = analysis_results['stress_analysis']
        report.append(f"• Stress Reduction: {stress.get('improvement_percent', 0):.1f}%")
        report.append(f"• Statistical Significance: p={stress.get('p_value', 'N/A')}")
        report.append(f"• Effect Size (Cohen's d): {stress.get('effect_size', 'N/A')}")
        report.append("")
        
        report.append("2. Adaptation Improvements")
        report.append("-------------------------")
        adapt = analysis_results['adaptation_analysis']
        report.append(f"• Adaptation Score Increase: {adapt.get('improvement_percent', 0):.2f}% improvement")
        report.append(f"• Mean Score Before: {adapt.get('mean_before', 0):.2f}")
        report.append(f"• Mean Score After: {adapt.get('mean_after', 0):.2f}")
        report.append(f"• Statistical Significance: p={adapt.get('p_value', 'N/A')}")
        report.append("")
        
        report.append("3. Social Network Impact")
        report.append("----------------------")
        social = analysis_results['social_network']
        report.append(f"• Network Density Improvement: {social.get('density_improvement', 0):.1f}%")
        report.append(f"• Average Connections (With Solution): {social.get('avg_connections_with', 0):.2f}")
        report.append(f"• Average Connections (Without Solution): {social.get('avg_connections_without', 0):.2f}")
        report.append(f"• Network Density (Before): {social.get('network_density_before', 0):.2f}")
        report.append(f"• Network Density (After): {social.get('network_density_after', 0):.2f}")
        
        return "\n".join(report)

    def plot_solution_impact(self, with_solution_data: pd.DataFrame, without_solution_data: pd.DataFrame):
        """Create publication-quality visualizations"""
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_stress_levels(ax1, with_solution_data, without_solution_data)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_adaptation(ax2, with_solution_data, without_solution_data)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_social_network(ax3, with_solution_data, without_solution_data)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_performance(ax4, with_solution_data, without_solution_data)
        
        plt.savefig(self.results_dir / 'solution_impact_analysis.pdf', bbox_inches='tight')
        plt.close()

    def _plot_stress_levels(self, ax, with_data, without_data):
        combined_data = pd.concat([
            with_data.assign(Condition='With Solution'),
            without_data.assign(Condition='Without Solution')
        ])
        
        sns.violinplot(data=combined_data, x='Condition', y='stress_level',
                      inner='box', ax=ax)
        
        ax.set_title('Stress Level Distribution', fontsize=12, pad=20)
        ax.set_ylabel('Stress Level')
        
        t_stat, p_val = stats.ttest_ind(
            with_data['stress_level'],
            without_data['stress_level']
        )
        ax.text(0.05, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes)

    def _plot_adaptation(self, ax, with_data, without_data):
        combined_data = pd.concat([
            with_data.assign(Condition='With Solution'),
            without_data.assign(Condition='Without Solution')
        ])
        
        sns.scatterplot(data=combined_data, 
                       x='stress_level', 
                       y='adaptation_score',
                       hue='Condition',
                       size='social_connections',
                       sizes=(50, 200),
                       alpha=0.6,
                       ax=ax)
        
        ax.set_title('Stress vs Adaptation', fontsize=12, pad=20)
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Adaptation Score')

    def _plot_social_network(self, ax, with_data, without_data):
        combined_data = pd.concat([
            with_data.assign(Condition='With Solution'),
            without_data.assign(Condition='Without Solution')
        ])
        
        sns.boxplot(data=combined_data,
                   x='Condition',
                   y='social_connections',
                   ax=ax)
        
        ax.set_title('Social Network Analysis', fontsize=12, pad=20)
        ax.set_ylabel('Number of Connections')

    def _plot_performance(self, ax, with_data, without_data):
        metrics = pd.DataFrame([
            {
                'Metric': metric,
                'With': with_data[col].mean(),
                'Without': without_data[col].mean()
            }
            for metric, col in [
                ('Success Rate', 'success_rate'),
                ('Adaptation', 'adaptation_score'),
                ('Social', 'social_connections')
            ]
        ])
        
        metrics_melted = pd.melt(metrics, 
                                id_vars=['Metric'],
                                var_name='Condition',
                                value_name='Score')
        
        sns.barplot(data=metrics_melted,
                   x='Metric',
                   y='Score',
                   hue='Condition',
                   ax=ax)
        
        ax.set_title('Performance Metrics', fontsize=12, pad=20)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
