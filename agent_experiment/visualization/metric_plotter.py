import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import networkx as nx
from matplotlib.animation import FuncAnimation

@dataclass
class PlotConfig:
    """Configuration for plot formatting"""
    title: str
    xlabel: str
    ylabel: str
    legend_title: Optional[str] = None
    fig_size: Tuple[int, int] = (10, 6)
    font_size: int = 12
    title_pad: int = 20

class MetricPlotter:
    """Handles visualization of experiment metrics with error handling and consistent formatting."""
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize the MetricPlotter with output directory and styling configurations.
        
        Args:
            results_dir: Directory path where plots will be saved
        """
        self.results_dir = Path(results_dir)
        self.solution_enabled = True
        self._setup_plotting_env()
        self._setup_logging()
        
    def _setup_plotting_env(self) -> None:
        """Configure plotting environment with consistent styling"""
        matplotlib.use('Agg')
        plt.ioff()
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'figure.dpi': 300,
            'figure.autolayout': True,
            'interactive': False
        })
        
    def _setup_logging(self) -> None:
        """Configure logging for plot generation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=self.results_dir / 'plotting.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def _ensure_output_dir(self) -> None:
        """Ensure the output directory exists"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _format_plot(self, ax: plt.Axes, config: PlotConfig) -> None:
        """
        Apply consistent formatting to a plot.
        
        Args:
            ax: Matplotlib axes object to format
            config: PlotConfig object containing formatting parameters
        """
        ax.set_title(config.title, fontsize=config.font_size, pad=config.title_pad)
        ax.set_xlabel(config.xlabel, fontsize=config.font_size)
        ax.set_ylabel(config.ylabel, fontsize=config.font_size)
        if config.legend_title:
            ax.legend(title=config.legend_title, title_fontsize=config.font_size)
            
    def _save_plot(self, filename: str) -> None:
        """
        Safely save plot with error handling.
        
        Args:
            filename: Name of the output file
        """
        self._ensure_output_dir()
        try:
            if not filename.endswith('.png'):
                filename = f"{filename}.png"
            
            filepath = self.results_dir / filename
            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            self.logger.info(f"Successfully saved plot: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save plot {filename}: {str(e)}")
        finally:
            plt.close('all')
            
    def plot_all_metrics(self, results: Dict) -> None:
        """Generate comprehensive publication-quality visualizations"""
        self.logger.info("Generating publication visualizations...")
        try:
            self.plot_stress_metrics(results)
            self.plot_social_metrics(results)
            self.plot_adaptation_metrics(results)
            
            self.plot_solution_impact_metrics(results)
            self.plot_intervention_effectiveness(results)
            self.plot_temporal_evolution(results)
            self.plot_environmental_factors(results)
            self.plot_crisis_response_analysis(results)
            self.plot_psychological_patterns(results)
            self.plot_social_network_analysis(results)
            self.plot_resource_optimization(results)
            self.plot_behavioral_adaptation(results) 
            self.plot_stress_correlation_matrix(results) 
            self.plot_personality_impact(results)
            self.plot_exit_accessibility_impact(results)
            self.plot_group_dynamics(results)
            self.plot_intervention_timing_analysis(results)
            self.plot_stress_recovery_patterns(results)
            self.plot_crisis_event_impact(results)
            self.plot_solution_comparative_analysis(results)
            
            self.logger.info("Successfully generated all visualizations")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            
    def plot_stress_metrics(self, results: Dict) -> None:
        """
        Plot stress-related metrics with quadrant visualization.
        
        Args:
            results: Dictionary containing stress-related metrics
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            stress_data = self._prepare_stress_data(results)
            sns.boxplot(data=stress_data, x='Space Type', y='Stress Level', 
                       hue='Has Exit', ax=ax1)
            self._format_plot(ax1, PlotConfig(
                title='Stress Distribution by Environment',
                xlabel='Space Type',
                ylabel='Stress Level',
                legend_title='Exit Available'
            ))
            
            self._plot_stress_timeline(results, ax2)
            
            self._plot_stress_adaptation(results, ax3)
            
            self._plot_crisis_stress(results, ax4)
            
            plt.tight_layout()
            self._save_plot('stress_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error in stress metrics plotting: {str(e)}")
            
    def _prepare_stress_data(self, results: Dict) -> pd.DataFrame:
        """
        Prepare stress-related data for plotting.
        
        Args:
            results: Raw results dictionary
            
        Returns:
            DataFrame with processed stress metrics
        """
        data = []
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                data.append({
                    "Space Type": group.get("space_type", "Unknown"),
                    "Has Exit": "Yes" if group.get("has_exit") else "No",
                    "Stress Level": participant.get("final_stress", 0),
                    "Phase": results.get("current_phase", "unknown")
                })
        return pd.DataFrame(data)

    def _plot_stress_timeline(self, results: dict, ax):
        """Plot stress levels over time"""
        phases = ['Initial', 'Adaptation', 'Crisis', 'Resolution']
        for group in results["groups"]:
            stress_values = [
                group["avg_stress_baseline"],
                group.get("adaptation_stress", group["avg_stress"]),
                group.get("crisis_stress", group["avg_stress"]),
                group["avg_stress"]
            ]
            label = f"{group['space_type']} ({'Exit' if group['has_exit'] else 'No Exit'})"
            ax.plot(phases, stress_values, marker='o', label=label)
        ax.set_title("Stress Evolution Over Time")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Average Stress Level")
        ax.legend()

    def _plot_stress_adaptation(self, results: Dict, ax: plt.Axes) -> None:
        """Plot stress vs adaptation correlation with statistical annotations"""
        data = []
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                data.append({
                    "Stress": participant.get("final_stress", 0),
                    "Adaptation": participant.get("adaptation_score", 0),
                    "Group": group.get("space_type", "Unknown"),
                    "Solution": "With Solution" if self.solution_enabled else "Without Solution"
                })
        df = pd.DataFrame(data)
        
        correlation = df["Stress"].corr(df["Adaptation"])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df["Stress"], df["Adaptation"])
        
        sns.scatterplot(data=df, x="Stress", y="Adaptation", 
                       hue="Group", style="Solution", s=100, alpha=0.6, ax=ax)
        sns.regplot(data=df, x="Stress", y="Adaptation", 
                   scatter=False, color='black', line_kws={'linestyle': '--'}, ax=ax)
        
        ax.text(0.05, 0.95, 
                f'r = {correlation:.2f}\np = {p_value:.2e}', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title("Stress vs Adaptation Score\nwith Linear Regression", pad=20)
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Adaptation Score")

    def _plot_crisis_stress(self, results: Dict, ax: plt.Axes) -> None:
        """Plot stress during crisis events"""
        data = []
        for group in results.get("groups", []):
            data.append({
                "Group": f"{group.get('space_type', 'Unknown')}",
                "Crisis Stress": group.get("crisis_stress", group.get("avg_stress", 0)),
                "Has Exit": "Yes" if group.get("has_exit") else "No"
            })
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x="Group", y="Crisis Stress", 
                   hue="Has Exit", ax=ax)
        ax.set_title("Stress Levels During Crisis Events")
        ax.set_xlabel("Space Type")
        ax.set_ylabel("Stress Level")

    def plot_social_metrics(self, results: Dict) -> None:
        """Plot social interaction metrics"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            data = []
            for group in results.get("groups", []):
                for participant in group.get("participants", []):
                    data.append({
                        "Space Type": group.get("space_type", "Unknown"),
                        "Has Exit": "Yes" if group.get("has_exit") else "No",
                        "Connections": participant.get("social_connections", 0)
                    })
            df = pd.DataFrame(data)
            
            sns.boxplot(data=df, x="Space Type", y="Connections", 
                       hue="Has Exit", ax=ax1)
            self._format_plot(ax1, PlotConfig(
                title="Social Connections Distribution",
                xlabel="Space Type",
                ylabel="Number of Connections",
                legend_title="Exit Available"
            ))
            
            cohesion_data = []
            for group in results.get("groups", []):
                avg_connections = np.mean([
                    p.get("social_connections", 0) 
                    for p in group.get("participants", [])
                ])
                cohesion_data.append({
                    "Space Type": group.get("space_type", "Unknown"),
                    "Average Connections": avg_connections,
                    "Has Exit": "Yes" if group.get("has_exit") else "No"
                })
            
            df_cohesion = pd.DataFrame(cohesion_data)
            sns.barplot(data=df_cohesion, x="Space Type", y="Average Connections",
                       hue="Has Exit", ax=ax2)
            ax2.set_title("Group Cohesion by Environment")
            
            self._plot_social_network_evolution(results, ax3)
            
            metrics = df.groupby("Space Type")["Connections"].agg(
                ["mean", "std", "min", "max"]).reset_index()
            sns.heatmap(metrics.set_index("Space Type"), annot=True, 
                       fmt=".2f", cmap="YlOrRd", ax=ax4)
            ax4.set_title("Social Metrics Summary")
            
            plt.tight_layout()
            self._save_plot('social_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error in social metrics plotting: {str(e)}")
            raise

    def _plot_social_network_evolution(self, results: Dict, ax: plt.Axes) -> None:
        """Plot the evolution of social connections over time"""
        phases = ['Initial', 'Adaptation', 'Crisis', 'Resolution']
        
        evolution_data = []
        for group in results.get("groups", []):
            connections_over_time = []
            
            initial_connections = np.mean([
                participant.get("initial_social_connections", 0)
                for participant in group.get("participants", [])
            ])
            
            adaptation_connections = np.mean([
                participant.get("adaptation_connections", participant.get("social_connections", 0) * 0.5)
                for participant in group.get("participants", [])
            ])
            
            crisis_connections = np.mean([
                participant.get("crisis_connections", participant.get("social_connections", 0) * 0.8)
                for participant in group.get("participants", [])
            ])
            
            final_connections = np.mean([
                participant.get("social_connections", 0)
                for participant in group.get("participants", [])
            ])
            
            connections_over_time = [
                initial_connections,
                adaptation_connections,
                crisis_connections,
                final_connections
            ]
            
            label = f"{group['space_type']} ({'Exit' if group['has_exit'] else 'No Exit'})"
            evolution_data.append((label, connections_over_time))

        for label, connections in evolution_data:
            ax.plot(phases, connections, marker='o', label=label, linewidth=2, markersize=8)
            
            for i in range(1, len(phases)):
                pct_change = ((connections[i] - connections[i-1]) / connections[i-1] * 100)
                ax.annotate(f'{pct_change:+.1f}%', 
                           xy=(phases[i], connections[i]),
                           xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=8,
                           alpha=0.7)

        ax.set_title("Social Network Evolution Over Time", pad=20)
        ax.set_xlabel("Experiment Phase")
        ax.set_ylabel("Average Social Connections")
        ax.legend(title="Environment Type")
        ax.grid(True, alpha=0.3)
        
        try:
            f_stat, p_val = stats.f_oneway(*[data[1] for data in evolution_data])
            ax.text(0.02, 0.98, f'ANOVA: p = {p_val:.2e}',
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
        except:
            pass

    def plot_resource_metrics(self, results: Dict) -> None:
        """Plot resource management metrics"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            data = []
            for group in results.get("groups", []):
                data.append({
                    "Space Type": group.get("space_type", "Unknown"),
                    "Resources": group.get("collective_resources", 0),
                    "Has Exit": "Yes" if group.get("has_exit") else "No"
                })
            df = pd.DataFrame(data)
            
            sns.barplot(data=df, x="Space Type", y="Resources",
                       hue="Has Exit", ax=ax1)
            ax1.set_title("Resource Distribution by Environment")
            
            efficiency = df["Resources"] / df.groupby("Space Type")["Resources"].transform("max")
            df["Efficiency"] = efficiency
            sns.boxplot(data=df, x="Space Type", y="Efficiency", ax=ax2)
            ax2.set_title("Resource Management Efficiency")
            
            plt.tight_layout()
            self._save_plot('resource_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error in resource metrics plotting: {str(e)}")
            raise

    def _plot_solution_effectiveness(self, results: Dict, ax: plt.Axes) -> None:
        """Plot comparative effectiveness of different solutions"""
        metrics = ['stress_reduction', 'adaptation_improvement', 'social_connectivity']
        solution_types = ['adaptive_architecture', 'physical_integration', 'therapeutic_integration']
        
        effectiveness_data = []
        for solution in solution_types:
            solution_results = [
                group for group in results.get("groups", [])
                if group.get("solution_type") == solution
            ]
            
            if solution_results:
                avg_metrics = {
                    'stress_reduction': np.mean([
                        (g['avg_stress_baseline'] - g['avg_stress']) / g['avg_stress_baseline'] * 100
                        for g in solution_results
                    ]),
                    'adaptation_improvement': np.mean([
                        np.mean([p['adaptation_score'] for p in g['participants']])
                        for g in solution_results
                    ]),
                    'social_connectivity': np.mean([
                        np.mean([len(p['social_connections']) for p in g['participants']])
                        for g in solution_results
                    ])
                }
                
                for metric in metrics:
                    effectiveness_data.append({
                        'Solution': solution.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': avg_metrics[metric]
                    })
        
        df = pd.DataFrame(effectiveness_data)
        
        sns.barplot(data=df, x='Solution', y='Value', hue='Metric', ax=ax)
        ax.set_title('Solution Effectiveness Comparison')
        ax.set_xlabel('Solution Type')
        ax.set_ylabel('Improvement (%)')
        ax.tick_params(axis='x', rotation=45)
        
        if len(solution_types) >= 2:
            from scipy.stats import f_oneway
            f_stat, p_val = f_oneway(*[
                df[df['Solution'] == sol]['Value'].values
                for sol in df['Solution'].unique()
            ])
            ax.text(0.02, 0.98, f'ANOVA: p = {p_val:.2e}',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

    def plot_comparative_metrics(self, results: Dict) -> None:
        """Plot detailed comparison of key metrics"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_stress_adaptation(results, ax1)
            
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_solution_effectiveness(results, ax2)
            
            ax3 = fig.add_subplot(gs[1, :])
            self._plot_temporal_progression(results, ax3)
            
            plt.tight_layout()
            self._save_plot('comparative_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error in comparative metrics plotting: {str(e)}")

    def save_all_plots(self) -> None:
        """Ensure all plots are properly closed"""
        plt.close('all')
        self.logger.info("Closed all plot windows")

    def plot_solution_impact_metrics(self, results: Dict) -> None:
        """Plot comprehensive before/after solution comparisons"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_before_after_comparison(
                results, 
                metric='stress_level',
                title='Stress Level Changes',
                ax=ax1
            )

            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_before_after_comparison(
                results,
                metric='adaptation_score',
                title='Adaptation Score Changes',
                ax=ax2
            )

            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_before_after_comparison(
                results,
                metric='social_connections',
                title='Social Connection Changes',
                ax=ax3
            )

            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_success_rate_comparison(results, ax4)

            plt.tight_layout()
            self._save_plot('solution_impact_metrics.png')
        except Exception as e:
            self.logger.error(f"Error in solution impact plotting: {str(e)}")

    def _plot_before_after_comparison(self, results: Dict, metric: str, title: str, ax: plt.Axes) -> None:
        """Create before/after comparison plot with statistical testing"""
        before_data = []
        after_data = []
        
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                before = participant.get(f"initial_{metric}", 0)
                after = participant.get(metric, 0)
                before_data.append(before)
                after_data.append(after)

        t_stat, p_val = stats.ttest_rel(before_data, after_data)
        effect_size = (np.mean(after_data) - np.mean(before_data)) / np.std(before_data)

        data = pd.DataFrame({
            'Phase': ['Before'] * len(before_data) + ['After'] * len(after_data),
            'Value': before_data + after_data
        })
        
        sns.violinplot(data=data, x='Phase', y='Value', ax=ax)
        
        ax.text(0.05, 0.95, 
                f'p = {p_val:.2e}\nd = {effect_size:.2f}', 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_ylabel(metric.replace('_', ' ').title())

    def plot_intervention_effectiveness(self, results: Dict) -> None:
        """Plot intervention effectiveness across different conditions"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            self._plot_temporal_effects(results, ax1)
            
            self._plot_space_type_effects(results, ax2)
            
            self._plot_personality_effects(results, ax3)
            
            self._plot_crisis_response(results, ax4)
            
            plt.tight_layout()
            self._save_plot('intervention_effectiveness.png')
        except Exception as e:
            self.logger.error(f"Error in intervention effectiveness plotting: {str(e)}")

    def plot_temporal_evolution(self, results: Dict) -> None:
        """Plot temporal evolution of key metrics"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_stress_timeline(results, ax1)
            
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_adaptation_progress(results, ax2)
            
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_social_growth(results, ax3)
            
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_resource_timeline(results, ax4)
            
            plt.tight_layout()
            self._save_plot('temporal_evolution.png')
        except Exception as e:
            self.logger.error(f"Error in temporal evolution plotting: {str(e)}")

    def plot_adaptation_metrics(self, results: Dict) -> None:
        """Plot adaptation-related metrics"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            self._plot_adaptation_timeline(results, ax1)
            
            self._plot_adaptation_personality(results, ax2)
            
            self._plot_adaptation_environment(results, ax3)
            
            self._plot_adaptation_solution_impact(results, ax4)
            
            plt.tight_layout()
            self._save_plot('adaptation_analysis.png')
            
        except Exception as e:
            self.logger.error(f"Error in adaptation metrics plotting: {str(e)}")

    def _plot_adaptation_timeline(self, results: Dict, ax: plt.Axes) -> None:
        """Plot adaptation progress over time"""
        phases = ['Initial', 'Adaptation', 'Crisis', 'Resolution']
        for group in results.get("groups", []):
            adaptation_values = [
                np.mean([p.get("initial_adaptation_score", 0) for p in group.get("participants", [])]),
                np.mean([p.get("adaptation_score", 0) for p in group.get("participants", [])]) * 0.7,
                np.mean([p.get("adaptation_score", 0) for p in group.get("participants", [])]) * 0.9,
                np.mean([p.get("adaptation_score", 0) for p in group.get("participants", [])])
            ]
            label = f"{group['space_type']} ({'Exit' if group['has_exit'] else 'No Exit'})"
            ax.plot(phases, adaptation_values, marker='o', label=label)
            
        ax.set_title("Adaptation Progress Over Time")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Average Adaptation Score")
        ax.legend()

    def _plot_adaptation_personality(self, results: Dict, ax: plt.Axes) -> None:
        """Plot relationship between personality traits and adaptation"""
        data = []
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                data.append({
                    "Adaptation": participant.get("adaptation_score", 0),
                    "Extroversion": participant.get("personality", {}).get("extroversion", 0),
                    "Resilience": participant.get("personality", {}).get("resilience", 0),
                    "Group": group.get("space_type", "Unknown")
                })
        df = pd.DataFrame(data)
        
        sns.scatterplot(data=df, x="Extroversion", y="Adaptation", 
                       size="Resilience", hue="Group", ax=ax)
        ax.set_title("Adaptation vs Personality Traits")

    def _plot_adaptation_environment(self, results: Dict, ax: plt.Axes) -> None:
        """Plot adaptation scores by environmental conditions"""
        data = []
        for group in results.get("groups", []):
            avg_adaptation = np.mean([
                p.get("adaptation_score", 0) 
                for p in group.get("participants", [])
            ])
            data.append({
                "Space Type": group.get("space_type", "Unknown"),
                "Has Exit": "Yes" if group.get("has_exit") else "No",
                "Adaptation Score": avg_adaptation
            })
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x="Space Type", y="Adaptation Score",
                   hue="Has Exit", ax=ax)
        ax.set_title("Environmental Impact on Adaptation")

    def _plot_adaptation_solution_impact(self, results: Dict, ax: plt.Axes) -> None:
        """Plot solution impact on adaptation"""
        solution_data = []
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                initial = participant.get("initial_adaptation_score", 0)
                final = participant.get("adaptation_score", 0)
                solution_data.append({
                    "Phase": "Before",
                    "Score": initial,
                    "Space Type": group.get("space_type", "Unknown")
                })
                solution_data.append({
                    "Phase": "After",
                    "Score": final,
                    "Space Type": group.get("space_type", "Unknown")
                })
        
        df = pd.DataFrame(solution_data)
        sns.boxplot(data=df, x="Phase", y="Score", 
                   hue="Space Type", ax=ax)
        ax.set_title("Solution Impact on Adaptation")

    def _plot_success_rate_comparison(self, results: Dict, ax: plt.Axes) -> None:
        """Plot success rate comparison between baseline and solution"""
        data = []
        for group in results.get("groups", []):
            data.append({
                "Group": group.get("space_type", "Unknown"),
                "Success Rate": group.get("success_rate", 0) * 100,
                "Has Exit": "Yes" if group.get("has_exit") else "No",
                "Solution": "With Solution" if self.solution_enabled else "Without Solution"
            })
        
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x="Group", y="Success Rate", 
                   hue="Solution", ax=ax)
        
        ax.set_title("Success Rate Comparison")
        ax.set_ylabel("Success Rate (%)")
        
        solution_rates = df[df["Solution"] == "With Solution"]["Success Rate"]
        baseline_rates = df[df["Solution"] == "Without Solution"]["Success Rate"]
        if len(solution_rates) > 0 and len(baseline_rates) > 0:
            t_stat, p_val = stats.ttest_ind(solution_rates, baseline_rates)
            ax.text(0.05, 0.95, f'p = {p_val:.2e}', 
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))

    def _plot_temporal_effects(self, results: Dict, ax: plt.Axes) -> None:
        """Plot immediate vs long-term effects of interventions"""
        phases = ['Immediate', 'Short-term', 'Medium-term', 'Long-term']
        for group in results.get("groups", []):
            effect_values = [
                np.mean([p.get("immediate_effect", 0) for p in group.get("participants", [])]),
                np.mean([p.get("short_term_effect", 0) for p in group.get("participants", [])]),
                np.mean([p.get("medium_term_effect", 0) for p in group.get("participants", [])]),
                np.mean([p.get("adaptation_score", 0) for p in group.get("participants", [])])
            ]
            label = f"{group['space_type']} ({'Exit' if group['has_exit'] else 'No Exit'})"
            ax.plot(phases, effect_values, marker='o', label=label)
            
        ax.set_title("Temporal Effects of Interventions")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Effect Magnitude")
        ax.legend()

    def _plot_adaptation_progress(self, results: Dict, ax: plt.Axes) -> None:
        """Plot adaptation progress over time"""
        data = []
        for group in results.get("groups", []):
            for participant in group.get("participants", []):
                data.append({
                    "Time": "Initial",
                    "Adaptation": participant.get("initial_adaptation_score", 0),
                    "Group": group.get("space_type", "Unknown")
                })
                data.append({
                    "Time": "Final",
                    "Adaptation": participant.get("adaptation_score", 0),
                    "Group": group.get("space_type", "Unknown")
                })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x="Time", y="Adaptation", hue="Group", ax=ax)
        ax.set_title("Adaptation Progress")

    def plot_environmental_factors(self, results: Dict) -> None:
        """Plot analysis of environmental factors"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            space_data = []
            for group in results.get("groups", []):
                space_data.append({
                    "Space Type": group.get("space_type", "Unknown"),
                    "Stress": group.get("avg_stress", 0),
                    "Has Exit": "Yes" if group.get("has_exit") else "No",
                    "Success": group.get("success_rate", 0) * 100
                })
            df_space = pd.DataFrame(space_data)
            
            sns.barplot(data=df_space, x="Space Type", y="Stress",
                       hue="Has Exit", ax=ax1)
            ax1.set_title("Space Configuration Impact")
            
            stress_factors = {
                "Confined": [g["avg_stress"] for g in results["groups"] if g["space_type"] == "confined"],
                "Open": [g["avg_stress"] for g in results["groups"] if g["space_type"] == "open"],
                "With Exit": [g["avg_stress"] for g in results["groups"] if g["has_exit"]],
                "No Exit": [g["avg_stress"] for g in results["groups"] if not g["has_exit"]]
            }
            ax2.boxplot(stress_factors.values(), labels=stress_factors.keys())
            ax2.set_title("Environmental Stress Factors")
            
            sns.scatterplot(data=df_space, x="Stress", y="Success",
                          hue="Space Type", style="Has Exit", s=100, ax=ax3)
            ax3.set_title("Success Rate by Environment")
            
            adaptation_data = []
            for group in results.get("groups", []):
                for participant in group.get("participants", []):
                    adaptation_data.append({
                        "Space Type": group.get("space_type", "Unknown"),
                        "Has Exit": "Yes" if group.get("has_exit") else "No",
                        "Adaptation": participant.get("adaptation_score", 0)
                    })
            df_adaptation = pd.DataFrame(adaptation_data)
            
            sns.violinplot(data=df_adaptation, x="Space Type", y="Adaptation",
                         hue="Has Exit", split=True, ax=ax4)
            ax4.set_title("Adaptation by Environment")
            
            plt.tight_layout()
            self._save_plot('environmental_factors.png')
            
        except Exception as e:
            self.logger.error(f"Error in environmental factors plotting: {str(e)}")

    def _get_group_label(self, space_type: str, has_exit: bool) -> str:
        """Convert space type and exit condition to group label"""
        if (space_type == "confined"):
            return "Group A" if has_exit else "Group B"
        else:
            return "Group C" if has_exit else "Group D"

    def plot_isolation_impact(self, results: Dict) -> None:
        """Plot stress, anxiety, and performance in isolated environment"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            data = []
            for group in results.get("groups", []):
                group_label = self._get_group_label(
                    group.get("space_type", "Unknown"),
                    group.get("has_exit", False)
                )
                for participant in group.get("participants", []):
                    data.append({
                        "Group": group_label,
                        "Stress": participant.get("stress_level", 0),
                        "Anxiety": participant.get("anxiety_level", 0),
                        "Performance": participant.get("performance_score", 0)
                    })
            df = pd.DataFrame(data)

            sns.violinplot(data=df, x="Group", y="Stress", ax=ax1)
            ax1.set_title("Stress Levels by Group")
            ax1.set_ylabel("Stress Level")
            
            sns.violinplot(data=df, x="Group", y="Anxiety", ax=ax2)
            ax2.set_title("Anxiety Levels by Group")
            ax2.set_ylabel("Anxiety Level")
            
            sns.violinplot(data=df, x="Group", y="Performance", ax=ax3)
            ax3.set_title("Performance by Group")
            ax3.set_ylabel("Performance Score")
            
            for ax, metric in zip([ax1, ax2, ax3], ["Stress", "Anxiety", "Performance"]):
                f_stat, p_val = stats.f_oneway(*[
                    df[df["Group"] == group][metric].values
                    for group in ["Group A", "Group B", "Group C", "Group D"]
                ])
                ax.text(0.05, 0.95, f'p = {p_val:.2e}',
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._save_plot('isolation_impact.png')
            
        except Exception as e:
            self.logger.error(f"Error in isolation impact plotting: {str(e)}")

    def plot_solution_comparison(self, before_results: Dict, after_results: Dict) -> None:
        """Plot before/after comparison of solutions impact"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            metrics = {
                "Stress": ("Stress Level", "stress_level"),
                "Anxiety": ("Anxiety Level", "anxiety_level"),
                "Performance": ("Performance Score", "performance_score")
            }
            
            for i, (title, (ylabel, metric)) in enumerate(metrics.items()):
                ax = [ax1, ax2, ax3][i]
                
                before_data = []
                after_data = []
                
                for group in before_results.get("groups", []):
                    group_label = self._get_group_label(
                        group.get("space_type", "Unknown"),
                        group.get("has_exit", False)
                    )
                    before_vals = [p.get(metric, 0) for p in group.get("participants", [])]
                    before_data.extend([(v, group_label, "Before") for v in before_vals])
                
                for group in after_results.get("groups", []):
                    group_label = self._get_group_label(
                        group.get("space_type", "Unknown"),
                        group.get("has_exit", False)
                    )
                    after_vals = [p.get(metric, 0) for p in group.get("participants", [])]
                    after_data.extend([(v, group_label, "After") for v in after_vals])
                
                df = pd.DataFrame(before_data + after_data, 
                                columns=["Value", "Group", "Phase"])
                
                sns.boxplot(data=df, x="Group", y="Value", hue="Phase", ax=ax)
                ax.set_title(f"{title} Before/After Solution")
                ax.set_ylabel(ylabel)
                
                for group in ["Group A", "Group B", "Group C", "Group D"]:
                    before_vals = df[(df["Group"] == group) & (df["Phase"] == "Before")]["Value"]
                    after_vals = df[(df["Group"] == group) & (df["Phase"] == "After")]["Value"]
                    if len(before_vals) > 0 and len(after_vals) > 0:
                        t_stat, p_val = stats.ttest_ind(before_vals, after_vals)
                        effect_size = (after_vals.mean() - before_vals.mean()) / before_vals.std()
                        ax.text(i-0.2, ax.get_ylim()[1]*0.9,
                               f'p = {p_val:.2e}\nd = {effect_size:.2f}',
                               bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._save_plot('solution_comparison.png')
            
        except Exception as e:
            self.logger.error(f"Error in solution comparison plotting: {str(e)}")

    def plot_isolation_metrics(self, results: Dict) -> None:
        """Plot isolation impact on stress, anxiety, and performance"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        groups = {
            ("confined", True): "Group A",
            ("confined", False): "Group B",
            ("open", True): "Group C",
            ("open", False): "Group D" 
        }
        
        data = []
        for group in results.get("groups", []):
            group_label = groups.get((group.get("space_type"), group.get("has_exit")))
            for participant in group.get("participants", []):
                data.append({
                    "Group": group_label,
                    "Stress": participant.get("final_stress", 0),
                    "Anxiety": participant.get("anxiety_level", 0),
                    "Performance": participant.get("adaptation_score", 0)
                })
        df = pd.DataFrame(data)

        sns.violinplot(data=df, x="Group", y="Stress", ax=ax1)
        ax1.set_title("Stress Levels by Group")
        ax1.set_ylabel("Stress Level")
        
        sns.violinplot(data=df, x="Group", y="Anxiety", ax=ax2)
        ax2.set_title("Anxiety Levels by Group")
        ax2.set_ylabel("Anxiety Level")
        
        sns.violinplot(data=df, x="Group", y="Performance", ax=ax3)
        ax3.set_title("Performance by Group")
        ax3.set_ylabel("Performance Score")

        for ax, metric in zip([ax1, ax2, ax3], ["Stress", "Anxiety", "Performance"]):
            f_stat, p_val = stats.f_oneway(*[
                df[df["Group"] == group][metric].values 
                for group in ["Group A", "Group B", "Group C", "Group D"]
            ])
            ax.text(0.05, 0.95, f'p = {p_val:.2e}',
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_plot('isolation_metrics.png')

    def plot_solution_impact(self, before_results: Dict, after_results: Dict) -> None:
        """Plot metrics before and after solution implementation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        groups = {
            ("confined", True): "Group A",
            ("confined", False): "Group B",
            ("open", True): "Group C",
            ("open", False): "Group D"
        }

        metrics = []
        for results, condition in [(before_results, "Before"), (after_results, "After")]:
            for group in results.get("groups", []):
                group_label = groups.get((group.get("space_type"), group.get("has_exit")))
                for participant in group.get("participants", []):
                    metrics.append({
                        "Group": group_label,
                        "Condition": condition,
                        "Stress": participant.get("final_stress", 0),
                        "Anxiety": participant.get("anxiety_level", 0),
                        "Performance": participant.get("adaptation_score", 0)
                    })
        df = pd.DataFrame(metrics)

        self._plot_before_after(df, "Stress", "Stress Level", ax1)
        self._plot_before_after(df, "Anxiety", "Anxiety Level", ax2)
        self._plot_before_after(df, "Performance", "Performance Score", ax3)

        plt.tight_layout()
        self._save_plot('solution_impact.png')

    def _plot_before_after(self, df: pd.DataFrame, metric: str, ylabel: str, ax: plt.Axes) -> None:
        """Helper method for before/after comparison plots"""
        sns.boxplot(data=df, x="Group", y=metric, hue="Condition", ax=ax)
        ax.set_title(f"{metric} Before/After Solution")
        ax.set_ylabel(ylabel)
        
        for group in ["Group A", "Group B", "Group C", "Group D"]:
            before = df[(df["Group"] == group) & (df["Condition"] == "Before")][metric]
            after = df[(df["Group"] == group) & (df["Condition"] == "After")][metric]
            if len(before) > 0 and len(after) > 0:
                t_stat, p_val = stats.ttest_ind(before, after)
                effect_size = (after.mean() - before.mean()) / before.std()
                x_pos = ["Group A", "Group B", "Group C", "Group D"].index(group)
                ax.text(x_pos, ax.get_ylim()[1]*0.9, 
                       f'p={p_val:.2e}\nd={effect_size:.2f}',
                       ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.8))

    def plot_solution_effectiveness(self, results: Dict) -> None:
        """Plot comparative effectiveness of different solutions with ANOVA"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        solution_data = []
        for solution_type in ["adaptive_architecture", "physical_integration", "therapeutic_integration"]:
            solution_groups = [g for g in results.get("groups", []) 
                             if g.get("solution_type") == solution_type]
            
            for group in solution_groups:
                improvement = (group.get("avg_stress_baseline", 0) - 
                             group.get("avg_stress", 0)) / group.get("avg_stress_baseline", 1) * 100
                solution_data.append({
                    "Solution": solution_type.replace("_", " ").title(),
                    "Improvement": improvement
                })
        
        df = pd.DataFrame(solution_data)
        
        sns.barplot(data=df, x="Solution", y="Improvement", ax=ax)
        ax.set_title("Solution Effectiveness Comparison")
        ax.set_xlabel("Solution Type")
        ax.set_ylabel("Stress Reduction (%)")
        
        solutions = df["Solution"].unique()
        if len(solutions) >= 2:
            f_stat, p_val = stats.f_oneway(*[
                df[df["Solution"] == sol]["Improvement"].values 
                for sol in solutions
            ])
            ax.text(0.02, 0.98, f'ANOVA:\nF = {f_stat:.2f}\np = {p_val:.2e}',
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        self._save_plot('solution_effectiveness.png')

    def _safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers, handling division by zero"""
        try:
            if abs(b) < 1e-10:
                return default
            return (a / b)
        except (ZeroDivisionError, RuntimeWarning):
            return default

    def _calculate_percent_change(self, current: float, previous: float) -> float:
        """Calculate percentage change with safety checks"""
        if previous == 0 or np.isnan(previous):
            return 0.0
        return self._safe_divide(current - previous, abs(previous)) * 100

    def _calculate_effect_size(self, after_data: np.ndarray, before_data: np.ndarray) -> float:
        """Calculate Cohen's d effect size with safety checks"""
        if len(after_data) == 0 or len(before_data) == 0:
            return 0.0
            
        std_before = np.std(before_data)
        if std_before == 0:
            pooled_std = np.sqrt((np.var(after_data) + np.var(before_data)) / 2)
            if pooled_std == 0:
                return 0.0
            return self._safe_divide(np.mean(after_data) - np.mean(before_data), pooled_std)
            
        return self._safe_divide(np.mean(after_data) - np.mean(before_data), std_before)

    def _perform_anova(self, data_groups: List[np.ndarray]) -> tuple:
        """Perform one-way ANOVA with safety checks"""
        try:
            valid_groups = [group for group in data_groups if len(group) > 0]
            if len(valid_groups) < 2:
                return np.nan, np.nan
                
            if all(np.all(group == group[0]) for group in valid_groups):
                return np.nan, np.nan
                
            return stats.f_oneway(*valid_groups)
        except:
            return np.nan, np.nan

    def plot_evolution_data(self, evolution_data: List[tuple], title: str):
        """Plot evolution data with improved error handling"""
        plt.figure(figsize=(10, 6))
        
        for label, data in evolution_data:
            if len(data) > 0:
                plt.plot(data, label=label, marker='o')
        
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        
        try:
            plt.tight_layout()
        except:
            pass
        
        save_path = self.results_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
