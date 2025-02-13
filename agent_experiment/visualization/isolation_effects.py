import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List

class IsolationEffectsVisualizer:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.group_labels = {
            ("confined", True): "Group A", 
            ("confined", False): "Group B",
            ("open", True): "Group C",
            ("open", False): "Group D"
        }

    def plot_isolation_effects(self, results: Dict):
        """Plot stress, anxiety, and performance in isolation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        data = self._prepare_data(results)
        
        self._plot_metric(ax1, data, "Stress", "Stress Levels by Group")
        
        self._plot_metric(ax2, data, "Anxiety", "Anxiety Levels by Group")
        self._plot_metric(ax3, data, "Performance", "Performance by Group")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'isolation_effects.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_solution_impact(self, before_results: Dict, after_results: Dict):
        """Plot metrics before and after solution implementation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        before_data = self._prepare_data(before_results)
        after_data = self._prepare_data(after_results)
        
        self._plot_before_after(ax1, before_data, after_data, "Stress", 
                              "Stress Levels Before/After Solutions")
        
        self._plot_before_after(ax2, before_data, after_data, "Anxiety",
                              "Anxiety Levels Before/After Solutions")
        
        self._plot_before_after(ax3, before_data, after_data, "Performance",
                              "Performance Before/After Solutions")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'solution_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_solution_effectiveness(self, results: Dict):
        """Plot comparative effectiveness of different solutions with ANOVA"""
        solutions = ["adaptive_architecture", "physical_integration", "therapeutic_integration"]
        metrics = ["Stress Reduction", "Anxiety Reduction", "Performance Improvement"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = []
        for solution in solutions:
            solution_data = results.get(solution, {})
            for metric in metrics:
                data.append({
                    "Solution": solution.replace("_", " ").title(),
                    "Metric": metric,
                    "Value": solution_data.get(metric, 0)
                })
        
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x="Solution", y="Value", hue="Metric", ax=ax)
        ax.set_title("Solution Effectiveness Comparison")
        ax.set_xlabel("Solution Type")
        ax.set_ylabel("Improvement (%)")
        
        # Add ANOVA results
        f_stat, p_val = stats.f_oneway(*[
            df[df["Solution"] == sol]["Value"].values 
            for sol in df["Solution"].unique()
        ])
        
        ax.text(0.02, 0.98, 
                f'ANOVA Results:\nF = {f_stat:.2f}\np = {p_val:.2e}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'solution_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_final_isolation_analysis(self, baseline_results: Dict, solution_results: Dict):
        """Create final publication-quality graphs with [FINAL] prefix"""
        # First set of graphs - Isolation effects
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig1.suptitle("[FINAL] Isolation Effects Analysis", fontsize=16, y=1.05)
        
        baseline_data = self._prepare_data(baseline_results)
        
        metrics = {
            "Stress": ("Stress Level", "red"),
            "Anxiety": ("Anxiety Level", "orange"),
            "Performance": ("Performance Score", "blue")
        }
        
        axes = [ax1, ax2, ax3]
        for ax, (metric, (ylabel, color)) in zip(axes, metrics.items()):
            sns.violinplot(data=baseline_data, x="Group", y=metric, ax=ax, color=color)
            ax.set_title(f"[FINAL] {metric} in Isolation")
            ax.set_ylabel(ylabel)
            
            # Add statistical testing
            f_stat, p_val = stats.f_oneway(*[
                baseline_data[baseline_data["Group"] == group][metric].values 
                for group in ["Group A", "Group B", "Group C", "Group D"]
            ])
            
            ax.text(0.05, 0.95, f'ANOVA\np = {p_val:.2e}',
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
            
            # Add group labels explanation
            if ax == ax1:
                ax.text(-0.2, -0.15, 
                       "Group A: Confined with exit\nGroup B: Confined w/o exit\n" + 
                       "Group C: Open with exit\nGroup D: Open without exit",
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.results_dir / 'FINAL_isolation_effects.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Second set of graphs - Solution impact
        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig2.suptitle("[FINAL] Solution Implementation Impact", fontsize=16, y=1.05)
        
        solution_data = self._prepare_data(solution_results)
        
        for ax, (metric, (ylabel, color)) in zip(axes, metrics.items()):
            data = pd.concat([
                baseline_data.assign(Condition="Before"),
                solution_data.assign(Condition="After")
            ])
            
            sns.boxplot(data=data, x="Group", y=metric, hue="Condition", ax=ax,
                       palette=[color, "green"])
            ax.set_title(f"[FINAL] {metric} Before/After Solutions")
            ax.set_ylabel(ylabel)
            
            # Add statistical testing for each group
            for group in ["Group A", "Group B", "Group C", "Group D"]:
                before = baseline_data[baseline_data["Group"] == group][metric]
                after = solution_data[solution_data["Group"] == group][metric]
                t_stat, p_val = stats.ttest_ind(before, after)
                effect_size = (after.mean() - before.mean()) / before.std()
                
                x_pos = ["Group A", "Group B", "Group C", "Group D"].index(group)
                ax.text(x_pos, ax.get_ylim()[1]*0.95,
                       f'p={p_val:.2e}\nd={effect_size:.2f}',
                       ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.results_dir / 'FINAL_solution_impact.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _prepare_data(self, results: Dict) -> pd.DataFrame:
        """Prepare data for plotting"""
        data = []
        for group in results.get("groups", []):
            group_label = self.group_labels.get(
                (group.get("space_type"), group.get("has_exit")),
                "Unknown"
            )
            for participant in group.get("participants", []):
                data.append({
                    "Group": group_label,
                    "Stress": participant.get("stress_level", 0),
                    "Anxiety": participant.get("anxiety_level", 0),
                    "Performance": participant.get("adaptation_score", 0)
                })
        return pd.DataFrame(data)

    def _plot_metric(self, ax: plt.Axes, data: pd.DataFrame, metric: str, title: str):
        """Plot a single metric with statistical annotations"""
        sns.violinplot(data=data, x="Group", y=metric, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(metric)
        
        # Add statistical test
        f_stat, p_val = stats.f_oneway(*[
            data[data["Group"] == group][metric].values 
            for group in ["Group A", "Group B", "Group C", "Group D"]
        ])
        
        ax.text(0.05, 0.95, f'p = {p_val:.2e}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

    def _plot_before_after(self, ax: plt.Axes, before_data: pd.DataFrame, 
                          after_data: pd.DataFrame, metric: str, title: str):
        """Plot before/after comparison for a metric"""
        combined = pd.concat([
            before_data.assign(Condition="Before"),
            after_data.assign(Condition="After")
        ])
        
        sns.boxplot(data=combined, x="Group", y=metric, hue="Condition", ax=ax)
        ax.set_title(title)
        
        # Add statistical testing
        for group in ["Group A", "Group B", "Group C", "Group D"]:
            before = before_data[before_data["Group"] == group][metric]
            after = after_data[after_data["Group"] == group][metric]
            t_stat, p_val = stats.ttest_ind(before, after)
            effect_size = (after.mean() - before.mean()) / before.std()
            
            x_pos = ["Group A", "Group B", "Group C", "Group D"].index(group)
            ax.text(x_pos, ax.get_ylim()[1]*0.9,
                   f'p={p_val:.2e}\nd={effect_size:.2f}',
                   ha='center', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.8))
