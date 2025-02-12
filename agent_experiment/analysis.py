import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from models import Group, Participant
import seaborn as sns
from scipy import stats

class ExperimentAnalyzer:
    def __init__(self, groups: List[Group]):
        self.groups = groups
        
    def generate_stress_report(self) -> pd.DataFrame:
        data = []
        for group in self.groups:
            for participant in group.participants:
                data.append({
                    "group_id": group.id,
                    "space_type": group.space_type.value,
                    "has_exit": group.has_exit,
                    "participant_id": participant.id,
                    "stress_level": participant.stress_level,
                    "survival_points": participant.survival_points
                })
        return pd.DataFrame(data)
    
    def plot_stress_distributions(self):
        df = self.generate_stress_report()
        
        plt.figure(figsize=(10, 6))
        for space_type in df['space_type'].unique():
            data = df[df['space_type'] == space_type]['stress_level']
            plt.hist(data, alpha=0.5, label=space_type)
        
        plt.title("Stress Level Distribution by Space Type")
        plt.xlabel("Stress Level")
        plt.ylabel("Frequency")
        plt.legend()
    
    def generate_summary_statistics(self) -> Dict:
        df = self.generate_stress_report()
        return {
            "mean_stress": df['stress_level'].mean(),
            "mean_survival": df['survival_points'].mean(),
            "stress_by_space_type": df.groupby('space_type')['stress_level'].mean().to_dict(),
            "stress_by_exit": df.groupby('has_exit')['stress_level'].mean().to_dict()
        }
    
    def compute_stress_pvalue(self) -> float:
        df = self.generate_stress_report()
        groups = [group['stress_level'].values for _, group in df.groupby('space_type')]
        if len(groups) >= 2:
            f_val, p_val = stats.f_oneway(*groups)
            return p_val
        return float('nan')
    
    def plot_advanced_graphs(self):
        df = self.generate_stress_report()
        p_val = self.compute_stress_pvalue()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.violinplot(x='space_type', y='stress_level', data=df, ax=axes[0])
        axes[0].set_title("Stress Distribution by Space Type\n(Violin Plot)")
        axes[0].set_xlabel("Space Type")
        axes[0].set_ylabel("Stress Level")
        if not pd.isna(p_val):
            axes[0].text(0.05, 0.95, f'p-value: {p_val:.3e}', transform=axes[0].transAxes, 
                         verticalalignment='top')
        
        sns.boxplot(x='space_type', y='stress_level', data=df, ax=axes[1])
        axes[1].set_title("Stress Distribution by Space Type\n(Box Plot)")
        axes[1].set_xlabel("Space Type")
        axes[1].set_ylabel("Stress Level")
        if not pd.isna(p_val):
            axes[1].text(0.05, 0.95, f'p-value: {p_val:.3e}', transform=axes[1].transAxes, 
                         verticalalignment='top')
        
        plt.tight_layout()
