import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from .models import Group, Participant

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
        plt.show()
    
    def generate_summary_statistics(self) -> Dict:
        df = self.generate_stress_report()
        return {
            "mean_stress": df['stress_level'].mean(),
            "mean_survival": df['survival_points'].mean(),
            "stress_by_space_type": df.groupby('space_type')['stress_level'].mean().to_dict(),
            "stress_by_exit": df.groupby('has_exit')['stress_level'].mean().to_dict()
        }
