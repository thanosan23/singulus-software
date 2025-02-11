from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class SPAPProtocol:
    """Synchronous Physical-Psychological Activation Protocol"""
    morning_bdnf_window: int = 37
    circadian_optimal_window: int = 30
    psychological_reset_point: int = 6
    intervention_efficacy: float = 2.78

@dataclass
class TherapeuticIntegration:
    eser_efficacy: float = 0.47 
    pss_intervals: List[int] = (30, 60, 90)
    cognitive_improvement: float = 0.58

class IntegratedSolution:
    def __init__(self):
        self.spap = SPAPProtocol()
        self.therapy = TherapeuticIntegration()
        
    def apply_morning_activation(self, agent_state: Dict) -> Dict:
        """Apply morning BDNF activation protocol"""
        if self._is_within_optimal_window(agent_state['time_since_wake']):
            return {
                'stress_reduction': 0.35,
                'cognitive_boost': 0.45,
                'adaptation_improvement': 0.40,
                'bdnf_activation': True
            }
        return {'bdnf_activation': False}
    
    def apply_psychological_reset(self, agent_state: Dict) -> Dict:
        """Apply midday psychological reset protocol"""
        if self._is_reset_time(agent_state['time_of_day']):
            return {
                'stress_reduction': 0.25,
                'emotional_stability': 0.30,
                'social_connectivity': 0.20
            }
        return {}
    
    def calculate_intervention_impact(self, agent_state: Dict) -> float:
        base_impact = 1.0
        if agent_state.get('bdnf_activation'):
            base_impact *= self.spap.intervention_efficacy
        if agent_state.get('eser_active'):
            base_impact *= (1 + self.therapy.eser_efficacy)
        return base_impact
    
    def _is_within_optimal_window(self, time_since_wake: int) -> bool:
        return 0 <= time_since_wake <= self.spap.circadian_optimal_window
    
    def _is_reset_time(self, time_of_day: float) -> bool:
        return 11.5 <= time_of_day <= 14.5
