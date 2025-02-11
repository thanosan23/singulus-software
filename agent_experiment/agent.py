import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AgentPersonality:
    extroversion: float
    neuroticism: float
    agreeableness: float
    openness: float
    conscientiousness: float
    stress_resilience: float
    social_needs: float
    leadership_tendency: float

class Agent:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.personality = AgentPersonality(
            extroversion=np.random.normal(0.5, 0.15),
            neuroticism=np.random.normal(0.5, 0.15),
            agreeableness=np.random.normal(0.6, 0.1),
            openness=np.random.normal(0.5, 0.15),
            conscientiousness=np.random.normal(0.6, 0.1),
            stress_resilience=np.random.normal(0.5, 0.2),
            social_needs=np.random.normal(0.5, 0.15),
            leadership_tendency=np.random.beta(2, 5)
        )
        self.stress = 0
        self.energy = 100
        self.mood = 0.5
        self.relationships = {}
        self.role = None
        self.task_history = []
        
    def update(self, environment: Dict, others: List['Agent']) -> Dict[str, float]:
        confined_impact = 0.3 if environment.get('is_confined') else 0.05
        social_satisfaction = self._calculate_social_satisfaction(others)
        stress_change = self._calculate_stress_change(environment, social_satisfaction)
        
        self.stress = np.clip(self.stress + stress_change, 0, 1)
        self.mood = np.clip(
            self.mood + 
            (0.1 * social_satisfaction) + 
            (-0.2 * self.stress) +
            np.random.normal(0, 0.05),
            -1, 1
        )
        
        return {
            'stress_change': stress_change,
            'mood': self.mood,
            'social_satisfaction': social_satisfaction
        }
    
    def make_decision(self, options: List[str], context: Dict) -> str:
        weights = []
        for option in options:
            weight = 1.0
            
            if 'leadership' in option:
                weight *= (1.5 * self.personality.leadership_tendency)
            
            if 'collaborate' in option:
                weight *= (
                    1.2 * self.personality.agreeableness * 
                    (1 + self.personality.extroversion) *
                    (1 - self.stress * 0.5)
                )
            
            if 'risky' in option:
                weight *= (
                    (1 - self.personality.neuroticism) * 
                    self.personality.openness *
                    (1 - self.stress)
                )
            
            if 'cautious' in option:
                weight *= (
                    self.personality.conscientiousness * 
                    (1 + self.stress)
                )
            
            weights.append(max(0.1, weight))
        
        weights = np.array(weights) / sum(weights)
        return np.random.choice(options, p=weights)
    
    def _calculate_social_satisfaction(self, others: List['Agent']) -> float:
        if not others:
            return 0
        
        satisfaction = 0
        for other in others:
            compatibility = self._calculate_compatibility(other)
            relationship = self.relationships.get(other.id, 0.5)
            satisfaction += compatibility * relationship
        
        return np.clip(satisfaction / len(others), 0, 1)
    
    def _calculate_stress_change(self, environment: Dict, social_satisfaction: float) -> float:
        base_change = np.random.normal(0, 0.05)
        
        stressors = [
            environment.get('is_confined', False) * 0.2,
            (1 - social_satisfaction) * 0.3,
            (1 - self.personality.stress_resilience) * 0.2,
            -self.personality.openness * 0.1
        ]
        
        return sum(stressors) + base_change
