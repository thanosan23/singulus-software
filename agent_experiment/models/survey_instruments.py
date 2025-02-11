from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PsychologicalAssessment:
    """Implements standardized psychological assessment tools"""
    
    @staticmethod
    def calculate_stai_score(responses: Dict[str, int]) -> float:
        anxiety_indicators = [
            'feeling_tense', 'feeling_nervous', 'worrying_thoughts',
            'lack_of_confidence', 'feeling_restless'
        ]
        return sum(responses.get(indicator, 0) for indicator in anxiety_indicators) / len(anxiety_indicators)
    
    @staticmethod
    def calculate_poms_scores(responses: Dict[str, int]) -> Dict[str, float]:
        return {
            'tension': sum(responses.get(k, 0) for k in ['tense', 'shaky', 'restless']) / 3,
            'depression': sum(responses.get(k, 0) for k in ['sad', 'unworthy', 'hopeless']) / 3,
            'anger': sum(responses.get(k, 0) for k in ['angry', 'annoyed', 'resentful']) / 3,
            'vigor': sum(responses.get(k, 0) for k in ['energetic', 'active', 'vigorous']) / 3,
            'fatigue': sum(responses.get(k, 0) for k in ['tired', 'exhausted', 'weary']) / 3
        }
    
    @staticmethod
    def calculate_pss_score(responses: Dict[str, int]) -> float:
        stress_indicators = [
            'feeling_overwhelmed', 'unable_to_control', 'feeling_nervous',
            'difficulty_coping', 'feeling_irritable'
        ]
        return sum(responses.get(indicator, 0) for indicator in stress_indicators) / len(stress_indicators)

@dataclass
class SociometricQuestionnaire:
    """Implements social relationship assessment tools"""
    
    @staticmethod
    def calculate_group_cohesion(responses: List[Dict]) -> float:
        cohesion_factors = [
            'trust_in_group',
            'feeling_of_belonging',
            'group_cooperation',
            'communication_quality'
        ]
        return sum(r.get('cohesion_score', 0) for r in responses) / len(responses)
    
    @staticmethod
    def map_social_hierarchy(responses: List[Dict]) -> Dict:
        leadership_scores = {}
        influence_scores = {}
        for response in responses:
            agent_id = response['agent_id']
            leadership_scores[agent_id] = response.get('perceived_leadership', 0)
            influence_scores[agent_id] = response.get('social_influence', 0)
        return {
            'leadership_hierarchy': leadership_scores,
            'influence_network': influence_scores
        }
