from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class TaskPerformanceMetrics:
    """Implements task performance measurement tools"""
    
    @staticmethod
    def evaluate_group_performance(task_results: Dict) -> Dict[str, float]:
        return {
            'solution_quality': TaskPerformanceMetrics._calculate_solution_quality(task_results),
            'collaboration_efficiency': TaskPerformanceMetrics._calculate_collaboration_efficiency(task_results),
            'crisis_response': TaskPerformanceMetrics._evaluate_crisis_response(task_results),
            'resource_management': TaskPerformanceMetrics._evaluate_resource_management(task_results),
            'adaptation_speed': TaskPerformanceMetrics._calculate_adaptation_speed(task_results)
        }
    
    @staticmethod
    def _calculate_solution_quality(results: Dict) -> float:
        quality_factors = [
            results.get('technical_accuracy', 0),
            results.get('solution_completeness', 0),
            results.get('innovation_level', 0),
            results.get('feasibility', 0)
        ]
        return np.mean(quality_factors)
    
    @staticmethod
    def _calculate_collaboration_efficiency(results: Dict) -> float:
        collaboration_metrics = [
            results.get('communication_effectiveness', 0),
            results.get('role_synergy', 0),
            results.get('resource_sharing', 0),
            results.get('decision_consensus', 0)
        ]
        return np.mean(collaboration_metrics)
    
    @staticmethod
    def _evaluate_crisis_response(results: Dict) -> float:
        response_metrics = [
            results.get('response_time', 0),
            results.get('solution_effectiveness', 0),
            results.get('team_coordination', 0),
            results.get('adaptation_quality', 0)
        ]
        return np.mean(response_metrics)
