import numpy as np
from typing import Dict, List, Set
import networkx as nx

class SocialDynamicsManager:
    def __init__(self, group_size: int):
        self.group_size = group_size
        self.social_graph = nx.Graph()
        self.personality_compatibility_matrix = None
        self.stress_threshold = 75
        
    def initialize_social_dynamics(self, agents: List['AIAgent']):
        """Initialize social network based on personality compatibility"""
        self.personality_compatibility_matrix = np.zeros((self.group_size, self.group_size))
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    compatibility = self._calculate_personality_compatibility(
                        agent1.personality,
                        agent2.personality
                    )
                    self.personality_compatibility_matrix[i, j] = compatibility
                    
                    if compatibility > 0.6:
                        self.social_graph.add_edge(agent1.id, agent2.id, weight=compatibility)
    
    def update_social_connections(self, agents: List['AIAgent'], environment: Dict):
        """Update social connections based on stress, personality, and environment"""
        is_confined = environment.get('confined', False)
        is_crisis = environment.get('crisis_event', False)
        
        for agent in agents:
            others = [a for a in agents if a != agent]
            potential_connections = set()
            
            for other in others:
                connect_prob = self._calculate_connection_probability(
                    agent, other, is_confined, is_crisis
                )
                
                if agent.stress_level > self.stress_threshold:
                    connect_prob *= 0.5
                
                if is_confined:
                    connect_prob *= 1.2
                if is_crisis:
                    connect_prob *= 1.3
                
                if np.random.random() < connect_prob:
                    potential_connections.add(other)
                    self.social_graph.add_edge(agent.id, other.id)
                elif other in agent.social_connections:
                    maintain_prob = 0.8 * (1 - agent.stress_level/100)
                    if np.random.random() > maintain_prob:
                        agent.social_connections.remove(other)
                        if self.social_graph.has_edge(agent.id, other.id):
                            self.social_graph.remove_edge(agent.id, other.id)
            
            agent.social_connections.update(potential_connections)
    
    def _calculate_personality_compatibility(self, p1: Dict, p2: Dict) -> float:
        """Calculate compatibility between two personalities"""
        compatibility = 0.0
        
        extroversion_diff = abs(p1['extroversion'] - p2['extroversion'])
        compatibility += (1 - extroversion_diff) * 0.3
        
        neuroticism_diff = abs(p1['neuroticism'] - p2['neuroticism'])
        compatibility += (1 - neuroticism_diff) * 0.2
        
        resilience_complement = min(p1['resilience'], p2['resilience'])
        compatibility += resilience_complement * 0.3
        
        avg_agreeableness = (p1.get('agreeableness', 0.5) + p2.get('agreeableness', 0.5)) / 2
        compatibility += avg_agreeableness * 0.2
        
        return np.clip(compatibility, 0, 1)
    
    def _calculate_connection_probability(self, agent1: 'AIAgent', agent2: 'AIAgent',
                                       is_confined: bool, is_crisis: bool) -> float:
        """Calculate probability of social connection formation"""
        base_prob = self.personality_compatibility_matrix[agent1.id, agent2.id]
        
        stress_factor = (200 - agent1.stress_level - agent2.stress_level) / 200
        adaptation_factor = (agent1.adaptation_score + agent2.adaptation_score) / 200
        
        environment_modifier = 1.0
        if is_confined:
            environment_modifier *= 1.2
        if is_crisis:
            environment_modifier *= 1.3
            
        return base_prob * stress_factor * adaptation_factor * environment_modifier

    def get_social_metrics(self) -> Dict:
        """Calculate social network metrics"""
        return {
            'density': nx.density(self.social_graph),
            'avg_clustering': nx.average_clustering(self.social_graph),
            'centrality': nx.degree_centrality(self.social_graph),
            'communities': list(nx.community.greedy_modularity_communities(self.social_graph))
        }