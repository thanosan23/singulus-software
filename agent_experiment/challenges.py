from typing import List, Callable
import random
from .models import Group, Participant, Role

class Challenge:
    def __init__(self, name: str, difficulty: int, description: str):
        self.name = name
        self.difficulty = difficulty
        self.description = description
        self.completion_criteria: List[Callable] = []
        
    def add_criterion(self, criterion: Callable):
        self.completion_criteria.append(criterion)
        
    def is_completed(self, group: Group) -> bool:
        return all(c(group) for c in self.completion_criteria)

class ChallengeManager:
    def __init__(self):
        self.challenges = self._create_challenges()
        
    def _create_challenges(self) -> List[Challenge]:
        challenges = []
        
        life_support = Challenge(
            "Life Support Failure",
            3,
            "Oxygen generation system is malfunctioning. Team must reallocate resources."
        )
        life_support.add_criterion(
            lambda g: g.collective_score >= 800
        )
        challenges.append(life_support)
        
        comm_crisis = Challenge(
            "Communication Breakdown",
            2,
            "Internal communication systems are down. Team must establish alternative protocols."
        )
        comm_crisis.add_criterion(
            lambda g: any(p.role == Role.COMMS_OFFICER for p in g.participants)
        )
        challenges.append(comm_crisis)
        
        resource_crisis = Challenge(
            "Critical Resource Depletion",
            4,
            "Essential resources are running low. Team must optimize consumption."
        )
        resource_crisis.add_criterion(
            lambda g: all(p.survival_points > 50 for p in g.participants)
        )
        challenges.append(resource_crisis)
        
        return challenges
    
    def trigger_random_challenge(self, group: Group) -> Challenge:
        challenge = random.choice(self.challenges)
        self._apply_challenge_effects(challenge, group)
        return challenge
    
    def _apply_challenge_effects(self, challenge: Challenge, group: Group):
        if challenge.name == "Life Support Failure":
            for p in group.participants:
                p.update_points(-20)
                p.update_stress(15)
        elif challenge.name == "Communication Breakdown":
            for p in group.participants:
                p.update_stress(10)
        elif challenge.name == "Critical Resource Depletion":
            group.collective_score -= 100
            for p in group.participants:
                p.update_points(-10)
