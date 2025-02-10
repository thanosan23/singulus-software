from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Participant:
    id: int
    group_id: int
    survival_points: int = 100
    resource_cards: List[int] = None
    
    def __post_init__(self):
        if self.resource_cards is None:
            self.resource_cards = []
    
    def update_points(self, amount: int):
        self.survival_points = max(0, self.survival_points + amount)
        
    def add_resource_card(self, value: int):
        self.resource_cards.append(value)
        
    def use_resource_card(self, value: int) -> bool:
        if value in self.resource_cards:
            self.resource_cards.remove(value)
            return True
        return False

class Group:
    def __init__(self, id: int, condition: dict):
        self.id = id
        self.condition = condition
        self.participants: List[Participant] = []
        self.collective_score = 1000
        
    def add_participant(self, participant: Participant):
        self.participants.append(participant)
        
    def update_collective_score(self, amount: int):
        self.collective_score = max(0, self.collective_score + amount)
        
    def distribute_resources(self):
        for participant in self.participants:
            values = [10, 20, 30]
            participant.add_resource_card(random.choice(values))
