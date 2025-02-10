from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime

class SpaceType(Enum):
    CONFINED = "confined"
    OPEN = "open"

class Role(Enum):
    COMMANDER = "commander"
    ENGINEER = "engineer"
    SCIENTIST = "scientist"
    COMMS_OFFICER = "communications_officer"
    UNASSIGNED = "unassigned"

@dataclass
class Participant:
    id: int
    name: str
    age: int
    gender: str
    role: Role = Role.UNASSIGNED
    survival_points: int = 100
    stress_level: int = 0
    anxiety_level: int = 0
    resource_cards: List[int] = field(default_factory=list)
    survey_responses: Dict = field(default_factory=dict)
    
    def update_points(self, amount: int):
        self.survival_points = max(0, min(200, self.survival_points + amount))
        
    def update_stress(self, amount: int):
        self.stress_level = max(0, min(100, self.stress_level + amount))

@dataclass
class Group:
    id: int
    space_type: SpaceType
    has_exit: bool
    participants: List[Participant] = field(default_factory=list)
    collective_score: int = 1000
    challenges_completed: int = 0
    crisis_events_handled: int = 0
    
    def get_leader(self) -> Optional[Participant]:
        return next((p for p in self.participants if p.role == Role.COMMANDER), None)

@dataclass
class ExperimentSession:
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    groups: List[Group] = field(default_factory=list)
    current_phase: str = "setup"
    is_completed: bool = False
