from datetime import datetime, timedelta
import random
from typing import List
from .models import ExperimentSession, Group, Participant, SpaceType, Role
from .surveys import SurveyManager
from .challenges import ChallengeManager
from .agent import Agent, AgentPersonality

class ExperimentController:
    def __init__(self):
        self.session = None
        self.survey_manager = SurveyManager()
        self.challenge_manager = ChallengeManager()
    
    def setup_experiment(self, participant_count: int):
        self.session = ExperimentSession(
            id=random.randint(1000, 9999),
            start_time=datetime.now()
        )
        
        agents = []
        for i in range(participant_count):
            personality = AgentPersonality(
                extroversion=random.random(),
                neuroticism=random.random(),
                agreeableness=random.random(),
                leadership=random.random()
            )
            agent = Agent(i, f"Agent{i}", personality)
            agents.append(agent)
        
        groups = [
            Group(1, SpaceType.CONFINED, True),
            Group(2, SpaceType.CONFINED, False),
            Group(3, SpaceType.OPEN, True),
            Group(4, SpaceType.OPEN, False)
        ]
        
        for agent in agents:
            group = random.choice(groups)
            group.participants.append(agent)
        
        self.session.groups = groups
    
    def run_experiment(self):
        self.conduct_pre_surveys()
        self.run_main_phase()
        self.conduct_post_surveys()
        self.analyze_results()
    
    def run_main_phase(self):
        phases = [
            ("setup", 15),
            ("initial_collaboration", 30),
            ("crisis_one", 45),
            ("adaptation", 45),
            ("crisis_two", 45),
            ("final_resolution", 30)
        ]
        
        for phase, duration in phases:
            self._execute_phase(phase, duration)
    
    def _execute_phase(self, phase: str, duration: int):
        self.session.current_phase = phase
        
        # Update all agents
        for group in self.session.groups:
            environment = {"is_confined": group.space_type == SpaceType.CONFINED}
            for agent in group.participants:
                agent.update(environment, [p for p in group.participants if p != agent])
        
        if phase in ["crisis_one", "crisis_two"]:
            for group in self.session.groups:
                challenge = self.challenge_manager.trigger_random_challenge(group)
                self._handle_challenge(group, challenge)
        
    
    def _handle_challenge(self, group: Group, challenge: Challenge):
        options = ["solve_individually", "collaborate", "seek_leadership"]
        decisions = []
        for agent in group.participants:
            context = {"leadership_required": challenge.difficulty > 3}
            decision = agent.make_decision(options, context)
            decisions.append(decision)
        
        # Calculate success probability based on decisions and agent states
        collaboration_count = decisions.count("collaborate")
        leadership_seekers = decisions.count("seek_leadership")
        
        success_prob = (
            collaboration_count * 0.2 +  # Collaboration bonus
            leadership_seekers * 0.1 +  # Leadership bonus
            (1 - sum(a.stress for a in group.participants)/len(group.participants)) * 0.3  # Stress penalty
        )
        
        if random.random() < success_prob:
            group.challenges_completed += 1
            group.collective_score += 50
        else:
            group.collective_score -= 30
    
    def analyze_results(self):
        results = {
            "completion_time": datetime.now() - self.session.start_time,
            "group_performances": [],
            "stress_levels": [],
            "successful_challenges": 0
        }
        
        for group in self.session.groups:
            results["group_performances"].append({
                "group_id": group.id,
                "final_score": group.collective_score,
                "challenges_completed": group.challenges_completed
            })
            
            for participant in group.participants:
                results["stress_levels"].append({
                    "participant_id": participant.id,
                    "final_stress": participant.stress_level,
                    "final_points": participant.survival_points
                })
        
        return results

def main():
    controller = ExperimentController()
    controller.setup_experiment(120)  # 120 participants
    controller.run_experiment()
    
    # Generate and display results
    analyzer = ExperimentAnalyzer(controller.session.groups)
    analyzer.plot_stress_distributions()
    print("Summary Statistics:", analyzer.generate_summary_statistics())

if __name__ == "__main__":
    main()
