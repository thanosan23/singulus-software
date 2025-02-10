from datetime import datetime, timedelta
import random
from typing import List, Dict
import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ai_agent import AIAgent
import os
import dotenv
from surveys import SurveyManager, SurveyType, analyze_survey_results
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

dotenv.load_dotenv('.env')


class ExperimentController:
    def __init__(self):
        self.start_time = None
        self.groups = []
        self.current_phase = None
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        self.survey_manager = SurveyManager()
        self.max_workers = 4  # Number of parallel threads
        self.thread_lock = threading.Lock()
        self.pbar = None  # Progress bar for overall experiment
        print("Initializing Space Psychology Experiment Controller...")
        self.phase_durations = {  # Real-time duration in hours
            "briefing": 0.5,
            "adaptation": 2.0,
            "resource_crisis": 3.0,
            "social_phase": 2.0,
            "major_crisis": 3.0,
            "resolution": 2.0
        }
        self.participants_per_group = 1  # Keep it to 1 person per group
            
    def initialize_experiment(self):
        self.groups = self.create_groups()
        self.start_time = datetime.now()
        
        for group in self.groups:
            # Create exactly one agent per group
            agent = AIAgent(
                id=0,  # Always ID 0 since only one agent per group
                openai_key=self.openai_key
            )
            group["participants"] = [agent]
            group["id"] = self.groups.index(group)
            group["collective_resources"] = 1000
            group["briefing_completed"] = False
            group["crisis_count"] = 0
            group["cooperation_score"] = 0
        
    def create_groups(self) -> List[Dict]:
        conditions = [
            {"space": "confined", "exit": True},
            {"space": "confined", "exit": False},
            {"space": "open", "exit": True},
            {"space": "open", "exit": False}
        ]
        return [{"condition": c, "participants": []} for c in conditions]
    
    def run_experiment(self):
        print("\n=== Starting Space Psychology Experiment ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Initializing experiment setup...")
        self.initialize_experiment()
        
        print("\n1. Conducting Pre-Experiment Surveys")
        with tqdm(total=len(self.groups) * 1, desc="Pre-Surveys") as pbar:
            for group in self.groups:
                for agent in group["participants"]:
                    responses = agent.take_survey(self.survey_manager.get_survey(SurveyType.PRE))
                    agent.pre_survey = responses
                    pbar.update(1)
        
        print("\n2. Running Main Experiment")
        self.run_main_task()
        
        print("\n3. Conducting Post-Experiment Surveys")
        self.conduct_post_surveys()
        
        print("\n4. Processing Results")
        self.export_results()
        
        print("\n=== Experiment Completed ===")
        print(f"Duration: {datetime.now() - self.start_time}")
        print("Results have been exported to 'experiment_results' directory")
        
    def conduct_pre_surveys(self):
        survey = self.survey_manager.get_survey(SurveyType.PRE)
        for group in self.groups:
            for agent in group["participants"]:
                responses = agent.take_survey(survey)
                agent.pre_survey = responses
    
    def run_main_task(self):
        timeline = [
            ("briefing", timedelta(minutes=30)),
            ("adaptation", timedelta(hours=2)),
            ("resource_crisis", timedelta(hours=3)),
            ("social_phase", timedelta(hours=2)),
            ("major_crisis", timedelta(hours=3)),
            ("resolution", timedelta(hours=2))
        ]
        
        print("\nStarting experiment simulation...")
        try:
            with tqdm(total=len(timeline), desc="Overall Progress") as overall_pbar:
                for phase, duration in timeline:
                    tqdm.write(f"\nStarting phase: {phase}")
                    self.current_phase = phase
                    
                    # Process groups sequentially with parallel agent updates
                    with tqdm(total=len(self.groups), desc=f"  {phase} progress", leave=False) as group_pbar:
                        for group in self.groups:
                            try:
                                self._process_group_phase(group, phase, duration)
                                group_pbar.update(1)
                            except Exception as e:
                                tqdm.write(f"\nError processing group {group['id']} in {phase}: {str(e)}")
                                continue
                    
                    overall_pbar.update(1)
                    tqdm.write(f"Completed phase: {phase}")
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user. Cleaning up...")
            # Add any cleanup code here if needed
            raise
        except Exception as e:
            print(f"\nUnexpected error during experiment: {str(e)}")
            raise

    def _process_group_phase(self, group: Dict, phase: str, duration: timedelta):
        """Process a single group's phase"""
        with self.thread_lock:
            # Update environment info
            is_confined = group["condition"]["space"] == "confined"
            has_exit = group["condition"]["exit"]
            
            environment = {
                'confined': is_confined,
                'has_exit': has_exit,
                'crisis_event': phase in ['resource_crisis', 'major_crisis'],
                'phase': phase
            }
            
            # Apply phase-specific effects
            if phase == "resource_crisis":
                environment['severity'] = 0.8
            elif phase == "major_crisis":
                environment['severity'] = 0.9
            
            # Update all agents with the current environment
            for agent in group["participants"]:
                others = [a for a in group["participants"] if a != agent]
                social_context = {
                    'connections': list(agent.social_connections),
                    'others': others
                }
                agent.update_state(environment, social_context)
            
            # Handle phase-specific logic
            if phase == "briefing":
                group["start_time"] = datetime.now()
            elif phase in ["resource_crisis", "major_crisis"]:
                self._handle_crisis_event(group)

    def _update_group_agents(self, group: Dict):
        """Update all agents in a group concurrently"""
        agents = group["participants"]
        is_confined = group["condition"]["space"] == "confined"
        
        with ThreadPoolExecutor(max_workers=min(len(agents), self.max_workers)) as executor:
            futures = {}
            for agent in agents:
                others = [a for a in agents if a != agent]
                environment = {
                    'confined': is_confined,
                    'crisis_event': self.current_phase in ['resource_crisis', 'major_crisis']
                }
                social_context = {
                    'connections': list(agent.social_connections),
                    'others': others
                }
                future = executor.submit(agent.update_state, environment, social_context)
                futures[future] = agent
            
            # Wait for all updates to complete
            for future in as_completed(futures):
                try:
                    future.result(timeout=10)  # Add timeout to prevent hanging
                except Exception as e:
                    agent = futures[future]
                    tqdm.write(f"Error updating agent {agent.id}: {str(e)}")

    def _update_group(self, group: Dict, phase: str):
        """Update a single group and its agents"""
        phase_handlers = {
            "briefing": self._handle_briefing,
            "adaptation": self._handle_initial_adaptation,
            "resource_crisis": self._handle_resource_management,
            "social_phase": self._handle_crisis_event,
            "major_crisis": self._handle_final_challenge,
            "resolution": self._handle_resolution
        }
        
        if phase in phase_handlers:
            with self.thread_lock:  # Ensure thread safety for shared resources
                phase_handlers[phase]()
            self._update_group_agents(group)

    def _handle_crisis_event(self, group: Dict):
        """Handle crisis event for a specific group"""
        crisis_types = [
            {"type": "resource_shortage", "severity": 0.7},
            {"type": "system_failure", "severity": 0.8},
            {"type": "social_conflict", "severity": 0.5},
            {"type": "external_threat", "severity": 0.9}
        ]
        
        crisis = random.choice(crisis_types)
        group["crisis_count"] += 1
        
        # Calculate success chance with environmental factors
        agents = group["participants"]
        avg_adaptation = sum(agent.adaptation_score for agent in agents) / len(agents)
        avg_stress = sum(agent.stress_level for agent in agents) / len(agents)
        cooperation = sum(len(agent.social_connections) for agent in agents) / len(agents)
        
        # Environmental modifiers
        space_modifier = 0.8 if group["condition"]["space"] == "confined" else 1.2
        exit_modifier = 1.2 if group["condition"]["exit"] else 0.8
        
        base_success = (
            avg_adaptation * 0.4 +
            (100 - avg_stress) * 0.4 +
            cooperation * 0.2
        ) / 100.0
        
        # Apply environmental modifiers
        success_chance = base_success * space_modifier * exit_modifier * (1 - crisis["severity"])
        
        # Determine outcome
        group["crisis_handled"] = random.random() < success_chance

        # Rest of the method...

    def _handle_briefing(self):
        """Handle briefing phase"""
        for group in self.groups:
            with self.thread_lock:
                group["briefing_completed"] = True
                group["start_time"] = datetime.now()

    def _handle_initial_adaptation(self, group: Dict):
        """Handle adaptation phase for a specific group"""
        is_confined = group["condition"]["space"] == "confined"
        tqdm.write(f"\nInitializing adaptation for Group {group['id']} ({group['condition']['space']})")
        
        for agent in group["participants"]:
            environment = {
                'confined': is_confined,
                'crisis_event': False
            }
            agent.update_state(environment, {'connections': [], 'others': []})

    def _handle_resource_management(self, group: Dict):
        """Handle resource management phase"""
        for group in self.groups:
            # Calculate total available resources
            total_available = group["collective_resources"]
            tqdm.write(f"\nGroup {group['id']} managing resources. Available: {total_available}")
            
            # Each agent decides how much to share
            for agent in group["participants"]:
                shared = agent.share_resources()
                group["collective_resources"] += shared
                tqdm.write(f"Agent {agent.id} shared {shared} resources")

    def _handle_final_challenge(self, group: Dict):
        """Handle final challenge phase"""
        for group in self.groups:
            agents = group["participants"]
            # Calculate group performance
            avg_adaptation = sum(agent.adaptation_score for agent in agents) / len(agents)
            avg_stress = sum(agent.stress_level for agent in agents) / len(agents)
            cooperation = sum(len(agent.social_connections) for agent in agents) / len(agents)
            
            success_chance = (
                avg_adaptation * 0.4 +
                (100 - avg_stress) * 0.3 +
                cooperation * 0.3
            ) / 100.0
            
            group["final_score"] = success_chance * 100
            tqdm.write(f"\nGroup {group['id']} final challenge result: {success_chance:.2%}")

    def _handle_resolution(self):
        """Handle resolution phase"""
        for group in self.groups:
            group["end_time"] = datetime.now()
            duration = (group["end_time"] - group["start_time"]).total_seconds() / 3600
            group["duration"] = duration
            
            # Calculate final metrics
            agents = group["participants"]
            group["avg_stress"] = sum(agent.stress_level for agent in agents) / len(agents)
            group["success_rate"] = group["final_score"] / 100
            
            tqdm.write(f"\nGroup {group['id']} final results:")
            tqdm.write(f"- Duration: {duration:.2f} hours")
            tqdm.write(f"- Average Stress: {group['avg_stress']:.2f}")
            tqdm.write(f"- Success Rate: {group['success_rate']:.2%}")

    def conduct_post_surveys(self):
        survey = self.survey_manager.get_survey(SurveyType.POST)
        total_surveys = sum(min(5, len(group["participants"])) for group in self.groups)
        
        print("\nConducting post-experiment surveys...")
        with tqdm(total=total_surveys, desc="Surveys Progress") as survey_pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a dict to map futures to agents
                future_to_agent = {}
                
                # Submit all survey tasks
                for group in self.groups:
                    sampled_agents = random.sample(group["participants"], min(5, len(group["participants"])))
                    for agent in sampled_agents:
                        future = executor.submit(agent.take_survey, survey)
                        future_to_agent[future] = agent
                
                # Process completed surveys
                for future in as_completed(future_to_agent.keys()):
                    agent = future_to_agent[future]
                    try:
                        agent.survey_results = future.result()
                    except Exception as e:
                        print(f"\nSurvey failed for agent {agent.id}: {e}")
                        agent.survey_results = agent._generate_fallback_responses(survey)
                    finally:
                        survey_pbar.update(1)

    def export_results(self):
        print("\nExporting Results:")
        results_dir = Path("experiment_results")
        results_dir.mkdir(exist_ok=True)
        
        print("1. Preparing detailed results...")
        detailed_results = {
            "experiment_time": self.start_time.isoformat(),
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "groups": []
        }
        
        print("2. Processing group data...")
        with tqdm(total=len(self.groups), desc="Processing Groups") as pbar:
            for group in self.groups:
                # Calculate group metrics first
                agents = group["participants"]
                avg_stress = sum(agent.stress_level for agent in agents) / len(agents) if agents else 0
                avg_adaptation = sum(agent.adaptation_score for agent in agents) / len(agents) if agents else 0
                
                # Calculate success rate
                success_chance = (
                    avg_adaptation * 0.4 +
                    (100 - avg_stress) * 0.3 +
                    (group.get("collective_resources", 0) / 1000) * 0.3
                ) / 100.0
                
                group_data = {
                    "id": group["id"],
                    "space_type": group["condition"]["space"],
                    "has_exit": group["condition"]["exit"],
                    "collective_resources": group.get("collective_resources", 0),
                    "crisis_count": group.get("crisis_count", 0),
                    "avg_stress": avg_stress,
                    "success_rate": success_chance,
                    "participants": []
                }
                
                for agent in agents:
                    agent_data = {
                        "id": agent.id,
                        "final_stress": agent.stress_level,
                        "personality": agent.personality,
                        "social_connections": len(agent.social_connections),
                        "pre_survey": agent.pre_survey,
                        "post_survey": agent.survey_results,
                        "adaptation_score": agent.adaptation_score
                    }
                    group_data["participants"].append(agent_data)
                
                detailed_results["groups"].append(group_data)
                pbar.update(1)
        
        print("3. Saving files...")
        # Save JSON
        json_path = results_dir / f"detailed_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"   - Detailed results saved to: {json_path}")
        
        # Save CSV
        csv_path = results_dir / f"summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "Space Type", "Has Exit", "Average Stress", "Success Rate", "Duration (hours)"])
            for group in detailed_results["groups"]:
                writer.writerow([
                    group["id"],
                    group["space_type"],
                    group["has_exit"],
                    group["avg_stress"],
                    group["success_rate"],
                    detailed_results["duration_hours"]
                ])
        print(f"   - Summary saved to: {csv_path}")
        
        print("4. Generating visualizations...")
        self._create_visualization(results_dir, detailed_results)
        
        print("\nKey Statistics:")
        for group in detailed_results["groups"]:
            print(f"\nGroup {group['id']} ({group['space_type']}):")
            print(f"  - Average Stress: {group['avg_stress']:.2f}")
            print(f"  - Success Rate: {group['success_rate']:.2%}")
            print(f"  - Crisis Count: {group['crisis_count']}")

    def _create_visualization(self, results_dir: Path, results: Dict):
        print("\nGenerating visualizations...")
        # Create multiple visualizations
        plt.figure(figsize=(15, 10))
        
        # Stress distribution
        plt.subplot(2, 2, 1)
        for space_type in ["confined", "open"]:
            stress_data = [g["avg_stress"] for g in results["groups"] if g["space_type"] == space_type]
            plt.hist(stress_data, alpha=0.5, label=space_type)
        plt.title("Stress Distribution by Space Type")
        plt.xlabel("Average Stress Level")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Social connections
        plt.subplot(2, 2, 2)
        confined_connections = [p["social_connections"] for g in results["groups"] 
                              if g["space_type"] == "confined" 
                              for p in g["participants"]]
        open_connections = [p["social_connections"] for g in results["groups"] 
                          if g["space_type"] == "open" 
                          for p in g["participants"]]
        
        plt.boxplot([confined_connections, open_connections], labels=["Confined", "Open"])
        plt.title("Social Connections by Space Type")
        plt.ylabel("Number of Connections")
        
        plt.savefig(results_dir / f"analysis_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        print("Visualizations saved successfully!")

if __name__ == "__main__":
    controller = ExperimentController()
    controller.run_experiment()
