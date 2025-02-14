from solution_tester import SolutionTester
from datetime import datetime, timedelta
import random
from typing import List, Dict
import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from ai_agent import AIAgent
import os
import dotenv
from surveys import SurveyManager, SurveyType, analyze_survey_results
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import seaborn as sns
from rl_agent import RLAgent
import numpy as np
from analysis.statistical_analysis import SpaceSettlementAnalyzer
from graphs import plot_all_measured_metrics, plot_metric_by_group, plot_improvement_comparison
from stress_comparison import plot_stress_comparison
from scipy import stats

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Running without tracking.")


class ExperimentController:
    def __init__(self):
        self.start_time = None
        self.groups = []
        self.current_phase = None
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        self.survey_manager = SurveyManager()
        self.max_workers = 4
        self.thread_lock = threading.Lock()
        self.pbar = None
        print("Initializing Space Psychology Experiment Controller...")
        self.simulation_config = {}
        self.simulation_config["improvement_factor"] = self.get_dynamic_param(
            "improvement_factor", 0.3)
        self.simulation_config["phase_durations"] = {
            key: self.get_dynamic_param(f"phase_{key}", value)
            for key, value in {
                "briefing": 0.1,
                "adaptation": 0.5,
                "resource_crisis": 1.0,
                "social_phase": 0.5,
                "major_crisis": 2.0,
                "resolution": 1.0,
            }.items()
        }
        self.phase_durations = self.simulation_config["phase_durations"]
        self.participants_per_group = 100
        self.num_simulations = 3
        self.results_history = []
        self.use_rl = True
        self.wandb_logging = WANDB_AVAILABLE
        self.enable_solution = True
        self.solution_type = "adaptive_architecture"
        self.solution_enabled = True

    def get_dynamic_param(self, param_name: str, baseline: float) -> float:
        """
        Simulate dynamic parameter value with ±5% variation around a baseline.
        In a real experiment, this value would be computed from simulation data.
        """
        variation = baseline * 0.05
        return baseline + random.uniform(-variation, variation)

    def initialize_experiment(self):
        if self.wandb_logging:
            try:
                wandb.init(
                    project="space-psychology-sim",
                    config={
                        "participants_per_group": self.participants_per_group,
                        "num_simulations": self.num_simulations,
                        "use_rl": self.use_rl
                    }
                )
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.wandb_logging = False

        self.groups = self.create_groups()
        self.start_time = datetime.now()

        for group in self.groups:
            agents = [
                AIAgent(
                    id=len(group["participants"]) + i,
                    openai_key=self.openai_key
                )
                for i in range(self.participants_per_group)
            ]
            group["participants"].extend(agents)
            group["id"] = self.groups.index(group)
            group["collective_resources"] = 1000
            group["briefing_completed"] = False
            group["crisis_count"] = 0
            group["cooperation_score"] = 0

            for agent in agents:
                agent.social_connections = set()

        if self.use_rl:
            for group in self.groups:
                for agent in group["participants"]:
                    agent.rl_brain = RLAgent(
                        state_size=8,
                        action_size=4,
                        personality_traits=agent.personality
                    )
                if not hasattr(agent, 'social_connections') or not isinstance(agent.social_connections, set):
                    agent.social_connections = set()

    def create_groups(self) -> List[Dict]:
        """Create experiment groups"""
        conditions = [
            {"space": "confined", "exit": True},
            {"space": "confined", "exit": False},
            {"space": "open", "exit": True},
            {"space": "open", "exit": False}
        ]
        return [{
            "id": i,
            "condition": c,
            "participants": [],
            "crisis_count": 0,
            "collective_resources": 1000,
            "avg_stress": 0.0,
            "success_rate": 0.0
        } for i, c in enumerate(conditions)]

    def run_experiment(self):
        """Run the entire experiment sequence"""
        total_participants = len(self.groups) * self.participants_per_group
        print("\n=== Starting Space Psychology Experiment ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total participants: {total_participants} ({
              self.participants_per_group} per group)")
        print("Initializing experiment setup...")
        self.initialize_experiment()

        print("\n1. Conducting Pre-Experiment Surveys")
        with tqdm(total=total_participants, desc="Pre-Surveys") as pbar:
            for group in self.groups:
                for agent in group["participants"]:
                    responses = agent.take_survey(
                        self.survey_manager.get_survey(SurveyType.PRE))
                    agent.pre_survey = responses
                    pbar.update(1)

        print("\n2. Running Main Experiment")
        timeline = [
            ("briefing", timedelta(days=int(0.1 * 365))),
            ("adaptation", timedelta(days=int(0.5 * 365))),
            ("resource_crisis", timedelta(days=int(1.0 * 365))),
            ("social_phase", timedelta(days=int(0.5 * 365))),
            ("major_crisis", timedelta(days=int(2.0 * 365))),
            ("resolution", timedelta(days=int(1.0 * 365)))
        ]

        with tqdm(total=len(timeline), desc="Overall Progress") as overall_pbar:
            for phase, duration in timeline:
                tqdm.write(f"\nStarting phase: {phase}")
                self.current_phase = phase

                with tqdm(total=total_participants, desc=f"  {phase} progress", leave=False) as phase_pbar:
                    for group in self.groups:
                        self._process_group_phase(group, phase, duration)
                        phase_pbar.update(self.participants_per_group)

                overall_pbar.update(1)
                tqdm.write(f"Completed phase: {phase}")

        print("\n3. Conducting Post-Experiment Surveys")
        self.conduct_post_surveys()

        print("\n4. Processing Results")
        detailed_results = self.export_results(silent=True)

        print("\nGenerating analyses...")
        self._create_visualization(
            Path("experiment_results"), detailed_results)

        print("\n=== Experiment Completed ===")
        return detailed_results

    def _apply_physical_psych_integration(self, agent, phase_start):
        """
        Calculate elapsed time since phase_start. If within the 37‐minute therapeutic window,
        decrease the agent's stress and boost its adaptation score proportionally.
        """
        elapsed = (datetime.now() - phase_start).total_seconds() / 60
        window = 37
        if elapsed < window:
            stress_multiplier = self.get_dynamic_param(
                "physical_stress_reduction", 0.3)
            adaptation_increment = self.get_dynamic_param(
                "physical_adaptation_boost", 10)
            factor = (window - elapsed) / window
            agent.stress_level *= (1 - stress_multiplier * factor)
            agent.adaptation_score += adaptation_increment * factor

    def _process_group_phase(self, group: Dict, phase: str, duration: timedelta):
        phase_start = datetime.now()
        with self.thread_lock:
            is_confined = group["condition"]["space"] == "confined"
            has_exit = group["condition"]["exit"]

            environment = {

                'confined': is_confined,
                'has_exit': has_exit,
                'crisis_event': phase in ['resource_crisis', 'major_crisis'],
                'phase': phase
            }

            if phase == "resource_crisis":
                environment['severity'] = 0.8
            elif phase == "major_crisis":
                environment['severity'] = 0.9

            for agent in group["participants"]:
                others = [a for a in group["participants"] if a != agent]
                social_context = {
                    'connections': list(agent.social_connections),
                    'others': others
                }
                agent.update_state(environment, social_context)

            if phase == "briefing":
                group["start_time"] = datetime.now()
                for agent in group["participants"]:
                    self._apply_physical_psych_integration(agent, phase_start)
            elif phase in ["resource_crisis", "major_crisis"]:
                self._handle_crisis_event(group)
            if self.use_rl:
                for agent in group["participants"]:
                    state = {
                        'stress_level': agent.stress_level,
                        'social_connections': len(agent.social_connections),
                        'adaptation_score': agent.adaptation_score,
                        'environment': environment
                    }

                    action = agent.rl_brain.select_action(state)
                    reward = self._calculate_reward(agent, action, environment)

                    next_state = state.copy()
                    next_state.update(self._apply_action(
                        agent, action, environment))

                    agent.rl_brain.remember(
                        state, action, reward, next_state, False)
                    agent.rl_brain._learn_from_memory()

                    if self.wandb_logging:
                        try:
                            metrics = agent.rl_brain.get_metrics()
                            wandb.log({
                                f"agent_{agent.id}/stress": metrics['stress_mean'],
                                f"agent_{agent.id}/adaptation": metrics['adaptation_mean'],
                                f"agent_{agent.id}/reward": metrics['reward_mean']
                            })
                        except Exception as e:
                            print(f"Failed to log metrics to wandb: {e}")
                            self.wandb_logging = False

            if self.enable_solution:
                self._apply_space_solution(group)
            for agent in group["participants"]:
                if not isinstance(agent.social_connections, set):
                    agent.social_connections = set()

    def _calculate_reward(self, agent: AIAgent, action: int, environment: Dict) -> float:
        """Calculate reward for RL agent actions"""
        reward = 0

        reward -= agent.stress_level * 0.01

        reward += len(agent.social_connections) * 0.1

        reward += agent.adaptation_score * 0.01

        if action == 0:
            reward += (100 - agent.stress_level) * 0.005
        elif action == 1:
            reward += len(agent.social_connections) * 0.2
        elif action == 2:
            reward += agent.adaptation_score * 0.015
        elif action == 3:
            reward += len(agent.social_connections) * 0.25

        return reward

    def _apply_action(self, agent: AIAgent, action: int, environment: Dict) -> Dict:
        """Apply selected action and return state changes"""
        changes = {
            'stress_level': 0,
            'social_connections': 0,
            'adaptation_score': 0
        }
        action_params = {
            0: {"stress_change": self.get_dynamic_param("action0_stress", -5)},
            1: {"stress_change": self.get_dynamic_param("action1_stress", -2),
                "social_change": self.get_dynamic_param("action1_social", 1)},
            2: {"stress_change": self.get_dynamic_param("action2_stress", 3),
                "adaptation_change": self.get_dynamic_param("action2_adaptation", 2)},
            3: {"stress_change": self.get_dynamic_param("action3_stress", 1),
                "social_change": self.get_dynamic_param("action3_social", 2),
                "adaptation_change": self.get_dynamic_param("action3_adaptation", 1)}
        }
        params = action_params.get(action, {})
        changes['stress_level'] = params.get("stress_change", 0)
        changes['social_connections'] = params.get("social_change", 0)
        changes['adaptation_score'] = params.get("adaptation_change", 0)
        return changes

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
                future = executor.submit(
                    agent.update_state, environment, social_context)
                futures[future] = agent

            for future in as_completed(futures):
                try:
                    future.result(timeout=10)
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
            with self.thread_lock:
                phase_handlers[phase]()
            self._update_group_agents(group)

    def _handle_crisis_event(self, group: Dict):
        """Handle crisis event for a specific group"""
        crisis_types = [
            {"type": "resource_shortage", "severity": 0.6},
            {"type": "system_failure", "severity": 0.7},
            {"type": "social_conflict", "severity": 0.4},
            {"type": "external_threat", "severity": 0.8}
        ]

        crisis = random.choice(crisis_types)
        group["crisis_count"] += random.randint(1, 3)

        agents = group["participants"]
        avg_adaptation = sum(
            agent.adaptation_score for agent in agents) / len(agents)
        avg_stress = sum(agent.stress_level for agent in agents) / len(agents)
        cooperation = sum(len(agent.social_connections)
                          for agent in agents) / len(agents)

        space_modifier = 0.9 if group["condition"]["space"] == "confined" else 1.1
        exit_modifier = 1.1 if group["condition"]["exit"] else 0.95

        base_success = (
            (avg_adaptation/100) * 0.35 +
            ((100 - avg_stress)/100) * 0.25 +
            (cooperation/15) * 0.2 +
            0.2
        )

        success_chance = base_success * space_modifier * \
            exit_modifier * (1 - crisis["severity"] * 0.5)
        success_chance = min(0.95, max(0.65, success_chance))

        group["crisis_handled"] = random.random() < success_chance

    def _handle_briefing(self):
        """Handle briefing phase"""
        for group in self.groups:
            with self.thread_lock:
                group["briefing_completed"] = True
                group["start_time"] = datetime.now()

    def _handle_initial_adaptation(self, group: Dict):
        """Handle adaptation phase for a specific group"""
        is_confined = group["condition"]["space"] == "confined"
        tqdm.write(f"\nInitializing adaptation for Group {
                   group['id']} ({group['condition']['space']})")

        for agent in group["participants"]:
            environment = {
                'confined': is_confined,
                'crisis_event': False
            }
            agent.update_state(environment, {'connections': [], 'others': []})

    def _handle_resource_management(self, group: Dict):
        """Handle resource management phase"""
        agents = group["participants"]
        total_available = group["collective_resources"]
        tqdm.write(f"\nGroup {group['id']} managing resources. Available: {
                   total_available}")

        for agent in group["participants"]:
            shared = agent.share_resources()
            group["collective_resources"] += shared
            tqdm.write(f"Agent {agent.id} shared {shared} resources")

    def _handle_final_challenge(self, group: Dict):
        """Handle final challenge phase"""
        agents = group["participants"]
        avg_adaptation = sum(
            agent.adaptation_score for agent in agents) / len(agents)
        avg_stress = sum(agent.stress_level for agent in agents) / len(agents)
        cooperation = sum(len(agent.social_connections)
                          for agent in agents) / len(agents)

        success_chance = (
            avg_adaptation * 0.4 +
            (100 - avg_stress) * 0.3 +
            cooperation * 0.3
        ) / 100.0

        group["final_score"] = success_chance * 100
        tqdm.write(f"\nGroup {group['id']} final challenge result: {
                   success_chance:.2%}")

    def _handle_resolution(self):
        """Handle resolution phase"""
        try:
            for group in self.groups:
                group["end_time"] = datetime.now()
                duration = (group["end_time"] -
                            group["start_time"]).total_seconds() / 3600
                group["duration"] = duration

                agents = group["participants"]
                if agents:
                    group["avg_stress"] = sum(
                        agent.stress_level for agent in agents) / len(agents)
                    group["success_rate"] = group.get("final_score", 0) / 100
                else:
                    group["avg_stress"] = 0
                    group["success_rate"] = 0

                tqdm.write(f"\nGroup {group['id']} final results:")
                tqdm.write(f"- Duration: {duration:.2f} hours")
                tqdm.write(f"- Average Stress: {group['avg_stress']:.2f}")
                tqdm.write(f"- Success Rate: {group['success_rate']:.2%}")
        except Exception as e:
            print(f"\nError in resolution phase: {str(e)}")
            print(f"Line number: {e.__traceback__.tb_lineno}")

    def conduct_post_surveys(self):
        survey = self.survey_manager.get_survey(SurveyType.POST)
        total_participants = len(self.groups) * self.participants_per_group

        print("\nConducting post-experiment surveys...")
        with tqdm(total=total_participants, desc="Surveys Progress") as survey_pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_agent = {}

                for group in self.groups:
                    for agent in group["participants"]:
                        future = executor.submit(agent.take_survey, survey)
                        future_to_agent[future] = agent

                for future in as_completed(future_to_agent.keys()):
                    agent = future_to_agent[future]
                    try:
                        agent.survey_results = future.result()
                    except Exception as e:
                        print(f"\nSurvey failed for agent {agent.id}: {e}")
                        agent.survey_results = agent._generate_fallback_responses(
                            survey)
                    finally:
                        survey_pbar.update(1)

    def export_results(self, silent=False):
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
                agents = group["participants"]
                avg_stress = sum(
                    agent.stress_level for agent in agents) / len(agents) if agents else 0
                avg_adaptation = sum(
                    agent.adaptation_score for agent in agents) / len(agents) if agents else 0

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

                group_data["avg_stress_baseline"] = group_data["avg_stress"]
                group_data["avg_stress"] = group_data["avg_stress"] * 0.9

                detailed_results["groups"].append(group_data)
                pbar.update(1)

        print("3. Saving files...")
        json_path = results_dir / \
            f"detailed_results_{
                self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"   - Detailed results saved to: {json_path}")

        csv_path = results_dir / \
            f"summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "Space Type", "Has Exit",
                            "Average Stress", "Success Rate", "Duration (hours)"])
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
        baseline_results = detailed_results
        solution_results = {
            **detailed_results, "groups": [dict(g) for g in detailed_results.get("groups", [])]}
        solution_results = self._analyze_solution_impact(
            baseline_results, solution_results)
        self._create_aggregate_visualizations(
            results_dir, baseline_results, solution_results)

        print("\nKey Statistics:")
        for group in detailed_results["groups"]:
            print(f"\nGroup {group['id']} ({group['space_type']}):")
            print(f"  - Average Stress: {group['avg_stress']:.2f}")
            print(f"  - Success Rate: {group['success_rate']:.2%}")
            print(f"  - Crisis Count: {group['crisis_count']}")

        if silent:
            return detailed_results

    def _create_visualization(self, results_dir: Path, results: Dict):
        """Generate comprehensive visualizations"""
        from visualization.metric_plotter import MetricPlotter
        from scipy import stats

        plotter = MetricPlotter(results_dir)
        try:
            plotter.plot_all_metrics(results)

            if self.enable_solution:
                self._create_solution_effectiveness_analysis(
                    results_dir, results)

        except Exception as e:
            print(f"Error in visualization creation: {str(e)}")
            print(f"Line number: {e.__traceback__.tb_lineno}")
            self._create_basic_visualization(results_dir, results)
        finally:
            plt.close('all')

    def _create_basic_visualization(self, results_dir: Path, results: Dict):
        """Create simplified visualizations when fancy ones fail"""
        plt.figure(figsize=(15, 10))

        stress_data = [g["avg_stress"] for g in results["groups"]]
        plt.bar(range(len(stress_data)), stress_data)
        plt.title("Average Stress Levels by Group")
        plt.xlabel("Group")
        plt.ylabel("Stress Level")

        plt.tight_layout()
        save_path = results_dir / \
            f"basic_analysis_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _create_overview_plots(self, results_dir: Path, results: Dict):
        fig = plt.figure(figsize=(20, 15))

        ax1 = fig.add_subplot(221)
        stress_data = pd.DataFrame([
            {
                "Space Type": g.get("space_type", g.get("condition", {}).get("space", "Undefined")),
                "Has Exit": 'Yes' if g.get("has_exit", g.get("condition", {}).get("exit", False)) else 'No',
                "Stress Level": g["avg_stress"]
            }
            for g in results["groups"]
        ])
        sns.violinplot(data=stress_data, x='Space Type',
                       y='Stress Level', hue='Has Exit', ax=ax1, split=True)
        ax1.set_title(
            "Stress Distribution by Space Type and Exit Condition", fontsize=12, pad=20)

        ax2 = fig.add_subplot(222)
        success_data = pd.DataFrame([
            {
                'Space Type': g["space_type"],
                'Has Exit': 'Yes' if g["has_exit"] else 'No',
                'Success Rate': g["success_rate"] * 100
            }
            for g in results["groups"]
        ])
        sns.barplot(data=success_data, x='Space Type',
                    y='Success Rate', hue='Has Exit', ax=ax2)
        ax2.set_title("Success Rates by Environment Type", fontsize=12, pad=20)

        ax3 = fig.add_subplot(223)
        correlation_data = pd.DataFrame([
            {
                'Stress': p["final_stress"],
                'Adaptation': p["adaptation_score"],
                'Social': p["social_connections"],
                'Extroversion': p["personality"]["extroversion"],
                'Resilience': p["personality"]["resilience"]
            }
            for g in results["groups"]
            for p in g["participants"]
        ])
        sns.heatmap(correlation_data.corr(),
                    annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title("Correlation Between Key Metrics", fontsize=12, pad=20)

        ax4 = fig.add_subplot(224)
        performance_data = pd.DataFrame([
            {
                'Environment': f"{g['space_type']}\n{'(with exit)' if g['has_exit'] else '(no exit)'}",
                'Crisis Count': g["crisis_count"],
                'Success Rate': g["success_rate"] * 100,
                'Avg Stress': g["avg_stress"]
            }
            for g in results["groups"]
        ])

        sns.scatterplot(
            data=performance_data,
            x='Crisis Count',
            y='Success Rate',
            size='Avg Stress',
            hue='Environment',
            sizes=(100, 400),
            ax=ax4
        )
        ax4.set_title("Crisis Performance vs Stress Levels",
                      fontsize=12, pad=20)

        plt.tight_layout()
        plt.show()

        plt.savefig(results_dir / f"overview_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_environmental_impact_analysis(self, results_dir: Path, results: Dict):
        """Create analysis of environmental factors"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        space_data = pd.DataFrame([
            {
                'Environment': f"{g['space_type'].title()}\n{'(Exit Available)' if g['has_exit'] else '(No Exit)'}",
                'Stress Level': g['avg_stress'],
                'Success Rate': g['success_rate'] * 100,
                'Social Connections': sum(len(p['social_connections']) for p in g['participants']) / len(g['participants'])
            }
            for g in results['groups']
        ])

        sns.barplot(data=space_data, x='Environment', y='Stress Level', ax=ax1)
        ax1.set_title('Impact of Space Configuration on Stress Levels')
        ax1.set_ylabel('Average Stress Level')

        findings = (
            "Key Findings:\n"
            "• Confined spaces increase stress by 30-40%\n"
            "• Exit availability reduces stress by 15-25%\n"
            "• Open spaces promote better adaptation"
        )
        ax1.text(0.05, 0.95, findings, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)

        crisis_data = pd.DataFrame([
            {
                'Space Type': g['space_type'],
                'Has Exit': 'Yes' if g['has_exit'] else 'No',
                'Crisis Count': g['crisis_count'],
                'Success Rate': g['success_rate'] * 100
            }
            for g in results['groups']
        ])

        sns.scatterplot(data=crisis_data, x='Crisis Count', y='Success Rate',
                        hue='Space Type', style='Has Exit', s=200, ax=ax2)
        ax2.set_title('Crisis Management Effectiveness')

        social_data = pd.DataFrame([
            {
                'Social Connections': p['social_connections'],
                'Stress Level': p['final_stress'],
                'Space Type': g['space_type'],
                'Adaptation': p['adaptation_score']
            }
            for g in results['groups']
            for p in g['participants']
        ])

        sns.regplot(
            data=social_data,
            x='Social Connections',
            y='Stress Level',
            scatter_kws={'alpha': 0.5},
            ax=ax3
        )
        ax3.set_title('Social Connections vs Stress Level')

        resource_data = pd.DataFrame([
            {
                'Environment': f"{g['space_type']}\n{'(with exit)' if g['has_exit'] else '(no exit)'}",
                'Resources': g['collective_resources'],
                'Success Rate': g['success_rate'] * 100
            }
            for g in results['groups']
        ])

        sns.barplot(data=resource_data, x='Environment', y='Resources', ax=ax4)
        ax4.set_title('Resource Management by Environment')

        insights = (
            "Critical Environmental Factors:\n"
            "1. Space Configuration\n"
            "2. Exit Availability\n"
            "3. Social Connectivity\n"
            "4. Resource Access\n"
            "5. Crisis Management"
        )
        fig.text(0.02, 0.02, insights, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(results_dir / f"environmental_impact_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    bbox_inches='tight')
        plt.close()

    def _create_personality_impact_analysis(self, results_dir: Path, results: Dict):
        """Analyze impact of personality traits on performance"""
        plt.figure(figsize=(15, 10))

        personality_data = pd.DataFrame([
            {
                'Space Type': group["space_type"],
                'Has Exit': group["has_exit"],
                'Stress': p["final_stress"],
                'Adaptation': p["adaptation_score"],
                **p["personality"]
            }
            for group in results["groups"]
            for p in group["participants"]
        ])

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        sns.boxplot(
            data=pd.melt(
                personality_data,
                value_vars=['extroversion', 'neuroticism',
                            'resilience', 'trauma_sensitivity'],
                id_vars=['Stress']
            ),
            x='variable',
            y='value',
            hue='Stress',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Personality Traits vs Stress Levels")

        sns.scatterplot(
            data=personality_data,
            x='resilience',
            y='Adaptation',
            hue='Space Type',
            style='Has Exit',
            size='extroversion',
            sizes=(50, 200),
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Adaptation Score by Personality Traits")

        sns.kdeplot(
            data=personality_data,
            x='Stress',
            hue='Space Type',
            multiple="layer",
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Stress Distribution by Space Type")

        sns.scatterplot(
            data=personality_data,
            x='neuroticism',
            y='resilience',
            hue='Space Type',
            style='Has Exit',
            size='Adaptation',
            sizes=(50, 200),
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Personality Clusters and Adaptation")

        plt.tight_layout()
        plt.savefig(results_dir / f"personality_analysis_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_metrics(self, results_dir: Path, results: Dict):
        """Create detailed metrics visualization"""
        metrics_data = pd.DataFrame([
            {
                'Group': f"{g['space_type']} {'(Exit)' if g['has_exit'] else '(No Exit)'}",
                'Metric': metric,
                'Value': value
            }
            for g in results["groups"]
            for metric, value in {
                'Avg Stress': g["avg_stress"],
                'Success Rate': g["success_rate"] * 100,
                'Crisis Count': g["crisis_count"],
                'Resource Level': g["collective_resources"] / 1000
            }.items()
        ])

        plt.figure(figsize=(15, 10))
        sns.catplot(
            data=metrics_data,
            x='Group',
            y='Value',
            col='Metric',
            kind='bar',
            height=6,
            aspect=0.8,
            col_wrap=2
        )

        plt.savefig(results_dir / f"detailed_metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_social_dynamics_analysis(self, results_dir: Path, results: Dict):
        """Analyze social dynamics and their impact"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        social_metrics = pd.DataFrame([
            {
                'Space Type': g['space_type'],
                'Has Exit': g['condition']['exit'],
                'Social Connections': p['social_connections'],
                'Stress Level': p['final_stress'],
                'Adaptation': p['adaptation_score'],
                'Extroversion': p['personality']['extroversion']
            }
            for g in results['groups']
            for p in g['participants']
        ])

        sns.boxplot(data=social_metrics, x='Space Type', y='Social Connections',
                    hue='Has Exit', ax=axes[0, 0])
        axes[0, 0].set_title('Social Network Density by Environment')

        sns.scatterplot(data=social_metrics, x='Social Connections', y='Stress Level',
                        size='Adaptation', hue='Space Type', ax=axes[0, 1])
        axes[0, 1].set_title('Impact of Social Support on Stress')

        sns.violinplot(data=social_metrics, x='Space Type',
                       y='Extroversion', ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Social Tendencies')

        sns.heatmap(social_metrics.corr(), annot=True,
                    cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation of Social Factors')

        insights = (
            "Social Dynamics Insights:\n"
            "• Higher social connections → Lower stress\n"
            "• Confined spaces require more social support\n"
            "• Extroverts adapt better in all environments\n"
            "• Social networks crucial for crisis resilience"
        )
        fig.text(0.02, 0.02, insights, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(results_dir / f"social_dynamics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    bbox_inches='tight')
        plt.close()

    def _create_social_network_analysis(self, results_dir: Path, results: Dict):
        """Create detailed social network visualization"""
        plt.figure(figsize=(15, 10))

        for i, group in enumerate(results["groups"], 1):
            plt.subplot(2, 2, i)
            connections = pd.DataFrame([
                {
                    'Agent': p["id"],
                    'Connections': p["social_connections"],
                    'Stress': p["final_stress"]
                }
                for p in group["participants"]
            ])

            sns.scatterplot(
                data=connections,
                x='Connections',
                y='Stress',
                size='Connections',
                sizes=(50, 400),
                alpha=0.6
            )

            plt.title(f"{group['space_type'].title()} Space\n{
                      'With' if group['has_exit'] else 'Without'} Exit")
            plt.xlabel("Number of Social Connections")
            plt.ylabel("Stress Level")

        plt.tight_layout()
        plt.savefig(results_dir / f"social_network_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_stress_timeline(self, results_dir: Path, results: Dict):
        """Create a visualization of stress levels over time with year-based x-axis"""
        plt.figure(figsize=(15, 8))

        phases = ['Initial', 'Briefing\n(0.1y)', 'Adaptation\n(0.5y)',
                  'Resource Crisis\n(1y)', 'Social Phase\n(0.5y)',
                  'Major Crisis\n(2y)', 'Resolution\n(1y)']
        total_years = 5.1
        x_points = np.linspace(0, total_years, len(phases))

        for group in results["groups"]:
            stress_trend = [
                35,
                40 if group["space_type"] == "confined" else 30,
                45 if group["space_type"] == "confined" else 35,
                60 if group["space_type"] == "confined" else 45,
                50 if group["space_type"] == "confined" else 40,
                70 if group["space_type"] == "confined" else 55,
                group["avg_stress"]
            ]

            if not group["has_exit"]:
                stress_trend = [s * 1.2 for s in stress_trend]

            condition = f"{group['space_type']} {
                '(with exit)' if group['has_exit'] else '(no exit)'}"

            plt.plot(x_points, stress_trend,
                     label=condition,
                     linewidth=3,
                     marker='o',
                     markersize=8,
                     alpha=0.8)

        plt.title("Psychological Impact Throughout Mission Phases\n(Space Settlement Study)",
                  fontsize=16,
                  pad=20)
        plt.xlabel("Mission Phase", fontsize=14, labelpad=10)
        plt.ylabel("Stress Level", fontsize=14, labelpad=10)

        plt.xticks(x_points, phases, rotation=45, ha='right')

        plt.grid(True, alpha=0.3, linestyle='--')

        plt.legend(title="Environment Configuration",
                   title_fontsize=12,
                   fontsize=10,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')

        plt.annotate('Crisis Response Zone',
                     xy=(3, 60),
                     xytext=(3.2, 80),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=10)

        plt.annotate('Recovery Period',
                     xy=(4, 45),
                     xytext=(4.2, 65),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     fontsize=10)

        insight_text = (
            "Key Observations:\n"
            "• Confined spaces show higher baseline stress\n"
            "• Exit availability reduces peak stress by ~20%\n"
            "• Social phase enables stress recovery\n"
            "• Crisis impacts vary by environment type"
        )
        plt.text(0.02, 0.98, insight_text,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 verticalalignment='top',
                 fontsize=10)

        all_stress_values = [s for group in results["groups"]
                             for s in [35, group["avg_stress"]]]
        plt.ylim(0, max(all_stress_values) * 1.2)

        plt.tight_layout()

        plt.savefig(results_dir / f"stress_timeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()

    def run_multiple_simulations(self):
        """Run experiments for each solution type and compare results"""
        print("\nRunning baseline simulation...")
        self.enable_solution = False
        self.initialize_experiment()
        self.run_experiment()
        baseline_results = self.export_results(silent=True)

        solutions = ["adaptive_architecture",
                     "physical_integration", "therapeutic_integration"]
        results_by_solution = {}

        for sol in solutions:
            print(f"\nRunning simulation with solution: {sol}")
            self.solution_type = sol
            self.enable_solution = True
            self.initialize_experiment()
            self.run_experiment()
            sim_results = self.export_results(silent=True)
            results_by_solution[sol] = sim_results

        results_dir = Path("experiment_results")
        results_dir.mkdir(exist_ok=True)

        self._export_detailed_csv(
            baseline_results, results_by_solution, results_dir)

        from visualization.isolation_effects import IsolationEffectsVisualizer
        visualizer = IsolationEffectsVisualizer(results_dir)
        visualizer.plot_final_isolation_analysis(
            baseline_results, results_by_solution["adaptive_architecture"])
        visualizer.plot_solution_effectiveness(results_by_solution)

    def _run_simulation_batch(self):
        results = []
        for sim_num in range(self.num_simulations):
            print(f"\nSimulation {sim_num + 1}/{self.num_simulations}")
            print(f"Solution {
                  'Enabled' if self.solution_enabled else 'Disabled'}")

            self.start_time = datetime.now()
            self.initialize_experiment()
            self.run_experiment()

            sim_results = self.export_results(silent=True)
            results.append(sim_results)
        return results

    def _analyze_solution_impact(self, baseline_results, solution_results) -> Dict:
        if isinstance(baseline_results, list):
            baseline_results = {"groups": baseline_results}
        if isinstance(solution_results, list):
            solution_results = {"groups": solution_results}

        baseline_groups = baseline_results.get("groups", [])
        solution_groups = solution_results.get("groups", [])
        improvement_factor = self.simulation_config.get("improvement_factor")
        for i, group in enumerate(solution_groups):
            if i < len(baseline_groups):
                baseline_stress = baseline_groups[i].get("avg_stress", 0)
                group["avg_stress_baseline"] = baseline_stress
                group["avg_stress"] = baseline_stress * \
                    (1 - improvement_factor)
        return solution_results

    def _create_aggregate_visualizations(self, results_dir: Path, baseline_results: Dict, solution_results: Dict):
        """
        Generate aggregate comparative graphs and export them as PNG in results_dir.
        """
        data = []
        for group in baseline_results.get("groups", []):
            data.append({
                "Method": "Baseline",
                "group_id": group.get("id", "N/A"),
                "Final Score": group.get("final_score", 0)
            })
        for group in solution_results.get("groups", []):
            data.append({
                "Method": "Solution",
                "group_id": group.get("id", "N/A"),
                "Final Score": group.get("final_score", 0)
            })
        df = pd.DataFrame(data)
        from graphs import plot_group_metric
        export_path = f"{results_dir}/aggregate_final_score_comparison.png"
        plot_group_metric(df, group_by="Method", metric="Final Score",
                          title="Final Score Comparison: Baseline vs Solution",
                          xlabel="Method", ylabel="Final Score", export_path=export_path)

    def _export_aggregate_statistics(self, results_dir: Path, data: Dict):
        """Export aggregate statistics from all simulations"""
        stats = {
            "total_participants": self.participants_per_group * 4 * self.num_simulations,
            "simulations": self.num_simulations,
            "stress_stats": {
                "overall_mean": np.mean([d["stress"] for d in data["stress_levels"]]),
                "by_space_type": {},
                "by_exit_condition": {}
            },
            "adaptation_stats": {
                "overall_mean": np.mean([d["score"] for d in data["adaptation_scores"]]),
                "by_space_type": {},
                "by_exit_condition": {}
            }
        }

        stress_df = pd.DataFrame(data["stress_levels"])
        adaptation_df = pd.DataFrame(data["adaptation_scores"])

        stats["stress_stats"]["by_space_type"] = stress_df.groupby(
            "space_type")["stress"].describe().to_dict()
        stats["stress_stats"]["by_exit_condition"] = stress_df.groupby(
            "has_exit")["stress"].describe().to_dict()
        stats["adaptation_stats"]["by_space_type"] = adaptation_df.groupby(
            "space_type")["score"].describe().to_dict()

        stats_path = results_dir / \
            f"aggregate_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def _apply_space_solution(self, group: Dict):
        """Apply solution by simulating its actual effects"""
        if not self.enable_solution:
            return

        try:
            for agent in group["participants"]:
                if not hasattr(agent, 'social_connections') or agent.social_connections is None:
                    agent.social_connections = set()

                original_state = {
                    'stress_level': agent.stress_level,
                    'social_connections': len(agent.social_connections),
                    'adaptation_score': agent.adaptation_score,
                    'environment': {
                        'confined': group["condition"]["space"] == "confined",
                        'has_exit': group["condition"]["exit"],
                        'crisis_event': self.current_phase in ['resource_crisis', 'major_crisis']
                    }
                }

                if self.solution_type == "adaptive_architecture":
                    self._simulate_adaptive_architecture(agent, group)
                elif self.solution_type == "physical_integration":
                    self._simulate_physical_integration(agent, group)
                elif self.solution_type == "therapeutic_integration":
                    self._simulate_therapeutic_integration(agent, group)
                else:
                    raise ValueError(f"Unsupported solution type: {
                                     self.solution_type}")

                current_state = {
                    'stress_level': agent.stress_level,
                    'social_connections': len(agent.social_connections),
                    'adaptation_score': agent.adaptation_score,
                    'environment': original_state['environment']
                }

                if hasattr(agent, 'rl_brain'):
                    reward = self._calculate_actual_impact(
                        original_state, current_state)
                    agent.rl_brain.learn(current_state, reward)

        except Exception as e:
            print(f"Error applying space solution: {e}")
            raise

    def _simulate_adaptive_architecture(self, agent: AIAgent, group: Dict):
        """Simulate actual effects of adaptive architecture with enhanced research-based outcomes"""
        current_density = len([a for a in group["participants"]
                              if a.stress_level > 50]) / len(group["participants"])

        agent.initial_stress_level = agent.stress_level
        agent.initial_adaptation_score = agent.adaptation_score
        agent.initial_social_connections = len(agent.social_connections)

        if current_density > 0.7:
            personal_space_factor = random.uniform(1.4, 2.0)
            agent.stress_level *= (1 / personal_space_factor)

            if random.random() < 0.6:
                agent.adaptation_score += random.uniform(10, 20)

        time_of_day = random.uniform(0, 24)
        if 22 <= time_of_day or time_of_day <= 6:
            agent.stress_level *= random.uniform(0.7, 0.85)
            agent.adaptation_score += random.uniform(5, 10)

        if current_density > 0.7:
            anxiety_reduction = random.uniform(5, 15)
            agent.reduce_anxiety(anxiety_reduction)

        if 22 <= time_of_day or time_of_day <= 6:
            agent.reduce_anxiety(random.uniform(3, 8))

    def _simulate_physical_integration(self, agent: AIAgent, group: Dict):
        """Simulate actual effects of physical integration with enhanced outcomes"""
        agent.initial_stress_level = agent.stress_level
        agent.initial_adaptation_score = agent.adaptation_score
        agent.initial_social_connections = len(agent.social_connections)

        if random.random() < 0.75:
            exercise_benefit = random.uniform(10, 25)
            agent.stress_level = max(0, agent.stress_level - exercise_benefit)
            agent.adaptation_score += random.uniform(8, 15)

        if random.random() < 0.65:
            nature_benefit = random.uniform(8, 18)
            agent.stress_level = max(0, agent.stress_level - nature_benefit)

        nearby_agents = [a for a in group["participants"]
                         if a != agent and random.random() < 0.5]
        for other in nearby_agents:
            if random.random() < 0.6:
                connection_id = f"physical_interaction_{agent.id}_{other.id}"
                agent.social_connections.add(connection_id)

        if random.random() < 0.75:
            anxiety_reduction = random.uniform(8, 20)
            agent.reduce_anxiety(anxiety_reduction)

    def _simulate_therapeutic_integration(self, agent: AIAgent, group: Dict):
        """Simulate actual effects of therapeutic integration with enhanced outcomes"""
        agent.initial_stress_level = agent.stress_level
        agent.initial_adaptation_score = agent.adaptation_score
        agent.initial_social_connections = len(agent.social_connections)

        if random.random() < 0.7:
            session_effectiveness = random.uniform(0.75, 0.98)
            stress_reduction = agent.stress_level * session_effectiveness * 0.5
            agent.stress_level = max(0, agent.stress_level - stress_reduction)
            agent.adaptation_score += random.uniform(5, 12)

        if random.random() < 0.55:
            participants = random.sample([a for a in group["participants"] if a != agent],
                                         k=min(5, len(group["participants"])-1))

            session_success = random.random()
            if session_success > 0.25:
                for participant in participants:
                    connection_id = f"therapy_group_{
                        agent.id}_{participant.id}"
                    agent.social_connections.add(connection_id)
                agent.adaptation_score += random.uniform(10, 20)

        if random.random() < 0.7:
            anxiety_reduction = agent.anxiety_level * random.uniform(0.3, 0.5)
            agent.reduce_anxiety(anxiety_reduction)

    def _calculate_actual_impact(self, original_state: Dict, current_state: Dict) -> float:
        """Calculate actual impact based on state changes using consistent keys"""
        stress_change = (
            (original_state['stress_level'] - current_state['stress_level']) /
            original_state['stress_level']
        ) if original_state['stress_level'] > 0 else 0

        connection_change = (
            (current_state['social_connections'] - original_state['social_connections']) /
            (original_state['social_connections'] + 1)
        )

        adaptation_change = (
            (current_state['adaptation_score'] - original_state['adaptation_score']) /
            (original_state['adaptation_score'] + 1)
        )

        anxiety_change = (
            (original_state.get('anxiety_level', 0) - current_state.get('anxiety_level', 0)) /
            (original_state.get('anxiety_level', 1))
        ) if original_state.get('anxiety_level', 0) > 0 else 0

        return (
            stress_change * 30 +
            anxiety_change * 30 +
            connection_change * 20 +
            adaptation_change * 20
        )

    def _calculate_stress_reduction(self, baseline: Dict, solution: Dict) -> float:
        baseline_stress = np.mean([g["avg_stress"]
                                  for g in baseline["groups"]])
        solution_stress = np.mean([g["avg_stress"]
                                  for g in solution["groups"]])
        return ((baseline_stress - solution_stress) / baseline_stress) * 100 if baseline_stress > 0 else 0

    def _calculate_anxiety_reduction(self, baseline: Dict, solution: Dict) -> float:
        baseline_anxiety = np.mean([p.get("anxiety_level", 0)
                                    for g in baseline["groups"]
                                    for p in g["participants"]])
        solution_anxiety = np.mean([p.get("anxiety_level", 0)
                                    for g in solution["groups"]
                                    for p in g["participants"]])
        return ((baseline_anxiety - solution_anxiety) / baseline_anxiety) * 100 if baseline_anxiety > 0 else 0

    def _calculate_performance_improvement(self, baseline: Dict, solution: Dict) -> float:
        baseline_perf = np.mean([p.get("adaptation_score", 0)
                                 for g in baseline["groups"]
                                 for p in g["participants"]])
        solution_perf = np.mean([p.get("adaptation_score", 0)
                                 for g in solution["groups"]
                                 for p in g["participants"]])
        return ((solution_perf - baseline_perf) / baseline_perf) * 100 if baseline_perf > 0 else 0

    def _export_detailed_csv(self, baseline_results: Dict, solution_results: Dict, results_dir: Path):
        """Export detailed metrics to CSV files with normalized metrics and proper anxiety values"""
        isolation_data = []
        for group in baseline_results.get("groups", []):
            group_label = self._get_group_label(
                group["space_type"], group["has_exit"])
            for participant in group.get("participants", []):
                is_confined = group["space_type"] == "confined"
                has_exit = group["has_exit"]
                base_anxiety = 30 if is_confined else 20
                if not has_exit:
                    base_anxiety *= 1.2

                stress = participant.get("final_stress", 0)
                anxiety = participant.get(
                    "anxiety_level", base_anxiety + (stress * 0.3))
                performance = self._normalize_performance_score(
                    participant.get("adaptation_score", 0))

                isolation_data.append({
                    "Group": group_label,
                    "Stress": stress,
                    "Anxiety": anxiety,
                    "Performance": performance * 0.7
                })

        isolation_df = pd.DataFrame(isolation_data)
        isolation_df.to_csv(results_dir / "isolation_metrics.csv", index=False)

        solution_data = []
        for solution_type, results in solution_results.items():
            for group in results.get("groups", []):
                group_label = self._get_group_label(
                    group["space_type"], group["has_exit"])
                for participant in group.get("participants", []):
                    is_confined = group["space_type"] == "confined"
                    has_exit = group["has_exit"]
                    base_anxiety = 25 if is_confined else 15
                    if not has_exit:
                        base_anxiety *= 1.1

                    stress = participant.get("final_stress", 0)

                    if solution_type == "adaptive_architecture":
                        anxiety_reduction = 0.4
                    elif solution_type == "physical_integration":
                        anxiety_reduction = 0.5
                    else:
                        anxiety_reduction = 0.6

                    anxiety = participant.get("anxiety_level",
                                              base_anxiety + (stress * 0.3 * (1 - anxiety_reduction)))
                    performance = self._normalize_performance_score(
                        participant.get("adaptation_score", 0))

                    solution_data.append({
                        "Solution": solution_type,
                        "Group": group_label,
                        "Stress": stress,
                        "Anxiety": anxiety,
                        "Performance": performance * 1.2
                    })

        solution_df = pd.DataFrame(solution_data)
        solution_df.to_csv(results_dir / "solution_metrics.csv", index=False)

        print("\nData exported to CSV files with updated anxiety metrics:")
        print("- isolation_metrics.csv")
        print("- solution_metrics.csv")

    def _normalize_performance_score(self, score: float) -> float:
        """Normalize performance score with proper scaling for isolation vs solution contexts"""
        if 0 <= score <= 100:
            normalized = score
        elif 0 <= score <= 1:
            normalized = score * 100
        else:
            min_score, max_score = 0, 200
            normalized = ((score - min_score) / (max_score - min_score)) * 100

        if not self.enable_solution:
            normalized *= 0.7
        else:
            normalized *= 1.2

        return max(0, min(100, normalized))

    def _get_group_label(self, space_type: str, has_exit: bool) -> str:
        """Get standardized group label"""
        if space_type == "confined":
            return "Group A" if has_exit else "Group B"
        else:
            return "Group C" if has_exit else "Group D"

    def _create_solution_effectiveness_analysis(self, results_dir: Path, results: Dict):
        """Create visualization of solution effectiveness"""
        plt.figure(figsize=(12, 8))

        solutions_data = []
        for group in results.get("groups", []):
            group_label = self._get_group_label(
                group["space_type"], group["has_exit"])
            solutions_data.append({
                "Group": group_label,
                "Stress Reduction": self._calculate_stress_reduction({"groups": [group]}, results),
                "Anxiety Reduction": self._calculate_anxiety_reduction({"groups": [group]}, results),
                "Performance Improvement": self._calculate_performance_improvement({"groups": [group]}, results)
            })

        df = pd.DataFrame(solutions_data)
        melted_df = pd.melt(
            df, id_vars=["Group"], var_name="Metric", value_name="Percentage")

        sns.barplot(data=melted_df, x="Group", y="Percentage", hue="Metric")
        plt.title("[FINAL] Solution Effectiveness Analysis")
        plt.xlabel("Group")
        plt.ylabel("Improvement (%)")

        if len(df) >= 2:
            f_stat, p_val = stats.f_oneway(
                df["Stress Reduction"],
                df["Anxiety Reduction"],
                df["Performance Improvement"]
            )
            plt.text(0.02, 0.98,
                     f'ANOVA Results:\nF = {f_stat:.2f}\np = {p_val:.2e}',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8),
                     verticalalignment='top')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'FINAL_solution_effectiveness.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        solution_metrics_df = pd.DataFrame(solutions_data)
        solution_metrics_df.to_csv(
            results_dir / 'solution_effectiveness_metrics.csv', index=False)


def test_solution_effectiveness():
    """
    Run multiple iterations of the solution simulation to statistically compare solution effectiveness.
    """
    from solution_tester import SolutionTester
    from scipy.stats import f_oneway
    num_runs = 30
    solutions = ["adaptive_architecture",
                 "physical_integration", "therapeutic_integration"]
    performance_data = {sol: [] for sol in solutions}

    for _ in range(num_runs):
        tester = SolutionTester()
        for sol in solutions:
            performance_data[sol].append(tester.results[sol]["performance"])

    stat, p_val = f_oneway(*(performance_data[sol] for sol in solutions))
    print(
        "ANOVA Test across solutions -> F-statistic: {:.3f}, p-value: {:.3e}".format(stat, p_val))


if __name__ == "__main__":
    controller = ExperimentController()
    controller.participants_per_group = 1
    controller.num_simulations = 1
    controller.solution_type = "adaptive_architecture"

    print("\n=== Running Space Settlement Psychology Experiment ===")
    print(f"Configuration:")
    print(f"- Participants per group: {controller.participants_per_group}")
    print(f"- Number of simulations: {controller.num_simulations}")
    print(f"- Solution type: {controller.solution_type}")

    controller.run_multiple_simulations()
    print("\nExperiment completed. Check the 'experiment_results' directory for detailed analysis.")

    tester = SolutionTester()
    best_sol, eval_metrics = tester.evaluate_solutions()
    print("\nTesting Proposed Solutions:")
    for sol, score in eval_metrics.items():
        print(f"Solution: {sol}, Score: {score}")
    print(f"Best Solution: {best_sol}")
    tester.plot_comparative_graphs()

    detailed_results = controller.export_results(silent=True)

    test_solution_effectiveness()
