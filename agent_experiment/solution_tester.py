import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ai_agent import AIAgent

import os
from dotenv import load_dotenv

load_dotenv('.env')

class SolutionTester:
    def __init__(self):
        self.solutions = ["adaptive_architecture", "physical_integration", "therapeutic_integration"]
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.results = {sol: self._simulate_solution(sol) for sol in self.solutions}

    def _simulate_solution(self, solution: str) -> dict:
        agents = [AIAgent(id=i, openai_key=self.openai_key) for i in range(30)]
        total_adapt = 0
        total_stress = 0
        for agent in agents:
            base_adapt = agent.adaptation_score
            base_stress = agent.stress_level
            if solution == "adaptive_architecture":
                reduction = random.uniform(0.15, 0.35)
                increase = random.uniform(10, 30)
            elif solution == "physical_integration":
                reduction = random.uniform(0.10, 0.30)
                increase = random.uniform(5, 15)
            elif solution == "therapeutic_integration":
                reduction = random.uniform(0.12, 0.32)
                increase = random.uniform(8, 20)
            else:
                reduction = 0.0
                increase = 0

            adjusted_stress = max(0, base_stress * (1 - reduction))
            adjusted_adapt = base_adapt + increase

            total_adapt += adjusted_adapt
            total_stress += adjusted_stress

        avg_adapt = total_adapt / len(agents)
        avg_stress = total_stress / len(agents)
        performance = avg_adapt - avg_stress
        return {"avg_adapt": avg_adapt, "avg_stress": avg_stress, "performance": performance}

    def evaluate_solutions(self):
        evaluation = {}
        for sol, metrics in self.results.items():
            evaluation[sol] = metrics["performance"]
        best_solution = max(evaluation, key=evaluation.get)
        return best_solution, evaluation

    def compare_stress_reduction(self):
        baseline_results = self._simulate_solution("baseline")
        print(f"Baseline Average Stress: {baseline_results['avg_stress']:.2f}")
        for sol in self.solutions:
            sol_results = self._simulate_solution(sol)
            reduction = baseline_results['avg_stress'] - sol_results['avg_stress']
            print(f"Solution '{sol}' Reduced Stress by: {reduction:.2f} (Baseline: {baseline_results['avg_stress']:.2f} vs {sol_results['avg_stress']:.2f})")

    def plot_comparative_graphs(self):
        df = pd.DataFrame([
            {"Solution": sol,
             "Adaptation": vals["avg_adapt"],
             "Stress": vals["avg_stress"],
             "Performance": vals["performance"]}
            for sol, vals in self.results.items()
        ])
        
        plt.figure(figsize=(8,6))
        plt.bar(df["Solution"], df["Performance"], color=['blue','orange','green'])
        plt.title("Overall Performance by Solution (Higher is Better)")
        plt.xlabel("Solution")
        plt.ylabel("Performance Score")
        plt.ylim(min(df["Performance"])-5, max(df["Performance"])+5)
        plt.savefig("experiment_results/solution_performance.png", bbox_inches='tight')
        print("Comparative graph saved to experiment_results/solution_performance.png")

if __name__ == "__main__":
    tester = SolutionTester()
    best_sol, eval_metrics = tester.evaluate_solutions()
    print("Evaluation Metrics:", eval_metrics)
    print("Best Solution:", best_sol)
    tester.plot_comparative_graphs()
