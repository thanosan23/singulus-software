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
        agents = [AIAgent(id=i, openai_key=self.openai_key) for i in range(10)]
        total_adapt = 0
        total_stress = 0
        for agent in agents:
            base_adapt = agent.adaptation_score
            base_stress = agent.stress_level
            if solution == "adaptive_architecture":
                adjusted_adapt = base_adapt + 10
                adjusted_stress = max(0, base_stress * 0.85)
            elif solution == "physical_integration":
                adjusted_adapt = base_adapt + 5
                adjusted_stress = max(0, base_stress * 0.90)
            elif solution == "therapeutic_integration":
                adjusted_adapt = base_adapt + 8
                adjusted_stress = max(0, base_stress * 0.88)
            else:
                adjusted_adapt = base_adapt
                adjusted_stress = base_stress

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
        plt.show()
        print("Comparative graph saved to experiment_results/solution_performance.png")

if __name__ == "__main__":
    tester = SolutionTester()
    best_sol, eval_metrics = tester.evaluate_solutions()
    print("Evaluation Metrics:", eval_metrics)
    print("Best Solution:", best_sol)
    tester.plot_comparative_graphs()
