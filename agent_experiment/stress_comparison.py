import matplotlib.pyplot as plt
import numpy as np

def plot_stress_comparison(before_stress: dict, after_stress: dict):
    groups = sorted(before_stress.keys())
    before_values = [before_stress[g] for g in groups]
    after_values = [after_stress[g] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before_values, width, label='Before')
    rects2 = ax.bar(x + width/2, after_values, width, label='After')
    
    ax.set_ylabel('Average Stress Level')
    ax.set_title('Stress Level Comparison per Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("experiment_results/stress_comparison.png", bbox_inches='tight')
    plt.show()
