import matplotlib.pyplot as plt
import numpy as np

def plot_stress_comparison(before: dict, after: dict, save_path: str = None):
    """
    Plots a double bar chart comparing stress levels per group before and after solution implementation.
    'before' and 'after' should be dicts mapping group labels to average stress values.
    """
    groups = list(before.keys())
    before_values = [before[g] for g in groups]
    after_values = [after.get(g, 0) for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before_values, width, label='Before')
    rects2 = ax.bar(x + width/2, after_values, width, label='After')

    ax.set_ylabel('Average Stress Level')
    ax.set_xlabel('Group')
    ax.set_title('Stress Level Comparison per Group\nBefore and After Solution Implementation')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
