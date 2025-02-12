import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metric_distribution(data: pd.DataFrame, metric: str, title: str, xlabel: str, ylabel: str, export_path: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[metric], kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_group_metric(data: pd.DataFrame, group_by: str, metric: str, title: str, xlabel: str, ylabel: str, export_path: str = None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_by, y=metric, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_all_measured_metrics(data: pd.DataFrame, export_dir: str = None):
    metrics = {
        "stress_level": ("Stress Level Distribution", "Stress Level", "Frequency"),
        "survival_points": ("Survival Points Distribution", "Survival Points", "Frequency"),
        "adaptation_score": ("Adaptation Score Distribution", "Adaptation Score", "Frequency"),
        "social_connections": ("Social Connections Distribution", "Number of Connections", "Frequency")
    }
    for metric, (title, xlabel, ylabel) in metrics.items():
        if metric in data.columns:
            export_path = f"{export_dir}/{metric}_distribution.png" if export_dir else None
            plot_metric_distribution(data, metric, title, xlabel, ylabel, export_path)

def plot_metric_by_group(data: pd.DataFrame, group_col: str = "group_id", export_dir: str = None):
    metrics = ["stress_level", "survival_points", "adaptation_score", "social_connections"]
    for metric in metrics:
        if metric in data.columns:
            title = f"{metric.replace('_', ' ').title()} by {group_col.title()}"
            export_path = f"{export_dir}/{metric}_by_{group_col}.png" if export_dir else None
            plot_group_metric(data, group_col, metric, title, group_col.title(), metric.replace('_', ' ').title(), export_path)

def plot_all_measured_metrics(df: pd.DataFrame) -> None:
    # Improved overall metrics graph
    plt.figure(figsize=(12, 8))
    metrics = [col for col in df.columns if col not in ['Group', 'Method']]
    for metric in metrics:
        sns.kdeplot(df[metric], label=metric, fill=True, alpha=0.5)
    plt.title("Distribution of Measured Metrics")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

def plot_metric_by_group(df: pd.DataFrame) -> None:
    # Boxplot comparing metrics across groups
    metrics = [col for col in df.columns if col not in ['Group', 'Method']]
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.boxplot(x='Group', y=metric, data=df, ax=ax)
        ax.set_title(f"{metric} by Group")
    plt.tight_layout()

def plot_improvement_comparison(baseline_df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    # Combine dataframes with an extra column indicating method
    baseline_df = baseline_df.copy()
    solution_df = solution_df.copy()
    baseline_df["Method"] = "Baseline"
    solution_df["Method"] = "Solution"
    combined = pd.concat([baseline_df, solution_df])
    
    # Plot improvements per metric
    metrics = [col for col in combined.columns if col not in ['Group', 'Method']]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.barplot(x="Method", y=metric, data=combined, ax=ax, ci="sd")
        ax.set_title(f"Comparison of {metric}")
        ax.set_ylabel(f"{metric} Value")
    plt.tight_layout()

def plot_confusion_matrix(matrix: np.ndarray, title: str, export_path: str = None) -> None:
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')