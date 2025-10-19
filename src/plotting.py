import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')

from config import MODEL_THRESHOLD


def plot_dist_with_scores(scores, output_path_score):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(scores, bins=50, kde=True)
    plt.title('Distribution of Predicted Fraud Scores')
    plt.xlabel('Fraud Score')
    plt.ylabel('Frequency')
    
    plt.savefig(output_path_score)
    plt.close()

def plot_dist_with_labels(scores, output_path_labels):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(np.where(scores >= MODEL_THRESHOLD, 1, 0))
    plt.title('Distribution of Predicted Fraud Labels')
    plt.xlabel('Fraud Label')
    plt.ylabel('Frequency')
    
    plt.savefig(output_path_labels)
    plt.close()

def plot_score_distribution(scores, output_path_score, output_path_labels):
    plot_dist_with_scores(scores, output_path_score)
    plot_dist_with_labels(scores, output_path_labels)
    