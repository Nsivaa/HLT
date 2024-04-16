import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# create confusion matrix of multilabel classification
def multilabel_confusion_matrix(y_true, y_pred, label_true, label_pred, normalize=False):
    n_true = len(label_true)
    n_pred = len(label_pred)
    confusion_matrix = np.zeros((n_true, n_pred))
    for true_el, pred_el in zip(y_true, y_pred):
        for i in range(n_true):
            if true_el[i] == 1:
                confusion_matrix[i,:] += pred_el
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    return confusion_matrix

def plot_multilabel_confusion_heatmap(y_true, y_pred, label_true, label_pred, normalize=False):
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, label_true, label_pred, normalize)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confusion_matrix, annot=True, ax=ax, xticklabels=label_pred, yticklabels=label_true, cmap='coolwarm')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return ax

def tune_sigmoid_threshold(y_true, y_pred, metric_fun=accuracy_score, metric_params={}, is_maximization=True):
    thresholds = np.arange(0, 1, 0.01)
    scores = [metric_fun(y_true, y_pred > t, **metric_params) for t in thresholds]
    best_threshold = thresholds[np.argmax(scores)] if is_maximization else thresholds[np.argmin(scores)]
    return best_threshold, scores

def plot_threshold_tuning(y_true, y_pred, metric_fun=accuracy_score, metric_params={}, plot=False, is_maximization=True, metric_name='Accuracy'):
    best_threshold, scores = tune_sigmoid_threshold(y_true, y_pred, metric_fun, metric_params, is_maximization)
    if plot:
        plt.plot(np.arange(0, 1, 0.01), scores)
        plt.xlabel('Threshold')
        plt.ylabel(metric_name)
        # get average type if provided
        if 'average' in metric_params:
            metric_name += f' ({metric_params["average"]})'
        plt.title(f'{metric_name} over sigmoid threshold')
        plt.show()
        print(f'Best threshold: {best_threshold}')
        print(f'Best {metric_name}: {max(scores) if is_maximization else min(scores)}')