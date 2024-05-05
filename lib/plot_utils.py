import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

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
        sum = confusion_matrix.sum(axis=1, keepdims=True)
        # avoid division by zero
        sum[sum == 0] = 1
        confusion_matrix = confusion_matrix / sum
        # keep only last 2 digits
        confusion_matrix = np.round(confusion_matrix, 2)
    return confusion_matrix

def plot_multilabel_confusion_heatmap(y_true, y_pred, label_true, label_pred, normalize=False):
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, label_true, label_pred, normalize)
    fig, ax = plt.subplots(figsize=(15,10))
    # transform items to percentage
    if normalize:
        confusion_matrix = confusion_matrix * 100
    sns.heatmap(confusion_matrix, annot=True, ax=ax, xticklabels=label_pred, yticklabels=label_true, cmap='coolwarm', fmt='.0f')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()

def tune_sigmoid_threshold(y_true, y_pred, metric_fun=accuracy_score, metric_params={}, is_maximization=True, return_only_best=True):
    thresholds = np.arange(0, 1, 0.01)
    scores = [metric_fun(y_true, y_pred > t, **metric_params) for t in thresholds]
    best_id = np.argmax(scores) if is_maximization else np.argmin(scores)
    best_threshold = thresholds[best_id]
    if return_only_best:
        scores = scores[best_id]
    return best_threshold, scores

def plot_threshold_tuning(y_true, y_pred, metric_fun=accuracy_score, metric_params={}, plot=False, is_maximization=True, metric_name='Accuracy'):
    best_threshold, scores = tune_sigmoid_threshold(y_true, y_pred, metric_fun, metric_params, is_maximization, return_only_best=False)
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

def plot_score_barplot(y_true, y_pred, class_names, metric_fun=f1_score, metric_params={'average': None, 'zero_division':0}, metric_name='F1 score'):
    class_scores = metric_fun(y_true, y_pred, **metric_params)
    plt.figure(figsize=(10,5))
    # rotate x labels
    plt.xticks(rotation=90)
    plt.bar(class_names, class_scores, color=sns.color_palette("viridis", len(class_names)))
    plt.xlabel('Class')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for each class')