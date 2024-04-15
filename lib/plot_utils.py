import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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