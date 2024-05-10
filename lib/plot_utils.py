import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report
from lib.scores import membership_score

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

def plot_learning_curves(tr_loss, val_loss, score_name = 'loss'):
    plt.plot(tr_loss, label='train')
    plt.plot(val_loss, label='validation', color='orange', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel(score_name)
    plt.legend()
    plt.title(f'{score_name} over epochs')
    plt.show()

def model_analysis(model, val_df, target_cols, test_df=None):
    # plot learning curves
    tr_scores, val_scores = model.get_train_scores(), model.get_val_scores()
    tr_loss, val_loss = model.get_train_loss(), model.get_val_loss()
    plot_learning_curves(tr_loss, val_loss)
    plot_learning_curves(tr_scores['f1_macro'], val_scores['f1_macro'], 'Macro F1')
    # get predictions on validation set
    out = model.predict(val_df)
    target = val_df[target_cols].values
    # plot threshold tuning
    plot_threshold_tuning(target, out, plot=True)
    plot_threshold_tuning(target, out, plot=True, metric_params={'average':'micro', 'zero_division':0}, metric_fun=f1_score, metric_name='F1 Score')
    plot_threshold_tuning(target, out, plot=True, metric_params={'average':'macro', 'zero_division':0}, metric_fun=f1_score, metric_name='F1 Score')
    # get best threshold
    thresh, _ = tune_sigmoid_threshold(target, out, metric_params={'average':'macro', 'zero_division':0}, metric_fun=f1_score)
    # plot the confusion matrix for the best threshold
    best_out = (out > thresh).astype(int)
    plot_multilabel_confusion_heatmap(target, best_out, label_true=target_cols, label_pred=target_cols, normalize=True)
    # bar plot over classes
    plot_score_barplot(target, best_out, target_cols)
    # print classification report
    print(classification_report(target, best_out, target_names=target_cols))
    # print additional metrics
    print('Jaccard Samples Score:', jaccard_score(target, best_out, zero_division=0, average='samples'))
    print('Jaccard Macro Score:', jaccard_score(target, best_out, zero_division=0, average='macro'))
    print('Membership Score:', membership_score(target, out))
    if test_df is not None:
        # print results on test set using threshold from validation set
        # get predictions on test set
        out = model.predict(test_df)
        target = test_df[target_cols].values
        # plot the confusion matrix for the best threshold
        best_out = (out > thresh).astype(int)
        plot_multilabel_confusion_heatmap(target, best_out, label_true=target_cols, label_pred=target_cols, normalize=True)
        # bar plot over classes
        plot_score_barplot(target, best_out, target_cols)
        # print classification report
        print(classification_report(target, best_out, target_names=target_cols))
        # print additional metrics
        print('Jaccard Samples Score:', jaccard_score(target, best_out, zero_division=0, average='samples'))
        print('Jaccard Macro Score:', jaccard_score(target, best_out, zero_division=0, average='macro'))
        print('Membership Score:', membership_score(target, out))