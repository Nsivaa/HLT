import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report
from lib.scores import membership_score, tune_sigmoid_threshold
import pandas as pd

# create confusion matrix of multilabel classification
def multilabel_confusion_matrix(_y_true, _y_pred, _label_true, _label_pred, normalize=False, transpose=False):
    y_true = _y_true if not transpose else _y_pred
    y_pred = _y_pred if not transpose else _y_true
    label_true = _label_true if not transpose else _label_pred
    label_pred = _label_pred if not transpose else _label_true
    n_true = len(label_true)
    n_pred = len(label_pred)
    if len(y_true) != len(y_pred):
        raise ValueError('Number of true and predicted labels must be the same')
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

def plot_multilabel_confusion_heatmap(y_true, y_pred, label_true, label_pred, normalize=False, transpose=False):
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, label_true, label_pred, normalize, transpose)
    x_label = label_pred if not transpose else label_true
    y_label = label_true if not transpose else label_pred
    fig, ax = plt.subplots(figsize=(15,10))
    # transform items to percentage
    if normalize:
        confusion_matrix = confusion_matrix * 100
    sns.heatmap(confusion_matrix, annot=True, ax=ax, xticklabels=x_label, yticklabels=y_label, cmap='coolwarm', fmt='.0f')
    ax.set_xlabel('Predicted' if not transpose else 'True')
    ax.set_ylabel('True' if not transpose else 'Predicted')
    plt.show()

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
    plt.show()

def plot_learning_curves(tr_loss, val_loss, score_name = 'loss'):
    plt.plot(tr_loss, label='train')
    plt.plot(val_loss, label='validation', color='orange', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel(score_name)
    plt.legend()
    plt.title(f'{score_name} over epochs')
    # set xticks to integers starting from 1
    plt.xticks(np.arange(0, len(tr_loss), 1), np.arange(1, len(tr_loss) + 1, 1))
    plt.show()
    
def custom_classification_report(scores_dict, labels_list):
    # print classification report
    print("Classification Report:")
    # print header
    print(f"{'Label':<20}{'Precision':>20}{'Recall':>20}{'F1-Score':>20}{'Jaccard':>20}{'Support':>20}")
    # print scores for each label
    for label in labels_list:
        # print scores with 2 decimal places
        print(f"{label:<20}{scores_dict[label]['precision']:20.2f}{scores_dict[label]['recall']:20.2f}{scores_dict[label]['f1-score']:20.2f}{scores_dict[label]['jaccard']:20.2f}{scores_dict[label]['support']:20.0f}")
    # print aggregated scores
    print(f"{'Macro avg':<20}{scores_dict['macro avg']['precision']:20.2f}{scores_dict['macro avg']['recall']:20.2f}{scores_dict['macro avg']['f1-score']:20.2f}{scores_dict['jaccard']['macro']:20.2f}{scores_dict['macro avg']['support']:20.0f}")
    print(f"{'Micro avg':<20}{scores_dict['micro avg']['precision']:20.2f}{scores_dict['micro avg']['recall']:20.2f}{scores_dict['micro avg']['f1-score']:20.2f}{scores_dict['jaccard']['micro']:20.2f}{scores_dict['micro avg']['support']:20.0f}")
    print(f"{'Weighted avg':<20}{scores_dict['weighted avg']['precision']:20.2f}{scores_dict['weighted avg']['recall']:20.2f}{scores_dict['weighted avg']['f1-score']:20.2f}{scores_dict['jaccard']['weighted']:20.2f}{scores_dict['weighted avg']['support']:20.0f}")
    # print membership score and jaccard samples
    print(f"{'Membership Score':<20}{scores_dict['membership']:20.2f}")
    print(f"{'Jaccard Samples':<20}{scores_dict['jaccard']['samples']:20.2f}")
    # print accuracy
    print(f"{'Accuracy':<20}{scores_dict['accuracy']:20.2f}")

def get_scores_dict(predictions, test_df, labels_list):
    # collect scores in a dictionary
    scores = classification_report(test_df[labels_list].values, predictions.values, target_names=labels_list, output_dict=True, zero_division=0)
    # add additional metrics
    # compute jaccard scores
    to_add = jaccard_score(test_df[labels_list].values, predictions.values, zero_division=0, average=None)
    for i, label in enumerate(labels_list):
        scores[label]['jaccard'] = to_add[i]
    # add aggregated jaccard scores
    scores['jaccard'] = {}
    scores['jaccard']['samples'] = jaccard_score(test_df[labels_list].values, predictions.values, zero_division=0, average='samples')
    scores['jaccard']['macro'] = jaccard_score(test_df[labels_list].values, predictions.values, zero_division=0, average='macro')
    scores['jaccard']['micro'] = jaccard_score(test_df[labels_list].values, predictions.values, zero_division=0, average='micro')
    scores['jaccard']['weighted'] = jaccard_score(test_df[labels_list].values, predictions.values, zero_division=0, average='weighted')
    # add membership score
    scores['membership'] = membership_score(test_df[labels_list].values, predictions.values)
    # add accuracy
    scores['accuracy'] = accuracy_score(test_df[labels_list].values, predictions.values)
    return scores

def model_analysis(model, val_df, target_cols):
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
    # print scores
    best_out = pd.DataFrame(best_out, columns=target_cols)
    scores_dict = get_scores_dict(best_out, val_df, target_cols)
    custom_classification_report(scores_dict, target_cols)

def twitter_model_analysis(model, val_df, target_cols):
    # plot learning curves
    tr_scores, val_scores = model.get_train_scores(), model.get_val_scores()
    tr_loss, val_loss = model.get_train_loss(), model.get_val_loss()
    plot_learning_curves(tr_loss, val_loss)
    plot_learning_curves(tr_scores['f1_macro'], val_scores['f1_macro'], 'Macro F1')
    # get predictions on validation set
    out = model.predict(val_df)
    target = val_df[target_cols].values
    # take argmax
    out = np.argmax(out, axis=1)
    plot_multilabel_confusion_heatmap(target, out, label_true=target_cols, label_pred=target_cols, normalize=True)
    # bar plot over classes
    plot_score_barplot(target, out, target_cols)
    # print scores
    out = pd.DataFrame(out, columns=target_cols)
    scores_dict = get_scores_dict(out, val_df, target_cols)
    custom_classification_report(scores_dict, target_cols)