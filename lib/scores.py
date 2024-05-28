from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import numpy as np

# tuning implicitly done in score calculation :)
def accuracy(y_true, y_pred):
    _, best_res = tune_sigmoid_threshold(y_true, y_pred, accuracy_score)
    return best_res

def jaccard(y_true, y_pred):
    _, best_res = tune_sigmoid_threshold(y_true, y_pred, jaccard_score, {'average': 'macro', 'zero_division': 0})
    return best_res

def jaccard_samples(y_true, y_pred):
    _, best_res = tune_sigmoid_threshold(y_true, y_pred, jaccard_score, {'average': 'samples', 'zero_division': 0})
    return best_res

def f1(y_true, y_pred):
    _, best_res = tune_sigmoid_threshold(y_true, y_pred, f1_score, {'average': 'macro', 'zero_division': 0})
    return best_res

def f1_micro(y_true, y_pred):
    _, best_res = tune_sigmoid_threshold(y_true, y_pred, f1_score, {'average': 'micro', 'zero_division': 0})
    return best_res
'''
weaker accuracy, each prediction is considered correct if its maximum probability class is one of the true ones
'''
def membership_score(y_true, y_pred):
    n_correct = 0
    for t_pattern, p_pattern in zip(y_true, y_pred):
        n_correct += t_pattern[np.argmax(p_pattern)] == 1
    return n_correct / len(y_true)

def tune_sigmoid_threshold(y_true, y_pred, metric_fun=accuracy_score, metric_params={}, is_maximization=True, return_only_best=True):
    thresholds = np.arange(0, 1, 0.01)
    scores = [metric_fun(y_true, y_pred > t, **metric_params) for t in thresholds]
    best_id = np.argmax(scores) if is_maximization else np.argmin(scores)
    best_threshold = thresholds[best_id]
    if return_only_best:
        scores = scores[best_id]
    return best_threshold, scores

def sl_f1_macro(y_true, y_pred):
    # flatten one hot encoded predictions and true labels
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='macro')

def sl_f1_micro(y_true, y_pred):
    # flatten one hot encoded predictions and true labels
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='micro')

def sl_accuracy(y_true, y_pred):
    # flatten one hot encoded predictions and true labels
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)