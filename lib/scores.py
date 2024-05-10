from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from lib.plot_utils import tune_sigmoid_threshold
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
weaker accuracy, each prediction is considered correct if its maximum probability class is one of the true classes
'''
def membership_score(y_true, y_pred):
    n_correct = 0
    for t_pattern, p_pattern in zip(y_true, y_pred):
        n_correct += t_pattern[np.argmax(p_pattern)] == 1
    return n_correct / len(y_true)