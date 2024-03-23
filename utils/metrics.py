import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score


def performance_metrics(true_label: list, predicted_label: list):
    true_label = true_label.argmax(axis=1)
    predicted_label = predicted_label.argmax(axis=1)
    cmatrix = confusion_matrix(true_label, predicted_label,
                               labels=[0, 1])
    cmatrix = cmatrix.astype("float") / cmatrix.sum(axis=1)[:, np.newaxis]
    f1s = f1_score(true_label, predicted_label)
    sen = float(cmatrix[1][1]/np.sum(cmatrix[1]))
    spec = float(cmatrix[0][0]/np.sum(cmatrix[0]))
    acc = accuracy_score(true_label, predicted_label)
    return f1s, sen, spec, acc

def roc_score(true_label: list, predicted_label: list):
    return roc_auc_score(true_label, predicted_label)
    