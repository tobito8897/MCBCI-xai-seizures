#!/usr/bin/python3.7
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from . import settings


__all__ = ["plot_validation_curve", "plot_confusion_matrix"]


def plot_validation_curve(training_acc: list, validation_acc: list,
                          filename: str, title: str):
    plt.title("Validation Curve of ML model")
    lw = 2
    plt.figure(0)
    plt.plot(list(range(len(training_acc))), training_acc,
             label="Training", color="darkorange", lw=lw,
             marker="o")
    plt.plot(list(range(len(validation_acc))), validation_acc,
             label="Validation", color="navy", lw=lw,
             marker="o")
    plt.ylabel(title)
    plt.xlabel("Epochs")
    plt.legend(loc="best")
    plt.show()

    plt.savefig(filename + ".png", dpi=300, transparent=False, facecolor="w")
    plt.close()


def plot_confusion_matrix(true_label: list, predicted_label: list,
                          filename: str):
    cmatrix = confusion_matrix(true_label, predicted_label,
                               labels=[0, 1])
    f1s = f1_score(true_label, predicted_label)
    sen = float(cmatrix[1][1]/np.sum(cmatrix[1]))
    spec = float(cmatrix[0][0]/np.sum(cmatrix[0]))
    f1s = round(f1s*100)/100
    sen = round(sen*100)/100
    spec = round(spec*100)/100

    df_cm = pd.DataFrame(cmatrix)
    sn.set(font_scale=1.4)
    plt.figure(1)
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    ax.set(xticks=np.arange(len(settings["classes"])) + 0.5,
           yticks=np.arange(len(settings["classes"])) + 0.5,
           xticklabels=list(settings["classes"].values()),
           yticklabels=list(settings["classes"].values()))
    ax.set_title("F1={}, Sen={}, Spec={}".format(f1s, sen, spec))
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.show()
    plt.savefig(filename + ".png", dpi=300, transparent=False, facecolor="w")
