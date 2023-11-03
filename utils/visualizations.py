#!/usr/bin/python3.7
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_eeg_windows(eeg_array: np.array, channels_list: list, fs: int,
                     save: str):

    grid_specs = GridSpec(1, 1, wspace=0.08, hspace=0.25)
    fig = plt.figure(figsize=(5.5, 4))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    time = np.arange(eeg_array.shape[1])/fs
    space = np.max(np.max(eeg_array))/1.5
    channels_list = ["-".join(x) for x in channels_list]

    for count, channel in enumerate(channels_list):
        ax.plot(time, eeg_array[count, :] - space*count, label=channel,
                linewidth=1)
    ax.set(yticks=np.arange(-space*count, 1, space),
           yticklabels=reversed(channels_list))
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Channels", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_database_comparison(dataframe: pd.DataFrame,
                             x_column: str,
                             y_column: str,
                             hue_column: str, save: str):
    grid_specs = GridSpec(2, 1, wspace=0.08, hspace=0.15)
    fig = plt.figure(figsize=(4.1, 6))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    sns.boxplot(ax=ax, data=dataframe, x=x_column,
                y=y_column, hue=hue_column)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)
    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim([0, 1])
    ax.legend(prop=dict(size=8), loc="lower right")

    if save:
        plt.savefig(save, dpi=400, bbox_inches="tight", pad_inches=0.1,
                    transparent=False, facecolor='white')
    plt.show()


def plot_seizure_comparison(dataframe_1: pd.DataFrame,
                            dataframe_2: pd.DataFrame,
                            x_column: str,
                            y_column: str,
                            save: str):
    grid_specs = GridSpec(1, 2, wspace=0.08, hspace=0.15)
    fig = plt.figure(figsize=(5.1, 3.5))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    sns.stripplot(ax=ax, data=dataframe_1, x=x_column,
                  y=y_column)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)
    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim([0, 1])
    ax.legend([], [], frameon=False)
    ax.text(1.0, -0.3, "(a)", fontsize=7)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    sns.stripplot(ax=ax, data=dataframe_2, x=x_column,
                  y=y_column)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)
    ax.xaxis.set_tick_params(rotation=45, labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim([0, 1])
    ax.set_yticks([])
    ax.set_ylabel("", fontsize=7)
    ax.legend([], [], frameon=False)
    ax.text(2.0, -0.3, "(b)", fontsize=7)

    if save:
        plt.savefig(save, dpi=400, bbox_inches="tight", pad_inches=0.1,
                    transparent=False, facecolor='white')
    plt.show()


def plot_explanation_paper(eeg_array: np.array, shap_array: np.array,
                           channels_list: list, save: str, xai_method: str):

    fs = int(eeg_array.shape[1]/2)
    channels_list = ["-".join(x) for x in channels_list]

    grid_specs = GridSpec(1, 2, wspace=0.08, hspace=0.25)
    fig = plt.figure(figsize=(5.5, 4))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    time = np.arange(eeg_array.shape[1])/fs

    for count, channel in enumerate(channels_list):
        norm_eeg_array = ((eeg_array[count, :] + np.min(eeg_array[count, :])) /
                          (np.max(eeg_array[count, :]) - np.min(eeg_array[count, :])))
        ax.plot(time, norm_eeg_array + count*0.8, label=channel, linewidth=1)
    ax.set(yticks=np.arange(0, (count+1)*0.8, 0.8),
           yticklabels=channels_list)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Channels", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.text(0.9, -5.4, "(a)", fontsize=8)

    ########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    max_time = shap_array.shape[1]/fs

    max_color = np.max(np.abs(shap_array))
    shap_array = np.flip(shap_array, axis=0)
    im = ax.imshow(shap_array, interpolation="nearest", cmap=plt.cm.bwr,
                   extent=[0, shap_array.shape[1], 0, shap_array.shape[0]],
                   aspect="auto", vmin=-max_color, vmax=max_color)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        aspect=10, pad=0.05)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(xai_method, fontsize=8)

    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set(xticks=np.linspace(0, shap_array.shape[1], 6),
           xticklabels=np.round(np.linspace(0, max_time, 6), 2))
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.text(245, -3.9, "(b)", fontsize=8)

    if save:
        plt.savefig(save, dpi=400, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_distribution_cofficients(dataframe_shap: pd.DataFrame,
                                  dataframe_lime: pd.DataFrame,
                                  x_column: str, save: str):
    grid_specs = GridSpec(1, 2, wspace=0.08, hspace=0.15)
    fig = plt.figure(figsize=((5.5, 4)))
    
    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    sns.histplot(dataframe_shap, x=x_column)
    ax.plot([-0.3]*201, list(range(-1, 200)), color='b', linestyle="--")
    ax.plot([0.3]*201, list(range(-1, 200)), color='b', linestyle="--")
    ax.plot([-0.5]*201, list(range(-1, 200)), color='r', linestyle="--")
    ax.plot([0.5]*201, list(range(-1, 200)), color='r', linestyle="--")
    ax.yaxis.label.set_size(7)
    ax.xaxis.label.set_size(7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=90)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([0, 200])
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_xlabel("Spearman's correlation", fontsize=7)
    ax.text(0, -40, "(a)", fontsize=7)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    sns.histplot(dataframe_lime, x=x_column)
    ax.plot([-0.3]*201, list(range(-1, 200)), color='b', linestyle="--")
    ax.plot([0.3]*201, list(range(-1, 200)), color='b', linestyle="--")
    ax.plot([-0.5]*201, list(range(-1, 200)), color='r', linestyle="--")
    ax.plot([0.5]*201, list(range(-1, 200)), color='r', linestyle="--")
    ax.yaxis.label.set_size(7)
    ax.xaxis.label.set_size(7)
    ax.xaxis.set_tick_params(labelsize=7, rotation=90)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([0, 200])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylabel("", fontsize=7)
    ax.set_xlabel("Spearman's correlation", fontsize=7)
    ax.text(0, -40, "(b)", fontsize=7)

    if save:
        plt.savefig(save, dpi=400, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()
    plt.close()
