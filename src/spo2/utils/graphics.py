from utils.consts import save_directory
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Function which creates a figure according to the number of subplots
def create_figure(**kwargs):
    fig = plt.figure(figsize=kwargs.get('figsize', (15, 8)))
    subplots = kwargs.get('subplots', (1, 1))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    axes = np.empty(subplots, dtype=object)
    ax_num = 1
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if ax_num == 1:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            elif sharex and sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0], sharey=axes[0][0])
            elif sharex:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0])
            elif sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharey=axes[0][0])
            else:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            ax_num += 1

    if kwargs.get('tight_layout', False):
        fig.tight_layout()

    return fig, axes


# Function which completes a figure with the different titles.
# Should be called after the creation of the figure and plotting the data
def complete_figure(fig, axes, **kwargs):
    SNAPSHOTS_DIR = save_directory
    xticks_fontsize = kwargs.get('xticks_fontsize', 12)
    yticks_fontsize = kwargs.get('yticks_fontsize', 12)
    xlabel_fontsize = kwargs.get('xlabel_fontsize', 20)
    ylabel_fontsize = kwargs.get('ylabel_fontsize', 20)
    lim_80_100 = kwargs.get('lim_80_100', False)
    lim_y_1 = kwargs.get('lim_y_1', False)
    lim_0_1 = kwargs.get('lim_0_1', False)
    lim_07_1 = kwargs.get('lim_07_1', False)
    rotation = kwargs.get('rotation', False)
    sharex23 = kwargs.get('sharex23', False)
    frameon = kwargs.get('frameon', False)
    subplot_xlabel = kwargs.get('subplot_xlabel', False)
    x_titles = kwargs.get('x_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    y_titles = kwargs.get('y_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    put_legend = kwargs.get('put_legend', True * np.ones(axes.shape, dtype=bool))
    loc_legend = kwargs.get('loc_legend', 'best' * np.ones(axes.shape, dtype=object))
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xlabel(x_titles[i][j], fontsize=xlabel_fontsize)
            axes[i][j].set_ylabel(y_titles[i][j], fontsize=ylabel_fontsize)
            axes[i][j].tick_params(axis='x', labelsize=xticks_fontsize)
            axes[i][j].tick_params(axis='y', labelsize=yticks_fontsize)
            if put_legend[i][j]:
                axes[i][j].legend(fontsize=kwargs.get('legend_fontsize', 28), loc=loc_legend[i][j], frameon=frameon)
            # axes[i][j].set_xlim(x_lim[i])
            if lim_80_100:
                axes[i][j].set_ylim([84, 100])
                axes[i][j].set_yticks([84, 88, 92, 96, 100])
                axes[i][j].set_xlim([0, 500])
            if lim_y_1:
                axes[i][j].set_xlim([0, 0.05])
                # axes[i][j].set_ylim([0, 0.5])
            if lim_0_1:
                axes[i][j].set_ylim([0.4, 1])
                axes[i][j].set_yticks([0.4, 0.6, 0.8, 1.0])
            if lim_07_1:
                axes[i][j].set_ylim([0.7, 1])
            if rotation:
                axes[i][j].xticks(rotation=45)
            if sharex23 and (i != 0):
                axes[i][j].set_ylim([80, 100])
                axes[i][j].set_yticks([80, 90, 100])
                if i == 2:
                    axes[i][j].set_xlim([0, 4000])
                    axes[i][j].set_xticks([0, 1000, 2000, 3000, 4000])
            # axes[i][j].set_xticks(x_ticks[i][j])
            # axes[i][j].set_yticks(y_ticks[i][j])
            # axes[i][j].set_xticklabels(x_ticks_labels[i][j])
            # axes[i][j].set_yticklabels(y_ticks_labels[i][j])
    if subplot_xlabel:
        for ax in axes.flat:
            ax.label_outer()
    plt.suptitle(kwargs.get('suptitle', ''), fontsize=kwargs.get('suptitle_fontsize', 28))
    if kwargs.get('savefig', False):
        plt.savefig(SNAPSHOTS_DIR + (kwargs.get('main_title', 'NoName') + '.png'))
    plt.close()


def model_metrics(data, label, predicted, beta=1):
    AUROC = roc_auc_score(label, data)
    accuracy = accuracy_score(label, predicted)
    TN, FP, FN, TP = confusion_matrix(label, predicted).ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    sensitivity = recall
    specificity = TN / (TN + FP)
    PPV = precision
    NPV = TN / (TN + FN)
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    print("Accuracy: " + str(accuracy))
    print("F" + str(beta) + "-Score: " + str(fbeta))
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))
    print("PPV: " + str(PPV))
    print("NPV: " + str(NPV))
    print("AUROC: " + str(AUROC))
    print(confusion_matrix(label, predicted))
    return 2 * precision * recall / (precision + recall), AUROC, sensitivity, specificity, PPV
