# Figures for new benchmark

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import matplotlib.cm as cm

from signet import DATA
from signet.utilities.io import read_methods_guesses
from signet.utilities.plotting import plot_percentage_all_methods

def plot_by_sig(list_of_guesses, list_of_methods, label, sigsnames, show=True):
    fig_width_cm = 21                                # A4 page
    fig_height_cm = 29.7
    inches_per_cm = 1 / 2.54                         # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm         # width in inches
    fig_height = fig_height_cm * inches_per_cm       # height in inches
    fig_size = [fig_width, fig_height]

    fig, axs = plt.subplots(nrows=len(list_of_methods), ncols=1, figsize=fig_size)
    num_muts = np.unique(label[:,-1].detach().numpy())
    colors = cm.rainbow(np.linspace(1, 0, num_muts.shape[0]))
    num_muts = num_muts[:-1]
    num_sigs = 72
    num_sigs_list = list(range(num_sigs))
    for method_index in range(len(list_of_methods)):
        values = np.zeros((num_sigs, len(num_muts)))
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            for j in range(num_sigs):
                guesses_j = list_of_guesses[method_index][indexes, j]
                labels_j = label[indexes, j]
                metric = torch.sum(guesses_j[labels_j>0]>0.01)/torch.sum(labels_j>0)
                values[j, i] = metric
        sigs_list = np.array(num_sigs_list)[values.sum(axis=1)>0]
        sigsnames_list = np.array(sigsnames)[sigs_list]
        values = values[values.sum(axis=1)>0,:]
        for k in range(len(values[0])):
            axs[method_index].scatter(range(len(sigs_list)), values[:,k], color=colors[k], s=3.0*(k+1), alpha=0.3)
        axs[method_index].set_xticks(range(len(sigs_list)))
        axs[method_index].set_xticklabels(['']*len(sigs_list), rotation=90)
        # axs[method_index].set_ylabel(list_of_methods[method_index])
        stylize_axes(axs[method_index], '', "", "%s \n TP/(real P)"%list_of_methods[method_index])
    axs[-1].set_xticklabels(sigsnames_list, rotation=90)
    # fig.subplots_adjust()
    plt.tight_layout()
    fig.legend(loc=7, labels=num_muts, prop={'size': 8})
    if show:
        plt.show()
    else:
        fig.savefig('fig_by_sig.pdf')

def plot_positives(list_of_guesses, list_of_methods, label, show=True):
    cutoff = 0.01
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    num_muts = np.unique(label[:,-1].detach().numpy())
    num_muts = num_muts[:-1]
    values = np.zeros((len(list_of_methods), len(num_muts)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            guesses_j = list_of_guesses[method_index][indexes, :]
            labels_j = label[indexes, :-1]
            guess_mask = (guesses_j>cutoff).type(torch.int).float()
            labels_mask = (labels_j>cutoff).type(torch.int).float()
            p = torch.sum(guess_mask)
            tp = torch.sum(torch.einsum("bi,bi->b", guess_mask, labels_mask))
            values[method_index, i] = p/tp
    marker_size = 3
    line_width = 0.5
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#fad6a5', '#36454f', '#bcbd22', '#17becf']
    p=axs.plot(np.log10(num_muts), np.transpose(values), marker='o',linewidth=line_width, markersize=marker_size)
    for idx, color in enumerate(colors):
        p[idx].set_color(color)
    stylize_axes(axs, '', "log(N)", "P/TP")
    fig.tight_layout()
    legend_adjustment = 0.75
    fig.subplots_adjust(right=legend_adjustment)   
    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    if show:
        plt.show()
    else:
        fig.savefig('positives.pdf')

def plot_F1(list_of_guesses, list_of_methods, label, show=True):
    cutoff = 0.01
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    num_muts = np.unique(label[:,-1].detach().numpy())
    num_muts = num_muts[:-1]
    values = np.zeros((len(list_of_methods), len(num_muts)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            guesses_j = list_of_guesses[method_index][indexes, :]
            labels_j = label[indexes, :-1]
            guess_mask = (guesses_j>cutoff).type(torch.int).float()
            labels_mask = (labels_j>cutoff).type(torch.int).float()
            fp = torch.sum(labels_mask - guess_mask < -0.1)
            fn = torch.sum(labels_mask - guess_mask > 0.1)
            tp = torch.sum(torch.einsum("bi,bi->b", guess_mask, labels_mask))
            tn = torch.sum(torch.einsum("bi,bi->b", 1 - guess_mask, 1 - labels_mask))
            sensitivity = tp/torch.sum(labels_mask)
            precision = tp / (tp + fp)
            F1 = 2*precision*sensitivity/(precision+sensitivity)
            values[method_index, i] = F1
    marker_size = 3
    line_width = 0.5
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#fad6a5', '#36454f', '#bcbd22', '#17becf']
    p = axs.plot(np.log10(num_muts), np.transpose(values), marker='o',linewidth=line_width, markersize=marker_size)
    for idx, color in enumerate(colors):
        p[idx].set_color(color)
    stylize_axes(axs, '', "log(N)", "F1")
    fig.tight_layout()
    legend_adjustment = 0.75
    fig.subplots_adjust(right=legend_adjustment)   
    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    if show:
        plt.show()
    else:
        fig.savefig('F1.pdf')

def plot_MCC(list_of_guesses, list_of_methods, label, show=True):
    cutoff = 0.01
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    num_muts = np.unique(label[:,-1].detach().numpy())
    num_muts = num_muts[:-1]
    values = np.zeros((len(list_of_methods), len(num_muts)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            guesses_j = list_of_guesses[method_index][indexes, :]
            labels_j = label[indexes, :-1]
            guess_mask = (guesses_j>cutoff).type(torch.int).float()
            labels_mask = (labels_j>cutoff).type(torch.int).float()
            fp = torch.sum(labels_mask - guess_mask < -0.1)
            fn = torch.sum(labels_mask - guess_mask > 0.1)
            tp = torch.sum(torch.einsum("bi,bi->b", guess_mask, labels_mask))
            tn = torch.sum(torch.einsum("bi,bi->b", 1 - guess_mask, 1 - labels_mask))
            MCC = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            values[method_index, i] = MCC
    marker_size = 3
    line_width = 0.5
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#fad6a5', '#36454f', '#bcbd22', '#17becf']
    p = axs.plot(np.log10(num_muts), np.transpose(values), marker='o',linewidth=line_width, markersize=marker_size)
    for idx, color in enumerate(colors):
        p[idx].set_color(color)
    stylize_axes(axs, '', "log(N)", "MCC")
    fig.tight_layout()
    legend_adjustment = 0.75
    fig.subplots_adjust(right=legend_adjustment)   
    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    if show:
        plt.show()
    else:
        fig.savefig('MCC.pdf')

def stylize_axes(ax, title, xlabel, ylabel):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA", "deconstructSigs", "mutationalCone", "QPsig", "sigLASSO", "MuSiCal", "NNLS", "SigNet"]
list_of_guesses, label = read_methods_guesses('cpu', "exp_all", list_of_methods, data_folder="../../data/")

sigsnames = list(pd.read_excel(DATA + "/data.xlsx").columns)[1:]

# plot_by_sig(list_of_guesses, list_of_methods, label, sigsnames, show=False)
# plot_positives(list_of_guesses, list_of_methods, label, show=False)
# plot_F1(list_of_guesses, list_of_methods, label, show=False)
plot_MCC(list_of_guesses, list_of_methods, label, show=False)

plot_percentage_all_methods(label, list_of_guesses, list_of_methods, sigsnames, plot_path='percentage_present_guessed_corrected.pdf', show=False, title=None)