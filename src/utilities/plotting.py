import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch

from utilities.metrics import accuracy, false_random, false_realistic, get_classification_metrics, get_pi_metrics, get_reconstruction_error
from utilities.io import create_dir

# from utilities.metrics import get_classification_metrics, get_pi_metrics



def stylize_axes(ax, title, xlabel, ylabel):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

# SIGNATURE PLOTS:
def plot_signature(signature, labels):
    plt.bar(range(96), signature, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()

# DATA GENERATION PLOTS:
def plot_prop_signatures(weights_0, weights_augmented):
    prop_0 = torch.sum(weights_0>0, dim=0)/weights_0.shape[0]*100
    prop_augmented = torch.sum(weights_augmented>0, dim=0)/weights_augmented.shape[0]*100

    num_classes = weights_0.shape[1]
    data = pd.read_excel("../../data/data.xlsx")
    sigs_names = list(data.columns)[2:]
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),prop_0, align='center', width=0.4, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.4, prop_augmented, width=0.4, align='center')
    ax.set_ylabel('Proportion present (%)')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.legend(['Original weights', 'Augmented weights'])
    plt.tight_layout()
    plt.show()

# CLASSIFIER PLOTS:
def plot_metric_vs_mutations_classifier(guess, label, num_muts_list, plot_path = None):
    fig, axs = plt.subplots(3, figsize=(8,4))
    fig.suptitle("Detector Performance")
    
    num_muts = np.unique(num_muts_list.detach().numpy())
    
    marker_size = 3
    line_width = 0.5
    values = np.zeros((3, len(num_muts)))
    for i, num_mut in enumerate(num_muts):
        indexes = num_muts_list == num_mut
        values[0,i] = accuracy(label=label[indexes], prediction=guess[indexes])
        values[1,i] = false_realistic(label=label[indexes], prediction=guess[indexes])
        values[2,i] = false_random(label=label[indexes], prediction=guess[indexes])
        
    axs[0].plot(np.log10(num_muts), values[0,:], marker='o',linewidth=line_width, markersize=marker_size)
    axs[1].plot(np.log10(num_muts), values[1,:], marker='o',linewidth=line_width, markersize=marker_size)
    axs[2].plot(np.log10(num_muts), values[2,:], marker='o',linewidth=line_width, markersize=marker_size)

    y_labels = ["Accuracy (%)", "False Realistic (%)", "False Random (%)"]
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', 'log(N)', y_labels[i])
    # stylize_axes(axs, '', 'log(N)', y_labels)

    fig.tight_layout()
    plt.show()
    # fig.savefig(plot_path+'detector.pdf')
    
# FINETUNER PLOTS:
def plot_crossval(values, num_muts):
    marker_size = 3
    line_width = 0.5

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axs[0, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 0]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[0, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 1]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[1, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 2]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[1, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 3]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axs[0, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 4]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[0, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 5]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[1, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 6]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    axs[1, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 7]),
                    marker='o', linewidth=line_width, markersize=marker_size)
    plt.show()
    plt.close()

def plot_crossval_benchmark(list_of_methods, list_of_guesses, label, values_finetuner, folder_path=None, show=False):
    '''
    Plot:
    MAE_p   MAE_n
    FPR      FNR
    and in another plot
    Accuracy    Precision
    Sensitivity Specificity
    '''
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

    num_muts = np.unique(label[:,-1].detach().numpy())
    list_of_metrics = ["MAE", "KL", "fpr", "fnr", "accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]

    values = np.zeros((len(list_of_methods)+1, len(num_muts), len(list_of_metrics)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            metrics = get_classification_metrics(label_batch=label[indexes, :-1],
                                                 prediction_batch=list_of_guesses[method_index][indexes, :])
            for metric_index, metric in enumerate(list_of_metrics):
                values[method_index, i, metric_index] = metrics[metric]
    values[-1,:,:] = np.mean(values_finetuner, axis=0)
    marker_size = 3
    line_width = 0.5
    legend_adjustment = 0.75
    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,0]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,1]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,2]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,3]), marker='o',linewidth=line_width, markersize=marker_size)

    xlabel = 'log(N)'
    ylabel = ["MAE", "KL", "FPR", "FNR"]
    # fig.suptitle("Metrics vs Number of Mutations")
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', xlabel, ylabel[i])
        # axes.ticklabel_format(axis="both", style="sci")


    fig.legend(loc=7, labels=list_of_methods+['SigNet Refitter'], prop={'size': 8})
    fig.tight_layout()
    fig.subplots_adjust(right=legend_adjustment)   
    # create_dir(folder_path)
    if show:
        plt.show()
    # plt.savefig(folder_path + '/metrics_low.svg')
    plt.close()
    ############################################################################################
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

    list_of_metrics = ["accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]

    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,4]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,5]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,6]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,7]), marker='o',linewidth=line_width, markersize=marker_size)

    xlabel = 'log(N)'
    ylabel = ["Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)"]
    # fig.suptitle("Metrics vs Number of Mutations")
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', xlabel, ylabel[i])
        # axes.ticklabel_format(axis="y", style="sci")
        

    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    fig.tight_layout()
    fig.subplots_adjust(right=legend_adjustment)   
    plt.show()
    # plt.savefig(folder_path + '/metrics_high.svg')
    plt.close()

def plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label, folder_path=None, show=False):
    '''
    Plot:
    MAE_p   MAE_n
    FPR      FNR
    and in another plot
    Accuracy    Precision
    Sensitivity Specificity
    '''
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

    num_muts = np.unique(label[:,-1].detach().numpy())
    list_of_metrics = ["MAE", "KL", "fpr", "fnr", "accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]

    values = np.zeros((len(list_of_methods), len(num_muts), len(list_of_metrics)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            metrics = get_classification_metrics(label_batch=label[indexes, :-1],
                                                 prediction_batch=list_of_guesses[method_index][indexes, :])
            for metric_index, metric in enumerate(list_of_metrics):
                values[method_index, i, metric_index] = metrics[metric]

    marker_size = 3
    line_width = 0.5
    legend_adjustment = 0.75
    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,0]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,1]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,2]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,3]), marker='o',linewidth=line_width, markersize=marker_size)

    xlabel = 'log(N)'
    ylabel = ["MAE", "KL", "FPR", "FNR"]
    # fig.suptitle("Metrics vs Number of Mutations")
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', xlabel, ylabel[i])
        # axes.ticklabel_format(axis="both", style="sci")


    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    fig.tight_layout()
    fig.subplots_adjust(right=legend_adjustment)   
    # create_dir(folder_path)
    if show:
        plt.show()
    # plt.savefig(folder_path + '/metrics_low.svg')
    plt.close()
    ############################################################################################
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

    list_of_metrics = ["accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]

    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,4]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,5]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,6]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,7]), marker='o',linewidth=line_width, markersize=marker_size)

    xlabel = 'log(N)'
    ylabel = ["Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)"]
    # fig.suptitle("Metrics vs Number of Mutations")
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', xlabel, ylabel[i])
        # axes.ticklabel_format(axis="y", style="sci")
        

    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    fig.tight_layout()
    fig.subplots_adjust(right=legend_adjustment)   
    plt.show()
    # plt.savefig(folder_path + '/metrics_high.svg')
    plt.close()

    ############################################################################################
    # mean_values = np.mean(values, axis=1)
    # # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,6))

    # width = 1/(len(list_of_methods)+3)
    # list_of_metrics = ["accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]
    # for method_index in range(len(list_of_methods)):
    #     axs[0].bar(np.array(range(len(list_of_metrics)))+width*method_index, mean_values[method_index,4:], align='center', width=width)
    # axs[0].set_xticks(np.array(range(len(list_of_metrics)))+width*len(list_of_methods)/2)
    # axs[0].set_xticklabels(["Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)"])
    # axs[0].hlines(100, axs[0].get_xlim()[0], axs[0].get_xlim()[1],linestyles = 'dashed', color = 'gray', label='_nolegend_')

    # list_of_metrics = ["MAE_p", "MAE_n"] 
    # for method_index in range(len(list_of_methods)):
    #     axs[1].bar(np.array(range(len(list_of_metrics)))+width*method_index, mean_values[method_index,:2], align='center', width=width)
    # axs[1].set_xticks(np.array(range(len(list_of_metrics)))+width*len(list_of_methods)/2)
    # axs[1].set_xticklabels(["MAE postives", "MAE negatives"])

    # list_of_metrics = ["fpr", "fnr"] 
    # for method_index in range(len(list_of_methods)):
    #     axs[2].bar(np.array(range(len(list_of_metrics)))+width*method_index, mean_values[method_index,2:4], align='center', width=width)
    # axs[2].set_xticks(np.array(range(len(list_of_metrics)))+width*len(list_of_methods)/2)
    # axs[2].set_xticklabels(["FPR", "FNR"])
    
    # for i, axes in enumerate(axs.flat):
    #     stylize_axes(axes, '', '', 'Metrics')

    # fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    # fig.tight_layout()
    # fig.subplots_adjust(right=legend_adjustment)
    # if show:
    #     plt.show()
    # # plt.savefig(folder_path + '/metrics_mean.png')
    # plt.close()


def final_plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label, folder_path=None, signatures=None, mutation_distributions=None):
    '''
    Plot:
    MAE                 KLdiv
    Accuracy            Precision
    Sensitivity         Specificity
    ReconstructionMSE   Legend
    '''
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,10))

    num_muts = np.unique(label[:,-1].detach().numpy())
    list_of_metrics = ["MAE", "KL", "accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %", "reconstruction_error"]

    values = np.zeros((len(list_of_methods), len(num_muts), len(list_of_metrics)))
    for method_index in range(len(list_of_methods)):
        for i, num_mut in enumerate(num_muts):
            indexes = label[:, -1] == num_mut
            metrics = get_classification_metrics(label_batch=label[indexes, :-1],
                                                 prediction_batch=list_of_guesses[method_index][indexes, :])
            for metric_index, metric in enumerate(list_of_metrics):
                if metric == "reconstruction_error":  # TODO(oleguer) Fix this, its super sketchy
                    assert(signatures is not None)
                    assert(mutation_distributions is not None)
                    rec_error = get_reconstruction_error(mutation_dist=mutation_distributions[indexes, :],
                                                            guess=list_of_guesses[method_index][indexes, :],
                                                            signatures=signatures)
                    values[method_index, i, metric_index] = torch.mean(rec_error)
                else:
                    values[method_index, i, metric_index] = metrics[metric]

    marker_size = 3
    line_width = 0.5
    p1 = axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,0]), marker='o',linewidth=line_width, markersize=marker_size)
    p2 = axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,1]), marker='o',linewidth=line_width, markersize=marker_size)
    p3 = axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,2]), marker='o',linewidth=line_width, markersize=marker_size)
    p4 = axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,3]), marker='o',linewidth=line_width, markersize=marker_size)
    p5 = axs[2,0].plot(np.log10(num_muts), np.transpose(values[:,:,4]), marker='o',linewidth=line_width, markersize=marker_size)
    p6 = axs[2,1].plot(np.log10(num_muts), np.transpose(values[:,:,5]), marker='o',linewidth=line_width, markersize=marker_size)
    p7 = axs[3,0].plot(np.log10(num_muts), np.transpose(values[:,:,6]), marker='o',linewidth=line_width, markersize=marker_size)
    
    # lines_labels = [ax.get_legend_handles_labels() for ax in axs.flat]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # print(lines)
    # print(labels)

    xlabel = 'log(N)'
    ylabel = ["MAE", "KL divergence", "Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)", "Reconstruction MSE"]

    for i, axes in enumerate(axs.flat):
        if i<len(axs.flat)-1:
            stylize_axes(axes, '', '', ylabel[i])
            if i == 5 or i ==6:
                stylize_axes(axes, '', xlabel, ylabel[i])
        else:
            axes.set_axis_off()
            axes.legend(p7, list_of_methods, loc='center left')#, prop={'size': 28})
    fig.tight_layout()
    plt.show()
    # plt.savefig(folder_path + '/benchmark.pdf')
    plt.close()

def plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, plot_path=None, show=False, signatures=None, mutation_distributions=None):
    fig, axs = plt.subplots(len(list_of_metrics), figsize=(8,6))
    fig.suptitle("Metrics vs Number of Mutations")
    
    num_muts = np.unique(label[:,-1].detach().numpy())

    for metric_index, metric in enumerate(list_of_metrics):
        values = np.zeros((len(list_of_methods), len(num_muts)))
        for method_index in range(len(list_of_methods)):
            for i, num_mut in enumerate(num_muts):
                indexes = label[:, -1] == num_mut
                metrics = get_classification_metrics(label_batch=label[indexes, :-1],
                                                     prediction_batch=list_of_guesses[method_index][indexes, :])
                
                if metric == "reconstruction_error":  # TODO(oleguer) Fix this, its super sketchy
                    assert(signatures is not None)
                    assert(mutation_distributions is not None)
                    rec_error = get_reconstruction_error(mutation_dist=mutation_distributions[indexes, :],
                                                         guess=list_of_guesses[method_index][indexes, :],
                                                         signatures=signatures)
                    values[method_index, i] = torch.mean(rec_error)
                else:
                    values[method_index, i] = metrics[metric]

    marker_size = 3
    line_width = 0.5
    axs.plot(np.log10(num_muts), np.transpose(values), marker='o',linewidth=line_width, markersize=marker_size)
    stylize_axes(axs, '', "log(N)", "Reconstruction MSE")
    fig.tight_layout()
    legend_adjustment = 0.75
    fig.subplots_adjust(right=legend_adjustment)   
    fig.legend(loc=7, labels=list_of_methods, prop={'size': 8})
    if show:
        plt.show()
    if plot_path is not None:
        # create_dir(plot_path)
        fig.savefig(plot_path)

def plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, plot_path=None, show=False):
    fig, axs = plt.subplots(len(list_of_metrics))
    fig.suptitle("Metrics vs Number of Signatures")
    
    num_sigs_ind = torch.sum(label[:, :-1]>0, 1)
    num_sigs = np.unique(num_sigs_ind.detach().numpy())
    for metric_index, metric in enumerate(list_of_metrics):
        values = np.zeros((len(list_of_methods), len(num_sigs)))
        for method_index in range(len(list_of_methods)):
            for i, sigs_index in enumerate(num_sigs):
                metrics = get_classification_metrics(label_batch=label[num_sigs_ind==sigs_index, :-1], prediction_batch=list_of_guesses[method_index][num_sigs_ind==sigs_index,:])
                values[method_index,i] = metrics[metric]
        
        handles = axs[metric_index].plot(num_sigs, np.transpose(values))
        axs[metric_index].set_ylabel(metric)
        if metric_index == len(list_of_metrics)-1:
            axs[metric_index].set_xlabel("N")

        # Shrink current axis by 3%
        box = axs[metric_index].get_position()
        axs[metric_index].set_position([box.x0, box.y0, box.width * 0.97, box.height])
    fig.legend(handles = handles, labels=list_of_methods, bbox_to_anchor=(1, 0.5))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    if show:
        plt.show()
    if plot_path is not None:
        # create_dir(plot_path)
        fig.savefig(plot_path)

def plot_reconstruction(input, weight_guess, signatures, ind_list, plot_path):
    create_dir(plot_path)
    reconstruction = torch.einsum("ij,bj->bi", (signatures, torch.tensor(weight_guess)))
    for i in ind_list:
        plt.bar(range(96), input[i,:], width=0.4)
        plt.bar(np.array(range(96))+0.4, reconstruction[i,:].detach().numpy(), width=0.4)
        plt.legend(["Input", "Reconstruction"])
        plt.show()
        # plt.savefig(plot_path + "_%s.png"%i)
        plt.close()


# ERRORLEARNER PLOTS:
def final_plot_interval_metrics_vs_mutations(label, pred_upper, pred_lower, sigs_names, plot_path=None, show=False):
    plt.figure(figsize=(8,6))

    num_muts = np.unique(label[:,-1].detach().numpy())
    values = np.zeros((6,len(num_muts)))
    for i, num_mut in enumerate(num_muts):
        k = -1
        indexes = label[:, -1] == num_mut
        metrics = get_pi_metrics(label[indexes, :-1], pred_lower[indexes, :], pred_upper[indexes, :])
        for metric in ["in_prop", "mean_interval_width"]:
            k += 1
            values[k,i] = metrics[metric]
    marker_size = 3
    line_width = 0.5
    xlabels = ["log(N)"]
    ylabels = ["Proportion in (%)", "Interval Width"]

    ax = plt.subplot(2,2,1)
    ax.plot(np.log10(num_muts), values[0], marker='o',linewidth=line_width, markersize=marker_size)
    stylize_axes(ax, '', xlabels[0], ylabels[0])
    ax = plt.subplot(2,2,2)
    ax.plot(np.log10(num_muts), values[1], marker='o',linewidth=line_width, markersize=marker_size)
    stylize_axes(ax, '', xlabels[0], ylabels[1])
    
    label_batch = label[:,:-1]
    lower = label_batch - pred_lower
    upper = pred_upper - label_batch
    num_error = torch.sum(lower<0, dim=0)
    num_error += torch.sum(upper<0, dim=0)
    num_error = num_error / label_batch.shape[0]
    num_classes = 72

    ax = plt.subplot(2,1,2)
    ax.bar(range(num_classes), 100*num_error, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    stylize_axes(ax, '', '', "Percentage of error (%)")
    xt = range(num_classes)
    xl = sigs_names
    # ax.set_xticks([xt[i] for i in range(num_classes) if i%2==0])
    ax.set_xticks(xt)
    ax.set_xticklabels([xl[i] if i%2==0 else '' for i in range(num_classes)], rotation=80)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.tight_layout()

    if show:
        plt.show()
    if plot_path is not None:
        plt.savefig(plot_path + 'interval_performance.pdf')


def plot_error_by_sig(label, pred_upper, pred_lower, sigs_names):
    lower = label - pred_lower
    upper = pred_upper - label
    num_error = torch.sum(lower < 0, dim=0)
    num_error += torch.sum(upper < 0, dim=0)
    num_error = num_error / label.shape[0]
    num_classes = 72
    
    fig, ax = plt.subplots()
    ax.bar(range(num_classes), 100*num_error, align='center', width=1, alpha=0.8, ecolor='black', capsize=10)
    stylize_axes(ax, '', '', "Percentage of error (%)")
    xt = range(num_classes)
    xl = sigs_names
    ax.set_xticks(xt)
    ax.set_xticklabels([xl[i] if i%2==0 else '' for i in range(num_classes)], rotation=80)
    return fig

def plot_width_by_sig(pred_upper, pred_lower, sigs_names):
    width = torch.mean(torch.abs(pred_upper - pred_lower), dim=0)
    num_classes = 72
    
    fig, ax = plt.subplots()
    ax.bar(range(num_classes), width, align='center', width=1, alpha=0.8, ecolor='black', capsize=10)
    stylize_axes(ax, '', '', "Width of interval")
    xt = range(num_classes)
    xl = sigs_names
    ax.set_xticks(xt)
    ax.set_xticklabels([xl[i] if i%2==0 else '' for i in range(num_classes)], rotation=80)
    return fig
    
    
def plot_interval_metrics_vs_mutations(label, pred_upper, pred_lower, plot_path=None, show=False):
    fig, axs = plt.subplots(2,2, figsize=(8,6))
    # fig.suptitle("Interval Metrics vs Number of Mutations")

    num_muts = np.unique(label[:,-1].detach().numpy())
    values = np.zeros((4,len(num_muts)))
    for i, num_mut in enumerate(num_muts):
        k = -1
        indexes = label[:, -1] == num_mut
        metrics = get_pi_metrics(label[indexes, :-1], pred_lower[indexes, :], pred_upper[indexes, :])
        for metric in metrics.keys():
            k += 1
            values[k,i] = metrics[metric]
    marker_size = 3
    line_width = 0.5
    axs[0,0].plot(np.log10(num_muts), values[0], marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), values[1], marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), values[2], marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), values[3], marker='o',linewidth=line_width, markersize=marker_size)

    xlabels = ["log(N)"]
    ylabels = ["Proportion in (%)", "Interval Width", "Interval Width Present", "Interval Width Absent" ]
    for i, axes in enumerate(axs.flat):
        stylize_axes(axes, '', xlabels[0], ylabels[i])

    fig.tight_layout()
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path)

def plot_interval_metrics_vs_sigs(label, pred_upper, pred_lower, plot_path=None, show=False):
    fig, axs = plt.subplots(2,2)
    fig.suptitle("Interval metrics vs Number of Signatures")

    num_sigs = list(range(1, 11))
    num_sigs_ind = torch.sum(label[:, :-1]>0, 1)
    values = np.zeros((4,len(num_sigs)))
    for i in range(len(num_sigs)):
        metrics = get_pi_metrics(label[num_sigs_ind==i+1, :-1], pred_lower[num_sigs_ind==i+1, :], pred_upper[num_sigs_ind==i+1, :])
        for k, metric in enumerate(metrics.keys()):
            values[k,i] = metrics[metric]
        
    axs[0,0].plot(num_sigs, values[0])
    axs[0,0].set_ylabel("Proportion in")
    axs[0,0].set_xlabel("N")

    axs[0,1].plot(num_sigs, values[1])
    axs[0,1].set_ylabel("Interval width")
    axs[0,1].set_xlabel("N")

    axs[1,0].plot(num_sigs, values[2])
    axs[1,0].set_ylabel("Interval width present")
    axs[1,0].set_xlabel("N")

    axs[1,1].plot(num_sigs, values[3])
    axs[1,1].set_ylabel("Interval width absent")
    axs[1,1].set_xlabel("N")

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path)

def plot_interval_performance(label_batch, pred_upper, pred_lower, sigs_names, plot_path=None, show=False): # Returns x,y
    label_batch = label_batch[:,:-1]
    lower = label_batch - pred_lower
    upper = pred_upper - label_batch
    num_error = torch.sum(lower<0, dim=0)
    num_error += torch.sum(upper<0, dim=0)
    num_error = num_error / label_batch.shape[0]
    num_classes = 72

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    # fig.suptitle('Confidence intervals performance')
    ax.bar(range(num_classes), 100*num_error, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Percentage of error (%)")
    
    stylize_axes(ax, '', '', "Percentage of error (%)")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')

    fig.tight_layout()
    if show:
        plt.show()
    if plot_path is not None:
        fig.savefig(plot_path)
    return range(num_classes), 100*num_error

def plot_interval_width_vs_mutations(label, upper, lower, show=True): # Returns x,y
    num_muts = np.unique(label[:,-1].detach().numpy())
    mean_width = np.zeros((len(num_muts)))
    for i in range(len(num_muts)):
        # Mean across signatures and across samples
        mean_width[i] = torch.mean(upper[1000*i:1000*(i+1),:] - lower[1000*i:1000*(i+1),:]).detach().numpy()
    
    plt.plot(np.log10(num_muts), mean_width)
    plt.ylabel("Mean interval width")
    plt.title("Mean interval width vs number of mutations")
    plt.show()
    return np.log10(num_muts), mean_width

def plot_interval_width_vs_mutations_some_sigs(label, upper, lower, list_of_sigs_ind, sigs_names, plot = True): # Returns x,y
    num_muts = np.unique(label[:,-1].detach().numpy())
    mean_width = np.zeros((len(num_muts),72))
    for i in range(len(num_muts)):
        # Mean across signatures and across samples
        mean_width[i] = torch.mean(upper[1000*i:1000*(i+1),:] - lower[1000*i:1000*(i+1),:], 0).detach().numpy()

    plt.plot(np.log10(num_muts), mean_width[:,list_of_sigs_ind])
    plt.ylabel("Mean interval width")
    plt.xlabel("log(N)")
    plt.title("Mean interval width vs number of mutations")
    plt.legend(np.array(sigs_names)[list_of_sigs_ind], title='Signatures', bbox_to_anchor=(1.02, 0.5), ncol=2)
    #plt.show()
    plt.savefig('../../plots/exp_0/width_vs_muts_some_sigs.png')
    return np.log10(num_muts), mean_width[:,list_of_sigs_ind]

def plot_propin_vs_mutations(label, upper, lower, plot = True): # Returns x,y
    num_muts = np.unique(label[:,-1].detach().numpy())
    mean_width = np.zeros((len(num_muts)))
    for i in range(len(num_muts)):
        # Mean across signatures and across samples
        mean_width[i] = torch.mean(upper[1000*i:1000*(i+1),:] - lower[1000*i:1000*(i+1),:]).detach().numpy()
    
    plt.plot(np.log10(num_muts), mean_width)
    plt.ylabel("Mean interval width")
    plt.title("Mean interval width vs number of mutations")
    plt.show()
    return np.log10(num_muts), mean_width


# WHOLE SIGNATURES-NET PLOTS:
def plot_confusion_matrix(label_list, predicted_list, class_names):
    conf_mat = confusion_matrix(label_list.numpy(), predicted_list.numpy())
    plt.figure(figsize=(15, 10))

    df_cm = pd.DataFrame(conf_mat, index=class_names,
                         columns=class_names).astype(int)
    heatmap = sn.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_weights(guessed_labels, pred_upper, pred_lower, sigs_names, plot_path):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots(figsize=(12,8))
    guessed_error_neg = guessed_labels - pred_lower
    guessed_error_pos = pred_upper - guessed_labels
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', alpha=0.5, ecolor='black', capsize=10)
    stylize_axes(ax, '', '', 'Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    ax.set_ylim([0,1])
    plt.tight_layout()
    plt.show()
    # fig.savefig(plot_path)
    plt.close()

def plot_weights_comparison(true_labels, guessed_labels, pred_upper, pred_lower, sigs_names, plot_path):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    guessed_error_neg = guessed_labels - pred_lower
    guessed_error_pos = pred_upper - guessed_labels
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10, label="guessed")
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center', label="true")
    # ax.set_ylim([0,1])
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    # fig.savefig(plot_path)

def plot_weights_comparison_deconstructSigs(true_labels, deconstructSigs_labels, guessed_labels, pred_upper, pred_lower, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    guessed_error_neg = guessed_labels - pred_lower
    guessed_error_pos = pred_upper - guessed_labels
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.bar(np.array(range(num_classes))-0.2,deconstructSigs_labels, width=0.2, align='center')
    ax.axhline(0.05, 0, num_classes, linestyle='--', c='red')
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.show()

def plot_bars(data, max=None):
    """Data is a dictionary with keys labels and values arrays of the same size
    """
    x = np.array(list(range(len(list(data.items())[0][1][0]))))[:max]
    width = 1/len(data.keys())
    bars = []
    for i, key in enumerate(data.keys()):
        y = torch.mean(data[key], dim=0).detach().numpy()[:max]
        bars.append(plt.bar(x + width*i, y, width, align='edge'))
    plt.legend(bars, data.keys())
    plt.show()

def get_correlation_matrix(data, signatures, plot = True):
    if plot:
        fig = plt.figure()
        df = pd.DataFrame(data.cpu().detach().numpy(), columns=signatures.columns[1:])
        corrMatrix = df.corr()
        sn.heatmap(corrMatrix, annot=False)
        return fig, corrMatrix
    else:
        df = pd.DataFrame(data.cpu().detach().numpy(), columns=signatures.columns[1:])
        corrMatrix = df.corr()
        return corrMatrix, corrMatrix

def plot_correlation_matrix(data, signatures):
    fig, corrMatrix = get_correlation_matrix(data, signatures)
    plt.show()
    return corrMatrix

def plot_histograms(data_dict, bins=100):
    for key in data_dict:
        data = data_dict[key].detach().cpu().numpy()
        plt.hist(data, density=True, bins=bins, label=key, alpha=0.5)  # density=False would make counts
    plt.legend()
    plt.show()

if __name__ == "__main__":
    deconstructSigs_labels = [0.1, 0.7, 0.2]
    real_labels = [0.2, 0.5, 0.3]
    guessed_labels = [0.25, 0.6, 0.2]
    guessed_error = [0.01, 0.04, 0.001]
    sigs_names = ["SBS1", "SBS2", "SBS3"]

    plot_weights_comparison_deconstructSigs(real_labels, deconstructSigs_labels, guessed_labels, guessed_error, sigs_names)