import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

from utilities.metrics import accuracy, false_random, false_realistic, get_classification_metrics, get_pi_metrics
from utilities.io import create_dir

# from utilities.metrics import get_classification_metrics, get_pi_metrics

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
def plot_metric_vs_mutations_classifier(guess, label, num_muts_list, plot_path = '../../plots/exp_classifier/performance_classifier.png'):
    fig, axs = plt.subplots(3)
    fig.suptitle("Metrics vs Number of Mutations")
    
    num_muts_list = num_muts_list[num_muts_list<=100000]
    num_muts = np.unique(num_muts_list.detach().numpy())
    
    values = np.zeros((3, len(num_muts)))
    for i in range(len(num_muts)):
        values[0,i] = accuracy(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        values[1,i] = false_realistic(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        values[2,i] = false_random(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        
    axs[0].plot(np.log10(num_muts), values[0,:])
    axs[0].set_ylabel("Accuracy (%)")

    axs[1].plot(np.log10(num_muts), values[1,:])
    axs[1].set_ylabel("False Realistic (%)")

    axs[2].plot(np.log10(num_muts), values[2,:])
    axs[2].set_ylabel("False Random (%)")

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig(plot_path)
    
# FINETUNER PLOTS:
def plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, plot_path):
    fig, axs = plt.subplots(len(list_of_metrics))
    fig.suptitle("Metrics vs Number of Mutations")
    
    num_muts = np.unique(label[:,-1].detach().numpy())

    for metric_index, metric in enumerate(list_of_metrics):
        values = np.zeros((len(list_of_methods), len(num_muts)))
        for method_index in range(len(list_of_methods)):
            for i, num_mut in enumerate(num_muts):
                indexes = label[:, -1] == num_mut
                metrics = get_classification_metrics(label_batch=label[indexes, :-1],
                                                     prediction_batch=list_of_guesses[method_index][indexes, :])
                values[method_index, i] = metrics[metric]
        
        handles = axs[metric_index].plot(np.log10(num_muts), np.transpose(values))
        axs[metric_index].set_ylabel(metric)
        if metric_index == len(list_of_metrics) - 1:
            axs[metric_index].set_xlabel("log(N)")

        # Shrink current axis by 3%
        box = axs[metric_index].get_position()
        axs[metric_index].set_position([box.x0, box.y0, box.width * 0.97, box.height])
    
    fig.legend(handles = handles, labels=list_of_methods, bbox_to_anchor=(1, 0.5))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    create_dir(plot_path)
    fig.savefig(plot_path)

def plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, plot_path):
    #TODO: Adapt this function to work like plot_metric_vs_mutations
    
    fig, axs = plt.subplots(len(list_of_metrics))
    fig.suptitle("Metrics vs Number of Signatures")
    
    num_sigs = list(range(1, 11))
    num_sigs_ind = torch.sum(label[:, :-1]>0, 1)
    for m, metric in enumerate(list_of_metrics):
        values = np.zeros((len(list_of_methods), len(num_sigs)))

        for k in range(len(list_of_methods)):
            for i in range(len(num_sigs)):
                if label[num_sigs_ind==i+1, :-1].shape[0] == 0:  # TODO: There is a bug in this for loop, we should take a look
                    continue
                metrics = get_classification_metrics(label_batch=label[num_sigs_ind==i+1, :-1], prediction_batch=list_of_guesses[k][num_sigs_ind==i+1,:])
                values[k,i] = metrics[metric]
        
        handles = axs[m].plot(num_sigs, np.transpose(values))
        axs[m].set_ylabel(metric)
        if m == len(list_of_metrics)-1:
            axs[m].set_xlabel("N")

        # Shrink current axis by 3%
        box = axs[m].get_position()
        axs[m].set_position([box.x0, box.y0, box.width * 0.97, box.height])
    fig.legend(handles = handles, labels=list_of_methods, bbox_to_anchor=(1, 0.5))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig(plot_path)

# ERRORLEARNER PLOTS:
def plot_interval_metrics_vs_mutations(label, pred_upper, pred_lower, plot_path):
    fig, axs = plt.subplots(2,2)
    fig.suptitle("Interval metrics vs Number of Mutations")

    num_muts = np.unique(label[:,-1].detach().numpy())
    values = np.zeros((4,len(num_muts)))
    for i in range(len(num_muts)):
        k = -1
        metrics = get_pi_metrics(label[1000*i:1000*(i+1), :-1], pred_lower[1000*i:1000*(i+1), :], pred_upper[1000*i:1000*(i+1), :])
        for metric in metrics.keys():
            k += 1
            values[k,i] = metrics[metric]
        
    axs[0,0].plot(np.log10(num_muts), values[0])
    axs[0,0].set_ylabel("Proportion in")
    axs[0,0].set_xlabel("log(N)")

    axs[0,1].plot(np.log10(num_muts), values[1])
    axs[0,1].set_ylabel("Interval width")
    axs[0,1].set_xlabel("log(N)")

    axs[1,0].plot(np.log10(num_muts), values[2])
    axs[1,0].set_ylabel("Interval width present")
    axs[1,0].set_xlabel("log(N)")

    axs[1,1].plot(np.log10(num_muts), values[3])
    axs[1,1].set_ylabel("Interval width absent")
    axs[1,1].set_xlabel("log(N)")

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig(plot_path)

def plot_interval_metrics_vs_sigs(label, pred_upper, pred_lower, plot_path):
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
    plt.show()
    fig.savefig(plot_path)

def plot_interval_performance(label_batch, pred_upper, pred_lower, sigs_names, plot_path): # Returns x,y
    lower = label_batch - pred_lower
    upper = pred_upper - label_batch
    num_error = torch.sum(lower<0, dim=0)
    num_error += torch.sum(upper<0, dim=0)
    num_error = num_error / label_batch.shape[0]
    num_classes = 72
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Confidence intervals performance')
    ax.bar(range(num_classes), 100*num_error, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Percentage of error (%)")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig(plot_path)
    return range(num_classes), 100*num_error

def plot_interval_width_vs_mutations(label, upper, lower, plot = True): # Returns x,y
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
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return conf_mat

def plot_weights(guessed_labels, pred_upper, pred_lower, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    guessed_error_neg = guessed_labels - pred_lower
    guessed_error_pos = pred_upper - guessed_labels
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    plt.show()

def plot_weights_comparison(true_labels, guessed_labels, pred_upper, pred_lower, sigs_names, plot_path):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    guessed_error_neg = guessed_labels - pred_lower
    guessed_error_pos = pred_upper - guessed_labels
    ax.bar(range(num_classes),guessed_labels, yerr=[abs(guessed_error_neg), abs(guessed_error_pos)], align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.set_ylim([0,1])
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig(plot_path)

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

if __name__ == "__main__":
    deconstructSigs_labels = [0.1, 0.7, 0.2]
    real_labels = [0.2, 0.5, 0.3]
    guessed_labels = [0.25, 0.6, 0.2]
    guessed_error = [0.01, 0.04, 0.001]
    sigs_names = ["SBS1", "SBS2", "SBS3"]

    plot_weights_comparison_deconstructSigs(real_labels, deconstructSigs_labels, guessed_labels, guessed_error, sigs_names)