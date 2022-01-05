import os
import sys

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from utilities.io import read_data, read_data_generator, read_real_data, read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs

def plot_weights_comparison(true_labels, guessed_labels, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),guessed_labels, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.set_ylim([0,1])
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.legend(['Input', 'Output'])
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()

model_directory = "../../trained_models/exp_good/generator"

# # Load data
# train_data, val_data = read_data_generator(device="cpu", data_folder="../../data")

# Load generator and get predictions
generator = read_model(model_directory)
# generator_output, mean, var = generator(x=val_data.inputs, noise=False)
    
# input0 = val_data.inputs[0,:]
# output0, mean, var = generator(x=input0, noise=False)
# data = pd.read_excel("../../data/data.xlsx")


from sklearn.decomposition import PCA


# Read data
train_data, val_data = read_data_generator(device="cpu", data_folder="../../data")
points_real = val_data.inputs

num_points = points_real.size()[0]
# num_points = 5000

train_data, val_data = read_data(device="cpu", experiment_id="exp_good", source="realistic_large", data_folder="../../data", include_baseline=False, include_labels=True)
points_realistic = val_data.labels[:num_points, :].detach().numpy()

pca = PCA(n_components=2, whiten=True).fit(points_real)
points_real_2d = pca.transform(points_real)

points_generator = generator.generate(num_points)
points_generator_2d = pca.transform(points_generator.detach().numpy())

points_realistic_2d = pca.transform(points_realistic)

plt.scatter(x=points_realistic_2d[:, 0], y=points_realistic_2d[:, 1], c="green")
plt.scatter(x=points_generator_2d[:, 0], y=points_generator_2d[:, 1], c="orange")
plt.scatter(x=points_real_2d[:, 0], y=points_real_2d[:, 1], c="blue")
plt.legend(["Realistic", "Generated", "Real"])
plt.show()


pca = PCA(n_components=3, whiten=True).fit(points_real)
points_real_3d = pca.transform(points_real)

points_generator = generator.generate(num_points)
points_generator_3d = pca.transform(points_generator.detach().numpy())

points_realistic_3d = pca.transform(points_realistic)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(points_realistic_3d[:, 0], points_realistic_3d[:, 1], points_realistic_3d[:, 2], c="green")
ax.scatter(points_generator_3d[:, 0], points_generator_3d[:, 1], points_generator_3d[:, 2], c="orange")
ax.scatter(points_real_3d[:, 0], points_real_3d[:, 1], points_real_3d[:, 2], c="blue")
plt.legend(["Realistic", "Generated", "Real"])
plt.show()


# fig = px.scatter(components, x=0, y=1, color=df['species'])
# fig.show()

print(val_data.inputs)
print(generator_output)
print(mean)
print(var)
