import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from utilities.io import read_data_generator, read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs

model_directory = "../../trained_models/exp_good/generator"

# Load data
train_data, val_data = read_data_generator(device="cpu", data_folder="../../data")

# Load generator and get predictions
generator = read_model(model_directory)
generator_output, mean, var = generator(x=val_data.inputs, noise=False)

print(val_data.inputs)
print(generator_output)
print(mean)
print(var)