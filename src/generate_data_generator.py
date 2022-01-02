import torch
import matplotlib.pyplot as plt

from utilities.io import read_data, read_model, read_data_generator

def make_hist(data):
    indexes = []
    for example in data:
        idx = (example > 0).nonzero().flatten()
        indexes += idx.tolist()
    plt.hist(indexes, bins=72)
    plt.show()

if __name__ == "__main__":
    generator = read_model("../trained_models/exp_good/generator", device="cuda")
    generator.to("cuda")
    examples = generator.generate(5000)
    examples[examples <= 0.01] = 0
    make_hist(examples)

    real_data, val_real_data = read_data_generator(device="cpu")
    real_data.inputs[real_data.inputs<=0.01] = 0
    make_hist(real_data.inputs)
