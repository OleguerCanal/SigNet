import os
import sys
from unicodedata import name

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_model, read_signatures
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from utilities.data_generator import DataGenerator

signatures = read_signatures("../../data/data.xlsx")

datagenerator = DataGenerator(signatures)

sbs90 = [0]*72
sbs90[-1] = 1
sbs89 = [0]*72
sbs89[-2] = 1
sbs88 = [0]*72
sbs88[-3] = 1
labels = torch.Tensor([sbs90, sbs89, sbs88])
inputs, labels = datagenerator.make_input(labels, "test", "", normalize=True, seed = 0)
print(labels)

# Run Classifier
classifier = read_model("../../trained_models/exp_final/classifier")
classifier_guess = classifier(inputs, labels[:, -1].view(-1, 1))
print(classifier_guess)
print(torch.reshape(classifier_guess, (10,3)))
guess90 = [classifier_guess[i].item() for i in range(len(classifier_guess)) if i%3 == 0]
guess89 = [classifier_guess[i].item() for i in range(len(classifier_guess)) if i%3 == 1]
guess88 = [classifier_guess[i].item() for i in range(len(classifier_guess)) if i%3 == 2]
# plt.bar(np.array(range(len(guess90))), np.log(guess90), alpha = 0.7, label="SBS90")
# plt.bar(np.array(range(len(guess90)))+0.1, np.log(guess89), alpha = 0.7, label="SBS89")
# plt.bar(np.array(range(len(guess90)))+0.2, np.log(guess88), alpha = 0.7, label="SBS88")
# plt.legend()
# plt.show()

# Run Baseline
print("Running Baseline")
sf = Baseline(signatures)
test_baseline = sf.get_weights_batch(inputs, n_workers=4)
print(test_baseline)

# Run Finetuner
models_path = "../../trained_models/exp_final"
finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "/finetuner_low",
                                large_mum_mut_dir=models_path + "/finetuner_large")
finetuner_guess = finetuner(mutation_dist=inputs,
                            baseline_guess=test_baseline,
                            num_mut=labels[:, -1].view(-1, 1))

print(finetuner_guess)
for i in range(3):
    plt.bar(range(72), test_baseline[i,:], alpha = 0.7, label='Baseline')
    plt.bar(range(72), finetuner_guess[i,:], alpha = 0.7, label='Finetuner')
    plt.legend()
    plt.show()