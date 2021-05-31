import os
import sys

import numpy as np
import pandas as pd
import torch

from tests.model_tester import ModelTester

data_folder = "../data/other_methods/"

if __name__=="__main__":
    list_of_methods = ["decompTumor2Sig_51", "MutationalPatterns_51",  "SignatureEstimationQP_51", "YAPSA_51"]
    #list_of_methods = ["decompTumor2sig", "deconstructSigs", "MutationalPatterns", "sigfit", "signature_estimationQP", "signature_estimationSA", "YAPSA"]
    for method in list_of_methods:
        inferred_results_file = "inferred.exposures_%s.csv"%method
        inferred = pd.read_csv(os.path.join(data_folder, inferred_results_file), index_col=0)

        # Create dataframe filled with 0's
        real = pd.DataFrame(0, index=inferred.index.values.tolist(), columns=inferred.columns.values.tolist())

        # Update rows that have signature
        real_different_zero = pd.read_csv(os.path.join(data_folder, "ground.truth.syn.exposures_test.csv"), index_col=0)
        real.update(real_different_zero)

        # To tensor
        inferred_tensor = torch.transpose(torch.from_numpy(np.array(inferred.values, dtype=np.float)), 0, 1)
        real_tensor = torch.transpose(torch.from_numpy(np.array(real.values, dtype=np.float)), 0, 1)

        # Normalize
        inferred_tensor = inferred_tensor / torch.sum(inferred_tensor, dim=1).reshape(-1,1)
        real_tensor = real_tensor / torch.sum(real_tensor, dim=1).reshape(-1,1)

        model_tester = ModelTester(num_classes=51)
        print(method)
        model_tester.test(guessed_labels=inferred_tensor, true_labels=real_tensor)

