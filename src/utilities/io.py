import os

import pandas as pd
import torch

def read_data(device, experiment_id, source, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    assert(source in ["random", "realistic"])
    path = os.path.join(data_folder, experiment_id)

    def csv_to_tensor(file):
        input_tensor = torch.tensor(pd.read_csv(file, header=None).values, dtype=torch.float)
        assert(not torch.isnan(input_tensor).any())
        assert(torch.count_nonzero(torch.sum(input_tensor, axis=1)) == input_tensor.shape[0])
        return input_tensor.to(device)

    train_input = csv_to_tensor(path + "/train_%s_input.csv"%source)
    train_baseline = csv_to_tensor(path + "/train_%s_baseline.csv"%source)
    train_label = csv_to_tensor(path + "/train_%s_label.csv"%source)

    val_input = csv_to_tensor(path + "/val_%s_input.csv"%source)
    val_baseline = csv_to_tensor(path + "/val_%s_baseline.csv"%source)
    val_label = csv_to_tensor(path + "/val_%s_label.csv"%source)

    return train_input, train_baseline, train_label, val_input, val_baseline, val_label

def read_methods_random_data(device, data_folder="../data"):
    label = torch.tensor(pd.read_csv(
        data_folder + "/random_data/test_label_w01.csv", header=None).values, dtype=torch.float)
    label = label.to(device)
    decompTumor2Sig_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/decompTumor2Sig_random_test_guess.csv", header=None).values, dtype=torch.float)
    decompTumor2Sig_guess = decompTumor2Sig_guess.to(device)
    deconstructSigs_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/deconstructSigs_random_test_guess.csv", header=None).values, dtype=torch.float)
    deconstructSigs_guess = deconstructSigs_guess.to(device)
    MutationalPatterns_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/MutationalPatterns_random_test_guess.csv", header=None).values, dtype=torch.float)
    MutationalPatterns_guess = MutationalPatterns_guess.to(device)
    mutSignatures_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/mutSignatures_random_test_guess.csv", header=None).values, dtype=torch.float)
    mutSignatures_guess = mutSignatures_guess.to(device)
    SignatureEstimationQP_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/SignatureEstimationQP_random_test_guess.csv", header=None).values, dtype=torch.float)
    SignatureEstimationQP_guess = SignatureEstimationQP_guess.to(device)
    YAPSA_guess = torch.tensor(pd.read_csv(
        data_folder + "/random_data/methods/YAPSA_random_test_guess.csv", header=None).values, dtype=torch.float)
    YAPSA_guess = YAPSA_guess.to(device)
    return label, decompTumor2Sig_guess, deconstructSigs_guess, MutationalPatterns_guess, mutSignatures_guess, SignatureEstimationQP_guess, YAPSA_guess

def read_realistic_test_methods(device, data_folder="../data"):
    label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/test_realistic_label.csv", header=None).values, dtype=torch.float)
    label = label.to(device)
    decompTumor2Sig_guess = torch.tensor(pd.read_csv(   
        data_folder + "/realistic_test/methods/decompTumor2Sig_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    decompTumor2Sig_guess = decompTumor2Sig_guess.to(device)
    deconstructSigs_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/methods/deconstructSigs_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    deconstructSigs_guess = deconstructSigs_guess.to(device)
    MutationalPatterns_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/methods/MutationalPatterns_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    MutationalPatterns_guess = MutationalPatterns_guess.to(device)
    mutSignatures_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/methods/mutSignatures_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    mutSignatures_guess = mutSignatures_guess.to(device)
    SignatureEstimationQP_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/methods/SignatureEstimationQP_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    SignatureEstimationQP_guess = SignatureEstimationQP_guess.to(device)
    YAPSA_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_test/methods/YAPSA_realistic_larger_test_guess.csv", header=None).values, dtype=torch.float)
    YAPSA_guess = YAPSA_guess.to(device)
    return label, decompTumor2Sig_guess, deconstructSigs_guess, MutationalPatterns_guess, mutSignatures_guess, SignatureEstimationQP_guess, YAPSA_guess

def read_methods_realistic_data(device, data_folder="../data"):
    label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/realistic_test_label.csv", header=None).values, dtype=torch.float)
    label = label.to(device)
    decompTumor2Sig_guess = torch.tensor(pd.read_csv(   
        data_folder + "/realistic_data/methods/decompTumor2Sig_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    decompTumor2Sig_guess = decompTumor2Sig_guess.to(device)
    deconstructSigs_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/methods/deconstructSigs_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    deconstructSigs_guess = deconstructSigs_guess.to(device)
    MutationalPatterns_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/methods/MutationalPatterns_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    MutationalPatterns_guess = MutationalPatterns_guess.to(device)
    mutSignatures_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/methods/mutSignatures_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    mutSignatures_guess = mutSignatures_guess.to(device)
    SignatureEstimationQP_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/methods/SignatureEstimationQP_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    SignatureEstimationQP_guess = SignatureEstimationQP_guess.to(device)
    YAPSA_guess = torch.tensor(pd.read_csv(
        data_folder + "/realistic_data/methods/YAPSA_realistic_test_guess.csv", header=None).values, dtype=torch.float)
    YAPSA_guess = YAPSA_guess.to(device)
    return label, decompTumor2Sig_guess, deconstructSigs_guess, MutationalPatterns_guess, mutSignatures_guess, SignatureEstimationQP_guess, YAPSA_guess

def read_data_realistic(device, data_folder="../data/realistic_data/train_default"):
    train_input = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_input.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_guess_0 = train_guess_0.to(device)
    train_label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_label.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_input.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_label.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)
    return train_input, train_guess_0, train_label, val_input, val_guess_0, val_label

def read_data_realistic_larger(device, data_folder="../data/realistic_data"):
    train_input = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_input.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_guess_0 = train_guess_0.to(device)
    train_label = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_label.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_input.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_label.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)