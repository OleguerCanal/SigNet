import numpy as np
import pandas as pd
import torch


def read_data(device, data_folder="../data", realistic_data = False, num_classes=72):
    if realistic_data == False:
        train_input = torch.tensor(pd.read_csv(
        data_folder + "/train_input_w01.csv", header=None).values, dtype=torch.float)
        train_input = train_input.to(device)
        train_guess_0 = torch.tensor(pd.read_csv(
            data_folder + "/train_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
        train_guess_0 = train_guess_0.to(device)
        train_label = torch.tensor(pd.read_csv(
            data_folder + "/train_label_w01.csv", header=None).values, dtype=torch.float)
        train_label = train_label.to(device)
    else:
        train_input = pd.read_csv(data_folder + "/realistic_data/ground.truth.syn.catalog_train.csv", index_col=[0, 1])
        train_input = torch.transpose(torch.from_numpy(np.array(train_input.values, dtype=np.float32)), 0, 1).to("cpu")
        train_input = train_input / torch.sum(train_input, dim=1).reshape(-1, 1)
        train_input = train_input.to(device)
        train_guess_0 = torch.tensor(pd.read_csv(
            data_folder + "/realistic_data/w0_train_fixed.csv", header=0).values, dtype=torch.float)
        train_guess_0 = train_guess_0.to(device)
        
        train_label = pd.read_csv(data_folder + "/realistic_data/ground.truth.syn.exposures_train.csv", index_col=0)
        train_label_full = pd.DataFrame(np.zeros((num_classes, train_label.shape[1])))
        data = pd.read_excel(data_folder + "/data.xlsx")
        train_label_full.index = data.columns[2:]
        train_label_full.columns = train_label.columns
        train_label_full.update(train_label)

        train_label_full = torch.transpose(torch.from_numpy(np.array(train_label_full.values, dtype=np.float32)), 0, 1).to("cpu")
        number_mutations = torch.sum(train_label_full, dim=1)
        train_label_full = train_label_full / torch.sum(train_label_full, dim=1).reshape(-1, 1)
        train_label_full = torch.cat((train_label_full, number_mutations.reshape(-1,1)), dim=1)
        train_label_full = train_label_full.to(device)
        train_label = train_label_full

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/validation_input_w01.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/validation_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/validation_label_w01.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)
    return train_input, train_guess_0, train_label, val_input, val_guess_0, val_label

def read_test_data(device, data_folder="../data", realistic_data = True, num_classes=72):
    if realistic_data == True:
        test_input = pd.read_csv(data_folder + "/realistic_data/ground.truth.syn.catalog_test.csv", index_col=[0, 1])
        test_input = torch.transpose(torch.from_numpy(np.array(test_input.values, dtype=np.float32)), 0, 1).to("cpu")
        test_input = test_input / torch.sum(test_input, dim=1).reshape(-1, 1)
        test_input = test_input.to(device)

        test_guess_0 = torch.tensor(pd.read_csv(
            data_folder + "/realistic_data/w0_test.csv", header=0).values, dtype=torch.float)
        test_guess_0 = test_guess_0.to(device)

        test_label = pd.read_csv(data_folder + "/realistic_data/ground.truth.syn.exposures_test.csv", index_col=0)
        test_label_full = pd.DataFrame(np.zeros((num_classes, test_label.shape[1])))
        data = pd.read_excel(data_folder + "/data.xlsx")
        test_label_full.index = data.columns[2:]
        test_label_full.columns = test_label.columns
        test_label_full.update(test_label)

        test_label_full = torch.transpose(torch.from_numpy(np.array(test_label_full.values, dtype=np.float32)), 0, 1).to("cpu")
        number_mutations = torch.sum(test_label_full, dim=1)
        test_label_full = test_label_full / torch.sum(test_label_full, dim=1).reshape(-1, 1)
        test_label_full = torch.cat((test_label_full, number_mutations.reshape(-1,1)), dim=1)
        test_label_full = test_label_full.to(device)
        test_label = test_label_full

        return test_input, test_guess_0, test_label