import torch

from utilities.io import read_signatures, tensor_to_csv, csv_to_tensor, read_model, read_data, read_test_data

def get_labels(data, random_finetuner, realistic_finetuner, signatures):
    
    def get_reconstruction_MSE(model):
        guess = model(mutation_dist=data.inputs,
                      baseline_guess=data.prev_guess,
                      num_mut=data.num_mut)
        guess_reconstruction = torch.einsum("ij,bj->bi", signatures, guess)
        label_reconstruction = torch.einsum("ij,bj->bi", signatures, data.labels)
        errors = torch.nn.MSELoss(reduce=False)(guess_reconstruction, label_reconstruction)
        return errors

    realistic_errors = get_reconstruction_MSE(model=realistic_finetuner)
    random_errors = get_reconstruction_MSE(model=random_finetuner)

    classification_label = (realistic_errors < random_errors).to(torch.float)  # 1 if realistic has less error 
    print("mean classification:", torch.mean(classification_label).item(), "(this should be close to 0)")
    return classification_label.reshape(-1, 1)

def shuffle(inputs, labels):
    indexes = torch.randperm(inputs.shape[0])
    return inputs[indexes, ...], labels[indexes, ...]

if __name__ == "__main__":
    data_folder = "../data"
    models_folder = "../trained_models"
    signatures = read_signatures(data_folder + "/data.xlsx")
    
    # Label all realistic data as a 1
    train_realistic_inputs = csv_to_tensor(data_folder + "/exp_0/train_realistic_input.csv")
    train_realistic_nummut = csv_to_tensor(data_folder + "/exp_0/train_realistic_label.csv")[:, -1]
    val_realistic_inputs = csv_to_tensor(data_folder + "/exp_0/val_realistic_input.csv")
    val_realistic_nummut = csv_to_tensor(data_folder + "/exp_0/val_realistic_label.csv")[:, -1]
    train_realistic_labels = torch.ones((train_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    val_realistic_labels = torch.ones((val_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)

    # Label random data according to which model achieves a better reconstruction
    train_random, val_random = read_data(device="cpu",
                                         experiment_id="exp_0",
                                         source="random",
                                         data_folder=data_folder)

    random_finetuner = read_model(models_folder + "/exp_0/finetuner_random")
    realistic_finetuner = read_model(models_folder + "/exp_0/finetuner_realistic")

    train_random_labels = get_labels(data=train_random,
                                     random_finetuner=random_finetuner,
                                     realistic_finetuner=realistic_finetuner,
                                     signatures=signatures)
    
    val_random_labels = get_labels(data=val_random,
                                   random_finetuner=random_finetuner,
                                   realistic_finetuner=realistic_finetuner,
                                   signatures=signatures)

    train_classification_inputs = torch.cat((train_realistic_inputs, train_random.inputs))
    train_classification_labels = torch.cat((train_realistic_labels, train_random_labels))
    train_classification_inputs, train_classification_labels = shuffle(inputs=train_classification_inputs,
                                                                       labels=train_classification_labels)

    val_classification_inputs = torch.cat((val_realistic_inputs, val_random.inputs))
    val_classification_labels = torch.cat((val_realistic_labels, val_random_labels))
    val_classification_inputs, val_classification_labels = shuffle(inputs=val_classification_inputs,
                                                                   labels=val_classification_labels)
    
    tensor_to_csv(train_classification_inputs, data_folder + "/exp_classifier_corrected/train_input.csv")
    tensor_to_csv(train_classification_labels, data_folder + "/exp_classifier_corrected/train_label.csv")
    tensor_to_csv(train_realistic_nummut, data_folder + "/exp_classifier_corrected/train_num_mut.csv")
    tensor_to_csv(val_classification_inputs, data_folder + "/exp_classifier_corrected/val_input.csv")
    tensor_to_csv(val_classification_labels, data_folder + "/exp_classifier_corrected/val_label.csv")
    tensor_to_csv(val_realistic_nummut, data_folder + "/exp_classifier_corrected/val_num_mut.csv")
