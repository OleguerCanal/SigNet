import numpy as np
import torch

from utilities.io import read_signatures, tensor_to_csv, csv_to_tensor, read_model, read_data, read_test_data
from utilities.data_partitions import DataPartitions
from modules.combined_finetuner import CombinedFinetuner
from tqdm import tqdm

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
    print("realistic_error mean:", torch.mean(realistic_errors).item())
    print("random_error mean:", torch.mean(random_errors).item())

    classification_label = (realistic_errors < random_errors).to(torch.float)  # 1 if realistic has less error 
    print("mean classification:", torch.mean(classification_label).item(), "(this should be close to 0)")
    return classification_label.reshape(-1, 1)

def shuffle(inputs, labels, num_mut):
    indexes = torch.randperm(inputs.shape[0])
    return inputs[indexes, ...], labels[indexes, ...], num_mut[indexes, ...]

def print_intersections(A, B):
    arr1 = A.numpy().view(np.int32)
    arr2 = B.numpy().view(np.int32)
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view, return_indices=True)
    print("intersected:", intersected[0], len(intersected[0][0]))

def array_row_intersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   result = a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]
   print(result.shape)

if __name__ == "__main__":
    data_folder = "../data"
    models_folder = "../trained_models/exp_2_nets/"
    signatures = read_signatures(data_folder + "/data.xlsx")

    # Read models
    random_finetuner = CombinedFinetuner(low_mum_mut_dir=models_folder + "finetuner_random", 
                                         large_mum_mut_dir=models_folder + "finetuner_random_large")
    realistic_finetuner = CombinedFinetuner(low_mum_mut_dir=models_folder + "finetuner_realistic", 
                                         large_mum_mut_dir=models_folder + "finetuner_realistic_large")
    
    # Read all realistic data    
    train_realistic_inputs = csv_to_tensor(data_folder + "/exp_0/train_realistic_input.csv")
    train_realistic_nummut = csv_to_tensor(data_folder + "/exp_0/train_realistic_label.csv")[:, -1].view(-1, 1)
    val_realistic_inputs = csv_to_tensor(data_folder + "/exp_0/val_realistic_input.csv")
    val_realistic_nummut = csv_to_tensor(data_folder + "/exp_0/val_realistic_label.csv")[:, -1].view(-1, 1)
    test_realistic_inputs = csv_to_tensor(data_folder + "/exp_0/test_realistic/test_realistic_input.csv")
    test_realistic_nummut = csv_to_tensor(data_folder + "/exp_0/test_realistic/test_realistic_label.csv")[:, -1].view(-1, 1)
    
    # Label all realistic data as a 1
    train_realistic_labels = torch.ones((train_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    val_realistic_labels = torch.ones((val_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    test_realistic_labels = torch.ones((test_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)

    # Read random data
    train_random, val_random = read_data(device="cpu",
                                         experiment_id="exp_0",
                                         source="random",
                                         data_folder=data_folder)
    test_random_inputs_ = csv_to_tensor(data_folder + "/exp_0/test_random/test_random_input.csv")
    test_random_labels_ = csv_to_tensor(data_folder + "/exp_0/test_random/test_random_label.csv")
    test_random_baseline_ = csv_to_tensor(data_folder + "/exp_0/test_random/test_random_baseline_yapsa.csv")
    test_random = DataPartitions(inputs=test_random_inputs_,
                                 labels=test_random_labels_,
                                 prev_guess=test_random_baseline_)

    print("--train:")
    train_random_labels = get_labels(data=train_random,
                                     random_finetuner=random_finetuner,
                                     realistic_finetuner=realistic_finetuner,
                                     signatures=signatures)
    
    # print("--val:")
    val_random_labels = get_labels(data=val_random,
                                   random_finetuner=random_finetuner,
                                   realistic_finetuner=realistic_finetuner,
                                   signatures=signatures)
    
    print("--test:")
    test_random_labels = get_labels(data=test_random,
                                   random_finetuner=random_finetuner,
                                   realistic_finetuner=realistic_finetuner,
                                   signatures=signatures)

    train_classification_inputs = torch.cat((train_realistic_inputs, train_random.inputs))
    train_classification_labels = torch.cat((train_realistic_labels, train_random_labels))
    train_classification_numut = torch.cat((train_realistic_nummut, train_random.num_mut))
    train_classification_inputs, train_classification_labels, train_classification_numut =\
        shuffle(inputs=train_classification_inputs,
                labels=train_classification_labels,
                num_mut=train_classification_numut)

    val_classification_inputs = torch.cat((val_realistic_inputs, val_random.inputs))
    val_classification_labels = torch.cat((val_realistic_labels, val_random_labels))
    val_classification_numut = torch.cat((val_realistic_nummut, val_random.num_mut))
    val_classification_inputs, val_classification_labels, val_classification_numut =\
        shuffle(inputs=val_classification_inputs,
                labels=val_classification_labels,
                num_mut=val_classification_numut)
    
    # split val-train
    # val_classification_inputs = train_classification_inputs[-10000:, ...]
    # val_classification_labels = train_classification_labels[-10000:, ...]
    # val_classification_numut = train_classification_numut[-10000:, ...]

    # train_classification_inputs = train_classification_inputs[:-10000, ...]
    # train_classification_labels = train_classification_labels[:-10000, ...]
    # train_classification_numut = train_classification_numut[:-10000, ...]
    
    test_classification_inputs = torch.cat((test_realistic_inputs, test_random.inputs))
    test_classification_labels = torch.cat((test_realistic_labels, test_random_labels))
    test_classification_numut = torch.cat((test_realistic_nummut, test_random.num_mut))
    test_classification_inputs, test_classification_labels, test_classification_numut =\
         shuffle(inputs=test_classification_inputs,
                 labels=test_classification_labels,
                 num_mut=test_classification_numut)

    print("--summary:")
    print("proportion of real train:", torch.mean(train_classification_labels).item(), train_classification_labels.shape)
    print("proportion of real val:", torch.mean(val_classification_labels).item(), val_classification_labels.shape)
    print("proportion of real test:", torch.mean(test_classification_labels).item(), test_classification_labels.shape)

    # check for repetitions
    # print("train-val")
    # print_intersections(train_classification_labels[:100, ...], val_classification_labels)
    # array_row_intersection(train_classification_inputs.numpy()[0:1000, ...], val_classification_inputs.numpy())
    # count = 0.0
    # for i in tqdm(range(val_classification_inputs.shape[0])):
    #     for j in range(train_classification_inputs.shape[0]):
    #         if torch.all(torch.eq(val_classification_inputs[i, ...], train_classification_inputs[j, ...])):
    #             count += 1.0
    #             print(count/val_classification_inputs.shape[0])
    # print("train-test")
    # print_intersections(train_classification_inputs, test_classification_inputs)
    
    # store everything
    tensor_to_csv(train_classification_inputs, data_folder + "/exp_classifier_corrected/train_input.csv")
    tensor_to_csv(train_classification_labels, data_folder + "/exp_classifier_corrected/train_label.csv")
    tensor_to_csv(train_classification_numut, data_folder + "/exp_classifier_corrected/train_num_mut.csv")

    tensor_to_csv(val_classification_inputs, data_folder + "/exp_classifier_corrected/val_input.csv")
    tensor_to_csv(val_classification_labels, data_folder + "/exp_classifier_corrected/val_label.csv")
    tensor_to_csv(val_classification_numut, data_folder + "/exp_classifier_corrected/val_num_mut.csv")

    tensor_to_csv(test_classification_inputs, data_folder + "/exp_classifier_corrected/test_input.csv")
    tensor_to_csv(test_classification_labels, data_folder + "/exp_classifier_corrected/test_label.csv")
    tensor_to_csv(test_classification_numut, data_folder + "/exp_classifier_corrected/test_num_mut.csv")


