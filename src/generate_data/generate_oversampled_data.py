sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data_generator, read_signatures
from utilities.oversampler import CancerTypeOverSampler
from utilities.data_generator import DataGenerator

if __name__ == "__main__":
    train_data, val_data = read_data_generator(device=dev,
                                               data_id="real_data",
                                               data_folder = "../data/",
                                               cosmic_version = 'v3',
                                               type='real',
                                               prop_train = 0.8)
    oversampler = CancerTypeOverSampler(train_data.inputs, train_data.cancer_types)
    train_label = oversampler.get_even_set()         # Oversample to create set with same number of samples per cancer type
    val_label = val_data.inputs

    # Create inputs associated to the labels
    signatures = read_signatures("../data/data.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)
    train_input, train_label = data_generator.make_input(train_label, "train", config["network_type"], normalize=True)
    val_input, val_label = data_generator.make_input(val_label, "val", config["network_type"], normalize=True)
    
    # Run Baseline
    sf = Baseline(signatures)
    train_baseline = sf.get_weights_batch(train_input, n_workers=2)
    val_baseline = sf.get_weights_batch(val_input, n_workers=2)
    
    # Create DataPartitions
    train_data = DataPartitions(inputs=train_input,
                                prev_guess=train_baseline,
                                labels=train_label)
    val_data = DataPartitions(inputs=val_input,
                                prev_guess=val_baseline,
                                labels=val_label)

    tensor_to_csv(train_data.inputs, "../data/exp_oversample/train_%s_input.csv"%config["network_type"])
    tensor_to_csv(train_label, "../data/exp_oversample/train_%s_label.csv"%config["network_type"])
    tensor_to_csv(train_data.prev_guess, "../data/exp_oversample/train_%s_baseline.csv"%config["network_type"])

    tensor_to_csv(val_data.inputs, "../data/exp_oversample/val_%s_input.csv"%config["network_type"])
    tensor_to_csv(val_label, "../data/exp_oversample/val_%s_label.csv"%config["network_type"])
    tensor_to_csv(val_data.prev_guess, "../data/exp_oversample/val_%s_baseline.csv"%config["network_type"])
        