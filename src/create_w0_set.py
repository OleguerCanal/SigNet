from utilities.metrics import get_jensen_shannon
from models.signature_finder import SignatureFinder
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    signatures_path = "../data/data.xlsx"
    data_path = "../data/realistic_data/ground.truth.syn.catalog_train.csv"
    output_file = "../data/realistic_data/w0_train.csv"

    signatures = pd.read_excel(signatures_path)
    signatures_list = [torch.tensor(signatures.iloc[:, i]).type(torch.float32)
                       for i in range(2, 74)]

    training_data = pd.read_csv(data_path, index_col=[0, 1])
    print(len(training_data.columns))
    training_data = torch.transpose(torch.from_numpy(
        np.array(training_data.values, dtype=np.float32)), 0, 1).to("cpu")
    training_data = training_data / \
        torch.sum(training_data, dim=1).reshape(-1, 1)

    sf = SignatureFinder(signatures_list, metric=get_jensen_shannon)

    dataset_size = training_data.shape[0]
    batch_size = 1000  # How often to save
    for i in range(int(1 + int(dataset_size)/int(batch_size))):
        print("Progress:", 100*float(i)/float(1 + dataset_size/batch_size), "%")
        index_min = min(i*batch_size, dataset_size)
        index_max = min((i+1)*batch_size, dataset_size)
        if index_min == index_max:
            continue
        input_batch = training_data[index_min:index_max]
        sol = sf.get_weights_batch(input_batch=input_batch, n_workers=10)
        sol = sol.detach().numpy()
        df = pd.DataFrame(sol)
        df.to_csv(output_file,
                  header=signatures.columns.tolist()[2:][index_min:index_max],
                  mode='a',
                  index=False)  # Append to csv
        print("Training done!")
