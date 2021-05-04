import numpy as np
import pandas as pd
import torch

def normalize_data(input_file, abundances):
    abundances_tensor = torch.Tensor(abundances)
    return torch.div(input_file, abundances_tensor)



def create_opportunities(abundances_path):
    # Create opportunities
    order = ['ACA','ACC','ACG','ACT','CCA','CCC','CCG','CCT','GCA','GCC','GCG','GCT','TCA','TCC','TCG','TCT',
             'ACA','ACC','ACG','ACT','CCA','CCC','CCG','CCT','GCA','GCC','GCG','GCT','TCA','TCC','TCG','TCT',
             'ACA','ACC','ACG','ACT','CCA','CCC','CCG','CCT','GCA','GCC','GCG','GCT','TCA','TCC','TCG','TCT',
             'ATA','ATC','ATG','ATT','CTA','CTC','CTG','CTT','GTA','GTC','GTG','GTT','TTA','TTC','TTG','TTT',
             'ATA','ATC','ATG','ATT','CTA','CTC','CTG','CTT','GTA','GTC','GTG','GTT','TTA','TTC','TTG','TTT',
             'ATA','ATC','ATG','ATT','CTA','CTC','CTG','CTT','GTA','GTC','GTG','GTT','TTA','TTC','TTG','TTT']
    
    opp_file = open(abundances_path, 'r')
    opp_file_lines = opp_file.readlines()
    opp_dic = {}
    for line in opp_file_lines:
        opp_dic[line.split("\t")[0]] = int(line.strip("\n").split("\t")[1])
    opp = np.zeros(96)
    for i in range(len(order)):
        tri = order[i]
        opp[i] = opp_dic[tri]
        b1 = tri[0]
        b2 = tri[1]
        b3 = tri[2]
        complement_base = complement(b3) + complement(b2) + complement(b1)
        opp[i] = opp[i] + opp_dic[complement_base]
    return opp



def complement(base):
    if base == "A":
        return "T"
    if base == "C":
        return "G"
    if base == "G":
        return "C"
    if base == "T":
        return "A"



# Example:
# data = torch.tensor(pd.read_csv("../data/test_input_w01.csv", header=None).values, dtype=torch.float)
# label_mut_batch = torch.tensor(pd.read_csv("../data/test_label_w01.csv", header=None).values, dtype=torch.float)
# label_batch = label_mut_batch[:,:72]
# num_mut = torch.reshape(label_mut_batch[:,72], (list(label_mut_batch.size())[0],1))
# data_total = data*num_mut
# data_2 = data_total[:2,:]
# print(data_2)
# opp = create_opportunities("../data/3mer_rn6.txt")
# normalized_data = normalize_data(data_2, opp) 
# print(normalized_data)