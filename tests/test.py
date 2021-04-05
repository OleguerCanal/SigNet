import sys
import os

import numpy as np
import torch 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *

a = np.random.uniform(size=(1,10))
b = np.random.uniform(size=(1,10))

#a = np.array([1,2], dtype=np.float)
#b = np.array([4,6], dtype=np.float)

print((get_MSE(torch.tensor(a), torch.tensor(b))*torch.tensor(a).shape[-1])**0.5)
#print(get_MSE(torch.tensor(a),torch.tensor(b)))
print(((np.linalg.norm(a-b, ord=2))))
print(a.shape[0])
print(a)