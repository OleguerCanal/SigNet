

import numpy as np

def do_bootstrap(input_sample):
    N_bootstrap = 1000
    N = len(input_sample)

    for i in range(N_bootstrap):
        new_labels = np.random.choice(input_sample, size=N, replace=True)
