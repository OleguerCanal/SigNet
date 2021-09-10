import torch
import torch.nn as nn


class FineTuner(nn.Module):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 cutoff=0.05):
        super(FineTuner, self).__init__()
        self._cutoff = cutoff
        self._EPS = 1e-6

        # Num units of the mutations path
        num_units_branch_mut = 10
        num_units_joined_path = 2*num_units + num_units_branch_mut

        self.layer1_1 = nn.Linear(num_classes, num_units)  # Baseline guess path
        # 96 = total number of possible muts
        self.layer1_2 = nn.Linear(96, num_units)  # Input path
        # Number of mutations path
        self.layer1_3 = nn.Linear(1, num_units_branch_mut)

        self.layer2_1 = nn.Linear(num_units, num_units)
        self.layer2_2 = nn.Linear(num_units, num_units)
        self.layer2_3 = nn.Linear(num_units_branch_mut, num_units_branch_mut)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units_joined_path, num_units_joined_path)
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units_joined_path, num_classes)
        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                mutation_dist,
                weights,
                num_mut):
        # Input head
        mutation_dist = self.activation(self.layer1_2(mutation_dist))
        mutation_dist = self.activation(self.layer2_2(mutation_dist))

        # Baseline head
        weights = self.activation(self.layer1_1(weights))
        weights = self.activation(self.layer2_1(weights))

        # Number of mutations head
        # num_mut = torch.log10(num_mut)
        num_mut = nn.Sigmoid()((num_mut-500000)/100000)
        assert(not torch.isnan(num_mut).any())
        num_mut = self.activation(self.layer1_3(num_mut))
        num_mut = self.activation(self.layer2_3(num_mut))

        # Concatenate
        comb = torch.cat([mutation_dist, weights, num_mut], dim=1)
        assert(not torch.isnan(comb).any())

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        comb = self.output_layer(comb)
        comb = self.softmax(comb)

        # If in eval mode, send small values to 0
        if not self.training:
            mask = (comb > self._cutoff).type(torch.int).float()
            comb = comb*mask
            comb = torch.div(comb, torch.sum(
                comb, axis=1).reshape((-1, 1)) + self._EPS)
        return comb


def baseline_guess_to_finetuner_guess(finetuner_args, trained_finetuner_file, data):
    # Load finetuner and compute guess_1
    import gc
    finetuner = FineTuner(**finetuner_args)
    finetuner.to("cpu")
    finetuner.load_state_dict(torch.load(
        trained_finetuner_file, map_location=torch.device('cpu')))
    finetuner.eval()

    with torch.no_grad():
        data.prev_guess = finetuner(mutation_dist=data.inputs,
                                    weights=data.prev_guess,
                                    num_mut=data.num_mut)
    del finetuner
    gc.collect()
    torch.cuda.empty_cache()
    return data