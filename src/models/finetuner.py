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

        self.layer1_1 = nn.Linear(num_classes, num_units).double()  # Baseline guess path
        # 96 = total number of possible muts
        self.layer1_2 = nn.Linear(96, num_units).double()  # Input path
        # Number of mutations path
        self.layer1_3 = nn.Linear(1, num_units_branch_mut).double()

        self.layer2_1 = nn.Linear(num_units, num_units).double()
        self.layer2_2 = nn.Linear(num_units, num_units).double()
        self.layer2_3 = nn.Linear(num_units_branch_mut, num_units_branch_mut).double()

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units_joined_path, num_units_joined_path).double()
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units_joined_path, num_classes).double()
        self.activation = nn.LeakyReLU(0.1).double()

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                mutation_dist,
                weights,
                num_mut):
        assert(not torch.isnan(mutation_dist).any())
        assert(not torch.isnan(weights).any())
        assert(not torch.isnan(num_mut).any())

        all_layers = [self.layer1_1, self.layer1_2, self.layer1_3,
                        self.layer2_1, self.layer2_2, self.layer2_3,
                        self.output_layer]
        for layer in all_layers:
            print("l_mean:", layer, torch.mean(torch.abs(layer.weight)).item())
            assert(not torch.isnan(layer.weight).any())
        
        for layer in self.hidden_layers:
            print("l_mean:", layer, torch.mean(torch.abs(layer.weight)).item())
            assert(not torch.isnan(layer.weight).any())
        print("####")

        # import pdb; pdb.set_trace()

        # Input head
        mutation_dist = self.activation(self.layer1_2(mutation_dist))
        mutation_dist = self.activation(self.layer2_2(mutation_dist))

        # Baseline head
        weights = self.activation(self.layer1_1(weights))
        weights = self.activation(self.layer2_1(weights))

        # Number of mutations head
        num_mut = torch.log10(num_mut)
        assert(not torch.isnan(num_mut).any())
        num_mut = self.activation(self.layer1_3(num_mut))
        num_mut = self.activation(self.layer2_3(num_mut))

        # Concatenate
        comb = torch.cat([mutation_dist, weights, num_mut], dim=1)
        assert(not torch.isnan(comb).any())

        # Apply shared layers
        for layer in self.hidden_layers:
            # print("comb mean:", torch.mean(torch.abs(comb)).item())
            # print("layer_mean:", torch.mean(torch.abs(layer.weight)).item())
            # print("####")
            comb = self.activation(layer(comb))
        assert(not torch.isnan(comb).any())
        # Apply output layer
        comb = self.output_layer(comb)
        assert(not torch.isnan(comb).any())
        comb = self.softmax(comb)

        # If in eval mode, send small values to 0
        if not self.training:
            # print("eval")
            mask = (comb > self._cutoff).type(torch.int).float()
            comb = comb*mask
            comb = torch.div(comb, torch.sum(
                comb, axis=1).reshape((-1, 1)) + self._EPS)
        # print("train")
        assert(not torch.isnan(comb).any())

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
    # del data.prev_guess
    del finetuner
    gc.collect()
    torch.cuda.empty_cache()
    # data.next_guess = next_guess
    return data