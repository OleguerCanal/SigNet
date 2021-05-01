from error_finder import ErrorFinder
from finetuner import FineTuner
from signature_finder import SignatureFinder


class SignatureNet:
    def __init__(self,
                 signature_finder_params,
                 finetuner_params,
                 error_learner_params):
        self.signature_finder = SignatureFinder(**signature_finder_params)
        self.finetuner = FineTuner(**finetuner_params)
        self.error_finder = ErrorFinder(**error_learner_params)

    def __call__(self, mutation_vec):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        num_mutations = np.sum(mutation_vec, axis=1)
        normalized_mutation_vec = mutation_vec/num_mutations
        weight_guess_0 = self.signature_finder.get_weights_batch(
            normalized_mutation_vec)
        weight_guess_1 = self.finetuner(mutation_dist=normalized_mutation_vec,
                                           weights=weight_guess_0)
        positive_errors, negative_errors = self.error_finder(weights=weight_guess_1,
                                                             num_mutations=num_mutations)
