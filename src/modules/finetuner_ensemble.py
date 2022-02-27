import torch

class FineTunerEnsemble:
    def __init__(self, models):
        self.models = models

    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):
        """Get weights of each signature in lexicographic wrt 1-mer
        """
        guesses = []
        for model in self.models:
            guess = model(mutation_dist, baseline_guess, num_mut)
            guess = self.__small_to_zero(guess)
            guesses.append(guess.unsqueeze(0))
        print(guesses)
        guesses = torch.cat(guesses)
        guesses = torch.mean(guesses, dim=0)
        guesses = guesses/torch.sum(guesses, dim=1).view(-1, 1)
        return guesses

    def __small_to_zero(self, a, thr = 0.01):
        """
        Small values to unknown category
        """
        a[a<thr] = 0
        return a