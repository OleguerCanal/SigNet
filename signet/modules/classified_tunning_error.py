import logging

class ClassifiedFinetunerErrorfinder:

    def __init__(self,
                 classifier,
                 finetuner,
                 errorfinder,
                 classification_cutoff=0.5,
                 device="cpu"):
        """Instantiate a ClassifiedFinetuner

        Args:
            classifier (Classifier): Model to discriminate between realistic and random data
            realistic_finetuner (Finetuner or CombinedFinetuner): Model which improves baseline guess for random data
            random_finetuner (Finetuner or CombinedFinetuner): Model which improves baseline guess for random data
            classification_cutoff (float, optional): Cuttoff at which we decide something is realistic. Defaults to 0.5.
            device (str, optional): Device to use (cuda or cpu). Defaults to "cpu".
        """

        self.classification_cutoff = classification_cutoff
        self.device = device

        self.classifier = classifier
        self.finetuner = finetuner
        self.errorfinder = errorfinder

    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):

        logging.info("Detecting out-of-train-distribution points...")
        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)
        logging.info("Detecting out-of-train-distribution points... DONE")
        
        logging.info("Finetuning NNLS guesses...")
        finetuner_guess = self.finetuner(mutation_dist=mutation_dist,
                                         baseline_guess = baseline_guess,
                                         num_mut=num_mut)
        logging.info("Finetuning NNLS guesses... DONE")

        logging.info("Estimating errorbars...")
        upper, lower = self.errorfinder(weights=finetuner_guess[:,:-1],
                                                num_mutations=num_mut,
                                                classification=classification.reshape(-1, 1))
        logging.info("Estimating errorbars... DONE")
        
        result = {"finetuner_guess": finetuner_guess,
                  "error_upper": upper,
                  "error_lower": lower,
                  "classification": classification}
        return result