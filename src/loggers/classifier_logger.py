import torch
import wandb


class ClassifierLogger:
    def __init__(self):
        self.metrics = {
            # "mse": get_MSE,
            # "cos": get_negative_cosine_similarity,
            # "cross_ent": get_cross_entropy2,
            # "KL": get_kl_divergence,
            # "JS": get_jensen_shannon,
            # "W": get_wasserstein_distance,
        }
        self.threshold = 0.5

    def accuracy(self, prediction, label):
        prediction = (prediction > self.threshold).float()*1
        return torch.sum(prediction == label)/torch.numel(prediction)*100
    
    def false_realistic(self, prediction, label):
        prediction = (prediction > self.threshold).float()
        return torch.sum(label[prediction == 1] == 0)/torch.numel(label[prediction == 1])*100

    def false_random(self, prediction, label):
        prediction = (prediction > self.threshold).float()
        return torch.sum(label[prediction == 0] == 1)/torch.numel(label[prediction == 0])*100

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            val_loss,
            val_prediction,
            val_label,
            step):

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        wandb.log({"train_accuracy": self.accuracy(train_prediction, train_label).item()})
        wandb.log({"val_accuracy": self.accuracy(val_prediction, val_label).item()})

        wandb.log({"train_false_realistic": self.false_realistic(train_prediction, train_label).item()})
        wandb.log({"val_false_realistic": self.false_realistic(val_prediction, val_label).item()})

        wandb.log({"train_false_random": self.false_random(train_prediction, train_label).item()})
        wandb.log({"val_false_random": self.false_random(val_prediction, val_label).item()})
