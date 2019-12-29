import matplotlib.pyplot as plt
import numpy as np
import sklearn


class PytorchHistory:
    def __init__(self, metric=sklearn.metrics.r2_score, metric_name='r2'):
        self.train_loss = []
        self.test_loss = []
        self.train_r2 = []
        self.test_r2 = []
        self.metric = metric
        self.metric_name = metric_name
        self.true_tracker, self.pred_tracker = [], []

    def log_loss(self, loss, train=True):
        if train:
            self.train_loss.append(loss)
        else:
            self.test_loss.append(loss)

    def track_metric(self, pred, value):
        self.pred_tracker.append(pred)
        self.true_tracker.append(value)

    def get_last_metric(self, train=True):
        if train:
            return self.train_r2[-1]
        else:
            return self.test_r2[-1]

    def log_metric(self, r2=None, train=True, internal=False):
        if internal:
            self.true_tracker = np.concatenate(self.true_tracker).flatten()
            self.pred_tracker = np.concatenate(self.pred_tracker).flatten()
            r2 = self.metric(self.true_tracker, self.pred_tracker)
            self.true_tracker, self.pred_tracker = [], []
        if train:
            self.train_r2.append(r2)
        else:
            self.test_r2.append(r2)

    def plot_loss(self, save_file=None, title='Loss', figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.plot(list(range(len(self.train_loss))), self.train_loss, label='Train Loss')
        plt.plot(list(range(len(self.test_loss))), self.test_loss, label='Test Loss')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file, bbox_inches='tight', dpi=300)

    def plot_metric(self, save_file=None, title='Loss', figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.plot(list(range(len(self.train_r2))), self.train_r2, label='Train r2')
        plt.plot(list(range(len(self.test_r2))), self.test_r2, label='Test r2')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_name)

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
