import torch.nn as nn


class VectorModel(nn.Module):

    def __init__(self, flen, dropout_rate, intermediate_rep=128):
        super(VectorModel, self).__init__()
        self.feature_length = flen

        self.model = nn.Sequential(
            nn.Linear(flen, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
        )

    def forward(self, features):
        return self.model(features)
