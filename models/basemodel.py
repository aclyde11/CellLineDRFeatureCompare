import torch
import torch.nn as nn


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


class BaseModel(nn.Module):
    def __init__(self, rflen, dr, intermediate_rep_drugs=128, featureModel=None, **kwargs):
        super(BaseModel, self).__init__()
        self.feature_length = rflen

        self.feature_model = featureModel(dropout_rate=dr, intermediate_rep=intermediate_rep_drugs, **kwargs)

        self.dropout = nn.Dropout(dr)
        self.rnamodel = nn.Sequential(
            nn.Linear(rflen, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            self.dropout,

            nn.Linear(256, 64),
            nn.ReLU(),
            self.dropout,

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.dropout,

            nn.Linear(64, 64),
        )

        self.basemodel = nn.Sequential(
            nn.Linear(intermediate_rep_drugs + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.dropout,

            nn.Linear(128, 64),
            nn.ReLU(),
            self.dropout,

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            self.dropout,

            nn.Linear(32, 1)
        )

    def forward(self, rnafeatures, *args):
        print('origin shape', rnafeatures.shape)
        drug_latent = self.feature_model(*args)
        rnafeatures = self.rnamodel(rnafeatures)
        print(drug_latent.shape, rnafeatures.shape)
        x = torch.cat([rnafeatures, drug_latent], dim=-1)
        return self.basemodel(x)
