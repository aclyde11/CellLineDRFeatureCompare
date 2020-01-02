import torch
import torch.nn as nn


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)

class BaseModelAttend(nn.Module):
    def __init__(self, rflen, dr, intermediate_rep_drugs=128, rnaseq=None, featureModel=None, **kwargs):
        super(BaseModelAttend, self).__init__()
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
        self.rnafeatures = rnaseq
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

    def forward(self, *args):
        drug_latent = self.feature_model(*args)
        rnafeatures = self.rnamodel(self.rnafeatures)
        print(rnafeatures, drug_latent.shape)
        x = torch.cat([rnafeatures.repeat([drug_latent.shape[0], 1]), drug_latent], dim=-1)
        return self.basemodel(x)

class BaseModel(nn.Module):
    def __init__(self, rflen, dr, intermediate_rep_drugs=128, featureModel=None, return_attns=False, **kwargs):
        super(BaseModel, self).__init__()
        self.feature_length = rflen
        self.return_attns=return_attns

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
        drug_latent = self.feature_model(*args)
        if isinstance(drug_latent, tuple):
            drug_latent, attn = drug_latent
        else:
            attn  = None
        rnafeatures = self.rnamodel(rnafeatures)
        x = torch.cat([rnafeatures, drug_latent], dim=-1)

        if self.return_attns:
            return self.basemodel(x), attn
        else:
            return self.basemodel(x)
