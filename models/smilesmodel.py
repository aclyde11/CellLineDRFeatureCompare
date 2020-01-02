import torch.nn as nn
import torch

class SmilesModel(nn.Module):

    def __init__(self, flen, dropout_rate, intermediate_rep=128, maxlen=320, vocab=512, embeds=None):
        super(SmilesModel, self).__init__()
        self.feature_length = flen

        self.embedding_layer = nn.Embedding(len(vocab), 96)
        if embeds is not None:
            print(torch.from_numpy(embeds).float().shape)
            self.embedding_layer.from_pretrained(torch.from_numpy(embeds).float())

        self.lstm = nn.GRU(96, 64, num_layers=4, dropout=dropout_rate, batch_first=True, )
        self.model = nn.Sequential(
            nn.Linear(64 * maxlen, intermediate_rep)
        )

    def forward(self, features):
        emb = self.embedding_layer(features)
        emb, _ = self.lstm(emb)
        emb = emb.reshape(emb.shape[0], -1)
        return self.model(emb)
