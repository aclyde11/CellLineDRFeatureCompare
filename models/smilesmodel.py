import torch.nn as nn


class SmilesModel(nn.Module):

    def __init__(self, flen, dropout_rate, intermediate_rep=128, maxlen=320, vocab_size=512):
        super(SmilesModel, self).__init__()
        self.feature_length = flen

        self.embedding_layer = nn.Embedding(vocab_size, 96)

        self.lstm = nn.GRU(96, 64, num_layers=4, dropout=dropout_rate, batch_first=True, )
        self.model(
            nn.Linear(64 * maxlen, intermediate_rep)
        )

    def forward(self, features):
        emb = self.embedding_layer(features)
        emb, _ = self.lstm(emb)
        emb = emb.view(self.emb[0], -1)
        return self.model(emb)
