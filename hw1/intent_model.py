from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embedding_dim = embeddings.size(1)
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=1)
            )

    def forward(self, batch):
        out1 = self.embed(batch)
        out2, _ = self.rnn(out1)
        out2 = out2[:, -1, :]
        out = self.fc(out2)
        return out
