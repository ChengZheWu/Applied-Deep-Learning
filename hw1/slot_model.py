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
        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.BatchNorm1d(35),
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=2)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=2)
            )

    def forward(self, batch):
        out = self.embed(batch)
        out, _ = self.rnn(out)
        out = self.fc(out)
        return out
