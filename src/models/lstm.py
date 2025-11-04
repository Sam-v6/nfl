#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidir, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden_size * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):  # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)        # out: (B, T, H)
        last = out[:, -1, :]                  # use last timestep representation
        logits = self.head(last)              # (B, C)
        return logits