#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

import torch
import torch.nn as nn

class ManZoneTransformer(nn.Module):

  def __init__(self, feature_len=5, model_dim=64, num_heads=2, num_layers=4, dim_feedforward=256, dropout=0.1, output_dim=2):
      super(ManZoneTransformer, self).__init__()
      self.feature_norm_layer = nn.BatchNorm1d(feature_len)

      self.feature_embedding_layer = nn.Sequential(
          nn.Linear(feature_len, model_dim),
          nn.ReLU(),
          nn.LayerNorm(model_dim),
          nn.Dropout(dropout),
      )

      transformer_encoder_layer = nn.TransformerEncoderLayer(
          d_model=model_dim,
          nhead=num_heads,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          batch_first=True,
      )
      self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

      self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

      self.decoder = nn.Sequential(
          nn.Linear(model_dim, model_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(model_dim, model_dim // 4),
          nn.ReLU(),
          nn.LayerNorm(model_dim // 4),
          nn.Linear(model_dim // 4, output_dim),
      )

  def forward(self, x):
      # x shape: (batch_size, num_players, feature_len)
      x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
      x = self.feature_embedding_layer(x)
      x = self.transformer_encoder(x)
      x = self.player_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
      x = self.decoder(x)
      return x