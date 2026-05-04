import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super(TransformerExtractor, self).__init__(observation_space, features_dim)
        
        # Observation space shape is (1, window_size)
        # We treat this as (batch, seq_len, input_dim) where input_dim is 1
        self.seq_len = observation_space.shape[1]
        self.input_dim = 1
        
        # Linear layer to project input to d_model
        self.embedding = nn.Linear(self.input_dim, features_dim)
        
        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_len, features_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global Average Pooling or just take the last element?
        # Let's use the last element representation for trading
        self.flatten = nn.Flatten()
        
        # Recompute features_dim based on flattening if necessary, 
        # but here we want to return features_dim. 
        # So we'll add a final linear layer to map to features_dim.
        self.output_layer = nn.Linear(features_dim * self.seq_len, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, 1, seq_len) -> need (batch, seq_len, 1)
        x = observations.transpose(1, 2)
        
        # Embed and add positional encoding
        x = self.embedding(x) + self.pos_encoding
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Flatten and project to features_dim
        x = self.flatten(x)
        return self.output_layer(x)
