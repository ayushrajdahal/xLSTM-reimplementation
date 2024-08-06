"""
LSTM: Long Short-Term Memory Network

This module implements a standard LSTM network for reference.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class LSTMConfig:
    """
    Configuration for the LSTM model.
    """
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float = 0.0
    bidirectional: bool = False

class LSTMBlock(nn.Module):
    """
    A single block of the LSTM network.
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Input weight vectors between inputs x_t and cell input, input gate, forget gate, and output gate, respectively.
        self.Wzx = nn.Linear(config.input_size, config.hidden_size)
        self.Wix = nn.Linear(config.input_size, config.hidden_size)
        self.Wfx = nn.Linear(config.input_size, config.hidden_size)
        self.Wox = nn.Linear(config.input_size, config.hidden_size)

        # Recurrent weights between hidden state h_t−1 and cell input, input gate, forget gate, and output gate, respectively.
        self.Rzh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rih = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rfh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Roh = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x, h_prev, c_prev):

        if self.training:
            x = self.dropout(x) # Apply dropout to the input during training

        z = torch.tanh(self.Wzx(x) + self.Rzh(h_prev)) # Cell input
        i = torch.sigmoid(self.Wix(x) + self.Rih(h_prev)) # Input gate
        f = torch.sigmoid(self.Wfx(x) + self.Rfh(h_prev)) # Forget gate
        o = torch.sigmoid(self.Wox(x) + self.Roh(h_prev)) # Output gate

        # New cell state
        c = f * c_prev + i * z
        # New hidden state
        h = o * torch.tanh(c)

        return h, c

class LSTM(nn.Module):
    """
    The main LSTM network.
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.lstm_blocks = nn.ModuleList([LSTMBlock(config) for _ in range(config.num_layers)])
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        
        # Initialization
        h = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)] # Hidden state
        c = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)] # Cell state

        # Loop through the sequence
        for t in range(seq_len):
            x = input_seq[:, t, :]
            for layer in range(self.config.num_layers):
                h[layer], c[layer] = self.lstm_block(x, h[layer], c[layer])
                x = h[layer]

        # Output layer (final hidden state)
        output = self.output_layer(h[-1])
        
        return output