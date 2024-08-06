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
        self.lstm_block = LSTMBlock(config)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        
        # Initialization
        h = torch.zeros(batch_size, self.config.hidden_size).to(device) # Hidden state
        c = torch.zeros(batch_size, self.config.hidden_size).to(device) # Cell state

        # Loop through the sequence
        for t in range(seq_len):
            h, c = self.lstm_block(input_seq[:, t, :], h, c)

        # Output layer
        output = self.output_layer(h)
        
        return output