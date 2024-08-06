import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float = 0.0
    bidirectional: bool = False


class LSTMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input weight vectors between inputs x_t and cell input, input gate, forget gate, and output gate, respectively
        self.Wzx = nn.Linear(config.input_size, config.hidden_size)
        self.Wix = nn.Linear(config.input_size, config.hidden_size)
        self.Wfx = nn.Linear(config.input_size, config.hidden_size)
        self.Wox = nn.Linear(config.input_size, config.hidden_size)

        # recurrent weights between hidden state h_t−1 and cell input, input gate, forget gate, and output gate, respectively
        self.Rzh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rih = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rfh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Roh = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        z = torch.tanh(self.Wzx(x) + self.Rzh(h_prev)) # cell input
        i = torch.sigmoid(self.Wix(x) + self.Rih(h_prev)) # input gate
        f = torch.sigmoid(self.Wfx(x) + self.Rfh(h_prev)) # forget gate
        o = torch.sigmoid(self.Wox(x) + self.Roh(h_prev)) # output gate

        c = f * c_prev + i * z # new cell state
        h = o * torch.tanh(c) # new hidden state

        return h, c


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm_block = LSTMBlock(config)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        h = torch.zeros(batch_size, self.config.hidden_size).to(device)
        c = torch.zeros(batch_size, self.config.hidden_size).to(device)

        for t in range(seq_len):
            h, c = self.lstm_block(input_seq[:, t, :], h, c)

        output = self.output_layer(h)
        return output