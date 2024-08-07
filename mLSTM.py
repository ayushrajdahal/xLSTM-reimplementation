import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class mLSTMConfig:
    """
    Configuration for the mLSTM model.
    """
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float = 0.0

class mLSTMBlock(nn.Module):
    """
    A single block of the mLSTM network.
    """
    def __init__(self, config: mLSTMConfig):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        
        self.wix = nn.Linear(config.input_size, config.hidden_size) # input gate weights
        self.wfx = nn.Linear(config.hidden_size, config.hidden_size) # forget gate weights
        self.Wox = nn.Linear(config.hidden_size, (config.hidden_size ** 2)) # output gate weights
        self.Wkqv = nn.Linear(config.input_size, (config.hidden_size ** 2) * 3) # key, query, value weights

    def forward(self, x, h_prev, C_prev, n_prev):
        i = torch.exp(self.wix(x)) # input gate
        f = torch.exp(self.wfx(h_prev)) # forget gate
        o = torch.sigmoid(self.Wox(h_prev)) # output gate

        k, q, v = self.Wkqv(x).chunk(3, dim=-1)

        k = k / math.sqrt(k.size(-1))

        C = f @ C_prev + i @ (v @ k.T) # cell state
        n = f @ n_prev + i @ k # normalizer state

        h_hat = C @ q / torch.max(torch.abs(n.T @ q), 1)
        h = o * h_hat

        return h, C, n

class mLSTM(nn.Module):
    """
    The main mLSTM network.
    """
    def __init__(self, config: mLSTMConfig):
        super().__init__()
        self.config = config
        self.lstm_blocks = nn.ModuleList([mLSTMBlock(config) for _ in range(config.num_layers)])
        self.output_layer = nn.Linear(config.hidden_size ** 2, config.output_size)

    def forward(self, input_seq):
        
        device = input_seq.device

        batch_size, seq_len, _ = input_seq.size()
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)]
        C = [torch.zeros(batch_size, batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)]
        n = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)]

        # for x in input_seq.split(1, dim=0):
        #     h, C, n = self.block(x.squeeze(0), h, C, n)

        for t in range(seq_len):
            x = input_seq[:, t, :]
            for layer in range(self.config.num_layers):
                h[layer], C[layer], n[layer] = self.lstm_blocks[layer](x, h[layer], C[layer], n[layer])
                x = h[layer] # Output of the current layer is the input to the next
        
        output = self.output_layer(h[-1])

        return output