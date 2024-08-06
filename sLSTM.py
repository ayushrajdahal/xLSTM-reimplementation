"""
sLSTM: Scalar Long Short-Term Memory Network

This module implements an enhanced version of the traditional LSTM with several key improvements:
1) Scalar or sequence-level updates for memory cells, which simplifies the memory structure compared to traditional LSTMs.
2) Exponential gating: Exponential gating is a significant enhancement in xLSTM that addresses the limitation of traditional LSTMs in revising storage decisions. This technique involves:
    - Using exponential activation functions for the input and forget gates.
    - Providing more flexible control over information flow in the memory cell.
    - Enabling the model to better update stored values when more relevant information is encountered.
    - Improving the model's ability to perform tasks like nearest neighbor search.
3) Memory mixing: sLSTM supports multiple memory cells and allows for memory mixing via recurrent connections. This means that information can be shared and mixed across different cells within the same head, enhancing the model's ability to extract complex patterns and track states.
4) Normalization and stabilization: To prevent numerical instabilities, sLSTM introduces a normalizer state that tracks the product of input and future forget gates.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class sLSTMConfig:
    """
    Configuration for the sLSTM model.
    """
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float = 0.0
    bidirectional: bool = False

class sLSTMBlock(nn.Module):
    """
    A single block of the sLSTM network.
    """
    def __init__(self, config: sLSTMConfig):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Input weight vectors between inputs xt and cell input, input gate, forget gate, and output gate, respectively.
        self.Wzx = nn.Linear(config.input_size, config.hidden_size)
        self.Wix = nn.Linear(config.input_size, config.hidden_size)
        self.Wfx = nn.Linear(config.input_size, config.hidden_size)
        self.Wox = nn.Linear(config.input_size, config.hidden_size)

        # Recurrent weights between hidden state ht−1 and cell input, input gate, forget gate, and output gate, respectively.
        self.Rzh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rih = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rfh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Roh = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x, h_prev, c_prev, n_prev, m_prev):

        x = self.dropout(x) # Apply dropout to the input

        z = torch.tanh(self.Wzx(x) + self.Rzh(h_prev))      # Cell input
        i = torch.exp(self.Wix(x) + self.Rih(h_prev))       # Input gate (MODIFIED)
        f = torch.exp(self.Wfx(x) + self.Rfh(h_prev))       # Forget gate (MODIFIED): can be exp OR sigmoid
        o = torch.sigmoid(self.Wox(x) + self.Roh(h_prev))   # Output gate

        # Excerpt from the paper:
        # We broadcast the original LSTM gating techniques, i.e., input- and/or hidden-dependent gating plus bias term, to the new architectures.
        # Exponential activation functions can lead to large values that cause overflows. Therefore, we stabilize gates with an additional state m_t (Milakov & Gimelshein, 2018).
        # We show in Appendix A.2, that replacing ft by ft_hat and it by it_hat in the forward pass does neither change the output of the whole network nor the derivatives of the loss wrt the parameters.
        
        # Stabilizer state (NEW)
        m = torch.max(torch.log(f) + m_prev, torch.log(i))

        # Stabilized input gate (NEW)
        i_hat = torch.exp(torch.log(i) - m)

        # Stabilized forget gate (NEW)
        f_hat = torch.exp(torch.log(f) + m_prev - m)

        # New cell state (MODIFIED)
        c = f_hat * c_prev + i_hat * z

        # Normalizer state (NEW + MODIFIED)
        n = f_hat * n_prev + i_hat

        # New hidden state
        h = o * c / n

        return h, c, n, m

class sLSTM(nn.Module):
    """
    The main sLSTM network.
    """
    def __init__(self, config: sLSTMConfig):
        super().__init__()
        self.config = config
        self.lstm_block = sLSTMBlock(config)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        
        # Initialization
        h = torch.zeros(batch_size, self.config.hidden_size).to(device) # Hidden state
        c = torch.zeros(batch_size, self.config.hidden_size).to(device) # Cell state
        n = torch.ones(batch_size).to(device) # Normalizer state
        m = torch.zeros(batch_size).to(device) # Stabilizer state

        # Loop through the sequence
        for t in range(seq_len):
            h, c, n, m = self.lstm_block(input_seq[:, t, :], h, c, n, m)

        # Output layer
        output = self.output_layer(h)
        
        return output