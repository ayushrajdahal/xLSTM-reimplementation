import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


@dataclass
class sLSTMConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.0
    bidirectional: bool = False

    
class sLSTMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # input weight vectors between inputs xt and cell input, input gate, forget gate, and output gate, respectively.
        self.Wzx = nn.Linear(config.input_size, config.hidden_size)
        self.Wix = nn.Linear(config.input_size, config.hidden_size)
        self.Wfx = nn.Linear(config.input_size, config.hidden_size)
        self.Wox = nn.Linear(config.input_size, config.hidden_size)

        # recurrent weights between hidden state ht−1 and cell input, input gate, forget gate, and output gate, respectively.
        self.Rzh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rih = nn.Linear(config.hidden_size, config.hidden_size)
        self.Rfh = nn.Linear(config.hidden_size, config.hidden_size)
        self.Roh = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        z = torch.tanh(self.Wzx(x) + self.Rzh(h_prev))      # cell input
        i = torch.exp(self.Wix(x) + self.Rih(h_prev))       # input gate (MODIFIED)
        f = torch.exp(self.Wfx(x) + self.Rfh(h_prev))       # forget gate (MODIFIED): can be exp OR sigmoid
        o = torch.sigmoid(self.Wox(x) + self.Roh(h_prev))   # output gate

        # Excerpt from the paper:
        # We broadcast the original LSTM gating techniques, i.e., input- and/or hidden-dependent gating plus bias term, to the new architectures.
        # Exponential activation functions can lead to large values that cause overflows. Therefore, we stabilize gates with an additional state m_t (Milakov & Gimelshein, 2018).
        # We show in Appendix A.2, that replacing ft by ft_hat and it by it_hat in the forward pass does neither change the output of the whole network nor the derivatives of the loss wrt the parameters.
        
        m = torch.max(torch.log(f) + m_prev, torch.log(i)) # stabilizer state (NEW)

        i_hat = torch.exp(torch.log(i) - m)                 # stabilized input gate (NEW)
        f_hat = torch.exp(torch.log(f) + m_prev - m)        # stabilized forget gate (NEW)

        # f and i are replaced with their stabilized equivalents below:
        c = f_hat * c_prev + i_hat * z  # new cell state (MODIFIED)
        n = f_hat * n_prev + i_hat      # normalizer state (NEW + MODIFIED)

        h = o * c / n           # new hidden state

        return h, c, n, m

        
class sLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm_block = sLSTMBlock(config)
        self.output_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        
        # initialization
        h = torch.zeros(batch_size, self.config.hidden_size).to(device) # hidden state
        c = torch.zeros(batch_size, self.config.hidden_size).to(device) # cell state
        n = torch.ones(batch_size).to(device) # normalizer state
        m = torch.zeros(batch_size).to(device) # stabilizer state

        # loop through the sequence
        for t in range(seq_len):
            h, c, n, m = self.lstm_block(input_seq[:, t, :], h, c, n, m)

        # output layer
        output = self.output_layer(h)
        
        return output