import torch
import torch.nn as nn
from sLSTM import sLSTMBlock, sLSTMConfig

class sLSTM(nn.Module):
    """
    The main sLSTM network along with bidirectional functionality.
    """
    def __init__(self, config: sLSTMConfig):
        super().__init__()
        self.config = config

        self.lstm_blocks_forward = nn.ModuleList([sLSTMBlock(config) for _ in range(config.num_layers)])
        if config.bidirectional:
            self.lstm_blocks_backward = nn.ModuleList([sLSTMBlock(config) for _ in range(config.num_layers)])
        
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_seq):
        
        device = input_seq.device

        batch_size, seq_len, _ = input_seq.size()
        
        # Initialization
        h_forward = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)] # Hidden state
        c_forward = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)] # Cell state
        n_forward = [torch.ones(batch_size).to(device) for _ in range(self.config.num_layers)] # Normalizer state
        m_forward = [torch.zeros(batch_size).to(device) for _ in range(self.config.num_layers)] # Stabilizer state

        if self.config.bidirectional:
            h_backward = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)]
            c_backward = [torch.zeros(batch_size, self.config.hidden_size).to(device) for _ in range(self.config.num_layers)]
            n_backward = [torch.ones(batch_size).to(device) for _ in range(self.config.num_layers)]
            m_backward = [torch.zeros(batch_size).to(device) for _ in range(self.config.num_layers)]
        

        # Loop through the sequence
        for t in range(seq_len):
            x = input_seq[:, t, :]
            for layer in range(self.config.num_layers):
                h_forward[layer], c_forward[layer], n_forward[layer], m_forward[layer] = self.lstm_blocks_forward[layer](x, h_forward[layer], c_forward[layer], n_forward[layer], m_forward[layer])
                x = h_forward[layer] # Output of the current layer is the input to the next
        
        if self.config.bidirectional:
            for t in reversed(range(seq_len)):
                x = input_seq[:, t, :]
                for layer in range(self.config.num_layers):
                    h_backward[layer], c_backward[layer], n_backward[layer], m_backward[layer] = self.lstm_blocks_backward[layer](x, h_backward[layer], c_backward[layer], n_backward[layer], m_backward[layer])
                    x = h_backward[layer]
            # concatenate the forward and backward hidden states
            h = [torch.cat((h_forward[layer], h_backward[layer]), dim=-1) for layer in range(self.config.num_layers)]
        else:
            h = h_forward
        # Output layer (final hidden state)
        output = self.output_layer(h[-1])
        
        return output