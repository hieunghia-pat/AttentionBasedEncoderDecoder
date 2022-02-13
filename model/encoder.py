import torch 
from torch import nn 

class Encoder(nn.Module):
    def __init__(self, image_w, d_model):
        super().__init__()

        in_dim = (image_w // 32)*d_model # aflter passed through any CNN structures, the width is reduced by 32
        self.rnn = nn.GRU(in_dim, d_model, bidirectional = True)

        self.proj_out = nn.Linear(2*d_model, d_model)
        self.proj_hidden = nn.Linear(2*d_model, d_model)
        
    def forward(self, src):
        """
            src: (w, bs, in_dim)
            outputs: (w, bs, d_model)
            hidden_states: (1, bs, d_model)
        """
        
        outputs, hidden_states = self.rnn(src)
        outputs = self.proj_out(outputs)
        hidden_states = torch.cat([hidden_states[-2, :, :], hidden_states[-1, :, :]], dim=-1).unsqueeze(0) # (1, bs, 2*d_model)
        hidden_states = torch.tanh(self.proj_hidden(hidden_states)) # (1, bs, d_model)
        
        return outputs, hidden_states