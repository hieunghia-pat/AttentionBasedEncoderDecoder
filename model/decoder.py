import torch 
from torch import nn
from torch.nn import functional as F
from model.embedding import Embedding

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.attn = nn.Linear(d_model + d_model + d_model, d_model)
        self.w_v = nn.Linear(d_model, 1) # this attention is per-timestep calculation
        
    def forward(self, hidden_states, cell_states, encoded_features):
        """
            hidden_states: (1, bs, d_model)
            cell_states: (1, bs, d_model)
            encoded_features: (w, bs, d_model)
            outputs: (bs, 1, w)
        """

        w = encoded_features.shape[0]
        hidden_states = hidden_states.repeat(w, 1, 1) # (w, bs, d_model)
        cell_states = cell_states.repeat(w, 1, 1) # (w, bs, d_model)

        correlation = torch.tanh(self.attn(torch.cat((hidden_states, cell_states, encoded_features), dim=-1))) # (w, bs, d_model)
        attn_weights = self.w_v(correlation).permute(1, 2, 0) # (bs, 1, w)
        
        return F.softmax(attn_weights, dim=-1)

class Decoder(nn.Module):
    def __init__(self, vocab, embedding_dim, d_model):
        super().__init__()

        self.attention = AttentionLayer(d_model)
        self.embedding = Embedding(len(vocab), embedding_dim, d_model, vocab.padding_idx, vocab.vectors)
        self.rnn = nn.LSTM(2*d_model, d_model, batch_first=True)
        
    def forward(self, inputs, hidden_states, cell_states, encoder_outputs):
        '''
            inputs: (bs, 1, 1)
            encoder_outputs: (w, bs, d_model)
            outputs: (bs, 1, d_model)
            hiddent_states: (1, bs, d_model)
            cell_states: (1, bs, d_model)
        '''
        
        embedded = self.embedding(inputs) # (bs, 1, d_model)
        attn_weights = self.attention(hidden_states, cell_states, encoder_outputs) # (bs, 1, w)
        contexts = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2)) # (bs, 1, d_model)

        rnn_input = torch.cat((embedded, contexts), dim=-1) # (bs, 1, 2*d_model)
        
        outputs, (hidden_states, cell_states) = self.rnn(rnn_input, (hidden_states, cell_states))
        
        return outputs, (hidden_states, cell_states)