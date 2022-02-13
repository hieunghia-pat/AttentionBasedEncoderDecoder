from torch import nn

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size, dropout=0.5):
        super(Generator, self).__init__()

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, outputs):
        '''
            outputs: (bs, w, d_model)
        '''
        outputs = self.dropout(self.fc(outputs))

        return outputs