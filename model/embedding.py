from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, padding_idx, weights=None):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embedding_dim, d_model)

        if weights:
            self.embedding.from_pretrained(weights, padding_idx=padding_idx)

    def forward(self, input):
        embedded = self.embedding(input)

        return self.fc(embedded)