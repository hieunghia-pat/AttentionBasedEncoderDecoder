import torch
from torch import nn
from model.feature_extractor import FeatureExtractor
from model.encoder import Encoder
from model.decoder import Decoder
from model.generator import Generator
from torch.nn import functional as F

class AttentionBasedEncoderDecoder(nn.Module):
    def __init__(self, extractor, vocab, image_w, d_model, embedding_dim, dropout=0.1):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(extractor, d_model)
        self.encoder = Encoder(image_w, d_model)
        self.decoder = Decoder(vocab, embedding_dim, d_model)
        self.generator = Generator(d_model, len(vocab), dropout)

    def forward(self, images, targets):
        '''
            images: (bs, c, h, w)
            targets: (bs, w)
        '''
        features = self.feature_extractor(images)
        encoded_features, hidden_states = self.encoder(features)
        # applying the teacher-forcing mechanisms
        outputs = torch.tensor([], dtype=images.dtype, device=images.device)
        cell_states = torch.zeros_like(hidden_states)
        for t in range(targets.shape[1]):
            input = targets[:, t].unsqueeze(-1)
            output, (hidden_states, cell_states) = self.decoder(input, hidden_states, cell_states, encoded_features)
            output = self.generator(output)
            outputs = torch.cat([outputs, output], dim=1) # append new predicted word to the sequence

        return F.log_softmax(outputs, dim=-1) # (bs, w, vocab_size)

    def get_predictions(self, images, vocab, max_len):
        self.eval()
        
        features = self.feature_extractor(images)
        encoded_features, hidden_states = self.encoder(features)
        bs = images.shape[0]
        # applying the teacher-forcing mechanisms
        targets = torch.tensor([vocab.sos_idx]*bs, dtype=images.dtype, device=images.device).reshape(bs, 1)
        cell_states = torch.zeros_like(hidden_states)
        for t in range(max_len):
            input = targets[:, -1].unsqueeze(-1)
            output, (hidden_states, cell_states) = self.decoder(input, hidden_states, cell_states, encoded_features)
            output = self.generator(output)
            # append new predicted word to the sequence
            target = output.argmax(dim=-1) # (bs, 1)
            targets = torch.cat([targets, target], dim=-1)

        self.train()

        return targets # (bs, w)