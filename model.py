import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttentionLayer(nn.Module):
    def __init__(self, in_dim, att_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, att_dim)
        self.v = torch.tensor(att_dim, requires_grad=True)

    def forward(self, x):
        '''
        Input size: (seq_len, batch_size, in_dim)
        Output size: (batch_size, in_dim)
        '''
        mu = torch.tanh(self.linear(x))
        v_mu = torch.sum(mu * self.v, dim=2)
        att_w = F.softmax(v_mu, dim=0)
        return torch.sum(att_w.unsqueeze(2) * x, dim=0)


class SpoilerNet(nn.Module):
    def __init__(self, cell_dim, att_dim, vocab_size, emb_size, dropout_rate, pretrained_emb):
        super().__init__()

        self.emb_layer = nn.Embedding(vocab_size, emb_size, 0)
        self.word_encoder = nn.GRU(emb_size, cell_dim, bidirectional=True)
        self.word_att = WordAttentionLayer(2 * cell_dim, att_dim)
        self.sent_encoder = nn.GRU(2 * cell_dim, cell_dim, bidirectional=True)

    def _init_embedding(self):
        pass

    def forward(self, x):
        '''
        x size: (sent_seq_len, word_seq_len, batch, word_idx)
        '''
        batch_size = x.size(2)

        emb = self.emb_layer(x)

        sent_tensors = []
        for word_seq in emb:
            word_h0 = torch.zeros(2, batch_size, self.cell_dim)
            # word_enc size (seq_len, batch, num_directions * cell_dim)
            word_enc, _ = self.word_encoder(word_seq, word_h0)

            # word_with_att size (batch, num_directions * cell_dim)
            word_with_att = self.word_att(word_enc)
            sent_tensors.append(word_with_att)

        # sents size (sent_seq_len, batch, num_directions * cell_dim)
        sents = torch.cat(sent_tensors)

        sent_h0 = torch.zeros(2, batch_size, self.cell_dim)

        sent_enc, _ = self.sent_encoder(sents, sent_h0)

        return sent_enc
