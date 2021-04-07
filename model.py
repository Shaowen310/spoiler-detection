import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttentionLayer(nn.Module):
    def __init__(self, in_dim, att_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, att_dim)
        self.v = torch.tensor(att_dim, dtype=torch.float32, requires_grad=True)

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
    def __init__(self,
                 cell_dim,
                 att_dim,
                 vocab_size,
                 emb_size,
                 dropout_rate=0.5,
                 pretrained_emb=None):
        super().__init__()

        self.cell_dim = cell_dim
        self.att_dim = att_dim
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.emb_layer = nn.Embedding(vocab_size, emb_size, 0)
        self.word_encoder = nn.GRU(emb_size + 1, cell_dim, bidirectional=True)
        self.word_att = WordAttentionLayer(2 * cell_dim, att_dim)
        self.sent_encoder = nn.GRU(2 * cell_dim, cell_dim, bidirectional=True)
        self.out_linear = nn.Linear(2 * cell_dim, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.cell_dim)

    def forward(self, x, word_h0, sent_h0, x_df_idf):
        '''
        x size: (batch, sent_seq_len, word_seq_len)
        '''
        x = x.permute(1, 0, 2)
        # (sent_seq_len, batch, word_seq_len)
        x_df_idf = x_df_idf.permute(1, 0, 2)

        sentlv_sent_enc_list = []
        for i in range(len(x)):
            sent = x[i]
            sent_df_idf = x_df_idf[i]

            sent = sent.permute(1, 0)
            # (word_seq_len, batch)
            sent_df_idf = sent_df_idf.permute(1, 0).unsqueeze(2)

            word_emb = self.emb_layer(sent)
            word_emb_idf = torch.cat((word_emb, sent_df_idf), 2)
            sentlv_word_encs, word_h0 = self.word_encoder(word_emb_idf, word_h0)
            # (word_seq_len, batch, num_directions * hidden_size)

            sentlv_sent_enc = self.word_att(sentlv_word_encs)
            # (batch, num_directions * hidden_size)

            sentlv_sent_enc_list.append(sentlv_sent_enc)

        sentlv_sent_encs = torch.stack(sentlv_sent_enc_list)
        # (sent_seq_len, batch, num_directions * hidden_size)

        doclv_sent_enc, sent_h0 = self.sent_encoder(sentlv_sent_encs, sent_h0)
        # (sent_seq_len, batch, num_directions * hidden_size)

        doclv_sent_enc = doclv_sent_enc.permute(1, 0, 2)
        # (batch, sent_seq_len, num_directions * hidden_size)

        doclv_sent_enc = self.out_linear(doclv_sent_enc)

        doclv_sent_enc = doclv_sent_enc.view(-1)
        # (batch * sent_seq_len)

        return doclv_sent_enc, word_h0, sent_h0
