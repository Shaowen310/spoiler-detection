import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')

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

class CoWordAttentionLayer(nn.Module):
    def __init__(self, ab_in_dim, sent_in_dim, att_dim):
        super().__init__()

        self.linear_ab = nn.Linear(ab_in_dim, att_dim)
        self.linear_sent = nn.Linear(sent_in_dim, att_dim)
        # self.v = torch.tensor(att_dim, dtype=torch.float32, requires_grad=True)

    def forward(self, x, ab):
        '''
        Input size x: (seq_len, batch_size, in_dim)
        Input size ab: (batch_size, max_n_key, in_dim)
        Output size: (batch_size, in_dim)
        '''
        mu = torch.tanh(self.linear_sent(x))
        mv = torch.tanh(self.linear_ab(ab))

        mu = mu.permute(1, 0, 2)
        mv = mv.permute(0, 2, 1)
        att_w = mu.matmul(mv) # batch, seq_len, max_n_key

        # att_w = torch.max(att_w, 2)[0]
        att_w = torch.sum(att_w, dim=2)

        att_w = F.softmax(att_w, dim=1)
        output = att_w.unsqueeze(1).matmul(x.permute(1,0,2))
        return output.squeeze()


class SpoilerNet(nn.Module):
    def __init__(self,
                 cell_dim,
                 att_dim,
                 vocab_size,
                 emb_size,
                 attent_type,
                 use_idf,
                 char_emb_size,
                 char_cell_dim,
                 use_char,
                 char_vocab_size,
                 dropout_rate=0.5,
                 pretrained_emb=None):
        super().__init__()

        self.cell_dim = cell_dim
        self.att_dim = att_dim
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.use_idf = use_idf
        self.use_char = use_char
        self.attent_type = attent_type

        self.emb_layer = nn.Embedding(vocab_size, emb_size, 0)
        # self.sentlv_word_emb_size = emb_size + 1 if use_idf else emb_size
        if use_idf and use_char:
            self.sentlv_word_emb_size = emb_size + 2*char_cell_dim + 1
        elif use_char:
            self.sentlv_word_emb_size = emb_size + 2*char_cell_dim
        elif use_idf:
            self.sentlv_word_emb_size = emb_size + 1
        else:
            self.sentlv_word_emb_size = emb_size

        print("sentlv_word_emb_size,", self.sentlv_word_emb_size)

        self.word_encoder = nn.GRU(self.sentlv_word_emb_size, cell_dim, bidirectional=True)

        if attent_type == "coAtt":
            self.word_att = CoWordAttentionLayer(emb_size, 2 * cell_dim, att_dim)
        else:
            self.word_att = WordAttentionLayer(2 * cell_dim, att_dim)

        self.sent_encoder = nn.GRU(2 * cell_dim, cell_dim, bidirectional=True)
        self.out_linear = nn.Linear(2 * cell_dim, 1)

        self.char_emb_layer = nn.Embedding(char_vocab_size, char_emb_size, 0)
        self.char_lstm = nn.LSTM(char_emb_size, char_cell_dim, num_layers=1, bidirectional=True)
        self.char_emb_size = char_emb_size
        self.char_cell_dim = char_cell_dim

        self.drop = nn.Dropout(dropout_rate)

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.cell_dim)

    def forward(self, x, word_h0, sent_h0, x_df_idf=None, chars=None, doc_ab=None):
        '''
        x size: (batch, sent_seq_len, word_seq_len)
        chars: (batch, sent_seq_len, word_seq_len, max_n_chars)
        doc_ab: (batch, max_n_key)
        '''
        doc_ab = self.emb_layer(doc_ab) 

        x = x.permute(1, 0, 2)
        # (sent_seq_len, batch, word_seq_len)
        if self.use_idf:
            x_df_idf = x_df_idf.permute(1, 0, 2)
        if self.use_char:
            chars = chars.permute(1, 0, 2, 3)

        sentlv_sent_enc_list = []
        for i in range(len(x)):
            sent = x[i]
            sent = sent.permute(1, 0)
            # (word_seq_len, batch)

            if self.use_char:
                sent_chars = chars[i].permute(1, 0, 2)
                # (word_seq_len, batch, max_n_chars)
                word_seq_len = sent_chars.size()[0]
                sent_out = []

                for j in range(word_seq_len):
                    batch_chars = sent_chars[j].permute(1, 0)
                    chars_embeds = self.char_emb_layer(batch_chars)
                    # (max_n_chars, batch, emb)

                    # char_output, (hn, cn) = self.char_lstm(chars_embeds)
                    # # (max_n_chars, batch, 2*hidden_size)
                    # char_output = char_output[-1]

                    chars_embeds = chars_embeds.permute(1, 0, 2)
                    char_output = torch.mean(chars_embeds, dim=1)

                    sent_out.append(char_output)
                char_sent_out = torch.stack(sent_out, dim=0)
                # (word_seq_len, batch, emb)

            if self.use_idf:
                sent_df_idf = x_df_idf[i]
                sent_df_idf = sent_df_idf.permute(1, 0).unsqueeze(2)

            word_emb = self.emb_layer(sent)
            if self.use_idf:
                word_emb = torch.cat((word_emb, sent_df_idf), 2)
            if self.use_char:
                word_emb = torch.cat((word_emb, char_sent_out), 2)

            sentlv_word_encs, word_h0 = self.word_encoder(word_emb, word_h0)
            # (word_seq_len, batch, num_directions * hidden_size)

            sentlv_word_encs = self.drop(sentlv_word_encs)

            if self.attent_type == "coAtt":
                sentlv_sent_enc = self.word_att(sentlv_word_encs, doc_ab)
            else:
                sentlv_sent_enc = self.word_att(sentlv_word_encs)
            # (batch, num_directions * hidden_size)

            sentlv_sent_enc_list.append(sentlv_sent_enc)

        sentlv_sent_encs = torch.stack(sentlv_sent_enc_list)
        # (sent_seq_len, batch, num_directions * hidden_size)

        doclv_sent_enc, sent_h0 = self.sent_encoder(sentlv_sent_encs, sent_h0)
        # (sent_seq_len, batch, num_directions * hidden_size)

        doclv_sent_enc = self.drop(doclv_sent_enc)

        doclv_sent_enc = doclv_sent_enc.permute(1, 0, 2)
        # (batch, sent_seq_len, num_directions * hidden_size)

        doclv_sent_enc = self.out_linear(doclv_sent_enc)

        doclv_sent_enc = doclv_sent_enc.view(-1)
        # (batch * sent_seq_len)

        return doclv_sent_enc, word_h0, sent_h0
