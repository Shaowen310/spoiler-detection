import itertools
import pickle

import nltk
import numpy as np
import torch


class Dictionary:
    def __init__(self, idx2word=None):
        if idx2word is None:
            self.word2idx = {}
            self.idx2word = []
        else:
            self.idx2word = idx2word
            self.word2idx = {word: idx for idx, word in enumerate(idx2word)}

    def __len__(self):
        return len(self.idx2word)

    def save(self, fp):
        pickle.dump(self.idx2word, open(fp, 'wb'))

    def load(self, fp):
        self.idx2word = pickle.load(open(fp, 'rb'))
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}


class GoodreadsReviewsSpoilerDataset(torch.utils.data.Dataset):
    '''
    Credits: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
    '''
    base_folder = 'goodreads-reviews-spoiler'
    URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
    filename = 'goodreads_reviews_spoiler.json.gz'
    word_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def __init__(self, doc_label_sents, doc_df_idf, doc_keys, itow, max_n_words, max_n_sents, ctoi):
        super().__init__()

        self.max_n_words = max_n_words
        self.max_n_sents = max_n_sents

        self.itow = itow
        self.wtoi = {w: i for i, w in enumerate(self.itow)}


        char_length = [len(w) for w in self.wtoi]
        char_length = sorted(char_length)
        self.max_n_chars = char_length[int(0.99*len(char_length))]

        key_length = [len(d) for d in doc_keys]
        key_length = sorted(key_length)
        self.max_n_keys = key_length[int(0.9*len(key_length))]


        docs, labels, doc_len_masks, doc_sent_lens, doc_chars, doc_abs = self.pad(doc_label_sents, doc_keys, itow=itow, ctoi=ctoi)
        self.docs = torch.from_numpy(docs)
        self.labels = torch.from_numpy(labels)
        self.doc_len_masks = torch.from_numpy(doc_len_masks)
        self.doc_sent_lens = doc_sent_lens

        self.doc_chars = torch.from_numpy(doc_chars)
        self.doc_abs = torch.from_numpy(doc_abs)

        self.doc_dfidf = self.paddfidf(doc_df_idf)

    def pad(self, doc_label_sents, doc_keys, pad_idx=0, itow=None, ctoi=None):
        docs, labels, doc_lens, doc_sent_lens, doc_chars, doc_abs = [], [], [], [], [], []
        for k, label_sent_encodes in enumerate(doc_label_sents):

            # refers
            _abs = doc_keys[k]
            doc_ab = np.full((self.max_n_keys), pad_idx, dtype=np.long)
            ab_len = min((self.max_n_keys, len(_abs)))
            doc_ab[:ab_len] = _abs[:ab_len]
            doc_abs.append(doc_ab)

            # chars
            doc_char = np.full((self.max_n_sents, self.max_n_words, self.max_n_chars), pad_idx, dtype=np.long)
            doc = np.full((self.max_n_sents, self.max_n_words), pad_idx, dtype=np.long)
            sent_labels = []
            sent_lens = []
            for i, (label, sent) in enumerate(itertools.islice(label_sent_encodes,
                                                               self.max_n_sents)):

                sent_len = min((self.max_n_words, len(sent)))
                doc[i, :sent_len] = sent[:sent_len]
                sent_labels.append(label)
                sent_lens.append(sent_len)

                for j, wid in enumerate(sent[:sent_len]):
                    word = itow[wid]
                    w_c_list = []
                    for char in word:
                        w_c_list.append(ctoi[char])

                    word_len = min((self.max_n_chars, len(w_c_list)))
                    doc_char[i, j, :word_len] = np.array(w_c_list)[:word_len]

            doc_len = min((self.max_n_sents, len(label_sent_encodes)))
            sent_labels = np.pad(np.array(sent_labels, dtype=np.long),
                                 ((0, self.max_n_sents - doc_len)))
            docs.append(doc)
            labels.append(sent_labels)
            doc_sent_lens.append(np.array(sent_lens))
            doc_chars.append(doc_char)


        docs = np.array(docs)
        labels = np.array(labels)
        doc_lens = np.array(list(map(len, doc_sent_lens)))
        doc_len_masks = np.zeros((len(doc_lens), self.max_n_sents), dtype=np.float32)
        for idx, doc_len in enumerate(doc_lens):
            doc_len_masks[idx, :doc_len] = 1
        doc_chars = np.array(doc_chars)
        doc_abs = np.array(doc_abs)
        return docs, labels, doc_len_masks, doc_sent_lens, doc_chars, doc_abs

    def paddfidf(self, doc_df_idf):
        docs = []
        for doc_df_idf_ in doc_df_idf:
            doc = np.full((self.max_n_sents, self.max_n_words), 0., dtype=np.float32)
            for i, sent in enumerate(itertools.islice(doc_df_idf_, self.max_n_sents)):
                sent_len = min((self.max_n_words, len(sent)))
                doc[i, :sent_len] = sent[:sent_len]
            docs.append(doc)
        docs = np.array(docs)
        return docs

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx], self.doc_len_masks[idx], self.doc_dfidf[idx], self.doc_chars[idx], self.doc_abs[idx]

    def __len__(self):
        return len(self.docs)
