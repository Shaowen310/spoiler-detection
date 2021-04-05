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

    def __init__(self, doc_label_sents, itow, max_n_words, max_n_sents):
        super().__init__()

        self.max_n_words = max_n_words
        self.max_n_sents = max_n_sents

        self.itow = itow
        self.wtoi = {w: i for i, w in enumerate(self.itow)}

        docs, labels, doc_lens, doc_sent_lens = self.pad(doc_label_sents)
        self.docs = torch.from_numpy(docs)
        self.labels = torch.from_numpy(labels)
        self.doc_lens = torch.from_numpy(doc_lens)
        self.doc_sent_lens = list(map(torch.from_numpy, doc_sent_lens))

    def pad(self, doc_label_sents, pad_idx=0):
        docs, labels, doc_lens, doc_sent_lens = [], [], [], []
        for label_sent_encodes in doc_label_sents:
            doc = np.full((self.max_n_sents, self.max_n_words), pad_idx, dtype=np.long)
            sent_labels = []
            sent_lens = []
            for i, (label, sent) in enumerate(itertools.islice(label_sent_encodes,
                                                               self.max_n_sents)):
                sent_len = min((self.max_n_words, len(sent)))
                doc[i, :sent_len] = sent[:sent_len]
                sent_labels.append(label)
                sent_lens.append(sent_len)
            doc_len = min((self.max_n_sents, len(label_sent_encodes)))
            sent_labels = np.pad(np.array(sent_labels, dtype=np.long),
                                 ((0, self.max_n_sents - doc_len)))
            docs.append(doc)
            labels.append(sent_labels)
            doc_sent_lens.append(np.array(sent_lens))
        docs = np.array(docs)
        labels = np.array(labels)
        doc_lens = np.array(list(map(len, doc_sent_lens)))
        return docs, labels, doc_lens, doc_sent_lens

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx], self.doc_lens[idx]

    def __len__(self):
        return len(self.docs)
