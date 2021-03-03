import collections
import itertools
import functools
import os
import gzip
import json
import pickle

import gdown
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

    def __init__(self, root='data_', download=True):
        super().__init__()

        self.root = root

        if download:
            self.download()

        self.max_n_words = 10
        self.max_n_sents = 10

        self.data = None

    def download(self):
        file_dir = os.path.join(self.root, self.base_folder)
        file_path = os.path.join(file_dir, self.filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            gdown.download(self.URL, output=file_path)

    def load_records(self, limit=None):
        return list(self.get_record_gen(limit))

    def get_record_gen(self, limit=None):
        file_path = os.path.join(self.root, self.base_folder, self.filename)
        count = 0
        with gzip.open(file_path) as fin:
            for l in fin:
                d = json.loads(l)
                count += 1

                yield d

                if (limit is not None) and (count >= limit):
                    break

    def get_sent_words(self, sent):
        # tokenize
        words = self.word_tokenizer.tokenize(sent)
        # transform & filter
        # => lower? stop words? punctuation? HTTP? digits? misspelled?
        return words

    @staticmethod
    def get_word_dict(wc, n_most_common):
        idx2word = ['<pad>']
        idx2word.extend([word for (word, _) in wc.most_common(n_most_common)])
        idx2word.append('<unk>')
        return Dictionary(idx2word)

    def get_label_sent_words(self, record):
        return [(lb_s[0], self.get_sent_words(lb_s[1])) for lb_s in record['review_sentences']]

    def get_word_count(self, records):
        wc_artwork = collections.defaultdict(lambda: collections.Counter())
        for record in records:
            artwork_id = record['book_id']
            for (_, words) in self.get_label_sent_words(record):
                wc_artwork[artwork_id].update(words)
        wc = functools.reduce((lambda x, y: x + y), wc_artwork.values())
        return wc, wc_artwork

    def get_doc_label_sent_encodes(self, record, word2idx):
        return [(lb_sw[0], [word2idx[w if w in word2idx else '<unk>'] for w in lb_sw[1]])
                for lb_sw in self.get_label_sent_words(record)]

    def get_doc_label_sent_encodes_gen(self, records, word2idx):
        return map(lambda record: self.get_doc_label_sent_encodes(record, word2idx), records)

    def encode(self, n_most_common, limit=None):
        record_gen = self.get_record_gen(limit)
        wc, _ = self.get_word_count(record_gen)
        word_dict = __class__.get_word_dict(wc, n_most_common)

        record_gen = self.get_record_gen(limit)
        doc_label_sent_encodes_gen = self.get_doc_label_sent_encodes_gen(record_gen, word_dict.word2idx)

        return doc_label_sent_encodes_gen, word_dict

    def pad(self, doc_label_sent_encodes, pad_idx=0):
        docs, labels, doc_lens = [], [], []
        for label_sent_encodes in doc_label_sent_encodes:
            doc = np.full((self.max_n_sents, self.max_n_words), pad_idx, dtype=np.long)
            sent_labels = []
            for i, (label, sent) in enumerate(itertools.islice(label_sent_encodes, self.max_n_sents)):
                sent_len = min((self.max_n_words, len(sent)))
                doc[i, :sent_len] = sent[:sent_len]
                sent_labels.append(label)
            doc_len = min((self.max_n_sents, len(label_sent_encodes)))
            sent_labels = np.pad(sent_labels, ((0, self.max_n_sents - doc_len)), dtype=np.float)
            docs.append(doc)
            labels.append(sent_labels)
            doc_lens.append(doc_len)
        docs = np.vstack(docs)
        labels = np.vstack(labels)
        doc_lens = np.array(doc_lens)
        return docs, labels, doc_lens

    def __getitem__(self):
        pass

    def __len__(self):
        pass
