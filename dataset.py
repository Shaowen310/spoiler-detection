import collections
import functools
import os
import gzip
import json
import pickle
import gdown
import nltk
from nltk import tokenize

from torch.utils.data import Dataset


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

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def save(self, fp):
        pickle.dump(self.idx2word, open(fp, 'wb'))

    def load(self, fp):
        self.idx2word = pickle.load(open(fp, 'rb'))
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}


class GoodreadsReviewsSpoilerDataset(Dataset):
    '''
    Credits: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
    '''
    base_folder = 'goodreads-reviews-spoiler'
    URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
    filename = 'goodreads_reviews_spoiler.json.gz'
    word_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def __init__(self, root='.data', download=True):
        super().__init__()

        self.root = root

        if download:
            self.download()

        file_path = os.path.join(self.root, self.base_folder, self.filename)

    def download(self):
        file_dir = os.path.join(self.root, self.base_folder)
        file_path = os.path.join(file_dir, self.filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            gdown.download(self.URL, output=file_path)

    def load_json(self, limit=None):
        return list(self.get_json_iter(limit))

    def get_json_iter(self, limit=None):
        file_path = os.path.join(self.root, self.base_folder, self.filename)
        count = 0
        with gzip.open(file_path) as fin:
            for l in fin:
                d = json.loads(l)
                count += 1

                yield d

                if (limit is not None) and (count > limit):
                    break

    def get_label_word_token_iter(self, record):
        review_sents = record['review_sentences']
        for (label, sent) in review_sents:
            words = self.word_tokenizer.tokenize(sent)
            yield (label, words)

    def get_word_count_by_document(self, json_iter):
        wc_doc = collections.defaultdict(lambda: collections.Counter())
        for record in json_iter:
            doc_id = record['book_id']
            label_word_token_iter = self.get_label_word_token_iter(record)
            for label_word_token in label_word_token_iter:
                wc_doc[doc_id].update(label_word_token[1])
        wc = functools.reduce((lambda x, y: x + y), wc_doc.values())
        return wc_doc, wc
