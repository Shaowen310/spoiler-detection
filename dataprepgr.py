'''
Data credits: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
'''
import collections
import itertools
import functools
import os
import gzip
import json
import pickle

import gdown
import nltk

import loggingutil

# %%
root = 'data_'
base_folder = 'goodreads-reviews-spoiler'
URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
filename = 'goodreads_reviews_spoiler.json.gz'

word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
n_most_common = 5000

logger = loggingutil.get_logger('dataprepgr')

# Download
file_dir = os.path.join(root, base_folder)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
file_path = os.path.join(file_dir, filename)
if not os.path.exists(file_path):
    gdown.download(URL, output=file_path)


# Load
def get_record_gen(limit=None):
    count = 0
    with gzip.open(file_path) as fin:
        for l in itertools.islice(fin, limit):
            d = json.loads(l)
            count += 1
            if not (count % 10000):
                logger.debug('Processed {} records'.format(count))
            yield d


def load_records(limit=None):
    return list(get_record_gen(limit))


# %%
# Reusable
def get_sent_words(sent):
    # tokenize
    words = word_tokenizer.tokenize(sent)
    # transform & filter
    # => lower? stop words? punctuation? HTTP? digits? misspelled?
    return words


def get_label_sent_words(label_sents):
    return [(lb_s[0], get_sent_words(lb_s[1])) for lb_s in label_sents]


def get_word_dict(wc, n_most_common):
    idx2word = ['<pad>', '<unk>']
    idx2word.extend([word for (word, _) in wc.most_common(n_most_common)])
    word2idx = {w: i for i, w in enumerate(idx2word)}
    logger.info("Vocabulary size = {}".format(len(word2idx)))
    return word2idx, idx2word


def get_doc_label_sent_encodes(label_sents, word2idx):
    return [(lb_sw[0], [word2idx[w if w in word2idx else '<unk>'] for w in lb_sw[1]])
            for lb_sw in get_label_sent_words(label_sents)]


# %%
def build_vocab_meta(records, n_most_common):
    # word frequency
    wc_artwork = collections.defaultdict(lambda: collections.Counter())
    for record in records:
        artwork_id = record['book_id']
        for (_, words) in get_label_sent_words(record['review_sentences']):
            wc_artwork[artwork_id].update(words)
    wc = functools.reduce((lambda x, y: x + y), wc_artwork.values(), collections.Counter())

    # vocab
    return get_word_dict(wc, n_most_common)


def process(records, word2idx):
    doc_encode = [get_doc_label_sent_encodes(record['review_sentences'], word2idx) for record in records]
    return doc_encode


def save(obj, filename):
    with open(os.path.join(file_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(obj, f)


# %%
(word2idx, idx2word) = build_vocab_meta(get_record_gen(), n_most_common)
doc_encode = process(get_record_gen(), word2idx)
obj = {'doc_label_sent_encodes': doc_encode, 'idx2word': idx2word}
save(obj, 'mappings')

# %%
