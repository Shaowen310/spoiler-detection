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
import re
import string
import math

import gdown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.preprocessing import StandardScaler

import loggingutil

# %%
root = 'data_'
base_folder = 'goodreads-reviews-spoiler'
URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
filename = 'goodreads_reviews_spoiler.json.gz'

logger = loggingutil.get_logger('dataprepgr')

# %%
STOP_WORDS = set(stopwords.words('english'))
_porter = PorterStemmer()
_stdscale = StandardScaler()
# %%
# Download
file_dir = os.path.join(root, base_folder)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
file_path = os.path.join(file_dir, filename)
if not os.path.exists(file_path):
    gdown.download(URL, output=file_path)


# Load
def generate_records(limit=None, log_every=100000):
    count = 0
    with gzip.open(file_path) as fin:
        for l in itertools.islice(fin, limit):
            d = json.loads(l)
            count += 1
            if not (count % log_every):
                logger.debug('Processed {} records'.format(count))
            yield d


def load_records(limit=None):
    return list(generate_records(limit))


# %%
# Reusable
def remove_punc(token):
    return token.translate(str.maketrans('', '', string.punctuation))


def get_sent_words(sent):
    # case-folding
    sent = sent.lower()

    # replace http
    sent = re.sub(r'(http|https)://\S+', '^http', sent)

    # replace digits
    sent = re.sub(r'\d+', ' ^num ', sent)

    # tokenize
    words = word_tokenize(sent)

    # split '-'
    words = itertools.chain(*map(lambda w: w.split('-'), words))

    # remove punctuation
    words = map(lambda w: remove_punc(w) if w not in ['^http', '^num'] else w, words)

    # remove empty
    words = filter(lambda w: w, words)

    # remove stopwords
    words = filter(lambda w: w not in STOP_WORDS, words)

    # stemming
    # words = map(lambda w: _porter.stem(w), words)

    words = list(words)
    return words


def get_label_sent_words(label_sents):
    return [(lb_s[0], get_sent_words(lb_s[1])) for lb_s in label_sents]


def get_word_dict(wc, n_most_common):
    itow = ['<pad>', '<unk>']
    itow.extend([word for (word, _) in wc.most_common(n_most_common)])
    wtoi = {w: i for i, w in enumerate(itow)}
    logger.info("Vocabulary size = {}".format(len(wtoi)))
    return wtoi, itow


def get_doc_label_sent_encodes(label_sents, word2idx):
    return [(lb_sw[0], [word2idx[w if w in word2idx else '<unk>'] for w in lb_sw[1]])
            for lb_sw in get_label_sent_words(label_sents)]


# %%
def word_count(records):
    wc_review = []
    wc_artwork = collections.defaultdict(lambda: collections.Counter())
    wc = collections.Counter()
    for record in records:
        artwork_id = record['book_id']
        wc_review_ = collections.Counter()
        for (_, words) in get_label_sent_words(record['review_sentences']):
            wc_review_.update(words)
        wc_review.append(wc_review_)
        wc_artwork[artwork_id].update(wc_review_)
        wc.update(wc_review_)

    logger.info('# of books: {}'.format(len(wc_artwork)))

    # vocab
    return wc, wc_artwork, wc_review


def process(records, word2idx):
    doc_artwork, doc_encode = [], []
    for record in records:
        doc_artwork.append(record['book_id'])
        doc_label_sent_encodes = get_doc_label_sent_encodes(record['review_sentences'], word2idx)
        doc_encode.append(doc_label_sent_encodes)

    return doc_encode, doc_artwork


def prepare_invmap(doc_artwork, wc_review, wc_artwork):
    atod = collections.defaultdict(lambda: [])
    for d, a in enumerate(doc_artwork):
        atod[a].append(d)
    wtor = collections.defaultdict(lambda: [])
    for r, wc in enumerate(wc_review):
        for w in wc.keys():
            wtor[w].append(r)
    wtoa = collections.defaultdict(lambda: [])
    for a, wc in wc_artwork.items():
        for w in wc.keys():
            wtoa[w].append(a)
    return dict(atod), dict(wtor), dict(wtoa)


def df_idf(word, artwork_id, atod, wtor, wtoa):
    d_i = len(atod[artwork_id])
    d_wi = len(wtor[word])
    df = d_wi / d_i
    e = 1
    l_w = len(wtoa[word])
    l = len(atod.keys())
    iif = math.log((l + e) / (l_w + e))
    return df * iif


def process_df_idf(doc_encode, doc_artwork, itow, atod, wtor, wtoa, log_every=1000):
    all_df_idf = []
    for i in range(len(doc_encode)):
        doc_label_sent_encodes = doc_encode[i]
        artwork_id = doc_artwork[i]
        for lb_s in doc_label_sent_encodes:
            for w in lb_s[1]:
                word = itow[w]
                dfidf = 0
                if word != '<unk>':
                    dfidf = df_idf(word, artwork_id, atod, wtor, wtoa)
                all_df_idf.append(dfidf)
    all_df_idf = _stdscale.fit_transform(np.array(all_df_idf).reshape(-1,1)).ravel()

    doc_df_idf = []
    c = 0
    for doc_label_sent_encodes in doc_encode:
        doc = []
        for lb_s in doc_label_sent_encodes:
            sent=[]
            for w in lb_s[1]:
                sent.append(all_df_idf[c])
                c += 1
            doc.append(sent)
        doc_df_idf.append(doc)

    return doc_df_idf


# %%
limit = 20000
n_most_common = None

logger.info('# of reviews: {}'.format(limit))
logger.info('Getting word counts...')
wc, wc_artwork, wc_review = word_count(generate_records(limit))
logger.info('Building dictionaries...')
wtoi, itow = get_word_dict(wc, n_most_common)
logger.info('Encoding reviews...')
doc_encode, doc_artwork = process(generate_records(limit), wtoi)
logger.info('Calculating DF-IDF...')
atod, wtor, wtoa = prepare_invmap(doc_artwork, wc_review, wc_artwork)
doc_df_idf = process_df_idf(doc_encode, doc_artwork, itow, atod, wtor, wtoa)
logger.info('Saving...')
obj = {
    'doc_label_sents': doc_encode,
    'itow': itow,
    'wc': dict(wc),
    'wc_artwork': dict(wc_artwork),
    'doc_artwork': doc_artwork,
    'doc_df_idf': doc_df_idf
}
filename = 'mappings_{}_{}'.format('all' if limit is None else str(limit),
                                   'all' if n_most_common is None else str(n_most_common))
with open(os.path.join(file_dir, filename + '.pkl'), 'wb') as f:
    pickle.dump(obj, f)

# %%
