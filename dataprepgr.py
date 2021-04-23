'''
Data credits: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
'''
import collections
import itertools
import os
import gzip
import json
import pickle
import re
import string
import math

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

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

# keywords
keyword_path = "data_/book_id_keywords.json"
keywords = json.load(open(keyword_path, "r"))


# Load
def generate_records(limit=None, log_every=100000):
    count = 0
    with gzip.open(file_path) as fin:
        for l in itertools.islice(fin, limit):
            d = json.loads(l)

            # process book abstract
            book_id = d['book_id']
            if book_id not in keywords:
                continue

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


def get_word_dict(wc, n_most_common, freq=1):
    itow = ['<pad>', '<unk>']
    wcs = wc.most_common(n_most_common)
    wcs = filter(lambda wc: wc[1] >= freq, wcs)
    words = map(lambda wc: wc[0], wcs)
    itow.extend(words)
    wtoi = {w: i for i, w in enumerate(itow)}
    logger.info("Vocabulary size = {}".format(len(wtoi)))
    return wtoi, itow


def get_doc_label_sent_encodes(label_sents, word2idx):
    return [(lb_sw[0], [word2idx[w if w in word2idx else '<unk>'] for w in lb_sw[1]], lb_sw[1])
            for lb_sw in get_label_sent_words(label_sents)]


# %%
def word_count(records):
    wc_doc = []
    wc_artwork = collections.defaultdict(lambda: collections.Counter())
    wc = collections.Counter()
    for record in records:
        artwork_id = record['book_id']
        wc_doc_ = collections.Counter()
        for (_, words) in get_label_sent_words(record['review_sentences']):
            wc_doc_.update(words)
        wc_doc.append(wc_doc_)
        wc_artwork[artwork_id].update(wc_doc_)
        wc.update(wc_doc_)

    logger.info('# of books: {}'.format(len(wc_artwork)))

    # vocab
    return wc, wc_artwork, wc_doc


def process(records, word2idx, ctoi):
    doc_artwork, doc_encode, doc_key_encode, doc_char_encode = [], [], [], []
    for record in records:

        key_encode = []
        book_id = record['book_id']
        doc_keys = keywords[book_id]
        char_encode = []

        for phase in doc_keys:
            words = word_tokenize(phase)
            for word in words:
                if word in word2idx:
                    key_encode.append(word2idx[word])

        if not key_encode:
            key_encode = [word2idx['<unk>']]
        doc_key_encode.append(key_encode)

        doc_artwork.append(record['book_id'])
        encodes = get_doc_label_sent_encodes(record['review_sentences'], word2idx)

        doc_label_sent_encodes = []
        doc_char_sent_encodes = []
        for encode in encodes:
            doc_label_sent_encodes.append(encode[:2])

            words = encode[2]
            char_encode = []
            for word in words:
                chars = []
                for char in word:
                    chars.append(ctoi[char])
                char_encode.append(chars)
            doc_char_sent_encodes.append(char_encode)

        doc_encode.append(doc_label_sent_encodes)
        doc_char_encode.append(doc_char_sent_encodes)

    return doc_encode, doc_artwork, doc_key_encode, doc_char_encode


def prepare_invmap(doc_artwork, wc_review, wc_artwork):
    atod = collections.defaultdict(lambda: [])
    for d, a in enumerate(doc_artwork):
        atod[a].append(d)
    wtod = collections.defaultdict(lambda: [])
    for r, wc in enumerate(wc_review):
        for w in wc.keys():
            wtod[w].append(r)
    wtoa = collections.defaultdict(lambda: [])
    for a, wc in wc_artwork.items():
        for w in wc.keys():
            wtoa[w].append(a)
    return dict(atod), dict(wtod), dict(wtoa)


def df_idf(word, artwork_id, atod, wtod, wtoa):
    d_i_docs = atod[artwork_id]
    d_i = len(d_i_docs)
    d_w_docs = wtod[word]
    d_wi_docs = set.intersection(set(d_i_docs), set(d_w_docs))
    d_wi = len(d_wi_docs)
    df = d_wi / d_i
    e = 1
    l_w = len(wtoa[word])
    l = len(atod.keys())
    iif = math.log((l + e) / (l_w + e))
    return df * iif


def process_df_idf(doc_encode, doc_artwork, itow, atod, wtod, wtoa, log_every=100000):
    all_df_idf = []
    c = 0
    for i in range(len(doc_encode)):
        doc_label_sent_encodes = doc_encode[i]
        artwork_id = doc_artwork[i]
        for lb_s in doc_label_sent_encodes:
            for w in lb_s[1]:
                word = itow[w]
                dfidf = 1.
                if word != '<unk>':
                    dfidf = df_idf(word, artwork_id, atod, wtod, wtoa)
                all_df_idf.append(dfidf)
                c += 1
                if log_every and not c % log_every:
                    print('Processed {} words'.format(c / log_every))
    all_df_idf = _stdscale.fit_transform(np.array(all_df_idf).reshape(-1, 1)).ravel()

    doc_df_idf = []
    c = 0
    for doc_label_sent_encodes in doc_encode:
        doc = []
        for lb_s in doc_label_sent_encodes:
            sent = []
            for w in lb_s[1]:
                sent.append(all_df_idf[c].item())
                c += 1
            doc.append(sent)
        doc_df_idf.append(doc)

    return doc_df_idf


# char-process
def get_char_dict(wc):
    itoc = set()
    for word in wc:
        word = word[0]
        word = set(word)
        itoc = itoc | word

    itoc = list(itoc)
    ctoi = {c: i for i, c in enumerate(itoc)}
    logger.info("Char Vocabulary size = {}".format(len(ctoi)))
    return ctoi, itoc


# %%
if __name__ == '__main__':
    limit = 10000
    n_most_common = None
    freq_ge = 5

    logger.info('# of reviews: {}'.format(limit))
    logger.info('Getting word counts...')
    wc, wc_artwork, wc_doc = word_count(generate_records(limit))
    logger.info('Building dictionaries...')
    wtoi, itow = get_word_dict(wc, n_most_common, freq_ge)
    logger.info('Building char-level dictionaries...')
    ctoi, itoc = get_char_dict(wc)

    logger.info('Encoding reviews...')
    doc_encode, doc_artwork, doc_key_encode, doc_char_encode = process(
        generate_records(limit), wtoi, ctoi)
    logger.info('Calculating DF-IDF...')
    atod, wtod, wtoa = prepare_invmap(doc_artwork, wc_doc, wc_artwork)
    doc_df_idf = process_df_idf(doc_encode, doc_artwork, itow, atod, wtod, wtoa)
    logger.info('Saving...')
    obj = {
        'doc_label_sents': doc_encode,
        'itow': itow,
        'wc': dict(wc),
        'wc_artwork': dict(wc_artwork),
        'doc_artwork': doc_artwork,
        'doc_df_idf': doc_df_idf,
        'ctoi': ctoi,
        'doc_key_encode': doc_key_encode,
        "doc_char_encode": doc_char_encode
    }
    filename = 'mappings_{}_{}_ge{}'.format('all' if limit is None else str(limit),
                                            'all' if n_most_common is None else str(n_most_common),
                                            str(freq_ge))
    with open(os.path.join(file_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(obj, f)

# %%
