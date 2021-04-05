# %%
import collections
import os
import pickle
import itertools

import numpy as np
import matplotlib.pyplot as plt

import dataset

# %%
data_dir = 'data_/goodreads-reviews-spoiler'
data_file = os.path.join(data_dir, 'mappings.pkl')

# %%
# Load
with open(data_file, 'rb') as f:
    data = pickle.load(f)
doc_label_sents = data['doc_label_sent_encodes']
itow = data['idx2word']

# %%
# doc len, sent len
n_docs = len(doc_label_sents)
print('# of docs: {}'.format(n_docs))

doc_lens = [len(label_sents) for label_sents in doc_label_sents]
doc_len_count = collections.Counter(doc_lens)
min_doc_len, max_doc_len = min(doc_len_count.keys()), max(doc_len_count.keys())

fig, axes = plt.subplots(1, 1)
axes.set_xlim(xmin=0, xmax=max_doc_len)
axes.hist(doc_lens, bins=max_doc_len)
plt.savefig('plot_/hist_doc_len.pdf')

fig, axes = plt.subplots(1, 1)
axes.set_ylim(ymin=0, ymax=max_doc_len)
axes.boxplot(doc_lens)
plt.savefig('plot_/box_doc_len.pdf')

print('25, 50, 75 percentile of doc len: {}'.format(str(np.percentile(doc_lens, [25, 50, 75]))))

max_doc_len = 30
print('percentile for doc len {}: {}'.format(max_doc_len, np.count_nonzero(np.array(doc_lens) <= max_doc_len)/n_docs))

# %%
sent_lens = [len(label_sent[1]) for label_sent in itertools.chain(*doc_label_sents)]
n_sents = len(sent_lens)

print('# of sents: {}'.format(n_sents))

sent_len_count = collections.Counter(sent_lens)
min_sent_len, max_sent_len = min(sent_len_count.keys()), max(sent_len_count.keys())

fig, axes = plt.subplots(1, 1)
axes.set_xlim(xmin=0, xmax=max_sent_len)
axes.hist(sent_lens, bins=max_sent_len)
plt.savefig('plot_/hist_sent_len.pdf')

fig, axes = plt.subplots(1, 1)
axes.set_ylim(ymin=0, ymax=max_sent_len)
axes.boxplot(sent_lens)
plt.savefig('plot_/box_sent_len.pdf')

print('25, 50, 75 percentile of sent len: {}'.format(str(np.percentile(sent_lens, [25, 50, 75]))))

max_sent_len =  25
print('percentile for doc len {}: {}'.format(max_sent_len, np.count_nonzero(np.array(sent_lens) <= max_sent_len)/n_sents))

# %%
# tf-idf
doc_label_sent_encodes_gen, word_dict = ds.encode(10, 10)
doc_label_sents = list(doc_label_sent_encodes_gen)
docs, labels, doc_lens = ds.pad(doc_label_sents)

# %%