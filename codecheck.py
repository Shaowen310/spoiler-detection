# %%
import collections
import os
import pickle
from typing import Sequence

# %%
data_dir = 'data_/goodreads-reviews-spoiler'
data_file = os.path.join(data_dir, 'mappings_20000_all.pkl')

with open(data_file, 'rb') as f:
    data = pickle.load(f)

# %%
# df-idf
doc_label_sents = data['doc_label_sents']
doc_df_idf = data['doc_df_idf']


# check size
def check_seq_same_shape(a: Sequence, b: Sequence):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if len(a[i][j][1]) != len(b[i][j]):
                return False
    return True


print('Same shape: {}'.format(check_seq_same_shape(doc_label_sents, doc_df_idf)))

# %%
