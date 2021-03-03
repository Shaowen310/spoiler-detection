# %%
import dataset

# %%
ds = dataset.GoodreadsReviewsSpoilerDataset(download=True)

# %%
dlist = ds.load_records(limit=10)

# %%
wc, wc_doc = ds.get_word_count(dlist)

# %%
# tf-idf
doc_label_sent_encodes_gen, word_dict = ds.encode(10, 10)
doc_label_sent_encodes = list(doc_label_sent_encodes_gen)
docs, labels, doc_lens = ds.pad(doc_label_sent_encodes)

# %%