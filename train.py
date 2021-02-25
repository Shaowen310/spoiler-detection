# %%
import dataset

# %%
ds = dataset.GoodreadsReviewsSpoilerDataset(download=True)

# %%
dlist = ds.load_json(limit=10)
diter = ds.load_json(limit=10)
# %%
wc_doc, wc = ds.get_word_count_by_document(dlist)
wc_doc['22551730']
# %%