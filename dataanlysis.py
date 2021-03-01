# %%
import dataset

# %%
ds = dataset.GoodreadsReviewsSpoilerDataset(download=True)

# %%
dlist = ds.load_records(limit=10)

# %%
wc, wc_doc = ds.get_word_count(dlist)
wc_doc['22551730']

# %%
# tf-idf
processed_records = list(ds.get_processed_record_gen(10,10))

# %%