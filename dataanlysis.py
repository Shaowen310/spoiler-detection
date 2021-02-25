import os
import gzip
import json
import gdown

root = '.data'
base_folder = 'goodreads-reviews-spoiler'
URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
filename = 'goodreads_reviews_spoiler.json.gz'

# %%
# Goodreads spoiler

# %%
# Download


def download():
    file_dir = os.path.join(root, base_folder)
    file_path = os.path.join(file_dir, filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(file_path):
        gdown.download(URL, output=file_path)


def load_json(limit=None):
    file_path = os.path.join(root, base_folder, filename)
    count = 0
    records = []
    with gzip.open(file_path) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            records.append(d)

            if (limit is not None) and (count > limit):
                break
    return records


download()
records = load_json(limit=100)


# %%
# tf-idf

