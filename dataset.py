import os
import gzip
import json
import pandas as pd
import gdown

from torch.utils.data import Dataset


class GoodreadsReviewsSpoilerDataset(Dataset):
    '''
    Credits: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
    '''
    base_folder = 'goodreads-reviews-spoiler'
    URL = 'https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS'
    filename = 'goodreads_reviews_spoiler.json.gz'

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
        file_path = os.path.join(self.root, self.base_folder, self.filename)
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
