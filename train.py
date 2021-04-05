# %%
import os
import time
import math
import pickle
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics

from dataset import GoodreadsReviewsSpoilerDataset
from model import SpoilerNet

# %% [markdown]
# ## Data
# %%
data_dir = 'data_/goodreads-reviews-spoiler'
data_file = os.path.join(data_dir, 'mappings_5000_all.pkl')
max_n_words = 10
max_n_sents = 30
batch_size = 32
train_portion, dev_portion, test_portion = 0.8, 0.1, 0.1

# %%
# Load
with open(data_file, 'rb') as f:
    data = pickle.load(f)
doc_label_sents = data['doc_label_sents']
itow = data['itow']


# %%
# Split train, dev, test
def train_dev_test_split(d: Sequence, train_size: float, dev_size: float):
    n_d = len(d)
    n_train = math.floor(n_d * train_size)
    n_dev = math.floor(n_d * dev_size)
    rand_idx = np.random.choice(n_d, n_d, replace=False)
    d_train = [d[idx] for idx in rand_idx[:n_train]]
    d_dev = [d[idx] for idx in rand_idx[n_train:n_train + n_dev]]
    d_test = [d[idx] for idx in rand_idx[n_train + n_dev:]]
    return d_train, d_dev, d_test


d_train, d_dev, d_test = train_dev_test_split(doc_label_sents, train_portion, test_portion)

ds_train = GoodreadsReviewsSpoilerDataset(d_train, itow, max_n_words, max_n_sents)
ds_dev = GoodreadsReviewsSpoilerDataset(d_dev, itow, max_n_words, max_n_sents)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=batch_size)


# %% [markdown]
# ## Training
# %%
def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval=1000):
    model.to(device)
    criterion.to(device)

    model.train()

    epoch_loss = 0
    log_loss = 0
    start_time = time.time()

    for batch, (elems, labels, sentmasks) in enumerate(dataloader):
        elems = elems.to(device)
        labels = labels.float().view(-1).to(device)
        sentmasks = sentmasks.view(-1).to(device)

        optimizer.zero_grad()

        word_h0 = model.init_hidden(len(elems)).to(device)
        sent_h0 = model.init_hidden(len(elems)).to(device)
        preds, word_h0, sent_h0 = model(elems, word_h0, sent_h0)
        loss = criterion(preds, labels)
        loss *= sentmasks
        loss = torch.sum(loss) / torch.count_nonzero(sentmasks)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        log_loss += batch_loss

        if log_interval and batch % log_interval == 0 and batch > 0:
            cur_loss = log_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | {:5.2f} ms/batch  | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(epoch, batch, elapsed * 1000 / log_interval,
                                                        cur_loss, math.exp(cur_loss)))
            log_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


def f1_score(y_true, y_pred, mask, threshold=0.5):
    y_true = y_true[np.nonzero(mask)]
    y_pred = y_pred[np.nonzero(mask)]
    y_pred = np.nonzero(y_pred >= threshold)
    return metrics.f1_score(y_true, y_pred)


def evaluate(model, dataloader, criterion, device='cpu'):
    model.to(device)
    criterion.to(device)

    model.eval()

    epoch_loss = 0
    predss = []
    labelss = []
    sentmaskss = []
    with torch.no_grad():
        for elems, labels, sentmasks in dataloader:
            elems = elems.to(device)
            labels = labels.float().view(-1).to(device)
            sentmasks = sentmasks.view(-1).to(device)

            word_h0 = model.init_hidden(len(elems)).to(device)
            sent_h0 = model.init_hidden(len(elems)).to(device)
            preds, word_h0, sent_h0 = model(elems, word_h0, sent_h0)

            loss = criterion(preds, labels)
            loss *= sentmasks
            loss = torch.sum(loss) / torch.count_nonzero(sentmasks)

            labelss.append(labels)
            predss.append(preds)
            sentmaskss.append(sentmasks)

    labels = torch.cat(labelss)
    preds = torch.cat(predss)
    sentmasks = torch.cat(sentmaskss)
    f1 = f1_score(labels, preds, sentmasks)

    return epoch_loss / len(dataloader), f1


# %%
cell_dim = 64
att_dim = 32
vocab_size = len(itow)
emb_size = 50
lr = 0.01
mom = 0.9

model = SpoilerNet(cell_dim=cell_dim, att_dim=att_dim, vocab_size=vocab_size, emb_size=emb_size)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

device = torch.device('cuda:0')
model.to(device)
criterion.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# %%
for epoch in range(5):
    epoch_loss = 0

    epoch_loss = train_one_epoch(epoch, model, dl_train, optimizer, criterion, device)

    print('| epoch {} | epoch_loss {} |'.format(epoch, epoch_loss))

    evaluate(model, dl_dev, criterion)

# %%