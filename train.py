# %%
import os
import time
import math
import pickle
from typing import Sequence

import numpy as np
import torch
import sklearn.metrics as metrics

import loggingutil
from dataset import GoodreadsReviewsSpoilerDataset
from model import SpoilerNet
from paramstore import ParamStore

_logger = loggingutil.get_logger('train')
paramstore = ParamStore()
params = {}

# %% [markdown]
# ## Data
# %%
data_dir = 'data_/goodreads-reviews-spoiler'
data_file = os.path.join(data_dir, 'mappings_100000_all.pkl')
max_sent_len = 25
max_doc_len = 30
batch_size = 32
train_portion, dev_portion = 0.5, 0.25

params['max_sent_len'] = max_sent_len
params['max_doc_len'] = max_doc_len
# %%
# Load
with open(data_file, 'rb') as f:
    data = pickle.load(f)
doc_label_sents = data['doc_label_sents']
doc_df_idf = data['doc_df_idf']
itow = data['itow']


# %%
# Split train, dev, test
def train_dev_test_split_idx(rand_idx, d: Sequence, n_train: int, n_dev: int):
    d_train = [d[idx] for idx in rand_idx[:n_train]]
    d_dev = [d[idx] for idx in rand_idx[n_train:n_train + n_dev]]
    d_test = [d[idx] for idx in rand_idx[n_train + n_dev:]]
    return d_train, d_dev, d_test


n_d = len(doc_label_sents)
n_train = math.floor(n_d * train_portion)
n_dev = math.floor(n_d * dev_portion)
rand_idx = np.random.choice(n_d, n_d, replace=False)

d_train, d_dev, d_test = train_dev_test_split_idx(rand_idx, doc_label_sents, n_train, n_dev)
d_idf_train, d_idf_dev, d_idf_test = train_dev_test_split_idx(rand_idx, doc_df_idf, n_train, n_dev)

ds_train = GoodreadsReviewsSpoilerDataset(d_train, d_idf_train, itow, max_sent_len, max_doc_len)
ds_dev = GoodreadsReviewsSpoilerDataset(d_dev, d_idf_dev, itow, max_sent_len, max_doc_len)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_dev = torch.utils.data.DataLoader(ds_dev, batch_size=batch_size)

# %%
model_name = 'spoilernet'
cell_dim = 64
att_dim = 32
vocab_size = len(itow)
emb_size = 50

params['cell_dim'] = cell_dim
params['att_dim'] = att_dim
params['vocab_size'] = vocab_size
params['emb_size'] = emb_size

model_id = paramstore.add(model_name, params)

_logger = loggingutil.get_logger(model_id)

_logger.info('Data file: {}'.format(data_file))

model = SpoilerNet(cell_dim=cell_dim, att_dim=att_dim, vocab_size=vocab_size, emb_size=emb_size)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

device = torch.device('cuda:0')
model.to(device)
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


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

    for batch, (elems, labels, sentmasks, dfidf) in enumerate(dataloader):
        elems = elems.to(device)
        labels = labels.float().view(-1).to(device)
        sentmasks = sentmasks.view(-1).to(device)
        dfidf = dfidf.to(device)

        optimizer.zero_grad()

        word_h0 = model.init_hidden(len(elems)).to(device)
        sent_h0 = model.init_hidden(len(elems)).to(device)
        preds, word_h0, sent_h0 = model(elems, word_h0, sent_h0, dfidf)
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


def f1_score(y_true, y_pred, threshold=0.5):
    y_pred_lb = y_pred >= threshold
    return metrics.f1_score(y_true, y_pred_lb)


def evaluate(model, dataloader, criterion, device='cpu'):
    model.to(device)
    criterion.to(device)

    model.eval()

    epoch_loss = 0
    predss = []
    labelss = []
    sentmaskss = []
    with torch.no_grad():
        for elems, labels, sentmasks, dfidf in dataloader:
            elems = elems.to(device)
            labels = labels.float().view(-1).to(device)
            sentmasks = sentmasks.view(-1).to(device)
            dfidf = dfidf.to(device)

            word_h0 = model.init_hidden(len(elems)).to(device)
            sent_h0 = model.init_hidden(len(elems)).to(device)
            preds, word_h0, sent_h0 = model(elems, word_h0, sent_h0, dfidf)

            loss = criterion(preds, labels)
            loss *= sentmasks
            loss = torch.sum(loss) / torch.count_nonzero(sentmasks)

            epoch_loss += loss.item()

            labelss.append(labels)
            predss.append(preds)
            sentmaskss.append(sentmasks)

        labels = torch.cat(labelss).detach().cpu().numpy()
        preds = torch.sigmoid(torch.cat(predss).detach()).cpu().numpy()
        sentmasks = torch.cat(sentmaskss).detach().cpu().numpy()
        labels = labels[np.nonzero(sentmasks)]
        preds = preds[np.nonzero(sentmasks)]

        f1 = f1_score(labels, preds, 0.05)
        roc_auc = metrics.roc_auc_score(labels, preds)

    return epoch_loss / len(dataloader), f1, roc_auc


# %%
dev_loss_lowest = 1000
patience = 10
no_drop_epochs = 0
n_epochs = 50
for epoch in range(n_epochs):
    if no_drop_epochs >= patience:
        break
    epoch_loss = 0

    epoch_loss = train_one_epoch(epoch, model, dl_train, optimizer, criterion, device)

    dev_loss, dev_f1, dev_roc_auc = evaluate(model, dl_dev, criterion, device)

    _logger.info(
        '| epoch {} | epoch_loss {:.6f} | dev_loss {:.6f} | dev_f1 {:.3f} | dev_roc_auc {:.3f}'.
        format(epoch, epoch_loss, dev_loss, dev_f1, dev_roc_auc))

    if dev_loss < dev_loss_lowest:
        dev_loss_lowest = dev_loss
        torch.save(model.state_dict(), os.path.join('model_', model_id + '.pt'))
        no_drop_epochs = 0
    else:
        no_drop_epochs += 1

# %%
