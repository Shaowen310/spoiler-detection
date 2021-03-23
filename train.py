# %%
import os
import time
import math

import torch

from dataset import GoodreadsReviewsSpoilerDataset
from model import SpoilerNet

# %% [markdown]
# ## Data
# %%
data_dir = 'data_/goodreads-reviews-spoiler'
train_file = os.path.join(data_dir, 'mappings.pkl')
max_n_words = 20
max_n_sents = 10

batch_size = 32

ds_train = GoodreadsReviewsSpoilerDataset(train_file, max_n_words, max_n_sents)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)


# %% [markdown]
# ## Training
# %%
def train_one_epoch(epoch,
                    model,
                    dataloader,
                    optimizer,
                    criterion,
                    device,
                    log_enabled=True,
                    log_interval=1000):
    model.to(device)
    criterion.to(device)

    model.train()

    epoch_loss = 0
    log_loss = 0
    start_time = time.time()

    for batch, (elems, labels, _) in enumerate(dataloader):
        elems = elems.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        word_h0 = model.init_hidden(len(elems)).to(device)
        sent_h0 = model.init_hidden(len(elems)).to(device)
        preds, word_h0, sent_h0 = model(elems, word_h0, sent_h0)
        labels = labels.view(-1)
        loss = criterion(preds, labels)

        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        log_loss += batch_loss

        if log_enabled and batch % log_interval == 0 and batch > 0:
            cur_loss = log_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | {:5.2f} ms/batch  | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(epoch, batch, elapsed * 1000 / log_interval,
                                                        cur_loss, math.exp(cur_loss)))
            log_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


# %%
cell_dim = 15
att_dim = 15
vocab_size = 5002
emb_size = 50
lr = 0.1
mom = 0.9

model = SpoilerNet(cell_dim=cell_dim, att_dim=att_dim, vocab_size=vocab_size, emb_size=emb_size)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cpu')
model.to(device)
criterion.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# %%
for epoch in range(5):
    epoch_loss = 0

    epoch_loss = train_one_epoch(epoch, model, dl_train, optimizer, criterion, device)

    print('| epoch {} | epoch_loss {} |'.format(epoch, epoch_loss))

# %%