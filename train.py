import torch
import torch.nn as nn
import model
import math
from torch.nn import functional as F
import config
from data import (
    load_dailydialog, load_or_train_tokenizer,
    DAILYDIALOG_DIR, DAILYDIALOG_DATASET, DAILYDIALOG_REVISION,
)


model_config = config.ModelConfig()
train_config = config.TrainConfig()
batch_size = train_config.batch_size
block_size = model_config.block_size
max_iters = train_config.max_iters
learning_rate = train_config.learning_rate
eval_interval = train_config.eval_interval
eval_iters = train_config.eval_iters
device = 'cuda' if torch.cuda.is_available() else 'cpu' #Device to run the model on
n_enbd = model_config.n_embed
n_head = model_config.n_head
n_layers = model_config.n_layer
dropout = model_config.dropout


# Keep DailyDialog's original splits; do not split conversations by character.
train_text = load_dailydialog("train")
val_text = load_dailydialog("validation")
tokenizer = load_or_train_tokenizer(train_text, path=DAILYDIALOG_DIR / "tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
train_data = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
val_data = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)

torch.manual_seed(1337)


def get_batch(split='train'):
    #generate a batch of data from the specified split
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out



model = model.BigramLanguageModel(vocab_size=vocab_size).to(device)
idx = train_data[:1].unsqueeze(0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Enable AMP only when running on CUDA; on CPU autocast is a no-op / may warn
use_amp = (device == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for iter_num in range(max_iters):
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)

    # Mixed precision forward pass (no-op on CPU)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        logits, loss = model(xb, yb)

    # Scaled backward + step; scaler is disabled on CPU so this is safe
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "tokenizer_json": tokenizer.to_str(),
    "dataset_name": DAILYDIALOG_DATASET,
    "dataset_revision": DAILYDIALOG_REVISION,
}, train_config.model_path)
#Print out sample result
print(tokenizer.decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
