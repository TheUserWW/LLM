import torch
import torch.nn as nn
import config
from datasets import load_dataset
from torch.nn import functional as F

#hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu' #Device to run the model on


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.ModelConfig().n_embed, head_size, bias=False)
        self.query = nn.Linear(config.ModelConfig().n_embed, head_size, bias=False)
        self.value = nn.Linear(config.ModelConfig().n_embed, head_size, bias=False)
        # Retained for compatibility with existing checkpoints; SDPA handles masking.
        self.register_buffer("tril", torch.tril(torch.ones(config.ModelConfig().block_size, config.ModelConfig().block_size)))

        self.dropout = nn.Dropout(config.ModelConfig().dropout)

    def forward(self, x):
        # SDPA expects (batch, heads, sequence, head_size).
        k = self.key(x).unsqueeze(1)
        q = self.query(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        return out.squeeze(1)

class MultiHeadAttention(nn.Module):
    #Just do single-head attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.ModelConfig().n_embed)
        self.dropout = nn.Dropout(config.ModelConfig().dropout)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Apply each attention head to the input and concatenate the results
        #x: (B, T, C)
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1) #concatenate along the last dimension
        out = self.proj(out) #project back to the original embedding dimension
        out = self.dropout(out)
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, n_enbd=config.ModelConfig().n_embed):
        n_enbd = config.ModelConfig().n_embed
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_enbd, n_enbd*4 ),
            nn.ReLU(),
            #Projection layer to map back to the original embedding dimension
            nn.Linear(n_enbd*4, n_enbd),
            nn.Dropout(config.ModelConfig().dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #Transformer block
    def __init__(self, n_enbd=config.ModelConfig().n_embed, num_heads=config.ModelConfig().n_head):
        super().__init__()
        head_size = n_enbd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForwardNetwork(n_enbd)
        self.ln1 = nn.LayerNorm(n_enbd)
        self.ln2 = nn.LayerNorm(n_enbd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size=config.ModelConfig().vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.ModelConfig().n_embed)
        self.positional_embedding_table = nn.Embedding(config.ModelConfig().block_size, config.ModelConfig().n_embed)
        self.blocks = nn.Sequential(
        *[Block(config.ModelConfig().n_embed, num_heads=config.ModelConfig().n_head) for _ in range(config.ModelConfig().n_layer)],)
        #Final layer norm before the output projection
        self.ln_f = nn.LayerNorm(config.ModelConfig().n_embed)
        self.lm_head = nn.Linear(config.ModelConfig().n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_enb = self.token_embedding_table(idx) #B, T, C
        pos_enb = self.positional_embedding_table(torch.arange(T, device=idx.device)) #T, C
        x = tok_enb + pos_enb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #B, T, vocab_size

        if targets is None:
            return logits, None
        else:
            # B: batch size, T: sequence length, C: number of classes (vocab size)
            B,T,C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
            return logits,loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.ModelConfig().block_size:]
            #get predictions for the next token
            logits, loss = self.forward(idx_cond)
            #focus on the last token's logits for prediction
            logits = logits[:, -1, :] / temperature
            #apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution of probabilitiess
            idx_next = torch.multinomial(probs, num_samples=1)
            #append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx





