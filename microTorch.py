import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy
from pathlib import Path


def pick_device(preferred_gpu_name="3080"):
    if not torch.cuda.is_available():
        if os.environ.get("ALLOW_CPU") == "1":
            print("Warning: CUDA not available; running on CPU due to ALLOW_CPU=1.")
            return torch.device("cpu")
        raise SystemExit("Error: CUDA not available. Set ALLOW_CPU=1 to run on CPU.")

    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    gpu_index = next((i for i, name in enumerate(devices) if preferred_gpu_name in name), 0)
    device = torch.device(f"cuda:{gpu_index}")
    print(f"CUDA is active: {device} ({devices[gpu_index]})")
    return device


device = pick_device()

ROOT = Path(__file__).resolve().parent
raw = (ROOT / "input.txt").read_text(encoding="utf-8")

docs = []
for block in raw.split("<END>"):
    block = block.strip()
    if block.startswith("<HAIKU>"):
        docs.append(block + "\n<END>")

random.shuffle(docs)

def tokenize_text(s):
    return re.findall(r"<[^>]+>|[a-zA-Z']+|[0-9]+|\n|[^\w\s]", s) 

tokenized = [tokenize_text(d) for d in docs]
itos = sorted(set(t for d in tokenized for t in d))
stoi = {t:i for i,t in enumerate(itos)}
vocab_size = len(itos)

data = []
for d in tokenized:
    data.extend([stoi[t] for t in d])
    data.append(stoi["\n"])

data = torch.tensor(data, dtype=torch.long)

block_size = 128
batch_size = 64

def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # causal mask: token can only attend backward
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))

        return logits, loss
    
model = TinyGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(5000):
    x, y = get_batch()
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

@torch.no_grad()
def generate(prompt, max_new_tokens=300, temperature=0.8):
    model.eval()
    toks = [stoi[t] for t in tokenize_text(prompt)]
    idx = torch.tensor([toks], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        if itos[next_id.item()] == "<END>":
            break

    return "".join(itos[i] for i in idx[0].tolist())

# print(generate("<HAIKU>\n<L1><5>"))
for i in range(10):
    print(f"\n--- sample {i+1} ---")
    print(generate("<HAIKU>\n<L1><5>", temperature=0.5))