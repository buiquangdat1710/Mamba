from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. Load dataset IWSLT'15
# =========================
dataset = load_dataset("opus100", "en-vi")
# Dataset có các split: train, validation, test
print(dataset)

# =========================
# 2. Lấy 1500 sample từ tập train
# =========================
subset = dataset["train"].select(range(1500))

# =========================
# 3. Xây vocab đơn giản (token = từ cách nhau bởi space)
# =========================
src_vocab = {"<pad>":0, "<unk>":1}
tgt_vocab = {"<pad>":0, "<unk>":1}

def add_to_vocab(sentence, vocab):
    for token in sentence.strip().split():
        if token not in vocab:
            vocab[token] = len(vocab)

for sample in subset:
    add_to_vocab(sample["translation"]["en"], src_vocab)
    add_to_vocab(sample["translation"]["vi"], tgt_vocab)

src_itos = {i:s for s,i in src_vocab.items()}
tgt_itos = {i:s for s,i in tgt_vocab.items()}

# =========================
# 4. Encode sentences
# =========================
def encode(sentence, vocab, max_len=20):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in sentence.strip().split()]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, subset, src_vocab, tgt_vocab, max_len=20):
        self.data = []
        for sample in subset:
            src = sample["translation"]["en"]
            tgt = sample["translation"]["vi"]
            self.data.append((encode(src, src_vocab, max_len),
                              encode(tgt, tgt_vocab, max_len)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = TranslationDataset(subset, src_vocab, tgt_vocab, max_len=20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# 2. Import Cobra model (bạn đã code ở trên)
# =========================
from cobra import Cobra  # giả sử bạn đã lưu Cobra model trong cobra.py

# =========================
# 3. Train Cobra
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = Cobra(
    dim=256,
    dt_rank=8,
    dim_inner=256,
    d_state=256,
    channels=64,
    num_tokens=len(src_vocab),  # vocab nguồn
    depth=2,   # giảm depth cho nhẹ
).to(device)

# Output head để map sang vocab đích
head = nn.Linear(256, len(tgt_vocab)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

EPOCHS = 20
for epoch in range(EPOCHS):
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        out = model(src)   # (B,L,256)
        logits = head(out) # (B,L,vocab_tgt)
        loss = criterion(logits.view(-1, len(tgt_vocab)), tgt.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss {loss.item():.4f}")

# =========================
# 4. Suy luận
# =========================
def translate(sentence):
    model.eval()
    with torch.no_grad():
        src = encode(sentence, src_vocab).unsqueeze(0).to(device)
        out = model(src)            # (1,L,256)
        logits = head(out)          # (1,L,vocab_tgt)
        pred = logits.argmax(-1).squeeze(0) # (L,)
        tokens = [tgt_itos[i.item()] for i in pred if i.item() != 0]
    return " ".join(tokens)


torch.save({
    "model_state_dict": model.state_dict(),
    "head_state_dict": head.state_dict(),
    "src_vocab": src_vocab,
    "tgt_vocab": tgt_vocab
}, "cobra_translation.pth")

print(translate("hello"))
print(translate("i love you"))
print(translate("thank you"))
