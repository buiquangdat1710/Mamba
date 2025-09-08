import torch
import torch.nn as nn
from cobra import Cobra

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
checkpoint = torch.load("cobra_translation.pth", map_location=device)

model = Cobra(
    dim=256,
    dt_rank=8,
    dim_inner=256,
    d_state=256,
    channels=64,
    num_tokens=len(checkpoint["src_vocab"]),
    depth=2,
).to(device)

head = nn.Linear(256, len(checkpoint["tgt_vocab"])).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
head.load_state_dict(checkpoint["head_state_dict"])

src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
src_itos = {i:s for s,i in src_vocab.items()}
tgt_itos = {i:s for s,i in tgt_vocab.items()}
def encode(sentence, vocab, max_len=20):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in sentence.strip().split()]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def translate(sentence):
    model.eval()
    with torch.no_grad():
        src = encode(sentence, src_vocab).unsqueeze(0).to(device)
        out = model(src)            # (1,L,256)
        logits = head(out)          # (1,L,vocab_tgt)
        pred = logits.argmax(-1).squeeze(0) # (L,)
        tokens = [tgt_itos[i.item()] for i in pred if i.item() != 0]
    return " ".join(tokens)

print(translate("My name is John"))
