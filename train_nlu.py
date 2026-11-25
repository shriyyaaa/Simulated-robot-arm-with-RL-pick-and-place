"""Fine-tune a small BERT model to parse block-placement commands.

    pip install transformers
    python nlu_data.py          # optional: just to look at the data
    python train_nlu.py         # ~5-15 minutes on CPU with distilbert
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from nlu import HEADS, CommandNLU, save_model
from nlu_data import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="distilbert-base-uncased")  # or bert-base-uncased
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="nlu_model")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = build_dataset(seed=args.seed)
    print(f"{len(train)} train / {len(val)} val sentences, device={device}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = CommandNLU(AutoModel.from_pretrained(args.base)).to(device)

    def collate(rows):
        enc = tok([r["text"] for r in rows], return_tensors="pt", padding=True,
                  truncation=True, max_length=32)
        labels = {k: torch.tensor([r[k] for r in rows]) for k in HEADS}
        return enc, labels

    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val, batch_size=64, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for enc, labels in train_dl:
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = sum(ce(logits[k], labels[k].to(device)) for k in HEADS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        model.eval()
        correct = {k: 0 for k in HEADS}
        exact = 0
        with torch.no_grad():
            for enc, labels in val_dl:
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                ok = torch.ones(len(labels["x"]), dtype=torch.bool)
                for k in HEADS:
                    hit = logits[k].argmax(-1).cpu() == labels[k]
                    correct[k] += hit.sum().item()
                    ok &= hit
                exact += ok.sum().item()
        n = len(val)
        accs = " ".join(f"{k}={correct[k] / n:.3f}" for k in HEADS)
        print(f"epoch {epoch + 1}: loss={total / len(train_dl):.3f}  val acc: {accs}  exact={exact / n:.3f}")

    save_model(model.cpu(), tok, args.out)
    print(f"Saved to {args.out}/. Test with: python nlu.py")


if __name__ == "__main__":
    main()
