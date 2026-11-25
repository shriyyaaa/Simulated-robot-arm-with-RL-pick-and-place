"""BERT-based command parser: text -> (forward/back, left/right, height) offsets."""
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

HEADS = {"x": 3, "y": 3, "z": 3, "intent": 2}
X_NAMES = ["back", "centre", "forward"]
Y_NAMES = ["right", "centre", "left"]
Z_NAMES = ["on the table", "in the air", "very high"]


class CommandNLU(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in HEADS.items()})

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.drop(h)
        return {k: head(h) for k, head in self.heads.items()}


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))


class CommandParser:
    """Loads a saved model folder (no internet needed once trained)."""

    def __init__(self, path="nlu_model", min_conf=0.6):
        if not os.path.exists(os.path.join(path, "model.pt")):
            raise FileNotFoundError(f"No trained model in '{path}'. Run: python train_nlu.py")
        self.tok = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path))
        self.model = CommandNLU(encoder)
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu"))
        self.model.eval()
        self.min_conf = min_conf

    @torch.no_grad()
    def parse(self, text):
        enc = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=32)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = {k: torch.softmax(v[0], dim=-1) for k, v in logits.items()}
        pred = {k: int(p.argmax()) for k, p in probs.items()}
        conf = min(float(p.max()) for p in probs.values())
        ok = pred["intent"] == 0 and conf >= self.min_conf
        return {
            "ok": ok,
            "dx": pred["x"] - 1,        # -1 back, 0 centre, +1 forward
            "dy": pred["y"] - 1,        # -1 right, 0 centre, +1 left
            "z": pred["z"],             # 0 table, 1 air, 2 very high
            "confidence": conf,
            "description": f"{X_NAMES[pred['x']]} / {Y_NAMES[pred['y']]} / {Z_NAMES[pred['z']]}",
        }


if __name__ == "__main__":
    # Try phrasings the generator never produced.
    parser = CommandParser()
    tests = [
        "put it foward", "towards the robot", "shift the block a little to the left",
        "bring it back and to the right", "lift it high in the middle",
        "hold it up on the left", "what's the weather like", "place it far right",
    ]
    for t in tests:
        r = parser.parse(t)
        print(f"{t!r:45} -> {r['description']:35} ok={r['ok']} conf={r['confidence']:.2f}")
