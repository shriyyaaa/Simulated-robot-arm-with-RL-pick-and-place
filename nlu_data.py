"""Generate labelled training data for the command parser.

Each sentence gets four labels:
  x      0 = back (toward the robot), 1 = centre, 2 = forward (away from the robot)
  y      0 = right, 1 = centre, 2 = left
  z      0 = on the table, 1 = in the air, 2 = very high
  intent 0 = a block-moving command, 1 = something unrelated

    python nlu_data.py        # writes nlu_data.json and prints examples
"""
import json
import random

X_BACK = ["back", "backward", "backwards", "to the back", "behind", "toward the robot",
          "towards the robot", "closer", "nearer", "closer to me", "near the robot",
          "towards me", "closer to the base"]
X_FWD = ["forward", "forwards", "ahead", "to the front", "in front", "farther away",
         "further out", "away from the robot", "away from me", "far", "straight ahead"]
Y_LEFT = ["left", "to the left", "on the left", "on the left side", "leftwards",
          "towards the left"]
Y_RIGHT = ["right", "to the right", "on the right", "on the right side", "rightwards",
           "towards the right"]
CENTRE = ["in the middle", "at the center", "in the centre", "centered",
          "straight down the middle", "in the center"]

Z_TABLE = ["on the table", "down on the table", "onto the table", "at table level",
           "on the surface", "down low"]
Z_UP = ["up", "in the air", "above the table", "raised", "elevated", "up in the air"]
Z_HIGH = ["very high", "as high as possible", "as high as you can", "at the highest point",
          "really high", "way up high", "all the way up", "at maximum height", "super high"]

LEADS = ["put it", "place it", "move it", "move the block", "put the block", "place the block",
         "bring it", "take it", "carry it", "can you put it", "please move the block",
         "could you place the block", "send it", "position the block", "set the block",
         "bring the cube", "put the object", "go", "move", "please put it"]
LEADS_UP = ["lift it", "raise it", "lift the block", "raise the block", "hold it",
            "hold the block", "pick it up", "pick up the block", "hold it up", "lift"]
LEADS_DOWN = ["put it down", "set it down", "drop it", "lower it", "place it down",
              "put the block down", "lower the block"]

UNRELATED = [
    "hello", "hi there", "how are you", "what is the weather today", "tell me a joke",
    "what time is it", "sing a song", "who are you", "what is your name", "thank you",
    "good morning", "open the door", "play some music", "turn off the lights",
    "how does reinforcement learning work", "i like pizza", "what is the capital of france",
    "move on to the next question", "go to the next slide", "what does left mean",
    "how do robots work", "asdf qwer", "the quick brown fox", "i am hungry",
    "what is your favorite color", "good night", "call my friend", "set an alarm",
    "show me the results", "explain hindsight experience replay", "what can you do",
]


def typo(word, rng):
    i = rng.randrange(1, len(word))
    op = rng.choice(["delete", "swap", "dup"])
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def augment(sentence, rng, p=0.3):
    if rng.random() > p:
        return sentence
    words = sentence.split()
    idxs = [i for i, w in enumerate(words) if len(w) > 3]
    if not idxs:
        return sentence
    for i in rng.sample(idxs, k=min(len(idxs), rng.choice([1, 1, 2]))):
        words[i] = typo(words[i], rng)
    return " ".join(words)


def make_move(rng):
    x, y, z = rng.choice([0, 1, 2]), rng.choice([0, 1, 2]), rng.choice([0, 0, 1, 1, 2])
    lead = rng.choice(LEADS)
    frags = []
    if x == 0:
        frags.append(rng.choice(X_BACK))
    elif x == 2:
        frags.append(rng.choice(X_FWD))
    if y == 0:
        frags.append(rng.choice(Y_RIGHT))
    elif y == 2:
        frags.append(rng.choice(Y_LEFT))
    if x == 1 and y == 1 and rng.random() < 0.4:
        frags.append(rng.choice(CENTRE))
    if z == 0 and rng.random() < 0.3:
        lead = rng.choice(LEADS_DOWN)
    elif z == 0 and rng.random() < 0.5:
        frags.append(rng.choice(Z_TABLE))
    elif z == 1:
        if rng.random() < 0.35:
            lead = rng.choice(LEADS_UP)
        else:
            frags.append(rng.choice(Z_UP))
    elif z == 2:
        if rng.random() < 0.3:
            lead = rng.choice(LEADS_UP)
        frags.append(rng.choice(Z_HIGH))
    rng.shuffle(frags)
    parts = [lead]
    for f in frags:
        parts.append(rng.choice([" and ", " ", ", "]) + f)
    text = parts[0] + "".join(parts[1:])
    return {"text": augment(text.replace(" ,", ",").strip(), rng),
            "x": x, "y": y, "z": z, "intent": 0}


def make_unrelated(rng):
    return {"text": augment(rng.choice(UNRELATED), rng, p=0.2),
            "x": 1, "y": 1, "z": 0, "intent": 1}


def build_dataset(seed=0, n_move=6000, n_neg=700, val_frac=0.15):
    rng = random.Random(seed)
    seen, rows = set(), []
    tries = 0
    while len([r for r in rows if r["intent"] == 0]) < n_move and tries < n_move * 20:
        r = make_move(rng)
        tries += 1
        if r["text"] not in seen:
            seen.add(r["text"])
            rows.append(r)
    for _ in range(n_neg):
        rows.append(make_unrelated(rng))
    rng.shuffle(rows)
    n_val = int(len(rows) * val_frac)
    return rows[n_val:], rows[:n_val]


if __name__ == "__main__":
    train, val = build_dataset()
    with open("nlu_data.json", "w") as f:
        json.dump({"train": train, "val": val}, f, indent=1)
    print(f"{len(train)} train / {len(val)} val sentences. Examples:")
    for r in train[:12]:
        print(r)
