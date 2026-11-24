# Simulated Robot Arm: RL Pick-and-Place

A simulated Fetch robot arm (MuJoCo) learns to pick up a block and place it at a
target position, using SAC + Hindsight Experience Replay (HER). Software only,
no robot hardware required.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Roadmap (do these in order)
| Stage | Command | Rough budget | Goal |
|---|---|---|---|
| 1. Sanity check | `python train.py --env FetchReach-v4 --steps 50000` | minutes | ~100% success |
| 2. Push | `python train.py --env FetchPush-v4 --steps 1000000` | hours on CPU | >80% |
| 3. Pick-and-place | `python train.py --env FetchPickAndPlace-v4 --steps 3000000` | many hours on CPU | >70% |

Budgets are rough estimates. Pick-and-place is the hardest of the three because the reward is
sparse (success or not); HER relabels failed attempts with the goal actually reached,
which is what makes learning feasible.

## Monitor and evaluate
    tensorboard --logdir runs
    python evaluate.py --model runs/FetchPickAndPlace-v4_seed0/best/best_model.zip
    python evaluate.py --model <path> --video demo.mp4   # headless recording

## Experiments for a report
- Ablation: HER on vs. off (expect off to fail on PickAndPlace).
- Goal selection strategy: `future` vs `final` vs `episode`.
- `n_sampled_goal` in {2, 4, 8}.
- Seeds: run 3 seeds and plot mean +/- std of success rate.
- Compare SAC vs TD3 (same HER setup).
- Curriculum: warm-start PickAndPlace from a Push checkpoint.

## Stretch: language-conditioned goals
`command_demo.py` turns phrases like "put the block up and to the left" into a target
position for the trained policy. It uses keyword rules; replace with a small intent model
or a speech-to-text front end for a conversational-AI angle.

## If it's too slow
- Start with Reach and Push, and report PickAndPlace as partial progress with learning curves.
- Lower `--n-envs` if RAM is tight, and `buffer_size` in `train.py` (1M entries is large).
- Checkpoints save every 100k steps, so you can stop and evaluate any time.

## Notes
- Written against Gymnasium-Robotics `Fetch*-v4` and Stable-Baselines3 2.x. Env IDs and the
  `HerReplayBuffer` API change between versions; if something errors, check those first.
- Not run end-to-end in my sandbox (no network to install MuJoCo), so expect to debug small things.
