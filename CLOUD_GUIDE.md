# ☁️ Cloud GPU Training Guide — DigiCow Challenge

Step-by-step guide to run training on your **Vast.ai RTX 5060 Ti** instance.

---

## 1. Connect to Your Instance

Your Vast.ai instance will provide an SSH command. It will look like:

```bash
ssh -p <PORT> root@141.0.85.201
```

> [!TIP]
> Find the exact SSH command in the Vast.ai dashboard under your instance → "Connect".

---

## 2. Get Project Files onto the Instance

### Option A: Git Clone (Recommended)

Since you pushed to GitHub, just clone directly on the instance:

```bash
git clone https://github.com/Calebnzm/digi-cow-challenge.git /workspace/digi-cow-challenge
cd /workspace/digi-cow-challenge
```

Then upload the **data files** (these aren't in git) from your **local machine**:

```bash
scp -P <PORT> "Original Data/Train.csv" "Original Data/Prior.csv" "Original Data/Test.csv" \
    root@141.0.85.201:/workspace/digi-cow-challenge/'Original Data'/
```

### Option B: SCP everything

```bash
scp -P <PORT> -r "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/." \
    root@141.0.85.201:/workspace/digi-cow-challenge/
```

---

## 3. Run Setup Script

On the **cloud instance**:

```bash
cd /workspace/digi-cow-challenge
chmod +x cloud_setup.sh
./cloud_setup.sh
```

This installs all dependencies and verifies your GPU is accessible.

---

## 4. Start Training

```bash
cd /workspace/digi-cow-challenge
source .venv/bin/activate
```

### Quick Smoke Test (~5 min)
```bash
python train.py --models rf --skip-ablation
```

### Full Training (All Models)
```bash
python train.py
```

### Custom Configurations
```bash
# Only neural network models
python train.py --models nn rwn --skip-ablation

# Faster run with 3 folds instead of 5
python train.py --n-splits 3

# Save outputs to a specific directory
python train.py --output-dir results/

# All options
python train.py --help
```

---

## 5. Using tmux (Run Training in the Background)

**Why tmux?** If your SSH connection drops (laptop sleeps, Wi-Fi hiccup, etc.), any running command dies with it. `tmux` keeps your training alive on the server even if you disconnect.

Think of it as a "virtual terminal" that lives on the server.

### Step-by-step

**1. Start a new tmux session** (give it a name like "train"):
```bash
tmux new -s train
```

You'll see a green bar at the bottom — that means you're inside tmux.

**2. Run your training command** (you're now inside tmux):
```bash
cd /workspace/digi-cow-challenge
source .venv/bin/activate
python train.py 2>&1 | tee training.log
```

> [!NOTE]
> `2>&1 | tee training.log` saves all output to `training.log` AND still shows it on screen. This way you have a record even after the session ends.

**3. Detach from tmux** (training keeps running in background):
- Press **`Ctrl+B`**, release both keys, then press **`D`**
- You'll see `[detached (from session train)]`
- You can now safely close your SSH connection — training continues!

**4. Come back later** — reconnect SSH and reattach:
```bash
ssh -p <PORT> root@141.0.85.201
tmux attach -t train
```

You'll see your training output exactly where you left off.

**5. When training is done**, exit tmux:
```bash
exit
```

### Visual Flow

```
You (laptop) ──SSH──▶ Cloud Instance
                          │
                     ┌────┴────┐
                     │  tmux   │  ◀── lives on the server
                     │ "train" │
                     ├─────────┤
                     │ python  │  ◀── your training runs here
                     │ train.py│
                     └─────────┘

  SSH drops? ──✗──▶  tmux keeps running ✓
  Reconnect: tmux attach -t train
```

### tmux Commands Cheat Sheet

| Action | What to type |
|---|---|
| Start a new session | `tmux new -s train` |
| Detach (go to background) | `Ctrl+B`, then `D` |
| Reattach to session | `tmux attach -t train` |
| List all sessions | `tmux ls` |
| Kill a session | `tmux kill-session -t train` |
| Scroll up in tmux | `Ctrl+B`, then `[`, then arrow keys. Press `q` to exit scroll |

### Running with Different Arguments in tmux

It works exactly the same — just type your full command inside the tmux session:

```bash
# Start tmux
tmux new -s train

# Then run whatever command you want:
python train.py --models nn rwn --n-splits 3 --skip-ablation 2>&1 | tee training.log

# Detach: Ctrl+B, then D
```

To run a **different training** later, just kill the old session and start a new one:

```bash
tmux kill-session -t train
tmux new -s train
python train.py --models rf 2>&1 | tee training_rf.log
```

---

## 5. Monitor GPU Usage

In a separate terminal/tmux pane:

```bash
# One-shot GPU status
nvidia-smi

# Live monitoring (updates every 1s)
watch -n 1 nvidia-smi
```

---

## 6. Download Results

After training completes, from your **local machine**:

```bash
# Download all submission CSVs
scp -P <PORT> root@141.0.85.201:/workspace/digi-cow-challenge/submission_*.csv \
    "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/"

# Download training log
scp -P <PORT> root@141.0.85.201:/workspace/digi-cow-challenge/training.log \
    "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/"
```

---

## 7. When You're Done

Destroy or stop your Vast.ai instance from the dashboard to stop billing.

> [!CAUTION]
> Your instance costs **$0.121/hr**. Remember to stop it when training is complete!

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA not available` | Run `nvidia-smi` to check driver. Restart instance if needed. |
| `ModuleNotFoundError` | Run `source .venv/bin/activate` then `uv pip install -e .` |
| `FileNotFoundError: Train.csv` | Ensure data is in `Original Data/` subdirectory |
| SSH connection drops | Use `tmux` (see step 4) so training survives disconnects |
| Out of VRAM | Reduce batch size: `--batch-size 64` or hidden dim: `--nn-hidden 256` |
