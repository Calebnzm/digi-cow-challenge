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

## 2. Upload Project Files

From your **local machine**, upload the project:

```bash
# Upload everything in one go
scp -P <PORT> -r "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/." \
    root@141.0.85.201:/workspace/digicow/
```

Or upload only the essential files (faster):

```bash
# Create remote directory
ssh -p <PORT> root@141.0.85.201 "mkdir -p /workspace/digicow/'Original Data'"

# Upload essentials
scp -P <PORT> train.py pyproject.toml cloud_setup.sh \
    root@141.0.85.201:/workspace/digicow/

scp -P <PORT> "Original Data/Train.csv" "Original Data/Prior.csv" "Original Data/Test.csv" \
    root@141.0.85.201:/workspace/digicow/'Original Data'/
```

---

## 3. Run Setup Script

On the **cloud instance**:

```bash
cd /workspace/digicow
chmod +x cloud_setup.sh
./cloud_setup.sh
```

This installs all dependencies and verifies your GPU is accessible.

---

## 4. Start Training

```bash
cd /workspace/digicow
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

### Run in Background (Recommended)

Use `tmux` so training continues if you disconnect:

```bash
tmux new -s train
python train.py 2>&1 | tee training.log
```

| tmux Command | Action |
|---|---|
| `Ctrl+B`, then `D` | Detach (training continues) |
| `tmux attach -t train` | Reattach to see progress |
| `tmux kill-session -t train` | Stop the session |

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
scp -P <PORT> root@141.0.85.201:/workspace/digicow/submission_*.csv \
    "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/"

# Download training log
scp -P <PORT> root@141.0.85.201:/workspace/digicow/training.log \
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
