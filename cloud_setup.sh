#!/bin/bash
# ============================================================
# DigiCow Cloud Setup Script — Vast.ai / Cloud GPU Instance
# ============================================================
# Run this ONCE after SSH-ing into your cloud instance.
# It installs all dependencies and verifies GPU access.
#
# Usage:
#   chmod +x cloud_setup.sh
#   ./cloud_setup.sh
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo "  DigiCow Cloud Setup — Starting..."
echo "============================================================"

# --- 1. System Packages ---
echo ""
echo "[1/6] Updating system packages..."
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git curl > /dev/null 2>&1
echo "  ✓ System packages updated"

# --- 2. Check Python ---
echo ""
echo "[2/6] Checking Python..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "  ❌ Python not found! Installing..."
    apt-get install -y python3 python3-pip python3-venv
    PYTHON_CMD="python3"
fi
echo "  ✓ Python: $($PYTHON_CMD --version)"

# --- 3. Install uv (fast Python package manager) ---
echo ""
echo "[3/6] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # Add to shell profile for future sessions
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "  ✓ uv: $(uv --version)"

# --- 4. Setup project ---
echo ""
echo "[4/6] Setting up project environment..."
PROJECT_DIR="${1:-/workspace/digicow}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Creating project directory: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    echo ""
    echo "  ⚠  You need to upload your project files to: $PROJECT_DIR"
    echo "     Required files:"
    echo "       - train.py"
    echo "       - pyproject.toml"
    echo "       - Original Data/Train.csv"
    echo "       - Original Data/Prior.csv"
    echo "       - Original Data/Test.csv"
    echo ""
    echo "     From your LOCAL machine, run:"
    echo "       scp -P <PORT> -r '/path/to/DigiCow Farmer Training Adoption Challenge/.' root@141.0.85.201:$PROJECT_DIR/"
    echo ""
fi

cd "$PROJECT_DIR"

# Create venv and install dependencies
if [ -f "pyproject.toml" ]; then
    echo "  Found pyproject.toml, installing dependencies..."
    uv venv --python 3.12 .venv 2>/dev/null || uv venv .venv
    source .venv/bin/activate

    # Install all deps + tqdm for progress bars
    uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .
    uv pip install tqdm
    echo "  ✓ Dependencies installed"
else
    echo "  ⚠  pyproject.toml not found in $PROJECT_DIR"
    echo "     Upload your project files first, then re-run this script."
fi

# --- 5. Verify CUDA ---
echo ""
echo "[5/6] Verifying CUDA / GPU access..."
$PYTHON_CMD -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram = props.total_mem / (1024**3)
        print(f'  GPU {i}: {props.name} ({vram:.1f} GB VRAM)')
    print('  ✓ GPU is ready!')
else:
    print('  ⚠  CUDA not detected. Training will run on CPU.')
"

# --- 6. Verify NVIDIA driver ---
echo ""
echo "[6/6] NVIDIA driver info..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
    echo "  ✓ NVIDIA driver OK"
else
    echo "  ⚠  nvidia-smi not found"
fi

# --- Done ---
echo ""
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "  To start training, run:"
echo ""
echo "    cd $PROJECT_DIR"
echo "    source .venv/bin/activate"
echo ""
echo "    # Quick smoke test (RF only, ~5 min):"
echo "    python train.py --models rf --skip-ablation"
echo ""
echo "    # Full training (all models):"
echo "    python train.py"
echo ""
echo "    # Run in background with tmux:"
echo "    tmux new -s train"
echo "    python train.py 2>&1 | tee training.log"
echo "    # (Ctrl+B, then D to detach. 'tmux attach -t train' to reattach)"
echo ""
echo "============================================================"
