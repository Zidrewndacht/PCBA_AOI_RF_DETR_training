r"""
Train RF-DETR-Large on pre-sliced COCO dataset.
Constants only. No defensive checks. Let it fail naturally.

# Using conda (only working for single GPU)
# conda create -p .\conda python=3.11 -y
# conda activate .\conda
# pip install "rfdetr[train,loggers]"
# pip uninstall torch torchvision torchaudio -y #because rfdetr[train] installs the CPU version by default
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

Using Docker:
    docker compose build
    docker compose up

"""
import yaml
from rfdetr import RFDETRLarge
from pathlib import Path

# Fixed container paths (Implementation detail of the Docker environment)
DATASET_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/output")

# RF-DETR saves full training states (optimizer, epoch, etc.) in .ckpt files.
# 'last.ckpt' is the standard file used for resuming interrupted training.
CHECKPOINT_PATH = OUTPUT_DIR / "last.ckpt"

if CHECKPOINT_PATH.exists():
    print(f"Resuming training from {CHECKPOINT_PATH}")
    resume_arg = str(CHECKPOINT_PATH)
else:
    print("No checkpoint found. Starting from scratch.")
    resume_arg = None

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config.yaml")
    
    # Auto-calculate gradient accumulation
    grad_accum = max(1, cfg["target_effective_batch_size"] // (cfg["batch_size"] * cfg["devices"]))
    actual_effective = cfg["batch_size"] * cfg["devices"] * grad_accum
    
    print(f"Effective batch size: {cfg['batch_size']} (per-GPU) x {cfg['devices']} (GPUs) x {grad_accum} (accum) = {actual_effective}")
    
    # Initialize Model
    model = RFDETRLarge()
    
    # Start Training
    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=grad_accum,
        lr=cfg["lr"],
        seed=cfg["seed"],
        
        # EMA & Checkpoints
        use_ema=cfg["use_ema"],
        checkpoint_interval=cfg["checkpoint_interval"],
        skip_best_epochs=cfg["skip_best_epochs"],
        
        # Early Stopping
        early_stopping=cfg["early_stopping"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
        early_stopping_use_ema=cfg["early_stopping_use_ema"],

        # Hardware config:
        device="cuda",
        strategy="ddp",
        devices=cfg["devices"],

        # We don't want naive augmentations here, 
        # this will get augmented sources from physics-based rendering directly
        aug_config={},

        resume=resume_arg
    )