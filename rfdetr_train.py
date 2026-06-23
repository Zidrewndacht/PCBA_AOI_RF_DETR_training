"""
Train RF-DETR-Large on pre-sliced COCO dataset.
Constants only. No defensive checks. Let it fail naturally.

# Using conda (not really, didn't work)
# conda create -p .\conda python=3.11 -y
# conda activate .\conda
# pip install "rfdetr[train,loggers]"
# pip uninstall torch torchvision torchaudio -y #because rfdetr[train] installs the old version by default
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

Using Docker:
Transfer data to volume (to avoid constant slow host disk reading):
    docker run --rm -v /mnt/d/!staging/@Mestrado/PCBA-AOI/output_sliced:/src -v rfdetr_dataset:/dest alpine sh -c "cp -a /src/. /dest/"
Run script with arguments as required:
    docker compose run --rm rfdetr python rfdetr_train.py --resume checkpoint.pth

"""
import yaml
from rfdetr import RFDETRLarge

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
    model = RFDETRLarge(resolution=cfg["resolution"])
    
    # Start Training
    model.train(
        dataset_dir=cfg["dataset_dir"],
        output_dir=cfg["output_dir"],
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
        # strategy="ddp",
        # devices=cfg["devices"],
    )