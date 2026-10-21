"""Cache the pinned official general-purpose BiRefNet weights locally."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "birefnet"

if __name__ == "__main__":
    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["config.json", "model.safetensors"],
        local_dir=MODEL_DIR,
    )
    print(f"Model ready: {MODEL_DIR}")
