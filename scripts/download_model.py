from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolLM2-135M"

path = snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-135M",
    local_dir=MODEL_DIR,
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
print("Downloaded to:", path)