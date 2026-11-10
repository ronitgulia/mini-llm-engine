import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolLM2-135M"
WEIGHTS = MODEL_DIR / "model.safetensors"

with open(WEIGHTS, "rb") as f:
    header_size = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(header_size))

header.pop("__metadata__", None)

print("Tensors in file:", len(header))
print("lm_head.weight stored separately:", "lm_head.weight" in header)
print()

total = 0
for name in sorted(header):
    info = header[name]
    shape = info["shape"]
    count = 1
    for size in shape:
        count *= size
    total += count
    if name.startswith("model.layers.") and not name.startswith("model.layers.0."):
        continue
    print(f"{name:50} {info['dtype']:5} {str(shape):14} {count:>12,}")

print()
print("Total parameters in file:", f"{total:,}")

config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))

vocab = config["vocab_size"]
hidden = config["hidden_size"]
layers = config["num_hidden_layers"]
n_heads = config["num_attention_heads"]
n_kv_heads = config["num_key_value_heads"]
inter = config["intermediate_size"]

head_dim = hidden // n_heads
q_size = n_heads * head_dim
kv_size = n_kv_heads * head_dim

embedding = vocab * hidden
attention = hidden * q_size + 2 * hidden * kv_size + q_size * hidden
mlp = 3 * hidden * inter
norms = 2 * hidden
per_layer = attention + mlp + norms

predicted = embedding + layers * per_layer + hidden
if not config["tie_word_embeddings"]:
    predicted += vocab * hidden

print()
print("Predicted from config.json:", f"{predicted:,}")
print("Match:", predicted == total)

file_size = WEIGHTS.stat().st_size
print()
print("File size in bytes:       ", f"{file_size:,}")
print("8 + header + numbers x 2: ", f"{8 + header_size + total * 2:,}")