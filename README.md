# mini-llm-engine

A small LLM inference engine built from scratch in PyTorch, as a learning project.

The goal is to load a real open-weights language model, re-implement its
transformer by hand, and then make it fast the way production systems do.

Status: work in progress, built step by step.

## Roadmap

- [x] Project setup (Python, virtual environment, PyTorch, Git)
- [ ] Inspect a real model: config, weights and tokenizer
- [ ] Re-implement the transformer and match Hugging Face outputs exactly
- [ ] Text generation loop with a KV cache
- [ ] Benchmarks: tokens/sec and latency
- [ ] Continuous batching and paged attention
- [ ] int8 / int4 quantization
- [ ] Triton / CUDA kernels
- [ ] Comparison with vLLM and a write-up

## Setup (Windows, CPU)

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python check_setup.py