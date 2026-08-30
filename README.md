# jepa.cpp

CPU inference for the JEPA family (I-JEPA, V-JEPA 1/2/2.1, V-JEPA 2-AC, LeJEPA-style ViTs, LeWorldModel)
in plain C/C++ on top of [ggml](https://github.com/ggml-org/ggml). Inference only — no training code.

Status: **scaffold** (see `docs/architecture.md`, `docs/gguf-schema.md`, `docs/parity.md`).

```bash
git clone --recursive <this repo>
cmake -B build -G Ninja && cmake --build build -j
# python side (conversion + golden references only)
uv venv .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install transformers safetensors numpy gguf pillow huggingface_hub av
scripts/download_models.sh small      # LeJEPA ViT-S, LeWM, V-JEPA 2.1 ViT-B
scripts/download_fixtures.sh          # 8 images + a few clips
```
