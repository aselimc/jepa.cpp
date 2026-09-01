# jepa-cpp

Python bindings for [jepa.cpp](https://github.com/aselimc/jepa.cpp) — CPU (and optional CUDA)
inference for the JEPA family: I-JEPA, V-JEPA 2, V-JEPA 2.1, LeJEPA, LeVJEPA and LeWorldModel.
One GGUF file in, numpy arrays out.

The package is a thin wrapper over the C API in `include/jepa.h`. The engine, the preprocessing —
the bit-exact torchvision-style resize the golden fixtures were generated against — and every
floating-point operation are on the C side; Python only marshals arguments and copies results into
numpy. `python/tests/test_parity.py` gates that: on a CPU the bindings reproduce
`build/jepa-embed`'s output **bit for bit**.

## Install

```bash
pip install jepa-cpp
```

The wheel bundles one self-contained shared library (libjepa plus ggml, linked whole), so there is
no ggml to install and nothing to compile.

To build it from a checkout instead — the only way to get a CUDA-enabled or `-march=native`
library:

```bash
git clone --recursive https://github.com/aselimc/jepa.cpp && cd jepa.cpp
pip install ./python                 # -march=native, matching the repository's own build/
```

An installed package can also be pointed at a library you built yourself, which is how the parity
tests make the bindings and `build/jepa-embed` the very same code:

```bash
export JEPA_CPP_LIB=/path/to/libjepa.so
```

## Ten lines

```python
import jepa_cpp, numpy as np

with jepa_cpp.Model("models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf", threads=8) as m:
    print(m.family, m.embed_dim, m.file_type_name)          # hfvit 384 f16
    tokens = m.encode("cat.jpg")                            # [197, 384] float32, every token
    feature = m.encode("cat.jpg", pool="cls")               # [384]
    batch = m.encode(["cat.jpg", "dog.jpg"], pool="cls")    # [2, 384], one encoder graph

with jepa_cpp.Model("models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf", threads=32) as m:
    clip = np.load("clip.npy")                              # uint8 [T, H, W, 3]
    print(m.classify(clip).top(5))                          # [(index, label, probability), ...]
```

## What is on `Model`

| | |
|---|---|
| `Model(path, device=…, threads=…, flash_attn=…, flash_kv=…, max_batch=…, verbose=…)` | load and build a context; `device` is `"cpu"`, `"cuda:0"`, an index, or `None` to follow `$JEPA_DEVICE` |
| `encode(x, pool=None, as_video=None)` | one item → `[n_tokens, dim]`, a list → `[batch, n_tokens, dim]`; `pool="mean" / "cls" / "lewm"` collapses each to `[dim]` |
| `pool(tokens, mode)`, `preprocess(x)`, `load_image(path)`, `token_grid(t, h, w)` | the pieces `encode` is made of |
| `classify(x)` | attentive-pool head → `Classification(logits, probs, pooled, labels)` with `.top(k)` |
| `predict(tokens, target, context, mask_index=1, modality="video")` | the masked predictor → `[n_target, out_dim]` |
| `lewm_project(tokens)`, `lewm_project_rows(rows)`, `lewm_predict(embs, actions)`, `lewm_rollout(embs, actions, n_steps)` | the LeWorldModel path |
| `family`, `name`, `embed_dim`, `patch_size`, `tubelet_size`, `img_size`, `n_frames`, `n_layer`, `n_head`, `has_cls`, `n_registers`, `n_prefix_tokens`, `has_predictor`, `has_head`, `has_projector`, `n_classes`, `labels`, `file_type`, `file_type_name`, `n_bytes`, `device`, `device_name`, `is_gpu`, `is_video` | the model |
| `threads`, `max_batch`, `last_batch`, `last_compute_ms`, `mul_mat_prec_f32` | the context |

Module level: `jepa_cpp.version()`, `jepa_cpp.devices()`, `jepa_cpp.system_info()`,
`jepa_cpp.library_path()`, and `jepa_cpp.JepaError`, which carries the text the C library logged
for the failure.

Every C entry point is also reachable unwrapped as `jepa_cpp._api.jepa_*`, with the header's own
signatures and its own conventions (0 / -1 return codes, `jepa_free` on returned buffers).
`tests/test_api_coverage.py` parses `include/jepa.h` and fails if any of them is missing.

## Images, clips and batches

`encode` follows `jepa-embed`'s rules so that the two agree:

- a path, a `uint8 [H, W, 3]` array, or a preprocessed `float32 [3, T, H, W]` array is one item;
- a `uint8 [T, H, W, 3]` array is one clip;
- a **list** is a batch of independent items — unless the model is a video family and the list is
  longer than one, or `as_video=True`, in which case the entries are the frames of a single clip
  and may have different sizes;
- a still image through LeVJEPA is repeated to the trained clip length, because that is how its
  model card feeds one. `jepa_encode` itself never repeats: a one-frame clip stays a one-frame
  clip.

Image families fold up to `max_batch` items (default 32) into one encoder graph, which is
bit-identical to encoding them one at a time and considerably faster. Video families run one clip
per graph.

## Threads and lifetime

A `Model` owns one `jepa_context`, which is per-thread compute state in C, so calls on it are
serialised on a lock: sharing a `Model` between Python threads is safe but not concurrent. Give
each thread its own `Model`, or raise `threads` to parallelise inside one graph. `Model` is a
context manager; `close()` frees the weights and is idempotent.

## Tests

```bash
pip install -e ".[test]" && pytest tests
```

`tests/test_parity.py` is the one that matters: it runs `build/jepa-embed` and
`build/jepa-worldmodel` as subprocesses and requires the bindings' arrays to match theirs bit for
bit on a CPU, then compares both to the PyTorch golden dumps in `tests/fixtures/ref/` with the same
per-family, per-dtype bars `tests/test-parity.cpp` uses. `tests/test_api_coverage.py` parses
`include/jepa.h` and fails when an entry point is unbound; `tests/test_model_api.py` covers
lifetime, argument checking and error text. GGUF files and golden dumps are not in the repository,
so a test whose assets are missing skips.

## Licence

MIT, the same as jepa.cpp.
