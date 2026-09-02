# Docker images

Two images, both running `jepa-server` and nothing else: `Dockerfile.cpu` on a Debian slim base and
`Dockerfile.cuda` on the NVIDIA CUDA runtime image. Weights are **never** baked in — a GGUF is
hundreds of megabytes to gigabytes, and several of them are non-commercial licensed — so a model
comes in as a read-only volume at run time.

Both build from the repository root, because the ggml submodule has to be in the context:

```bash
git clone --recursive https://github.com/aselimc/jepa.cpp && cd jepa.cpp
docker build -f docker/Dockerfile.cpu  -t jepa-server:cpu  .
docker build -f docker/Dockerfile.cuda -t jepa-server:cuda .
```

`.dockerignore` keeps `models/`, the build trees and the fixtures out of the context.

## Running

```bash
# CPU: publish on the host's loopback, mount the models directory read-only
docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD/models:/models:ro" jepa-server:cpu \
    -m /models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --threads 8 --workers 2

# CUDA: one container per device
docker run --rm --gpus '"device=0"' -p 127.0.0.1:8080:8080 -v "$PWD/models:/models:ro" \
    jepa-server:cuda -m /models/gguf/ijepa_vith14_1k-f16.gguf --gpu 0
```

A container sees only the cards it was given, renumbered from zero, so `--gpu 0` inside is whichever
device the run command exposed. Where the daemon rejects `--gpus` with *"invoking the NVIDIA
Container Runtime Hook directly … is not supported"* — a snap-packaged Docker does — the equivalent
is the runtime plus the device list:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -p 127.0.0.1:8080:8080 -v "$PWD/models:/models:ro" \
    jepa-server:cuda -m /models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --gpu 0
```

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
     -d "{\"input\": {\"b64\": \"$(base64 -w0 cat.jpg)\"}}"
```

The entry point is `jepa-server --host 0.0.0.0 --port 8080`, so everything after the image name is a
server flag. `0.0.0.0` is the container's own loopback-plus-bridge, not the host's: what decides who
can reach the server is the `-p` mapping, and `-p 127.0.0.1:8080:8080` keeps it on the host's
loopback. **`jepa-server` has no TLS and no authentication** — publishing it as `-p 8080:8080` puts
an unauthenticated inference endpoint on every interface the host has.

The server runs as uid 10001, not root. There is no `HEALTHCHECK`, because neither image carries an
HTTP client to write one with; `GET /health` is the probe, and an orchestrator makes that request
itself.

One process serves one device — a model and its contexts never straddle two backends — so two cards
are two containers, each given its own device and its own published port.

## What is in them

| | CPU | CUDA |
|---|---|---|
| base | `debian:bookworm-slim` | `nvidia/cuda:13.0.3-runtime-ubuntu24.04` |
| builder | `debian:bookworm-slim` + cmake/ninja | `nvidia/cuda:13.0.3-devel-ubuntu24.04` |
| binaries | `jepa-server`, `jepa-info` | `jepa-server`, `jepa-info` |
| ggml | linked in (`BUILD_SHARED_LIBS=OFF`) | `libggml*.so` in `/usr/local/lib` |
| GPU architectures | – | `89-real;80-virtual` (`--build-arg CUDA_ARCHITECTURES=...`) |
| size | 124 MB | 4.31 GB |

Both images are built and run here, on this box: the CPU one serves embeddings, and the CUDA one
reports `NVIDIA RTX 4500 Ada Generation, compute capability 8.9, VMM: yes` from `jepa-info --devices`
and answers `/v1/embeddings` with `"backend": "CUDA0"`. The CUDA builder needs one thing a host build
does not: `libggml-cuda.so` calls the driver API, its `DT_NEEDED` is the stub's SONAME
`libcuda.so.1`, and a dependency's `DT_NEEDED` is resolved through `-rpath-link` rather than `-L` —
hence the symlink and the linker flag in `Dockerfile.cuda`.

Both are built with `JEPA_NATIVE=OFF`. That is the one thing to know about their numbers: an image
is a distributable artefact, `-march=native` bakes in whatever ISA the build host had, and the
engine's arithmetic follows the ISA — so a native image would be an image whose results depend on
where it was built.

Measured on a Threadripper 7995WX, LeJEPA-S f16, `--pool cls` on `coco_000000000139.jpg` at 8
threads: the CPU image's vector is **bit-identical** to a `JEPA_NATIVE=OFF` host build of the same
commit (0 of 384 float32 bit patterns differ, max |a−b| = 0), and differs from the repository's
default `-march=native` host build at cosine 0.999999958 with max |a−b| = 4.9e-04, all 384 patterns
differing. The gap is the AVX-512 `mul_mat` path, not the container. Edit `JEPA_NATIVE` back to `ON`
for a private image if you want the host's ISA and can promise the image will only ever run on that
host.

## Reproduce

```bash
docker build -f docker/Dockerfile.cpu -t jepa-server:cpu .
docker run -d --rm --name jepa-smoke -p 127.0.0.1:18080:8080 \
    -v "$PWD/models:/models:ro" jepa-server:cpu \
    -m /models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --threads 8 --workers 1
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    -i tests/fixtures/media/coco_000000000139.jpg --pool cls -t 8 -o /tmp/native.npy
# ... then POST the same image to :18080/v1/embeddings and compare the float32 bit patterns
docker stop jepa-smoke
```

No image is published anywhere; these files build them, and that is all they do.
