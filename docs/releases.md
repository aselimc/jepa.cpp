# Releases and compatibility

Releases live on the [GitHub releases page](https://github.com/aselimc/jepa.cpp/releases). Each one is
a tag `v<version>`, a section of [`CHANGELOG.md`](https://github.com/aselimc/jepa.cpp/blob/main/CHANGELOG.md),
and a prebuilt archive for Linux x86-64.

## What ships

| asset | contents |
|---|---|
| `jepa-<version>-linux-x86_64.tar.gz` | `bin/jepa-{info,embed,classify,worldmodel,quantize,bench}`, `include/jepa.h`, `lib/lib{jepa,ggml,ggml-base,ggml-cpu}.a`, `LICENSE`, `README.md`, `CHANGELOG.md`, `SHA256SUMS` |
| `SHA256SUMS` | the checksum of the archive itself |

The binaries are CPU only and built for **x86-64-v3** — SSE4.2, AVX, AVX2, FMA, F16C, BMI2, i.e. any
Haswell-or-newer CPU — against **glibc 2.35** (Ubuntu 22.04), with `libstdc++` and `libgcc` linked
statically. The one external runtime dependency is `libgomp.so.1` (`apt install libgomp1`,
`dnf install libgomp`). ggml is linked statically, so there is nothing else to place beside the
binaries. `ffmpeg` and `ffprobe` on the `PATH` are optional: `jepa-embed` and `jepa-classify` run
them to decode a `--video` clip, and without them that one flag reports how to install them while
everything else — including the `--frames-npy` route into the same models — works unchanged.

An x86-64-v3 binary takes ggml's AVX2 kernels where a build from source on an AVX-512 machine takes
the AVX-512 ones, and the two accumulate in a different order. The outputs agree to round-off, not
bit for bit: on the LeJEPA CLS vector of one fixture image the prebuilt `jepa-embed` and a native
build differ by 4.9e-4 at a vector norm of 9.05, i.e. 5e-5 relative, well inside the f16 tier the
parity suite judges with. The CI `parity` job runs this same generic configuration against the
golden dumps on every push. The bit-exact f32 figures on the [Accuracy](accuracy.md) and
[Performance](performance.md) pages come from a native build (`-DJEPA_NATIVE=ON`, the default when
building from source), which is also the faster one on an AVX-512 CPU.

No GPU archive ships yet: `libggml-cuda.so` is 45 MB per GPU architecture and GitHub-hosted runners
have no NVIDIA device to test it on. Build one from source with `-DJEPA_CUDA=ON`
([CUDA build](getting-started.md#cuda-build)).

```bash
V=0.1.1
curl -sSLO https://github.com/aselimc/jepa.cpp/releases/download/v$V/jepa-$V-linux-x86_64.tar.gz
curl -sSLO https://github.com/aselimc/jepa.cpp/releases/download/v$V/SHA256SUMS
sha256sum -c SHA256SUMS
tar xzf jepa-$V-linux-x86_64.tar.gz
jepa-$V-linux-x86_64/bin/jepa-info --version        # 0.1.1
```

`SHA256SUMS` inside the archive lists every packaged file, so `sha256sum -c SHA256SUMS` from the
unpacked directory checks the contents as well as the download.

## Version numbers

The version is [semantic](https://semver.org/spec/v2.0.0.html) and lives in exactly one place, the
`project(... VERSION ...)` line of `CMakeLists.txt`. The build compiles it into the library, so
`jepa_version()` and `jepa-info --version` report it and a tag cannot disagree with the binaries it
publishes — `scripts/package_release.sh` refuses to package a tree where it does.

While the major version is **0**:

- **Patch** (`0.1.0` → `0.1.1`) — fixes. No new API, no schema change, no measured number moves
  outside its stated tolerance.
- **Minor** (`0.1.0` → `0.2.0`) — new model families, new API, new tools, new metadata keys. Behaviour
  of what already exists may change where the change is a fix; the changelog says which.
- **Major** — the first stable API and the first schema break, whichever comes first.

## C API stability

`include/jepa.h` is **append-only**. A declaration that ships in a release keeps its name, signature
and meaning; new calls go into the `APPEND-ONLY` blocks at the end of the header. So code compiled
against `0.1.0` compiles against every later 0.x, and the generated
[C API reference](api.md) only ever grows. Removing or changing a declaration is a major-version
change. `src/` and the tools carry no such promise: they are the implementation, not the interface.

## GGUF schema

Every file carries `jepa.schema_version`, and it is **1** for the whole 0.x line — version 1 is what
[GGUF format](gguf-schema.md) documents, and `jepa-info` prints it. A file written by any 0.x
converter loads in any 0.x engine that knows its family, so converted files never need regenerating
for a patch or minor release; the changelog says so explicitly when a release adds keys.

New families and new optional keys arrive *inside* schema v1: a key an older engine has never seen
belongs to a family it cannot build anyway, and the loader refuses a family it does not know. The
schema version bumps only when an existing key changes meaning, and that bump comes with a major
release, which is also when the loader starts range-checking the field — today it reads
`jepa.schema_version` and reports it without rejecting a value from the future.

## Continuous integration

[`ci.yml`](https://github.com/aselimc/jepa.cpp/blob/main/.github/workflows/ci.yml) runs on every push
to `main` and every pull request: builds and `ctest` on Ubuntu 22.04 and 24.04 and on macOS arm64, a
Metal build on macOS, an MSVC build on Windows, an ASAN+UBSAN build that also runs the error-path
and thread-contract suites and compiles the fuzz target, and the generator checks that keep
`docs/api.md`, the figures and the accuracy tables in step with their sources.
[`release.yml`](https://github.com/aselimc/jepa.cpp/blob/main/.github/workflows/release.yml) builds,
packages, smoke-tests and publishes the archive from a `v*` tag.
