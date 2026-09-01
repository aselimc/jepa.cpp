#!/usr/bin/env python3
"""Publish the converted GGUFs and the parity fixtures to Hugging Face, reproducibly.

Every number in a generated card is read at run time from a committed artifact -- the GGUF itself
(via `jepa-info`), `docs/parity.md`, `docs/quantization.md`, `tests/results/accuracy-{image,video,ssv2}.json`
and `tests/results/benchmarks{,-gpu}.json`. Only names, prose and licence attributions live in this file,
so re-running it after a re-measurement produces cards that agree with the docs by construction.

    scripts/hf_publish.py cards  [--out tmp/hf-cards]              # write the 7 model cards + the dataset card
    scripts/hf_publish.py upload [--models NAME...] [--fixtures]   # upload, one repo at a time, resumable
    scripts/hf_publish.py check                                    # hub sha256/size/cards vs local; exit 1 on drift

    NAME    ijepa | lejepa | lewm | vjepa2 | vjepa2-ssv2 | vjepa21 | levjepa  (or the GGUF basename)
    --org   Hugging Face organisation, default jepacpp
    --out   where `cards` writes, default <root>/tmp/hf-cards (git-ignored)

`upload` with neither --models nor --fixtures uploads every model repo and the dataset. It needs a
write token (`hf auth login`); `cards` and `check` need none for public repos.

Layout on the hub, fixed -- other tooling (scripts/download_models.sh, scripts/download_fixtures.sh)
depends on it:

    <org>/<basename>-GGUF                 <basename>-{f32,f16,q8_0,q4_0,q4_k}.gguf + README.md
    <org>/jepa.cpp-fixtures  (dataset)    ref/<model>/...                          + README.md

Requires: models/gguf populated, tests/fixtures/ref populated (for --fixtures), a built `build/jepa-info`
(`$JEPA_INFO` overrides the path) and `huggingface_hub` (the project .venv has both).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "models" / "gguf"
FIXTURES = ROOT / "tests" / "fixtures"
RESULTS = ROOT / "tests" / "results"
DOCS = ROOT / "docs"
JEPA_INFO = pathlib.Path(os.environ.get("JEPA_INFO") or ROOT / "build" / "jepa-info")
DTYPES = ["f32", "f16", "q8_0", "q4_0", "q4_k"]
DATASET = "jepa.cpp-fixtures"
SITE = "https://aselimc.github.io/jepa.cpp"
REPO = "https://github.com/aselimc/jepa.cpp"

# ---------------------------------------------------------------------------------------------
# Per-model constants: names, prose and attribution only. Anything measurable is parsed, not typed.
# ---------------------------------------------------------------------------------------------

MODELS = [
    dict(
        base="ijepa_vith14_1k",
        short="ijepa",
        title="I-JEPA ViT-H/14 (IN1k)",
        parity_name="ijepa-vith14-1k",
        ref_dir="ijepa-vith14-1k",
        quant_section="I-JEPA ViT-H/14",
        image_model="ijepa-vith14-1k",
        video_model=None,
        base_model="facebook/ijepa_vith14_1k",
        pipeline="image-feature-extraction",
        tags=["i-jepa", "image-feature-extraction", "vision"],
        blurb=(
            "Meta's **I-JEPA** ViT-H/14 image encoder, self-supervised on ImageNet-1k, converted to GGUF for\n"
            "[jepa.cpp]({repo}) — a ggml C/C++ engine that runs it on a plain CPU with no Python and no\n"
            "PyTorch. No CLS token: the feature is the mean over the patch tokens after the final LayerNorm."
        ),
        licence=(
            "**CC BY-NC 4.0 — non-commercial use only.** The source checkpoint is published by **Meta AI (FAIR)**\n"
            "under [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/):\n"
            "`license: cc-by-nc-4.0` on the model card, and the full CC text as\n"
            "[LICENSE](https://github.com/facebookresearch/ijepa/blob/main/LICENSE) in `facebookresearch/ijepa`.\n"
            "These GGUF files are Adapted Material — the same weights re-serialised into the GGUF container,\n"
            "quantized where the file name says so — so they carry the same licence, credit Meta, and are marked\n"
            "as modified. Cite the [I-JEPA paper](https://arxiv.org/abs/2301.08243) (Assran et al., 2023)."
        ),
        convert="python scripts/convert.py --family ijepa --src models/facebook/ijepa_vith14_1k --ftype f16",
        run=[
            "# one image -> a pooled feature vector",
            "build/jepa-embed -m {base}-f16.gguf -i photo.jpg --pool mean -t 32 -o feat.npy",
        ],
        below_bar={},
        note=(
            "A small fraction of this model's tokens fall well below the pooled cosine at q8_0 — the\n"
            "low-variance tokens that the final LayerNorm amplifies — while every pooled feature stays close to 1.\n"
            "Use q8_0 and below for pooled features and retrieval, f16 or f32 for dense per-token work; the\n"
            "token-level analysis is in [quantization]({site}/quantization/)."
        ),
    ),
    dict(
        base="lejepa-vits16-pretrain-in1k",
        short="lejepa",
        title="LeJEPA ViT-S/16 (IN1k)",
        parity_name="lejepa-vits16",
        ref_dir="lejepa-vits16",
        quant_section="LeJEPA ViT-S/16",
        image_model="lejepa-vits16",
        video_model=None,
        base_model="OK-AI/lejepa-vits16-pretrain-in1k",
        pipeline="image-feature-extraction",
        tags=["lejepa", "image-feature-extraction", "vision"],
        blurb=(
            "The community **LeJEPA** ViT-S/16 image encoder (a DINOv2-style ViTv2 backbone) pretrained on\n"
            "ImageNet-1k, converted to GGUF for [jepa.cpp]({repo}) — a ggml C/C++ engine that runs it on a plain\n"
            "CPU with no Python and no PyTorch. The smallest and fastest model in the set."
        ),
        licence=(
            "**Apache-2.0.** The source checkpoint is published by **Open-Knowledge-AI (OK-AI)**, trained from\n"
            "scratch on ImageNet-1k with their `lite_ssl` codebase. `license: apache-2.0` in the model card's\n"
            "metadata is the repository's only licence statement — it ships no `LICENSE` file. These GGUF files\n"
            "are the same weights re-serialised into the GGUF container, quantized where the file name says so.\n"
            "Cite the [LeJEPA paper](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025)."
        ),
        convert=(
            "python scripts/convert.py --family hfvit --src models/OK-AI/lejepa-vits16-pretrain-in1k --ftype f16"
        ),
        run=[
            "# one image -> a CLS feature",
            "build/jepa-embed -m {base}-f16.gguf -i photo.jpg --pool cls -t 32 -o feat.npy",
        ],
        below_bar={"q4_k": "the derived `cls` feature misses the low-bit tier's 0.99 bar, on the CPU and on CUDA alike"},
        note=(
            "This model's matrices are not all a multiple of the K-quant block, so `q4_k` falls back per tensor to\n"
            "`q4_0` for most of them and the two files come out the same size; `general.file_type` reads `2 (q4_0)`\n"
            "as a result. The fallback rule is in [quantization]({site}/quantization/)."
        ),
    ),
    dict(
        base="lewm-pusht",
        short="lewm",
        title="LeWorldModel Push-T",
        parity_name="lewm-pusht",
        ref_dir="lewm-pusht",
        quant_section="LeWM PushT",
        image_model="lewm-pusht",
        video_model=None,
        base_model="quentinll/lewm-pusht",
        pipeline="image-feature-extraction",
        tags=["world-model", "robotics", "image-feature-extraction"],
        blurb=(
            "**LeWorldModel** Push-T: a ViT-Ti/14 image encoder plus a latent, action-conditioned world-model\n"
            "predictor, converted to GGUF for [jepa.cpp]({repo}) — a ggml C/C++ engine that runs it on a plain CPU\n"
            "with no Python and no PyTorch. The encoder's CLS token is projected to a world-model state, and six\n"
            "AdaLN-zero causal blocks conditioned on a 10-d action roll that state forward."
        ),
        licence=(
            "**MIT.** The source checkpoint is published by **quentinll** for the `le-wm` project\n"
            "([github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), MIT); `license: mit` on the model\n"
            "card. These GGUF files are the same weights re-serialised into the GGUF container, quantized where the\n"
            "file name says so, and carry the MIT notice. Cite the\n"
            "[LeWorldModel paper](https://arxiv.org/abs/2603.19312)."
        ),
        convert="python scripts/convert.py --family lewm --src models/quentinll/lewm-pusht --ftype f16",
        run=[
            "# one image -> a world-model state, then 8 random action steps",
            "build/jepa-embed      -m {base}-f16.gguf -i photo.jpg --pool lewm -t 32",
            "build/jepa-worldmodel -m {base}-f16.gguf --image photo.jpg --random-actions 8 -t 32",
        ],
        below_bar={},
        note=(
            "The most quantization-robust model in the set — narrow, shallow, with BatchNorm folded into the MLPs —\n"
            "and the only one whose predictor `tests/test-predictor` gates with a bit-exactness causality check:\n"
            "perturbing step *t* leaves every prediction before *t* bit-identical."
        ),
    ),
    dict(
        base="vjepa2-vitl-fpc64-256",
        short="vjepa2",
        title="V-JEPA 2 ViT-L/16 (fpc64, 256)",
        parity_name="vjepa2-vitl-fpc64-256",
        ref_dir="vjepa2-vitl-fpc64-256",
        quant_section="V-JEPA 2 ViT-L/16 fpc64",
        image_model=None,
        video_model="vjepa2-vitl-fpc64-256",
        base_model="facebook/vjepa2-vitl-fpc64-256",
        pipeline=None,
        tags=["v-jepa", "v-jepa-2", "video", "video-feature-extraction"],
        blurb=(
            "Meta's **V-JEPA 2** ViT-L/16 video encoder with its masked latent predictor, converted to GGUF for\n"
            "[jepa.cpp]({repo}) — a ggml C/C++ engine that runs it on a plain CPU with no Python and no PyTorch.\n"
            "Tubelets of two frames and 3-D RoPE in Meta's *tiled* layout; a whole clip goes through one graph."
        ),
        licence=(
            "**MIT.** The source checkpoint is published by **Meta AI (FAIR)**: `license: mit` on the model card and\n"
            "[LICENSE](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE) in `facebookresearch/vjepa2`\n"
            "(Copyright (c) Meta Platforms, Inc. and affiliates). There is no separate weights licence, no gating and\n"
            "no acceptable-use policy. These GGUF files are the same weights re-serialised into the GGUF container,\n"
            "quantized where the file name says so. Cite the [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)."
        ),
        convert="python scripts/convert.py --family vjepa2 --src models/facebook/vjepa2-vitl-fpc64-256 --ftype f16",
        run=[
            "# a clip (THWC uint8 .npy, written by scripts/video_frames.py) -> a pooled feature",
            "build/jepa-embed -m {base}-f16.gguf --frames-npy clip.npy --pool mean -t 32 -o feat.npy",
        ],
        below_bar={},
        note=(
            "**Use f32 if you consume individual tokens of this model.** Its activation range contains a degenerate\n"
            "low-norm token cluster that the F16 activation rounding inside ggml's `mul_mat` collapses — a property\n"
            "of the checkpoint, reproduced in numpy, not an engine defect, and the reason its f16 and q8_0 worst-token\n"
            "columns above read so much lower than its pooled ones. Everything pooled is unaffected. The mechanism is\n"
            "worked through in [accuracy]({site}/accuracy/)."
        ),
    ),
    dict(
        base="vjepa2-vitl-fpc16-256-ssv2",
        short="vjepa2-ssv2",
        title="V-JEPA 2 ViT-L/16 SSv2 classifier",
        parity_name="vjepa2-vitl-fpc16-256-ssv2",
        ref_dir="vjepa2-vitl-fpc16-256-ssv2",
        quant_section="V-JEPA 2 ViT-L/16 SSv2 classifier",
        image_model=None,
        video_model=None,
        base_model="facebook/vjepa2-vitl-fpc16-256-ssv2",
        pipeline="video-classification",
        tags=["v-jepa", "v-jepa-2", "video", "video-classification", "action-recognition"],
        blurb=(
            "Meta's **V-JEPA 2** ViT-L/16 encoder with its Something-Something-v2 attentive-pooler head — 174 action\n"
            "classes from one clip — converted to GGUF for [jepa.cpp]({repo}), a ggml C/C++ engine that runs it on a\n"
            "plain CPU with no Python and no PyTorch. The 174 label strings travel inside the GGUF, so no side file\n"
            "is needed."
        ),
        licence=(
            "**MIT.** The source checkpoint is published by **Meta AI (FAIR)**: `license: mit` on the model card and\n"
            "[LICENSE](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE) in `facebookresearch/vjepa2`\n"
            "(Copyright (c) Meta Platforms, Inc. and affiliates). There is no separate weights licence, no gating and\n"
            "no acceptable-use policy. These GGUF files are the same weights re-serialised into the GGUF container,\n"
            "quantized where the file name says so. Cite the [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985). The\n"
            "head was fine-tuned on\n"
            "[Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset),\n"
            "whose dataset terms are a separate matter from this weight licence."
        ),
        convert=(
            "python scripts/convert.py --family vjepa2 --src models/facebook/vjepa2-vitl-fpc16-256-ssv2 --ftype f16"
        ),
        run=[
            "# a 16-frame clip -> the top 5 of 174 actions, or the pooled feature",
            "build/jepa-classify -m {base}-f16.gguf --frames-npy clip.npy -k 5 -t 32",
            "build/jepa-embed    -m {base}-f16.gguf --frames-npy clip.npy --pool mean -t 32",
        ],
        below_bar={"q4_k": "misses even the advisory derived-tensor bar (logits and pooled), on the CPU and on CUDA alike"},
        note=(
            "**f16 is the recommendation for classifier work.** Quantization moves decisions without moving the\n"
            "score: read the agreement column, not the top-1 delta, as the cost of a low-bit file. The full argument,\n"
            "with the per-clip breakdown, is in [accuracy]({site}/accuracy/)."
        ),
    ),
    dict(
        base="vjepa2_1-vitb-384",
        short="vjepa21",
        title="V-JEPA 2.1 ViT-B/16 @384",
        parity_name="vjepa2_1-vitb-384",
        ref_dir="vjepa2_1-vitb-384",
        quant_section="V-JEPA 2.1 ViT-B/16 384",
        image_model=None,
        video_model="vjepa2_1-vitb-384",
        base_model=None,
        pipeline=None,
        tags=["v-jepa", "v-jepa-2", "video", "video-feature-extraction", "image-feature-extraction"],
        blurb=(
            "Meta's **V-JEPA 2.1** ViT-B/16 encoder at 384x384, distilled from ViT-g, with its masked latent\n"
            "predictor — converted to GGUF for [jepa.cpp]({repo}), a ggml C/C++ engine that runs it on a plain CPU\n"
            "with no Python and no PyTorch. The one model here that handles **both modalities**: a still image goes\n"
            "through the image tokenizer and its own modality vector, a clip through the tubelet tokenizer."
        ),
        licence=(
            "**MIT.** The source is a bare `.pt` checkpoint published by **Meta AI (FAIR)** from the V-JEPA 2.1\n"
            "[checkpoint table](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints) of\n"
            "`facebookresearch/vjepa2`, whose [LICENSE](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE)\n"
            "is MIT (Copyright (c) Meta Platforms, Inc. and affiliates). The file itself carries no licence metadata,\n"
            "so MIT here is the repository-level grant that publishes it rather than a per-file one; there is no\n"
            "gating and no acceptable-use policy. These GGUF files are the same weights re-serialised into the GGUF\n"
            "container, quantized where the file name says so. Cite the\n"
            "[V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)."
        ),
        convert=(
            "python scripts/convert.py --family vjepa2_1 --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \\\n"
            "                          --out models/gguf/vjepa2_1-vitb-384-f16.gguf"
        ),
        run=[
            "# an image and a clip through the same file",
            "build/jepa-embed -m {base}-f16.gguf -i photo.jpg          --pool mean -t 32",
            "build/jepa-embed -m {base}-f16.gguf --frames-npy clip.npy --pool mean -t 32 -o feat.npy",
        ],
        below_bar={},
        note=(
            "Far more forgiving at reduced precision than the V-JEPA 2 ViT-L encoders — its worst-token column stays\n"
            "high at f16 — so f16 is a per-token-grade configuration here, not only a pooled-feature-grade one."
        ),
    ),
    dict(
        base="levjepa-vitl16",
        short="levjepa",
        title="LeVJEPA ViT-L/16 (VideoMix)",
        parity_name="levjepa-vitl16",
        ref_dir="levjepa-vitl16",
        quant_section="LeVJEPA ViT-L/16",
        image_model=None,
        video_model="levjepa-vitl16",
        base_model="galilai-group/LeVJEPA-VideoMix-Large",
        pipeline=None,
        tags=["levjepa", "video", "video-feature-extraction"],
        blurb=(
            "The community **LeVJEPA** ViT-L/16 video encoder, trained from scratch on VideoMix (Kinetics-710, SSv2,\n"
            "Walking Tours, PE-Video), converted to GGUF for [jepa.cpp]({repo}) — a ggml C/C++ engine that runs it on\n"
            "a plain CPU with no Python and no PyTorch. Tubelet 1 and a **block-causal** attention mask —\n"
            "bidirectional inside a frame, causal across frames, with the CLS token a read-only sink. The feature is\n"
            "that CLS token."
        ),
        licence=(
            "**CC BY-NC 4.0 — non-commercial use only.** The source checkpoint is published by **galilai-group** under\n"
            "[Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/):\n"
            "`license: cc-by-nc-4.0` in the model card's metadata, the repository's only licence statement (it ships\n"
            "no `LICENSE` file). The weights were trained from scratch, so the restriction is the publisher's own\n"
            "choice rather than inherited. These GGUF files are Adapted Material — the same weights re-serialised\n"
            "into the GGUF container, quantized where the file name says so — so they carry the same licence, credit\n"
            "galilai-group, and are marked as modified."
        ),
        convert=(
            "python scripts/convert.py --family levjepa --src models/galilai-group/LeVJEPA-VideoMix-Large \\\n"
            "                          --out models/gguf/levjepa-vitl16-f16.gguf --ftype f16"
        ),
        run=[
            "# a clip -> a CLS feature (a still image is repeated to the model's frame count, as its card does)",
            "build/jepa-embed -m {base}-f16.gguf --frames-npy clip.npy --pool cls -t 32 -o feat.npy",
            "build/jepa-embed -m {base}-f16.gguf -i photo.jpg          --pool cls -t 32",
        ],
        below_bar={},
        note=(
            "**No low-cosine token tail.** Where the V-JEPA 2 ViT-L encoders drop individual tokens badly at f16,\n"
            "not one of this model's tokens does on any fixture — its reference row norms sit in a narrow band, so\n"
            "the F16 activation rounding has no degenerate low-norm cluster to amplify. Its rows are all a multiple\n"
            "of the K-quant block, so the K-quants never fall back."
        ),
    ),
]
BY_BASE = {m["base"]: m for m in MODELS}
BY_SHORT = {m["short"]: m for m in MODELS}

TIERS = {
    "f32": "exact",
    "f16": "parity",
    "q8_0": "parity",
    "q4_0": "advisory",
    "q4_k": "advisory",
}

# ---------------------------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------------------------


def strip_md(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def md_tables(path: pathlib.Path):
    """Yield (heading, header_cells, row_cells) for every pipe table, tracking the enclosing heading."""
    heading = ""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            yield heading, header, rows
            continue
        i += 1


def read_parity() -> dict:
    """(parity_name, ftype) -> list of dicts of the CPU test-parity rows of docs/parity.md."""
    out: dict = {}
    for heading, header, rows in md_tables(DOCS / "parity.md"):
        low = [h.lower() for h in header]
        if "model" not in low or "ftype" not in low or "cos mean" not in low:
            continue
        if "cuda" in heading.lower() or "gpu" in heading.lower():
            continue  # the GPU section has the same shape; the cards report the CPU backend
        idx = {h: k for k, h in enumerate(low)}
        for r in rows:
            if len(r) != len(header):
                continue
            rec = {h: strip_md(r[k]) for h, k in idx.items()}
            out.setdefault((rec["model"], rec["ftype"]), []).append(rec)
    return out


def read_quant() -> dict:
    """(quant_section, type) -> {tensor: {metric: value}}, plus ('__source__', section) -> ftype."""
    out: dict = {}
    for heading, header, rows in md_tables(DOCS / "quantization.md"):
        low = [h.lower() for h in header]
        if not low or low[0] != "type" or "tensor" not in low:
            continue
        section = heading.split("(")[0].strip()
        m = re.search(r"source (f16|f32)", heading)
        if m:
            out[("__source__", section)] = m.group(1)
        idx = {h: k for k, h in enumerate(low)}
        cur = None
        for r in rows:
            if len(r) != len(header):
                continue
            first = strip_md(r[0])
            if first:
                cur = first.split("(")[0].strip()
                out.setdefault((section, cur), {})["__size__"] = strip_md(r[idx["file size"]])
            if cur is None:
                continue
            tensor = strip_md(r[idx["tensor"]])
            if not tensor:
                continue
            out.setdefault((section, cur), {})[tensor] = {h: strip_md(r[k]) for h, k in idx.items()}
    return out


def read_json(name: str):
    return json.loads((RESULTS / name).read_text())


def jepa_info(path: pathlib.Path) -> dict:
    """Parse `jepa-info FILE` into a dict of the fields the cards use."""
    tool = JEPA_INFO if JEPA_INFO.exists() else shutil.which("jepa-info")
    if not tool:
        sys.exit(f"{JEPA_INFO} not found -- build the tools (cmake --build build) or set $JEPA_INFO")
    text = subprocess.run([str(tool), str(path)], capture_output=True, text=True, check=True).stdout
    info: dict = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(.*)$", line)
        if m:
            info[m.group(1)] = m.group(2).strip()
    m = re.search(r"weights:\s+([\d.]+) MiB in (\d+) tensors", text)
    if m:
        info["weights_mib"] = float(m.group(1))
        info["n_tensors"] = int(m.group(2))
    for key in ("D", "layers", "heads", "patch", "tubelet", "img", "frames"):
        m = re.search(rf"\b{key}=(\d+)", text)
        if m:
            info[key] = int(m.group(1))
    info["has_cls"] = "cls=yes" in text
    return info


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_many(paths):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(paths, ex.map(sha256, paths)))


def mib(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MiB"


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


# ---------------------------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------------------------


def measured_cell(m: dict, ftype: str, parity: dict, quant: dict) -> str:
    """One 'measured against the PyTorch reference' cell, from parity.md if the row exists, else quantization.md."""
    rows = parity.get((m["parity_name"], ftype), [])
    if rows:
        row = min(rows, key=lambda r: float(r.get("cos mean", "1") or 1))
        bits = [f"cos mean {row['cos mean']}"]
        if row.get("cos med"):
            bits.append(f"median {row['cos med']}")
        bits.append(f"worst {row['cos min']}")
        for name, label in (("pooled", "`pooled_mean`"), ("cls", "`cls`"), ("logits", "logits")):
            v = row.get(name)
            if v and v not in ("-", "–", "—"):
                bits.append(f"{label} {v}")
        top = row.get("top-1/top-5")
        if top and top not in ("-", "–", "—"):
            bits.append(f"top-1/top-5 {top}")
        if ftype == "f32" and row.get("rel_max"):
            bits.append(f"`rel_max` {row['rel_max']}")
        return ", ".join(bits) + " " + PARITY_MARK
    tensors = quant.get((m["quant_section"], ftype), {})
    if not tensors:
        return "—"
    bits = []
    lhs = tensors.get("last_hidden_state")
    if lhs:
        bits.append(f"cos mean {lhs['cos mean']}, worst {lhs['cos min (worst token)']}")
    for name, label in (("pooled_mean", "`pooled_mean`"), ("cls", "`cls`"), ("emb", "`emb`"),
                        ("pred_next", "`pred_next`"), ("logits", "logits")):
        t = tensors.get(name)
        if t:
            bits.append(f"{label} {t['cos mean']}")
            if name == "logits" and t.get("top-1 / top-5"):
                bits.append(f"top-1/top-5 {t['top-1 / top-5']}")
    return ", ".join(bits) + " " + QUANT_MARK


PARITY_MARK = "ᵖ"
QUANT_MARK = "ᵈ"
FOOT = (
    f"<sub>{PARITY_MARK} `tests/test-parity` on the CPU backend, stored reference input, 32 threads, worst sample "
    f"— [docs/parity.md]({SITE}/parity/). {QUANT_MARK} `scripts/gguf_dequant_selftest.py`: the dequantized weights "
    f"through the numpy reference graph at f32 activations, so the figure is the weight error alone — "
    f"[docs/quantization.md]({SITE}/quantization/). `cos mean` is the mean per-token cosine of "
    f"`last_hidden_state`, `worst` its single worst token.</sub>"
)


def knn_image_table(m: dict, img: dict, info: dict) -> str:
    """The Imagenette k-NN table for this model, straight from tests/results/accuracy-image.json."""
    rows = [r for r in img["rows"] if r["model"] == m["image_model"]]
    if not rows:
        return ""
    want = "emb" if info.get("family", "").startswith("lewm") else ("cls" if info["has_cls"] else "mean")
    groups = {}
    for r in rows:
        groups.setdefault((r["feature"], r["gallery"], r["n_gallery"], r["n_query"]), []).append(r)
    key = max(groups, key=lambda k: (k[0] == want, k[2]))
    feature, gallery, n_gallery, n_query = key
    p = img["protocol"]
    out = [
        f"**Imagenette k-NN** — {n_query} queries against a gallery of {n_gallery} (`{gallery}`), the frozen "
        f"`{feature}` feature, k = {p['k']} cosine vote. Nothing is trained.",
        "",
        "| backend | dtype | k-NN top-1 % | centroid top-1 % | agreement with PyTorch % | mean feature cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(groups[key], key=lambda r: (r["backend"] != "pytorch", DTYPES.index(r["dtype"]))):
        ref = r["backend"] == "pytorch"
        top1 = f"**{pct(r['knn_top1'])}**" if ref else pct(r["knn_top1"])
        agree = "—" if ref else pct(r["agreement"])
        cos = "1" if ref else f"{r['feat_cos_mean']:.6f}"
        out.append(f"| {r['backend']} | {r['dtype']} | {top1} | {pct(r['centroid_top1'])} | {agree} | {cos} |")
    return "\n".join(out)


def knn_video_table(m: dict, vid: dict) -> str:
    """The UCF-101 k-NN table for this model, straight from tests/results/accuracy-video.json."""
    entry = vid["models"].get(m["video_model"])
    if not entry:
        return ""
    split = "val+test"
    ds = vid["dataset"]
    n_query = entry["backends"]["torch"]["splits"][split]["n"]
    out = [
        f"**UCF-101 k-NN** — {len(ds['classes'])} classes, {n_query} query clips ({split}) against a gallery of "
        f"{ds['gallery']['n']}, {ds['frames_per_clip']} frames per clip, k = {vid['protocol']['knn']['k']} cosine "
        f"vote over frozen features. Nothing is trained.",
        "",
        "| backend | dtype | k-NN top-1 % | centroid top-1 % | k-NN agreement % | centroid agreement % | feature cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, b in entry["backends"].items():
        s = b["splits"][split]
        ref = b["backend"] == "pytorch"
        a = b.get("agreement", {}).get(split, {})
        agree_k = "—" if ref else f"**{pct(a['knn'])}**" if a["knn"] == 1.0 else pct(a["knn"])
        agree_c = "—" if ref else f"**{pct(a['centroid'])}**" if a["centroid"] == 1.0 else pct(a["centroid"])
        cos = "—" if ref else f"{a['feat_cos']:.6f}"
        out.append(
            f"| {b['backend']} | {b['dtype']} | {pct(s['knn_top1'])} | {pct(s['centroid_top1'])} | "
            f"{agree_k} | {agree_c} | {cos} |"
        )
    return "\n".join(out)


def ssv2_table(ssv2: dict) -> str:
    """The SSv2 validation tables, straight from tests/results/accuracy-ssv2.json."""
    full = [r for r in ssv2["runs"] if r["scope"] == "full"]
    sub = [r for r in ssv2["runs"] if r["scope"] == "sub10" and r["backend"] in ("cpu", "pytorch")]
    n_full = full[0]["n_clips"]
    n_cls = ssv2["dataset"].get("n_classes") or ssv2["model"].get("n_classes")
    out = [
        f"**Something-Something-v2 validation, all {n_full:,} clips**"
        + (f", {n_cls} classes" if n_cls else "")
        + ", one view per clip and no test-time augmentation. Both engines read the same uniformly sampled "
        "frames and each runs the preprocessing itself.",
        "",
        "| backend | dtype | top-1 % | top-5 % | top-1 agreement with PyTorch % | logit cosine, mean / worst clip |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def row(r, engine, ref_top1, ref_top5):
        v = r.get("vs_pytorch") or {}
        ref = r["backend"] == "pytorch"
        # bold a jepa.cpp rate only where it lands on the reference's, to the precision shown
        top1 = f"**{pct(r['top1'])}**" if ref or pct(r["top1"]) == ref_top1 else pct(r["top1"])
        top5 = f"**{pct(r['top5'])}**" if ref or pct(r["top5"]) == ref_top5 else pct(r["top5"])
        agree = "—" if ref else pct(v["top1_agreement"])
        cos = "—" if ref else f"{v['logit_cos_mean']:.7f} / {v['logit_cos_min']:.8g}"
        return f"| {engine} | {r['dtype']} | {top1} | {top5} | {agree} | {cos} |"

    ref_full = next(r for r in full if r["backend"] == "pytorch")
    for r in sorted(full, key=lambda r: (r["backend"] != "pytorch", DTYPES.index(r["dtype"]))):
        out.append(row(r, "PyTorch (float32, TF32 off)" if r["backend"] == "pytorch" else "jepa.cpp CUDA",
                       pct(ref_full["top1"]), pct(ref_full["top5"])))
    n_sub = sub[0]["n_clips"]
    out += [
        "",
        f"On the CPU, over a fixed {n_sub:,}-clip subset (every 10th clip of the validation order):",
        "",
        "| backend | dtype | top-1 % | top-5 % | top-1 agreement with PyTorch % | logit cosine, mean / worst clip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    ref_sub = next(r for r in sub if r["backend"] == "pytorch")
    for r in sorted(sub, key=lambda r: (r["backend"] != "pytorch", DTYPES.index(r["dtype"]))):
        out.append(row(r, "PyTorch" if r["backend"] == "pytorch" else "jepa.cpp CPU, 32 threads",
                       pct(ref_sub["top1"]), pct(ref_sub["top5"])))
    return "\n".join(out)


def speed_line(m: dict, bench: dict, gpu: dict) -> str:
    """One sentence of measured latency, from tests/results/benchmarks{,-gpu}.json."""
    enc = [r for r in bench["rows"]
           if r["model"] == m["base"] and r["mode"] == "encoder" and r["ftype"] == "f16" and r["threads"] == 32]
    if not enc:
        return ""
    box = bench.get("box", {})
    cpu = box.get("cpu") or box.get("name") or "the reference box"
    bits = []
    for r in sorted(enc, key=lambda r: r["tokens"]):
        unit = "image" if r["frames"] == 1 else f"{r['frames']}-frame clip"
        piece = f"{r['ms_mean']:.0f} ms per {unit}"
        if r.get("pytorch_ms") and r.get("pytorch_comparable"):
            piece += f" against PyTorch's {r['pytorch_ms']:.0f} ms"
        bits.append(piece)
    line = f"**Speed** — the encoder graph at f16 on 32 threads ({cpu}): " + "; ".join(bits) + "."
    grows = [r for r in gpu.get("rows", [])
             if r.get("model") == m["base"] and r.get("mode") == "encoder" and r.get("ftype") == "f16"]
    if grows:
        g = min(grows, key=lambda r: r["tokens"])
        dev = gpu.get("box", {}).get("device", "one GPU")
        line += f" The same shape on {dev}: {g['ms_mean']:.1f} ms."
    rss = [r for r in enc if r.get("peak_rss_mib")]
    if rss:
        line += f" Peak RSS at f16: {max(r['peak_rss_mib'] for r in rss):.0f} MiB."
    return line


def render_card(m: dict, ctx: dict) -> str:
    base, org = m["base"], ctx["org"]
    repo_id = f"{org}/{base}-GGUF"
    info = ctx["info"][base]
    sizes = ctx["sizes"][base]
    shas = ctx["shas"][base]
    params = round(info["weights_mib"] * 1024 * 1024 / 4 / 1e6)
    fmt = dict(repo=REPO, site=SITE, base=base, org=org)

    out = ["---", f"license: {info['license']}"]
    if m["base_model"]:
        out.append(f"base_model: {m['base_model']}")
    out.append("library_name: jepa.cpp")
    if m["pipeline"]:
        out.append(f"pipeline_tag: {m['pipeline']}")
    out.append("tags:")
    for t in ["jepa", "ggml", "gguf", "jepa.cpp"] + m["tags"]:
        out.append(f"  - {t}")
    out += ["---", "", f"# {m['title']} — GGUF for jepa.cpp", "", m["blurb"].format(**fmt), ""]

    shape = f"D = {info['D']}, {info['layers']} layers, {info['heads']} heads, patch {info['patch']}"
    if info.get("tubelet", 1) > 1:
        shape += f", tubelet {info['tubelet']}"
    shape += f", {info['img']}x{info['img']}"
    out += [
        f"**{params} M** parameters; {shape}. Everything the engine needs — dimensions, positional scheme,",
        "preprocessing recipe, and class labels where there are any — travels inside the file, so inference needs",
        "one binary and one GGUF and nothing else.",
        "",
        "## Run it",
        "",
        "```bash",
        f"git clone --recursive {REPO} && cd jepa.cpp",
        "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        f"hf download {repo_id} {base}-f16.gguf --local-dir models/gguf",
        "",
    ]
    out += [line.format(**fmt) for line in m["run"]]
    out += [
        "```",
        "",
        "`--pool` selects `mean`, `cls`, `lewm` or `none` (the full token map); `-o` writes a `.npy`.",
        f"`scripts/download_models.sh` fetches whole sets at once. The C API is one header,",
        f"[`include/jepa.h`]({REPO}/blob/main/include/jepa.h) — full reference on the [C API page]({SITE}/api/).",
        "",
        "## Files",
        "",
        "| file | size | sha256 (first 16) | tier | measured against the PyTorch reference |",
        "|---|---|---|---|---|",
    ]
    for d in DTYPES:
        fn = f"{base}-{d}.gguf"
        tier = TIERS[d]
        if d in m["below_bar"]:
            tier += ", **below the bar**"
        out.append(f"| `{fn}` | {mib(sizes[d])} | `{shas[d][:16]}` | {tier} | "
                   f"{measured_cell(m, d, ctx['parity'], ctx['quant'])} |")
    out += ["", FOOT, ""]
    out += [
        "**Tiers.** `exact` — reproduces the PyTorch reference to the printed precision on the CPU. `parity` —",
        "passes its family's `test-parity` thresholds. `advisory` — below 8 bits per weight, which is not a parity",
        "configuration: the results are reported, only the derived tensors and the top-1 are gated. Which file to",
        f"ship: [Accuracy → which dtype]({SITE}/accuracy/#which-dtype-to-ship).",
    ]
    for d, why in m["below_bar"].items():
        out.append(f"`{base}-{d}.gguf` {why}.")
    out += ["", "Full checksums:", "", "```"]
    out += [f"{shas[d]}  {base}-{d}.gguf" for d in DTYPES]
    out += [
        "```",
        "",
        "Verify a download with `sha256sum -c`. The other types `jepa-quantize` can produce (`q4_1`, `q5_0`,",
        f"`q5_1`, `q5_k`, `q6_k`, measured in [quantization]({SITE}/quantization/)) are not published here; make",
        f"them locally with `build/jepa-quantize {base}-f16.gguf out.gguf q6_k -t 32`.",
        "",
        "## Measured",
        "",
        f"Every figure below is read from a committed artifact of jepa.cpp [`{ctx['commit']}`]({REPO}/commit/"
        f"{ctx['commit']}) by `scripts/hf_publish.py` — [parity]({SITE}/parity/), "
        f"[quantization]({SITE}/quantization/), [accuracy]({SITE}/accuracy/), [performance]({SITE}/performance/) "
        "and `tests/results/*.json`.",
        "",
    ]
    blocks = []
    if m["image_model"]:
        blocks.append(knn_image_table(m, ctx["img"], info))
    if m["video_model"]:
        blocks.append(knn_video_table(m, ctx["vid"]))
    if m["base"] == ctx["ssv2"]["model"]["name"]:
        blocks.append(ssv2_table(ctx["ssv2"]))
    blocks.append(speed_line(m, ctx["bench"], ctx["gpu"]))
    blocks.append(m["note"].format(**fmt))
    out += ["\n\n".join(b for b in blocks if b), ""]

    src = info.get("source", "")
    src_name = src.replace("https://huggingface.co/", "").replace("https://", "")
    out += [
        "## Source, licence and attribution",
        "",
        f"Converted from [`{src_name}`]({src}).",
        "",
        m["licence"].format(**fmt),
        "",
        "The licence travels inside every GGUF as `general.license` and the origin as `general.source_url`;",
        "`build/jepa-info <file> --kv` prints them. jepa.cpp's own code is MIT.",
        "",
        "## Conversion",
        "",
        f"Produced by jepa.cpp [`{ctx['commit']}`]({REPO}/commit/{ctx['commit']}):",
        "",
        "```bash",
        f"scripts/download_models.sh --convert {m['short']}",
        m["convert"],
    ]
    if "--out" in m["convert"]:
        out.append(f"#   ... and again with --ftype f32 --out models/gguf/{base}-f32.gguf for the f32 file")
    else:
        out.append("#   ... and again with --ftype f32 for the f32 file")
    quant_src = ctx["quant"].get(("__source__", m["quant_section"]), "f16")
    out += [
        "",
        "for q in q8_0 q4_0 q4_k; do",
        f"  build/jepa-quantize models/gguf/{base}-{quant_src}.gguf \\",
        f"      models/gguf/{base}-$q.gguf $q -t 32",
        "done",
        "```",
        "",
        "`jepa-quantize` re-types only the 2-D attention / FFN / projection / classifier matrices; patch",
        "embeddings, position tables, norms and biases keep the source type. The rules are in",
        f"[docs/gguf-schema.md]({SITE}/gguf-schema/).",
        "",
        "## Links",
        "",
        f"- Code: <{REPO}>",
        f"- Documentation: <{SITE}/>",
        f"- Parity fixtures: <https://huggingface.co/datasets/{org}/{DATASET}>",
        f"- All jepa.cpp GGUFs: <https://huggingface.co/{org}>",
    ]
    return "\n".join(out) + "\n"


def render_dataset_card(ctx: dict) -> str:
    org = ctx["org"]
    dirs = sorted(p for p in (FIXTURES / "ref").iterdir() if p.is_dir())
    n_files = sum(1 for p in (FIXTURES / "ref").rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in (FIXTURES / "ref").rglob("*") if p.is_file())
    by_ref = {m["ref_dir"]: m for m in MODELS}
    nc = sorted({by_ref[d.name]["title"] for d in dirs
                 if by_ref.get(d.name) and ctx["info"][by_ref[d.name]["base"]]["license"] == "cc-by-nc-4.0"})

    out = [
        "---",
        "license: cc-by-nc-4.0",
        "pretty_name: jepa.cpp parity fixtures",
        "size_categories:",
        "  - n<1K",
        "tags:",
    ]
    for t in ["jepa", "jepa.cpp", "ggml", "gguf", "parity", "golden-references", "regression-testing"]:
        out.append(f"  - {t}")
    out += [
        "---",
        "",
        "# jepa.cpp parity fixtures",
        "",
        f"PyTorch golden reference dumps for [**jepa.cpp**]({REPO}), a ggml-based C/C++ inference engine for the",
        "JEPA family. `tests/test-parity` and `tests/test-predictor` replay these tensors through the engine and",
        "gate per-token cosine, pooled outputs and classifier top-1/top-5 against per-family thresholds.",
        "",
        f"Generated from jepa.cpp `main` @ [`{ctx['commit']}`]({REPO}/commit/{ctx['commit']}) with",
        "`scripts/dump_reference.py --model all`, in float32 eval mode on 32 CPU threads, no autocast.",
        "",
        "## Contents",
        "",
        "| directory | model | files | size |",
        "|---|---|---:|---:|",
    ]
    for d in dirs:
        files = [p for p in d.rglob("*") if p.is_file()]
        title = by_ref[d.name]["title"] if d.name in by_ref else d.name
        out.append(f"| `ref/{d.name}/` | {title} | {len(files)} | {sum(p.stat().st_size for p in files) / 1e6:.0f} MB |")
    out += [
        "",
        f"{n_files} files, {total / 1e6:.0f} MB in total.",
        "",
        "## Layout",
        "",
        "One directory per model, each with a `manifest.json` and one `.npy` per tensor per sample, named",
        "`<sample>.<tensor>.npy`. All arrays are float32 C-order except `frames_u8` (uint8) and `top5_idx`",
        "(int64). Shapes carry no batch dimension except `input`, which is stored exactly as fed to the model.",
        "",
        "The manifest records the model id, the hyper-parameters, the preprocessing pipeline that produced",
        "`input`, the PyTorch forward wall time per sample (`timing_s.forward_s`, the baseline for the speed",
        "tables of the jepa.cpp docs), the frame indices sampled from each clip, and the label strings for the",
        "classifier. Per-model tensor lists and the exact preprocessing recipe:",
        f"[**docs/fixtures.md**]({SITE}/fixtures/).",
        "",
        "## How jepa.cpp consumes it",
        "",
        "```bash",
        f"git clone --recursive {REPO} && cd jepa.cpp",
        "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        "",
        "scripts/download_fixtures.sh          # this dataset -> tests/fixtures/ref, plus the media it needs",
        f"scripts/download_models.sh small      # GGUFs from https://huggingface.co/{org}",
        "",
        "cmake -S . -B build && ctest --test-dir build      # the parity suites register at configure time",
        "```",
        "",
        "`test-parity` runs two passes per file: first the stored `input` tensor (bypassing preprocessing, so a",
        "graph bug shows up alone), then jepa.cpp's own preprocessor on the source media (so a preprocessing",
        "mismatch shows up separately). The second pass needs `tests/fixtures/media/`, which is **not** part of",
        "this dataset — see below.",
        "",
        "## Input provenance",
        "",
        "The dumps are model outputs computed on two public research sets. Neither the source images nor the",
        "source videos are redistributed here; only the reference activations and the decoded frame tensors the",
        "tests replay are.",
        "",
        "| input | source | fetched by |",
        "|---|---|---|",
        "| 8 images `coco_*.jpg` | [COCO val2017](https://cocodataset.org/), images subject to their original "
        "Flickr terms | tracked in the jepa.cpp git repo |",
        "| 6 short clips | [`nateraw/kinetics-mini`](https://huggingface.co/datasets/nateraw/kinetics-mini), a "
        "small sample of Kinetics-400 | `scripts/download_fixtures.sh` |",
        "",
        "The `input` and `frames_u8` arrays inside `ref/` are preprocessed pixels of those images and clips, kept",
        "because the parity tests must feed the network exactly the tensor PyTorch saw. If you hold rights in any",
        f"of the underlying material and want it removed, open an issue on [github.com/aselimc/jepa.cpp]({REPO}/issues).",
        "",
        "## Licence",
        "",
        "`cc-by-nc-4.0`, the most restrictive licence among the checkpoints whose outputs are stored here: the",
        f"dumps of {' and '.join(nc)} are outputs of CC BY-NC 4.0 models, the rest of MIT / Apache-2.0 ones.",
        "Research and non-commercial use.",
        "",
        "## Regenerating instead of downloading",
        "",
        "```bash",
        "scripts/download_fixtures.sh media                        # the source media",
        "scripts/download_models.sh --convert all                  # the source checkpoints",
        ".venv/bin/python scripts/dump_reference.py --model all    # ~1 min on 32 cores",
        "```",
        "",
        "## Links",
        "",
        f"- Code: <{REPO}>",
        f"- Documentation: <{SITE}/>",
        f"- GGUF models: <https://huggingface.co/{org}>",
    ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------


def build_context(org: str, need_fixtures: bool = False) -> dict:
    missing = [f"{m['base']}-{d}.gguf" for m in MODELS for d in DTYPES
               if not (GGUF_DIR / f"{m['base']}-{d}.gguf").exists()]
    if missing:
        sys.exit(f"missing {len(missing)} GGUF(s) under {GGUF_DIR}, e.g. {missing[0]} "
                 f"-- run scripts/download_models.sh --ftype all")
    if need_fixtures and not (FIXTURES / "ref").is_dir():
        sys.exit(f"missing {FIXTURES / 'ref'} -- run scripts/download_fixtures.sh")
    # the provenance of the numbers is the last commit that touched the artifacts a card quotes, not HEAD:
    # tying it to HEAD would make `check` report drift after every unrelated commit
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--abbrev=7", "--format=%h", "--",
         "docs/parity.md", "docs/quantization.md", "docs/accuracy.md", "tests/results", "scripts/convert.py"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not commit:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    paths = [GGUF_DIR / f"{m['base']}-{d}.gguf" for m in MODELS for d in DTYPES]
    digests = sha256_many(paths)
    ctx = dict(
        org=org,
        commit=commit,
        parity=read_parity(),
        quant=read_quant(),
        img=read_json("accuracy-image.json"),
        vid=read_json("accuracy-video.json"),
        ssv2=read_json("accuracy-ssv2.json"),
        bench=read_json("benchmarks.json"),
        gpu=read_json("benchmarks-gpu.json"),
        info={},
        sizes={},
        shas={},
    )
    for m in MODELS:
        b = m["base"]
        ctx["info"][b] = jepa_info(GGUF_DIR / f"{b}-f32.gguf")
        ctx["sizes"][b] = {d: (GGUF_DIR / f"{b}-{d}.gguf").stat().st_size for d in DTYPES}
        ctx["shas"][b] = {d: digests[GGUF_DIR / f"{b}-{d}.gguf"] for d in DTYPES}
    return ctx


def write_cards(ctx: dict, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for m in MODELS:
        p = out_dir / f"{m['base']}.md"
        p.write_text(render_card(m, ctx))
        written[m["base"]] = p
    p = out_dir / "fixtures.md"
    p.write_text(render_dataset_card(ctx))
    written[DATASET] = p
    return written


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


def resolve(names):
    if not names:
        return list(MODELS)
    out = []
    for n in names:
        m = BY_SHORT.get(n) or BY_BASE.get(n)
        if m is None:
            sys.exit(f"unknown model '{n}' -- one of {', '.join(sorted(BY_SHORT))}")
        out.append(m)
    return out


def cmd_cards(args):
    ctx = build_context(args.org, need_fixtures=True)
    for base, p in write_cards(ctx, args.out).items():
        print(f"{p}  ({p.stat().st_size} bytes)")


def cmd_upload(args):
    from huggingface_hub import HfApi

    api = HfApi()
    models = resolve(args.models)
    do_fixtures = args.fixtures or not args.models
    ctx = build_context(args.org, need_fixtures=do_fixtures)
    cards = write_cards(ctx, args.out)

    for m in models:
        repo_id = f"{args.org}/{m['base']}-GGUF"
        files = [f"{m['base']}-{d}.gguf" for d in DTYPES]
        size = sum(ctx["sizes"][m["base"]][d] for d in DTYPES)
        print(f"[{repo_id}] {len(files)} files, {size / 1e6:.0f} MB", flush=True)
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=str(cards[m["base"]]), path_in_repo="README.md",
                        repo_id=repo_id, commit_message="Update model card")
        api.upload_folder(folder_path=str(GGUF_DIR), path_in_repo="", repo_id=repo_id,
                          allow_patterns=files,
                          commit_message="Add " + "/".join(DTYPES) + " GGUFs")
        print(f"  done {repo_id}", flush=True)

    if do_fixtures:
        repo_id = f"{args.org}/{DATASET}"
        print(f"[{repo_id}] ref/ ({FIXTURES / 'ref'})", flush=True)
        if not args.dry_run:
            api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
            api.upload_file(path_or_fileobj=str(cards[DATASET]), path_in_repo="README.md",
                            repo_id=repo_id, repo_type="dataset", commit_message="Update dataset card")
            api.upload_folder(folder_path=str(FIXTURES / "ref"), path_in_repo="ref", repo_id=repo_id,
                              repo_type="dataset", commit_message="Add PyTorch golden reference dumps")
            print(f"  done {repo_id}", flush=True)


def cmd_check(args):
    import requests
    from huggingface_hub import HfApi

    api = HfApi()
    ctx = build_context(args.org, need_fixtures=True)
    cards = write_cards(ctx, args.out)
    drift = []
    n_ok = 0

    def remote_card(repo_id, repo_type="models"):
        url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}/raw/main/README.md"
        r = requests.get(url, timeout=60)
        return r.text if r.status_code == 200 else None

    for m in MODELS:
        base = m["base"]
        repo_id = f"{args.org}/{base}-GGUF"
        files = [f"{base}-{d}.gguf" for d in DTYPES]
        try:
            infos = {i.path: i for i in api.get_paths_info(repo_id, files, expand=True)}
        except Exception as exc:  # repo missing, network, auth
            drift.append(f"{repo_id}: cannot list ({exc.__class__.__name__})")
            continue
        for d, fn in zip(DTYPES, files):
            i = infos.get(fn)
            if i is None:
                drift.append(f"{repo_id}/{fn}: not published")
                continue
            remote = getattr(getattr(i, "lfs", None), "sha256", None)
            if remote is None:
                drift.append(f"{repo_id}/{fn}: no LFS sha256 on the hub")
            elif remote != ctx["shas"][base][d]:
                drift.append(f"{repo_id}/{fn}: sha256 {remote[:16]} != local {ctx['shas'][base][d][:16]}")
            elif i.size != ctx["sizes"][base][d]:
                drift.append(f"{repo_id}/{fn}: size {i.size} != local {ctx['sizes'][base][d]}")
            else:
                n_ok += 1
        rc = remote_card(repo_id)
        if rc is None:
            drift.append(f"{repo_id}/README.md: not published")
        elif rc != cards[base].read_text():
            drift.append(f"{repo_id}/README.md: differs from the generated card ({args.out}/{base}.md)")
        else:
            n_ok += 1

    repo_id = f"{args.org}/{DATASET}"
    d_ok = 0
    try:
        tree = [e for e in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True)
                if getattr(e, "size", None) is not None and e.path not in ("README.md", ".gitattributes")]
    except Exception as exc:
        tree = []
        drift.append(f"{repo_id}: cannot list ({exc.__class__.__name__})")
    local_files = {str(p.relative_to(FIXTURES)) for p in (FIXTURES / "ref").rglob("*") if p.is_file()}
    for e in tree:
        p = FIXTURES / e.path
        if not p.exists():
            drift.append(f"{repo_id}/{e.path}: published but not local")
            continue
        remote = getattr(getattr(e, "lfs", None), "sha256", None)
        if remote is None:  # small files are plain git blobs; compare the bytes
            body = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/{e.path}", timeout=60).content
            if hashlib.sha256(body).hexdigest() != sha256(p):
                drift.append(f"{repo_id}/{e.path}: content differs")
                continue
        elif remote != sha256(p):
            drift.append(f"{repo_id}/{e.path}: sha256 differs")
            continue
        d_ok += 1
    for missing in sorted(local_files - {e.path for e in tree}):
        drift.append(f"{repo_id}/{missing}: local but not published")
    rc = remote_card(repo_id, "dataset")
    if rc is None:
        drift.append(f"{repo_id}/README.md: not published")
    elif rc != cards[DATASET].read_text():
        drift.append(f"{repo_id}/README.md: differs from the generated card ({args.out}/fixtures.md)")
    else:
        d_ok += 1

    print(f"models:  {n_ok} of {len(MODELS) * (len(DTYPES) + 1)} checks pass "
          f"({len(MODELS)} repos x {len(DTYPES)} files + card)")
    print(f"dataset: {d_ok} of {len(local_files) + 1} checks pass ({len(local_files)} files + card)")
    if drift:
        print(f"\n{len(drift)} DRIFT:")
        for d in drift:
            print(f"  {d}")
        return 1
    print("\nno drift: every published file matches the local sha256 and size, and every card matches "
          "the artifacts it is generated from.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", default="jepacpp", help="Hugging Face organisation (default: jepacpp)")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "tmp" / "hf-cards",
                    help="where the generated cards go (default: tmp/hf-cards, git-ignored)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cards", help="write the model cards and the dataset card")
    up = sub.add_parser("upload", help="upload the GGUFs, the fixtures and the cards")
    up.add_argument("--models", nargs="*", default=None, metavar="NAME")
    up.add_argument("--fixtures", action="store_true", help="also (or only) upload the fixtures dataset")
    up.add_argument("--dry-run", action="store_true")
    sub.add_parser("check", help="compare the hub with the local files and the generated cards")
    args = ap.parse_args()
    return {"cards": cmd_cards, "upload": cmd_upload, "check": cmd_check}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
