"""Parity: the bindings against the C tools, and both against the PyTorch golden dumps.

Two questions, kept apart:

* **Is the Python path the same code?** ``build/jepa-embed`` and ``build/jepa-worldmodel`` are run
  as subprocesses and their ``.npy`` output is compared to the bindings' arrays *bit for bit* —
  identical float32 bit patterns, ``max|a-b| == 0.0``. Anything less would mean Python is doing
  arithmetic it should be delegating. This is a CPU claim; a GPU is bit-reproducible only against
  itself, and the GPU test below uses the GPU tier accordingly.
* **Is the engine still right?** The same arrays are compared to
  ``tests/fixtures/ref/<model>/*.npy`` — the PyTorch reference — with the bars
  ``tests/test-parity.cpp`` uses per family and dtype, reproduced in ``conftest.py``. Those
  comparisons run on the reference's own stored ``input.npy``, so what is being measured is the
  graph and not the JPEG decoder.

Assets (GGUF files, golden dumps, built tools) are git-ignored, so a missing one skips.
"""

from __future__ import annotations

import numpy as np
import pytest

import jepa_cpp
from conftest import (
    TIERS,
    assert_bit_identical,
    assert_within,
    gguf,
    media,
    ref,
)

IMAGE = "coco_000000000139.jpg"
IMAGES = [
    "coco_000000000139.jpg",
    "coco_000000000285.jpg",
    "coco_000000000632.jpg",
    "coco_000000000724.jpg",
]


def stored_input(path) -> np.ndarray:
    """A reference ``input.npy`` as one preprocessed item, ``[C, T, H, W]``.

    The dumps are ``[1, 3, H, W]`` for images and ``[1, 3, T, H, W]`` for clips.
    """
    a = np.load(path).astype(np.float32)
    if a.ndim == 4:  # [N, C, H, W] — one frame
        a = a[:, :, None]
    assert a.ndim == 5 and a.shape[0] == 1, a.shape
    return np.ascontiguousarray(a[0])


@pytest.fixture(scope="module")
def lejepa(threads):
    with jepa_cpp.Model(gguf("lejepa-vits16-pretrain-in1k-f16.gguf"), threads=threads) as m:
        yield m


@pytest.fixture(scope="module")
def lejepa_f32(threads):
    with jepa_cpp.Model(gguf("lejepa-vits16-pretrain-in1k-f32.gguf"), threads=threads) as m:
        yield m


@pytest.fixture(scope="module")
def vjepa21(threads):
    with jepa_cpp.Model(gguf("vjepa2_1-vitb-384-f16.gguf"), threads=threads) as m:
        yield m


@pytest.fixture(scope="module")
def lewm(threads):
    with jepa_cpp.Model(gguf("lewm-pusht-f16.gguf"), threads=threads) as m:
        yield m


# ===============================================================================================
# LeJEPA ViT-S/16 — image
# ===============================================================================================
def test_lejepa_introspection(lejepa):
    assert lejepa.family == "hfvit"
    assert (lejepa.embed_dim, lejepa.patch_size, lejepa.img_size) == (384, 16, 224)
    assert lejepa.has_cls and not lejepa.is_video
    assert lejepa.file_type_name == "f16"
    assert lejepa.token_grid(1, 224, 224) == (197, 1, 14, 14)


def test_lejepa_tokens_are_bit_identical_to_jepa_embed(lejepa, run_tool):
    got = lejepa.encode(media(IMAGE))
    want, _ = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("lejepa-vits16-pretrain-in1k-f16.gguf")),
            "-i",
            str(media(IMAGE)),
            "--pool",
            "none",
            "--print-n",
            "0",
        ],
    )
    assert got.shape == (197, 384)
    assert_bit_identical(got, want, "LeJEPA f16 tokens vs jepa-embed --pool none")


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_lejepa_pooled_is_bit_identical_to_jepa_embed(lejepa, run_tool, pool):
    got = lejepa.encode(media(IMAGE), pool=pool)
    want, stats = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("lejepa-vits16-pretrain-in1k-f16.gguf")),
            "-i",
            str(media(IMAGE)),
            "--pool",
            pool,
            "--print-n",
            "0",
        ],
        want_json=True,
    )
    assert stats["family"] == lejepa.family
    assert stats["file_type"] == lejepa.file_type_name
    assert stats["dim"] == lejepa.embed_dim
    assert stats["n_clips"] == 1
    assert_bit_identical(got, want[0], f"LeJEPA f16 --pool {pool} vs jepa-embed")


def test_lejepa_f16_against_the_golden_dump(lejepa):
    d = ref("lejepa-vits16")
    sample = IMAGE.replace(".jpg", "")
    tokens = lejepa.encode(stored_input(d / f"{sample}.input.npy"))
    tier = TIERS[("image", "f16")]
    print(
        assert_within(
            tier,
            tokens,
            np.load(d / f"{sample}.last_hidden_state.npy"),
            "LeJEPA f16 last_hidden_state",
            derived=False,
        )
    )
    print(
        assert_within(
            tier,
            lejepa.pool(tokens, "cls"),
            np.load(d / f"{sample}.cls.npy"),
            "LeJEPA f16 cls",
            derived=True,
        )
    )
    print(
        assert_within(
            tier,
            lejepa.pool(tokens, "mean"),
            np.load(d / f"{sample}.pooled_mean.npy"),
            "LeJEPA f16 pooled_mean",
            derived=True,
        )
    )


def test_lejepa_f32_against_the_golden_dump(lejepa_f32):
    """The f32 tier is the tight one — it also gates the relative error, which cosine cannot see."""
    d = ref("lejepa-vits16")
    sample = IMAGE.replace(".jpg", "")
    tokens = lejepa_f32.encode(stored_input(d / f"{sample}.input.npy"))
    tier = TIERS[("image", "f32")]
    print(
        assert_within(
            tier,
            tokens,
            np.load(d / f"{sample}.last_hidden_state.npy"),
            "LeJEPA f32 last_hidden_state",
            derived=False,
        )
    )
    print(
        assert_within(
            tier,
            lejepa_f32.pool(tokens, "cls"),
            np.load(d / f"{sample}.cls.npy"),
            "LeJEPA f32 cls",
            derived=True,
        )
    )


# ===============================================================================================
# Batching
# ===============================================================================================
@pytest.mark.parametrize("model", ["lejepa", "lejepa_f32"])
def test_a_batch_equals_encoding_one_at_a_time(request, model):
    m = request.getfixturevalue(model)
    paths = [media(n) for n in IMAGES]
    batched = m.encode(paths)
    assert batched.shape == (len(paths), 197, m.embed_dim)
    assert m.last_batch == len(paths), "the four items should have shared one graph"
    for i, p in enumerate(paths):
        assert_bit_identical(
            batched[i], m.encode(p), f"{m.file_type_name} item {i} batched vs alone"
        )


def test_a_batch_is_bit_identical_to_jepa_embed(lejepa, run_tool):
    paths = [media(n) for n in IMAGES]
    got = lejepa.encode(paths, pool="cls")
    args = ["-m", str(gguf("lejepa-vits16-pretrain-in1k-f16.gguf"))]
    for p in paths:
        args += ["-i", str(p)]
    want, _ = run_tool("jepa-embed", [*args, "--pool", "cls", "--print-n", "0"])
    assert_bit_identical(got, want, "LeJEPA f16 batch of 4 --pool cls vs jepa-embed")


def test_max_batch_one_gives_the_same_rows(lejepa):
    """`max_batch = 1` is the pre-batching path; the header promises the same rows on the CPU."""
    paths = [media(n) for n in IMAGES]
    grouped = lejepa.encode(paths)
    previous = lejepa.max_batch
    try:
        lejepa.max_batch = 1
        one_by_one = lejepa.encode(paths)
        assert lejepa.last_batch == 1
    finally:
        lejepa.max_batch = previous
    assert_bit_identical(grouped, one_by_one, "batched graph vs one graph per item")


# ===============================================================================================
# V-JEPA 2.1 ViT-B — a still image and a 16-frame clip
# ===============================================================================================
def test_vjepa21_introspection(vjepa21):
    assert vjepa21.family == "vjepa2_1"
    assert vjepa21.is_video and vjepa21.tubelet_size == 2
    assert vjepa21.embed_dim == 768
    assert vjepa21.max_batch == 1, "video models default to one clip per call, as jepa-embed does"
    assert vjepa21.token_grid(16, 384, 384) == (4608, 8, 24, 24)
    assert vjepa21.token_grid(1, 384, 384) == (576, 1, 24, 24), "the 2.1 image tokenizer"


def test_vjepa21_image_is_bit_identical_to_jepa_embed(vjepa21, run_tool):
    got = vjepa21.encode(media(IMAGE))
    want, _ = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("vjepa2_1-vitb-384-f16.gguf")),
            "-i",
            str(media(IMAGE)),
            "--pool",
            "none",
            "--print-n",
            "0",
        ],
    )
    assert got.shape == (576, 768)
    assert_bit_identical(got, want, "V-JEPA 2.1 f16 image tokens vs jepa-embed")


def test_vjepa21_image_against_the_golden_dump(vjepa21):
    d = ref("vjepa2_1-vitb-384")
    sample = IMAGE.replace(".jpg", "")
    tokens = vjepa21.encode(stored_input(d / f"{sample}.input.npy"))
    tier = TIERS[("video", "f16")]
    print(
        assert_within(
            tier,
            tokens,
            np.load(d / f"{sample}.last_hidden_state.npy"),
            "V-JEPA 2.1 f16 image last_hidden_state",
            derived=False,
        )
    )
    print(
        assert_within(
            tier,
            vjepa21.pool(tokens, "mean"),
            np.load(d / f"{sample}.pooled_mean.npy"),
            "V-JEPA 2.1 f16 image pooled_mean",
            derived=True,
        )
    )


def test_vjepa21_clip_is_bit_identical_to_jepa_embed(vjepa21, run_tool):
    d = ref("vjepa2_1-vitb-384")
    frames = np.load(d / "archery_f16.frames_u8.npy")
    assert frames.shape == (16, 360, 480, 3) and frames.dtype == np.uint8
    got = vjepa21.encode(frames)
    want, _ = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("vjepa2_1-vitb-384-f16.gguf")),
            "--frames-npy",
            str(d / "archery_f16.frames_u8.npy"),
            "--pool",
            "none",
            "--print-n",
            "0",
        ],
    )
    assert got.shape == (4608, 768)
    assert_bit_identical(got, want, "V-JEPA 2.1 f16 16-frame clip vs jepa-embed --frames-npy")

    pooled = vjepa21.pool(got, "mean")
    want_pooled, _ = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("vjepa2_1-vitb-384-f16.gguf")),
            "--frames-npy",
            str(d / "archery_f16.frames_u8.npy"),
            "--pool",
            "mean",
            "--print-n",
            "0",
        ],
    )
    assert_bit_identical(pooled, want_pooled[0], "V-JEPA 2.1 f16 clip --pool mean vs jepa-embed")


def test_vjepa21_clip_against_the_golden_dump(vjepa21):
    d = ref("vjepa2_1-vitb-384")
    tokens = vjepa21.encode(stored_input(d / "archery_f16.input.npy"))
    tier = TIERS[("video", "f16")]
    print(
        assert_within(
            tier,
            tokens,
            np.load(d / "archery_f16.last_hidden_state.npy"),
            "V-JEPA 2.1 f16 clip last_hidden_state",
            derived=False,
        )
    )
    print(
        assert_within(
            tier,
            vjepa21.pool(tokens, "mean"),
            np.load(d / "archery_f16.pooled_mean.npy"),
            "V-JEPA 2.1 f16 clip pooled_mean",
            derived=True,
        )
    )


def test_a_clip_from_frames_matches_the_stored_preprocessed_input(vjepa21):
    """The bindings' preprocessing is the library's: same pixels in, same tensor out."""
    d = ref("vjepa2_1-vitb-384")
    frames = np.load(d / "archery_f16.frames_u8.npy")
    got = vjepa21.preprocess(frames)
    assert_bit_identical(
        got,
        stored_input(d / "archery_f16.input.npy"),
        "own preprocessing of the stored frames vs the reference input",
    )


# ===============================================================================================
# The masked predictor (V-JEPA 2 ViT-L)
# ===============================================================================================
def test_masked_predictor_against_the_golden_dump(threads):
    d = ref("vjepa2-vitl-fpc64-256")
    enc = np.load(d / "archery_f16.last_hidden_state.npy")
    want = np.load(d / "archery_f16.predictor_last_hidden_state.npy")
    ids = np.arange(enc.shape[0], dtype=np.int32)
    with jepa_cpp.Model(gguf("vjepa2-vitl-fpc64-256-f16.gguf"), threads=threads) as m:
        assert m.has_predictor
        got = m.predict(enc, target=ids, context=ids, mask_index=1, modality="video")
    assert got.shape == want.shape
    print(
        assert_within(
            TIERS[("video", "f16")],
            got,
            want,
            "V-JEPA 2 f16 predictor_last_hidden_state",
            derived=False,
        )
    )


def test_the_modality_argument_reaches_the_predictor(vjepa21):
    """V-JEPA 2.1's image and video modality vectors are not interchangeable (docs/parity.md:
    the wrong one costs two digits of cosine), so the two calls must differ."""
    d = ref("vjepa2_1-vitb-384")
    enc = np.load(d / f"{IMAGE.replace('.jpg', '')}.last_hidden_state.npy")
    ids = np.arange(enc.shape[0], dtype=np.int32)
    as_image = vjepa21.predict(enc, target=ids, context=ids, modality="image")
    as_video = vjepa21.predict(enc, target=ids, context=ids, modality="video")
    # 2.1's predictor projects into the ViT-G teacher's width, so out_dim is not embed_dim.
    assert as_image.shape == as_video.shape == (enc.shape[0], as_image.shape[1])
    assert as_image.shape[1] != vjepa21.embed_dim
    assert not np.array_equal(as_image, as_video)
    # AUTO picks IMAGE for ids spanning a single temporal slice when the file has an image vector.
    assert_bit_identical(
        vjepa21.predict(enc, target=ids, context=ids, modality="auto"),
        as_image,
        "modality='auto' on a single temporal slice vs modality='image'",
    )


# ===============================================================================================
# The attentive-pool head (V-JEPA 2 ViT-L, Something-Something v2)
# ===============================================================================================
@pytest.fixture(scope="module")
def ssv2(threads):
    with jepa_cpp.Model(gguf("vjepa2-vitl-fpc16-256-ssv2-f16.gguf"), threads=threads) as m:
        yield m


def test_classify_matches_jepa_embed_and_the_golden_dump(ssv2, run_tool):
    d = ref("vjepa2-vitl-fpc16-256-ssv2")
    assert ssv2.has_head and ssv2.n_classes == 174
    assert len(ssv2.labels) == 174 and all(ssv2.labels)

    frames = np.load(d / "archery_f16.frames_u8.npy")
    got = ssv2.classify(frames)
    _, _, want_logits = run_tool(
        "jepa-embed",
        [
            "-m",
            str(gguf("vjepa2-vitl-fpc16-256-ssv2-f16.gguf")),
            "--frames-npy",
            str(d / "archery_f16.frames_u8.npy"),
            "--pool",
            "mean",
            "--print-n",
            "0",
        ],
        want_logits=True,
    )
    assert_bit_identical(got.logits, want_logits[0], "SSv2 f16 logits vs jepa-embed --logits")

    tier = TIERS[("video", "f16")]
    print(
        assert_within(
            tier, got.logits, np.load(d / "archery_f16.logits.npy"), "SSv2 f16 logits", derived=True
        )
    )
    print(
        assert_within(
            tier,
            got.pooled,
            np.load(d / "archery_f16.pooled.npy"),
            "SSv2 f16 pooler output",
            derived=True,
        )
    )
    top5 = [i for i, _label, _p in got.top(5)]
    want_top5 = list(np.load(d / "archery_f16.top5_idx.npy"))
    assert top5[0] == want_top5[0], f"top-1 {top5[0]} vs reference {want_top5[0]}"
    assert len(set(top5) & set(want_top5)) >= 4, f"top-5 {top5} vs reference {want_top5}"
    assert abs(float(got.probs.sum()) - 1.0) < 1e-5
    assert got.top(3) == got.top(5)[:3]


# ===============================================================================================
# LeWorldModel — projector, predictor and rollout
# ===============================================================================================
def test_lewm_projector_and_predictor_against_the_golden_dump(threads):
    d = ref("lewm-pusht")
    with jepa_cpp.Model(gguf("lewm-pusht-f32.gguf"), threads=threads) as m:
        assert m.has_projector and (m.lewm_n_frames, m.lewm_action_dim) == (3, 10)
        tier = TIERS[("image", "f32")]
        sample = IMAGE.replace(".jpg", "")
        emb = m.encode(stored_input(d / f"{sample}.input.npy"), pool="lewm")
        print(
            assert_within(tier, emb, np.load(d / f"{sample}.emb.npy"), "LeWM f32 emb", derived=True)
        )

        one = m.lewm_predict(emb, np.load(d / f"{sample}.action.npy"))
        print(
            assert_within(
                tier,
                one[-1],
                np.load(d / f"{sample}.pred_next.npy"),
                "LeWM f32 pred_next",
                derived=True,
            )
        )

        seq = m.lewm_predict(np.load(d / "seq.emb_seq.npy"), np.load(d / "seq.action_seq.npy"))
        print(
            assert_within(
                tier,
                seq,
                np.load(d / "seq.pred_seq.npy"),
                "LeWM f32 pred_seq (3 causal frames)",
                derived=True,
            )
        )


def test_lewm_rollout_is_bit_identical_to_jepa_worldmodel(lewm, run_tool, tmp_path):
    steps = 8
    actions = (
        np.random.default_rng(0).standard_normal((steps, lewm.lewm_action_dim)).astype(np.float32)
    )
    # atof() reads a double and narrows to float, so repr() of the exactly-representable double
    # round-trips the float32 without loss.
    spec = ";".join(",".join(repr(float(v)) for v in row) for row in actions)

    emb = lewm.encode(media(IMAGE), pool="lewm")
    got = lewm.lewm_rollout(emb, actions, steps)
    assert got.shape == (steps, lewm.embed_dim)

    want, _ = run_tool(
        "jepa-worldmodel",
        ["-m", str(gguf("lewm-pusht-f16.gguf")), "--image", str(media(IMAGE)), "--actions", spec],
    )
    assert_bit_identical(got, want, "LeWM f16 8-step rollout vs jepa-worldmodel")


def test_lewm_rollout_grows_the_window_causally(lewm):
    """A rollout is the predictor fed its own output: step 0 has to equal one predictor call."""
    actions = np.random.default_rng(1).standard_normal((3, lewm.lewm_action_dim)).astype(np.float32)
    emb = lewm.encode(media(IMAGE), pool="lewm")
    rollout = lewm.lewm_rollout(emb, actions, 3)
    first = lewm.lewm_predict(emb[None], actions[:1])
    assert_bit_identical(rollout[0], first[-1], "rollout step 0 vs a single lewm_predict")


# ===============================================================================================
# GPU
# ===============================================================================================
gpu = pytest.mark.skipif(
    not jepa_cpp.devices(),
    reason="the loaded library has no GPU device — build one with -DJEPA_CUDA=ON "
    "(cmake -S python -B build-py-cuda -DJEPA_CUDA=ON) and point $JEPA_CPP_LIB at it",
)


@gpu
def test_gpu_encoder_agrees_with_the_cpu(threads):
    """A GPU is not bit-reproducible against a CPU by construction (TF32 GEMMs, F16 flash K/V,
    a one-pass ggml_norm variance), so this is the GPU tier, not bit-identity."""
    path = gguf("lejepa-vits16-pretrain-in1k-f16.gguf")
    with jepa_cpp.Model(path, device="cpu", threads=threads) as cpu:
        want = cpu.encode(media(IMAGE))
        want_cls = cpu.pool(want, "cls")
    with jepa_cpp.Model(path, device="cuda:0", threads=threads) as gpu_model:
        assert gpu_model.is_gpu and gpu_model.device == 0
        assert gpu_model.device_name.startswith("CUDA")
        assert gpu_model.mul_mat_prec_f32, "GPU contexts default to GGML_PREC_F32"
        got = gpu_model.encode(media(IMAGE))
        got_cls = gpu_model.pool(got, "cls")
    tier = TIERS[("image", "gpu")]
    print(assert_within(tier, got, want, "LeJEPA f16 tokens CUDA0 vs CPU", derived=False))
    print(assert_within(tier, got_cls, want_cls, "LeJEPA f16 cls CUDA0 vs CPU", derived=True))
