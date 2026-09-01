"""The wrapper's own behaviour: lifetime, argument checking, and errors carrying the C text.

Nothing numeric is asserted here — that is ``test_parity.py``'s job. What this covers is the
marshalling layer: that a failed C call becomes an exception with the library's own message in it,
that a closed model stays closed, and that the arguments a caller can get wrong are rejected in
Python rather than reaching the engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import jepa_cpp
from conftest import gguf, media

MODEL = "lejepa-vits16-pretrain-in1k-f16.gguf"


@pytest.fixture(scope="module")
def model(threads):
    with jepa_cpp.Model(gguf(MODEL), threads=threads) as m:
        yield m


def test_library_identity():
    assert jepa_cpp.library_path().is_file()
    assert jepa_cpp.version()  # whatever JEPA_VERSION says
    assert jepa_cpp.__version__.startswith("0.1.0")


def test_a_failed_load_carries_the_c_message(tmp_path):
    missing = tmp_path / "not-a-model.gguf"
    with pytest.raises(jepa_cpp.JepaError) as exc:
        jepa_cpp.Model(missing)
    assert str(missing) in str(exc.value)
    assert "GGUF" in str(exc.value), "the exception should quote what the library logged"


def test_a_failed_encode_carries_the_c_message(model):
    # 100 is not a multiple of the patch size, which the encoder refuses.
    bad = np.zeros((3, 1, 100, 100), dtype=np.float32)
    with pytest.raises(jepa_cpp.JepaError) as exc:
        model.encode(bad)
    assert "patch size" in str(exc.value)


def test_the_error_text_does_not_leak_between_calls(model):
    with pytest.raises(jepa_cpp.JepaError):
        model.encode(np.zeros((3, 1, 100, 100), dtype=np.float32))
    # A good call afterwards must not resurrect the previous message.
    assert model.encode(media("coco_000000000139.jpg")).shape == (197, model.embed_dim)


def test_close_is_idempotent_and_final():
    m = jepa_cpp.Model(gguf(MODEL), threads=2)
    assert "Model" in repr(m)
    m.close()
    m.close()
    assert "closed" in repr(m)
    with pytest.raises(jepa_cpp.JepaError):
        m.encode(media("coco_000000000139.jpg"))


def test_pool_arguments_are_checked_in_python(model):
    with pytest.raises(ValueError):
        model.encode(media("coco_000000000139.jpg"), pool="nonsense")
    with pytest.raises(ValueError):
        model.pool(np.zeros((197, model.embed_dim), np.float32), "nonsense")


def test_a_model_without_a_head_or_a_predictor_says_so(model):
    assert not model.has_head and not model.has_predictor and not model.has_projector
    with pytest.raises(jepa_cpp.JepaError):
        model.classify(media("coco_000000000139.jpg"))
    with pytest.raises(ValueError):
        model.encode(media("coco_000000000139.jpg"), pool="lewm")


@pytest.mark.parametrize(
    "device,index",
    [("cpu", -1), ("CPU", -1), ("cuda:0", 0), ("gpu:3", 3), ("7", 7), (2, 2), (None, None)],
)
def test_device_strings(device, index):
    from jepa_cpp.model import _parse_device

    assert _parse_device(device) == index


def test_a_bad_device_string_is_rejected():
    from jepa_cpp.model import _parse_device

    with pytest.raises(ValueError):
        _parse_device("tpu:0")


def test_context_settings_round_trip(model):
    assert model.threads == 8
    previous = model.max_batch
    model.max_batch = 4
    assert model.max_batch == 4
    model.max_batch = previous
    assert model.max_batch == previous
    model.mul_mat_prec_f32 = False
    assert model.mul_mat_prec_f32 is False
    model.mul_mat_prec_f32 = True
    assert model.mul_mat_prec_f32 is True


def test_preprocess_params_are_the_files_own(model):
    p = model.preprocess_params
    assert p.crop == model.img_size
    assert p.rescale == pytest.approx(1.0 / 255.0)
    assert all(0.0 < v < 1.0 for v in p.mean)
    assert all(0.0 < v < 1.0 for v in p.std)


def test_load_image_and_preprocess_agree_with_encode(model):
    path = media("coco_000000000139.jpg")
    rgb = jepa_cpp.Model.load_image(path)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    x = model.preprocess(rgb)
    assert x.shape == (3, 1, model.img_size, model.img_size) and x.dtype == np.float32
    a = model.encode(x)
    b = model.encode(path)
    assert np.array_equal(a, b), "preprocess() then encode() must be encode()"


def test_encode_shapes(model):
    path = media("coco_000000000139.jpg")
    assert model.encode(path).shape == (197, 384)
    assert model.encode(path, pool="cls").shape == (384,)
    assert model.encode([path, path]).shape == (2, 197, 384)
    assert model.encode([path, path], pool="mean").shape == (2, 384)
    # An image family has no clips: a [T, H, W, 3] stack is T independent images.
    stack = np.stack([jepa_cpp.Model.load_image(path)] * 2)
    assert model.encode(stack).shape == (2, 197, 384)
    # Forcing the clip reading on one anyway is the header's documented behaviour, not an error:
    # an image family encodes every (batch, frame) slice on its own and concatenates the rows.
    assert model.encode(stack, as_video=True).shape == (2 * 197, 384)


def test_unreadable_inputs_raise_before_the_engine(model):
    with pytest.raises(TypeError):
        model.encode(42)
    with pytest.raises(ValueError):
        model.encode(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        model.encode([])
