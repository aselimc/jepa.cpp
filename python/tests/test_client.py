"""``jepa_cpp.client`` against a real ``jepa-server``.

The server is started for the module on a loopback port it picks itself (``--port 0``, whose number
comes back on stdout), served a small model, and shut down afterwards. The load-bearing assertion is
the same one ``tests/test-server.cpp`` makes from C++ and ``test_parity.py`` makes for the bindings:
what comes back over HTTP is the float32 bit pattern ``jepa-embed`` writes for the same file. What
is checked here beyond that is the *client* — input normalization, both encodings, batching through
one request, and the exception a refused request raises.

World-model rollouts live in ``test_rollout_client`` below and use the small LeWM file; the V-JEPA
2-AC planner is a ViT-g and belongs to the load test, not to a suite that has to stay fast.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time

import numpy as np
import pytest

from conftest import assert_bit_identical, gguf, media, tool
from jepa_cpp.client import Client, ServerError, clip

MODEL = "lejepa-vits16-pretrain-in1k-f16.gguf"
IMAGES = [
    "coco_000000000139.jpg",
    "coco_000000000285.jpg",
    "coco_000000000632.jpg",
    "coco_000000000724.jpg",
]
# The server and jepa-embed must agree on this: on the CPU concurrent and batched encodes are
# bit-identical to serial ones *at the same n_threads*, and only at the same n_threads.
THREADS = 8


def _wait_for_banner(proc, timeout=60.0):
    """The port out of the server's first stdout line, which it prints once it is bound."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"jepa-server exited with {proc.returncode}")
            continue
        m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
        if m:
            return int(m.group(1))
    raise RuntimeError("jepa-server never printed its port")


@pytest.fixture(scope="module")
def server(request):
    """One jepa-server for the module, on a port it chose, with the client pointed at it."""
    model = gguf(MODEL)
    exe = tool("jepa-server")
    proc = subprocess.Popen(
        [
            str(exe),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--threads",
            str(THREADS),
            "--workers",
            "2",
            "--max-batch",
            "8",
            "--max-wait-ms",
            "20",
            "--model-name",
            "lejepa-s",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        port = _wait_for_banner(proc)
        yield Client(f"http://127.0.0.1:{port}", timeout=120.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - only if the server wedges
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()


@pytest.fixture(scope="module")
def cli_rows(run_tool):
    """``jepa-embed --pool cls`` over the four images: [4, 384], one row per image."""
    args = ["-m", str(gguf(MODEL))]
    for name in IMAGES:
        args += ["-i", str(media(name))]
    rows, _ = run_tool("jepa-embed", [*args, "--pool", "cls", "--as-images", "--print-n", "0"])
    return rows


# --- health and introspection --------------------------------------------------------------------
def test_health_describes_the_loaded_model(server):
    h = server.health()
    assert h["status"] == "ok"
    assert h["model"] == "lejepa-s"
    assert h["family"] == "hfvit"
    assert h["backend"] == "CPU"
    assert h["threads"] == THREADS
    assert h["workers"] == 2
    assert h["max_batch"] == 8
    assert h["allow_local_files"] is False


def test_models_lists_the_one_model(server):
    models = server.models()
    assert len(models) == 1
    assert models[0]["id"] == "lejepa-s"
    assert models[0]["object"] == "model"
    assert models[0]["embed_dim"] == 384


def test_metrics_is_prometheus_text(server):
    server.embed(media(IMAGES[0]))
    text = server.metrics()
    assert "# TYPE jepa_requests_total counter" in text
    assert "jepa_request_duration_seconds_bucket" in text
    assert "jepa_batch_size_count" in text
    assert 'jepa_requests_total{endpoint="/v1/embeddings",status="200"}' in text


# --- the bit-identity claim -----------------------------------------------------------------------
@pytest.mark.parametrize("encoding_format", ["base64", "float"])
def test_embed_is_bit_identical_to_jepa_embed(server, cli_rows, encoding_format):
    got = server.embed([media(n) for n in IMAGES], pool="cls", encoding_format=encoding_format)
    assert got.shape == cli_rows.shape == (len(IMAGES), 384)
    assert_bit_identical(got, cli_rows, f"server /v1/embeddings ({encoding_format}) vs jepa-embed")


def test_one_request_of_four_matches_four_requests(server):
    """Dynamic batching is a scheduling decision, never a numeric one."""
    batched = server.embed([media(n) for n in IMAGES], pool="cls")
    one_by_one = np.stack([server.embed(media(n), pool="cls")[0] for n in IMAGES])
    assert_bit_identical(batched, one_by_one, "one batched request vs four single ones")


def test_both_encodings_carry_the_same_bits(server):
    a = server.embed(media(IMAGES[0]), encoding_format="base64")
    b = server.embed(media(IMAGES[0]), encoding_format="float")
    assert_bit_identical(a, b, "base64 vs float encoding_format")


def test_bytes_and_path_inputs_agree(server):
    from_path = server.embed(media(IMAGES[0]))
    from_bytes = server.embed(media(IMAGES[0]).read_bytes())
    assert_bit_identical(from_path, from_bytes, "a path read here vs the same bytes")


def test_pool_none_returns_every_token(server):
    tokens = server.embed(media(IMAGES[0]), pool="none")
    assert tokens.shape == (1, 197, 384)


def test_pool_mean_differs_from_cls(server):
    cls = server.embed(media(IMAGES[0]), pool="cls")
    mean = server.embed(media(IMAGES[0]), pool="mean")
    assert cls.shape == mean.shape
    assert not np.array_equal(cls, mean)


def test_usage_reports_the_token_count(server):
    res = server.embed_response([media(n) for n in IMAGES[:2]])
    assert res["object"] == "list"
    assert res["model"] == "lejepa-s"
    assert res["usage"]["total_tokens"] == 2 * 197


def test_clip_sends_one_item(server):
    """An image model encodes each frame of a clip on its own, so a two-frame clip is two rows."""
    res = server.embed_response([clip([media(IMAGES[0]), media(IMAGES[1])])], pool="none")
    assert len(res["data"]) == 1
    assert res["usage"]["total_tokens"] == 2 * 197


# --- errors ---------------------------------------------------------------------------------------
def test_classify_without_a_head_raises(server):
    with pytest.raises(ServerError) as e:
        server.classify(media(IMAGES[0]))
    assert e.value.status == 400
    assert "no classification head" in str(e.value)


def test_unknown_model_name_raises(server):
    with pytest.raises(ServerError) as e:
        server.embed(media(IMAGES[0]), model="not-this-one")
    assert e.value.status == 404


def test_bytes_that_are_not_an_image_raise(server):
    with pytest.raises(ServerError) as e:
        server.embed(b"this is not a PNG")
    assert e.value.status == 400
    assert "decode" in str(e.value)


def test_a_local_path_is_refused_by_default(server):
    with pytest.raises(ServerError) as e:
        server.embed({"path": "/etc/passwd"})
    assert e.value.status == 400
    assert "--allow-local-files" in str(e.value)


def test_rollout_on_a_model_without_a_predictor_raises(server):
    with pytest.raises(ServerError) as e:
        server.rollout(media(IMAGES[0]), [[[0.0] * 10]])
    assert e.value.status == 400
    assert "world model" in str(e.value)


def test_an_unreachable_server_raises(unused_port):
    with pytest.raises(ServerError) as e:
        Client(f"http://127.0.0.1:{unused_port}", timeout=5.0).health()
    assert "cannot reach" in str(e.value)
    assert e.value.status is None


@pytest.fixture(scope="module")
def unused_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def test_bad_json_body_is_a_400_and_the_server_survives(server):
    """The client cannot send this, so it goes out by hand — and /health must still answer."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        server.base_url + "/v1/embeddings",
        data=b"{ this is not json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        with urllib.request.urlopen(req, timeout=30):  # pragma: no cover - always raises
            pass
    body = json.loads(e.value.read())
    e.value.close()
    assert e.value.code == 400
    assert "not valid JSON" in body["error"]["message"]
    assert server.health()["status"] == "ok"


# --- /rollout, on the small world model -----------------------------------------------------------
@pytest.fixture(scope="module")
def lewm_server():
    model = gguf("lewm-pusht-f32.gguf")
    exe = tool("jepa-server")
    proc = subprocess.Popen(
        [
            str(exe),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--threads",
            str(THREADS),
            "--workers",
            "1",
            "--model-name",
            "lewm",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        port = _wait_for_banner(proc)
        yield Client(f"http://127.0.0.1:{port}", timeout=120.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()


def test_rollout_scores_every_candidate_and_step(lewm_server):
    actions = [
        [[0.1] * 10, [0.2] * 10],
        [[-0.1] * 10, [-0.2] * 10],
        [[0.0] * 10, [0.0] * 10],
    ]
    res = lewm_server.rollout(media(IMAGES[0]), actions, goal=media(IMAGES[1]))
    assert res["object"] == "rollout"
    assert res["n_candidates"] == 3
    assert res["horizon"] == 2
    energies = np.asarray(res["energies"], dtype=np.float32)
    assert energies.shape == (3, 2)
    assert np.all(energies > 0)
    # `best` is the candidate with the lowest energy at the final step.
    assert res["best"]["index"] == int(np.argmin(energies[:, -1]))
    assert res["best"]["energy"] == pytest.approx(float(energies[:, -1].min()), rel=1e-6)


def test_rollout_without_a_goal_scores_against_the_last_observed_frame(lewm_server):
    res = lewm_server.rollout(media(IMAGES[0]), [[[0.1] * 10]])
    assert np.asarray(res["energies"]).shape == (1, 1)


def test_rollout_latents_round_trip(lewm_server):
    from jepa_cpp.client import latents

    dim = lewm_server.health()["embed_dim"]
    res = lewm_server.rollout(media(IMAGES[0]), [[[0.1] * 10, [0.2] * 10]], return_latents=True)
    z = latents(res)
    assert z.shape == (1, 2, 1, dim)
    assert z.dtype == np.float32
    assert np.isfinite(z).all()


def test_a_wrong_action_width_is_refused(lewm_server):
    with pytest.raises(ServerError) as e:
        lewm_server.rollout(media(IMAGES[0]), [[[0.1] * 7]])
    assert e.value.status == 400
    assert "10 numbers" in str(e.value)


def test_plan_is_refused_on_lewm(lewm_server):
    with pytest.raises(ServerError) as e:
        lewm_server.rollout(media(IMAGES[0]), goal=media(IMAGES[1]), plan={"samples": 4})
    assert e.value.status == 400
    assert "AC world model" in str(e.value)
