"""Fixtures and the parity bars, shared by the test modules.

Everything the parity suite needs beyond the repository is git-ignored — the converted GGUF files
under ``models/gguf/`` and the PyTorch golden dumps under ``tests/fixtures/ref/`` (see
``scripts/download_models.sh``, ``scripts/convert.py`` and ``scripts/dump_reference.py``) — so a
missing asset skips a test rather than failing it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
GGUF_DIR = REPO / "models" / "gguf"
REF_DIR = REPO / "tests" / "fixtures" / "ref"
MEDIA_DIR = REPO / "tests" / "fixtures" / "media"


# --- the parity bars ---------------------------------------------------------------------------
# Copied from tests/test-parity.cpp POLICY[BK_CPU], which is where they are derived and justified
# (docs/parity.md). `lhs` gates the token map, `derived` the single-row tensors users consume.
# Only the cells this suite exercises are listed.
class Tier:
    def __init__(
        self,
        *,
        lhs_mean=None,
        lhs_med=None,
        lhs_min=None,
        derived_mean=None,
        derived_min=None,
        rel_max=None,
    ):
        self.lhs_mean, self.lhs_med, self.lhs_min = lhs_mean, lhs_med, lhs_min
        self.derived_mean, self.derived_min, self.rel_max = derived_mean, derived_min, rel_max


TIERS = {
    ("image", "f32"): Tier(lhs_mean=0.9999, lhs_min=0.9999, derived_min=0.9999, rel_max=1e-3),
    ("image", "f16"): Tier(lhs_mean=0.9999, lhs_min=0.99, derived_min=0.9995),
    ("video", "f32"): Tier(
        lhs_mean=0.9999, lhs_med=0.9999, lhs_min=0.9999, derived_min=0.9999, rel_max=1e-3
    ),
    ("video", "f16"): Tier(lhs_med=0.999, lhs_mean=0.99, derived_min=0.9995),
    # On a GPU there is no f32 tier (TF32 GEMMs, F16 flash K/V, one-pass ggml_norm variance), so an
    # f32 file is judged with its family's f16 bars — tests/test-parity.cpp, "The GPU dimension".
    ("image", "gpu"): Tier(lhs_mean=0.9999, lhs_min=0.99, derived_min=0.9995, rel_max=5e-2),
}


def rel_bound(base: float, n_rows: int) -> float:
    """test-parity.cpp's rel_bound: the f32 bound grows with the row count."""
    return base * max(1.0, (n_rows / 2048.0) ** 0.5)


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine of two ``[rows, dim]`` blocks, in float64 as the C tests do."""
    x = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])
    y = np.asarray(b, dtype=np.float64).reshape(-1, b.shape[-1])
    num = (x * y).sum(1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-30
    return num / den


def assert_within(tier: Tier, got: np.ndarray, ref: np.ndarray, what: str, *, derived: bool):
    got = np.atleast_2d(np.asarray(got, dtype=np.float32))
    ref = np.atleast_2d(np.asarray(ref, dtype=np.float32))
    assert got.shape == ref.shape, f"{what}: shape {got.shape} vs reference {ref.shape}"
    c = cosines(got, ref)
    rel = float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-30))
    report = (
        f"{what}: cos mean {c.mean():.7f} median {np.median(c):.7f} min {c.min():.7f} "
        f"rel {rel:.2e} over {got.shape[0]} rows"
    )
    if derived:
        if tier.derived_mean is not None:
            assert c.mean() >= tier.derived_mean, report
        if tier.derived_min is not None:
            assert c.min() >= tier.derived_min, report
    else:
        if tier.lhs_mean is not None:
            assert c.mean() >= tier.lhs_mean, report
        if tier.lhs_med is not None:
            assert np.median(c) >= tier.lhs_med, report
        if tier.lhs_min is not None:
            assert c.min() >= tier.lhs_min, report
    if tier.rel_max is not None:
        assert rel <= rel_bound(tier.rel_max, got.shape[0]), report
    return report


def assert_bit_identical(a: np.ndarray, b: np.ndarray, what: str):
    """Equality of the float32 *bit patterns*, which is what "bit for bit" has to mean."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    n_diff = int((a.view(np.uint32) != b.view(np.uint32)).sum())
    max_abs = float(np.abs(a - b).max()) if a.size else 0.0
    assert n_diff == 0, f"{what}: {n_diff}/{a.size} values differ, max|a-b| = {max_abs:g}"
    assert max_abs == 0.0, f"{what}: max|a-b| = {max_abs:g}"


# --- assets ------------------------------------------------------------------------------------
def gguf(name: str) -> pathlib.Path:
    p = GGUF_DIR / name
    if not p.is_file():
        pytest.skip(f"{p} is not there — run scripts/download_models.sh and scripts/convert.py")
    return p


def ref(model: str) -> pathlib.Path:
    p = REF_DIR / model
    if not (p / "manifest.json").is_file():
        pytest.skip(f"{p} has no reference dump — run scripts/dump_reference.py")
    return p


def media(name: str) -> pathlib.Path:
    p = MEDIA_DIR / name
    if not p.is_file():
        pytest.skip(f"{p} is not there — run scripts/download_fixtures.sh")
    return p


def tool(name: str) -> pathlib.Path:
    p = REPO / "build" / name
    if not p.is_file():
        pytest.skip(f"{p} is not built — cmake -S . -B build -G Ninja && cmake --build build")
    return p


@pytest.fixture(scope="session")
def threads() -> int:
    """A fixed thread count for every run, so the two sides never differ by scheduling."""
    return 8


@pytest.fixture(scope="session")
def run_tool(tmp_path_factory, threads):
    """Run one of the C tools and return its ``.npy`` output (plus the ``--json`` stats, if any)."""
    out_dir = tmp_path_factory.mktemp("tools")
    counter = [0]

    def _run(name: str, args: list[str], *, want_json: bool = False, want_logits: bool = False):
        counter[0] += 1
        stem = out_dir / f"{name}-{counter[0]}"
        npy = stem.with_suffix(".npy")
        cmd = [str(tool(name)), *args, "-o", str(npy), "-t", str(threads)]
        stats_path = stem.with_suffix(".json")
        logits_path = stem.with_name(stem.name + "-logits.npy")
        if want_json:
            cmd += ["--json", str(stats_path)]
        if want_logits:
            cmd += ["--logits", str(logits_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"{' '.join(cmd)} failed:\n{proc.stderr}"
        arr = np.load(npy)
        stats = json.loads(stats_path.read_text()) if want_json else None
        if want_logits:
            return arr, stats, np.load(logits_path)
        return arr, stats

    return _run
