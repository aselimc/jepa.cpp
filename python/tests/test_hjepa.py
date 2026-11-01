"""H-JEPA through the bindings: the authors' reference dumps, and ``build/jepa-plan``.

``test-predictor --hjepa`` already gates the engine against these dumps. What is tested here is the
marshalling: argument order, struct layout, shapes and the noise callback. So one f32 file with a
reference dump is enough, and the bars are the C test's. Missing assets skip.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

import jepa_cpp
from conftest import cosines, gguf, media, ref, tool

MODEL = "ant_hjepa_l3-seed42"
IMAGE = "coco_000000000139.jpg"
GOAL = "coco_000000000285.jpg"

# Field order of a `plan.*.plan` / `plan.hplan.levels` row: PLAN_FIELDS of scripts/hjepa_reference.py.
PLAN_FIELDS = [
    "horizon",
    "pre_decay_horizon",
    "n_samples",
    "n_steps",
    "lr",
    "weight_decay",
    "var_scale",
    "action_noise",
    "action_clip_sigma",
    "min_its",
    "early_stop_patience",
    "early_stop_rel_delta",
    "cost_last_n",
    "intermediate_cost_weight",
    "action_cost_weight",
    "num_subgoals",
    "horizon_one_goal_upper",
]
# test-predictor's bar for a replayed solve: tens of AdamW steps fed back into themselves.
SOLVE_BAR = (0.999, float("inf"))
INT_FIELDS = {
    "horizon",
    "pre_decay_horizon",
    "n_samples",
    "n_steps",
    "min_its",
    "early_stop_patience",
    "cost_last_n",
    "num_subgoals",
    "horizon_one_goal_upper",
}


def plan_of(row):
    return jepa_cpp.hjepa_level_plan(
        **{f: int(v) if f in INT_FIELDS else float(v) for f, v in zip(PLAN_FIELDS, row)}
    )


class Replay:
    """The reference's normal draws, handed back in the order it made them."""

    def __init__(self, noise):
        self.noise = np.asarray(noise, dtype=np.float32).ravel()
        self.pos = 0

    def __call__(self, n: int) -> np.ndarray:
        out = self.noise[self.pos : self.pos + n]
        assert out.size == n, "the planner asked for more draws than the reference made"
        self.pos += n
        return out

    @property
    def exact(self) -> bool:
        return self.pos == self.noise.size


def close(got, want, what: str, cos_min: float = 0.9999, rel_max: float = 1e-3) -> None:
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    assert got.shape == want.shape, f"{what}: shape {got.shape} vs {want.shape}"
    c = float(cosines(got.reshape(want.shape[0], -1), want.reshape(want.shape[0], -1)).min())
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    assert c >= cos_min and rel <= rel_max, f"{what}: cos min {c:.7f}, rel {rel:.2e}"


@pytest.fixture(scope="module")
def d():
    return ref(MODEL)


@pytest.fixture(scope="module")
def model(threads):
    with jepa_cpp.Model(gguf(f"{MODEL}-f32.gguf"), threads=threads) as m:
        yield m


def seq(d, level: int, what: str) -> np.ndarray:
    return np.load(d / f"seq.lvl{level}.{what}.npy")


def test_introspection_matches_the_reference_manifest(model, d):
    import json

    manifest = json.loads((d / "manifest.json").read_text())
    assert model.family == "hjepa"
    assert model.hjepa_n_levels == manifest["n_levels"]
    assert model.hjepa_proprio_dim == manifest["proprio_dim"]
    for lv in manifest["levels"]:
        k = lv["level"]
        assert model.hjepa_embed_dim(k) == lv["embed_dim"]
        assert model.hjepa_action_dim(k) == lv["action_dim"]
        assert model.hjepa_action_embed_dim(k) == lv["act_embed_dim"]
        assert model.hjepa_stride(k) == lv["stride"]
        assert model.hjepa_n_frames(k) == lv["pred_n_frames"]


def test_norm_stats_and_prior(model):
    mean, std = model.hjepa_norm_stats("action")
    assert mean.shape == std.shape == (model.hjepa_raw_action_dim,)
    assert (std > 0).all()
    assert model.hjepa_norm_stats("no-such-column") is None
    prior = model.hjepa_action_prior(2)
    assert prior.ndim == 2 and prior.shape[1] == model.hjepa_action_dim(2)


def test_embed_against_the_dump(model, d):
    states = model.hjepa_embed(np.load(d / "seq.cls.npy"), np.load(d / "seq.proprio.npy"))
    close(states, seq(d, 1, "state"), "embed")


def test_a_model_that_reads_proprio_refuses_to_go_without(model, d):
    with pytest.raises(ValueError, match="proprio"):
        model.hjepa_embed(np.load(d / "seq.cls.npy"))


@pytest.mark.parametrize("level", [2, 3])
def test_lift_and_action_encode_against_the_dump(model, d, level):
    close(model.hjepa_lift(level, seq(d, level - 1, "state")), seq(d, level, "state"), "lift")
    close(
        model.hjepa_action_encode(level, seq(d, level - 1, "act_emb")),
        seq(d, level, "action"),
        "action_encode",
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_action_embed_predict_and_rollout_against_the_dump(model, d, level):
    st, act = seq(d, level, "state"), seq(d, level, "action")
    close(model.hjepa_action_embed(level, act), seq(d, level, "act_emb"), "action_embed")

    want = seq(d, level, "pred")
    t = want.shape[0]
    close(model.hjepa_predict(level, st[:t], act[:t]), want, "predict")

    for history in (1, 3):
        want = seq(d, level, f"rollout_h{history}")
        n = want.shape[0]
        got = model.hjepa_rollout(level, st[:1], act[:n], n, history=history)
        close(got, want, f"rollout history {history}")


def test_rollout_checks_the_action_count(model):
    dim, adim = model.hjepa_embed_dim(1), model.hjepa_action_dim(1)
    with pytest.raises(ValueError, match="need 3 actions"):
        model.hjepa_rollout(1, np.zeros((1, dim)), np.zeros((2, adim)), 3)


def test_a_bad_level_carries_the_c_message(model):
    with pytest.raises(jepa_cpp.JepaError) as exc:
        model.hjepa_lift(99, np.zeros((1, model.hjepa_embed_dim(1))))
    assert "level" in str(exc.value)


@pytest.mark.parametrize(
    "level,mode", [(k, m) for k in (1, 2, 3) for m in ("own", "upper", "dense")]
)
def test_plan_cost_and_gradient_against_autograd(model, d, level, mode):
    b = f"plan.cost.L{level}.{mode}."
    if not (d / f"{b}meta.npy").is_file():
        pytest.skip(f"no {mode} goal at level {level}")
    meta = np.load(d / f"{b}meta.npy")
    ga = d / f"{b}goal_action.npy"
    actions = np.load(d / f"{b}actions.npy")
    actions = actions.reshape(int(meta[1]), int(meta[0]), -1)
    args = (level, seq(d, level, "state")[0], np.load(d / f"{b}goal.npy").reshape(int(meta[2]), -1),
            actions)
    kw = dict(
        plan=plan_of(np.load(d / f"{b}plan.npy").ravel()),
        goal_level=int(meta[3]),
        dense=bool(meta[4]),
        goal_actions=np.load(ga).reshape(int(meta[2]), -1) if ga.is_file() else None,
    )
    costs, grad = model.hjepa_plan_cost(*args, **kw)
    close(costs[None], np.load(d / f"{b}costs.npy").ravel()[None], "costs")
    close(grad, np.load(d / f"{b}grad.npy").reshape(grad.shape), "gradient")
    assert model.hjepa_plan_cost(*args, **kw, want_grad=False)[1] is None


def test_solve_replays_the_reference_on_its_own_noise(model, d):
    solved = 0
    for level in (1, 2, 3):
        for tag in ("a", "b"):
            b = f"plan.solve.L{level}.{tag}."
            if not (d / f"{b}meta.npy").is_file():
                continue
            meta = np.load(d / f"{b}meta.npy")
            ga = d / f"{b}goal_action.npy"
            noise = Replay(np.load(d / f"{b}noise.npy"))
            got = model.hjepa_solve(
                level,
                seq(d, level, "state")[0],
                np.load(d / f"{b}goal.npy").reshape(int(meta[0]), -1),
                plan_of(np.load(d / f"{b}plan.npy").ravel()),
                goal_level=int(meta[1]),
                goal_actions=np.load(ga).reshape(int(meta[0]), -1) if ga.is_file() else None,
                randn=noise,
            )
            close(got.actions.ravel()[None], np.load(d / f"{b}actions.npy").ravel()[None],
                  f"solve L{level}.{tag} actions", *SOLVE_BAR)
            close(got.states, np.load(d / f"{b}states.npy"), f"solve L{level}.{tag} states",
                  *SOLVE_BAR)
            assert noise.exact, f"solve L{level}.{tag}: the noise was not consumed exactly"
            solved += 1
    assert solved, "the dump has no solves"


def test_hierarchical_plan_replays_the_reference(model, d):
    n = model.hjepa_n_levels
    rows = np.load(d / "plan.hplan.levels.npy")
    cfg = [plan_of(rows[k]) for k in range(n)]
    interval, budget, n_calls = (int(x) for x in np.load(d / "plan.hplan.episode.npy")[:3])
    schedule = np.load(d / "plan.hplan.schedule.npy")
    for c in range(n_calls):
        got = model.hjepa_horizons(cfg, interval, budget, c * interval)
        assert got == [int(schedule[k][c]) for k in range(n)], f"call {c}"

    states = [np.load(d / f"plan.hplan.state_L{k}.npy").ravel() for k in range(1, n + 1)]
    goals = [np.load(d / f"plan.hplan.goal_L{k}.npy").ravel() for k in range(1, n + 1)]
    for c in np.load(d / "plan.hplan.calls.npy"):
        c = int(c)
        horizons = model.hjepa_horizons(cfg, interval, budget, c * interval)
        levels = [plan_of(rows[k]) for k in range(n)]
        for p, h in zip(levels, horizons):
            p.horizon = h
        noise = Replay(np.load(d / f"plan.hplan.c{c}.noise.npy"))
        got = model.hjepa_plan(states, goals, levels, randn=noise)
        want = np.load(d / f"plan.hplan.c{c}.actions.npy").reshape(got.actions.shape)
        close(got.actions.ravel()[None], want.ravel()[None], f"plan call {c}", *SOLVE_BAR)
        assert noise.exact, f"plan call {c}: the noise was not consumed exactly"
        assert len(got.planned) == len(got.cost) == n


def test_an_exception_in_randn_comes_back_out(model):
    states = [np.zeros(model.hjepa_embed_dim(k), dtype=np.float32) for k in (1, 2, 3)]
    levels = model.hjepa_default_levels()
    for p in levels:
        p.n_steps = 1

    def broken(n):
        raise KeyError("no draws today")

    with pytest.raises(KeyError, match="no draws today"):
        model.hjepa_plan(states, states, levels, randn=broken)


@pytest.mark.parametrize("name", [f"{MODEL}-f32.gguf", f"{MODEL}-f16.gguf"])
def test_plan_matches_jepa_plan(threads, name):
    """Default horizons, default proprio, the fixed-seed generator: the same call as the tool's."""
    obs, goal = media(IMAGE), media(GOAL)
    cmd = [str(tool("jepa-plan")), "-m", str(gguf(name)), "--obs", str(obs), "--goal", str(goal),
           "--quiet", "--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    with jepa_cpp.Model(gguf(name), threads=threads) as m:
        got = m.hjepa_plan(m.hjepa_states(obs), m.hjepa_states(goal)).actions
    printed = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in got)
    assert printed == proc.stdout.strip(), "the bindings and jepa-plan planned differently"


def test_a_closed_model_stays_closed(threads):
    m = jepa_cpp.Model(gguf(f"{MODEL}-f16.gguf"), threads=threads)
    m.close()
    with pytest.raises(Exception):
        m.hjepa_n_levels


def test_level_plan_rejects_unknown_fields():
    with pytest.raises(TypeError, match="no field"):
        jepa_cpp.hjepa_level_plan(horizonn=3)
    assert jepa_cpp.hjepa_level_plan(horizon=7).horizon == 7
    assert jepa_cpp.hjepa_level_plan().n_samples == 100
