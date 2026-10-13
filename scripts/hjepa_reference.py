#!/usr/bin/env python3
"""hjepa_reference.py — PyTorch reference activations for an H-JEPA checkpoint, from the authors' modules.

  scripts/hjepa_reference.py --ckpt models/jepa-world-models/h-jepa/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
      --hjepa-src tmp/hjepa/src [--out tests/fixtures/ref] [--media tests/fixtures/media]

writes tests/fixtures/ref/<run>-<seed>/{manifest.json, seq.*.npy}, the fixture `test-hjepa` and
`jepa_convert/selftest.py` compare against.

The released files are pickled module objects. Nothing is unpickled here: the tensors and the module
attributes are read with jepa_convert/torch_object.py (which runs none of the code a pickle names),
the same modules are then constructed from the authors' own classes (`--hjepa-src`, a checkout of
github.com/kevinghst/H-JEPA) and the tensors are loaded with `strict=True` — which proves the
rebuilt model has exactly the checkpoint's parameters. Every number below then comes out of the
authors' `JEPA.encode` / `JEPA.predict` / `JEPA.rollout`.

One sample, "seq": T1 level-1 steps, enough for 3 states at the top level.
  input          [T1, 3, H, W]   ImageNet-normalised frames (fixture images, squash-resized)
  proprio        [T1, P]         z-scored proprioception, N(0,1) draw (levels with a proprio stream only)
  last_hidden_state [1+N, Dv]    ViT tokens of frame 0 (CLS first)
  cls            [T1, Dv]        ViT CLS rows
  lvl{k}.state   [T_k, D_k]      the level's states s_k (level 1: JEPA.encode of the frames)
  lvl{k}.action  [T_k, A_k]      the level's actions (level 1: N(0,1) draw; above: action_encoder output)
  lvl{k}.act_emb [T_k, E_k]      action_embed(action): the predictor's conditioning, the level above's input
  lvl{k}.n_valid [T_k]           int64, level k >= 2: steps of each action chunk that exist (the rest is padding)
  lvl{k}.pred    [T, D_k]        JEPA.predict over the first T = min(T_k, n_frames) steps (teacher forced)
  lvl{k}.rollout_h{H} [n, D_k]   JEPA.rollout from state 0 with history_size H: the n predicted states

A second sample, "plan", is the authors' planner run on those states (plan_reference below):
  cost.L{k}.{own,upper,dense}.*  LevelCostModel.get_cost of fixed candidates and its autograd gradient,
                                 for a goal in the level's own space, one level up (with the action
                                 cost), and at the top level reached at stride 1
  solve.L{k}.{a,b,c}.*           GradientSolver.solve: plain; early-stopped with per-step noise;
                                 sigma-clipped. Every normal draw is stored, so the solve can be replayed.
  hplan.*                        HierarchicalSolver.solve at three points of a short episode, with the
                                 horizon schedule it derived
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_reference import RefWriter, image_to_tensor_imagenet, list_images  # noqa: E402
from jepa_convert.hjepa import default_name  # noqa: E402
from jepa_convert.torch_object import load_object, module_tree, root_attr  # noqa: E402

VIT_SIZES = {192: "tiny", 384: "small", 768: "base", 1024: "large", 1280: "huge"}


def build_model(ckpt: Path, hjepa_src: Path):
    """The authors' HJEPA, rebuilt from the checkpoint's own shapes and attributes, weights loaded strictly."""
    import torch
    from torch import nn

    sys.path.insert(0, str(hjepa_src / "h_jepa"))
    from models.encoders.hf_vit import create_hf_vit
    from models.encoders.seq_encoder import SequenceEncoder, SequenceMLPEncoder
    from models.hjepa import HJEPA
    from models.jepa import JEPA, FusionEncoder, ProjectedEncoder, ProjectedPredictor
    from models.module import MLP, Embedder, ResidualLatentMLP
    from models.predictors.predictors import ARPredictor, CausalTransformerPredictor

    obj = load_object(ckpt)
    T, M = module_tree(obj)

    def kind(n):
        return M[n].kind if n in M else "missing"

    def shape(k):
        return T[k].shape

    def projector(n):
        k = kind(n)
        if k == "Identity":
            return nn.Identity()
        if k == "LayerNorm":
            return nn.LayerNorm(shape(n + ".weight")[0])
        assert k == "MLP" and kind(n + ".net.1") == "BatchNorm1d", (n, k)
        h, i = shape(n + ".net.0.weight")
        return MLP(input_dim=i, hidden_dim=h, output_dim=shape(n + ".net.3.weight")[0], norm_fn=nn.BatchNorm1d)

    def vit(n):
        D, _, P, _ = shape(n + ".embeddings.patch_embeddings.projection.weight")
        img = M[n + ".embeddings"].attrs["image_size"][0]
        return create_hf_vit(size=VIT_SIZES[D], patch_size=P, image_size=img,
                             use_mask_token=(n + ".embeddings.mask_token") in T)

    def base_encoder(n):
        k = kind(n)
        if k == "ViTModel":
            return vit(n)
        if k == "Identity":
            return nn.Identity()
        if k == "ResidualLatentMLP":
            h, i = shape(n + ".fc1.weight")
            return ResidualLatentMLP(input_dim=i, hidden_dim=h, output_dim=shape(n + ".fc2.weight")[0],
                                     final_ln=kind(n + ".final_norm") == "LayerNorm")
        assert k == "MLP", (n, k)   # the proprio encoder: Linear -> LayerNorm -> GELU -> Linear
        h, i = shape(n + ".net.0.weight")
        return MLP(input_dim=i, hidden_dim=h, output_dim=shape(n + ".net.3.weight")[0])

    def encoder(n):
        if kind(n) == "FusionEncoder":
            return FusionEncoder(
                pixel_encoder=ProjectedEncoder(base_encoder(n + ".pixel_encoder.encoder"), projector(n + ".pixel_encoder.projector")),
                proprio_encoder=ProjectedEncoder(base_encoder(n + ".proprio_encoder.encoder"), projector(n + ".proprio_encoder.projector")),
                projector=projector(n + ".projector"), proprio_key=M[n].attrs.get("proprio_key", "proprio"))
        return ProjectedEncoder(base_encoder(n + ".encoder"), projector(n + ".projector"))

    def predictor(n, d_state):
        if kind(n) == "CausalTransformerPredictor":
            tr = n + ".transformer"
            D = shape(tr + ".norm.weight")[0]
            heads = M[tr + ".layers.0.attn"].attrs["heads"]
            return CausalTransformerPredictor(
                input_dim=d_state, action_dim=shape(n + ".action_embedder.0.weight")[1],
                depth=sum(1 for k in T if k.startswith(tr + ".layers.") and k.endswith(".attn.to_qkv.weight")),
                heads=heads, dim_head=shape(tr + ".layers.0.attn.to_qkv.weight")[0] // 3 // heads,
                mlp_ratio=shape(tr + ".layers.0.mlp.net.0.weight")[0] / D,
                max_seq_len=shape(n + ".pos_embedding")[1], predictor_dim=D)
        core, tr = n + ".predictor", n + ".predictor.transformer"
        D = shape(tr + ".norm.weight")[0]
        heads = M[tr + ".layers.0.attn"].attrs["heads"]
        d_out = shape(tr + ".output_proj.weight")[0] if (tr + ".output_proj.weight") in T else D
        ar = ARPredictor(
            num_frames=shape(core + ".pos_embedding")[1],
            depth=sum(1 for k in T if k.startswith(tr + ".layers.") and k.endswith(".attn.to_qkv.weight")),
            heads=heads, mlp_dim=shape(tr + ".layers.0.mlp.net.1.weight")[0], input_dim=d_state, hidden_dim=D,
            output_dim=d_out, dim_head=shape(tr + ".layers.0.attn.to_qkv.weight")[0] // 3 // heads)
        return ProjectedPredictor(ar, projector(n + ".projector"))

    def action_encoder(n, stride):
        k = kind(n)
        if k == "Identity":
            return None
        if k == "SequenceMLPEncoder":
            h, i = shape(n + ".net.net.0.weight")
            return SequenceMLPEncoder(output_dim=shape(n + ".net.net.3.weight")[0], hidden_dims=[h], input_dim=i // stride,
                                      temporal_stride=stride, final_ln=kind(n + ".net.final_norm") == "LayerNorm")
        assert k == "SequenceEncoder", (n, k)
        d, i = shape(n + ".step_mlp.fc1.weight")
        return SequenceEncoder(
            output_dim=shape(n + ".head.3.weight")[0], input_dim=i, d_model=d,
            nhead=M[n + ".encoder.layers.0.self_attn"].attrs["num_heads"],
            num_layers=sum(1 for k in T if k.startswith(n + ".encoder.layers.") and k.endswith(".self_attn.in_proj_weight")),
            ff_mult=shape(n + ".encoder.layers.0.linear1.weight")[0] // d, max_chunk=shape(n + ".pos")[0] - 1,
            norm_first=M[n + ".encoder.layers.0"].attrs["norm_first"])

    def action_embed(n):
        if kind(n) == "Identity":
            return nn.Identity()
        h, feat = shape(n + ".embed.0.weight")
        e = shape(n + ".embed.2.weight")[0]
        conv = (n + ".patch_embed.weight") in T
        return Embedder(input_dim=shape(n + ".patch_embed.weight")[1] if conv else feat, smoothed_dim=feat, emb_dim=e,
                        mlp_scale=h // e, patch_embed=conv)

    jepas, d_state = [], 0
    for i in range(len(M["jepas"].children)):
        j = f"jepas.{i}"
        a = M[j].attrs
        enc = encoder(j + ".encoder")
        if i == 0:
            if kind(j + ".encoder") == "FusionEncoder":
                d_state = shape(j + ".encoder.pixel_encoder.projector.net.3.weight")[0] + shape(j + ".encoder.proprio_encoder.encoder.net.3.weight")[0]
            elif kind(j + ".encoder.projector") == "MLP":
                d_state = shape(j + ".encoder.projector.net.3.weight")[0]
            else:
                d_state = shape(j + ".encoder.encoder.layernorm.weight")[0]
        elif kind(j + ".encoder.encoder") == "ResidualLatentMLP":
            d_state = shape(j + ".encoder.encoder.fc2.weight")[0]
        jepas.append(JEPA(
            encoder=enc, predictor=predictor(j + ".predictor", d_state), action_embed=action_embed(j + ".action_embed"),
            action_encoder=action_encoder(j + ".action_encoder", a["temporal_stride"]), level=a["level"],
            action_queue_size=a["action_queue_size"], temporal_stride=a["temporal_stride"],
            temporal_window_size=a["temporal_window_size"], target_length=a["target_length"]))
    model = HJEPA(jepas)
    # training-only attachments the rebuilt model does not have: the inverse-dynamics head, a probing decoder
    sd = {k: torch.from_numpy(np.array(v)) for k, v in T.items() if ".idm_loss_fn." not in k and "decoder_level" not in k}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, T, M, root_attr(obj, "normalizer_artifact") or {}


# The fields of jepa_hjepa_level_plan, in the order `plan.*.levels` stores them (one row per level).
PLAN_FIELDS = ("horizon", "pre_decay_horizon", "n_samples", "n_steps", "lr", "weight_decay", "var_scale", "action_noise",
               "action_clip_sigma", "min_its", "early_stop_patience", "early_stop_rel_delta", "cost_last_n",
               "intermediate_cost_weight", "action_cost_weight", "num_subgoals", "horizon_one_goal_upper")


def plan_reference(model, hjepa_src: Path, states: list, actions: list, frames, proprio) -> dict:
    """The authors' planner on this model: LevelCostModel costs with autograd gradients, GradientSolver
    solves and HierarchicalSolver calls, with every normal draw they make recorded.

    Returns {name: (array, layout)} for the "plan" sample; the layouts are the documentation.
    """
    import types
    from types import SimpleNamespace as NS

    import torch

    # The solver modules import gymnasium and hydra for type annotations and config instantiation,
    # neither of which a solve touches, and the stable_worldmodel package __init__ pulls in the
    # simulators. Empty stand-ins for the former, a bare package for the latter.
    for name in ("gymnasium", "hydra"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["gymnasium"].Space = object
    if "stable_worldmodel" not in sys.modules:
        pkg = types.ModuleType("stable_worldmodel")
        pkg.__path__ = [str(hjepa_src / "stable_worldmodel")]
        sys.modules["stable_worldmodel"] = pkg
    from hierarchical_solver import HierarchicalSolver, LevelCostModel
    from stable_worldmodel.solver.gd import GradientSolver

    N = model.num_levels
    jepas = list(model.jepas)

    def plain(x):   # the `seq` tensors were made under inference_mode, which autograd refuses to read
        return None if x is None else torch.from_numpy(x.numpy().copy())
    states, actions = [plain(s) for s in states], [plain(a) for a in actions]
    frames, proprio = plain(frames), plain(proprio)
    A = [a.shape[-1] for a in actions]
    out: dict[str, tuple] = {}
    g = torch.Generator().manual_seed(1)

    class Draws:   # records torch.randn while a solver runs
        def __enter__(self):
            self.rows, self.orig = [], torch.randn

            def randn(*args, **kw):
                x = self.orig(*args, **kw)
                self.rows.append(x.detach().reshape(-1).clone())
                return x
            torch.randn = randn
            return self

        def __exit__(self, *exc):
            torch.randn = self.orig

        def flat(self):
            return torch.cat(self.rows) if self.rows else torch.zeros(0)

    def cost_model(k):
        return LevelCostModel(model=jepas[k - 1], upper_models=jepas[k:])

    def info(k, S, goal, *, upper, goal_level=None, dense=False, goal_action=None, plan=None):
        d = {"pixels": torch.zeros(1, S, 1, 1), "embed_0": states[k - 1][:1].expand(1, S, 1, -1),
             "goal_embed_0": goal[None, None].expand(1, S, -1, -1), "goal_embed_0_in_upper_space": upper,
             "dense_upper_goal_projection": dense}
        if goal_level is not None:
            d["goal_embed_0_level"] = goal_level
        if goal_action is not None:
            d["goal_action"] = goal_action[None, None].expand(1, S, -1, -1)
        if plan is not None:
            d.update(cost_last_n=plan["cost_last_n"], intermediate_cost_weight=plan["intermediate_cost_weight"],
                     action_cost_weight=plan["action_cost_weight"])
        return d

    def prior(k):   # candidates drawn like the solver's initial ones: around the level's action prior
        q = jepas[k - 1].get_latent_action_queue()
        mean = q.mean(0) if q is not None else torch.zeros(A[k - 1])
        std = q.std(0, unbiased=False) if q is not None else torch.ones(A[k - 1])
        return mean, std

    def defaults(**kw):
        p = dict(horizon=1, pre_decay_horizon=0, n_samples=4, n_steps=10, lr=0.1, weight_decay=0.01, var_scale=1.0,
                 action_noise=0.0, action_clip_sigma=0.0, min_its=30, early_stop_patience=5, early_stop_rel_delta=1e-2,
                 cost_last_n=1, intermediate_cost_weight=1.0, action_cost_weight=0.0, num_subgoals=1, horizon_one_goal_upper=0)
        p.update(kw)
        return p

    def row(p):
        return [float(p[f]) for f in PLAN_FIELDS]

    # ---- 1. cost and gradient of fixed candidates, one case per goal placement
    cases = []
    for k in range(1, N + 1):
        cases.append((k, "own", k, False))
        if k < N:
            cases.append((k, "upper", k + 1, False))
            cases.append((k, "dense", N, True))
    for k, mode, gl, dense in cases:
        stride = int(jepas[k].temporal_stride) if k < N else 1
        S = 5
        if mode == "own":      # the level's own space: two scored states, the earlier one down-weighted
            T, plan = 3, defaults(cost_last_n=2, intermediate_cost_weight=0.5)
            goal, gact = states[k - 1][-1:], None
        elif mode == "upper":  # two waypoints of the level above and its two macro-actions to imitate
            T, plan = 2 * stride, defaults(cost_last_n=2, action_cost_weight=0.2)
            goal, gact = states[k][1:3], actions[k][0:2]
        else:                  # the real goal at the top level, every predicted step encoded up at stride 1
            T, plan = 3, defaults(cost_last_n=2)
            goal, gact = states[N - 1][-1:], None
        mean, std = prior(k)
        cand = (mean + std * torch.randn(1, S, T, A[k - 1], generator=g)).requires_grad_(True)
        cm = cost_model(k)
        costs = cm.get_cost(info(k, S, goal, upper=mode != "own", goal_level=gl if mode == "dense" else None, dense=dense,
                                 goal_action=gact, plan=plan), cand)
        costs.sum().backward()
        n = f"cost.L{k}.{mode}"
        out[f"{n}.meta"] = (torch.tensor([T, S, goal.shape[0], gl, int(dense)], dtype=torch.int64), "[5] horizon, samples, goals, goal level, dense")
        out[f"{n}.plan"] = (torch.tensor(row(plan)), f"[{len(PLAN_FIELDS)}] jepa_hjepa_level_plan fields: " + ", ".join(PLAN_FIELDS))
        out[f"{n}.goal"] = (goal, "[n_goal, D_goal_level]")
        if gact is not None:
            out[f"{n}.goal_action"] = (gact, "[n_goal, A_{k+1}]")
        out[f"{n}.actions"] = (cand.detach()[0], "[S, T, A_k] candidates")
        out[f"{n}.costs"] = (costs.detach()[0], "[S] LevelCostModel.get_cost")
        out[f"{n}.grad"] = (cand.grad[0], "[S, T, A_k] d sum(costs) / d actions (autograd)")

    # ---- solver plumbing
    def level_solver(k, p):
        s = GradientSolver(model=cost_model(k), n_steps=p["n_steps"], min_its=p["min_its"], batch_size=1,
                           var_scale=p["var_scale"], num_samples=p["n_samples"], action_noise=p["action_noise"],
                           early_stop_patience=p["early_stop_patience"], early_stop_rel_delta=p["early_stop_rel_delta"],
                           action_clip_sigma=p["action_clip_sigma"] or None, device="cpu", seed=100 + k,
                           optimizer_cls=torch.optim.AdamW, optimizer_kwargs={"lr": p["lr"], "weight_decay": p["weight_decay"]})
        return s

    def level_cfg(k, p, receding=1):
        return NS(horizon=p["horizon"], receding_horizon=receding, action_block=1, action_dim=A[k - 1],
                  pre_decay_horizon=p["pre_decay_horizon"] or None, num_subgoals=p["num_subgoals"], cost_last_n=p["cost_last_n"],
                  intermediate_cost_weight=p["intermediate_cost_weight"], action_cost_weight=p["action_cost_weight"],
                  horizon_one_goal_cost_space="upper" if p["horizon_one_goal_upper"] else "lower")

    space = NS(shape=(1, A[0]))

    # ---- 2. one level's solve, three flavours: plain, early-stopped with per-step noise, sigma-clipped
    for k in range(1, N + 1):
        stride = int(jepas[k].temporal_stride) if k < N else 1
        flavours = {
            "a": defaults(horizon=2 * stride if k < N else 3, n_samples=6, n_steps=12, cost_last_n=2,
                          action_cost_weight=0.2 if k < N else 0.0, lr=0.1 if k > 1 else 0.05),
            "b": defaults(horizon=3, n_samples=5, n_steps=40, min_its=4, early_stop_patience=2, action_noise=0.01,
                          cost_last_n=2, intermediate_cost_weight=0.5, lr=0.05),
            "c": defaults(horizon=2, n_samples=4, n_steps=8, var_scale=1.5, action_clip_sigma=2.0, lr=0.3, weight_decay=0.0),
        }
        for tag, p in flavours.items():
            upper = tag == "a" and k < N
            goal = states[k][1:3] if upper else states[k - 1][-1:]
            gact = actions[k][0:2] if upper else None
            solver = level_solver(k, p)
            solver.configure(action_space=space, n_envs=1, config=level_cfg(k, p), level=k)
            d = {"pixels": torch.zeros(1, 1, 1), "embed_0": states[k - 1][None, :1], "goal_embed_0": goal[None],
                 "goal_embed_0_in_upper_space": upper, "action_cost_weight": p["action_cost_weight"]}
            if gact is not None:
                d["goal_action"] = gact[None]
            with Draws() as draws:
                res = solver.solve(info_dict=d, planning_horizon=p["horizon"])
            best = res["actions"]                                    # [1, T, A]
            with torch.no_grad():
                final = cost_model(k).get_cost(info(k, 1, goal, upper=upper, goal_action=gact, plan=p), best[:, None])
            n = f"solve.L{k}.{tag}"
            out[f"{n}.plan"] = (torch.tensor(row(p)), "jepa_hjepa_level_plan fields")
            out[f"{n}.meta"] = (torch.tensor([goal.shape[0], k + 1 if upper else k, solver.solve_records[-1]["n_iters"][0]], dtype=torch.int64),
                                "[3] goals, goal level, iterations the reference ran")
            out[f"{n}.goal"] = (goal, "[n_goal, D]")
            if gact is not None:
                out[f"{n}.goal_action"] = (gact, "[n_goal, A_{k+1}]")
            out[f"{n}.noise"] = (draws.flat(), "every torch.randn value the solve drew, in order")
            out[f"{n}.actions"] = (best[0], "[T, A_k] the returned (lowest-cost) candidate")
            out[f"{n}.states"] = (res["predictions"]["predicted_embed_0"][0], "[T + 1, D_k] its rollout, the current state first")
            out[f"{n}.cost"] = (final.reshape(1), "[1] its cost")

    # ---- 3. the whole hierarchy, at three points of an episode
    if N > 1:
        strides = [1] + [int(j.temporal_stride) for j in jepas[1:]]
        # With the levels above skipped, the reference pools the candidates' actions at stride 1 before
        # finding there is nothing to compare them with. The MLP action encoder (DROID) has a fixed
        # chunk length and cannot do that, so its episode runs without the action cost.
        seq_pool = type(jepas[1].action_encoder).__name__ == "SequenceEncoder"
        plans = []
        for k in range(1, N + 1):
            top = k == N
            plans.append(defaults(
                horizon=3 if top else 2 * strides[k], pre_decay_horizon=0 if top else strides[k], n_samples=4, n_steps=8,
                lr=0.05 if k == 1 else 0.1, cost_last_n=2 if k == 1 else 1, action_cost_weight=0.2 if k == 1 and seq_pool else 0.0,
                horizon_one_goal_upper=1 if k > 1 else 0))
        receding, budget = 2, 24
        hs = HierarchicalSolver(level_solvers={k: level_solver(k, plans[k - 1]) for k in range(1, N + 1)}, level_models=jepas)
        cfg = NS(receding_horizon=receding, **{f"level{k}": level_cfg(k, plans[k - 1], receding) for k in range(1, N + 1)})
        hs.configure(action_space=space, n_envs=1, config=cfg)
        d = {"pixels": frames[None, :1], "goal": frames[None, -1:]}
        if proprio is not None:
            d["proprio"], d["goal_proprio"] = proprio[None, :1], proprio[None, -1:]
        schedules = hs._horizon_schedules(eval_budget=budget)
        calls = len(schedules[1])
        obs = hs._encode_all_levels(info_dict=d, key="pixels")
        goal = hs._encode_all_levels(info_dict=d, key="goal")
        out["hplan.levels"] = (torch.tensor([row(p) for p in plans]), f"[n_levels, {len(PLAN_FIELDS)}] configured plans, level 1 first")
        out["hplan.episode"] = (torch.tensor([receding, budget, calls], dtype=torch.int64), "[3] replanning interval, eval budget, planner calls")
        out["hplan.schedule"] = (torch.tensor([schedules[k] for k in range(1, N + 1)], dtype=torch.int64), "[n_levels, calls] horizon per call")
        for k in range(1, N + 1):
            out[f"hplan.state_L{k}"] = (obs[f"embed_{k}"][0, 0], "[D_k] the observation at level k")
            out[f"hplan.goal_L{k}"] = (goal[f"embed_{k}"][0, 0], "[D_k] the goal at level k")
        # the first call, one in the middle, and the first at which every upper level is down to 1
        last = next((c for c in range(calls) if all(schedules[k][c] == 1 for k in range(2, N + 1))), calls - 1)
        picks = sorted({0, calls // 2, last})
        out["hplan.calls"] = (torch.tensor(picks, dtype=torch.int64), "planner call indices dumped below")
        for c in picks:
            with Draws() as draws:
                res = hs.solve(dict(d), steps_taken=c * receding, eval_budget=budget)
            out[f"hplan.c{c}.noise"] = (draws.flat(), "every torch.randn value of the call, top level first")
            out[f"hplan.c{c}.actions"] = (res["actions"][0], "[horizon_1, A_1] the level-1 plan")
    return out


def main(argv=None) -> int:
    import torch

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path, help="<run>_object.ckpt from huggingface.co/jepa-world-models/h-jepa")
    ap.add_argument("--hjepa-src", required=True, type=Path, help="checkout of github.com/kevinghst/H-JEPA")
    ap.add_argument("--out", type=Path, default=root / "tests" / "fixtures" / "ref")
    ap.add_argument("--media", type=Path, default=root / "tests" / "fixtures" / "media")
    ap.add_argument("--name", default=None, help="fixture directory name (default: <run>-<seed>)")
    ap.add_argument("--top-steps", type=int, default=3, help="states at the top level (sets the level-1 length)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    t0 = time.time()
    model, T, M, norm = build_model(a.ckpt, a.hjepa_src)
    load_s = time.time() - t0
    n_levels = model.num_levels
    strides = [int(j.temporal_stride) for j in model.jepas]
    T1 = a.top_steps
    for s in reversed(strides[1:]):
        T1 = (T1 - 1) * s + 1
    j1 = model.jepas[0]
    fusion = hasattr(j1.encoder, "pixel_encoder")
    vit = j1.encoder.pixel_encoder.encoder if fusion else j1.encoder.encoder
    img = int(vit.config.image_size)
    name = a.name or default_name(a.ckpt)

    g = torch.Generator().manual_seed(0)
    emb1 = j1.action_embed
    A1 = emb1.patch_embed.in_channels if getattr(emb1, "patch_embed", None) is not None else \
        (emb1.embed[0].in_features if hasattr(emb1, "embed") else j1.predictor.action_embedder[0].in_features)
    actions = torch.randn(T1, A1, generator=g)
    P = j1.encoder.proprio_encoder.encoder.net[0].in_features if fusion else 0
    proprio = torch.randn(T1, P, generator=g) if fusion else None

    from PIL import Image
    files = list_images(a.media, 8)
    frames = []
    for t in range(T1):   # T1 distinct frames out of a handful of images: each pass rolls the picture further
        x = image_to_tensor_imagenet(Image.open(files[t % len(files)]), img)[0]
        frames.append(torch.roll(x, shifts=(t // len(files)) * (img // 8), dims=-1))
    pixels = torch.stack(frames)   # [T1, 3, H, W]

    try:
        commit = subprocess.run(["git", "-C", str(a.hjepa_src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    out: dict[str, tuple] = {}
    levels_meta = []
    plan_states, plan_actions = [], []
    t0 = time.time()
    with torch.inference_mode():
        tokens = vit(pixels, interpolate_pos_encoding=True).last_hidden_state   # [T1, 1+N, Dv]
        out["input"] = (pixels, "[T1, 3, H, W] normalised frames")
        out["last_hidden_state"] = (tokens[0], "[1+N_patches, Dv] frame 0, CLS first")
        out["cls"] = (tokens[:, 0], "[T1, Dv]")
        info = {"pixels": pixels[None], "action": actions[None]}
        if fusion:
            info["proprio"] = proprio[None]
            out["proprio"] = (proprio, "[T1, P] z-scored")
        enc = j1.encode(info)
        state, act_emb, act = enc["embed_0"], enc["action_0"], actions[None]
        if fusion:
            out["lvl1.pixel_embed"] = (enc["pixel_embed_0"][0], "[T1, pixel_dim]")
        for k in range(1, n_levels + 1):
            j = model.jepas[k - 1]
            if k > 1:
                # exactly HJEPA.encode_hierarchical's per-level step: chunk the level below, encode, pool the actions
                below = {"embed": state, "action": act_emb}
                chunked = j._chunk_temporal_info(dict(below), key="embed", stride=j.temporal_stride, window_size=j.temporal_window_size)
                mask = chunked["action_mask"]                         # [1, T_k, stride], True = padding
                act = j.action_encoder(chunked["action"], mask)       # [1, T_k, A_k] latent macro-actions
                enc = j.encode(dict(below), key="embed", chunk_temporal_inputs=True)
                state, act_emb2 = enc["embed_0"], enc["action_0"]
                assert torch.equal(act_emb2, j.action_embed(act))
                act_emb = act_emb2
                out[f"lvl{k}.n_valid"] = ((~mask[0]).sum(-1).to(torch.int64), "[T_k] valid steps per action chunk")
            Tk = state.shape[1]
            nf = j.predictor.predictor.pos_embedding.shape[1] if hasattr(j.predictor, "predictor") else j.predictor.pos_embedding.shape[1]
            Tp = min(Tk, nf)
            pred = j.predict(state[:, :Tp], act_emb[:, :Tp])
            plan_states.append(state[0].clone())
            plan_actions.append(act[0].clone())
            out[f"lvl{k}.state"] = (state[0], "[T_k, D_k]")
            out[f"lvl{k}.action"] = (act[0], "[T_k, A_k]")
            out[f"lvl{k}.act_emb"] = (act_emb[0], "[T_k, E_k]")
            out[f"lvl{k}.pred"] = (pred[0], "[min(T_k, n_frames), D_k] next-state prediction at every step")
            roll_h = sorted({1, min(nf, 3)})
            for H in roll_h:
                # JEPA.rollout: one seed state, Tk actions -> Tk predicted states, window = history_size
                r = j.rollout({"pixels": torch.zeros(1, 1, 1, 1), "embed_0": state[:, None, :1]}, act[:, None], history_size=H)
                out[f"lvl{k}.rollout_h{H}"] = (r["predicted_embed_0"][0, 0, 1:], "[T_k, D_k] states predicted from state 0")
            levels_meta.append({
                "level": k, "embed_dim": int(state.shape[-1]), "action_dim": int(act.shape[-1]), "act_embed_dim": int(act_emb.shape[-1]),
                "stride": int(j.temporal_stride), "window": int(j.temporal_window_size), "n_steps": int(Tk),
                "pred_n_frames": int(nf), "pred_steps": int(Tp), "rollout_history": roll_h,
                "encoder": type(j.encoder.encoder).__name__ if hasattr(j.encoder, "encoder") else type(j.encoder).__name__,
                "action_encoder": type(j.action_encoder).__name__, "predictor": type(j.predictor).__name__,
            })
    fwd = time.time() - t0

    stats = {c: {"mean": np.asarray(s["mean"]).reshape(-1).tolist(), "std": np.asarray(s["std"]).reshape(-1).tolist()}
             for c, s in (norm.get("stats") or {}).items() if c in ("action", "proprio")}
    w = RefWriter(
        a.out / name, name, a.ckpt,
        hf_id="jepa-world-models/h-jepa", ckpt_sha256=hashlib.sha256(a.ckpt.read_bytes()).hexdigest(),
        reference_code={"repo": "https://github.com/kevinghst/H-JEPA", "commit": commit,
                        "note": "modules built from the authors' classes with the checkpoint's own shapes; load_state_dict(strict=True)"},
        n_levels=n_levels, levels=levels_meta, img_size=img, proprio_dim=P, normalizer=stats,
        semantics={
            "frames": f"{T1} level-1 steps = fixture images squash-resized to {img}x{img} (bilinear, antialias, on uint8), ImageNet-normalised; "
                      "later passes over the image list are rolled horizontally so every frame differs",
            "actions": "level 1: N(0,1) draw, torch.Generator(0) (the model's inputs are z-scored, frameskip-stacked raw actions); "
                       "level k >= 2: action_encoder over the stride-long chunk of the level below's act_emb starting at the chunk's state, "
                       "zero-padded past the end of the stream (n_valid)",
            "state": "level 1: JEPA.encode(pixels[, proprio]); level k >= 2: encoder_k(state_{k-1}[::stride])",
            "pred": "JEPA.predict(state[:T], act_emb[:T]): row t is the predicted state t+1 given steps 0..t (causal)",
            "rollout": "JEPA.rollout(seed = state[0], actions = action[0:T_k], history_size = H): T_k predicted states; "
                       "step t conditions on action t and on the last H states (seed, then its own predictions)",
        })
    w.add("seq", [f.name for f in files], out, {"forward_s": round(fwd, 4)})
    t0 = time.time()
    plan = plan_reference(model, a.hjepa_src, plan_states, plan_actions, pixels, proprio)
    w.add("plan", None, plan, {"forward_s": round(time.time() - t0, 4)},
          note="the authors' planner (LevelCostModel / GradientSolver / HierarchicalSolver) on the states of `seq`")
    w.finish(load_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
