"""Executable numpy specification of the "hjepa" GGUF graph (docs/gguf-schema.md "hjepa").

Every function reads a GGUF through selftest.Model and mirrors one entry point of the C API
(src/hjepa.cpp). Levels are 1-based. `selftest.py --hjepa-ref REFDIR` checks each against a dump of
the authors' PyTorch modules (scripts/hjepa_reference.py); it needs no checkpoint and no checkout.
"""
from __future__ import annotations

import math

import numpy as np


def _ops():
    from . import selftest as S   # late: selftest imports this module
    return S


def hp_level(m, key: str, k: int):
    return m.hp[f"jepa.hjepa.{key}"][k - 1]


def _lin(m, x, name):
    S = _ops()
    return S.linear(x, m.get(name + ".weight"), m.opt(name + ".bias"))


def _ln(m, x, name, eps):
    S = _ops()
    return S.layer_norm(x, m.get(name + ".weight"), m.get(name + ".bias"), eps)


def embed(m, cls: np.ndarray, proprio: np.ndarray | None = None) -> np.ndarray:
    """Level-1 states from the ViT CLS rows [T, Dv] (and z-scored proprio [T, P]) -> [T, D_1]."""
    S = _ops()
    eps = m.hp["jepa.hjepa.ln_eps"]
    kind = m.hp["jepa.hjepa.proj_kind"]
    if kind == "mlp":
        x = _lin(m, S.gelu_erf(_lin(m, cls, "lvl.1.enc.proj.0")), "lvl.1.enc.proj.2")
    elif kind == "layernorm":
        x = _ln(m, cls, "lvl.1.enc.proj_ln", eps)
    else:
        x = cls
    if hp_level(m, "enc_kind", 1) == "vit_proprio":
        h = _ln(m, _lin(m, proprio, "lvl.1.enc.proprio.0"), "lvl.1.enc.proprio.ln", eps)
        x = np.concatenate([x, _lin(m, S.gelu_erf(h), "lvl.1.enc.proprio.2")], axis=-1)
    return x


def lift(m, k: int, below: np.ndarray) -> np.ndarray:
    """States of level k from the states of level k-1 [T, D_{k-1}]: every stride-th row, encoded."""
    S = _ops()
    x = below[:: hp_level(m, "stride", k)]
    if hp_level(m, "enc_kind", k) == "identity":
        return x
    eps = m.hp["jepa.hjepa.ln_eps"]
    p = f"lvl.{k}.enc."
    x = _lin(m, S.gelu_erf(_lin(m, _ln(m, x, p + "in_norm", eps), p + "fc1")), p + "fc2")
    return _ln(m, x, p + "out_norm", eps) if (p + "out_norm.weight") in m else x


def act_embed(m, k: int, actions: np.ndarray) -> np.ndarray:
    """JEPA.action_embed: the level's actions [T, A_k] -> [T, E_k] (identity when the file has none)."""
    S = _ops()
    p = f"lvl.{k}.act_embed"
    if (p + ".0.weight") not in m:
        return actions
    return _lin(m, S.silu(_lin(m, actions, p + ".0")), p + ".2")


def act_chunks(below_emb: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """[T, E] -> ([T_k, stride, E] zero-padded, n_valid [T_k]): chunk T starts at row T*stride."""
    T = below_emb.shape[0]
    starts = np.arange(0, T, stride)
    chunks = np.zeros((len(starts), stride, below_emb.shape[1]), np.float32)
    n_valid = np.zeros(len(starts), np.int64)
    for i, s in enumerate(starts):
        n = min(stride, T - s)
        chunks[i, :n] = below_emb[s:s + n]
        n_valid[i] = n
    return chunks, n_valid


def act_encode(m, k: int, chunks: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    """The level's latent actions [T_k, A_k] from chunks [T_k, stride, E_{k-1}] of the level below's
    action embeddings; steps >= n_valid[t] of a chunk are padding (zero rows, masked as keys)."""
    S = _ops()
    eps = m.hp["jepa.hjepa.ln_eps"]
    p = f"lvl.{k}.act_enc."
    Tk, L, E = chunks.shape
    if hp_level(m, "act_enc_kind", k) == "mlp":
        x = _lin(m, np.maximum(_lin(m, chunks.reshape(Tk, L * E), p + "fc1"), 0), p + "fc2")
        return _ln(m, x, p + "out_norm", eps) if (p + "out_norm.weight") in m else x
    H = hp_level(m, "act_enc_n_head", k)
    pos = m.get(p + "pos")
    out = []
    for t in range(Tk):
        x = np.concatenate([pos[:1], _lin(m, chunks[t], p + "step") + pos[1:L + 1]], 0)   # CLS + L steps
        keep = np.concatenate([[True], np.arange(L) < n_valid[t]])
        d = x.shape[1]
        hd = d // H
        for i in range(hp_level(m, "act_enc_n_layer", k)):
            b = f"{p}blk.{i}."
            h = _ln(m, x, b + "ln1", eps)
            q, kk, v = np.split(_lin(m, h, b + "attn_qkv"), 3, axis=-1)
            q, kk, v = (z.reshape(-1, H, hd).transpose(1, 0, 2) for z in (q, kk, v))
            s = (q @ kk.transpose(0, 2, 1)) / math.sqrt(hd)
            s = np.where(keep[None, None, :], s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            a = ((s / s.sum(-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(-1, d)
            x = x + _lin(m, a, b + "attn_out")
            h = _ln(m, x, b + "ln2", eps)
            x = x + _lin(m, np.maximum(_lin(m, h, b + "ffn_up"), 0), b + "ffn_down")
        y = _ln(m, S.gelu_erf(_lin(m, x[0], p + "head.0")), p + "head.ln", eps)
        out.append(S.layer_norm(_lin(m, y, p + "head.2"), m.get(p + "out_norm.weight"), m.get(p + "out_norm.bias"), 1e-6))
    return np.stack(out)


def predict(m, k: int, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """states [T, D_k], the level's actions [T, A_k] -> [T, D_k]; row t = predicted state t+1 given 0..t."""
    S = _ops()
    eps, aeps = m.hp["jepa.hjepa.ln_eps"], m.hp["jepa.hjepa.adaln_eps"]
    H = hp_level(m, "pred_n_head", k)
    bare = hp_level(m, "pred_kind", k) == "adaln_bare"
    p = f"lvl.{k}.pred."
    T = states.shape[0]
    c = act_embed(m, k, actions)
    if bare:
        c = _lin(m, S.silu(_lin(m, c, p + "act_embed.0")), p + "act_embed.2")
    x = states + m.get(p + "pos_embed")[:T]
    if (p + "in_proj.weight") in m:
        x = _lin(m, x, p + "in_proj")
    if (p + "cond_proj.weight") in m:
        c = _lin(m, c, p + "cond_proj")
    cs = S.silu(c)
    for i in range(hp_level(m, "pred_n_layer", k)):
        b = f"{p}blk.{i}."
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = np.split(_lin(m, cs, b + "adaln"), 6, axis=-1)
        h = S.layer_norm(x, None, None, aeps) * (1 + sc_a) + sh_a
        if not bare:
            h = _ln(m, h, b + "ln1", eps)
        q, kk, v = np.split(_lin(m, h, b + "attn_qkv"), 3, axis=-1)
        x = x + g_a * _lin(m, S.attention(q, kk, v, H, causal=True), b + "attn_out")
        h = S.layer_norm(x, None, None, aeps) * (1 + sc_m) + sh_m
        if not bare:
            h = _ln(m, h, b + "ln2", eps)
        x = x + g_m * _lin(m, S.gelu_erf(_lin(m, h, b + "ffn_up")), b + "ffn_down")
    x = _ln(m, x, p + "norm", eps)
    if (p + "out_proj.weight") in m:
        x = _lin(m, x, p + "out_proj")
    if (p + "proj.0.weight") in m:
        x = _lin(m, S.gelu_erf(_lin(m, x, p + "proj.0")), p + "proj.2")
    return x


def rollout(m, k: int, seeds: np.ndarray, actions: np.ndarray, history: int) -> np.ndarray:
    """JEPA.rollout: seeds [n_seed, D_k], actions [n_seed + n - 1, A_k] -> the n predicted states.
    Step t reads the last `history` states (seeds, then its own predictions) and their actions."""
    seq = [s for s in seeds]
    n = actions.shape[0] - len(seq) + 1
    for _ in range(n):
        w = min(history, len(seq))
        hi = len(seq)
        seq.append(predict(m, k, np.stack(seq[hi - w:hi]), actions[hi - w:hi])[-1])
    return np.stack(seq[len(seeds):])
