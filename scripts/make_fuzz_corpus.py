#!/usr/bin/env python3
"""make_fuzz_corpus.py — seed corpus for tests/fuzz/fuzz-gguf-load.cpp.

The fuzz target feeds bytes to `jepa_model_load`, so a useful seed is a file that reaches deep into
`src/jepa-gguf.cpp` before it is rejected. Three kinds are written, all from the two small GGUFs
(`lewm-pusht`, `lejepa-vits16`) so the corpus needs no new checkpoint:

  mini-<model>.gguf     the same metadata and the same tensor NAMES, with every hparam and every
                        tensor shrunk consistently (D=8, ffn=16, patch=2, img=8, 2 layers). A few
                        kilobytes, and it loads: the fuzzer therefore mutates a file that otherwise
                        runs the loader end to end, shape checks and tensor wiring included.
  trunc-<...>.gguf      truncations of the real file and of the mini at the header / KV / tensor-info
                        boundaries and at a spread of offsets in between.
  kv-<key>-<what>.gguf  the mini with one metadata key mutated: zero, negative, absurd, an unknown
                        enum string, or the wrong GGUF value type.

The shrink is a per-namespace substitution over dimension VALUES (192 -> 8, 576 -> 24, ...), built
from the source file's own hparams; the script refuses to run if two roles of one namespace share a
dimension value, which is what would make that substitution ambiguous.

    scripts/make_fuzz_corpus.py                       # -> tests/fuzz/corpus/
    scripts/make_fuzz_corpus.py --out DIR --models A.gguf B.gguf

--models takes any GGUF of any family, so a run that should reach the video encoder, the 3-D RoPE
tables or the masked predictor can seed itself from `vjepa2_1-vitb-384` and `levjepa-vitl16`; the
default is the two small image models, which is what the committed workflow uses.

Needs `gguf` (gguf-py): `pip install gguf`.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

try:
    import gguf
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    sys.exit(f"make_fuzz_corpus.py needs numpy and gguf (pip install gguf): {exc}")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODELS = [
    ROOT / "models/gguf/lewm-pusht-f32.gguf",
    ROOT / "models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf",
]

# The shrunk model. Keep n_head a divisor of D and patch_size a divisor of img_size, or the mini
# file would be refused by the very checks it exists to get past.
TINY = dict(embed_dim=8, n_head=2, ffn_dim=16, patch_size=2, img_size=8, hidden=16, action_dim=2,
            n_layer=2, n_classes=4)


def kv_value(reader: "gguf.GGUFReader", key: str):
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def dim_map(reader: "gguf.GGUFReader", ns: str) -> dict[int, int]:
    """old dimension value -> new one, for the `enc` / `pred` / `head` namespace of this file."""
    get = lambda k, d=0: int(kv_value(reader, f"jepa.{ns}.{k}") or d)  # noqa: E731
    d, f = TINY["embed_dim"], TINY["ffn_dim"]
    inner_new = TINY["n_head"] * (d // TINY["n_head"])
    if ns == "enc":
        old_d, old_f = get("embed_dim"), get("ffn_dim")
        old_inner = get("n_head") * (get("head_dim") or (old_d // get("n_head", 1) if get("n_head") else 0))
        patch, tub, chans = get("patch_size", 16), get("tubelet_size", 1), get("in_chans", 3)
        img, n_reg = get("img_size", 224), get("n_registers", 0)
        prefix = (1 if kv_value(reader, "jepa.enc.cls_token") else 0) + n_reg
        old = {
            "d": old_d, "3d": 3 * old_d, "ffn": old_f,
            "inner": old_inner, "3inner": 3 * old_inner,
            "patch_in": chans * tub * patch * patch,
            "n_pos": (img // patch) ** 2 + prefix if patch else 0,
        }
        new = {
            "d": d, "3d": 3 * d, "ffn": f,
            "inner": inner_new, "3inner": 3 * inner_new,
            "patch_in": chans * 1 * TINY["patch_size"] ** 2,
            "n_pos": (TINY["img_size"] // TINY["patch_size"]) ** 2 + prefix,
        }
        # the LeWM projector hidden width (enc.proj.0), when the file has one
        if _has(reader, "enc.proj.0.weight"):
            # enc.proj.0 is [D, hidden] in ggml `ne` order: the hidden width is the axis that is not D
            shape = [int(v) for v in reader.tensors[_index_of(reader, "enc.proj.0.weight")].shape if int(v) > 0]
            hidden = next((v for v in shape if v != old_d), 0)
            if hidden > 0:
                old["hidden"], new["hidden"] = hidden, TINY["hidden"]
    elif ns == "pred":
        old_d, old_f = get("embed_dim"), get("ffn_dim")
        old_inner = get("n_head") * (get("head_dim") or (old_d // get("n_head", 1) if get("n_head") else 0))
        # pred.embed.* and pred.proj.* straddle the two widths, so the encoder's dims belong in this
        # map too. They map to the same new value, so a collision between them is not one.
        enc_d = int(kv_value(reader, "jepa.enc.embed_dim") or 0)
        old = {"d": old_d, "3d": 3 * old_d, "6d": 6 * old_d, "ffn": old_f,
               "inner": old_inner, "3inner": 3 * old_inner, "enc_d": enc_d,
               "out": get("out_dim", old_d), "act": get("action_dim")}
        new = {"d": d, "3d": 3 * d, "6d": 6 * d, "ffn": f,
               "inner": inner_new, "3inner": 3 * inner_new, "enc_d": d,
               "out": d, "act": TINY["action_dim"]}
    else:
        eget = lambda k, dd=0: int(kv_value(reader, f"jepa.enc.{k}") or dd)  # noqa: E731
        old_d = eget("embed_dim")
        old_inner = eget("n_head") * (old_d // eget("n_head", 1) if eget("n_head") else 0)
        old = {"d": old_d, "3d": 3 * old_d, "ffn": eget("ffn_dim"),
               "inner": old_inner, "3inner": 3 * old_inner, "cls": get("n_classes")}
        new = {"d": d, "3d": 3 * d, "ffn": f,
               "inner": inner_new, "3inner": 3 * inner_new, "cls": TINY["n_classes"]}

    seen: dict[int, str] = {}
    out: dict[int, int] = {}
    for role, v in old.items():
        if v <= 0:
            continue
        if v in seen and new[seen[v]] != new[role]:
            sys.exit(f"{ns}: dimension {v} is both '{seen[v]}' and '{role}' — the substitution would be ambiguous")
        seen[v] = role
        out[v] = new[role]
    return out


def _has(reader, name: str) -> bool:
    return any(t.name == name for t in reader.tensors)


def _index_of(reader, name: str) -> int:
    for i, t in enumerate(reader.tensors):
        if t.name == name:
            return i
    raise KeyError(name)


def shrink(src: pathlib.Path, out: pathlib.Path) -> None:
    reader = gguf.GGUFReader(str(src))
    maps = {ns: dim_map(reader, ns) for ns in ("enc", "pred", "head")}

    n_layer_old = {ns: int(kv_value(reader, f"jepa.{ns}.n_layer") or 0) for ns in ("enc", "pred")}
    n_layer_old["head"] = int(kv_value(reader, "jepa.head.n_pool_layers") or 0)
    n_layer_new = {ns: min(TINY["n_layer"], n) for ns, n in n_layer_old.items()}

    overrides: dict[str, int] = {}
    for ns in ("enc", "pred"):
        if kv_value(reader, f"jepa.{ns}.embed_dim") is None:
            continue
        overrides[f"jepa.{ns}.embed_dim"] = TINY["embed_dim"]
        overrides[f"jepa.{ns}.n_head"] = TINY["n_head"]
        overrides[f"jepa.{ns}.ffn_dim"] = TINY["ffn_dim"]
        overrides[f"jepa.{ns}.n_layer"] = n_layer_new[ns]
        if kv_value(reader, f"jepa.{ns}.out_dim") is not None:
            overrides[f"jepa.{ns}.out_dim"] = TINY["embed_dim"]
        if kv_value(reader, f"jepa.{ns}.head_dim") is not None:
            overrides[f"jepa.{ns}.head_dim"] = TINY["embed_dim"] // TINY["n_head"]
        if kv_value(reader, f"jepa.{ns}.action_dim") is not None:
            overrides[f"jepa.{ns}.action_dim"] = TINY["action_dim"]
        if kv_value(reader, f"jepa.{ns}.grid_size") is not None:
            overrides[f"jepa.{ns}.grid_size"] = TINY["img_size"] // TINY["patch_size"]
    overrides["jepa.enc.patch_size"] = TINY["patch_size"]
    overrides["jepa.enc.img_size"] = TINY["img_size"]
    overrides["jepa.enc.tubelet_size"] = 1
    if kv_value(reader, "jepa.head.n_pool_layers") is not None:
        overrides["jepa.head.n_pool_layers"] = n_layer_new["head"]
    if kv_value(reader, "jepa.head.n_classes") is not None:
        overrides["jepa.head.n_classes"] = TINY["n_classes"]
    if kv_value(reader, "jepa.pre.resize_short") is not None:
        overrides["jepa.pre.resize_short"] = TINY["img_size"]
    if kv_value(reader, "jepa.pre.crop") is not None:
        overrides["jepa.pre.crop"] = TINY["img_size"]

    arch = str(kv_value(reader, "general.architecture") or "jepa")
    writer = gguf.GGUFWriter(str(out), arch)
    for key, field in reader.fields.items():
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
        if key in overrides:
            writer.add_key_value(key, overrides[key], gguf.GGUFValueType.INT32)
            continue
        value = field.contents()
        if value is None:
            continue
        types = field.types
        if types and types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, sub_type=types[1])
        else:
            writer.add_key_value(key, value, types[0])

    rng = np.random.default_rng(0x6a657061)
    for t in reader.tensors:
        name = t.name
        ns = name.split(".", 1)[0]
        block = _block_index(name)
        if block is not None and block >= n_layer_new.get(ns, 0):
            continue
        m = maps.get(ns, {})
        shape = tuple(int(m.get(int(v), int(v))) for v in t.shape if int(v) > 0)
        if not shape:
            continue
        # ggml `ne` is reversed relative to numpy's shape
        data = rng.standard_normal(size=tuple(reversed(shape)), dtype=np.float32) * 0.05
        writer.add_tensor(name, np.ascontiguousarray(data), raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _block_index(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "blk" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


# --------------------------------------------------------------------------------------------
# truncation and key mutation
# --------------------------------------------------------------------------------------------
def boundaries(path: pathlib.Path, cap: int) -> list[int]:
    """Offsets worth cutting at: the fixed header fields, then a spread up to `cap`."""
    raw = path.read_bytes()[: max(cap, 64)]
    offs = {4, 8, 16, 24, len(raw)}
    for frac in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89):
        offs.add(len(raw) * frac // 100)
    return sorted(o for o in offs if 0 < o <= len(raw))


def mutate_kv(reader_path: pathlib.Path, out_dir: pathlib.Path, tag: str) -> int:
    """Rewrite one key of the mini file at a time with a value the loader must refuse cleanly."""
    reader = gguf.GGUFReader(str(reader_path))
    written = 0
    cases: list[tuple[str, object, "gguf.GGUFValueType"]] = [
        ("jepa.enc.n_head", 0, gguf.GGUFValueType.INT32),
        ("jepa.enc.n_head", -1, gguf.GGUFValueType.INT32),
        ("jepa.enc.embed_dim", 0, gguf.GGUFValueType.INT32),
        ("jepa.enc.embed_dim", -8, gguf.GGUFValueType.INT32),
        ("jepa.enc.embed_dim", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.enc.n_layer", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.enc.n_layer", 0, gguf.GGUFValueType.INT32),
        ("jepa.enc.ffn_dim", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.enc.patch_size", 0, gguf.GGUFValueType.INT32),
        ("jepa.enc.patch_size", -3, gguf.GGUFValueType.INT32),
        ("jepa.enc.in_chans", 2 ** 30, gguf.GGUFValueType.INT32),
        ("jepa.enc.tubelet_size", 2 ** 30, gguf.GGUFValueType.INT32),
        ("jepa.enc.img_size", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.enc.n_frames", -1, gguf.GGUFValueType.INT32),
        ("jepa.enc.n_registers", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.enc.act", "not_an_activation", gguf.GGUFValueType.STRING),
        ("jepa.enc.proj_act", "relu6", gguf.GGUFValueType.STRING),
        ("jepa.enc.attn_mode", "causal", gguf.GGUFValueType.STRING),
        ("jepa.enc.attn_mode", "block_causal", gguf.GGUFValueType.STRING),
        ("jepa.enc.pos_type", "quaternion", gguf.GGUFValueType.STRING),
        ("jepa.family", "not_a_family", gguf.GGUFValueType.STRING),
        ("jepa.family", "", gguf.GGUFValueType.STRING),
        ("general.architecture", "llama", gguf.GGUFValueType.STRING),
        ("jepa.pred.kind", "masked", gguf.GGUFValueType.STRING),
        ("jepa.pred.kind", "lewm", gguf.GGUFValueType.STRING),
        ("jepa.pred.kind", "ac", gguf.GGUFValueType.STRING),
        ("jepa.pred.n_layer", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.pred.n_head", 0, gguf.GGUFValueType.INT32),
        ("jepa.pred.action_dim", 0, gguf.GGUFValueType.INT32),
        ("jepa.pred.grid_size", 0, gguf.GGUFValueType.INT32),
        ("jepa.head.kind", "attentive_pool", gguf.GGUFValueType.STRING),
        ("jepa.head.n_pool_layers", 2 ** 31 - 1, gguf.GGUFValueType.INT32),
        ("jepa.head.n_classes", -1, gguf.GGUFValueType.INT32),
        ("jepa.pre.crop", -1, gguf.GGUFValueType.INT32),
        ("jepa.pre.resize_short", 0, gguf.GGUFValueType.INT32),
        ("jepa.enc.ln_eps", float("nan"), gguf.GGUFValueType.FLOAT32),
        ("jepa.enc.rope_theta", 0.0, gguf.GGUFValueType.FLOAT32),
        # wrong value type for a key the loader reads as an integer / string
        ("jepa.enc.embed_dim", "eight", gguf.GGUFValueType.STRING),
        ("jepa.family", 7, gguf.GGUFValueType.INT32),
    ]
    for i, (key, value, vtype) in enumerate(cases):
        name = f"kv-{tag}-{i:02d}-{key.replace('.', '_')}.gguf"
        _rewrite(reader, out_dir / name, {key: (value, vtype)})
        written += 1
    return written


def _rewrite(reader: "gguf.GGUFReader", out: pathlib.Path, overrides: dict) -> None:
    arch = str(kv_value(reader, "general.architecture") or "jepa")
    if "general.architecture" in overrides:
        arch = str(overrides["general.architecture"][0])
    writer = gguf.GGUFWriter(str(out), arch)
    for key, field in reader.fields.items():
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
        if key in overrides:
            value, vtype = overrides[key]
            writer.add_key_value(key, value, vtype)
            continue
        value = field.contents()
        if value is None:
            continue
        types = field.types
        if types and types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, sub_type=types[1])
        else:
            writer.add_key_value(key, value, types[0])
    for key, (value, vtype) in overrides.items():
        if key not in reader.fields and key != "general.architecture":
            writer.add_key_value(key, value, vtype)
    for t in reader.tensors:
        arr = np.array(t.data)
        writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "tests/fuzz/corpus")
    ap.add_argument("--models", type=pathlib.Path, nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--cap", type=int, default=128 * 1024,
                    help="largest truncation taken from a real GGUF (bytes, default 128 KiB)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in args.models:
        if not src.exists():
            print(f"skip {src} (not present)", file=sys.stderr)
            continue
        tag = src.stem.replace("-f32", "").replace("-pretrain-in1k", "")
        mini = args.out / f"mini-{tag}.gguf"
        shrink(src, mini)
        n += 1
        print(f"{mini.name}: {mini.stat().st_size} bytes")

        mini_bytes = mini.read_bytes()
        for off in boundaries(mini, len(mini_bytes)):
            (args.out / f"trunc-mini-{tag}-{off}.gguf").write_bytes(mini_bytes[:off])
            n += 1
        for off in boundaries(src, args.cap):
            (args.out / f"trunc-real-{tag}-{off}.gguf").write_bytes(src.read_bytes()[:off])
            n += 1
        n += mutate_kv(mini, args.out, tag)

    # a handful of hand-written degenerate headers: the shapes a mutator rarely reaches by chance
    (args.out / "raw-empty.gguf").write_bytes(b"")
    (args.out / "raw-magic-only.gguf").write_bytes(b"GGUF")
    (args.out / "raw-wrong-magic.gguf").write_bytes(b"FUGG" + b"\x03\x00\x00\x00" + b"\x00" * 16)
    (args.out / "raw-version-0.gguf").write_bytes(b"GGUF" + struct.pack("<I", 0) + b"\x00" * 16)
    (args.out / "raw-huge-counts.gguf").write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<qq", 2 ** 62, 2 ** 62))
    n += 5

    print(f"{n} corpus files in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
