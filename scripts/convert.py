#!/usr/bin/env python3
"""convert.py — PyTorch / safetensors checkpoints -> jepa.cpp GGUF (docs/gguf-schema.md).

Examples
  scripts/convert.py --family ijepa --src models/facebook/ijepa_vith14_1k --ftype f16
  scripts/convert.py --family hfvit --src models/OK-AI/lejepa-vits16-pretrain-in1k --ftype f32
  scripts/convert.py --family lewm  --src models/quentinll/lewm-pusht --out models/gguf/lewm-pusht-f32.gguf
  scripts/convert.py --family vjepa2   --src models/facebook/vjepa2-vitl-fpc64-256
  scripts/convert.py --family vjepa2_1 --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
  scripts/convert.py --family levjepa --src models/galilai-group/LeVJEPA-VideoMix-Large --ftype f32

Default output name (when --out is omitted):
  <repo>/models/gguf/<basename of --src, extension stripped>-<ftype>.gguf
  e.g. models/gguf/ijepa_vith14_1k-f16.gguf, models/gguf/lewm-pusht-f32.gguf

--ftype f32 stores every tensor as F32; --ftype f16 stores the quantizable weights
(attn_*, ffn_*, *.proj, head.cls) as F16 and everything else (patch embed, norms, biases,
position tables, tokens, adaLN, action embed) as F32.  Q8_0 and friends are produced from an
f32/f16 file by tools/jepa-quantize, never here.

vjepa2 / vjepa2_1 / levjepa are standalone modules (scripts/jepa_convert/vjepa2.py, vjepa2_1.py,
levjepa.py) that depend only on `gguf`; this CLI dispatches to them if they are present.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_convert.common import FTYPES, default_output_path, log  # noqa: E402

FAMILIES = ("ijepa", "hfvit", "lewm", "vjepa2", "vjepa2_1", "levjepa")
LOCAL_FAMILIES = ("ijepa", "hfvit", "lewm")


def _dispatch_external(family: str, src: Path, out: Path, ftype: str, name: str | None) -> None:
    modname = f"jepa_convert.{family}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(family):
            raise SystemExit(
                f"family {family!r}: scripts/jepa_convert/{family}.py is not present in this checkout "
                f"(it is written as a standalone module by the video-family converter). Nothing was written.")
        raise
    if hasattr(mod, "convert"):
        import inspect
        if "name" in inspect.signature(mod.convert).parameters:
            mod.convert(src, out, ftype, name=name)
        else:
            mod.convert(src, out, ftype)
        return
    if hasattr(mod, "main"):
        argv = ["--src", str(src), "--out", str(out), "--ftype", ftype]
        if name:
            argv += ["--name", name]
        mod.main(argv)
        return
    raise SystemExit(f"{modname} exposes neither convert(src, out, ftype) nor main(argv)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="convert.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES, help="model family (see docs/gguf-schema.md)")
    ap.add_argument("--src", required=True, help="checkpoint directory (HF repo layout) or single .pt file")
    ap.add_argument("--out", default=None, help="output .gguf path (default: models/gguf/<src-basename>-<ftype>.gguf)")
    ap.add_argument("--ftype", default="f16", choices=sorted(FTYPES), help="storage dtype for quantizable weights (default f16)")
    ap.add_argument("--name", default=None, help="override general.name (default: basename of --src)")
    ap.add_argument("--allow-unmapped", action="store_true",
                    help="do not fail when the checkpoint has tensors the converter does not know")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"--src {src} does not exist")
    out = Path(args.out) if args.out else default_output_path(args.family, src, args.ftype)
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"convert: family={args.family} src={src} -> {out} (ftype={args.ftype})")

    if args.family in LOCAL_FAMILIES:
        mod = importlib.import_module(f"jepa_convert.{args.family}")
        mod.convert(src, out, args.ftype, name=args.name, allow_unmapped=args.allow_unmapped)
    else:
        _dispatch_external(args.family, src, out, args.ftype, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
