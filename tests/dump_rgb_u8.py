#!/usr/bin/env python3
"""Dump the fixture images as PIL-decoded RGB uint8 arrays (<stem>.rgb.npy, HWC) so that
`test-parity --rgb-dir DIR` can separate JPEG-decoder differences (stb_image vs libjpeg/PIL) from
resize/normalisation differences in its own-preprocessing pass.

  tests/dump_rgb_u8.py OUT_DIR [--media tests/fixtures/media]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--media", default=str(Path(__file__).resolve().parent / "fixtures" / "media"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.media).glob("*.jpg")) + sorted(Path(a.media).glob("*.png"))
    if not files:
        sys.exit(f"no images in {a.media}")
    for f in files:
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        np.save(out / f"{f.stem}.rgb.npy", np.ascontiguousarray(arr))
        print(f"{f.name}: {arr.shape} -> {out / (f.stem + '.rgb.npy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
