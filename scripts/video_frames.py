#!/usr/bin/env python3
"""Decode a video dataset to per-clip THWC uint8 .npy frame caches shared by both backends.

    scripts/video_frames.py --data DIR --out tmp/frames [--frames 16] [--splits train,val,test]
                            [--jobs 32] [--index tmp/frames/index.json]

`DIR` is a `<split>/<class>/<clip>.avi` tree (the UCF101 subset at
data/ucf101-subset/UCF101_subset).  Every clip is decoded once with PyAV as RGB24 and reduced to
`--frames` frames sampled uniformly over the whole clip, endpoints included, with exactly the rule
scripts/dump_reference.py uses:

    idx = round(linspace(0, T_total - 1, n))

The result is written as `<out>/<split>/<class>/<stem>.npy`, a (T, H, W, 3) uint8 C-order array —
the layout `jepa-embed --frames-npy` / `jepa-classify --frames-npy` read and the same thing the
HF video processor is fed on the PyTorch side, so both backends start from identical pixels.

An index JSON records every clip (split, class, label id, source path, npy path, decoded frame
count, sampled indices, fps, frame size) so downstream runs are reproducible without re-decoding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VIDEO_EXT = (".avi", ".mp4", ".mkv", ".webm", ".mov")


def decode_video(path: Path):
    """Decode every frame with PyAV as RGB24 -> (T, H, W, 3) uint8, plus fps.

    Identical to scripts/dump_reference.py:decode_video (kept in sync deliberately).
    """
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
        fps = float(s.average_rate) if s.average_rate else None
    return np.stack(frames), fps


def sample_frames(frames, n: int):
    """n frames uniformly over the clip, endpoints included: idx = round(linspace(0, T-1, n))."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.ascontiguousarray(frames[idx]), idx.tolist()


def list_clips(data: Path, splits: list[str]) -> list[dict]:
    """`<split>/<class>/<clip>` tree -> sorted clip records with a stable per-dataset label id."""
    classes = sorted({d.name for sp in splits for d in (data / sp).iterdir() if d.is_dir()
                      and any(p.suffix.lower() in VIDEO_EXT for p in d.iterdir())})
    cls_id = {c: i for i, c in enumerate(classes)}
    out = []
    for sp in splits:
        for c in classes:
            d = data / sp / c
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in VIDEO_EXT:
                    out.append({"split": sp, "class": c, "label": cls_id[c], "clip": str(p), "stem": p.stem})
    return out


def _one(job):
    rec, out_dir, n_frames, overwrite = job
    dst = Path(out_dir) / rec["split"] / rec["class"] / (rec["stem"] + ".npy")
    if dst.exists() and not overwrite:
        a = np.load(dst, mmap_mode="r")
        return {**rec, "npy": str(dst), "frame_size_hw": [int(a.shape[1]), int(a.shape[2])], "cached": True}
    t = time.time()
    frames, fps = decode_video(Path(rec["clip"]))
    fr, idx = sample_frames(frames, n_frames)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".npy.tmp")           # atomic: a killed run never leaves a half file
    with open(tmp, "wb") as f:                  # a handle, so numpy does not re-append ".npy"
        np.save(f, fr)
    tmp.replace(dst)
    return {**rec, "npy": str(dst), "n_frames": int(n_frames), "n_frames_total": int(len(frames)),
            "frame_indices": idx, "fps": fps, "frame_size_hw": [int(fr.shape[1]), int(fr.shape[2])],
            "decode_s": round(time.time() - t, 3), "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/ucf101-subset/UCF101_subset"))
    ap.add_argument("--out", type=Path, default=Path("tmp/frames"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--index", type=Path, default=None, help="index JSON (default <out>/index.json)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    splits = [s for s in a.splits.split(",") if s]
    if not a.data.is_dir():
        sys.exit(f"no dataset at {a.data}")
    clips = list_clips(a.data, splits)
    if not clips:
        sys.exit(f"no videos under {a.data}/{{{a.splits}}}")
    index_path = a.index or (a.out / "index.json")
    print(f"{len(clips)} clips, {a.frames} frames each -> {a.out} ({a.jobs} jobs)")

    t0 = time.time()
    jobs = [(r, str(a.out), a.frames, a.overwrite) for r in clips]
    recs = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.1f}s", flush=True)
    wall = time.time() - t0

    by_split = {}
    for r in recs:
        by_split.setdefault(r["split"], {}).setdefault(r["class"], 0)
        by_split[r["split"]][r["class"]] += 1
    classes = sorted({r["class"] for r in recs})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        "dataset": str(a.data), "frames": a.frames, "splits": splits, "classes": classes,
        "sampling": "idx = round(linspace(0, T_total-1, n)) over all PyAV-decoded rgb24 frames",
        "layout": "THWC uint8 .npy per clip", "n_clips": len(recs),
        "counts": {sp: sum(v.values()) for sp, v in by_split.items()},
        "wall_s": round(wall, 1), "clips": recs}, indent=1))
    print(f"decoded {sum(0 if r.get('cached') else 1 for r in recs)}/{len(recs)} clips in {wall:.1f}s "
          f"({len(classes)} classes) -> {index_path}")


if __name__ == "__main__":
    main()
