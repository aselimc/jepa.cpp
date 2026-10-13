#!/usr/bin/env python3
"""hjepa_batch.py — convert, verify and publish the released H-JEPA checkpoints in bulk.

    scripts/hjepa_batch.py run    --hjepa-src tmp/hjepa/src [--runs REGEX] [--seeds 42,1]
    scripts/hjepa_batch.py upload [--org jepacpp] [--dry-run]

`run`, for every selected run of huggingface.co/jepa-world-models/h-jepa:
    download <env>/<run>/seed<N>/<run>_object.ckpt  -> models/jepa-world-models/h-jepa/...
    convert.py --family hjepa --ftype f32           -> models/gguf/<run>-seed<N>-f32.gguf
    hjepa_reference.py (the authors' modules)       -> tests/fixtures/ref/<run>-seed<N>/
    jepa-quantize f16, q8_0                         -> models/gguf/<run>-seed<N>-{f16,q8_0}.gguf
    test-predictor --hjepa on all three             -> tests/results/hjepa-parity.json
A file is only ever published with a PASS in that JSON next to its sha256.

`upload` pushes the GGUFs to <org>/h-jepa-GGUF as <env>/<run>-seed<N>-<dtype>.gguf with a card
generated from the JSON, and the reference dumps to the <org>/jepa.cpp-fixtures dataset.

The default selection is one seed of every released configuration (seed 42 for the four simulated
environments, seed 1 for DROID): 31 of the 93 checkpoints. The other seeds are retrainings of the
same architectures; `--seeds` and `--runs` select them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_REPO = "jepa-world-models/h-jepa"
SRC_DIR = ROOT / "models" / "jepa-world-models" / "h-jepa"
GGUF_DIR = ROOT / "models" / "gguf"
REF_DIR = ROOT / "tests" / "fixtures" / "ref"
RESULTS = ROOT / "tests" / "results" / "hjepa-parity.json"
PY = sys.executable
DTYPES = ("f32", "f16", "q8_0")
ROW = re.compile(r"^\s+(?P<name>.+?)\s+rows=(?P<rows>\d+)\s+cos_mean=(?P<mean>[\d.]+) cos_min=(?P<min>[\d.]+) "
                 r"max\|d\|=(?P<mad>[\d.e+-]+) rel=(?P<rel>[\d.e+-]+).*?(?P<ok>OK|FAIL)\s*$")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def list_runs() -> list[dict]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{SRC_REPO}/tree/main?recursive=true&limit=1000") as r:
        tree = json.load(r)
    runs = []
    for e in tree:
        m = re.fullmatch(r"(\w+)/(\w+)/seed(\d+)/(\w+)_object\.ckpt", e["path"])
        if e["type"] == "file" and m and m.group(2) == m.group(4):
            runs.append({"env": m.group(1), "run": m.group(2), "seed": int(m.group(3)), "path": e["path"],
                         "size": e["size"], "sha256": (e.get("lfs") or {}).get("oid", "")})
    return sorted(runs, key=lambda r: (r["env"], r["run"], r["seed"]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(run: dict) -> Path:
    dst = SRC_DIR / run["path"]
    if dst.exists() and dst.stat().st_size == run["size"]:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = sh(["curl", "-sfL", "-o", dst, f"https://huggingface.co/{SRC_REPO}/resolve/main/{run['path']}"])
    if r.returncode != 0 or dst.stat().st_size != run["size"]:
        raise SystemExit(f"download of {run['path']} failed")
    return dst


def parity(gguf: Path, ref: Path, threads: int) -> dict:
    """Run test-predictor --hjepa and summarise its rows against the reference."""
    r = sh([ROOT / "build" / "test-predictor", "--hjepa", gguf, "--ref", ref, "--threads", threads])
    rows = [m.groupdict() for m in map(ROW.match, r.stdout.splitlines()) if m]
    # rows that compare against the PyTorch dump (the others are self-consistency checks)
    vs_ref = [x for x in rows if "causal" not in x["name"] and "== predict" not in x["name"]]
    chain = [x for x in vs_ref if x["name"].startswith("chain")]
    return {
        "pass": r.returncode == 0 and r.stdout.strip().endswith("PASS"),
        "checks": len(rows),
        "failed": [x["name"] for x in rows if x["ok"] == "FAIL"],
        "cos_min": min((float(x["min"]) for x in vs_ref), default=None),
        "rel_max": max((float(x["rel"]) for x in vs_ref), default=None),
        "chain_cos_min": float(chain[0]["min"]) if chain else None,
    }


def cmd_run(a) -> int:
    seeds = {int(s) for s in a.seeds.split(",")}
    runs = [r for r in list_runs() if r["seed"] in seeds and re.search(a.runs, r["run"])]
    # one seed per configuration: 42 for the simulated environments, 1 for DROID, unless asked otherwise
    if a.seeds == "42,1":
        runs = [r for r in runs if (r["seed"] == 1) == (r["env"] == "droid")]
    print(f"{len(runs)} runs selected")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    commit = sh(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).stdout.strip()
    for i, run in enumerate(runs, 1):
        name = f"{run['run']}-seed{run['seed']}"
        t0 = time.time()
        ckpt = download(run)
        f32 = GGUF_DIR / f"{name}-f32.gguf"
        r = sh([PY, ROOT / "scripts" / "convert.py", "--family", "hjepa", "--src", ckpt, "--ftype", "f32", "--out", f32])
        if r.returncode != 0:
            print(f"[{i}/{len(runs)}] {name}: CONVERT FAILED\n{r.stderr[-600:]}")
            results["runs"][name] = {"env": run["env"], "error": "convert: " + r.stderr.strip().splitlines()[-1][:200]}
            continue
        r = sh([PY, ROOT / "scripts" / "hjepa_reference.py", "--ckpt", ckpt, "--hjepa-src", a.hjepa_src, "--out", REF_DIR])
        if r.returncode != 0:
            print(f"[{i}/{len(runs)}] {name}: REFERENCE FAILED\n{r.stderr[-600:]}")
            results["runs"][name] = {"env": run["env"], "error": "reference: " + r.stderr.strip().splitlines()[-1][:200]}
            continue
        info = sh([ROOT / "build" / "jepa-info", f32]).stdout
        levels = [ln.strip() for ln in info.splitlines() if ln.strip().startswith("level ")]
        entry = {"env": run["env"], "run": run["run"], "seed": run["seed"], "source": run["path"],
                 "source_sha256": run["sha256"], "n_levels": len(levels), "levels": levels, "files": {}}
        for dt in DTYPES:
            g = GGUF_DIR / f"{name}-{dt}.gguf"
            if dt != "f32":
                q = sh([ROOT / "build" / "jepa-quantize", f32, g, dt])
                if q.returncode != 0:
                    entry["files"][dt] = {"pass": False, "error": q.stderr.strip()[-200:]}
                    continue
            entry["files"][dt] = {"file": g.name, "bytes": g.stat().st_size, "sha256": sha256(g),
                                  **parity(g, REF_DIR / name, a.threads)}
        results["runs"][name] = entry
        verdict = " ".join(f"{dt}:{'PASS' if entry['files'][dt].get('pass') else 'FAIL'}" for dt in DTYPES)
        print(f"[{i}/{len(runs)}] {name:28s} {len(levels)} levels  {verdict}  "
              f"f32 rel_max {entry['files']['f32'].get('rel_max')}  ({time.time() - t0:.0f}s)", flush=True)
        results["engine_commit"] = commit
        results["reference"] = "scripts/hjepa_reference.py: the authors' modules (github.com/kevinghst/H-JEPA), PyTorch CPU"
        results["bars"] = "tests/test-predictor.cpp thresholds_for(): f32 cos >= 0.9999 and rel <= 1e-3; f16 cos_mean >= 0.9999, cos_min >= 0.99; q8_0 cos_mean >= 0.999, cos_min >= 0.98"
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    bad = [n for n, e in results["runs"].items() if "error" in e or not all(f.get("pass") for f in e["files"].values())]
    print(f"wrote {RESULTS}: {len(results['runs'])} runs, {len(bad)} with a failure {bad if bad else ''}")
    return 1 if bad else 0


ENV_TITLES = {"ant": "Visual AntMaze", "cube": "OGBench Cube", "fourroom": "FourRoom Distractors",
              "pusht": "Push-T", "droid": "DROID"}


def card(results: dict, org: str) -> str:
    runs = {n: e for n, e in results["runs"].items() if "files" in e}
    lines = [
        "---", "license: mit", "library_name: jepa.cpp", "pipeline_tag: robotics", f"base_model: {SRC_REPO}",
        "tags: [gguf, jepa, h-jepa, world-models, planning, robotics, jepa.cpp]", "---", "",
        "# H-JEPA — GGUF for jepa.cpp", "",
        "**H-JEPA** ([arXiv 2610.06805](https://arxiv.org/abs/2610.06805), Zhang, Terver, Rabbat, LeCun, Balestriero) trains a",
        "hierarchy of action-conditioned JEPA world models end to end: level 1 predicts one step ahead in the latent space of a",
        "ViT image encoder, each level above re-encodes every *stride*-th state of the level below and predicts one of *its*",
        "steps ahead, conditioned on a latent macro-action. These are the authors' released checkpoints",
        f"([{SRC_REPO}](https://huggingface.co/{SRC_REPO}), code at [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA))",
        "converted to GGUF for [jepa.cpp](https://github.com/aselimc/jepa.cpp), a ggml C/C++ engine with no Python and no PyTorch",
        "at run time.", "",
        "`hjepa_l{N}` is the N-level H-JEPA, `hwm_l{N}` the HWM baseline (identity upper encoders), `lewm` the flat one-level",
        "baseline. One seed of every released configuration is here (seed 42 for the simulated environments, seed 1 for DROID).", "",
        "## What runs", "",
        "The encoder and every level's state encoder, action encoder, predictor and latent rollout (`jepa_hjepa_*` in",
        "`include/jepa.h`). The paper's top-down gradient planner is not in the engine yet.", "",
        "```sh",
        f"hf download {org}/h-jepa-GGUF ant/ant_hjepa_l3-seed42-f16.gguf --local-dir models/gguf",
        "jepa-info models/gguf/ant/ant_hjepa_l3-seed42-f16.gguf",
        "```", "",
        "## Parity with the authors' PyTorch", "",
        "Every file below was checked by `test-predictor --hjepa` against activations dumped from the authors' own modules",
        "(`scripts/hjepa_reference.py`): encoder tokens, each level's states, latent actions, action embeddings, teacher-forced",
        "predictions and autoregressive rollouts, and the whole chain from frames to top-level states. `cos min` is the worst",
        "row cosine over all of those comparisons, `rel max` the largest max-abs error relative to the reference's largest value.", "",
        "| file | levels | size | cos min | rel max |", "|---|---|---|---|---|",
    ]
    for env in ENV_TITLES:
        for name, e in sorted(runs.items()):
            if e["env"] != env:
                continue
            for dt in DTYPES:
                f = e["files"].get(dt, {})
                if not f.get("pass"):
                    continue
                lines.append(f"| `{env}/{f['file']}` | {e['n_levels']} | {f['bytes'] / 1e6:.1f} MB | {f['cos_min']:.7f} | {f['rel_max']:.1e} |")
    lines += [
        "", f"Measured at jepa.cpp `{results.get('engine_commit', '')}`, CPU. Bars: {results.get('bars', '')}.",
        "`q4_k` and below do not clear the q8_0 bars on these small models and are not published.", "",
        "## Inputs", "",
        "Frames are squash-resized to the encoder's image size and ImageNet-normalised (`jepa.pre.*`). Actions and",
        "proprioception are z-scored with the training statistics stored in the file (`norm.action.*`, `norm.proprio.*`,",
        "`jepa_hjepa_norm_stats`); a level-1 action is `jepa.hjepa.frameskip` consecutive raw actions stacked.", "",
        "## Licence", "",
        f"MIT, as the source checkpoints ([{SRC_REPO}](https://huggingface.co/{SRC_REPO})) and code. Cite the paper:", "",
        "```bibtex",
        "@article{zhang2026hjepa,",
        "  title   = {H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning},",
        "  author  = {Zhang, Wancong and Terver, Basile and Rabbat, Michael and LeCun, Yann and Balestriero, Randall},",
        "  journal = {arXiv preprint arXiv:2610.06805},", "  year    = {2026}", "}", "```", "",
    ]
    return "\n".join(lines)


def cmd_upload(a) -> int:
    results = json.loads(RESULTS.read_text())
    text = card(results, a.org)
    out = ROOT / "tmp" / "hf-cards" / "h-jepa-GGUF.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    todo = []
    for name, e in sorted(results["runs"].items()):
        for dt, f in e.get("files", {}).items():
            if f.get("pass"):
                p = GGUF_DIR / f["file"]
                if sha256(p) != f["sha256"]:
                    raise SystemExit(f"{p} changed since it was verified; re-run `hjepa_batch.py run`")
                todo.append((p, f"{e['env']}/{f['file']}"))
    total = sum(p.stat().st_size for p, _ in todo)
    print(f"card: {out}\n{len(todo)} GGUFs, {total / 1e9:.2f} GB -> {a.org}/h-jepa-GGUF; "
          f"{len(results['runs'])} reference dumps -> {a.org}/jepa.cpp-fixtures")
    if a.dry_run:
        return 0
    from huggingface_hub import HfApi
    api = HfApi()
    repo = f"{a.org}/h-jepa-GGUF"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo="README.md", repo_id=repo, repo_type="model",
                    commit_message="Model card")
    for i, (p, dst) in enumerate(todo, 1):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst, repo_id=repo, repo_type="model",
                        commit_message=f"Add {dst}")
        print(f"  [{i}/{len(todo)}] {dst}", flush=True)
    ds = f"{a.org}/jepa.cpp-fixtures"
    for name in sorted(results["runs"]):
        if (REF_DIR / name / "manifest.json").exists():
            api.upload_folder(folder_path=str(REF_DIR / name), path_in_repo=f"ref/{name}", repo_id=ds,
                              repo_type="dataset", commit_message=f"Add ref/{name} (H-JEPA)")
    print(f"done: https://huggingface.co/{repo}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--hjepa-src", required=True, type=Path, help="checkout of github.com/kevinghst/H-JEPA")
    r.add_argument("--runs", default=".", help="regex over the run name, e.g. '^ant_' (default: all)")
    r.add_argument("--seeds", default="42,1", help="comma-separated seeds (default 42,1 = one per configuration)")
    r.add_argument("--threads", type=int, default=8)
    u = sub.add_parser("upload")
    u.add_argument("--org", default="jepacpp")
    u.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_upload(a)


if __name__ == "__main__":
    sys.exit(main())
