#!/usr/bin/env python3
"""Generate docs/api.md — the C API reference — from include/jepa.h.

The header is the single source of truth: this script slices it into the sections its banner
comments define, extracts every declaration with the comment block that precedes it, and emits
a markdown page with a function index. Regenerate after editing the header:

    python scripts/gen_api_md.py            # writes docs/api.md
    python scripts/gen_api_md.py --check    # exit 1 if docs/api.md is stale (used by CI)
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / "include" / "jepa.h"
OUT = ROOT / "docs" / "api.md"

SECTION_RE = re.compile(r"^// (?:-{2,}|={2,})\s*(.*?)\s*(?:-{2,}|={2,})\s*$")
FUNC_RE = re.compile(r"\b(jepa_\w+)\s*\(")


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Return (title, lines) per banner-comment section; a leading unnamed section collects the rest."""
    sections: list[tuple[str, list[str]]] = [("Types and handles", [])]
    for line in text.splitlines():
        m = SECTION_RE.match(line.rstrip())
        if m and m.group(1):
            sections.append((m.group(1).rstrip(" -"), []))
            continue
        sections[-1][1].append(line)
    return sections


def tidy(lines: list[str]) -> str:
    """Drop include guards / cpp guards / pragma noise, collapse blank runs."""
    keep: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s in ('#pragma once', 'extern "C" {', "}", "#ifdef __cplusplus", "#endif"):
            continue
        if s.startswith("#include"):
            continue
        keep.append(ln.rstrip())
    out: list[str] = []
    for ln in keep:
        if ln == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    while out and out[-1] == "":
        out.pop()
    while out and out[0] == "":
        out.pop(0)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify docs/api.md is up to date")
    args = ap.parse_args()

    text = HEADER.read_text()
    sections = split_sections(text)

    funcs = sorted(set(FUNC_RE.findall(text)))
    index = " · ".join(f"`{f}`" for f in funcs)

    parts = [
        "# C API reference",
        "",
        "*Generated from [`include/jepa.h`](https://github.com/aselimc/jepa.cpp/blob/main/include/jepa.h)"
        " by `scripts/gen_api_md.py` — do not edit by hand.*",
        "",
        "The whole public surface is this one header: C linkage, opaque handles"
        " (`jepa_model`, `jepa_context`), plain structs, `malloc`-returned buffers freed with"
        " `jepa_free`. Typical flow: `jepa_model_load` → `jepa_context_new` → `jepa_preprocess_*`"
        " → `jepa_encode` → `jepa_pool_*` / `jepa_head` / `jepa_predict*` / `jepa_lewm_*`.",
        "",
        f"**Functions ({len(funcs)}):** {index}",
        "",
    ]
    for title, lines in sections:
        body = tidy(lines)
        if not body:
            continue
        parts.append(f"## {title}")
        parts.append("")
        parts.append("```c")
        parts.append(body)
        parts.append("```")
        parts.append("")
    rendered = "\n".join(parts)

    if args.check:
        if not OUT.exists() or OUT.read_text() != rendered:
            print("docs/api.md is stale — run scripts/gen_api_md.py", file=sys.stderr)
            return 1
        print("docs/api.md is up to date")
        return 0

    OUT.write_text(rendered)
    print(f"wrote {OUT} ({len(rendered.splitlines())} lines, {len(funcs)} functions, "
          f"{sum(1 for t, l in sections if tidy(l))} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
