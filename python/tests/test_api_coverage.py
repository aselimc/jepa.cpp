"""The check script: ``include/jepa.h`` is the contract, and this parses it.

Nothing here talks to a model. It reads the header, extracts every ``jepa_*`` declaration, every
``enum`` constant and every ``typedef struct``, and fails if :mod:`jepa_cpp._api` is missing any of
them, has them in a different order, or has a struct whose fields do not line up with the C one.
An entry point added to the header therefore breaks this test until it is bound.

Runnable on its own for a summary::

    python python/tests/test_api_coverage.py
"""

from __future__ import annotations

import ctypes
import pathlib
import re

import pytest

from jepa_cpp import _api

REPO = pathlib.Path(__file__).resolve().parents[2]
HEADER = REPO / "include" / "jepa.h"

pytestmark = pytest.mark.skipif(
    not HEADER.is_file(),
    reason=f"{HEADER} is not there — the coverage check needs the repository, not just the wheel",
)

# A declaration is a statement outside a comment that has `jepa_name(` in it and ends in `);`.
DECL_RE = re.compile(r"^[^/#\n].*?\bjepa_\w+\s*\([^;]*\);", re.M | re.S)
NAME_RE = re.compile(r"\b(jepa_\w+)\s*\(")
ENUM_RE = re.compile(r"^enum\s*\{([^}]*)\};", re.M)
STRUCT_RE = re.compile(r"typedef struct \{(.*?)\}\s*(jepa_\w+);", re.S)


def _header_text() -> str:
    """The header with its comments removed, so a name mentioned in prose is not a declaration."""
    text = HEADER.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def header_functions() -> list[str]:
    return [NAME_RE.search(d).group(1) for d in DECL_RE.findall(_header_text())]


def header_enum_constants() -> list[str]:
    out: list[str] = []
    for body in ENUM_RE.findall(_header_text()):
        for item in body.split(","):
            name = item.split("=")[0].strip()
            if name:
                out.append(name)
    return out


def header_structs() -> dict[str, list[str]]:
    """``{typedef name: [field names in order]}`` for the header's plain structs."""
    out: dict[str, list[str]] = {}
    for body, name in STRUCT_RE.findall(_header_text()):
        fields: list[str] = []
        for stmt in body.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            # "int n_batch, n_chans, n_frames" / "float mean[3]" / "const float * data"
            decl = re.sub(r"^(const\s+)?\w+\s+", "", stmt)
            for part in decl.split(","):
                ident = re.sub(r"[\[\]\d\*\s]", "", part)
                if ident:
                    fields.append(ident)
        out[name] = fields
    return out


def test_header_parses():
    """The parser itself has to find something, or every check below passes vacuously."""
    funcs = header_functions()
    assert len(funcs) > 60, funcs
    assert "jepa_encode" in funcs and "jepa_model_load" in funcs
    assert len(set(funcs)) == len(funcs), "the header declares a name twice"


def test_every_function_is_bound():
    missing = [f for f in header_functions() if not hasattr(_api, f)]
    assert (
        not missing
    ), f"include/jepa.h declares {len(missing)} function(s) _api.py does not bind: {missing}"


def test_no_invented_functions():
    """_api.py binds the header and nothing else — a stale name would fail to load at import."""
    declared = set(header_functions())
    extra = [name for name, _, _ in _api.FUNCTIONS if name not in declared]
    assert not extra, f"_api.py binds names that are not in include/jepa.h: {extra}"


def test_function_table_matches_header_order():
    """Header order is the review order; keeping the table in it makes drift obvious."""
    assert [name for name, _, _ in _api.FUNCTIONS] == header_functions()


def test_every_bound_function_has_a_signature():
    for name, restype, argtypes in _api.FUNCTIONS:
        fn = getattr(_api, name)
        assert fn.restype is restype, name
        assert list(fn.argtypes) == list(argtypes), name


def test_enum_constants():
    missing = [c for c in header_enum_constants() if not hasattr(_api, c)]
    assert not missing, f"unbound enum constants: {missing}"


def test_struct_fields_match_the_header():
    for name, fields in header_structs().items():
        struct = getattr(_api, name, None)
        assert struct is not None, f"_api.py has no {name}"
        assert [f for f, *_ in struct._fields_] == fields, name


def test_structs_are_not_padded_by_accident():
    """Sanity on the layouts the C side reads and writes: sizes on a 64-bit LP64 ABI."""
    assert ctypes.sizeof(_api.jepa_context_params) == 12
    assert ctypes.sizeof(_api.jepa_input) == 32
    assert ctypes.sizeof(_api.jepa_output) == 24
    assert ctypes.sizeof(_api.jepa_preprocess_params) == 48
    assert ctypes.sizeof(_api.jepa_model_params) == 8


def test_docs_api_md_is_generated_from_the_same_header():
    """docs/api.md quotes the function count; a new entry point has to reach the docs too."""
    doc = REPO / "docs" / "api.md"
    if not doc.is_file():
        pytest.skip("docs/api.md is not in this tree")
    m = re.search(r"\*\*Functions \((\d+)\):\*\*", doc.read_text())
    assert m, "docs/api.md has no function index"
    assert int(m.group(1)) == len(header_functions())


if __name__ == "__main__":  # pragma: no cover - manual summary
    funcs = header_functions()
    bound = [f for f in funcs if hasattr(_api, f)]
    print(
        f"include/jepa.h: {len(funcs)} functions, {len(bound)} bound, "
        f"{len(header_enum_constants())} enum constants, {len(header_structs())} structs"
    )
    for f in funcs:
        if not hasattr(_api, f):
            print("  MISSING", f)
    raise SystemExit(0 if len(bound) == len(funcs) else 1)
