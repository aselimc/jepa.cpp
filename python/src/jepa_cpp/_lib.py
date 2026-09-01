"""Finding and opening the jepa.cpp shared library.

The wheel ships one self-contained ``libjepa`` (all of libjepa plus all of ggml, linked whole —
python/CMakeLists.txt), so the common case is a single file next to this module. A checkout that
already has its own build can point ``JEPA_CPP_LIB`` at that library instead, which is what the
parity tests do when they want the bindings and ``build/jepa-embed`` to be the very same code.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys

__all__ = ["LIB_ENV", "library_path", "load_library"]

LIB_ENV = "JEPA_CPP_LIB"


def _names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libjepa.dylib",)
    if os.name == "nt":
        return ("jepa.dll", "libjepa.dll")
    return ("libjepa.so",)


def _package_roots() -> list[pathlib.Path]:
    """Every directory this package is assembled from.

    A wheel has one; an editable install has two — the source tree the .py files come from and the
    site-packages directory CMake's install step wrote ``lib/libjepa.so`` into — and only the
    second one holds the library, so both have to be searched.
    """
    roots: list[pathlib.Path] = []
    pkg = sys.modules.get(__package__ or "")
    for p in getattr(pkg, "__path__", None) or []:
        roots.append(pathlib.Path(p))
    here = pathlib.Path(__file__).resolve().parent
    if here not in roots:
        roots.append(here)
    return roots


def _candidates() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    override = os.environ.get(LIB_ENV)
    if override:
        p = pathlib.Path(override).expanduser()
        # A directory is allowed so callers can name a CMake build tree rather than a file.
        out += [p / n for n in _names()] if p.is_dir() else [p]
    for root in _package_roots():
        out += [root / "lib" / n for n in _names()]
        out += [root / n for n in _names()]
    return out


def library_path() -> pathlib.Path:
    """Path of the shared library that would be loaded. Raises if there is none."""
    tried = _candidates()
    for c in tried:
        if c.is_file():
            return c
    listed = "\n  ".join(str(c) for c in tried)
    raise OSError(
        "jepa_cpp cannot find the jepa.cpp shared library. Looked at:\n  "
        + listed
        + f"\nInstall the wheel (which bundles it), or set ${LIB_ENV} to a library built from the "
        "repository (`cmake -S python -B build-py && cmake --build build-py` produces one)."
    )


def load_library() -> ctypes.CDLL:
    path = library_path()
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # pragma: no cover - platform specific
        raise OSError(f"cannot load {path}: {exc}") from exc
    if not hasattr(lib, "jepa_version"):
        raise OSError(f"{path} does not export jepa_version() — this is not a jepa.cpp library")
    return lib
