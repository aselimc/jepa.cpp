"""jepa-cpp — Python bindings for `jepa.cpp <https://github.com/aselimc/jepa.cpp>`_.

A thin wrapper over the C API in ``include/jepa.h``: one GGUF file in, numpy arrays out, with the
engine, the preprocessing and the arithmetic all on the C side. The raw 1:1 ``ctypes`` layer is
:mod:`jepa_cpp._api`; :class:`Model` is the layer you normally want.

    >>> import jepa_cpp
    >>> with jepa_cpp.Model("models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf", threads=8) as m:
    ...     tokens = m.encode("cat.jpg")            # [197, 384]
    ...     feature = m.encode("cat.jpg", pool="cls")   # [384]
"""

from __future__ import annotations

from ._lib import LIB_ENV, library_path
from .model import (
    STILL_REPEAT_FAMILIES,
    VIDEO_FAMILIES,
    Classification,
    JepaError,
    Model,
    devices,
    system_info,
    version,
)

__all__ = [
    "Classification",
    "JepaError",
    "LIB_ENV",
    "Model",
    "STILL_REPEAT_FAMILIES",
    "VIDEO_FAMILIES",
    "__version__",
    "devices",
    "library_path",
    "library_version",
    "system_info",
    "version",
]

#: The package version, which the build reads from the repository's ``CMakeLists.txt`` — the one
#: place the project keeps it (docs/releases.md). The *loaded* library reports its own through
#: :func:`version`; the two differ only if ``$JEPA_CPP_LIB`` points at a library from another build.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("jepa-cpp")
except PackageNotFoundError:  # pragma: no cover - running from a source tree, uninstalled
    __version__ = "0+unknown"

#: Alias for :func:`version`, for when both versions are in play.
library_version = version
