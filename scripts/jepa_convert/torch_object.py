"""Read a `torch.save(nn.Module)` checkpoint without running any of the code it names.

`torch.load(weights_only=False)` on a pickled module object imports and calls whatever the pickle
says, so the file is a program. This reader is a `pickle.Unpickler` whose `find_class` hands back
only a short fixed list (`_SAFE`: the tensor / parameter rebuilders re-implemented here in numpy,
`OrderedDict`, `numpy.dtype`, and numpy's array / scalar / bytes constructors, also re-implemented) and a data-only stub class for every other
global. No torch import, no source checkout: the result is the module tree as plain data.

    obj = load_object("model_object.ckpt")
    tensors, modules = module_tree(obj)
      tensors   {"jepas.0.encoder.projector.net.0.weight": np.ndarray, ...}   == state_dict() order
      modules   {"jepas.0": ModuleInfo(cls="models.jepa.JEPA", attrs={"temporal_stride": 2, ...}), ...}

`attrs` holds the plain-valued instance attributes (ints, floats, strings, bools, tuples), which is
where the modules keep the hyper-parameters a state dict alone does not carry (`heads`, `eps`,
`temporal_stride`, ...).
"""
from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_STORAGE_DTYPES = {
    "DoubleStorage": np.float64, "FloatStorage": np.float32, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8,
    "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class Stub:
    """Stands in for every class or function the pickle names: it records, it never acts."""
    _qual = "?"

    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args

    def __setstate__(self, state: Any) -> None:
        self._state = state


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _NdArray:
    """numpy.ndarray as pickled by numpy (`_reconstruct` + `__setstate__`)."""

    def __init__(self, *args: Any):
        self.value: np.ndarray | None = None

    def __setstate__(self, state: tuple) -> None:
        _version, shape, dtype, fortran, raw = state
        if dtype.hasobject:
            raise pickle.UnpicklingError("object-dtype numpy array in checkpoint")
        a = np.frombuffer(raw, dtype=dtype)
        self.value = a.reshape(shape, order="F" if fortran else "C").copy()


def _rebuild_tensor_v2(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *_rest: Any) -> np.ndarray:
    size = tuple(int(s) for s in size)
    if len(size) == 0:
        return storage[offset:offset + 1].reshape(()).copy()
    if any(s == 0 for s in size):
        return np.zeros(size, dtype=storage.dtype)
    last = offset + sum((n - 1) * st for n, st in zip(size, stride))
    if offset < 0 or last >= storage.shape[0] or any(st < 0 for st in stride):
        raise pickle.UnpicklingError("tensor view reaches outside its storage")
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=tuple(int(st) * storage.itemsize for st in stride))
    return np.array(view)  # own copy, C-contiguous


def _rebuild_parameter(data: Any, _requires_grad: Any = None, _hooks: Any = None, *_rest: Any) -> Any:
    return data


def _numpy_scalar(dtype: np.dtype, raw: bytes) -> Any:
    if dtype.hasobject:
        raise pickle.UnpicklingError("object-dtype numpy scalar in checkpoint")
    return np.frombuffer(raw, dtype=dtype)[0]


def _codecs_encode(text: str, encoding: str = "latin1") -> bytes:
    # protocol-2 pickles spell a bytes object as _codecs.encode(<str>, "latin1")
    if not isinstance(text, str) or encoding not in ("latin1", "latin-1"):
        raise pickle.UnpicklingError("unexpected _codecs.encode call in checkpoint")
    return text.encode("latin1")


_SAFE = {
    ("_codecs", "encode"): _codecs_encode,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("collections", "OrderedDict"): OrderedDict,
    ("torch", "Size"): tuple,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): _NdArray,
    ("numpy.core.multiarray", "_reconstruct"): lambda cls, shape, typecode: _NdArray(),
    ("numpy._core.multiarray", "_reconstruct"): lambda cls, shape, typecode: _NdArray(),
    ("numpy.core.multiarray", "scalar"): _numpy_scalar,
    ("numpy._core.multiarray", "scalar"): _numpy_scalar,
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf: zipfile.ZipFile, root: str):
        super().__init__(f)
        self._zf, self._root = zf, root
        self._storages: dict[str, np.ndarray] = {}
        self._stubs: dict[str, type] = {}

    def find_class(self, module: str, name: str):
        hit = _SAFE.get((module, name))
        if hit is not None:
            return hit
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        if module == "torch" and name == "BFloat16Storage":
            raise pickle.UnpicklingError("bfloat16 tensors are not supported")
        key = f"{module}.{name}"
        if key not in self._stubs:
            self._stubs[key] = type(name, (Stub,), {"_qual": key})
        return self._stubs[key]

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and isinstance(pid[1], _StorageType)):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        key = str(pid[2])
        if key not in self._storages:
            self._storages[key] = np.frombuffer(self._zf.read(f"{self._root}/data/{key}"),
                                                dtype=_STORAGE_DTYPES[pid[1].name])
        return self._storages[key]


def load_object(path: str | Path) -> Any:
    """The unpickled object of a zip-format torch checkpoint, with stubs in place of every class."""
    zf = zipfile.ZipFile(str(path))
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")]
    if len(pkl) != 1:
        raise ValueError(f"{path}: not a zip-format torch checkpoint (found {len(pkl)} data.pkl entries)")
    with zf.open(pkl[0]) as f:
        return unwrap(_Unpickler(f, zf, pkl[0][: -len("/data.pkl")]).load())


def unwrap(o: Any, _seen: set[int] | None = None) -> Any:
    """Replace the numpy-array holders by their arrays, recursively through containers and stubs."""
    seen = set() if _seen is None else _seen
    if isinstance(o, _NdArray):
        return o.value
    if id(o) in seen:
        return o
    if isinstance(o, dict):
        seen.add(id(o))
        for k in list(o):
            o[k] = unwrap(o[k], seen)
    elif isinstance(o, list):
        seen.add(id(o))
        o[:] = [unwrap(v, seen) for v in o]
    elif isinstance(o, tuple):
        return tuple(unwrap(v, seen) for v in o)
    elif isinstance(o, Stub):
        seen.add(id(o))
        if hasattr(o, "_state"):
            o._state = unwrap(o._state, seen)
    return o


@dataclass
class ModuleInfo:
    cls: str                                    # qualified class name, e.g. "torch.nn.modules.normalization.LayerNorm"
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)   # names in _modules, None entries included

    @property
    def kind(self) -> str:
        return self.cls.rsplit(".", 1)[-1]


_PLAIN = (bool, int, float, str, type(None))


def _plain(v: Any) -> bool:
    return isinstance(v, _PLAIN) or (isinstance(v, (tuple, list)) and all(isinstance(e, _PLAIN) for e in v))


def module_tree(obj: Any) -> tuple["OrderedDict[str, np.ndarray]", "OrderedDict[str, ModuleInfo]"]:
    """Walk a stubbed nn.Module the way `state_dict()` does (parameters, buffers, then children)."""
    tensors: OrderedDict[str, np.ndarray] = OrderedDict()
    modules: OrderedDict[str, ModuleInfo] = OrderedDict()

    def walk(m: Any, prefix: str) -> None:
        state = getattr(m, "_state", None)
        if not isinstance(m, Stub) or not isinstance(state, dict) or "_modules" not in state:
            raise ValueError(f"{prefix or '<root>'}: not an nn.Module ({type(m).__name__})")
        info = ModuleInfo(cls=type(m)._qual)
        info.attrs = {k: v for k, v in state.items() if not k.startswith("_") and _plain(v)}
        for group in ("_parameters", "_buffers"):
            for k, v in (state.get(group) or {}).items():
                if isinstance(v, np.ndarray):
                    tensors[prefix + k] = v
        modules[prefix[:-1]] = info
        for k, child in (state.get("_modules") or {}).items():
            info.children.append(k)
            if child is not None:
                walk(child, f"{prefix}{k}.")

    walk(obj, "")
    return tensors, modules


def root_attr(obj: Any, name: str, default: Any = None) -> Any:
    """A non-module attribute of the root object (e.g. a dict the training script attached)."""
    state = getattr(obj, "_state", None)
    return state.get(name, default) if isinstance(state, dict) else default
