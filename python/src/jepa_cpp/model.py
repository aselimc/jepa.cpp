"""The Pythonic layer: numpy in, numpy out, over the C API in :mod:`jepa_cpp._api`.

Nothing here reimplements the engine. Preprocessing is the library's own
``jepa_preprocess_image_rgb`` — the bit-exact torchvision-style resize the parity fixtures were
generated against — pooling is ``jepa_pool_*``, prediction is ``jepa_predict_mod``, and the only
work this module does is marshalling: turning paths and numpy arrays into ``jepa_input``, grouping
items into encoder graphs the way ``tools/jepa-embed.cpp`` does, copying ``jepa_output`` buffers
into numpy before releasing them, and turning a non-zero return code into an exception carrying
the text the library logged.
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _api
from ._api import jepa_input, jepa_output

__all__ = [
    "Classification",
    "JepaError",
    "Model",
    "devices",
    "system_info",
    "version",
]

# The families whose jepa_encode takes the frames of one clip rather than independent images
# (tools/jepa-embed.cpp `video_model`).
VIDEO_FAMILIES = frozenset({"vjepa", "vjepa2", "vjepa2_1", "levjepa"})
# ...and the ones whose model card feeds a still image as that image repeated to the trained clip
# length (tools/jepa-embed.cpp `repeats_still_image`). jepa_encode() itself never repeats.
STILL_REPEAT_FAMILIES = frozenset({"levjepa"})

_KV = {"auto": _api.JEPA_KV_AUTO, "f16": _api.JEPA_KV_F16, "f32": _api.JEPA_KV_F32}
_MODALITY = {
    "auto": _api.JEPA_MODALITY_AUTO,
    "video": _api.JEPA_MODALITY_VIDEO,
    "image": _api.JEPA_MODALITY_IMAGE,
}


class JepaError(RuntimeError):
    """A jepa.cpp call failed. The message is the text the library logged for it."""


def version() -> str:
    """``jepa_version()`` — the version of the loaded shared library."""
    return _api.jepa_version().decode()


def system_info() -> None:
    """``jepa_print_system_info()`` — one line on stderr."""
    _api.jepa_print_system_info()


def devices() -> list[dict[str, Any]]:
    """The GPU devices the ggml backend registry can see; empty on a CPU-only build."""
    out = []
    for i in range(_api.jepa_device_count()):
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        _api.jepa_device_memory(i, ctypes.byref(free), ctypes.byref(total))
        out.append(
            {
                "index": i,
                "name": _api.jepa_device_name(i).decode(),
                "description": _api.jepa_device_description(i).decode(),
                "free_bytes": free.value,
                "total_bytes": total.value,
            }
        )
    return out


def _parse_device(device: str | int | None) -> int | None:
    """``"cpu"`` / ``"cuda:0"`` / ``"gpu:1"`` / ``"2"`` / an int → the C device index."""
    if device is None:
        return None
    if isinstance(device, int):
        return device
    s = str(device).strip().lower()
    if s == "cpu":
        return -1
    for prefix in ("cuda:", "gpu:", "device:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    else:
        if s in ("cuda", "gpu"):
            return 0
    try:
        return int(s)
    except ValueError as exc:
        raise ValueError(
            f"cannot read device {device!r} — expected 'cpu', 'cuda:N', 'gpu:N' or an index"
        ) from exc


@dataclass(frozen=True)
class Classification:
    """One item through the attentive-pool head."""

    logits: np.ndarray  # [n_classes] float32, raw
    probs: np.ndarray  # [n_classes] float32, softmax(logits)
    pooled: np.ndarray  # [embed_dim] float32, the pooler output = classifier input
    labels: Sequence[str]

    def top(self, k: int = 5) -> list[tuple[int, str, float]]:
        """The k largest logits as ``(class index, label, probability)``, descending."""
        n = int(self.logits.size)
        k = max(0, min(int(k), n))
        idx = np.empty(k, dtype=np.int32)
        got = _api.jepa_top_k(
            self.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            k,
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return [
            (int(i), self.labels[i] if i < len(self.labels) else "", float(self.probs[i]))
            for i in idx[:got]
        ]


def _as_f32(a, name: str, ndim: int | tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(a, dtype=np.float32)
    if ndim is not None:
        want = (ndim,) if isinstance(ndim, int) else ndim
        if arr.ndim not in want:
            raise ValueError(
                f"{name} must be {' or '.join(f'{d}-D' for d in want)}, got {arr.ndim}-D"
            )
    return arr


def _f32_ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32_ptr(a: np.ndarray | None):
    if a is None:
        return None
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _output_to_numpy(out: jepa_output, free: bool = True) -> np.ndarray:
    """Copy a library-owned ``jepa_output`` into numpy and release the C buffer."""
    if not out.data:
        raise JepaError("the library returned an empty output")
    rows, dim = int(out.n_tokens), int(out.dim)
    arr = np.ctypeslib.as_array(out.data, shape=(rows, dim)).copy()
    if free:
        _api.jepa_free(ctypes.cast(out.data, ctypes.c_void_p))
        out.data = None
    return arr


def _view_output(rows: np.ndarray) -> jepa_output:
    """A ``jepa_output`` header over a numpy block the library only reads. No ownership."""
    return jepa_output(_f32_ptr(rows), rows.shape[0], rows.shape[1])


class Model:
    """A loaded GGUF model and the compute context that runs it.

    ``Model`` owns one ``jepa_context``, which is per-thread compute state in the C library, so
    calls are serialised on an internal lock: several Python threads may share a ``Model`` safely,
    but they will not run graphs concurrently. For real parallelism give each thread its own
    ``Model`` (or raise ``threads``, which parallelises inside one graph). That lock is what makes
    this class safe under the C thread contract stated in ``include/jepa.h`` — a model may be
    shared, a context may not — and the same contract is why ``jepa_error_text()`` reads back only
    the calling thread's message, which is what the error handling here relies on.

    >>> with Model("models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf", threads=8) as m:
    ...     feat = m.encode("cat.jpg", pool="cls")     # [384]
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        device: str | int | None = None,
        threads: int = 0,
        flash_attn: bool = True,
        flash_kv: str = "auto",
        verbose: bool = False,
        max_batch: int | None = None,
        mul_mat_prec_f32: bool | None = None,
    ) -> None:
        """Load ``path`` and build a context for it.

        ``device`` is ``"cpu"``, ``"cuda:N"``, ``"gpu:N"`` or a device index; ``None`` (the
        default) follows ``$JEPA_DEVICE`` exactly as the command-line tools do. ``threads = 0``
        means the hardware concurrency. ``flash_kv`` is ``"auto"``, ``"f16"`` or ``"f32"``.
        ``max_batch`` caps how many image items share one encoder graph (the C default is 32, or
        ``$JEPA_MAX_BATCH``); video models default to one clip per call, which is what
        ``jepa-embed`` does, because the library never folds clips into one graph anyway.
        """
        self._lock = threading.RLock()
        self._model = None
        self._ctx = None

        if flash_kv not in _KV:
            raise ValueError(f"flash_kv must be one of {sorted(_KV)}, got {flash_kv!r}")

        mp = _api.jepa_model_default_params()
        mp.verbose = bool(verbose)
        dev = _parse_device(device)
        if dev is not None:
            mp.device = dev

        self.path = os.fspath(path)
        _api.jepa_error_reset()
        model = _api.jepa_model_load_ex(self.path.encode(), ctypes.byref(mp))
        if not model:
            raise JepaError(_why(f"cannot load {self.path}"))
        self._model = model

        cp = _api.jepa_context_default_params()
        cp.n_threads = int(threads)
        cp.use_flash_attn = bool(flash_attn)
        cp.verbose = bool(verbose)
        cp.flash_kv = _KV[flash_kv]
        _api.jepa_error_reset()
        ctx = _api.jepa_context_new(self._mp, cp)
        if not ctx:
            _api.jepa_model_free(self._model)
            self._model = None
            raise JepaError(_why("cannot create a context"))
        self._ctx = ctx

        if mul_mat_prec_f32 is not None:
            _api.jepa_context_set_mul_mat_prec_f32(self._ctx, bool(mul_mat_prec_f32))
        if max_batch is not None:
            _api.jepa_context_set_max_batch(self._ctx, int(max_batch))
        elif self.is_video:
            # jepa-embed's default: grouping clips only inflates the working set (measured 3.9x
            # peak RSS for +16 % wall), because the library runs one clip per graph regardless.
            _api.jepa_context_set_max_batch(self._ctx, 1)

    # --- lifetime ------------------------------------------------------------------------------
    def close(self) -> None:
        """Free the context and the weights. Idempotent; further calls raise."""
        with self._lock:
            if self._ctx is not None:
                _api.jepa_context_free(self._ctx)
                self._ctx = None
            if self._model is not None:
                _api.jepa_model_free(self._model)
                self._model = None

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._model is None:
            return "<jepa_cpp.Model (closed)>"
        return (
            f"<jepa_cpp.Model {self.name!r} family={self.family} dim={self.embed_dim} "
            f"{self.file_type_name} on {self.device_name}>"
        )

    def _handle(self):
        """The live ``jepa_context *``. Every compute call goes through this."""
        if self._model is None or self._ctx is None:
            raise JepaError("this Model is closed")
        return self._ctx

    @property
    def _mp(self):
        """The live ``jepa_model *``. Every read of the file goes through this.

        Handing ctypes a NULL model would not raise, it would crash the interpreter inside the C
        library, so a closed Model has to be caught here rather than one frame further down.
        """
        if self._model is None:
            raise JepaError("this Model is closed")
        return self._model

    # --- introspection -------------------------------------------------------------------------
    @property
    def family(self) -> str:
        return _api.jepa_model_family(self._mp).decode()

    @property
    def name(self) -> str:
        return _api.jepa_model_name(self._mp).decode()

    @property
    def embed_dim(self) -> int:
        return _api.jepa_model_embed_dim(self._mp)

    @property
    def patch_size(self) -> int:
        return _api.jepa_model_patch_size(self._mp)

    @property
    def tubelet_size(self) -> int:
        return _api.jepa_model_tubelet_size(self._mp)

    @property
    def img_size(self) -> int:
        return _api.jepa_model_img_size(self._mp)

    @property
    def n_frames(self) -> int:
        return _api.jepa_model_n_frames(self._mp)

    @property
    def n_layer(self) -> int:
        return _api.jepa_model_n_layer(self._mp)

    @property
    def n_head(self) -> int:
        return _api.jepa_model_n_head(self._mp)

    @property
    def has_cls(self) -> bool:
        return bool(_api.jepa_model_has_cls(self._mp))

    @property
    def n_registers(self) -> int:
        return _api.jepa_model_n_registers(self._mp)

    @property
    def n_prefix_tokens(self) -> int:
        return _api.jepa_model_n_prefix_tokens(self._mp)

    @property
    def has_predictor(self) -> bool:
        return bool(_api.jepa_model_has_predictor(self._mp))

    @property
    def has_head(self) -> bool:
        return bool(_api.jepa_model_has_head(self._mp))

    @property
    def has_projector(self) -> bool:
        return bool(_api.jepa_model_has_projector(self._mp))

    @property
    def n_classes(self) -> int:
        return _api.jepa_model_n_classes(self._mp)

    @property
    def labels(self) -> list[str]:
        out = []
        for i in range(self.n_classes):
            s = _api.jepa_model_label(self._mp, i)
            out.append(s.decode() if s else "")
        return out

    @property
    def file_type(self) -> int:
        return _api.jepa_model_file_type(self._mp)

    @property
    def file_type_name(self) -> str:
        return _api.jepa_model_file_type_name(self._mp).decode()

    @property
    def n_bytes(self) -> int:
        return _api.jepa_model_n_bytes(self._mp)

    @property
    def device(self) -> int:
        """-1 for the CPU, otherwise the GPU device index the weights live on."""
        return _api.jepa_model_device(self._mp)

    @property
    def device_name(self) -> str:
        return _api.jepa_model_device_name(self._mp).decode()

    @property
    def is_gpu(self) -> bool:
        return bool(_api.jepa_model_is_gpu(self._mp))

    @property
    def is_video(self) -> bool:
        """True for the families whose ``jepa_encode`` takes a clip rather than images."""
        return self.family in VIDEO_FAMILIES

    @property
    def threads(self) -> int:
        return _api.jepa_context_n_threads(self._handle())

    @property
    def max_batch(self) -> int:
        return _api.jepa_context_max_batch(self._handle())

    @max_batch.setter
    def max_batch(self, n: int) -> None:
        _api.jepa_context_set_max_batch(self._handle(), int(n))

    @property
    def last_batch(self) -> int:
        """Items per graph the last :meth:`encode` actually used."""
        return _api.jepa_context_last_batch(self._handle())

    @property
    def last_compute_ms(self) -> float:
        """Graph-compute wall time of the last encode / projector call."""
        return _api.jepa_context_last_compute_ms(self._handle())

    @property
    def mul_mat_prec_f32(self) -> bool:
        return bool(_api.jepa_context_mul_mat_prec_f32(self._handle()))

    @mul_mat_prec_f32.setter
    def mul_mat_prec_f32(self, on: bool) -> None:
        _api.jepa_context_set_mul_mat_prec_f32(self._handle(), bool(on))

    @property
    def lewm_n_frames(self) -> int:
        """The LeWM predictor's context window."""
        return _api.jepa_lewm_n_frames(self._mp)

    @property
    def lewm_action_dim(self) -> int:
        return _api.jepa_lewm_action_dim(self._mp)

    @property
    def preprocess_params(self) -> _api.jepa_preprocess_params:
        """The model's ``jepa.pre.*`` pipeline, as the editable C struct."""
        return _api.jepa_preprocess_default_params(self._mp)

    def token_grid(self, n_frames: int, height: int, width: int) -> tuple[int, int, int, int]:
        """``(n_tokens, grid_t, grid_h, grid_w)`` for one T x H x W item.

        ``n_tokens`` is 0 when the shape is not encodable by this model.
        """
        gt, gh, gw = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        n = _api.jepa_token_grid(
            self._mp,
            int(n_frames),
            int(height),
            int(width),
            ctypes.byref(gt),
            ctypes.byref(gh),
            ctypes.byref(gw),
        )
        return int(n), gt.value, gh.value, gw.value

    # --- preprocessing -------------------------------------------------------------------------
    def preprocess(self, item) -> np.ndarray:
        """One image or clip → the normalized ``[3, T, H, W]`` float32 the encoder takes.

        ``item`` is a path, an ``uint8`` ``[H, W, 3]`` array, an ``uint8`` ``[T, H, W, 3]`` array,
        or a sequence of those (the frames of one clip; they may differ in size, each is resized
        and cropped on its own). Frame handling follows ``jepa-embed``: a still image through a
        family whose card prescribes it is repeated to the trained clip length, and a clip that is
        not a multiple of the tubelet size gets its last frame repeated.
        """
        frames = _frames_of(item)
        return self._preprocess_frames(frames)

    def _preprocess_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("no frames to preprocess")
        video = self.is_video
        want = self.n_frames
        tubelet = self.tubelet_size
        repeat_first = 0
        if video and want > 1 and self.family in STILL_REPEAT_FAMILIES and len(frames) == 1:
            repeat_first = want
            frames = frames * want
        elif (
            video
            and tubelet > 1
            and len(frames) % tubelet != 0
            and self.token_grid(len(frames), self.img_size, self.img_size)[0] == 0
        ):
            frames = frames + [frames[-1]] * (tubelet - len(frames) % tubelet)

        n_t = len(frames)
        n_pre = 1 if repeat_first else n_t
        out: np.ndarray | None = None
        for t in range(n_pre):
            plane = self._preprocess_one(frames[t])
            if out is None:
                out = np.empty((3, n_t) + plane.shape[1:], dtype=np.float32)
            elif plane.shape[1:] != out.shape[2:]:
                raise ValueError(
                    f"frame {t} preprocesses to {plane.shape[1:]} but frame 0 to {out.shape[2:]}"
                )
            out[:, t] = plane
        assert out is not None
        if repeat_first:
            out[:, 1:] = out[:, :1]
        return out

    def _preprocess_one(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"a frame must be uint8 [H, W, 3], got {rgb.shape}")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        oh, ow = ctypes.c_int(0), ctypes.c_int(0)
        _api.jepa_error_reset()
        p = _api.jepa_preprocess_image_rgb(
            self._mp,
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            h,
            w,
            ctypes.byref(oh),
            ctypes.byref(ow),
        )
        if not p:
            raise JepaError(_why(f"cannot preprocess a {h}x{w} frame"))
        try:
            return np.ctypeslib.as_array(p, shape=(3, oh.value, ow.value)).copy()
        finally:
            _api.jepa_free(ctypes.cast(p, ctypes.c_void_p))

    @staticmethod
    def load_image(path: str | os.PathLike) -> np.ndarray:
        """Decode an image file to ``uint8 [H, W, 3]`` with the library's own decoder."""
        h, w = ctypes.c_int(0), ctypes.c_int(0)
        _api.jepa_error_reset()
        p = _api.jepa_load_image_rgb(os.fspath(path).encode(), ctypes.byref(h), ctypes.byref(w))
        if not p:
            raise JepaError(_why(f"cannot read {os.fspath(path)}"))
        try:
            return np.ctypeslib.as_array(p, shape=(h.value, w.value, 3)).copy()
        finally:
            _api.jepa_free(ctypes.cast(p, ctypes.c_void_p))

    # --- encoding ------------------------------------------------------------------------------
    def encode(self, x, *, pool: str | None = None, as_video: bool | None = None) -> np.ndarray:
        """Encode one item or a batch of them.

        ``x`` is a path, an ``uint8 [H, W, 3]`` image, an ``uint8 [T, H, W, 3]`` stack of frames,
        an already preprocessed ``float32 [C, T, H, W]`` item (or ``float32 [N, C, T, H, W]``
        batch, which is ``jepa_input``'s own layout), or a list of any of those.

        A list — and a ``[T, H, W, 3]`` stack — is one clip for a video family and a batch of
        independent items for an image family, which is the rule ``jepa-embed`` uses; ``as_video``
        forces either reading.

        Returns ``[n_tokens, dim]`` for one item and ``[batch, n_tokens, dim]`` for a batch; with
        ``pool`` set to ``"mean"``, ``"cls"`` or ``"lewm"``, ``[dim]`` and ``[batch, dim]``.
        """
        if pool is not None and pool not in ("mean", "cls", "lewm"):
            raise ValueError(f"pool must be None, 'mean', 'cls' or 'lewm', got {pool!r}")
        if pool == "cls" and not self.has_cls:
            raise ValueError(f"{self.name} has no CLS token — use pool='mean'")
        if pool == "lewm" and not self.has_projector:
            raise ValueError(f"{self.name} has no enc.proj projector")

        items, batched = self._items_of(x, as_video)
        rows = self._encode_items(items)
        if pool is not None:
            rows = [self._pool_one(r, pool) for r in rows]
        return np.stack(rows) if batched else rows[0]

    def pool(self, tokens: np.ndarray, mode: str = "mean") -> np.ndarray:
        """Pool an encoder output: ``"mean"`` over patch tokens, ``"cls"``, or ``"lewm"``.

        ``tokens`` is one item's ``[n_tokens, dim]`` (→ ``[dim]``) or a batch's
        ``[batch, n_tokens, dim]`` (→ ``[batch, dim]``).
        """
        arr = _as_f32(tokens, "tokens", (2, 3))
        if arr.ndim == 3:
            return np.stack([self._pool_one(a, mode) for a in arr])
        return self._pool_one(arr, mode)

    def _pool_one(self, rows: np.ndarray, mode: str) -> np.ndarray:
        rows = _as_f32(rows, "tokens", 2)
        enc = _view_output(rows)
        out = jepa_output()
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            if mode == "mean":
                rc = _api.jepa_pool_mean(self._mp, ctypes.byref(enc), ctypes.byref(out))
            elif mode == "cls":
                rc = _api.jepa_pool_cls(self._mp, ctypes.byref(enc), ctypes.byref(out))
            elif mode == "lewm":
                rc = _api.jepa_lewm_project(ctx, ctypes.byref(enc), ctypes.byref(out))
            else:
                raise ValueError(f"pool mode must be 'mean', 'cls' or 'lewm', got {mode!r}")
            if rc != 0:
                raise JepaError(_why(f"pooling ({mode}) failed"))
            return _output_to_numpy(out)[0]

    def _encode_items(self, items: list[np.ndarray]) -> list[np.ndarray]:
        """Encode preprocessed ``[3, T, H, W]`` items, grouping equal shapes into one graph."""
        out: list[np.ndarray] = []
        i = 0
        with self._lock:
            ctx = self._handle()
            cap = max(1, _api.jepa_context_max_batch(ctx))
            while i < len(items):
                j = i + 1
                while j < len(items) and j - i < cap and items[j].shape == items[i].shape:
                    j += 1
                group = items[i:j]
                c, t, h, w = group[0].shape
                block = np.ascontiguousarray(np.stack(group))
                inp = jepa_input(_f32_ptr(block), len(group), c, t, h, w)
                enc = jepa_output()
                _api.jepa_error_reset()
                if _api.jepa_encode(ctx, ctypes.byref(inp), ctypes.byref(enc)) != 0:
                    raise JepaError(
                        _why(f"encoding {len(group)} item(s) of shape {group[0].shape}")
                    )
                rows = _output_to_numpy(enc)
                if rows.shape[0] % len(group) != 0:
                    raise JepaError(
                        f"the encoder returned {rows.shape[0]} rows for {len(group)} items"
                    )
                per = rows.shape[0] // len(group)
                out += [rows[k * per : (k + 1) * per] for k in range(len(group))]
                i = j
        return out

    def _items_of(self, x, as_video: bool | None) -> tuple[list[np.ndarray], bool]:
        """Normalize any accepted input into preprocessed items + whether it was a batch."""
        if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 5:
            # already preprocessed, in jepa_input's own NCTHW layout
            return [np.ascontiguousarray(e) for e in x], True
        if isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.ndim == 4:
            # [T, H, W, 3]: the frames of one clip for a video family, which is the only family
            # that has clips; a stack of independent images for the image families.
            if as_video if as_video is not None else self.is_video:
                return [self._preprocess_frames(list(x))], False
            return [self._as_item(f) for f in x], True
        if _is_one(x):
            return [self._as_item(x)], False
        seq = list(x)
        if not seq:
            raise ValueError("nothing to encode")
        clip = as_video if as_video is not None else (self.is_video and len(seq) > 1)
        if clip:
            frames: list[np.ndarray] = []
            for e in seq:
                frames += _frames_of(e)
            return [self._preprocess_frames(frames)], False
        return [self._as_item(e) for e in seq], True

    def _as_item(self, e) -> np.ndarray:
        if isinstance(e, np.ndarray) and e.dtype == np.float32 and e.ndim == 4:
            return np.ascontiguousarray(e)  # already preprocessed [3, T, H, W]
        return self._preprocess_frames(_frames_of(e))

    # --- classification ------------------------------------------------------------------------
    def classify(self, x, *, as_video: bool | None = None):
        """Run the attentive-pool head. One item → :class:`Classification`, a batch → a list.

        Raises if the file has no head (``model.has_head``).
        """
        if not self.has_head:
            raise JepaError(f"{self.name} has no classification head")
        items, batched = self._items_of(x, as_video)
        rows = self._encode_items(items)
        labels = self.labels
        res = [self._head_one(r, labels) for r in rows]
        return res if batched else res[0]

    def _head_one(self, rows: np.ndarray, labels: Sequence[str]) -> Classification:
        # `block` has to outlive the call: _view_output only keeps the pointer, and _as_f32 may
        # have made a converted copy that nothing else references.
        block = _as_f32(rows, "tokens", 2)
        enc = _view_output(block)
        pooled = jepa_output()
        logits = jepa_output()
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            rc = _api.jepa_head_ex(
                ctx, ctypes.byref(enc), ctypes.byref(pooled), ctypes.byref(logits)
            )
            if rc != 0:
                raise JepaError(_why("the attentive-pool head failed"))
            p = _output_to_numpy(pooled)[0]
            lg = _output_to_numpy(logits)[0]
        probs = np.empty_like(lg)
        _api.jepa_softmax(_f32_ptr(lg), int(lg.size), _f32_ptr(probs))
        return Classification(logits=lg, probs=probs, pooled=p, labels=labels)

    # --- masked predictor ----------------------------------------------------------------------
    def predict(
        self,
        tokens,
        target: Iterable[int] | None = None,
        context: Iterable[int] | None = None,
        *,
        n_target: int | None = None,
        n_context: int | None = None,
        mask_index: int = 1,
        modality: str = "video",
    ) -> np.ndarray:
        """Predict encoder features at ``target`` token ids given ``context`` ones.

        ``tokens`` is one item's encoder output ``[n_tokens, dim]`` (what :meth:`encode` returns
        with ``pool=None``). ``context`` and ``target`` are token ids into that grid; passing
        ``None`` for either means ``0..n-1``, in which case give the count as ``n_context`` /
        ``n_target``. ``modality`` selects V-JEPA 2.1's ``pred.mod_embed_img`` vs
        ``pred.mod_embed_video`` — ``"video"`` (the C default of ``jepa_predict`` /
        ``jepa_predict_ex``), ``"image"``, or ``"auto"``; the wrong one costs ~0.14 cosine, so a
        2-frame clip must say ``"video"`` explicitly. Returns ``[n_target, pred_out_dim]``.
        """
        if modality not in _MODALITY:
            raise ValueError(f"modality must be one of {sorted(_MODALITY)}, got {modality!r}")
        rows = _as_f32(tokens, "tokens", 2)
        tgt = None if target is None else np.ascontiguousarray(target, dtype=np.int32).ravel()
        ctxi = None if context is None else np.ascontiguousarray(context, dtype=np.int32).ravel()
        n_t = int(n_target if tgt is None else tgt.size)
        n_c = int((n_context or 0) if ctxi is None else ctxi.size)
        if tgt is None and n_target is None:
            raise ValueError("give target ids or n_target")

        enc = _view_output(rows)
        out = jepa_output()
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            rc = _api.jepa_predict_mod(
                ctx,
                ctypes.byref(enc),
                _i32_ptr(ctxi),
                n_c,
                _i32_ptr(tgt),
                n_t,
                int(mask_index),
                _MODALITY[modality],
                ctypes.byref(out),
            )
            if rc != 0:
                raise JepaError(_why("the masked predictor failed"))
            return _output_to_numpy(out)

    # --- LeWM world model ----------------------------------------------------------------------
    def lewm_project(self, tokens) -> np.ndarray:
        """``enc.proj(CLS)`` for one item's encoder output ``[n_tokens, dim]`` → ``[dim]``."""
        return self._pool_one(_as_f32(tokens, "tokens", 2), "lewm")

    def lewm_project_rows(self, cls_rows) -> np.ndarray:
        """``enc.proj`` over explicit CLS rows ``[n, dim]`` → ``[n, dim]``."""
        rows = _as_f32(cls_rows, "cls_rows", (1, 2))
        if rows.ndim == 1:
            rows = rows[None]
        out = jepa_output()
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            rc = _api.jepa_lewm_project_rows(ctx, _f32_ptr(rows), rows.shape[0], ctypes.byref(out))
            if rc != 0:
                raise JepaError(_why("jepa_lewm_project_rows failed"))
            return _output_to_numpy(out)

    def lewm_predict(self, embs, actions) -> np.ndarray:
        """One predictor call over up to :attr:`lewm_n_frames` consecutive frames.

        ``embs`` is ``[n_frames, dim]`` of projected embeddings and ``actions``
        ``[n_frames, action_dim]``. The attention is causal over frames, so row *t* is the
        prediction after frames 0..t and the last row is the next-frame prediction.
        """
        e = _as_f32(embs, "embs", (1, 2))
        if e.ndim == 1:
            e = e[None]
        a = _as_f32(actions, "actions", (1, 2))
        if a.ndim == 1:
            a = a[None]
        if a.shape[0] != e.shape[0]:
            raise ValueError(f"{e.shape[0]} embeddings but {a.shape[0]} actions")
        out = jepa_output()
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            rc = _api.jepa_lewm_predict(
                ctx, _f32_ptr(e), _f32_ptr(a), e.shape[0], ctypes.byref(out)
            )
            if rc != 0:
                raise JepaError(_why("jepa_lewm_predict failed"))
            return _output_to_numpy(out)

    def lewm_rollout(self, embs, actions, n_steps: int | None = None) -> np.ndarray:
        """Autoregressive rollout: ``[n_steps, dim]``, predictions fed back as frames.

        ``embs`` is ``[n_seed, dim]`` (or one ``[dim]`` seed) and ``actions``
        ``[n_steps, action_dim]``; action *k* drives step *k*. ``n_steps`` defaults to the number
        of actions given.
        """
        e = _as_f32(embs, "embs", (1, 2))
        if e.ndim == 1:
            e = e[None]
        a = _as_f32(actions, "actions", (1, 2))
        if a.ndim == 1:
            a = a[None]
        steps = int(a.shape[0] if n_steps is None else n_steps)
        if steps < 1:
            raise ValueError("n_steps must be >= 1")
        if a.shape[0] < steps:
            raise ValueError(f"{steps} steps need {steps} actions, got {a.shape[0]}")
        out = np.empty((steps, e.shape[1]), dtype=np.float32)
        with self._lock:
            ctx = self._handle()
            _api.jepa_error_reset()
            rc = _api.jepa_lewm_rollout(
                ctx, _f32_ptr(e), e.shape[0], _f32_ptr(a), steps, _f32_ptr(out)
            )
            if rc != 0:
                raise JepaError(_why("jepa_lewm_rollout failed"))
        return out


# --- input normalization ---------------------------------------------------------------------
def _is_one(x) -> bool:
    """True when ``x`` is a single item rather than a sequence of them."""
    if isinstance(x, (str, bytes, os.PathLike)):
        return True
    if isinstance(x, np.ndarray):
        return True
    return False


def _frames_of(item) -> list[np.ndarray]:
    """One accepted item → its frames as ``uint8 [H, W, 3]`` arrays."""
    if isinstance(item, (str, os.PathLike)):
        return [Model.load_image(item)]
    if isinstance(item, np.ndarray):
        if item.dtype == np.uint8 and item.ndim == 3:
            return [item]
        if item.dtype == np.uint8 and item.ndim == 4:
            return [item[t] for t in range(item.shape[0])]
        raise ValueError(
            f"an image must be uint8 [H, W, 3] and a clip uint8 [T, H, W, 3], got "
            f"{item.dtype} {item.shape}"
        )
    if isinstance(item, (list, tuple)):
        frames: list[np.ndarray] = []
        for e in item:
            frames += _frames_of(e)
        return frames
    raise TypeError(f"cannot read an image or clip from {type(item).__name__}")


def _why(what: str) -> str:
    """``what`` plus whatever the library logged since the last reset."""
    text = _api.jepa_error_text()
    text = text.decode(errors="replace").strip() if text else ""
    return f"{what}: {text}" if text else what
