"""A client for ``jepa-server``, the HTTP front end in ``tools/jepa-server.cpp``.

Standard library only — :mod:`urllib.request` and :mod:`json` — plus the numpy the package already
depends on, so talking to a server adds nothing to the wheel. The engine is on the other end of the
socket: this module marshals, it never computes.

    >>> from jepa_cpp.client import Client
    >>> c = Client("http://127.0.0.1:8080")
    >>> c.health()["family"]
    'hfvit'
    >>> c.embed("cat.jpg").shape                      # doctest: +SKIP
    (1, 384)
    >>> c.embed([clip(["f0.jpg", "f1.jpg"])]).shape   # doctest: +SKIP
    (1, 1024)

An *item* is one thing to encode, and may be:

* a path (:class:`str` or :class:`os.PathLike`) or raw :class:`bytes` — one still image, read here
  and sent base64, so the server needs no ``--allow-local-files``;
* ``{"b64": "..."}`` or ``{"path": "..."}`` — the wire forms, passed through unchanged
  (``path`` names a file on the *server*, which must have been started with
  ``--allow-local-files``);
* ``{"frames": [item, ...]}`` — the frames of one clip, which :func:`clip` builds for you.

A list is always a list of items; the frames of one clip go inside ``frames``. That is the server's
own rule, and keeping it here means what a caller writes is what goes on the wire.

Vectors come back through ``encoding_format: "base64"`` by default: the little-endian float32 bytes,
which are smaller than the JSON array and settle any question about decimal round-trips.
``encoding_format="float"`` selects the OpenAI-compatible array of numbers, which round-trips the
same bits — the server writes the shortest decimal that reads back as the same value.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from .model import JepaError

__all__ = ["Client", "ServerError", "clip", "latents"]


class ServerError(JepaError):
    """A ``jepa-server`` request failed.

    The message is the server's own ``error.message`` where there was one — which for an engine
    failure is the text ``jepa_error_text()`` captured on the worker thread — and the transport
    error otherwise. :attr:`status` is the HTTP status, or ``None`` when the request never
    arrived.
    """

    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def clip(frames) -> dict[str, list]:
    """The frames of ONE clip as a single item: ``clip(["f0.jpg", "f1.jpg"])``.

    Without this, a list is read as several separate items — which is what you want for images and
    never what you want for a video model.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("a clip needs at least one frame")
    return {"frames": [_frame(f) for f in frames]}


# --- input normalization -------------------------------------------------------------------------
def _frame(x) -> dict[str, str]:
    """One frame in the wire form the server takes."""
    if isinstance(x, dict):
        if not ({"b64", "path"} & x.keys()):
            raise ValueError('a frame dict needs "b64" or "path"')
        return x
    if isinstance(x, bytes):
        return {"b64": base64.b64encode(x).decode("ascii")}
    if isinstance(x, (str, os.PathLike)):
        with open(x, "rb") as f:
            return {"b64": base64.b64encode(f.read()).decode("ascii")}
    raise TypeError(f"cannot use {type(x).__name__} as an image; pass a path, bytes or a dict")


def _is_one(x) -> bool:
    """True when ``x`` is a single item rather than a sequence of them."""
    return isinstance(x, (str, bytes, os.PathLike, dict))


def _items(x) -> list:
    """``x`` as the list of items the server's ``input`` field expects."""
    seq = [x] if _is_one(x) else list(x)
    if not seq:
        raise ValueError("no input items")
    out = []
    for item in seq:
        if isinstance(item, dict) and "frames" in item:
            out.append({"frames": [_frame(f) for f in item["frames"]]})
        else:
            out.append(_frame(item))
    return out


# --- vectors back ---------------------------------------------------------------------------------
def _vector(datum: dict) -> np.ndarray:
    """One ``data[]`` entry as float32, whichever encoding the server used.

    ``encoding_format: "base64"`` carries the values flat, so a multi-row vector (``pool="none"``)
    is reshaped with the ``dim`` the server reports beside it.
    """
    value = datum["embedding"]
    if isinstance(value, str):
        flat = np.frombuffer(base64.b64decode(value), dtype="<f4").astype(np.float32, copy=True)
        dim = int(datum.get("dim", flat.size)) or flat.size
        return flat.reshape(-1, dim) if flat.size > dim else flat
    return np.asarray(value, dtype=np.float32)


class Client:
    """Talks to one ``jepa-server``.

    :param base_url: where the server is, e.g. ``"http://127.0.0.1:8080"``.
    :param timeout: seconds to wait for a response. A ViT-g planning request is measured in
        seconds, so the default is generous.
    :param model: sent as the request's ``model`` field. A server holds one model and accepts
        either its name or no name at all, so this is a guard against pointing at the wrong
        process, not a selector.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 600.0,
        model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.model = model

    def __repr__(self) -> str:
        return f"Client({self.base_url!r}{', model=' + repr(self.model) if self.model else ''})"

    # --- transport ---------------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                raw = res.read()
                ctype = res.headers.get("Content-Type", "")
            if raw and "json" in ctype:
                return json.loads(raw)
            return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read()
            e.close()
            message, parsed = f"HTTP {e.code} from {url}", None
            try:
                parsed = json.loads(body)
                message = parsed["error"]["message"]
            except Exception:  # noqa: BLE001 - a non-JSON error body is still an error
                if body:
                    message += ": " + body.decode("utf-8", errors="replace")[:400]
            raise ServerError(message, status=e.code, body=parsed) from None
        except urllib.error.URLError as e:
            raise ServerError(f"cannot reach {url}: {e.reason}") from None

    def _body(self, payload: dict, model: str | None) -> dict:
        name = model if model is not None else self.model
        if name is not None:
            payload["model"] = name
        return payload

    # --- endpoints ---------------------------------------------------------------------------
    def health(self) -> dict:
        """``GET /health`` — the loaded model, the backend it runs on and the pool settings."""
        return self._request("GET", "/health")

    def models(self) -> list[dict]:
        """``GET /v1/models`` — the one model this server serves, OpenAI-shaped."""
        return self._request("GET", "/v1/models")["data"]

    def metrics(self) -> str:
        """``GET /metrics`` — the Prometheus exposition text, unparsed."""
        return self._request("GET", "/metrics")

    def embed(
        self,
        inputs,
        *,
        model: str | None = None,
        pool: str | None = None,
        encoding_format: str = "base64",
    ) -> np.ndarray:
        """``POST /v1/embeddings`` — items in, one float32 row per item out.

        :param inputs: one item or a sequence of them (see the module docstring).
        :param pool: ``"mean"``, ``"cls"``, ``"lewm"`` or ``"none"``. The default is the server's,
            which is the default ``jepa-embed`` picks: the CLS token when the model has one, the
            mean over patch tokens otherwise.
        :param encoding_format: ``"base64"`` (default) or ``"float"``. Both carry the same bits.
        :returns: ``[n_items, dim]``, or ``[n_items, n_tokens, dim]`` for ``pool="none"``.
        """
        payload: dict[str, Any] = {"input": _items(inputs), "encoding_format": encoding_format}
        if pool is not None:
            payload["pool"] = pool
        res = self._request("POST", "/v1/embeddings", self._body(payload, model))
        rows = [_vector(d) for d in res["data"]]
        return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)

    def embed_response(
        self,
        inputs,
        *,
        model: str | None = None,
        pool: str | None = None,
        encoding_format: str = "base64",
    ) -> dict:
        """:meth:`embed`, returning the whole response — for ``usage`` and the raw ``data`` list."""
        payload: dict[str, Any] = {"input": _items(inputs), "encoding_format": encoding_format}
        if pool is not None:
            payload["pool"] = pool
        return self._request("POST", "/v1/embeddings", self._body(payload, model))

    def classify(self, inputs, *, model: str | None = None, top_k: int = 5) -> list[list[dict]]:
        """``POST /classify`` — top-``k`` labels per item, highest probability first.

        :returns: one list of ``{"index", "label", "probability", "logit"}`` per input item.
        """
        payload: dict[str, Any] = {"input": _items(inputs), "top_k": int(top_k)}
        res = self._request("POST", "/classify", self._body(payload, model))
        return [d["predictions"] for d in res["data"]]

    def rollout(
        self,
        context,
        actions=None,
        *,
        goal=None,
        state=None,
        plan: dict | None = None,
        model: str | None = None,
        return_latents: bool = False,
    ) -> dict:
        """``POST /rollout`` — world-model rollout energies, and the CEM plan.

        :param context: the observed frame(s), as items. The last one is where planning starts.
        :param actions: ``[K][H][action_dim]`` candidate action sequences, or ``[H][action_dim]``
            for a single candidate. Optional when ``plan`` is given.
        :param goal: the goal frame. Energies are scored against it; without one they are scored
            against the last observed frame, which reads as how far the rollout has drifted.
        :param state: the seed pose (V-JEPA 2-AC).
        :param plan: the CEM parameters — ``samples``, ``topk``, ``cem_steps``, ``horizon``,
            ``maxnorm``, ``gripper_clamp``, ``seed`` — mirroring ``jepa_ac_plan``. Needs a goal.
        :param return_latents: also return the predicted latents, as ``{"shape", "b64"}``; decode
            with :func:`latents`.
        :returns: the response, with ``energies`` ``[K][H]``, ``best``, and ``plan`` when asked.
        """
        payload: dict[str, Any] = {"context": _items(context)}
        if goal is not None:
            payload["goal"] = (
                _frame(goal) if not isinstance(goal, dict) or "frames" not in goal else goal
            )
        if actions is not None:
            payload["actions"] = np.asarray(actions, dtype=np.float32).tolist()
        if state is not None:
            payload["state"] = [float(x) for x in state]
        if plan is not None:
            payload["plan"] = dict(plan)
        if return_latents:
            payload["return_latents"] = True
        return self._request("POST", "/rollout", self._body(payload, model))


def latents(response: dict) -> np.ndarray:
    """The ``latents`` block of a :meth:`Client.rollout` response as a float32 array.

    Shaped ``[n_candidates, horizon, rows, dim]`` — one row per token for V-JEPA 2-AC, one row per
    step for LeWM.
    """
    block = response["latents"]
    flat = np.frombuffer(base64.b64decode(block["b64"]), dtype="<f4").astype(np.float32, copy=True)
    return flat.reshape(tuple(int(n) for n in block["shape"]))
