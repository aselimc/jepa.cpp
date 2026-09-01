"""Raw ctypes bindings — ``include/jepa.h``, one Python name per C name, nothing else.

Every declaration in the header appears here with the same name, the same argument order and the
same types; no argument is defaulted, no return value is interpreted, no buffer is copied. The
Pythonic layer lives in :mod:`jepa_cpp.model`, and ``tests/test_api_coverage.py`` parses the header
and fails if anything here has drifted from it.

The convention the header sets is kept as it is: ``int`` results are 0 on success and -1 on
failure, pointer results are NULL on failure, and a ``jepa_output.data`` that came back from the
library belongs to the caller and must be released with :func:`jepa_free`.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_void_p,
)

from ._lib import load_library

__all__ = [
    "JEPA_KV_AUTO",
    "JEPA_KV_F16",
    "JEPA_KV_F32",
    "JEPA_MODALITY_AUTO",
    "JEPA_MODALITY_IMAGE",
    "JEPA_MODALITY_VIDEO",
    "JEPA_RESAMPLE_BICUBIC",
    "JEPA_RESAMPLE_BILINEAR",
    "JEPA_RESIZE_SHORTEST_EDGE",
    "JEPA_RESIZE_SQUASH",
    "FUNCTIONS",
    "jepa_context_params",
    "jepa_input",
    "jepa_model_params",
    "jepa_output",
    "jepa_preprocess_params",
    "lib",
]

# --- enums ------------------------------------------------------------------------------------
JEPA_KV_AUTO = 0
JEPA_KV_F16 = 1
JEPA_KV_F32 = 2

JEPA_RESAMPLE_BILINEAR = 0
JEPA_RESAMPLE_BICUBIC = 1

JEPA_RESIZE_SHORTEST_EDGE = 0
JEPA_RESIZE_SQUASH = 1

JEPA_MODALITY_AUTO = 0
JEPA_MODALITY_VIDEO = 1
JEPA_MODALITY_IMAGE = 2


# --- opaque handles ----------------------------------------------------------------------------
class jepa_model(ctypes.Structure):  # noqa: N801 - the C name
    """Opaque ``jepa_model``; only ever held as a pointer."""


class jepa_context(ctypes.Structure):  # noqa: N801 - the C name
    """Opaque ``jepa_context``; only ever held as a pointer."""


jepa_model_p = POINTER(jepa_model)
jepa_context_p = POINTER(jepa_context)


# --- plain structs -----------------------------------------------------------------------------
class jepa_context_params(ctypes.Structure):  # noqa: N801 - the C name
    _fields_ = [
        ("n_threads", c_int),
        ("use_flash_attn", c_bool),
        ("verbose", c_bool),
        ("flash_kv", c_int),
    ]


class jepa_input(ctypes.Structure):  # noqa: N801 - the C name
    _fields_ = [
        ("data", POINTER(c_float)),
        ("n_batch", c_int),
        ("n_chans", c_int),
        ("n_frames", c_int),
        ("height", c_int),
        ("width", c_int),
    ]


class jepa_output(ctypes.Structure):  # noqa: N801 - the C name
    _fields_ = [
        ("data", POINTER(c_float)),
        ("n_tokens", c_int64),
        ("dim", c_int64),
    ]


class jepa_preprocess_params(ctypes.Structure):  # noqa: N801 - the C name
    _fields_ = [
        ("mean", c_float * 3),
        ("std", c_float * 3),
        ("resize_short", c_int),
        ("crop", c_int),
        ("resample", c_int),
        ("resize_mode", c_int),
        ("rescale", c_float),
        ("fused_norm", c_bool),
    ]


class jepa_model_params(ctypes.Structure):  # noqa: N801 - the C name
    _fields_ = [
        ("verbose", c_bool),
        ("device", c_int),
    ]


u8_p = POINTER(c_uint8)
f32_p = POINTER(c_float)
i32_p = POINTER(c_int32)
int_p = POINTER(c_int)
size_p = POINTER(c_size_t)

# (name, restype, argtypes) for every function in include/jepa.h, in header order.
FUNCTIONS: list[tuple[str, object, list]] = [
    # --- lifecycle
    ("jepa_context_default_params", jepa_context_params, []),
    ("jepa_model_load", jepa_model_p, [c_char_p, c_bool]),
    ("jepa_model_free", None, [jepa_model_p]),
    ("jepa_context_new", jepa_context_p, [jepa_model_p, jepa_context_params]),
    ("jepa_context_free", None, [jepa_context_p]),
    ("jepa_context_n_threads", c_int, [jepa_context_p]),
    ("jepa_context_last_compute_ms", c_double, [jepa_context_p]),
    # --- introspection
    ("jepa_model_family", c_char_p, [jepa_model_p]),
    ("jepa_model_name", c_char_p, [jepa_model_p]),
    ("jepa_model_embed_dim", c_int, [jepa_model_p]),
    ("jepa_model_patch_size", c_int, [jepa_model_p]),
    ("jepa_model_tubelet_size", c_int, [jepa_model_p]),
    ("jepa_model_img_size", c_int, [jepa_model_p]),
    ("jepa_model_n_frames", c_int, [jepa_model_p]),
    ("jepa_model_n_layer", c_int, [jepa_model_p]),
    ("jepa_model_n_head", c_int, [jepa_model_p]),
    ("jepa_model_has_cls", c_bool, [jepa_model_p]),
    ("jepa_model_n_registers", c_int, [jepa_model_p]),
    ("jepa_model_n_prefix_tokens", c_int, [jepa_model_p]),
    ("jepa_model_has_predictor", c_bool, [jepa_model_p]),
    ("jepa_model_has_head", c_bool, [jepa_model_p]),
    ("jepa_model_has_projector", c_bool, [jepa_model_p]),
    ("jepa_model_n_classes", c_int, [jepa_model_p]),
    ("jepa_model_label", c_char_p, [jepa_model_p, c_int]),
    ("jepa_model_file_type", c_int, [jepa_model_p]),
    ("jepa_model_file_type_name", c_char_p, [jepa_model_p]),
    ("jepa_model_n_bytes", c_size_t, [jepa_model_p]),
    # --- preprocessing
    ("jepa_preprocess_default_params", jepa_preprocess_params, [jepa_model_p]),
    ("jepa_preprocess_image_file", f32_p, [jepa_model_p, c_char_p, int_p, int_p]),
    ("jepa_preprocess_image_rgb", f32_p, [jepa_model_p, u8_p, c_int, c_int, int_p, int_p]),
    (
        "jepa_preprocess_frames_rgb",
        f32_p,
        [jepa_model_p, POINTER(u8_p), c_int, c_int, c_int, int_p, int_p],
    ),
    (
        "jepa_preprocess_image_rgb_ex",
        f32_p,
        [POINTER(jepa_preprocess_params), u8_p, c_int, c_int, int_p, int_p],
    ),
    (
        "jepa_preprocess_frames_rgb_ex",
        f32_p,
        [POINTER(jepa_preprocess_params), POINTER(u8_p), c_int, c_int, c_int, int_p, int_p],
    ),
    ("jepa_load_image_rgb", u8_p, [c_char_p, int_p, int_p]),
    ("jepa_resize_antialias_u8", None, [u8_p, c_int, c_int, c_int, u8_p, c_int, c_int, c_int]),
    ("jepa_free", None, [c_void_p]),
    # --- inference
    ("jepa_encode", c_int, [jepa_context_p, POINTER(jepa_input), POINTER(jepa_output)]),
    ("jepa_pool_mean", c_int, [jepa_model_p, POINTER(jepa_output), POINTER(jepa_output)]),
    ("jepa_pool_cls", c_int, [jepa_model_p, POINTER(jepa_output), POINTER(jepa_output)]),
    ("jepa_lewm_project", c_int, [jepa_context_p, POINTER(jepa_output), POINTER(jepa_output)]),
    ("jepa_lewm_project_rows", c_int, [jepa_context_p, f32_p, c_int, POINTER(jepa_output)]),
    ("jepa_head", c_int, [jepa_context_p, POINTER(jepa_output), POINTER(jepa_output)]),
    (
        "jepa_predict",
        c_int,
        [jepa_context_p, POINTER(jepa_output), i32_p, c_int, i32_p, c_int, POINTER(jepa_output)],
    ),
    # --- video encoders and the attentive-pool head
    ("jepa_token_grid", c_int64, [jepa_model_p, c_int, c_int, c_int, int_p, int_p, int_p]),
    (
        "jepa_head_ex",
        c_int,
        [jepa_context_p, POINTER(jepa_output), POINTER(jepa_output), POINTER(jepa_output)],
    ),
    ("jepa_softmax", None, [f32_p, c_int, f32_p]),
    ("jepa_top_k", c_int, [f32_p, c_int, c_int, i32_p]),
    # --- misc
    ("jepa_version", c_char_p, []),
    ("jepa_print_system_info", None, []),
    # --- predictors and world model
    (
        "jepa_predict_ex",
        c_int,
        [
            jepa_context_p,
            POINTER(jepa_output),
            i32_p,
            c_int,
            i32_p,
            c_int,
            c_int,
            POINTER(jepa_output),
        ],
    ),
    (
        "jepa_predict_mod",
        c_int,
        [
            jepa_context_p,
            POINTER(jepa_output),
            i32_p,
            c_int,
            i32_p,
            c_int,
            c_int,
            c_int,
            POINTER(jepa_output),
        ],
    ),
    ("jepa_lewm_n_frames", c_int, [jepa_model_p]),
    ("jepa_lewm_action_dim", c_int, [jepa_model_p]),
    ("jepa_lewm_predict", c_int, [jepa_context_p, f32_p, f32_p, c_int, POINTER(jepa_output)]),
    ("jepa_lewm_rollout", c_int, [jepa_context_p, f32_p, c_int, f32_p, c_int, f32_p]),
    # --- V-JEPA 2-AC action-conditioned world model (jepa.pred.kind == "ac")
    ("jepa_ac_tokens_per_frame", c_int, [jepa_model_p]),
    ("jepa_ac_action_dim", c_int, [jepa_model_p]),
    ("jepa_ac_state_dim", c_int, [jepa_model_p]),
    ("jepa_ac_max_frames", c_int, [jepa_model_p]),
    ("jepa_ac_normalize_reps", c_bool, [jepa_model_p]),
    ("jepa_ac_normalize", None, [jepa_model_p, f32_p, c_int64, c_int64]),
    ("jepa_ac_predict", c_int,
     [jepa_context_p, f32_p, c_int, c_int, f32_p, f32_p, POINTER(jepa_output)]),
    ("jepa_ac_predict_all", c_int,
     [jepa_context_p, f32_p, c_int, c_int, f32_p, f32_p, POINTER(jepa_output)]),
    ("jepa_ac_rollout", c_int,
     [jepa_context_p, f32_p, c_int, f32_p, f32_p, f32_p, c_int, c_int, f32_p]),
    ("jepa_ac_next_state", None, [jepa_model_p, f32_p, f32_p, f32_p]),
    ("jepa_ac_energy", None, [f32_p, f32_p, c_int, c_int64, c_int64, f32_p]),
    # --- encoder batching
    ("jepa_context_set_max_batch", None, [jepa_context_p, c_int]),
    ("jepa_context_max_batch", c_int, [jepa_context_p]),
    ("jepa_context_last_batch", c_int, [jepa_context_p]),
    # --- backend / device selection
    ("jepa_model_default_params", jepa_model_params, []),
    ("jepa_model_load_ex", jepa_model_p, [c_char_p, POINTER(jepa_model_params)]),
    ("jepa_device_count", c_int, []),
    ("jepa_device_name", c_char_p, [c_int]),
    ("jepa_device_description", c_char_p, [c_int]),
    ("jepa_device_memory", None, [c_int, size_p, size_p]),
    ("jepa_model_device", c_int, [jepa_model_p]),
    ("jepa_model_device_name", c_char_p, [jepa_model_p]),
    ("jepa_model_is_gpu", c_bool, [jepa_model_p]),
    ("jepa_context_set_mul_mat_prec_f32", None, [jepa_context_p, c_bool]),
    ("jepa_context_mul_mat_prec_f32", c_bool, [jepa_context_p]),
    # --- diagnostics capture
    ("jepa_error_reset", None, []),
    ("jepa_error_text", c_char_p, []),
]

lib = load_library()

for _name, _restype, _argtypes in FUNCTIONS:
    _fn = getattr(lib, _name)
    _fn.restype = _restype
    _fn.argtypes = _argtypes
    globals()[_name] = _fn
    __all__.append(_name)

del _name, _restype, _argtypes, _fn
