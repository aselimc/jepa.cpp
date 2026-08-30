"""jepa_convert — PyTorch checkpoint -> jepa.cpp GGUF converters.

Families implemented here (see README.md in this directory):
  ijepa   facebook/ijepa_*        HF IJepaModel safetensors
  hfvit   OK-AI/lejepa-*          DINOv2/timm-style ViT ("ViTv2") safetensors
  lewm    quentinll/lewm-*        LeWorldModel weights.pt (HF ViT encoder + adaLN predictor)

vjepa2 / vjepa2_1 live in standalone modules written separately (they only import `gguf`).
The schema every converter implements is docs/gguf-schema.md.
"""

__all__ = ["common", "ijepa", "hfvit", "lewm"]
