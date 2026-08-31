# Quantization

`tools/jepa-quantize` turns an f32 / f16 jepa.cpp GGUF into a smaller one (f16, q8_0, q4_0, q4_1, q5_0, q5_1,
q4_k, q5_k, q6_k). This page has the rule, the CLI, the file sizes and the measured accuracy of every type against
the PyTorch references, plus a recommendation. All numbers below were produced without the C++ graph: the
quantized weights were dequantized in Python and pushed through the numpy reference graphs
(`scripts/gguf_dequant_selftest.py`), so they isolate the *weight* error from any kernel error.

## The rule

Only the 2-D weight matrices of the attention / FFN / projection / classifier layers change type
(`docs/gguf-schema.md`, "Quantization rules"):

| re-typed | kept in the type of the input file |
|---|---|
| `*.attn_qkv.weight`, `*.attn_q/k/v.weight`, `*.attn_out.weight`, `*.ffn_up.weight`, `*.ffn_down.weight` (encoder, predictor and head blocks) | `enc.patch_embed*`, `enc.pos_embed`, `enc.cls_token`, `enc.reg_tokens`, `enc.mod_embed_*`, `pred.mask_tokens`, `pred.pos_embed`, `pred.mod_embed_*` |
| `enc.proj.*.weight` (LeWM projector), `pred.proj*.weight` (incl. `pred.proj.0/2`, `pred.proj_context`), `pred.embed*.weight` (V-JEPA 2 / 2.1 context projection) | every LayerNorm (`ln1`, `ln2`, `norm`, `hier_norm`, `ln_kv`), every bias, `ls1/ls2` |
| `head.xattn.{q,k,v,ffn_up,ffn_down}.weight`, `head.cls.weight` | `pred.blk.*.adaln.*`, `pred.action_embed.*`, `pred.state_embed.*`, `head.query` |

`TYPE = f16` applies the same rule with F16 as the target (the converter's own `--ftype f16` rule: on
`lejepa-vits16-pretrain-in1k-f32.gguf` the tool reproduces `scripts/convert.py --ftype f16` **byte for byte**),
`TYPE = f32` widens / dequantizes every tensor. A tensor whose row length `ne[0]` is not a multiple of the block size
keeps its type: for the K-quants (block 256) the tool falls back **per tensor** to the legacy type of the same width
(`q4_k -> q4_0`, `q5_k -> q5_0`, `q6_k -> q8_0`) and prints it — e.g. all 384-wide matrices of ViT-S (`attn_qkv`,
`attn_out`, `ffn_up` have `ne[0] = 384`) and of the V-JEPA 2 predictors fall back, `ffn_down` (`ne[0] = 1536`) does
not; ViT-H (1280 / 5120) and ViT-L / ViT-B (1024 / 4096, 768 / 3072) never fall back. `general.file_type` is set to the
ggml ftype of the requested type (`7` for q8_0, `2` q4_0, `3` q4_1, `8` q5_0, `9` q5_1, `12` q4_k, `13` q5_k, `14`
q6_k, `1` f16, `0` f32) even when some tensors fell back ("mostly"). All other metadata is copied verbatim, in order.
Already-quantized tensors are refused unless `--allow-requant` is given (quantizing on top of a quantization is lossy
twice; `f32` / `f16` targets always dequantize).

## CLI

```
jepa-quantize IN.gguf OUT.gguf TYPE [-t threads] [--keep SUBSTR]... [--allow-requant] [--dry-run] [-v]
  TYPE           f32 | f16 | q8_0 | q4_0 | q4_1 | q5_0 | q5_1 | q4_k | q5_k | q6_k
  -t N           quantization threads (default min(32, hardware threads); rows of a matrix are split across threads)
  --keep SUBSTR  tensors whose name contains SUBSTR keep their type (repeatable), e.g. --keep ffn_down, --keep enc.blk.23.
  --allow-requant  re-quantize tensors that are already quantized
  --dry-run      print the plan (per-tensor types, fallbacks, byte summary), write nothing
  -v             also list the tensors that keep their type
```

```
$ build/jepa-quantize models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf models/gguf/lejepa-vits16-pretrain-in1k-q4_k.gguf q4_k -t 32
jepa-quantize: ...-f32.gguf (lejepa-vits16-pretrain-in1k, arch=jepa, file_type=0, 150 tensors, 33 kv) -> ...-q4_k.gguf [q4_k], 32 threads
  enc.blk.0.attn_qkv.weight     [1152,384]   f32 -> q4_0   1.7 MiB -> 243.0 KiB  fallback q4_K -> q4_0 (ne[0]=384 not a multiple of 256)
  enc.blk.0.attn_out.weight     [384,384]    f32 -> q4_0 576.0 KiB ->  81.0 KiB  fallback q4_K -> q4_0 (ne[0]=384 not a multiple of 256)
  enc.blk.0.ffn_up.weight       [1536,384]   f32 -> q4_0   2.2 MiB -> 324.0 KiB  fallback q4_K -> q4_0 (ne[0]=384 not a multiple of 256)
  enc.blk.0.ffn_down.weight     [384,1536]   f32 -> q4_K   2.2 MiB -> 324.0 KiB
  ...
plan: 48 tensors re-typed, 102 kept by the rule, 36 K-quant fallbacks, 0 re-quantized
tensor bytes by type, before:
  f32     150 tensors     86662656 bytes (82.6 MiB)
tensor bytes by type, after:
  f32     102 tensors      1728000 bytes (1.6 MiB)
  q4_0     36 tensors      8294400 bytes (7.9 MiB)
  q4_K     12 tensors      3649536 bytes (3.5 MiB)
tensor data: 86662656 bytes (82.6 MiB) -> 13671936 bytes (13.0 MiB), 0.158x
wrote ...-q4_k.gguf: 13681856 bytes (13.0 MiB), input 86672576 bytes (82.7 MiB), 0.158x, 0.1 s
```

Implementation notes (`tools/jepa-quantize.cpp`): gguf C API only (`gguf_init_from_file` with `no_alloc`, tensor data
is streamed from the input file tensor by tensor, so the peak memory is one tensor, not the model); the output is
written in two passes (`gguf_write_to_file(..., only_meta)` then the data appended with the GGUF alignment padding);
`ggml_quantize_chunk` does the quantization with the rows of each matrix split across `-t` threads; f16 / quantized
inputs are widened with the ggml type traits. The tool re-opens the written file and checks that every tensor has the
planned type and size. ViT-H/14 (1.2 GB f16) takes 1.5 s to q8_0 and 2.6 s to q4_k on 32 threads. Round trips are
exact: `f16 -> f16` on `ijepa_vith14_1k-f16.gguf` reproduces the input byte for byte.

## File sizes

Sizes of the files in `models/gguf/` (`jepa-quantize` from the f32 / f16 converter output). Ratios are against the
source file; q4_k and q4_0 (both 4.5 bit / weight) and q5_k and q5_0 (5.5) have identical sizes, so the K-quants only
ever buy accuracy, never bytes.

| model | source | f16 | q8_0 | q6_k | q5_k | q5_1 | q5_0 | q4_1 | q4_k | q4_0 |
|---|---|---|---|---|---|---|---|---|---|---|
| lejepa-vits16-pretrain-in1k (ViT-S/16) | f32 82.7 MiB | 42.2 | 23.2 (0.28x) | 21.5 | 15.6 | 16.8 | 15.6 | 14.3 | 13.0 | 13.0 (0.16x) |
| lewm-pusht (ViT-Ti/14 + predictor) | f32 68.8 MiB | 37.7 | 23.1 (0.34x) | 21.7 | — | — | 17.2 | — | 15.3 | 15.3 (0.22x) |
| ijepa_vith14_1k (ViT-H/14) | f16 1206 MiB | — | 644 (0.53x) | 498 | 419 | — | — | — | 344 | 344 (0.29x) |
| vjepa2_1-vitb-384 (ViT-B/16 + predictor) | f16 210 MiB | — | 113 (0.54x) | — | — | — | — | — | 62 | 62 (0.30x) |
| vjepa2-vitl-fpc64-256 (ViT-L/16 + predictor) | f16 623 MiB | — | 333 (0.53x) | — | — | — | — | — | 178 | 178 (0.29x) |
| vjepa2-vitl-fpc16-256-ssv2 (ViT-L/16 + attentive head) | f16 717 MiB | — | 383 (0.53x) | — | — | — | — | — | 205 | 205 (0.29x) |

The non-quantized remainder (patch embeddings, norms, biases, position tables) is 1.6 MiB for ViT-S, 6 MiB for ViT-H,
0.8-1.7 MiB for the V-JEPA 2 files, i.e. q8_0 is 0.53x of f16 and q4 is 0.29x, as expected from 8.5 / 4.5 bits per
weight (LeWM has a larger f32 remainder: adaLN, action embedder, position tables, 4 %).

## How the accuracy was measured

```
scripts/gguf_dequant_selftest.py --gguf models/gguf/<model>-<type>.gguf --ref tests/fixtures/ref/<model> [--samples N|names] [--json out.json]
```

reads the GGUF with the python `gguf` package, dequantizes every tensor (`gguf.quants.dequantize`), runs the numpy
graph of `scripts/jepa_convert/selftest.py` (`ijepa`, `hfvit`, `lewm`: encoder, LeWM projector / action embedder /
adaLN predictor) or of `scripts/jepa_convert/vjepa2_numpy_ref.py` (`vjepa2`, `vjepa2_1`: encoder with 3-D RoPE,
attentive-pool head, full-context predictor) on the **stored `input` tensors** of the reference set
(`tests/fixtures/README.md`, same preprocessed pixels as the PyTorch run) and compares with `scripts/compare.py`
metrics: per-token cosine (mean and worst token), `rel_max = max|a-b| / max|b|`, `rel_fro = ||a-b|| / ||b||`, and for
`logits` the top-1 / top-5 agreement. The exit status uses the Q8_0 threshold of `docs/architecture.md` (worst token
cosine >= 0.999); `--min-cos` relaxes it. The f32 rows of the tables are the floor of the method (numpy f32 vs
PyTorch f32: `rel_max` ~1e-5 on ViT-S, ~1e-6 on LeWM); the f16 rows are the converter output that the C++ engine
runs by default.

## Accuracy against the PyTorch references

Per tensor, aggregated over the samples: `cos mean` = mean over samples of the mean per-token cosine, `cos min` = worst token of the worst sample, `rel_max` / `rel_fro` = worst sample. The type column lists the tensor types actually present in the file (K-quant fallbacks show up as Q4_0 / Q5_0 / Q8_0 counts).

### LeJEPA ViT-S/16 (`hfvit`, source f32, 8 images)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f32** (F32:150) | 82.7 MiB | last_hidden_state | 8 | 1.000000 | 1.000000 | 1.09e-05 | 8.75e-06 |  |
|  |  | cls | 8 | 1.000000 | 1.000000 | 1.06e-06 | 1.44e-06 |  |
|  |  | pooled_mean | 8 | 1.000000 | 1.000000 | 5.61e-07 | 1.27e-06 |  |
| **f16** (F16:48,F32:102) | 42.2 MiB | last_hidden_state | 8 | 1.000000 | 1.000000 | 5.08e-04 | 5.75e-04 |  |
|  |  | cls | 8 | 1.000000 | 1.000000 | 3.25e-04 | 3.32e-04 |  |
|  |  | pooled_mean | 8 | 1.000000 | 1.000000 | 1.28e-04 | 2.91e-04 |  |
| **q8_0** (F32:102,Q8_0:48) | 23.2 MiB | last_hidden_state | 8 | 0.999875 | 0.999197 | 1.30e-02 | 1.71e-02 |  |
|  |  | cls | 8 | 0.999960 | 0.999952 | 6.87e-03 | 9.95e-03 |  |
|  |  | pooled_mean | 8 | 0.999973 | 0.999965 | 3.08e-03 | 8.34e-03 |  |
| **q6_k** (F32:102,Q6_K:12,Q8_0:36) | 21.5 MiB | last_hidden_state | 8 | 0.999431 | 0.997862 | 2.97e-02 | 3.69e-02 |  |
|  |  | cls | 8 | 0.999797 | 0.999756 | 2.34e-02 | 2.21e-02 |  |
|  |  | pooled_mean | 8 | 0.999860 | 0.999822 | 6.10e-03 | 1.89e-02 |  |
| **q5_k** (F32:102,Q5_0:36,Q5_K:12) | 15.6 MiB | last_hidden_state | 8 | 0.993131 | 0.976433 | 1.15e-01 | 1.31e-01 |  |
|  |  | cls | 8 | 0.997738 | 0.996823 | 8.94e-02 | 7.97e-02 |  |
|  |  | pooled_mean | 8 | 0.998519 | 0.997604 | 2.12e-02 | 6.98e-02 |  |
| **q5_1** (F32:102,Q5_1:48) | 16.8 MiB | last_hidden_state | 8 | 0.994324 | 0.981566 | 8.23e-02 | 1.18e-01 |  |
|  |  | cls | 8 | 0.997913 | 0.996885 | 7.13e-02 | 7.99e-02 |  |
|  |  | pooled_mean | 8 | 0.998712 | 0.998442 | 2.28e-02 | 5.58e-02 |  |
| **q5_0** (F32:102,Q5_0:48) | 15.6 MiB | last_hidden_state | 8 | 0.992123 | 0.969174 | 9.52e-02 | 1.36e-01 |  |
|  |  | cls | 8 | 0.997448 | 0.996915 | 5.85e-02 | 7.86e-02 |  |
|  |  | pooled_mean | 8 | 0.998335 | 0.997639 | 2.28e-02 | 6.87e-02 |  |
| **q4_1** (F32:102,Q4_1:48) | 14.3 MiB | last_hidden_state | 8 | 0.975445 | 0.912616 | 1.96e-01 | 2.37e-01 |  |
|  |  | cls | 8 | 0.990653 | 0.988645 | 1.58e-01 | 1.52e-01 |  |
|  |  | pooled_mean | 8 | 0.994229 | 0.993205 | 5.00e-02 | 1.17e-01 |  |
| **q4_k** (F32:102,Q4_0:36,Q4_K:12) | 13.0 MiB | last_hidden_state | 8 | 0.970211 | 0.843224 | 1.73e-01 | 2.65e-01 |  |
|  |  | cls | 8 | 0.988700 | 0.984663 | 1.77e-01 | 1.75e-01 |  |
|  |  | pooled_mean | 8 | 0.993358 | 0.992025 | 4.86e-02 | 1.26e-01 |  |
| **q4_0** (F32:102,Q4_0:48) | 13.0 MiB | last_hidden_state | 8 | 0.965778 | 0.830929 | 2.10e-01 | 2.79e-01 |  |
|  |  | cls | 8 | 0.987218 | 0.982443 | 1.43e-01 | 1.87e-01 |  |
|  |  | pooled_mean | 8 | 0.992347 | 0.990894 | 5.72e-02 | 1.35e-01 |  |

### LeWM PushT (`lewm`, source f32, 2 images + 3-frame rollout)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f32** (F32:243) | 68.8 MiB | last_hidden_state | 2 | 1.000000 | 1.000000 | 1.17e-06 | 6.24e-07 |  |
|  |  | cls | 2 | 1.000000 | 1.000000 | 4.12e-07 | 4.23e-07 |  |
|  |  | emb | 2 | 1.000000 | 1.000000 | 7.54e-07 | 7.21e-07 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 1.000000 | 1.000000 | 7.12e-07 | 7.17e-07 |  |
|  |  | emb_seq | 1 | 1.000000 | 1.000000 | 7.78e-07 | 7.44e-07 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 1.000000 | 1.000000 | 7.80e-07 | 7.62e-07 |  |
| **f16** (F16:76,F32:167) | 37.7 MiB | last_hidden_state | 2 | 1.000000 | 1.000000 | 1.75e-04 | 1.40e-04 |  |
|  |  | cls | 2 | 1.000000 | 1.000000 | 1.66e-04 | 1.64e-04 |  |
|  |  | emb | 2 | 1.000000 | 1.000000 | 3.06e-04 | 3.53e-04 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 1.000000 | 1.000000 | 3.00e-04 | 3.22e-04 |  |
|  |  | emb_seq | 1 | 1.000000 | 1.000000 | 3.35e-04 | 3.14e-04 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 1.000000 | 1.000000 | 3.27e-04 | 2.98e-04 |  |
| **q8_0** (F32:167,Q8_0:76) | 23.1 MiB | last_hidden_state | 2 | 0.999995 | 0.999985 | 4.35e-03 | 3.38e-03 |  |
|  |  | cls | 2 | 0.999991 | 0.999990 | 4.29e-03 | 4.49e-03 |  |
|  |  | emb | 2 | 0.999961 | 0.999954 | 8.76e-03 | 9.87e-03 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 0.999952 | 0.999933 | 1.00e-02 | 1.22e-02 |  |
|  |  | emb_seq | 1 | 0.999965 | 0.999954 | 8.20e-03 | 8.44e-03 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 0.999961 | 0.999933 | 8.36e-03 | 8.95e-03 |  |
| **q6_k** (F32:167,Q6_K:26,Q8_0:50) | 21.7 MiB | last_hidden_state | 2 | 0.999966 | 0.999912 | 9.60e-03 | 8.43e-03 |  |
|  |  | cls | 2 | 0.999958 | 0.999954 | 1.05e-02 | 9.56e-03 |  |
|  |  | emb | 2 | 0.999863 | 0.999839 | 1.86e-02 | 1.84e-02 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 0.999880 | 0.999879 | 1.82e-02 | 1.57e-02 |  |
|  |  | emb_seq | 1 | 0.999860 | 0.999839 | 1.95e-02 | 1.69e-02 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 0.999868 | 0.999859 | 1.74e-02 | 1.65e-02 |  |
| **q5_0** (F32:167,Q5_0:76) | 17.2 MiB | last_hidden_state | 2 | 0.999660 | 0.999146 | 3.21e-02 | 2.63e-02 |  |
|  |  | cls | 2 | 0.999512 | 0.999464 | 3.04e-02 | 3.27e-02 |  |
|  |  | emb | 2 | 0.998056 | 0.997984 | 6.60e-02 | 6.38e-02 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 0.998208 | 0.998199 | 5.59e-02 | 6.10e-02 |  |
|  |  | emb_seq | 1 | 0.998314 | 0.997984 | 6.18e-02 | 5.82e-02 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 0.998087 | 0.997575 | 6.19e-02 | 6.21e-02 |  |
| **q4_k** (F32:167,Q4_0:50,Q4_K:26) | 15.3 MiB | last_hidden_state | 2 | 0.998779 | 0.996803 | 6.88e-02 | 5.07e-02 |  |
|  |  | cls | 2 | 0.998297 | 0.998065 | 5.78e-02 | 6.22e-02 |  |
|  |  | emb | 2 | 0.993206 | 0.991479 | 1.12e-01 | 1.31e-01 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 0.991625 | 0.988878 | 1.33e-01 | 1.54e-01 |  |
|  |  | emb_seq | 1 | 0.992993 | 0.991479 | 1.23e-01 | 1.18e-01 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 0.992487 | 0.988878 | 1.11e-01 | 1.22e-01 |  |
| **q4_0** (F32:167,Q4_0:76) | 15.3 MiB | last_hidden_state | 2 | 0.998547 | 0.995504 | 6.58e-02 | 5.63e-02 |  |
|  |  | cls | 2 | 0.998194 | 0.997942 | 6.44e-02 | 6.42e-02 |  |
|  |  | emb | 2 | 0.992647 | 0.991803 | 1.21e-01 | 1.29e-01 |  |
|  |  | act_emb | 2 | 1.000000 | 1.000000 | 2.31e-07 | 1.37e-07 |  |
|  |  | pred_next | 2 | 0.991251 | 0.988378 | 1.59e-01 | 1.59e-01 |  |
|  |  | emb_seq | 1 | 0.992159 | 0.991183 | 1.29e-01 | 1.26e-01 |  |
|  |  | act_emb_seq | 1 | 1.000000 | 1.000000 | 2.31e-07 | 1.42e-07 |  |
|  |  | pred_seq | 1 | 0.991862 | 0.988378 | 1.33e-01 | 1.29e-01 |  |

### I-JEPA ViT-H/14 (`ijepa`, source f16, 2 images)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f16** (F16:128,F32:261) | 1206.2 MiB | last_hidden_state | 2 | 0.999999 | 0.999532 | 8.60e-03 | 2.58e-03 |  |
|  |  | pooled_mean | 2 | 1.000000 | 1.000000 | 3.28e-04 | 3.62e-04 |  |
| **q8_0** (F32:261,Q8_0:128) | 643.7 MiB | last_hidden_state | 2 | 0.999767 | 0.988348 | 9.40e-02 | 2.86e-02 |  |
|  |  | pooled_mean | 2 | 0.999974 | 0.999973 | 6.45e-03 | 7.69e-03 |  |
| **q6_k** (F32:261,Q6_K:128) | 498.4 MiB | last_hidden_state | 2 | 0.998199 | 0.913993 | 1.47e-01 | 7.48e-02 |  |
|  |  | pooled_mean | 2 | 0.999719 | 0.999663 | 2.01e-02 | 2.75e-02 |  |
| **q5_k** (F32:261,Q5_K:128) | 418.7 MiB | last_hidden_state | 2 | 0.993854 | 0.668111 | 3.00e-01 | 1.34e-01 |  |
|  |  | pooled_mean | 2 | 0.998895 | 0.998861 | 4.42e-02 | 4.78e-02 |  |
| **q4_k** (F32:261,Q4_K:128) | 343.7 MiB | last_hidden_state | 2 | 0.979426 | 0.525752 | 3.29e-01 | 2.62e-01 |  |
|  |  | pooled_mean | 2 | 0.995850 | 0.995366 | 9.15e-02 | 9.62e-02 |  |
| **q4_0** (F32:261,Q4_0:128) | 343.7 MiB | last_hidden_state | 2 | 0.972290 | 0.433550 | 7.25e-01 | 2.90e-01 |  |
|  |  | pooled_mean | 2 | 0.993563 | 0.993280 | 1.05e-01 | 1.16e-01 |  |

### V-JEPA 2.1 ViT-B/16 384 (`vjepa2_1`, source f16, 2 clips x 16 frames + 2 images)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f16** (F16:101,F32:214) | 209.7 MiB | last_hidden_state | 4 | 1.000000 | 0.999985 | 1.41e-03 | 5.13e-04 |  |
|  |  | pooled_mean | 4 | 1.000000 | 1.000000 | 2.46e-05 | 1.31e-04 |  |
| **q8_0** (F16:2,F32:214,Q8_0:99) | 113.3 MiB | last_hidden_state | 4 | 0.999934 | 0.978667 | 4.86e-02 | 1.32e-02 |  |
|  |  | pooled_mean | 4 | 0.999990 | 0.999988 | 7.95e-04 | 4.93e-03 |  |
| **q4_k** (F16:2,F32:214,Q4_0:38,Q4_K:61) | 61.9 MiB | last_hidden_state | 4 | 0.990260 | 0.751045 | 1.82e-01 | 1.64e-01 |  |
|  |  | pooled_mean | 4 | 0.998225 | 0.997961 | 1.18e-02 | 6.38e-02 |  |
| **q4_0** (F16:2,F32:214,Q4_0:99) | 61.9 MiB | last_hidden_state | 4 | 0.976396 | 0.734594 | 2.23e-01 | 2.90e-01 |  |
|  |  | pooled_mean | 4 | 0.993697 | 0.984555 | 9.01e-02 | 1.77e-01 |  |

### V-JEPA 2 ViT-L/16 SSv2 classifier (`vjepa2`, source f16, 2 clips x 16 frames)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f16** (F16:165,F32:331) | 717.1 MiB | last_hidden_state | 2 | 0.999995 | 0.997693 | 2.21e-02 | 3.70e-03 |  |
|  |  | logits | 2 | 1.000000 | 1.000000 | 6.20e-04 | 7.64e-04 | 2/2 / 1.00 |
| **q8_0** (F16:1,F32:331,Q8_0:164) | 383.2 MiB | last_hidden_state | 2 | 0.997179 | 0.288682 | 6.84e-01 | 8.17e-02 |  |
|  |  | logits | 2 | 0.999857 | 0.999774 | 1.73e-02 | 2.13e-02 | 2/2 / 1.00 |
| **q4_k** (F16:1,F32:331,Q4_0:37,Q4_K:127) | 205.1 MiB | last_hidden_state | 2 | 0.931075 | 0.220083 | 8.73e-01 | 3.84e-01 |  |
|  |  | logits | 2 | 0.985617 | 0.981231 | 1.74e-01 | 1.93e-01 | 2/2 / 0.90 |
| **q4_0** (F16:1,F32:331,Q4_0:164) | 205.1 MiB | last_hidden_state | 2 | 0.915000 | 0.200962 | 8.19e-01 | 4.23e-01 |  |
|  |  | logits | 2 | 0.988118 | 0.987313 | 1.44e-01 | 1.61e-01 | 2/2 / 0.90 |

### V-JEPA 2 ViT-L/16 fpc64 (`vjepa2`, source f16, 2 clips x 16 frames, encoder + predictor)

| type | file size | tensor | n | cos mean | cos min (worst token) | rel_max | rel_fro | top-1 / top-5 |
|---|---|---|---|---|---|---|---|---|
| **f16** (F16:147,F32:296) | 622.5 MiB | last_hidden_state | 2 | 0.999995 | 0.997693 | 2.21e-02 | 3.70e-03 |  |
|  |  | pooled_mean | 2 | 1.000000 | 1.000000 | 9.87e-05 | 3.15e-04 |  |
|  |  | predictor_last_hidden_state | 2 | 1.000000 | 0.999995 | 1.90e-03 | 7.54e-04 |  |
| **q8_0** (F16:1,F32:296,Q8_0:146) | 332.8 MiB | last_hidden_state | 2 | 0.997179 | 0.288682 | 6.84e-01 | 8.17e-02 |  |
|  |  | pooled_mean | 2 | 0.999969 | 0.999965 | 2.64e-03 | 8.44e-03 |  |
|  |  | predictor_last_hidden_state | 2 | 0.999846 | 0.990921 | 8.70e-02 | 1.81e-02 |  |
| **q4_k** (F16:1,F32:296,Q4_0:37,Q4_K:109) | 178.3 MiB | last_hidden_state | 2 | 0.931075 | 0.220083 | 8.73e-01 | 3.84e-01 |  |
|  |  | pooled_mean | 2 | 0.996089 | 0.996085 | 2.60e-02 | 8.86e-02 |  |
|  |  | predictor_last_hidden_state | 2 | 0.979838 | 0.916385 | 2.12e-01 | 2.08e-01 |  |
| **q4_0** (F16:1,F32:296,Q4_0:146) | 178.3 MiB | last_hidden_state | 2 | 0.915000 | 0.200962 | 8.19e-01 | 4.23e-01 |  |
|  |  | pooled_mean | 2 | 0.994207 | 0.994142 | 3.68e-02 | 1.09e-01 |  |
|  |  | predictor_last_hidden_state | 2 | 0.978168 | 0.918093 | 2.00e-01 | 2.17e-01 |  |


## Reading the numbers

* **The pooled / CLS features degrade much more slowly than the per-token worst case.** With q8_0 every pooled
  feature (`pooled_mean`, `cls`, LeWM `emb` / `pred_next`) has cosine >= 0.99995, while the worst single token
  drops to 0.98 on ViT-H / V-JEPA 2.1 and to 0.29 on V-JEPA 2 ViT-L. The bad tokens are the ones with the smallest variance before the
  final LayerNorm: on `coco_000000000139` the six worst I-JEPA tokens have a pre-norm std of 1.03-1.13 against a
  median of 1.67 (the best tokens ~4), and `corr(log(1-cos), log(std)) = -0.63`; the LayerNorm divides by that std
  and amplifies whatever error the quantized weights injected. Only 0.4 % of the ViT-H tokens fall below 0.99 at
  q8_0 and the mean per-token cosine is 0.99974. (`tmp/<branch>/ijepa_token_analysis.py` in the scratch dir
  reproduces this; the weights themselves dequantize to cosine 0.99998 / rel 4e-3 per tensor, i.e. the
  quantizer is not the culprit.)
* **f16 on ViT-H already shows the same effect** (worst token 0.9995, `rel_max` 8.6e-3 against a pooled 3e-4),
  so the per-token parity threshold for that model has to be read on the mean, not the minimum.
* **V-JEPA 2 ViT-L is by far the most token-sensitive model** (the fpc64 and SSv2 files contain the same encoder
  and give identical encoder rows). At q8_0 the *median* token cosine is 0.9998 but 18-23 % of the 2048 tokens fall
  below 0.999, 3 % below 0.99 and 0.3-0.8 % below 0.9 (worst 0.29); at f16 the worst token is 0.9977. The damage
  is spread over the whole network: q8_0 with any one of `attn_qkv` / `attn_out` / `ffn_up` / `ffn_down` kept at
  f16 (`--keep`) still has a worst token of 0.49 / 0.29 / 0.35 / 0.29. Everything downstream of the tokens is
  fine — `pooled_mean` 0.99997, predictor output 0.9998, logits 0.9999 with identical top-1 / top-5 — so this
  matters only if individual V-JEPA 2 tokens are consumed directly.
* **Depth and width matter more than the type family.** The 12-layer ViT-S loses more at q4 than the 32-layer
  ViT-H gains from K-quants: LeJEPA q4_0 mean token cosine 0.966 (worst 0.83), ViT-H 0.972 (worst 0.43). q4_k is
  consistently a little better than q4_0 at the same size (0.970 vs 0.966 on ViT-S, 0.979 vs 0.972 on ViT-H,
  0.9903 vs 0.9884 on V-JEPA 2.1), q6_k sits between q8_0 and q5.
* **The LeWM world model is the most robust** (192-d, shallow, BatchNorm-folded MLPs): q8_0 `pred_next` cosine
  0.99997, q6_k 0.9998, and even q4 keeps the predicted next-state embedding above 0.99.

## Recommendation

| use | type | why |
|---|---|---|
| parity / default deployment | **f16** | 0.5x of f32, indistinguishable from f32 for every feature (pooled 1e-4 relative, worst token 0.9977) |
| features (pooled / CLS / mean-pooled embeddings, retrieval, LeWM world-model rollouts) | **q8_0** | pooled cosine >= 0.99995 on every model, 0.53x of f16; passes the Q8_0 parity threshold (worst-token cosine 0.999) on ViT-S, LeWM and the V-JEPA 2.1 image path, and on the pooled outputs of every model |
| classification with the attentive-pool head (SSv2) | **f16** (q8_0 if memory matters) | logits cosine 0.9998, top-1 and top-5 identical to the reference on the fixtures; q4 still matched top-1 here but top-5 overlap fell to 0.9 and the logits moved by 17 % (`rel_max`) — do not trust q4 near a decision boundary; engine-level check on 105 real clips: top-1 agreement with PyTorch 99.0 % at f16 vs 94.3 % at q8_0 (`docs/accuracy-video.md`) |
| dense per-token features (segmentation-style use of `last_hidden_state`, per-token retrieval) | **q8_0**, at most q6_k; **f32 for V-JEPA 2 ViT-L** (the C++ engine's f16 matmuls round activations: worst token 0.51 at f16, `docs/parity.md`) | q6_k keeps the mean token cosine >= 0.998 but the worst tokens go to 0.91 on ViT-H; q5 and below distort individual tokens badly (worst 0.5-0.7); V-JEPA 2 ViT-L already has 3 % of tokens below 0.99 at q8_0 |
| small-footprint feature extraction where pooled ~0.99 is enough | **q4_k** (falls back to q4_0 on 384-wide layers) | pooled cosine 0.992-0.998 on every model, 0.29x of f16; not for parity testing, not for per-token use |

**On a GPU the speed half of this advice inverts.** Everything above about *accuracy* holds
unchanged, but the "quantised buys memory, not time" rule is a CPU statement: `GGML_LLAMAFILE`'s fast
sgemm covers only F32/F16/Q8_0, so K-quants fall back to ggml's generic vec-dot there. On CUDA every
type we ship takes `mmq`, a real INT8 tensor-core kernel, and **q4_k ties q8_0 and beats f16** — I-JEPA
ViT-H 7.8 ms (q4_k) / 8.0 (q8_0) / 15.5 (f16) against 198 / 129 / 147 ms on 32 CPU threads. See
`docs/results.md` "GPU (CUDA)" and `docs/parity.md` "Parity on a GPU" for the parity bars that apply
there (q4_k stays in the advisory low-bit tier on both backends).

q4_1 / q5_1 / q5_0 / q5_k are supported for completeness (q5_k and q5_0 have the same size, as do q4_k and q4_0)
but no row of the tables prefers them over q8_0 / q6_k / q4_k; there is no reason to ship them. When a single
component or layer is known to be sensitive, `--keep` can hold it at the source type at a small size cost (it did
not rescue the V-JEPA 2 ViT-L worst tokens — the sensitivity there is spread across the whole network).

## Reproduce

```bash
export PATH=$HOME/.local/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32 --target jepa-quantize
PY=.venv/bin/python
for t in q8_0 q6_k q5_k q5_1 q5_0 q4_1 q4_k q4_0; do build/jepa-quantize models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf models/gguf/lejepa-vits16-pretrain-in1k-$t.gguf $t -t 32; done
for t in q8_0 q6_k q5_0 q4_k q4_0; do build/jepa-quantize models/gguf/lewm-pusht-f32.gguf models/gguf/lewm-pusht-$t.gguf $t -t 32; done
for t in q8_0 q6_k q5_k q4_k q4_0; do build/jepa-quantize models/gguf/ijepa_vith14_1k-f16.gguf models/gguf/ijepa_vith14_1k-$t.gguf $t -t 32; done
for m in vjepa2_1-vitb-384 vjepa2-vitl-fpc64-256 vjepa2-vitl-fpc16-256-ssv2; do for t in q8_0 q4_k q4_0; do build/jepa-quantize models/gguf/$m-f16.gguf models/gguf/$m-$t.gguf $t -t 32; done; done
# accuracy rows (32 numpy threads; ViT-H: 2 images, V-JEPA 2 ViT-L: the 16-frame clips)
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/lejepa-vits16-pretrain-in1k-q8_0.gguf --ref tests/fixtures/ref/lejepa-vits16 --threads 32
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/lewm-pusht-q8_0.gguf --ref tests/fixtures/ref/lewm-pusht --threads 32
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/ijepa_vith14_1k-q8_0.gguf --ref tests/fixtures/ref/ijepa-vith14-1k --samples 2 --threads 32
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/vjepa2_1-vitb-384-q8_0.gguf --ref tests/fixtures/ref/vjepa2_1-vitb-384 --threads 32
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/vjepa2-vitl-fpc16-256-ssv2-q8_0.gguf --ref tests/fixtures/ref/vjepa2-vitl-fpc16-256-ssv2 --threads 32
$PY scripts/gguf_dequant_selftest.py --gguf models/gguf/vjepa2-vitl-fpc64-256-q8_0.gguf --ref tests/fixtures/ref/vjepa2-vitl-fpc64-256 --samples archery_f16,bowling_f16 --threads 32
# V-JEPA 2.1 q8_0 against the Meta PyTorch model on random inputs (the converter's own reference script also dequantizes)
$PY scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2_1-vitb-384-q8_0.gguf --meta models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt --vjepa2-src tmp/vjepa2-src
```

The 64-frame V-JEPA 2 samples (8192 tokens) also run through `gguf_dequant_selftest.py` but need ~20 GB for the
numpy attention matrices; pass `--samples archery_f64` explicitly.
