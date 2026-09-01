// jepa-quantize — re-type the weight matrices of a jepa.cpp GGUF (f32/f16 -> f16 / q8_0 / q4_0 / ... / q6_k).
//
//   jepa-quantize IN.gguf OUT.gguf TYPE [-t threads] [--allow-requant] [--dry-run] [-v]
//
// Rule (docs/gguf-schema.md "Quantization rules", docs/quantization.md): only the 2-D weight matrices of the
// attention / FFN / projection / classifier layers change type; patch embeddings, norms, biases, position tables,
// tokens, adaLN, action embedders and mask tokens keep the type they have in the input file.  All metadata is
// copied verbatim except `general.file_type`, which is set to the ggml ftype of the requested TYPE.
// K-quants need ne[0] % 256 == 0; tensors that do not qualify fall back per tensor (q4_k -> q4_0, q5_k -> q5_0,
// q6_k -> q8_0) and the fallback is printed.  Already-quantized tensors are refused unless --allow-requant.
//
// The file is written in two passes (metadata first, then the tensor data appended tensor by tensor), so the
// peak memory is one tensor, not the whole model.
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>

// 64-bit absolute seek: fseeko is POSIX, MSVC spells it _fseeki64.
static int jq_seek64(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, (long long) offset, SEEK_SET);
#else
    return fseeko(f, (off_t) offset, SEEK_SET);
#endif
}
#include <vector>

namespace {

struct type_spec {
    const char * name;
    ggml_type    type;
    ggml_ftype   ftype;
};

// TYPE argument -> ggml tensor type + general.file_type value
const type_spec TYPE_SPECS[] = {
    {"f32",  GGML_TYPE_F32,  GGML_FTYPE_ALL_F32},
    {"f16",  GGML_TYPE_F16,  GGML_FTYPE_MOSTLY_F16},
    {"q8_0", GGML_TYPE_Q8_0, GGML_FTYPE_MOSTLY_Q8_0},
    {"q4_0", GGML_TYPE_Q4_0, GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_TYPE_Q4_1, GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0, GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1, GGML_FTYPE_MOSTLY_Q5_1},
    {"q4_k", GGML_TYPE_Q4_K, GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_TYPE_Q5_K, GGML_FTYPE_MOSTLY_Q5_K},
    {"q6_k", GGML_TYPE_Q6_K, GGML_FTYPE_MOSTLY_Q6_K},
};

const type_spec * find_type(const std::string & s) {
    std::string l = s;
    std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    for (const auto & t : TYPE_SPECS) {
        if (l == t.name) return &t;
    }
    return nullptr;
}

// per-tensor fallback when a K-quant cannot be used (row size not a multiple of 256)
ggml_type kquant_fallback(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_K: return GGML_TYPE_Q4_0;
        case GGML_TYPE_Q5_K: return GGML_TYPE_Q5_0;
        case GGML_TYPE_Q6_K: return GGML_TYPE_Q8_0;
        default:             return t;
    }
}

bool is_kquant(ggml_type t) {
    return t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

bool contains(const std::string & s, const char * sub) { return s.find(sub) != std::string::npos; }
bool starts_with(const std::string & s, const char * pre) { return s.rfind(pre, 0) == 0; }
bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// docs/gguf-schema.md: quantize only the 2-D weights of attention / FFN / projection / classifier layers.
bool is_quantizable(const std::string & name, int n_dims) {
    if (n_dims != 2 || !ends_with(name, ".weight")) {
        return false;
    }
    // explicit keep list (most are 1-D anyway; listed so the rule is obvious and robust)
    static const char * const KEEP[] = {
        "patch_embed", "pos_embed", "cls_token", "reg_tokens", "mask_tokens", "mod_embed",
        ".adaln.", "action_embed", "state_embed", ".ln1.", ".ln2.", ".norm.", "hier_norm", "ln_kv", "head.query",
    };
    for (const char * k : KEEP) {
        if (contains(name, k)) return false;
    }
    // block components: enc.blk.* / pred.blk.* / head.blk.* / head.xattn.*
    static const char * const COMPONENTS[] = {
        ".attn_qkv.", ".attn_q.", ".attn_k.", ".attn_v.", ".attn_out.", ".ffn_up.", ".ffn_down.",
    };
    for (const char * c : COMPONENTS) {
        if (contains(name, c)) return true;
    }
    // projections / classifier
    static const char * const PREFIXES[] = {
        "pred.proj", "pred.embed", "enc.proj.", "head.cls.", "head.xattn.",
    };
    for (const char * p : PREFIXES) {
        if (starts_with(name, p)) return true;
    }
    return false;
}

// copy KV pair `i` of `src` into `dst`; gguf_set_kv() would copy everything at once but the replaced
// general.file_type would then move to the end, and we want the metadata order preserved
void copy_kv(gguf_context * dst, const gguf_context * src, int64_t i) {
    const char * key = gguf_get_key(src, i);
    switch (gguf_get_kv_type(src, i)) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst, key, gguf_get_val_u8  (src, i)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst, key, gguf_get_val_i8  (src, i)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst, key, gguf_get_val_u16 (src, i)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst, key, gguf_get_val_i16 (src, i)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst, key, gguf_get_val_u32 (src, i)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst, key, gguf_get_val_i32 (src, i)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst, key, gguf_get_val_f32 (src, i)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst, key, gguf_get_val_u64 (src, i)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst, key, gguf_get_val_i64 (src, i)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (dst, key, gguf_get_val_f64 (src, i)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst, key, gguf_get_val_bool(src, i)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str (dst, key, gguf_get_val_str (src, i)); break;
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(src, i);
            const size_t n = gguf_get_arr_n(src, i);
            if (at == GGUF_TYPE_STRING) {
                std::vector<const char *> strs(n);
                for (size_t j = 0; j < n; ++j) strs[j] = gguf_get_arr_str(src, i, j);
                gguf_set_arr_str(dst, key, strs.data(), n);
            } else {
                gguf_set_arr_data(dst, key, at, gguf_get_arr_data(src, i), n);
            }
        } break;
        default:
            GGML_ABORT("invalid KV type for key %s", key);
    }
}

// quantize `nrows` rows of `n_per_row` floats with ggml_quantize_chunk, rows split across threads
size_t quantize_rows(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int n_threads) {
    const size_t row_size = ggml_row_size(type, n_per_row);
    // do not bother spawning threads for tiny tensors (< 64k elements per thread)
    const int64_t max_threads = std::max<int64_t>(1, (nrows * n_per_row) / (64 * 1024));
    n_threads = (int) std::max<int64_t>(1, std::min<int64_t>({(int64_t) n_threads, nrows, max_threads}));
    if (n_threads == 1) {
        return ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, nullptr);
    }
    const int64_t rows_per_thread = (nrows + n_threads - 1) / n_threads;
    std::vector<size_t> sizes(n_threads, 0);
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t) {
        const int64_t r0 = t * rows_per_thread;
        const int64_t r1 = std::min(nrows, r0 + rows_per_thread);
        if (r0 >= r1) break;
        workers.emplace_back([=, &sizes]() {
            sizes[t] = ggml_quantize_chunk(type, src + r0 * n_per_row, (char *) dst + r0 * row_size, 0, r1 - r0, n_per_row, nullptr);
        });
    }
    for (auto & w : workers) w.join();
    size_t total = 0;
    for (size_t s : sizes) total += s;
    return total;
}

std::string fmt_bytes(double b) {
    char buf[64];
    if (b >= 1024.0 * 1024.0 * 1024.0) snprintf(buf, sizeof buf, "%.2f GiB", b / (1024.0 * 1024.0 * 1024.0));
    else if (b >= 1024.0 * 1024.0)     snprintf(buf, sizeof buf, "%.1f MiB", b / (1024.0 * 1024.0));
    else if (b >= 1024.0)              snprintf(buf, sizeof buf, "%.1f KiB", b / 1024.0);
    else                               snprintf(buf, sizeof buf, "%.0f B", b);
    return buf;
}

std::string fmt_shape(const int64_t * ne, int n_dims) {
    std::string s = "[";
    for (int i = n_dims - 1; i >= 0; --i) {  // print in PyTorch order (ne reversed)
        s += std::to_string(ne[i]);
        if (i) s += ",";
    }
    return s + "]";
}

int64_t file_size(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    const int64_t n = ftell(f);
    fclose(f);
    return n;
}

struct plan_item {
    std::string name;
    ggml_type   src_type;
    ggml_type   dst_type;
    int64_t     ne[GGML_MAX_DIMS];
    int         n_dims;
    size_t      src_offset;   // relative to the data section of the input
    size_t      src_bytes;
    size_t      dst_bytes;
    std::string note;
};

void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s IN.gguf OUT.gguf TYPE [-t threads] [--keep SUBSTR]... [--allow-requant] [--dry-run] [-v]\n"
            "  TYPE: f32 | f16 | q8_0 | q4_0 | q4_1 | q5_0 | q5_1 | q4_k | q5_k | q6_k\n"
            "  -t N              quantization threads (default: min(32, hardware threads))\n"
            "  --keep SUBSTR     tensors whose name contains SUBSTR keep their type (repeatable; e.g. --keep ffn_down)\n"
            "  --allow-requant   re-quantize tensors that are already quantized (lossy on top of lossy)\n"
            "  --dry-run         print the plan and the size summary, write nothing\n"
            "  -v                also list the tensors that keep their type\n",
            prog);
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // keep the progress lines ordered with stderr diagnostics
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const char * in_path  = argv[1];
    const char * out_path = argv[2];
    const type_spec * spec = find_type(argv[3]);
    if (!spec) {
        fprintf(stderr, "error: unknown TYPE '%s'\n", argv[3]);
        usage(argv[0]);
        return 1;
    }
    int  n_threads     = (int) std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    bool allow_requant = false;
    bool dry_run       = false;
    bool verbose       = false;
    std::vector<std::string> keep;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-t" || a == "--threads") && i + 1 < argc) {
            n_threads = std::max(1, atoi(argv[++i]));
        } else if (a == "--keep" && i + 1 < argc) {
            keep.push_back(argv[++i]);
        } else if (a == "--allow-requant") {
            allow_requant = true;
        } else if (a == "--dry-run") {
            dry_run = true;
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (strcmp(in_path, out_path) == 0) {
        fprintf(stderr, "error: IN and OUT must differ\n");
        return 1;
    }
    const ggml_type target = spec->type;
    if (ggml_is_quantized(target) && ggml_quantize_requires_imatrix(target)) {
        fprintf(stderr, "error: %s needs an importance matrix\n", spec->name);
        return 1;
    }

    const auto t_start = std::chrono::steady_clock::now();

    // ---- read the input metadata (tensor data stays on disk) ------------------------------------------------------
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * in = gguf_init_from_file(in_path, ip);
    if (!in) {
        fprintf(stderr, "error: cannot read GGUF '%s'\n", in_path);
        return 1;
    }
    const int64_t n_tensors = gguf_get_n_tensors(in);
    const size_t  in_data_offset = gguf_get_data_offset(in);
    {
        const int64_t k = gguf_find_key(in, "general.architecture");
        const char * arch = k >= 0 ? gguf_get_val_str(in, k) : "?";
        const int64_t kn = gguf_find_key(in, "general.name");
        const char * name = kn >= 0 ? gguf_get_val_str(in, kn) : "?";
        const int64_t kf = gguf_find_key(in, "general.file_type");
        const int64_t ft = kf >= 0 ? (int64_t) gguf_get_val_u32(in, kf) : -1;
        printf("jepa-quantize: %s (%s, arch=%s, file_type=%" PRId64 ", %" PRId64 " tensors, %" PRId64 " kv) -> %s [%s], %d threads\n",
               in_path, name, arch, ft, n_tensors, gguf_get_n_kv(in), out_path, spec->name, n_threads);
        if (strcmp(arch, "jepa") != 0) {
            fprintf(stderr, "warning: general.architecture is '%s', not 'jepa'; applying the jepa naming rule anyway\n", arch);
        }
    }

    // ---- output metadata: all KV pairs verbatim (same order), file_type updated, tensors re-typed ------------------
    gguf_context * out = gguf_init_empty();
    for (int64_t i = 0; i < gguf_get_n_kv(in); ++i) {
        const std::string key = gguf_get_key(in, i);
        if (key == "general.file_type") {
            gguf_set_val_u32(out, key.c_str(), (uint32_t) spec->ftype);
        } else if (key == "general.alignment") {
            // gguf_init_empty() fixes the output alignment at the default; a different input value is not carried over
            printf("note: dropping general.alignment=%u, the output uses the default alignment %zu\n",
                   gguf_get_kv_type(in, i) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(in, i) : 0u, gguf_get_alignment(out));
        } else {
            copy_kv(out, in, i);
        }
    }
    if (gguf_find_key(out, "general.file_type") < 0) {
        gguf_set_val_u32(out, "general.file_type", (uint32_t) spec->ftype);
    }

    ggml_init_params mp = { ggml_tensor_overhead() * (size_t) n_tensors + 4096, nullptr, /*no_alloc =*/ true };
    ggml_context * meta = ggml_init(mp);

    std::vector<plan_item> plan;
    plan.reserve(n_tensors);
    std::map<ggml_type, std::pair<int64_t, size_t>> before, after;  // type -> (count, bytes)
    int n_changed = 0, n_fallback = 0, n_kept_rule = 0, n_kept_user = 0, n_requant = 0;
    size_t bytes_before = 0, bytes_after = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        plan_item it;
        it.name       = gguf_get_tensor_name(in, i);
        it.src_type   = gguf_get_tensor_type(in, i);
        it.src_offset = gguf_get_tensor_offset(in, i);
        it.src_bytes  = gguf_get_tensor_size(in, i);
        const int64_t * ne = gguf_get_tensor_ne(in, i);
        memcpy(it.ne, ne, sizeof(it.ne));
        it.n_dims = GGML_MAX_DIMS;
        while (it.n_dims > 1 && it.ne[it.n_dims - 1] == 1) it.n_dims--;

        ggml_type dst = it.src_type;
        if (target == GGML_TYPE_F32) {
            dst = GGML_TYPE_F32;  // full dequantization / f16 -> f32 for every tensor
            if (it.src_type != GGML_TYPE_F32) it.note = ggml_is_quantized(it.src_type) ? "dequantized" : "widened";
        } else if (!is_quantizable(it.name, it.n_dims)) {
            it.note = "kept (rule)";
            n_kept_rule++;
        } else if (std::any_of(keep.begin(), keep.end(), [&](const std::string & k) { return it.name.find(k) != std::string::npos; })) {
            it.note = "kept (--keep)";
            n_kept_user++;
        } else {
            dst = target;
            if (ggml_is_quantized(it.src_type) && it.src_type != target && ggml_is_quantized(target)) {
                if (!allow_requant) {
                    fprintf(stderr, "error: %s is already %s; re-quantizing to %s needs --allow-requant\n",
                            it.name.c_str(), ggml_type_name(it.src_type), spec->name);
                    return 1;
                }
                it.note = "REQUANTIZED from " + std::string(ggml_type_name(it.src_type));
                n_requant++;
            }
            if (is_kquant(dst) && it.ne[0] % ggml_blck_size(dst) != 0) {
                const ggml_type fb = kquant_fallback(dst);
                it.note = std::string("fallback ") + ggml_type_name(dst) + " -> " + ggml_type_name(fb) +
                          " (ne[0]=" + std::to_string(it.ne[0]) + " not a multiple of " + std::to_string(ggml_blck_size(dst)) + ")";
                dst = fb;
                n_fallback++;
            }
            if (it.ne[0] % ggml_blck_size(dst) != 0) {
                it.note = std::string("kept: ne[0]=") + std::to_string(it.ne[0]) + " not a multiple of the " +
                          ggml_type_name(dst) + " block size " + std::to_string(ggml_blck_size(dst));
                dst = it.src_type;
            }
            if (dst == it.src_type && it.note.empty()) it.note = "already " + std::string(ggml_type_name(dst));
            if (ggml_is_quantized(it.src_type) && !ggml_is_quantized(dst) && dst != it.src_type) it.note = "dequantized";
        }
        it.dst_type  = dst;
        it.dst_bytes = ggml_row_size(dst, it.ne[0]) * (size_t) (it.ne[1] * it.ne[2] * it.ne[3]);

        ggml_tensor * t = ggml_new_tensor(meta, dst, it.n_dims, it.ne);
        ggml_set_name(t, it.name.c_str());
        gguf_add_tensor(out, t);

        before[it.src_type].first++;  before[it.src_type].second += it.src_bytes;
        after[dst].first++;           after[dst].second          += it.dst_bytes;
        bytes_before += it.src_bytes;
        bytes_after  += it.dst_bytes;
        const bool changed = dst != it.src_type;
        n_changed += changed;
        if (changed || verbose || (!it.note.empty() && it.note.rfind("kept (", 0) != 0 && it.note.rfind("already", 0) != 0)) {
            printf("  %-40s %-18s %5s -> %-5s %10s -> %-10s %s\n", it.name.c_str(), fmt_shape(it.ne, it.n_dims).c_str(),
                   ggml_type_name(it.src_type), ggml_type_name(dst), fmt_bytes((double) it.src_bytes).c_str(),
                   fmt_bytes((double) it.dst_bytes).c_str(), it.note.c_str());
        }
        plan.push_back(std::move(it));
    }

    printf("plan: %d tensors re-typed, %d kept by the rule, %d kept by --keep, %d K-quant fallbacks, %d re-quantized\n",
           n_changed, n_kept_rule, n_kept_user, n_fallback, n_requant);

    // If K-quant fallbacks re-typed the majority of quantized tensors to a different type than requested,
    // general.file_type would misdescribe the file (e.g. "q4_k" that is mostly q4_0). Record the majority.
    if (n_fallback > 0 && ggml_is_quantized(target)) {
        ggml_type major = target; int64_t major_n = -1; int64_t total_q = 0;
        for (const auto & kv : after) {
            if (!ggml_is_quantized(kv.first)) continue;
            total_q += kv.second.first;
            if (kv.second.first > major_n) { major = kv.first; major_n = kv.second.first; }
        }
        if (major != target) {
            for (const auto & ts : TYPE_SPECS) {
                if (ts.type == major) {
                    gguf_set_val_u32(out, "general.file_type", (uint32_t) ts.ftype);
                    printf("note: %lld/%lld quantized tensors are %s after fallbacks -> general.file_type set to %s (requested %s)\n",
                           (long long) major_n, (long long) total_q, ggml_type_name(major), ts.name, spec->name);
                    break;
                }
            }
        }
    }
    printf("tensor bytes by type, before:\n");
    for (const auto & kv : before) {
        printf("  %-6s %4" PRId64 " tensors %12zu bytes (%s)\n", ggml_type_name(kv.first), kv.second.first, kv.second.second, fmt_bytes((double) kv.second.second).c_str());
    }
    printf("tensor bytes by type, after:\n");
    for (const auto & kv : after) {
        printf("  %-6s %4" PRId64 " tensors %12zu bytes (%s)\n", ggml_type_name(kv.first), kv.second.first, kv.second.second, fmt_bytes((double) kv.second.second).c_str());
    }
    printf("tensor data: %zu bytes (%s) -> %zu bytes (%s), %.3fx\n", bytes_before, fmt_bytes((double) bytes_before).c_str(),
           bytes_after, fmt_bytes((double) bytes_after).c_str(), bytes_before ? (double) bytes_after / (double) bytes_before : 0.0);

    if (dry_run) {
        printf("dry run: nothing written\n");
        ggml_free(meta);
        gguf_free(out);
        gguf_free(in);
        return 0;
    }

    // ---- pass 1: metadata --------------------------------------------------------------------------------------------
    if (!gguf_write_to_file(out, out_path, /*only_meta =*/ true)) {
        fprintf(stderr, "error: cannot write '%s'\n", out_path);
        return 1;
    }
    const size_t alignment = gguf_get_alignment(out);

    // ---- pass 2: tensor data, appended in order ---------------------------------------------------------------------
    FILE * fin = fopen(in_path, "rb");
    FILE * fout = fopen(out_path, "ab");
    if (!fin || !fout) {
        fprintf(stderr, "error: cannot open files for the data pass\n");
        return 1;
    }
    std::vector<uint8_t> src;
    std::vector<float>   f32;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> zeros(alignment, 0);
    size_t written = 0;
    for (const plan_item & it : plan) {
        src.resize(it.src_bytes);
        if (jq_seek64(fin, in_data_offset + it.src_offset) != 0 || fread(src.data(), 1, it.src_bytes, fin) != it.src_bytes) {
            fprintf(stderr, "error: short read on %s\n", it.name.c_str());
            return 1;
        }
        const uint8_t * out_data = src.data();
        size_t out_bytes = it.src_bytes;
        if (it.dst_type != it.src_type) {
            const int64_t n_elem = it.ne[0] * it.ne[1] * it.ne[2] * it.ne[3];
            // source -> f32
            const float * xf;
            if (it.src_type == GGML_TYPE_F32) {
                xf = (const float *) src.data();
            } else {
                f32.resize(n_elem);
                const ggml_type_traits * tr = ggml_get_type_traits(it.src_type);
                if (!tr->to_float) {
                    fprintf(stderr, "error: cannot convert %s from %s\n", it.name.c_str(), ggml_type_name(it.src_type));
                    return 1;
                }
                // to_float works row by row for block types; whole tensor is fine since rows are contiguous
                tr->to_float(src.data(), f32.data(), n_elem);
                xf = f32.data();
            }
            dst.resize(it.dst_bytes);
            if (it.dst_type == GGML_TYPE_F32) {
                memcpy(dst.data(), xf, it.dst_bytes);
            } else if (it.dst_type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(xf, (ggml_fp16_t *) dst.data(), n_elem);
            } else {
                const int64_t nrows = n_elem / it.ne[0];
                const size_t got = quantize_rows(it.dst_type, xf, dst.data(), nrows, it.ne[0], n_threads);
                if (got != it.dst_bytes) {
                    fprintf(stderr, "error: %s: quantized %zu bytes, expected %zu\n", it.name.c_str(), got, it.dst_bytes);
                    return 1;
                }
            }
            out_data = dst.data();
            out_bytes = it.dst_bytes;
        }
        if (fwrite(out_data, 1, out_bytes, fout) != out_bytes) {
            fprintf(stderr, "error: short write on %s\n", it.name.c_str());
            return 1;
        }
        const size_t pad = GGML_PAD(out_bytes, alignment) - out_bytes;
        if (pad && fwrite(zeros.data(), 1, pad, fout) != pad) {
            fprintf(stderr, "error: short write (padding) on %s\n", it.name.c_str());
            return 1;
        }
        written += out_bytes + pad;
    }
    fclose(fin);
    fclose(fout);
    ggml_quantize_free();
    ggml_free(meta);
    gguf_free(out);
    gguf_free(in);

    // ---- verify: the output must parse and describe the same tensors -----------------------------------------------
    gguf_context * chk = gguf_init_from_file(out_path, ip);
    if (!chk) {
        fprintf(stderr, "error: the written file '%s' does not parse\n", out_path);
        return 1;
    }
    bool ok = gguf_get_n_tensors(chk) == n_tensors;
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        ok = plan[i].name == gguf_get_tensor_name(chk, i) && plan[i].dst_type == gguf_get_tensor_type(chk, i) &&
             plan[i].dst_bytes == gguf_get_tensor_size(chk, i);
    }
    const int64_t expected_size = (int64_t) gguf_get_data_offset(chk) + (int64_t) written;
    gguf_free(chk);
    const int64_t size_in = file_size(in_path), size_out = file_size(out_path);
    if (!ok || size_out != expected_size) {
        fprintf(stderr, "error: verification of '%s' failed (size %" PRId64 ", expected %" PRId64 ")\n", out_path, size_out, expected_size);
        return 1;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("wrote %s: %" PRId64 " bytes (%s), input %" PRId64 " bytes (%s), %.3fx, %.1f s\n", out_path, size_out,
           fmt_bytes((double) size_out).c_str(), size_in, fmt_bytes((double) size_in).c_str(),
           size_in ? (double) size_out / (double) size_in : 0.0, secs);
    return 0;
}
