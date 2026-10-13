// H-JEPA planning: the paper's top-down gradient planner (arXiv 2610.06805, section "Planning";
// h_jepa/hierarchical_solver.py and stable_worldmodel/solver/gd.py in github.com/kevinghst/H-JEPA).
//
// Every level optimises a sequence of ITS actions by gradient descent through its own world model:
//
//   candidates  a[S, T, A]   S samples, T = the level's planning horizon
//   rollout     s_0 = the current state, s_{t+1} = pred_k(s_t, a_t)          (context window 1)
//   cost        the last `cost_last_n` predicted states against the goal, as a sum of squared
//               errors -- in the level's own latent space, or after encoding the predictions with
//               the level(s) above when the goal is one of THEIR states;
//               optionally plus the same for the actions, pooled by the upper action encoder
//   update      AdamW on a, clip to the range of latent actions seen in training, early stop
//   result      the lowest-cost sample and the states it predicts
//
// and jepa_hjepa_plan runs the levels top-down, each level's first predicted state(s) becoming the
// goal of the level below.
//
// The gradient is hand-written and lives on the host. ggml's autodiff covers neither LayerNorm nor
// GELU, and the planner's rollout uses a context window of one state, where causal attention over
// a single token is the identity on its value projection -- so one predictor step is a chain of
// dense layers, LayerNorms and element-wise functions, and its reverse mode is short enough to
// write down and check against PyTorch's autograd (tests/test-predictor.cpp --hjepa, "plan").
// The weights it needs are read back from the model once per context, dequantised to f32.
//
// Samples are independent (the objective is the sum of their costs), so they are processed in
// small chunks, one tape each, across the context's threads. A sample's arithmetic does not depend
// on which chunk or thread it lands in: the result is deterministic for any thread count.
#include "predictor-internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------
// weights, as plain f32
// ---------------------------------------------------------------------------------------------
struct lin {
    int in = 0, out = 0;
    std::vector<float> w;    // [out][in]  (the backward pass walks rows of this)
    std::vector<float> wt;   // [in][out]  (the forward pass walks rows of this)
    std::vector<float> b;    // [out] or empty
    explicit operator bool() const { return in > 0; }
};

struct vecp {                // an optional f32 vector (norm weight / bias, position row)
    std::vector<float> v;
    explicit operator bool() const { return !v.empty(); }
};

static std::vector<float> tensor_f32(const jepa_model * m, const std::string & name) {
    std::vector<float> out;
    ggml_tensor * t = m->get(name);
    if (!t) return out;
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), n);
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        GGML_ASSERT(tr && tr->to_float);
        tr->to_float(raw.data(), out.data(), n);
    }
    return out;
}

// Linear `name`: weight [in, out] in ggml order, i.e. row-major [out][in]. `row0`/`rows` take a
// slice of the output rows (the value third of a fused qkv).
static lin load_lin(const jepa_model * m, const std::string & name, int64_t row0 = 0, int64_t rows = -1) {
    lin L;
    ggml_tensor * t = m->get(name + ".weight");
    if (!t) return L;
    std::vector<float> w = tensor_f32(m, name + ".weight");
    std::vector<float> b = tensor_f32(m, name + ".bias");
    const int64_t in = t->ne[0], out_all = t->ne[1];
    if (rows < 0) rows = out_all - row0;
    L.in = (int) in; L.out = (int) rows;
    L.w.assign(w.begin() + row0 * in, w.begin() + (row0 + rows) * in);
    if (!b.empty()) L.b.assign(b.begin() + row0, b.begin() + row0 + rows);
    L.wt.resize(L.w.size());
    for (int64_t o = 0; o < rows; o++) {
        for (int64_t i = 0; i < in; i++) L.wt[(size_t) (i * rows + o)] = L.w[(size_t) (o * in + i)];
    }
    return L;
}

static vecp load_vec(const jepa_model * m, const std::string & name) {
    vecp v;
    v.v = tensor_f32(m, name);
    return v;
}

struct block_w {            // one adaLN predictor block, as the T = 1 graph needs it
    lin adaln, v, out, up, down;
    vecp ln1_w, ln1_b, ln2_w, ln2_b;
};

struct seq_block_w {        // one pre-LN layer of the 'seq' action encoder
    lin qkv, out, up, down;
    vecp ln1_w, ln1_b, ln2_w, ln2_b;
};

struct level_w {
    jepa_hjepa_level hp;
    // predictor
    std::vector<float> pos0;                 // pos_embed row 0
    lin act0, act2;                          // act_embed (absent = identity)
    lin pact0, pact2;                        // adaln_bare: the predictor's own action embedder
    lin in_proj, cond_proj, out_proj, proj0, proj2;
    vecp norm_w, norm_b;
    std::vector<block_w> blocks;
    // state encoder of this level (levels >= 2): latent_mlp or identity
    vecp enc_in_w, enc_in_b, enc_out_w, enc_out_b;
    lin enc_fc1, enc_fc2;
    // action encoder of this level (levels >= 2)
    lin ae_step, ae_head0, ae_head2, ae_fc1, ae_fc2;
    std::vector<float> ae_pos;               // [max_chunk + 1][d]
    std::vector<seq_block_w> ae_blocks;
    vecp ae_headln_w, ae_headln_b, ae_out_w, ae_out_b;
    // prior over the level's actions: rows of latent actions recorded in training
    std::vector<float> queue;                // [n][action_dim]
};

struct plan_model {
    float ln_eps = 1e-5f, adaln_eps = 1e-6f;
    std::vector<level_w> levels;
};

static std::shared_ptr<plan_model> build_plan_model(const jepa_model * m) {
    auto pm = std::make_shared<plan_model>();
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    pm->ln_eps = hj.ln_eps;
    pm->adaln_eps = hj.adaln_eps;
    for (size_t li = 0; li < hj.levels.size(); li++) {
        const std::string p = "lvl." + std::to_string(li + 1) + ".";
        level_w L;
        L.hp = hj.levels[li];
        const int64_t inner = (int64_t) L.hp.pred_n_head * L.hp.pred_head_dim;
        std::vector<float> pos = tensor_f32(m, p + "pred.pos_embed");
        L.pos0.assign(pos.begin(), pos.begin() + L.hp.embed_dim);
        L.act0 = load_lin(m, p + "act_embed.0");
        L.act2 = load_lin(m, p + "act_embed.2");
        L.pact0 = load_lin(m, p + "pred.act_embed.0");
        L.pact2 = load_lin(m, p + "pred.act_embed.2");
        L.in_proj = load_lin(m, p + "pred.in_proj");
        L.cond_proj = load_lin(m, p + "pred.cond_proj");
        L.out_proj = load_lin(m, p + "pred.out_proj");
        L.proj0 = load_lin(m, p + "pred.proj.0");
        L.proj2 = load_lin(m, p + "pred.proj.2");
        L.norm_w = load_vec(m, p + "pred.norm.weight");
        L.norm_b = load_vec(m, p + "pred.norm.bias");
        for (int i = 0; i < L.hp.pred_n_layer; i++) {
            const std::string q = p + "pred.blk." + std::to_string(i) + ".";
            block_w B;
            B.adaln = load_lin(m, q + "adaln");
            // a single query attending a single key: softmax is 1, the output is the value row
            B.v = load_lin(m, q + "attn_qkv", 2 * inner, inner);
            B.out = load_lin(m, q + "attn_out");
            B.up = load_lin(m, q + "ffn_up");
            B.down = load_lin(m, q + "ffn_down");
            B.ln1_w = load_vec(m, q + "ln1.weight"); B.ln1_b = load_vec(m, q + "ln1.bias");
            B.ln2_w = load_vec(m, q + "ln2.weight"); B.ln2_b = load_vec(m, q + "ln2.bias");
            L.blocks.push_back(std::move(B));
        }
        L.enc_in_w = load_vec(m, p + "enc.in_norm.weight");   L.enc_in_b = load_vec(m, p + "enc.in_norm.bias");
        L.enc_out_w = load_vec(m, p + "enc.out_norm.weight"); L.enc_out_b = load_vec(m, p + "enc.out_norm.bias");
        L.enc_fc1 = load_lin(m, p + "enc.fc1");
        L.enc_fc2 = load_lin(m, p + "enc.fc2");
        L.ae_step = load_lin(m, p + "act_enc.step");
        L.ae_head0 = load_lin(m, p + "act_enc.head.0");
        L.ae_head2 = load_lin(m, p + "act_enc.head.2");
        L.ae_fc1 = load_lin(m, p + "act_enc.fc1");
        L.ae_fc2 = load_lin(m, p + "act_enc.fc2");
        L.ae_pos = tensor_f32(m, p + "act_enc.pos");
        L.ae_headln_w = load_vec(m, p + "act_enc.head.ln.weight"); L.ae_headln_b = load_vec(m, p + "act_enc.head.ln.bias");
        L.ae_out_w = load_vec(m, p + "act_enc.out_norm.weight");   L.ae_out_b = load_vec(m, p + "act_enc.out_norm.bias");
        for (int i = 0; i < L.hp.act_enc_n_layer; i++) {
            const std::string q = p + "act_enc.blk." + std::to_string(i) + ".";
            seq_block_w B;
            B.qkv = load_lin(m, q + "attn_qkv");
            B.out = load_lin(m, q + "attn_out");
            B.up = load_lin(m, q + "ffn_up");
            B.down = load_lin(m, q + "ffn_down");
            B.ln1_w = load_vec(m, q + "ln1.weight"); B.ln1_b = load_vec(m, q + "ln1.bias");
            B.ln2_w = load_vec(m, q + "ln2.weight"); B.ln2_b = load_vec(m, q + "ln2.bias");
            L.ae_blocks.push_back(std::move(B));
        }
        L.queue = tensor_f32(m, p + "act_queue");
        pm->levels.push_back(std::move(L));
    }
    return pm;
}

// ---------------------------------------------------------------------------------------------
// reverse-mode tape over row batches: a node is [n rows][d] f32, a row is one sample (or one token)
// ---------------------------------------------------------------------------------------------
struct tape {
    struct node {
        int n = 0, d = 0;
        std::vector<float> v, g;              // g is allocated when a gradient first arrives
        std::function<void(tape &, int)> back; // pushes this node's g into its inputs
    };
    std::deque<node> nodes;
    bool want_grad = true;

    int add(int n, int d) {
        nodes.emplace_back();
        node & x = nodes.back();
        x.n = n; x.d = d;
        x.v.assign((size_t) n * d, 0.0f);
        return (int) nodes.size() - 1;
    }
    float * v(int id) { return nodes[(size_t) id].v.data(); }
    float * g(int id) {
        node & x = nodes[(size_t) id];
        if (x.g.empty()) x.g.assign((size_t) x.n * x.d, 0.0f);
        return x.g.data();
    }
    bool has_g(int id) const { return !nodes[(size_t) id].g.empty(); }
    int n(int id) const { return nodes[(size_t) id].n; }
    int d(int id) const { return nodes[(size_t) id].d; }

    // Nodes are created in topological order, so one reverse sweep is the whole backward pass.
    // A node's gradient is dropped once it has been pushed on, except for the leaves the caller reads.
    void backward() {
        for (int i = (int) nodes.size() - 1; i >= 0; i--) {
            node & x = nodes[(size_t) i];
            if (x.g.empty() || !x.back) continue;
            x.back(*this, i);
            std::vector<float>().swap(x.g);
        }
    }
};

// y[r] += x[r][i] * wt[i][:] for every row: the weight row is reused across the batch while hot.
// This file is compiled with floating-point contraction off (CMakeLists.txt): with it on, the
// compiler fuses the multiply-add in the vectorised body of this loop and not in its scalar
// remainder, so which elements got an FMA depended on where malloc had put the buffer, and the
// same call returned different last bits from one run to the next.
static void gemm_acc(const float * x, int n, int in, const float * wt, int out, float * y) {
    for (int i = 0; i < in; i++) {
        const float * GGML_RESTRICT wr = wt + (size_t) i * out;
        for (int r = 0; r < n; r++) {
            const float a = x[(size_t) r * in + i];
            float * GGML_RESTRICT yr = y + (size_t) r * out;
            for (int o = 0; o < out; o++) yr[o] += a * wr[o];
        }
    }
}

static int op_input(tape & t, int n, int d, const float * rows) {
    const int y = t.add(n, d);
    memcpy(t.v(y), rows, (size_t) n * d * sizeof(float));
    return y;
}

static int op_linear(tape & t, int x, const lin & L) {
    const int n = t.n(x);
    GGML_ASSERT(t.d(x) == L.in);
    const int y = t.add(n, L.out);
    float * yv = t.v(y);
    if (!L.b.empty()) {
        for (int r = 0; r < n; r++) memcpy(yv + (size_t) r * L.out, L.b.data(), (size_t) L.out * sizeof(float));
    }
    gemm_acc(t.v(x), n, L.in, L.wt.data(), L.out, yv);
    if (t.want_grad) {
        const lin * Lp = &L;
        t.nodes[(size_t) y].back = [x, Lp](tape & tp, int self) {
            gemm_acc(tp.g(self), tp.n(self), Lp->out, Lp->w.data(), Lp->in, tp.g(x));
        };
    }
    return y;
}

// y = (x - mean) / sqrt(var + eps) * w + b, over each row; w / b optional
static int op_layer_norm(tape & t, int x, const vecp * w, const vecp * b, float eps) {
    const int n = t.n(x), d = t.d(x);
    const int y = t.add(n, d);
    auto xhat = std::make_shared<std::vector<float>>((size_t) n * d);
    auto inv = std::make_shared<std::vector<float>>((size_t) n);
    const float * xv = t.v(x);
    float * yv = t.v(y);
    const float * wv = w && *w ? w->v.data() : nullptr;
    const float * bv = b && *b ? b->v.data() : nullptr;
    for (int r = 0; r < n; r++) {
        const float * xr = xv + (size_t) r * d;
        double mean = 0;
        for (int i = 0; i < d; i++) mean += xr[i];
        mean /= d;
        double var = 0;
        for (int i = 0; i < d; i++) { const double c = xr[i] - mean; var += c * c; }
        var /= d;
        const float is = (float) (1.0 / std::sqrt(var + (double) eps));
        (*inv)[(size_t) r] = is;
        float * hr = xhat->data() + (size_t) r * d;
        float * yr = yv + (size_t) r * d;
        for (int i = 0; i < d; i++) {
            hr[i] = (float) (xr[i] - mean) * is;
            yr[i] = hr[i] * (wv ? wv[i] : 1.0f) + (bv ? bv[i] : 0.0f);
        }
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [x, xhat, inv, wv](tape & tp, int self) {
            const int n = tp.n(self), d = tp.d(self);
            const float * gy = tp.g(self);
            float * gx = tp.g(x);
            std::vector<float> gh((size_t) d);
            for (int r = 0; r < n; r++) {
                const float * hr = xhat->data() + (size_t) r * d;
                double m1 = 0, m2 = 0;
                for (int i = 0; i < d; i++) {
                    gh[(size_t) i] = gy[(size_t) r * d + i] * (wv ? wv[i] : 1.0f);
                    m1 += gh[(size_t) i];
                    m2 += (double) gh[(size_t) i] * hr[i];
                }
                m1 /= d; m2 /= d;
                const float is = (*inv)[(size_t) r];
                for (int i = 0; i < d; i++) {
                    gx[(size_t) r * d + i] += (float) ((gh[(size_t) i] - m1 - hr[i] * m2) * is);
                }
            }
        };
    }
    return y;
}

static int op_act(tape & t, int x, jepa_act_id act) {
    const int n = t.n(x), d = t.d(x);
    const int y = t.add(n, d);
    const float * xv = t.v(x);
    float * yv = t.v(y);
    const size_t N = (size_t) n * d;
    switch (act) {
        case JEPA_ACT_SILU: for (size_t i = 0; i < N; i++) yv[i] = xv[i] / (1.0f + std::exp(-xv[i])); break;
        case JEPA_ACT_RELU: for (size_t i = 0; i < N; i++) yv[i] = xv[i] > 0 ? xv[i] : 0.0f; break;
        default:            for (size_t i = 0; i < N; i++) yv[i] = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f)); break;
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [x, act](tape & tp, int self) {
            const size_t N = (size_t) tp.n(self) * tp.d(self);
            const float * xv = tp.v(x);
            const float * gy = tp.g(self);
            float * gx = tp.g(x);
            for (size_t i = 0; i < N; i++) {
                float dv;
                const float v = xv[i];
                if (act == JEPA_ACT_SILU) {
                    const float s = 1.0f / (1.0f + std::exp(-v));
                    dv = s * (1.0f + v * (1.0f - s));
                } else if (act == JEPA_ACT_RELU) {
                    dv = v > 0 ? 1.0f : 0.0f;
                } else {   // d/dx [x Phi(x)] = Phi(x) + x phi(x)
                    dv = 0.5f * (1.0f + std::erf(v * 0.70710678118654752f)) + v * 0.39894228040143268f * std::exp(-0.5f * v * v);
                }
                gx[i] += gy[i] * dv;
            }
        };
    }
    return y;
}

static int op_add(tape & t, int a, int b) {
    const int n = t.n(a), d = t.d(a);
    const int y = t.add(n, d);
    const float * av = t.v(a), * bv = t.v(b);
    float * yv = t.v(y);
    for (size_t i = 0; i < (size_t) n * d; i++) yv[i] = av[i] + bv[i];
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [a, b](tape & tp, int self) {
            const size_t N = (size_t) tp.n(self) * tp.d(self);
            const float * gy = tp.g(self);
            float * ga = tp.g(a);
            for (size_t i = 0; i < N; i++) ga[i] += gy[i];
            float * gb = tp.g(b);
            gy = tp.g(self);
            for (size_t i = 0; i < N; i++) gb[i] += gy[i];
        };
    }
    return y;
}

// y = x + row (the same constant row for every sample: a position embedding)
static int op_add_row(tape & t, int x, const float * row) {
    const int n = t.n(x), d = t.d(x);
    const int y = t.add(n, d);
    const float * xv = t.v(x);
    float * yv = t.v(y);
    for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) yv[(size_t) r * d + i] = xv[(size_t) r * d + i] + row[i];
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [x](tape & tp, int self) {
            const size_t N = (size_t) tp.n(self) * tp.d(self);
            const float * gy = tp.g(self);
            float * gx = tp.g(x);
            for (size_t i = 0; i < N; i++) gx[i] += gy[i];
        };
    }
    return y;
}

// The adaLN modulation: y = h * (1 + mod[:, sc]) + mod[:, sh], with sc / sh two D-wide column
// blocks of `mod` (D = width of h).
static int op_film(tape & t, int h, int mod, int sc_off, int sh_off) {
    const int n = t.n(h), d = t.d(h), md = t.d(mod);
    const int y = t.add(n, d);
    const float * hv = t.v(h), * mv = t.v(mod);
    float * yv = t.v(y);
    for (int r = 0; r < n; r++) {
        const float * sc = mv + (size_t) r * md + sc_off, * sh = mv + (size_t) r * md + sh_off;
        for (int i = 0; i < d; i++) yv[(size_t) r * d + i] = hv[(size_t) r * d + i] * (1.0f + sc[i]) + sh[i];
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [h, mod, sc_off, sh_off](tape & tp, int self) {
            const int n = tp.n(self), d = tp.d(self), md = tp.d(mod);
            const float * gy = tp.g(self);
            const float * hv = tp.v(h), * mv = tp.v(mod);
            float * gh = tp.g(h);
            float * gm = tp.g(mod);
            for (int r = 0; r < n; r++) {
                for (int i = 0; i < d; i++) {
                    const float gyi = gy[(size_t) r * d + i];
                    gh[(size_t) r * d + i] += gyi * (1.0f + mv[(size_t) r * md + sc_off + i]);
                    gm[(size_t) r * md + sc_off + i] += gyi * hv[(size_t) r * d + i];
                    gm[(size_t) r * md + sh_off + i] += gyi;
                }
            }
        };
    }
    return y;
}

// y = x + a * mod[:, gate]   (the gated residual of an adaLN block)
static int op_gated_add(tape & t, int x, int a, int mod, int gate_off) {
    const int n = t.n(x), d = t.d(x), md = t.d(mod);
    const int y = t.add(n, d);
    const float * xv = t.v(x), * av = t.v(a), * mv = t.v(mod);
    float * yv = t.v(y);
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < d; i++) {
            yv[(size_t) r * d + i] = xv[(size_t) r * d + i] + av[(size_t) r * d + i] * mv[(size_t) r * md + gate_off + i];
        }
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [x, a, mod, gate_off](tape & tp, int self) {
            const int n = tp.n(self), d = tp.d(self), md = tp.d(mod);
            const float * gy = tp.g(self);
            const float * av = tp.v(a), * mv = tp.v(mod);
            float * gx = tp.g(x);
            float * ga = tp.g(a);
            float * gm = tp.g(mod);
            for (int r = 0; r < n; r++) {
                for (int i = 0; i < d; i++) {
                    const float gyi = gy[(size_t) r * d + i];
                    gx[(size_t) r * d + i] += gyi;
                    ga[(size_t) r * d + i] += gyi * mv[(size_t) r * md + gate_off + i];
                    gm[(size_t) r * md + gate_off + i] += gyi * av[(size_t) r * d + i];
                }
            }
        };
    }
    return y;
}

// Columns side by side: [n][sum d_i]
static int op_concat_cols(tape & t, const std::vector<int> & xs) {
    const int n = t.n(xs[0]);
    int d = 0;
    for (int x : xs) d += t.d(x);
    const int y = t.add(n, d);
    float * yv = t.v(y);
    int off = 0;
    for (int x : xs) {
        const int dx = t.d(x);
        const float * xv = t.v(x);
        for (int r = 0; r < n; r++) memcpy(yv + (size_t) r * d + off, xv + (size_t) r * dx, (size_t) dx * sizeof(float));
        off += dx;
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [xs](tape & tp, int self) {
            const int n = tp.n(self), d = tp.d(self);
            int off = 0;
            for (int x : xs) {
                const int dx = tp.d(x);
                float * gx = tp.g(x);
                const float * gy = tp.g(self);
                for (int r = 0; r < n; r++) for (int i = 0; i < dx; i++) gx[(size_t) r * dx + i] += gy[(size_t) r * d + off + i];
                off += dx;
            }
        };
    }
    return y;
}

// Token sequences for the 'seq' action encoder: sample r becomes the rows r*(L+1) .. of the
// result, a constant CLS row followed by its L step rows (each plus its position row).
static int op_tokens(tape & t, const std::vector<int> & steps, const float * pos /* [L+1][d] */) {
    const int L = (int) steps.size(), n = t.n(steps[0]), d = t.d(steps[0]);
    const int y = t.add(n * (L + 1), d);
    float * yv = t.v(y);
    for (int r = 0; r < n; r++) {
        memcpy(yv + (size_t) r * (L + 1) * d, pos, (size_t) d * sizeof(float));
        for (int l = 0; l < L; l++) {
            const float * xv = t.v(steps[(size_t) l]) + (size_t) r * d;
            float * row = yv + ((size_t) r * (L + 1) + 1 + l) * d;
            for (int i = 0; i < d; i++) row[i] = xv[i] + pos[(size_t) (l + 1) * d + i];
        }
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [steps](tape & tp, int self) {
            const int L = (int) steps.size(), d = tp.d(self), n = tp.n(self) / (L + 1);
            for (int l = 0; l < L; l++) {
                float * gx = tp.g(steps[(size_t) l]);
                const float * gy = tp.g(self);
                for (int r = 0; r < n; r++) {
                    for (int i = 0; i < d; i++) gx[(size_t) r * d + i] += gy[((size_t) r * (L + 1) + 1 + l) * d + i];
                }
            }
        };
    }
    return y;
}

// Row 0 of every group of `group` rows
static int op_first_of_group(tape & t, int x, int group) {
    const int n = t.n(x) / group, d = t.d(x);
    const int y = t.add(n, d);
    for (int r = 0; r < n; r++) memcpy(t.v(y) + (size_t) r * d, t.v(x) + (size_t) r * group * d, (size_t) d * sizeof(float));
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [x, group](tape & tp, int self) {
            const int n = tp.n(self), d = tp.d(self);
            float * gx = tp.g(x);
            const float * gy = tp.g(self);
            for (int r = 0; r < n; r++) for (int i = 0; i < d; i++) gx[(size_t) r * group * d + i] += gy[(size_t) r * d + i];
        };
    }
    return y;
}

// Multi-head self-attention inside each group of `group` token rows. qkv: [rows][3 d] = [q | k | v].
static int op_mha(tape & t, int qkv, int group, int n_head) {
    const int rows = t.n(qkv), d = t.d(qkv) / 3, hd = d / n_head, G = group;
    const int y = t.add(rows, d);
    auto probs = std::make_shared<std::vector<float>>((size_t) rows * n_head * G);   // [row][head][key]
    const float scale = 1.0f / std::sqrt((float) hd);
    const float * x = t.v(qkv);
    float * yv = t.v(y);
    for (int g0 = 0; g0 < rows; g0 += G) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < G; i++) {
                const float * q = x + (size_t) (g0 + i) * 3 * d + h * hd;
                float * p = probs->data() + ((size_t) (g0 + i) * n_head + h) * G;
                float mx = -INFINITY;
                for (int j = 0; j < G; j++) {
                    const float * k = x + (size_t) (g0 + j) * 3 * d + d + h * hd;
                    float s = 0;
                    for (int c = 0; c < hd; c++) s += q[c] * k[c];
                    p[j] = s * scale;
                    mx = std::max(mx, p[j]);
                }
                float sum = 0;
                for (int j = 0; j < G; j++) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
                float * o = yv + (size_t) (g0 + i) * d + h * hd;
                for (int j = 0; j < G; j++) {
                    p[j] /= sum;
                    const float * v = x + (size_t) (g0 + j) * 3 * d + 2 * d + h * hd;
                    for (int c = 0; c < hd; c++) o[c] += p[j] * v[c];
                }
            }
        }
    }
    if (t.want_grad) {
        t.nodes[(size_t) y].back = [qkv, G, n_head, probs, scale](tape & tp, int self) {
            const int rows = tp.n(self), d = tp.d(self), hd = d / n_head;
            const float * x = tp.v(qkv);
            const float * go = tp.g(self);
            float * gx = tp.g(qkv);
            std::vector<float> gp((size_t) G);
            for (int g0 = 0; g0 < rows; g0 += G) {
                for (int h = 0; h < n_head; h++) {
                    for (int i = 0; i < G; i++) {
                        const float * p = probs->data() + ((size_t) (g0 + i) * n_head + h) * G;
                        const float * goi = go + (size_t) (g0 + i) * d + h * hd;
                        float dotp = 0;
                        for (int j = 0; j < G; j++) {
                            const float * v = x + (size_t) (g0 + j) * 3 * d + 2 * d + h * hd;
                            float * gv = gx + (size_t) (g0 + j) * 3 * d + 2 * d + h * hd;
                            float s = 0;
                            for (int c = 0; c < hd; c++) { s += goi[c] * v[c]; gv[c] += p[j] * goi[c]; }
                            gp[(size_t) j] = s;
                            dotp += s * p[j];
                        }
                        const float * q = x + (size_t) (g0 + i) * 3 * d + h * hd;
                        float * gq = gx + (size_t) (g0 + i) * 3 * d + h * hd;
                        for (int j = 0; j < G; j++) {
                            const float gs = p[j] * (gp[(size_t) j] - dotp) * scale;   // d cost / d score
                            const float * k = x + (size_t) (g0 + j) * 3 * d + d + h * hd;
                            float * gk = gx + (size_t) (g0 + j) * 3 * d + d + h * hd;
                            for (int c = 0; c < hd; c++) { gq[c] += gs * k[c]; gk[c] += gs * q[c]; }
                        }
                    }
                }
            }
        };
    }
    return y;
}

// ---------------------------------------------------------------------------------------------
// the model on the tape
// ---------------------------------------------------------------------------------------------
static int build_act_embed(tape & t, const level_w & L, int a) {
    if (!L.act0) return a;
    return op_linear(t, op_act(t, op_linear(t, a, L.act0), JEPA_ACT_SILU), L.act2);
}

// One predictor step with a context of one state: s [n][D], e = act_embed(a) [n][E] -> next state.
static int build_step(tape & t, const plan_model & pm, const level_w & L, int s, int e) {
    int c = e;
    if (L.pact0) c = op_linear(t, op_act(t, op_linear(t, c, L.pact0), JEPA_ACT_SILU), L.pact2);
    int x = op_add_row(t, s, L.pos0.data());
    if (L.in_proj) x = op_linear(t, x, L.in_proj);
    if (L.cond_proj) c = op_linear(t, c, L.cond_proj);
    const int cs = op_act(t, c, JEPA_ACT_SILU);
    const int D = t.d(x);
    for (const block_w & B : L.blocks) {
        const int mod = op_linear(t, cs, B.adaln);   // [shift_a | scale_a | gate_a | shift_m | scale_m | gate_m]
        int h = op_film(t, op_layer_norm(t, x, nullptr, nullptr, pm.adaln_eps), mod, D, 0);
        if (B.ln1_w) h = op_layer_norm(t, h, &B.ln1_w, &B.ln1_b, pm.ln_eps);
        x = op_gated_add(t, x, op_linear(t, op_linear(t, h, B.v), B.out), mod, 2 * D);
        h = op_film(t, op_layer_norm(t, x, nullptr, nullptr, pm.adaln_eps), mod, 4 * D, 3 * D);
        if (B.ln2_w) h = op_layer_norm(t, h, &B.ln2_w, &B.ln2_b, pm.ln_eps);
        x = op_gated_add(t, x, op_linear(t, op_act(t, op_linear(t, h, B.up), JEPA_ACT_GELU_ERF), B.down), mod, 5 * D);
    }
    x = op_layer_norm(t, x, &L.norm_w, &L.norm_b, pm.ln_eps);
    if (L.out_proj) x = op_linear(t, x, L.out_proj);
    if (L.proj0) x = op_linear(t, op_act(t, op_linear(t, x, L.proj0), JEPA_ACT_GELU_ERF), L.proj2);
    return x;
}

// The state encoder of `up` applied to one state of the level below.
static int build_encode_up(tape & t, const plan_model & pm, const level_w & up, int s) {
    if (up.hp.enc_kind == "identity") return s;
    int x = op_layer_norm(t, s, &up.enc_in_w, &up.enc_in_b, pm.ln_eps);
    x = op_linear(t, op_act(t, op_linear(t, x, up.enc_fc1), JEPA_ACT_GELU_ERF), up.enc_fc2);
    if (up.enc_out_w) x = op_layer_norm(t, x, &up.enc_out_w, &up.enc_out_b, pm.ln_eps);
    return x;
}

// The action encoder of `up` over one full chunk of the level below's action embeddings.
static int build_action_pool(tape & t, const plan_model & pm, const level_w & up, const std::vector<int> & chunk) {
    if (up.hp.act_enc_kind == "mlp") {
        int y = op_linear(t, op_act(t, op_linear(t, op_concat_cols(t, chunk), up.ae_fc1), JEPA_ACT_RELU), up.ae_fc2);
        if (up.ae_out_w) y = op_layer_norm(t, y, &up.ae_out_w, &up.ae_out_b, pm.ln_eps);
        return y;
    }
    std::vector<int> steps;
    for (int e : chunk) steps.push_back(op_linear(t, e, up.ae_step));
    const int G = (int) chunk.size() + 1;
    int x = op_tokens(t, steps, up.ae_pos.data());
    for (const seq_block_w & B : up.ae_blocks) {
        int h = op_layer_norm(t, x, &B.ln1_w, &B.ln1_b, pm.ln_eps);
        x = op_add(t, x, op_linear(t, op_mha(t, op_linear(t, h, B.qkv), G, up.hp.act_enc_n_head), B.out));
        h = op_layer_norm(t, x, &B.ln2_w, &B.ln2_b, pm.ln_eps);
        x = op_add(t, x, op_linear(t, op_act(t, op_linear(t, h, B.up), JEPA_ACT_RELU), B.down));
    }
    int y = op_first_of_group(t, x, G);
    y = op_act(t, op_linear(t, y, up.ae_head0), JEPA_ACT_GELU_ERF);
    y = op_layer_norm(t, y, &up.ae_headln_w, &up.ae_headln_b, pm.ln_eps);
    y = op_linear(t, y, up.ae_head2);
    return op_layer_norm(t, y, &up.ae_out_w, &up.ae_out_b, 1e-6f);
}

// ---------------------------------------------------------------------------------------------
// the cost of a batch of candidates, and its gradient
// ---------------------------------------------------------------------------------------------
struct cost_spec {
    int level = 1;                       // 1-based level being planned
    int horizon = 1;                     // T
    const float * s0 = nullptr;          // [D_level] the current state
    const float * goal = nullptr;        // [n_goal][D_goal_level]
    int n_goal = 1;
    int goal_level = 1;                  // level the goal states live at (>= level)
    bool dense = false;                  // goal_level > level reached at stride 1 (every predicted step is scored)
    const float * goal_action = nullptr; // [n_goal][A_{level+1}] or null
    int cost_last_n = 1;
    float intermediate_cost_weight = 1.0f;
    float action_cost_weight = 0.0f;
};

// Indices 0, stride, 2*stride, ... of a list
static std::vector<int> every(const std::vector<int> & xs, int stride) {
    std::vector<int> out;
    for (size_t i = 0; i < xs.size(); i += (size_t) stride) out.push_back(xs[i]);
    return out;
}

// Number of predicted states the cost reads once the predictions have been encoded up to the
// goal's level and the context state dropped. 0 = the horizon is too short to reach one.
static int scored_states(const plan_model & pm, const cost_spec & cs) {
    int len = cs.horizon + 1;
    for (int lv = cs.level + 1; lv <= cs.goal_level; lv++) {
        const int stride = cs.dense ? 1 : pm.levels[(size_t) lv - 1].hp.stride;
        len = (len - 1) / stride + 1;
    }
    return len - 1;
}

// Cost (and gradient with respect to the actions) of `n` candidates on one tape.
//   actions [n][T][A] row-major; costs [n]; grad [n][T][A] or null; states [T + 1][D] of
//   candidate 0 or null.
static void chunk_cost(const plan_model & pm, const cost_spec & cs, const float * actions, int n,
                       float * costs, float * grad, float * states) {
    const level_w & L = pm.levels[(size_t) cs.level - 1];
    const int D = L.hp.embed_dim, A = L.hp.action_dim, T = cs.horizon;
    tape t;
    t.want_grad = grad != nullptr;

    std::vector<float> buf((size_t) n * std::max(D, A));
    for (int r = 0; r < n; r++) memcpy(buf.data() + (size_t) r * D, cs.s0, (size_t) D * sizeof(float));
    std::vector<int> pred = { op_input(t, n, D, buf.data()) }, act_in, act_emb;
    for (int step = 0; step < T; step++) {
        for (int r = 0; r < n; r++) memcpy(buf.data() + (size_t) r * A, actions + ((size_t) r * T + step) * A, (size_t) A * sizeof(float));
        act_in.push_back(op_input(t, n, A, buf.data()));
        act_emb.push_back(build_act_embed(t, L, act_in.back()));
        pred.push_back(build_step(t, pm, L, pred.back(), act_emb.back()));
    }
    if (states) {
        for (int step = 0; step <= T; step++) memcpy(states + (size_t) step * D, t.v(pred[(size_t) step]), (size_t) D * sizeof(float));
    }

    // encode the predicted stream up to the goal's level
    std::vector<int> scored = pred;
    for (int lv = cs.level + 1; lv <= cs.goal_level; lv++) {
        const level_w & up = pm.levels[(size_t) lv - 1];
        scored = every(scored, cs.dense ? 1 : up.hp.stride);
        for (int & s : scored) s = build_encode_up(t, pm, up, s);
    }
    scored.erase(scored.begin());   // the context state is not a prediction
    const int Dg = pm.levels[(size_t) cs.goal_level - 1].hp.embed_dim;
    const int W = std::min(std::max(1, cs.cost_last_n), (int) scored.size());
    for (int r = 0; r < n; r++) costs[r] = 0.0f;
    for (int j = 0; j < W; j++) {
        const int id = scored[scored.size() - (size_t) W + (size_t) j];
        const float * g = cs.goal + (size_t) (cs.n_goal == 1 ? 0 : j) * Dg;
        const float w = (cs.intermediate_cost_weight != 1.0f && W > 1 && j < W - 1) ? cs.intermediate_cost_weight : 1.0f;
        const float * pv = t.v(id);
        float * pg = grad ? t.g(id) : nullptr;
        for (int r = 0; r < n; r++) {
            double acc = 0;
            for (int i = 0; i < Dg; i++) {
                const float diff = pv[(size_t) r * Dg + i] - g[i];
                acc += (double) diff * diff;
                if (pg) pg[(size_t) r * Dg + i] += 2.0f * w * diff;
            }
            costs[r] += w * (float) acc;
        }
    }

    // action cost: the candidates pooled into the upper level's macro-actions, against the ones
    // the upper level chose. Only whole chunks are compared (the reference drops the last one).
    if (cs.action_cost_weight != 0.0f && cs.goal_action && cs.goal_level == cs.level + 1 && !cs.dense) {
        const level_w & up = pm.levels[(size_t) cs.level];
        const int stride = up.hp.stride, Au = up.hp.action_dim;
        const int n_pooled = T / stride;   // chunks starting at 0, stride, ... <= T, minus the last
        const int Wa = std::min({ std::max(1, cs.cost_last_n), n_pooled, cs.n_goal });
        for (int j = 0; j < Wa; j++) {
            const int chunk = n_pooled - Wa + j;
            std::vector<int> ids(act_emb.begin() + (size_t) chunk * stride, act_emb.begin() + (size_t) (chunk + 1) * stride);
            const int id = build_action_pool(t, pm, up, ids);
            const float * g = cs.goal_action + (size_t) (cs.n_goal - Wa + j) * Au;
            const float * pv = t.v(id);
            float * pg = grad ? t.g(id) : nullptr;
            for (int r = 0; r < n; r++) {
                double acc = 0;
                for (int i = 0; i < Au; i++) {
                    const float diff = pv[(size_t) r * Au + i] - g[i];
                    acc += (double) diff * diff;
                    if (pg) pg[(size_t) r * Au + i] += 2.0f * cs.action_cost_weight * diff;
                }
                costs[r] += cs.action_cost_weight * (float) acc;
            }
        }
    }

    if (grad) {
        t.backward();
        for (int step = 0; step < T; step++) {
            const int id = act_in[(size_t) step];
            if (!t.has_g(id)) {
                for (int r = 0; r < n; r++) memset(grad + ((size_t) r * T + step) * A, 0, (size_t) A * sizeof(float));
                continue;
            }
            const float * g = t.g(id);
            for (int r = 0; r < n; r++) memcpy(grad + ((size_t) r * T + step) * A, g + (size_t) r * A, (size_t) A * sizeof(float));
        }
    }
}

// Candidates per tape at most: enough rows to keep a weight row hot across the batch, few enough
// that a tape (every activation of every step, kept for the backward pass) stays a few MiB.
const int PLAN_CHUNK = 8;

static void batch_cost(const plan_model & pm, const cost_spec & cs, const float * actions, int n_samples,
                       float * costs, float * grad, int n_threads) {
    const level_w & L = pm.levels[(size_t) cs.level - 1];
    const size_t row = (size_t) cs.horizon * L.hp.action_dim;
    // smaller chunks when there are more threads than full chunks: a sample's arithmetic does not
    // depend on its chunk, so this changes the speed and nothing else
    const int chunk = std::max(1, std::min(PLAN_CHUNK, (n_samples + std::max(1, n_threads) - 1) / std::max(1, n_threads)));
    const int n_chunks = (n_samples + chunk - 1) / chunk;
    std::atomic<int> next{0};
    auto work = [&]() {
        for (;;) {
            const int c = next.fetch_add(1);
            if (c >= n_chunks) return;
            const int r0 = c * chunk, n = std::min(chunk, n_samples - r0);
            chunk_cost(pm, cs, actions + (size_t) r0 * row, n, costs + r0, grad ? grad + (size_t) r0 * row : nullptr, nullptr);
        }
    };
    const int nt = std::max(1, std::min(n_threads, n_chunks));
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; i++) pool.emplace_back(work);
    work();
    for (std::thread & th : pool) th.join();
}

// ---------------------------------------------------------------------------------------------
// one level's gradient solver (stable_worldmodel.solver.GradientSolver)
// ---------------------------------------------------------------------------------------------
struct prior_stats { std::vector<float> mean, std, lo, hi; bool has_queue = false; };

static prior_stats prior_of(const level_w & L, float clip_sigma) {
    prior_stats p;
    const int A = L.hp.action_dim;
    const int64_t n = (int64_t) L.queue.size() / A;
    p.mean.assign((size_t) A, 0.0f);
    p.std.assign((size_t) A, 1.0f);
    p.has_queue = n > 0;
    if (p.has_queue) {
        for (int a = 0; a < A; a++) {
            double m = 0, v = 0;
            for (int64_t i = 0; i < n; i++) m += L.queue[(size_t) (i * A + a)];
            m /= (double) n;
            for (int64_t i = 0; i < n; i++) { const double c = L.queue[(size_t) (i * A + a)] - m; v += c * c; }
            p.mean[(size_t) a] = (float) m;
            p.std[(size_t) a] = std::max((float) std::sqrt(v / (double) n), 1e-4f);   // population std, floored
        }
    }
    if (clip_sigma > 0.0f) {
        for (int a = 0; a < A; a++) {
            p.lo.push_back(p.mean[(size_t) a] - clip_sigma * p.std[(size_t) a]);
            p.hi.push_back(p.mean[(size_t) a] + clip_sigma * p.std[(size_t) a]);
        }
    } else if (p.has_queue) {
        // the 2 % and 98 % quantiles of the recorded latent actions, linearly interpolated
        std::vector<float> col((size_t) n);
        for (int a = 0; a < A; a++) {
            for (int64_t i = 0; i < n; i++) col[(size_t) i] = L.queue[(size_t) (i * A + a)];
            std::sort(col.begin(), col.end());
            auto quantile = [&](double q) {
                const double pos = q * (double) (n - 1);
                const int64_t i0 = (int64_t) std::floor(pos), i1 = std::min(i0 + 1, n - 1);
                const double f = pos - (double) i0;
                return (float) ((double) col[(size_t) i0] + ((double) col[(size_t) i1] - (double) col[(size_t) i0]) * f);
            };
            p.lo.push_back(quantile(0.02));
            p.hi.push_back(quantile(0.98));
        }
    }
    return p;
}

struct noise_source {
    jepa_randn_fn fn = nullptr;
    void * user = nullptr;
    std::mt19937_64 rng;
    std::normal_distribution<float> dist{0.0f, 1.0f};
    void fill(float * out, size_t n) {
        if (fn) { fn(user, out, n); return; }
        for (size_t i = 0; i < n; i++) out[i] = dist(rng);
    }
};

struct solve_out {
    std::vector<float> actions;   // [T][A] the lowest-cost candidate
    std::vector<float> states;    // [T + 1][D] the states it predicts, the current state first
    float cost = 0;
    int iters = 0;
};

static solve_out solve_level(const plan_model & pm, const cost_spec & cs, const jepa_hjepa_level_plan & lp,
                             noise_source & noise, int n_threads) {
    const level_w & L = pm.levels[(size_t) cs.level - 1];
    const int A = L.hp.action_dim, T = cs.horizon, S = std::max(1, lp.n_samples);
    const size_t row = (size_t) T * A, N = (size_t) S * row;
    const prior_stats prior = prior_of(L, lp.action_clip_sigma);

    // every candidate starts at the prior mean; all but the first are perturbed
    std::vector<float> x(N), z(N);
    for (size_t i = 0; i < N; i++) x[i] = prior.mean[i % (size_t) A];
    if (S > 1) {
        noise.fill(z.data(), N - row);
        for (size_t i = 0; i < N - row; i++) x[row + i] += z[i] * prior.std[i % (size_t) A] * lp.var_scale;
    }

    // AdamW, PyTorch defaults
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    std::vector<float> m(N, 0.0f), v(N, 0.0f), grad(N), costs((size_t) S);
    solve_out out;
    float best = INFINITY;
    int stalled = 0;
    for (int step = 0; step < lp.n_steps; step++) {
        batch_cost(pm, cs, x.data(), S, costs.data(), grad.data(), n_threads);
        const float c1 = 1.0f - std::pow(b1, (float) (step + 1)), c2 = 1.0f - std::pow(b2, (float) (step + 1));
        const float step_size = lp.lr / c1, c2s = std::sqrt(c2);
        for (size_t i = 0; i < N; i++) {
            x[i] *= 1.0f - lp.lr * lp.weight_decay;
            m[i] = b1 * m[i] + (1.0f - b1) * grad[i];
            v[i] = b2 * v[i] + (1.0f - b2) * grad[i] * grad[i];
            x[i] -= step_size * m[i] / (std::sqrt(v[i]) / c2s + eps);
        }
        if (lp.action_noise > 0.0f) {
            noise.fill(z.data(), N);
            for (size_t i = 0; i < N; i++) x[i] += z[i] * lp.action_noise;
        }
        if (!prior.lo.empty()) {
            for (size_t i = 0; i < N; i++) x[i] = std::min(std::max(x[i], prior.lo[i % (size_t) A]), prior.hi[i % (size_t) A]);
        }
        out.iters = step + 1;
        if (lp.early_stop_patience > 0) {
            // on the costs of the candidates as they were BEFORE this update, as the reference does
            const float score = *std::min_element(costs.begin(), costs.end());
            const bool improved = !std::isfinite(best) || best - score > lp.early_stop_rel_delta * std::max(std::fabs(best), 1e-12f);
            if (improved) { best = score; stalled = 0; }
            else if (++stalled >= lp.early_stop_patience && step + 1 >= lp.min_its) break;
        }
    }
    batch_cost(pm, cs, x.data(), S, costs.data(), nullptr, n_threads);
    const size_t top = (size_t) (std::min_element(costs.begin(), costs.end()) - costs.begin());
    out.cost = costs[top];
    out.actions.assign(x.begin() + (std::ptrdiff_t) (top * row), x.begin() + (std::ptrdiff_t) ((top + 1) * row));
    out.states.resize((size_t) (T + 1) * L.hp.embed_dim);
    float c;
    chunk_cost(pm, cs, out.actions.data(), 1, &c, nullptr, out.states.data());
    return out;
}

// ---------------------------------------------------------------------------------------------
// argument checking shared by the entry points
// ---------------------------------------------------------------------------------------------
static std::shared_ptr<plan_model> plan_model_of(jepa_context * ctx, const char * fn) {
    if (!ctx->model->hp.hjepa.present) {
        jepa_log("jepa: %s: '%s' is not an H-JEPA model (jepa.family = '%s')\n", fn, ctx->model->hp.name.c_str(),
                 ctx->model->hp.family_str.c_str());
        return nullptr;
    }
    if (!ctx->hjepa_plan) ctx->hjepa_plan = build_plan_model(ctx->model);
    return std::static_pointer_cast<plan_model>(ctx->hjepa_plan);
}

static bool check_spec(const plan_model & pm, const cost_spec & cs, const char * fn) {
    const int n_levels = (int) pm.levels.size();
    if (cs.level < 1 || cs.level > n_levels) {
        jepa_log("jepa: %s: level %d is out of range, the model has levels 1..%d\n", fn, cs.level, n_levels);
        return false;
    }
    if (cs.goal_level < cs.level || cs.goal_level > n_levels) {
        jepa_log("jepa: %s: the goal's level %d has to be in %d..%d (the planned level up to the top)\n",
                 fn, cs.goal_level, cs.level, n_levels);
        return false;
    }
    if (cs.horizon < 1 || cs.horizon > JEPA_LIMIT_N_FRAMES) {
        jepa_log("jepa: %s: horizon %d is out of range [1, %d]\n", fn, cs.horizon, JEPA_LIMIT_N_FRAMES);
        return false;
    }
    if (!cs.s0 || !cs.goal || cs.n_goal < 1) {
        jepa_log("jepa: %s: the current state and at least one goal state are required\n", fn);
        return false;
    }
    if (cs.dense && cs.goal_level == cs.level) {
        jepa_log("jepa: %s: a dense goal projection needs a goal above the planned level\n", fn);
        return false;
    }
    const int scored = scored_states(pm, cs);
    if (scored < 1) {
        jepa_log("jepa: %s: horizon %d at level %d does not reach one state of level %d (strides multiply)\n",
                 fn, cs.horizon, cs.level, cs.goal_level);
        return false;
    }
    const int W = std::min(std::max(1, cs.cost_last_n), scored);
    if (cs.n_goal != 1 && cs.n_goal != W) {
        jepa_log("jepa: %s: %d goal states for %d scored predictions (horizon %d, cost_last_n %d): give one goal, "
                 "or exactly one per scored prediction\n", fn, cs.n_goal, W, cs.horizon, cs.cost_last_n);
        return false;
    }
    return true;
}

static cost_spec spec_from(int level, int horizon, const float * state, const jepa_hjepa_goal * goal,
                           const jepa_hjepa_level_plan * lp) {
    cost_spec cs;
    cs.level = level;
    cs.horizon = horizon;
    cs.s0 = state;
    if (goal) {
        cs.goal = goal->states;
        cs.n_goal = goal->n_states;
        cs.goal_level = goal->level > 0 ? goal->level : level;
        cs.dense = goal->dense != 0;
        cs.goal_action = goal->actions;
    }
    if (lp) {
        cs.cost_last_n = lp->cost_last_n;
        cs.intermediate_cost_weight = lp->intermediate_cost_weight;
        cs.action_cost_weight = lp->action_cost_weight;
    }
    return cs;
}

static noise_source make_noise(jepa_randn_fn randn, void * user, uint64_t seed) {
    noise_source ns;
    ns.fn = randn;
    ns.user = user;
    ns.rng.seed(seed);
    return ns;
}

// stable_worldmodel.solver.decreasing_horizon_schedule
static std::vector<int> decreasing_schedule(int num_calls, int horizon) {
    if (horizon == 1) return std::vector<int>((size_t) num_calls, 1);
    if (num_calls == 1) return { horizon };
    std::vector<int> s;
    if (num_calls < horizon) {
        for (int idx = 0; idx < num_calls; idx++) {
            const double v = 1.0 + (double) (horizon - 1) * (double) (num_calls - 1 - idx) / (double) (num_calls - 1);
            s.push_back(std::max(1, (int) std::ceil(v)));
        }
        return s;
    }
    const int per = num_calls / horizon, extra = num_calls % horizon;
    for (int i = 0; i < horizon; i++) {            // horizon values horizon, horizon - 1, ..., 1
        const int count = per + (i >= horizon - extra ? 1 : 0);
        s.insert(s.end(), (size_t) count, horizon - i);
    }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------------------------
extern "C" jepa_hjepa_level_plan jepa_hjepa_level_plan_default(void) {
    jepa_hjepa_level_plan p;
    p.horizon = 1;
    p.pre_decay_horizon = 0;
    p.n_samples = 100;
    p.n_steps = 90;
    p.lr = 0.1f;
    p.weight_decay = 0.01f;
    p.var_scale = 1.0f;
    p.action_noise = 0.0f;
    p.action_clip_sigma = 0.0f;
    p.min_its = 30;
    p.early_stop_patience = 5;
    p.early_stop_rel_delta = 1e-2f;
    p.cost_last_n = 1;
    p.intermediate_cost_weight = 1.0f;
    p.action_cost_weight = 0.0f;
    p.num_subgoals = 1;
    p.horizon_one_goal_upper = 0;
    return p;
}

extern "C" int jepa_hjepa_plan_cost(jepa_context * ctx, int level, const float * state, const jepa_hjepa_goal * goal,
                                    const jepa_hjepa_level_plan * plan, const float * actions, int n_samples,
                                    int horizon, float * costs, float * grad) {
    if (!ctx || !state || !goal || !actions || !costs || n_samples < 1) return -1;
    std::shared_ptr<plan_model> pm = plan_model_of(ctx, "jepa_hjepa_plan_cost");
    if (!pm) return -1;
    const cost_spec cs = spec_from(level, horizon, state, goal, plan);
    if (!check_spec(*pm, cs, "jepa_hjepa_plan_cost")) return -1;
    batch_cost(*pm, cs, actions, n_samples, costs, grad, ctx->n_threads);
    return 0;
}

extern "C" int jepa_hjepa_solve(jepa_context * ctx, int level, const float * state, const jepa_hjepa_goal * goal,
                                const jepa_hjepa_level_plan * plan, jepa_randn_fn randn, void * randn_user,
                                float * actions, float * states, float * cost, int * iters) {
    if (!ctx || !state || !goal || !plan || !actions) return -1;
    std::shared_ptr<plan_model> pm = plan_model_of(ctx, "jepa_hjepa_solve");
    if (!pm) return -1;
    const cost_spec cs = spec_from(level, plan->horizon, state, goal, plan);
    if (!check_spec(*pm, cs, "jepa_hjepa_solve")) return -1;
    if (plan->n_samples < 1 || plan->n_steps < 0) {
        jepa_log("jepa: jepa_hjepa_solve: n_samples %d and n_steps %d have to be >= 1 and >= 0\n", plan->n_samples, plan->n_steps);
        return -1;
    }
    noise_source ns = make_noise(randn, randn_user, 0x4a455041ull + (uint64_t) level);
    const solve_out r = solve_level(*pm, cs, *plan, ns, ctx->n_threads);
    memcpy(actions, r.actions.data(), r.actions.size() * sizeof(float));
    if (states) memcpy(states, r.states.data(), r.states.size() * sizeof(float));
    if (cost) *cost = r.cost;
    if (iters) *iters = r.iters;
    return 0;
}

extern "C" int jepa_hjepa_horizons(const jepa_model * model, const jepa_hjepa_level_plan * levels, int n_levels,
                                   int replanning_interval, int eval_budget, int steps_taken, int * horizons) {
    if (!model || !levels || !horizons || !model->hp.hjepa.present) return -1;
    if (n_levels != (int) model->hp.hjepa.levels.size() || replanning_interval < 1 || eval_budget < 1 || steps_taken < 0) {
        jepa_log("jepa: jepa_hjepa_horizons: need one plan per level (%zu), a positive replanning interval and budget, "
                 "and steps_taken >= 0\n", model->hp.hjepa.levels.size());
        return -1;
    }
    const int calls = (eval_budget + replanning_interval - 1) / replanning_interval;
    const int call = steps_taken / replanning_interval;
    if (call >= calls) {
        jepa_log("jepa: jepa_hjepa_horizons: step %d is past the evaluation budget of %d\n", steps_taken, eval_budget);
        return -1;
    }
    std::vector<int> above;
    for (int lv = n_levels; lv >= 1; lv--) {
        const jepa_hjepa_level_plan & lp = levels[lv - 1];
        if (lp.horizon < 1) {
            jepa_log("jepa: jepa_hjepa_horizons: level %d has horizon %d\n", lv, lp.horizon);
            return -1;
        }
        std::vector<int> s;
        if (lv == n_levels) {
            s = decreasing_schedule(calls, lp.horizon);
        } else {
            // hold the pre-decay horizon until the level above is down to one step, then decay
            const int pre = lp.pre_decay_horizon > 0 ? lp.pre_decay_horizon : model->hp.hjepa.levels[(size_t) lv].stride;
            const auto one = std::find(above.begin(), above.end(), 1);
            if (one == above.end()) {
                s.assign((size_t) calls, pre);
            } else {
                const int tail_start = (int) (one - above.begin());
                s.assign((size_t) tail_start, pre);
                const std::vector<int> tail = decreasing_schedule(calls - tail_start, lp.horizon);
                s.insert(s.end(), tail.begin(), tail.end());
            }
        }
        horizons[lv - 1] = s[(size_t) call];
        above = std::move(s);
    }
    return 0;
}

// HierarchicalSolver.solve: top-down, each planned level's first prediction(s) the goal of the next.
extern "C" int jepa_hjepa_plan(jepa_context * ctx, const float * const * states, const float * const * goals,
                               const jepa_hjepa_level_plan * levels, int n_levels, jepa_randn_fn randn,
                               void * randn_user, float * actions, jepa_hjepa_plan_info * info) {
    if (!ctx || !states || !goals || !levels || !actions) return -1;
    std::shared_ptr<plan_model> pm = plan_model_of(ctx, "jepa_hjepa_plan");
    if (!pm) return -1;
    if (n_levels != (int) pm->levels.size()) {
        jepa_log("jepa: jepa_hjepa_plan: %d level plans for a model with %zu levels\n", n_levels, pm->levels.size());
        return -1;
    }
    for (int lv = 1; lv <= n_levels; lv++) {
        if (!states[lv - 1] || !goals[lv - 1]) {
            jepa_log("jepa: jepa_hjepa_plan: the current and goal state of level %d are missing\n", lv);
            return -1;
        }
        if (levels[lv - 1].n_samples < 1 || levels[lv - 1].n_steps < 0 || levels[lv - 1].num_subgoals < 1) {
            jepa_log("jepa: jepa_hjepa_plan: level %d: n_samples, n_steps and num_subgoals have to be >= 1, >= 0, >= 1\n", lv);
            return -1;
        }
    }
    if (info) *info = jepa_hjepa_plan_info{};

    solve_out upper;                 // the plan of the level above, when it planned
    bool have_upper = false;
    int skipped_goal_level = 0;      // highest skipped level whose space the next planned level is scored in
    bool upper_anchored = true;      // the level above aimed at the true goal with its last step
    std::vector<float> goal_buf, goal_act;
    for (int lv = n_levels; lv >= 1; lv--) {
        const jepa_hjepa_level_plan & lp = levels[lv - 1];
        if (lv > 1 && lp.horizon == 1) {
            // a one-step plan up here gives the level below no waypoint: skip it and pass the real goal down
            skipped_goal_level = lp.horizon_one_goal_upper ? (skipped_goal_level ? skipped_goal_level : lv) : 0;
            have_upper = false;
            continue;
        }
        cost_spec cs;
        cs.level = lv;
        cs.horizon = lp.horizon;
        cs.s0 = states[lv - 1];
        cs.cost_last_n = lp.cost_last_n;
        cs.intermediate_cost_weight = lp.intermediate_cost_weight;
        cs.action_cost_weight = lp.action_cost_weight;
        bool anchored;
        if (!have_upper) {
            cs.goal_level = skipped_goal_level ? skipped_goal_level : lv;
            cs.goal = goals[cs.goal_level - 1];
            cs.n_goal = 1;
            cs.dense = cs.goal_level > lv;
            anchored = true;
        } else {
            const level_w & up = pm->levels[(size_t) lv];
            const int Du = up.hp.embed_dim, Au = up.hp.action_dim;
            const int n_up = (int) upper.states.size() / Du;     // the upper level's states, its current one first
            const int n_sub = std::min(levels[lv].num_subgoals, n_up - 1);
            goal_buf.assign(upper.states.begin() + Du, upper.states.begin() + (std::ptrdiff_t) (1 + n_sub) * Du);
            goal_act.assign(upper.actions.begin(), upper.actions.begin() + (std::ptrdiff_t) n_sub * Au);
            anchored = false;
            if (1 + levels[lv].num_subgoals >= n_up && upper_anchored) {
                // the upper plan's last state should have reached its goal: aim at the goal itself
                memcpy(goal_buf.data() + (size_t) (n_sub - 1) * Du, goals[lv], (size_t) Du * sizeof(float));
                anchored = true;
            }
            cs.goal_level = lv + 1;
            cs.goal = goal_buf.data();
            cs.n_goal = n_sub;
            cs.goal_action = goal_act.data();
            if (n_sub > 1) {
                const int expected = (lp.horizon + up.hp.stride) / up.hp.stride - 1;
                if (n_sub != expected) {
                    jepa_log("jepa: jepa_hjepa_plan: level %d: %d subgoals from above, but horizon %d at stride %d "
                             "reaches %d states of level %d\n", lv, n_sub, lp.horizon, up.hp.stride, expected, lv + 1);
                    return -1;
                }
            }
        }
        if (!check_spec(*pm, cs, "jepa_hjepa_plan")) return -1;
        noise_source ns = make_noise(randn, randn_user, 0x4a455041ull + (uint64_t) lv);
        upper = solve_level(*pm, cs, lp, ns, ctx->n_threads);
        if (info && lv <= JEPA_HJEPA_MAX_PLAN_LEVELS) {
            info->planned[lv - 1] = 1;
            info->cost[lv - 1] = upper.cost;
            info->iters[lv - 1] = upper.iters;
            info->goal_level[lv - 1] = cs.goal_level;
        }
        upper_anchored = anchored;
        skipped_goal_level = 0;
        have_upper = true;
    }
    // level 1 always plans, so `upper` now holds its result
    memcpy(actions, upper.actions.data(), upper.actions.size() * sizeof(float));
    return 0;
}
