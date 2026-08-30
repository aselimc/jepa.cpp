// jepa-info: print the hparams and the tensor table of a jepa.cpp GGUF.
//   jepa-info model.gguf [--no-tensors] [--kv]
#include "jepa-internal.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s model.gguf [--no-tensors] [--kv]\n"
                    "  --no-tensors   skip the tensor table\n"
                    "  --kv           also dump every general.* / jepa.* key verbatim\n", argv0);
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const char * path = nullptr;
    bool tensors = true, kv = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-tensors") == 0) tensors = false;
        else if (strcmp(argv[i], "--kv") == 0) kv = true;
        else if (argv[i][0] == '-') { usage(argv[0]); return 1; }
        else path = argv[i];
    }
    if (!path) { usage(argv[0]); return 1; }

    jepa_model * m = jepa_model_load(path, false);
    if (!m) return 1;
    const jepa_hparams & hp = m->hp;
    const jepa_enc_hparams & e = hp.enc;

    printf("file:        %s\n", path);
    printf("name:        %s\n", hp.name.c_str());
    printf("family:      %s (modality %s, schema v%u)\n", hp.family_str.c_str(), hp.modality.c_str(), hp.schema_version);
    printf("license:     %s\n", hp.license.c_str());
    printf("source:      %s\n", hp.source_url.c_str());
    printf("file_type:   %u (%s)\n", hp.file_type, jepa_file_type_name(hp.file_type));
    printf("weights:     %.1f MiB in %zu tensors\n", m->n_bytes_weights / (1024.0 * 1024.0), m->tensors.size());
    printf("encoder:     D=%d layers=%d heads=%d (head_dim %d) ffn=%d patch=%d tubelet=%d img=%d frames=%d chans=%d\n",
           e.embed_dim, e.n_layer, e.n_head, e.head_dim(), e.ffn_dim, e.patch_size, e.tubelet_size, e.img_size, e.n_frames, e.in_chans);
    printf("             ln_eps=%g act=%s pos=%s cls=%s registers=%d qkv_fused=%s layer_scale=%s\n",
           e.ln_eps, jepa_act_name(e.act), e.pos_type_str.c_str(), e.cls_token ? "yes" : "no", e.n_registers,
           e.qkv_fused ? "yes" : "no", e.layer_scale ? "yes" : "no");
    if (e.pos_type == JEPA_POS_ROPE3D) {
        printf("             rope: theta=%g layout=%s interpolate=%s ref_grid=%d\n", e.rope_theta, e.rope_freq_layout.c_str(),
               e.rope_interpolate ? "yes" : "no", e.rope_ref_grid);
    }
    if (e.modality_embed || e.image_patch_embed || !e.hier_layers.empty()) {
        printf("             modality_embed=%s image_patch_embed=%s hier_layers=[", e.modality_embed ? "yes" : "no", e.image_patch_embed ? "yes" : "no");
        for (size_t i = 0; i < e.hier_layers.size(); i++) printf("%s%d", i ? "," : "", e.hier_layers[i]);
        printf("]\n");
    }
    if (m->pos_embed) printf("             pos table: %" PRId64 " tokens x %" PRId64 "\n", m->pos_embed->ne[1], m->pos_embed->ne[0]);
    if (m->proj0_w) printf("             projector: enc.proj.0 [%" PRId64 "->%" PRId64 "] %s enc.proj.2 [%" PRId64 "->%" PRId64 "]\n",
                           m->proj0_w->ne[0], m->proj0_w->ne[1], jepa_act_name(e.proj_act), m->proj2_w->ne[0], m->proj2_w->ne[1]);
    if (hp.pred.present) {
        const jepa_pred_hparams & p = hp.pred;
        printf("predictor:   kind=%s D=%d layers=%d heads=%d (head_dim %d) ffn=%d out_dim=%d mask_tokens=%d\n",
               p.kind.c_str(), p.embed_dim, p.n_layer, p.n_head, p.head_dim_eff(), p.ffn_dim, p.out_dim, p.n_mask_tokens);
        printf("             ln_eps=%g adaln_eps=%g act=%s qkv_bias=%s action_dim=%d state_dim=%d n_frames=%d causal=%s\n",
               p.ln_eps, p.adaln_eps, jepa_act_name(p.act), p.qkv_bias ? "yes" : "no", p.action_dim, p.state_dim, p.n_frames,
               p.frame_causal ? "yes" : "no");
        if (!p.rope_freq_layout.empty()) {
            printf("             rope layout=%s interpolate=%s ref_grid=%d grid_size=%d n_hier_in=%d\n", p.rope_freq_layout.c_str(),
                   p.rope_interpolate ? "yes" : "no", p.rope_ref_grid, p.grid_size, p.n_hier_in);
        }
    } else {
        printf("predictor:   none\n");
    }
    if (hp.head.present) {
        printf("head:        kind=%s classes=%d pool_layers=%d labels=%zu\n", hp.head.kind.c_str(), hp.head.n_classes,
               hp.head.n_pool_layers, hp.head.labels.size());
    } else {
        printf("head:        none\n");
    }
    printf("preprocess:  mode=%s resize_short=%d crop=%d resample=%s rescale=%g mean=(%g,%g,%g) std=(%g,%g,%g)\n",
           hp.pre.resize_mode.c_str(), hp.pre.resize_short, hp.pre.crop, hp.pre.resample.c_str(), hp.pre.rescale,
           hp.pre.mean[0], hp.pre.mean[1], hp.pre.mean[2], hp.pre.std[0], hp.pre.std[1], hp.pre.std[2]);

    if (kv) {
        printf("\nmetadata (%zu keys):\n", hp.raw_kv.size());
        for (const auto & p : hp.raw_kv) printf("  %-36s %s\n", p.first.c_str(), p.second.c_str());
    }

    if (tensors) {
        printf("\n%-40s %-6s %-28s %12s\n", "tensor", "type", "shape (ggml ne)", "bytes");
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(m->ctx_w); t; t = ggml_get_next_tensor(m->ctx_w, t)) {
            std::string shape = "[";
            for (int d = 0; d < ggml_n_dims(t); d++) shape += (d ? ", " : "") + std::to_string(t->ne[d]);
            shape += "]";
            printf("%-40s %-6s %-28s %12zu\n", ggml_get_name(t), ggml_type_name(t->type), shape.c_str(), ggml_nbytes(t));
            total += ggml_nbytes(t);
        }
        printf("%-40s %-6s %-28s %12zu\n", "total", "", "", total);
    }
    jepa_model_free(m);
    return 0;
}
