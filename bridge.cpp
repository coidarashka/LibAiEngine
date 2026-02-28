#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>
#include <ctime>
#include <csignal>
#include <unistd.h>

using json = nlohmann::json;

#define TAG "MandreAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct GlobalConfig {
    int n_threads = 4;
    int n_threads_batch = 4;
    int n_ctx = 2048;
    int n_batch = 512;
    int img_max_tokens = 128;
    bool kv_quant = true;
};

static GlobalConfig g_conf;
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static mtmd_context* g_mtmd_ctx = nullptr;
static bool g_cancel_flag = false;

void signal_handler(int signum) {
    const char* sig_name = "UNKNOWN";
    switch(signum) {
        case SIGSEGV: sig_name = "SIGSEGV"; break;
        case SIGABRT: sig_name = "SIGABRT"; break;
        case SIGILL:  sig_name = "SIGILL";  break;
        case SIGFPE:  sig_name = "SIGFPE";  break;
    }
    LOGE("CRITICAL ENGINE CRASH: %s", sig_name);
    signal(signum, SIG_DFL);
    raise(signum);
}

extern "C" {
    int configure_engine(const char* json_str) {
        try {
            auto j = json::parse(json_str);
            if (j.contains("n_threads"))      g_conf.n_threads      = j["n_threads"];
            if (j.contains("n_ctx"))          g_conf.n_ctx          = j["n_ctx"];
            if (j.contains("n_batch"))        g_conf.n_batch        = j["n_batch"];
            if (j.contains("img_max_tokens")) g_conf.img_max_tokens = j["img_max_tokens"];
            LOGD("Config Applied: thr=%d, ctx=%d", g_conf.n_threads, g_conf.n_ctx);
            return 0;
        } catch (...) { LOGE("Config Parse Error!"); return -1; }
    }

    void register_crash_handlers() {
        signal(SIGSEGV, signal_handler); signal(SIGABRT, signal_handler);
        signal(SIGILL,  signal_handler); signal(SIGFPE,  signal_handler);
    }

    void set_inference_config(int sz) {}
    void cancel_inference() { g_cancel_flag = true; }

    int load_mmproj(const char* p) {
        if (!g_model) return -2;
        if (g_mtmd_ctx) mtmd_free(g_mtmd_ctx);
        mtmd_context_params params = mtmd_context_params_default();
        params.use_gpu = false;
        params.image_min_tokens = 32;
        params.image_max_tokens = g_conf.img_max_tokens;
        g_mtmd_ctx = mtmd_init_from_file(p, g_model, params);
        return g_mtmd_ctx ? 0 : -1;
    }

    int load_model(const char* p) {
        register_crash_handlers();
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.use_mmap = true;
        g_model = llama_model_load_from_file(p, mp);
        if (!g_model) return -1;
        g_vocab = llama_model_get_vocab(g_model);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = g_conf.n_ctx;
        cp.n_threads       = g_conf.n_threads;
        cp.n_threads_batch = g_conf.n_threads;
        cp.n_batch         = g_conf.n_batch;
        cp.n_ubatch        = g_conf.n_batch / 2;
        if (g_conf.kv_quant) {
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }

        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) return -2;

        auto sp = llama_sampler_chain_default_params();
        g_sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, 1.45f, 0.4f, 0.4f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(NULL)));
        LOGD("Engine Loaded. Kleidi: ON");
        return 0;
    }

    typedef void (*cb_t)(const char*);
    int infer(const char* pr, const char* img, cb_t cb) {
        if (!g_ctx || !g_sampler) return -1;
        g_cancel_flag = false;
        llama_memory_seq_rm(llama_get_memory(g_ctx), -1, -1, -1);

        if (img && g_mtmd_ctx && strlen(img) > 0) {
            long t0 = time(NULL);
            mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(g_mtmd_ctx, img);
            if (bmp) {
                std::string fp = std::string(mtmd_default_marker()) + "\n" + pr;
                mtmd_input_text it = {fp.c_str(), true, true};
                mtmd_input_chunks* ch = mtmd_input_chunks_init();
                const mtmd_bitmap* bmps[] = {bmp};
                if (mtmd_tokenize(g_mtmd_ctx, ch, &it, bmps, 1) == 0) {
                    llama_pos np = 0;
                    mtmd_helper_eval_chunks(g_mtmd_ctx, g_ctx, ch, 0, 0, g_conf.n_threads, true, &np);
                }
                LOGD("Vision Total: %ld seconds", time(NULL) - t0);
                mtmd_input_chunks_free(ch);
                mtmd_bitmap_free(bmp);
            }
        } else {
            std::vector<llama_token> tk(strlen(pr) + 16);
            int n = llama_tokenize(g_vocab, pr, strlen(pr), tk.data(), tk.size(), true, true);
            tk.resize(n);
            llama_decode(g_ctx, llama_batch_get_one(tk.data(), n));
        }

        for (int i = 0; i < 4096; i++) {
            if (g_cancel_flag) break;
            llama_token id = llama_sampler_sample(g_sampler, g_ctx, -1);
            if (llama_vocab_is_eog(g_vocab, id)) break;
            char b[256];
            int n = llama_token_to_piece(g_vocab, id, b, sizeof(b), 0, true);
            if (n > 0) cb(std::string(b, n).c_str());
            llama_sampler_accept(g_sampler, id);
            llama_batch batch = llama_batch_get_one(&id, 1);
            if (llama_decode(g_ctx, batch) != 0) break;
        }
        return 0;
    }

    void free_engine() {
        if (g_mtmd_ctx) mtmd_free(g_mtmd_ctx);
        if (g_sampler)  llama_sampler_free(g_sampler);
        if (g_model)    llama_model_free(g_model);
        if (g_ctx)      llama_free(g_ctx);
        llama_backend_free();
    }
}
