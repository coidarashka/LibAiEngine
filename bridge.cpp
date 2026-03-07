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

using json = nlohmann::json;

#define TAG "MandreAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct GlobalConfig {
    int n_threads = 4;
    int n_threads_batch = 8;
    int n_ctx = 0;           // 0 = авто
    int n_batch = 512;
    int max_tokens = 2048;      // Общий лимит ответа
    int max_think_tokens = 512;  // Лимит только для <think>
    bool kv_quant = true;
    bool flash_attn = true;
};

static GlobalConfig g_conf;
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static mtmd_context* g_mtmd_ctx = nullptr;
static bool g_cancel_flag = false;

extern "C" {
    // Освобождение ресурсов (предотвращает краш при повторной загрузке)
    void free_engine() {
        if (g_mtmd_ctx) { mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
        if (g_sampler)  { llama_sampler_free(g_sampler); g_sampler = nullptr; }
        if (g_ctx)      { llama_free(g_ctx); g_ctx = nullptr; }
        if (g_model)    { llama_model_free(g_model); g_model = nullptr; }
        llama_backend_free();
        LOGD("Engine Resources Cleared");

    int load_mmproj(const char* p) {
        if (!g_model) return -2;
        LOGD("Vision adapter load requested: %s", p ? p : "none");
        
        // Если путь пустой или "none", ничего не делаем
        if (!p || strlen(p) == 0 || strcmp(p, "none") == 0) return 0;

        if (g_mtmd_ctx) mtmd_free(g_mtmd_ctx);
        
        mtmd_context_params params = mtmd_context_params_default();
        params.use_gpu = false; // На андроиде лучше CPU для стабильности
        params.image_max_tokens = g_conf.max_think_tokens; // Используем лимит из конфига
        
        g_mtmd_ctx = mtmd_init_from_file(p, g_model, params);
        return g_mtmd_ctx ? 0 : -1;
    },
    }

    int configure_engine(const char* json_str) {
        try {
            auto j = json::parse(json_str);
            if (j.contains("n_threads"))        g_conf.n_threads = j["n_threads"];
            if (j.contains("n_threads_batch"))  g_conf.n_threads_batch = j["n_threads_batch"];
            if (j.contains("n_ctx"))            g_conf.n_ctx = j["n_ctx"];
            if (j.contains("max_tokens"))       g_conf.max_tokens = j["max_tokens"];
            if (j.contains("max_think_tokens")) g_conf.max_think_tokens = j["max_think_tokens"];
            if (j.contains("kv_quant"))         g_conf.kv_quant = j["kv_quant"];
            if (j.contains("flash_attn"))       g_conf.flash_attn = j["flash_attn"];
            return 0;
        } catch (...) { return -1; }
    }

    void cancel_inference() { g_cancel_flag = true; }

    int load_model(const char* p) {
        free_engine(); // Очищаем всё перед загрузкой новой модели
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.use_mmap = true;
        g_model = llama_model_load_from_file(p, mp);
        if (!g_model) return -1;
        g_vocab = llama_model_get_vocab(g_model);

        // Адаптация контекста под модель
        int model_train_ctx = llama_model_n_ctx_train(g_model);
        int final_ctx = (g_conf.n_ctx > 0) ? g_conf.n_ctx : std::min(model_train_ctx, 4096);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = final_ctx;
        cp.n_threads       = g_conf.n_threads;
        cp.n_threads_batch = g_conf.n_threads_batch;
        cp.n_batch         = g_conf.n_batch;
        
        // Исправлено: используем enum flash_attn_type
        cp.flash_attn_type = g_conf.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        if (g_conf.kv_quant) {
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }

        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) return -2;

        auto sp = llama_sampler_chain_default_params();
        g_sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(NULL)));

        LOGD("Model Loaded. Ctx: %d, FlashAttn: %d", final_ctx, g_conf.flash_attn);
        return 0;
    }

    typedef void (*cb_t)(const char*);
    int infer(const char* pr, const char* img, cb_t cb) {
        if (!g_ctx || !g_sampler) return -1;
        g_cancel_flag = false;

        // Сброс состояния для новой генерации (защита от вылетов на 2-й раз)
        llama_sampler_reset(g_sampler);
        llama_memory_seq_rm(llama_get_memory(g_ctx), -1, -1, -1);

        // Простая токенизация промпта
        std::vector<llama_token> tk(strlen(pr) + 16);
        int n = llama_tokenize(g_vocab, pr, strlen(pr), tk.data(), tk.size(), true, true);
        tk.resize(n);
        llama_decode(g_ctx, llama_batch_get_one(tk.data(), n));

        int think_tokens = 0;
        int normal_tokens = 0;
        bool in_think_tag = false;

        for (int i = 0; i < g_conf.max_tokens; i++) {
            if (g_cancel_flag) break;

            llama_token id = llama_sampler_sample(g_sampler, g_ctx, -1);
            if (llama_vocab_is_eog(g_vocab, id)) break;

            char b[256];
            int n_p = llama_token_to_piece(g_vocab, id, b, sizeof(b), 0, true);
            if (n_p > 0) {
                std::string piece(b, n_p);

                // Логика Think-токенов
                if (piece.find("<think>") != std::string::npos) in_think_tag = true;
                
                if (in_think_tag) {
                    think_tokens++;
                    if (think_tokens > g_conf.max_think_tokens) {
                        cb("\n[Think Limit Exceeded]\n");
                        // Принудительно закрываем тег или выходим
                        in_think_tag = false; 
                        break; 
                    }
                } else {
                    normal_tokens++;
                }

                cb(piece.c_str());
                if (piece.find("</think>") != std::string::npos) in_think_tag = false;
            }

            llama_sampler_accept(g_sampler, id);
            if (llama_decode(g_ctx, llama_batch_get_one(&id, 1)) != 0) break;
        }
        return 0;
    }
}
