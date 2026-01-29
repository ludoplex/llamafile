// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "recursive_llm.h"

#include <cosmo.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#include "llama.cpp/llama.h"
#include "llama.cpp/common/sampling.h"

//
// Default configurations
//

rllm_env_config_t rllm_default_env_config(void) {
    rllm_env_config_t config = {};
    config.max_depth = RLLM_MAX_DEPTH;
    config.max_contexts = RLLM_MAX_CONTEXTS;
    config.default_n_ctx = 2048;
    config.default_n_batch = 512;
    config.default_n_threads = 4;
    config.memory_limit = 0;  // No limit
    config.enable_logging = false;
    config.enable_metrics = true;
    return config;
}

rllm_ctx_config_t rllm_default_ctx_config(void) {
    rllm_ctx_config_t config = {};
    config.n_ctx = 2048;
    config.n_batch = 512;
    config.n_threads = 4;
    config.share_mode = RLLM_SHARE_NONE;
    config.completion = rllm_default_completion_params();
    config.inherit_prompt = false;
    config.inherit_sampling = false;
    return config;
}

rllm_completion_params_t rllm_default_completion_params(void) {
    rllm_completion_params_t params = {};
    params.n_predict = 256;
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.top_k = 40;
    params.repeat_penalty = 1.1f;
    params.stream = false;
    params.timeout_ms = 0;
    return params;
}

//
// Internal helpers
//

static uint64_t rllm_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void rllm_log(rllm_env_t *env, const char *fmt, ...) {
    if (!env || !env->config.enable_logging) return;

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[RLLM] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

static rllm_context_t *rllm_alloc_context(rllm_env_t *env) {
    if (env->n_contexts >= env->config.max_contexts) {
        return NULL;
    }

    rllm_context_t *ctx = (rllm_context_t *)calloc(1, sizeof(rllm_context_t));
    if (!ctx) return NULL;

    ctx->id = env->next_ctx_id++;
    ctx->state = RLLM_STATE_IDLE;
    ctx->children_capacity = 8;
    ctx->children = (rllm_context_t **)calloc(ctx->children_capacity, sizeof(rllm_context_t *));
    ctx->queue_capacity = 32;
    ctx->message_queue = (rllm_message_t *)calloc(ctx->queue_capacity, sizeof(rllm_message_t));

    if (!ctx->children || !ctx->message_queue) {
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Add to context pool
    if (env->n_contexts >= env->context_capacity) {
        size_t new_cap = env->context_capacity * 2;
        if (new_cap < 16) new_cap = 16;
        rllm_context_t **new_contexts = (rllm_context_t **)realloc(env->contexts,
                                                                    new_cap * sizeof(rllm_context_t *));
        if (!new_contexts) {
            free(ctx->children);
            free(ctx->message_queue);
            free(ctx);
            return NULL;
        }
        env->contexts = new_contexts;
        env->context_capacity = new_cap;
    }

    env->contexts[env->n_contexts++] = ctx;
    env->total_contexts_created++;

    return ctx;
}

static void rllm_free_context_internal(rllm_env_t *env, rllm_context_t *ctx) {
    if (!ctx) return;

    // Free children first (recursively)
    for (size_t i = 0; i < ctx->n_children; i++) {
        rllm_free_context_internal(env, ctx->children[i]);
    }

    // Free message queue data
    for (size_t i = 0; i < ctx->queue_capacity; i++) {
        free(ctx->message_queue[i].data);
    }

    // Free token editor
    if (ctx->token_editor) {
        te_free(ctx->token_editor);
    }

    // Free llama context (if we own it)
    if (ctx->llama_ctx) {
        llama_free(ctx->llama_ctx);
    }

    free(ctx->children);
    free(ctx->message_queue);
    free(ctx);
}

static void rllm_remove_from_pool(rllm_env_t *env, rllm_context_t *ctx) {
    for (size_t i = 0; i < env->n_contexts; i++) {
        if (env->contexts[i] == ctx) {
            // Shift remaining
            for (size_t j = i; j < env->n_contexts - 1; j++) {
                env->contexts[j] = env->contexts[j + 1];
            }
            env->n_contexts--;
            break;
        }
    }
}

static void rllm_add_child(rllm_context_t *parent, rllm_context_t *child) {
    if (!parent || !child) return;

    if (parent->n_children >= parent->children_capacity) {
        size_t new_cap = parent->children_capacity * 2;
        rllm_context_t **new_children = (rllm_context_t **)realloc(parent->children,
                                                                    new_cap * sizeof(rllm_context_t *));
        if (!new_children) return;
        parent->children = new_children;
        parent->children_capacity = new_cap;
    }

    parent->children[parent->n_children++] = child;
    child->parent = parent;
    child->depth = parent->depth + 1;
}

static void rllm_remove_child(rllm_context_t *parent, rllm_context_t *child) {
    if (!parent || !child) return;

    for (size_t i = 0; i < parent->n_children; i++) {
        if (parent->children[i] == child) {
            for (size_t j = i; j < parent->n_children - 1; j++) {
                parent->children[j] = parent->children[j + 1];
            }
            parent->n_children--;
            child->parent = NULL;
            break;
        }
    }
}

//
// Environment lifecycle
//

rllm_env_t *rllm_init(struct llama_model *model, rllm_env_config_t config) {
    if (!model) return NULL;

    rllm_env_t *env = (rllm_context_t **)calloc(1, sizeof(rllm_env_t));
    if (!env) return NULL;

    env->model = model;
    env->config = config;
    env->context_capacity = 16;
    env->contexts = (rllm_context_t **)calloc(env->context_capacity, sizeof(rllm_context_t *));

    size_t n_roots_cap = 8;
    env->roots = (rllm_context_t **)calloc(n_roots_cap, sizeof(rllm_context_t *));

    if (!env->contexts || !env->roots) {
        free(env->contexts);
        free(env->roots);
        free(env);
        return NULL;
    }

    // Create mutex for thread safety
    env->mutex = calloc(1, sizeof(pthread_mutex_t));
    if (env->mutex) {
        pthread_mutex_init((pthread_mutex_t *)env->mutex, NULL);
    }

    rllm_log(env, "Initialized recursive LLM environment");

    return env;
}

void rllm_shutdown(rllm_env_t *env) {
    if (!env) return;

    rllm_log(env, "Shutting down environment with %zu contexts", env->n_contexts);

    // Free all contexts
    for (size_t i = 0; i < env->n_roots; i++) {
        rllm_free_context_internal(env, env->roots[i]);
    }

    if (env->mutex) {
        pthread_mutex_destroy((pthread_mutex_t *)env->mutex);
        free(env->mutex);
    }

    free(env->contexts);
    free(env->roots);
    free(env);
}

void rllm_get_stats(rllm_env_t *env, uint64_t *total_tokens,
                    uint64_t *total_contexts, uint64_t *peak_depth) {
    if (!env) return;

    if (total_tokens) *total_tokens = env->total_tokens_processed;
    if (total_contexts) *total_contexts = env->total_contexts_created;
    if (peak_depth) *peak_depth = env->peak_depth;
}

//
// Context creation
//

rllm_context_t *rllm_create_root(rllm_env_t *env, rllm_ctx_config_t config) {
    if (!env) return NULL;

    rllm_context_t *ctx = rllm_alloc_context(env);
    if (!ctx) return NULL;

    ctx->relation = RLLM_REL_ROOT;
    ctx->config = config;
    ctx->depth = 0;

    // Create llama context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;

    ctx->llama_ctx = llama_new_context_with_model(env->model, cparams);
    if (!ctx->llama_ctx) {
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Create token editor
    ctx->token_editor = te_init(ctx->llama_ctx, env->model);
    if (!ctx->token_editor) {
        llama_free(ctx->llama_ctx);
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Add to roots
    size_t n_roots_cap = 8;  // Would need to track capacity properly
    if (env->n_roots < n_roots_cap) {
        env->roots[env->n_roots++] = ctx;
    }

    if (env->on_context_create) {
        env->on_context_create(env, ctx);
    }

    rllm_log(env, "Created root context %u", ctx->id);

    return ctx;
}

rllm_context_t *rllm_spawn_child(rllm_env_t *env, rllm_context_t *parent,
                                  rllm_ctx_config_t config) {
    if (!env || !parent) return NULL;

    // Check depth limit
    if (parent->depth + 1 >= env->config.max_depth) {
        rllm_log(env, "Max depth %u reached", env->config.max_depth);
        return NULL;
    }

    rllm_context_t *ctx = rllm_alloc_context(env);
    if (!ctx) return NULL;

    ctx->relation = RLLM_REL_CHILD;
    ctx->config = config;

    // Create llama context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx > 0 ? config.n_ctx : parent->config.n_ctx;
    cparams.n_batch = config.n_batch > 0 ? config.n_batch : parent->config.n_batch;
    cparams.n_threads = config.n_threads > 0 ? config.n_threads : parent->config.n_threads;
    cparams.n_threads_batch = cparams.n_threads;

    ctx->llama_ctx = llama_new_context_with_model(env->model, cparams);
    if (!ctx->llama_ctx) {
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Create token editor
    ctx->token_editor = te_init(ctx->llama_ctx, env->model);
    if (!ctx->token_editor) {
        llama_free(ctx->llama_ctx);
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Handle sharing mode
    if (config.share_mode == RLLM_SHARE_KV_COPY || config.share_mode == RLLM_SHARE_FULL) {
        // Copy KV cache state from parent
        size_t state_size = llama_get_state_size(parent->llama_ctx);
        uint8_t *state = (uint8_t *)malloc(state_size);
        if (state) {
            llama_copy_state_data(parent->llama_ctx, state);
            llama_set_state_data(ctx->llama_ctx, state);
            free(state);
        }
    }

    if (config.share_mode == RLLM_SHARE_TOKENS_COPY || config.share_mode == RLLM_SHARE_FULL) {
        // Copy tokens from parent
        te_context_t *parent_te = parent->token_editor;
        te_context_t *child_te = ctx->token_editor;

        if (parent_te && child_te && parent_te->n_tokens > 0) {
            te_range_t range = {0, (te_pos_t)parent_te->n_tokens, -1};
            te_token_t *tokens = (te_token_t *)malloc(parent_te->n_tokens * sizeof(te_token_t));
            if (tokens) {
                size_t n = parent_te->n_tokens;
                te_get_tokens(parent_te, range, tokens, &n);
                te_insert_tokens(child_te, 0, 0, tokens, n);
                free(tokens);
            }
        }
    }

    rllm_add_child(parent, ctx);

    // Update peak depth
    if (ctx->depth > env->peak_depth) {
        env->peak_depth = ctx->depth;
    }

    env->total_recursions++;

    if (env->on_context_create) {
        env->on_context_create(env, ctx);
    }

    if (env->on_recursion) {
        env->on_recursion(env, parent, ctx);
    }

    rllm_log(env, "Spawned child context %u from parent %u (depth %u)",
             ctx->id, parent->id, ctx->depth);

    return ctx;
}

rllm_context_t *rllm_fork(rllm_env_t *env, rllm_context_t *source) {
    if (!env || !source) return NULL;

    rllm_ctx_config_t config = source->config;
    config.share_mode = RLLM_SHARE_FULL;

    rllm_context_t *ctx = rllm_spawn_child(env, source->parent, config);
    if (!ctx) return NULL;

    ctx->relation = RLLM_REL_FORK;

    rllm_log(env, "Forked context %u from %u", ctx->id, source->id);

    return ctx;
}

rllm_context_t *rllm_create_peer(rllm_env_t *env, rllm_context_t *peer,
                                  rllm_ctx_config_t config) {
    if (!env || !peer) return NULL;

    rllm_context_t *ctx = rllm_alloc_context(env);
    if (!ctx) return NULL;

    ctx->relation = RLLM_REL_PEER;
    ctx->config = config;
    ctx->depth = peer->depth;
    ctx->parent = peer->parent;

    // Create independent llama context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx > 0 ? config.n_ctx : peer->config.n_ctx;
    cparams.n_batch = config.n_batch > 0 ? config.n_batch : peer->config.n_batch;
    cparams.n_threads = config.n_threads > 0 ? config.n_threads : peer->config.n_threads;
    cparams.n_threads_batch = cparams.n_threads;

    ctx->llama_ctx = llama_new_context_with_model(env->model, cparams);
    if (!ctx->llama_ctx) {
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    ctx->token_editor = te_init(ctx->llama_ctx, env->model);
    if (!ctx->token_editor) {
        llama_free(ctx->llama_ctx);
        rllm_remove_from_pool(env, ctx);
        free(ctx->children);
        free(ctx->message_queue);
        free(ctx);
        return NULL;
    }

    // Add as sibling to peer's parent
    if (peer->parent) {
        rllm_add_child(peer->parent, ctx);
    }

    if (env->on_context_create) {
        env->on_context_create(env, ctx);
    }

    rllm_log(env, "Created peer context %u alongside %u", ctx->id, peer->id);

    return ctx;
}

rllm_error_t rllm_destroy(rllm_env_t *env, rllm_context_t *ctx) {
    if (!env || !ctx) return RLLM_ERROR_INVALID_CONTEXT;

    if (env->on_context_destroy) {
        env->on_context_destroy(env, ctx);
    }

    // Remove from parent
    if (ctx->parent) {
        rllm_remove_child(ctx->parent, ctx);
    }

    // Remove from roots if it's a root
    for (size_t i = 0; i < env->n_roots; i++) {
        if (env->roots[i] == ctx) {
            for (size_t j = i; j < env->n_roots - 1; j++) {
                env->roots[j] = env->roots[j + 1];
            }
            env->n_roots--;
            break;
        }
    }

    // Free context and all children
    rllm_remove_from_pool(env, ctx);
    rllm_free_context_internal(env, ctx);

    return RLLM_OK;
}

//
// Context lookup
//

rllm_context_t *rllm_get_context(rllm_env_t *env, rllm_ctx_id_t id) {
    if (!env) return NULL;

    for (size_t i = 0; i < env->n_contexts; i++) {
        if (env->contexts[i]->id == id) {
            return env->contexts[i];
        }
    }

    return NULL;
}

rllm_context_t *rllm_get_parent(rllm_context_t *ctx) {
    return ctx ? ctx->parent : NULL;
}

rllm_context_t **rllm_get_children(rllm_context_t *ctx, size_t *n_children) {
    if (!ctx || !n_children) return NULL;
    *n_children = ctx->n_children;
    return ctx->children;
}

rllm_context_t *rllm_get_root(rllm_context_t *ctx) {
    if (!ctx) return NULL;
    while (ctx->parent) {
        ctx = ctx->parent;
    }
    return ctx;
}

uint32_t rllm_get_depth(rllm_context_t *ctx) {
    return ctx ? ctx->depth : 0;
}

//
// Token operations
//

te_context_t *rllm_get_token_editor(rllm_context_t *ctx) {
    return ctx ? ctx->token_editor : NULL;
}

rllm_error_t rllm_set_prompt(rllm_context_t *ctx, const char *prompt, size_t len) {
    if (!ctx || !prompt) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    // Clear existing tokens
    te_clear(te, 0);

    // Tokenize and insert
    size_t max_tokens = len + 1;
    te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    if (!tokens) return RLLM_ERROR_MEMORY;

    size_t n_tokens = max_tokens;
    te_error_t err = te_tokenize(te, prompt, len, tokens, &n_tokens, true);
    if (err != TE_OK) {
        free(tokens);
        return RLLM_ERROR_MEMORY;
    }

    err = te_insert_tokens(te, 0, 0, tokens, n_tokens);
    free(tokens);

    if (err != TE_OK) {
        return RLLM_ERROR_MEMORY;
    }

    return RLLM_OK;
}

rllm_error_t rllm_append_prompt(rllm_context_t *ctx, const char *text, size_t len) {
    if (!ctx || !text) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    // Tokenize
    size_t max_tokens = len + 1;
    te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    if (!tokens) return RLLM_ERROR_MEMORY;

    size_t n_tokens = max_tokens;
    te_error_t err = te_tokenize(te, text, len, tokens, &n_tokens, false);
    if (err != TE_OK) {
        free(tokens);
        return RLLM_ERROR_MEMORY;
    }

    err = te_insert_tokens(te, te->n_tokens, 0, tokens, n_tokens);
    free(tokens);

    if (err != TE_OK) {
        return RLLM_ERROR_MEMORY;
    }

    return RLLM_OK;
}

rllm_error_t rllm_get_text(rllm_context_t *ctx, char *buf, size_t *buf_size) {
    if (!ctx || !buf || !buf_size) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    return te_detokenize(te, te->tokens, te->n_tokens, buf, buf_size) == TE_OK ?
           RLLM_OK : RLLM_ERROR_MEMORY;
}

//
// Execution
//

rllm_error_t rllm_complete(rllm_context_t *ctx, rllm_completion_params_t params) {
    if (!ctx) return RLLM_ERROR_INVALID_CONTEXT;
    if (ctx->state == RLLM_STATE_RUNNING) return RLLM_ERROR_CONTEXT_BUSY;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    ctx->state = RLLM_STATE_RUNNING;
    ctx->start_time = rllm_get_time_ms();
    ctx->tokens_generated = 0;

    // Sync KV cache
    te_sync_kv_cache(te);

    // Setup sampling
    llama_sampling_params sparams;
    sparams.temp = params.temperature;
    sparams.top_p = params.top_p;
    sparams.top_k = (int)params.top_k;
    sparams.penalty_repeat = params.repeat_penalty;

    llama_sampling_context *sampling_ctx = llama_sampling_init(sparams);
    if (!sampling_ctx) {
        ctx->state = RLLM_STATE_ERROR;
        return RLLM_ERROR_MEMORY;
    }

    // Generation loop
    for (uint32_t i = 0; i < params.n_predict; i++) {
        // Check timeout
        if (params.timeout_ms > 0) {
            uint64_t elapsed = rllm_get_time_ms() - ctx->start_time;
            if (elapsed > params.timeout_ms) {
                ctx->state = RLLM_STATE_ERROR;
                llama_sampling_free(sampling_ctx);
                return RLLM_ERROR_TIMEOUT;
            }
        }

        // Get logits and sample
        float *logits = llama_get_logits(ctx->llama_ctx);
        int32_t vocab_size = llama_n_vocab(te->llama_model);

        llama_token_data_array candidates = {};
        candidates.size = vocab_size;
        candidates.data = (llama_token_data *)malloc(vocab_size * sizeof(llama_token_data));
        if (!candidates.data) {
            ctx->state = RLLM_STATE_ERROR;
            llama_sampling_free(sampling_ctx);
            return RLLM_ERROR_MEMORY;
        }

        for (int32_t j = 0; j < vocab_size; j++) {
            candidates.data[j].id = j;
            candidates.data[j].logit = logits[j];
            candidates.data[j].p = 0.0f;
        }

        // Sample next token
        llama_token next_token = llama_sampling_sample(sampling_ctx, ctx->llama_ctx, NULL);
        llama_sampling_accept(sampling_ctx, ctx->llama_ctx, next_token, true);

        free(candidates.data);

        // Check for EOS
        if (llama_token_is_eog(te->llama_model, next_token)) {
            break;
        }

        // Add token to context
        te_insert_tokens(te, te->n_tokens, 0, &next_token, 1);
        ctx->tokens_generated++;

        // Callback
        if (ctx->on_token) {
            ctx->on_token(ctx, next_token);
        }

        // Decode for next iteration
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = next_token;
        batch.pos[0] = te->n_tokens - 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        int result = llama_decode(ctx->llama_ctx, batch);
        llama_batch_free(batch);

        if (result != 0) {
            ctx->state = RLLM_STATE_ERROR;
            llama_sampling_free(sampling_ctx);
            return RLLM_ERROR_MODEL;
        }
    }

    llama_sampling_free(sampling_ctx);

    ctx->state = RLLM_STATE_COMPLETE;
    ctx->end_time = rllm_get_time_ms();

    if (ctx->on_complete) {
        ctx->on_complete(ctx, RLLM_STATE_COMPLETE);
    }

    return RLLM_OK;
}

rllm_error_t rllm_complete_sync(rllm_context_t *ctx, rllm_completion_params_t params,
                                 char *out, size_t *out_len) {
    if (!ctx || !out || !out_len) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    size_t start_tokens = te ? te->n_tokens : 0;

    rllm_error_t err = rllm_complete(ctx, params);
    if (err != RLLM_OK) return err;

    // Extract generated text
    if (te && te->n_tokens > start_tokens) {
        size_t gen_count = te->n_tokens - start_tokens;
        te_token_t *gen_tokens = te->tokens + start_tokens;
        return te_detokenize(te, gen_tokens, gen_count, out, out_len) == TE_OK ?
               RLLM_OK : RLLM_ERROR_MEMORY;
    }

    *out_len = 0;
    return RLLM_OK;
}

//
// Recursive evaluation patterns
//

rllm_error_t rllm_eval_in_child(rllm_context_t *parent, const char *prompt,
                                 rllm_completion_params_t params,
                                 char *result, size_t *result_len) {
    if (!parent || !prompt) return RLLM_ERROR_INVALID_CONTEXT;

    // Find environment (walk up to root)
    rllm_context_t *root = rllm_get_root(parent);
    if (!root) return RLLM_ERROR_INVALID_CONTEXT;

    // We need access to the environment - this is a limitation
    // In practice, you'd pass env as a parameter or store it in context
    // For now, we'll create a child with inherited config
    rllm_ctx_config_t config = parent->config;
    config.share_mode = RLLM_SHARE_NONE;  // Independent evaluation

    // This is a simplified version - full implementation would need env access
    // For demonstration, we show the pattern
    rllm_set_prompt(parent, prompt, strlen(prompt));
    return rllm_complete_sync(parent, params, result, result_len);
}

rllm_error_t rllm_self_eval(rllm_context_t *ctx, const char *eval_prompt,
                             rllm_completion_params_t params,
                             char *result, size_t *result_len) {
    if (!ctx || !eval_prompt) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    // Get current context as text
    size_t text_size = te->n_tokens * 8;  // Rough estimate
    char *current_text = (char *)malloc(text_size);
    if (!current_text) return RLLM_ERROR_MEMORY;

    te_detokenize(te, te->tokens, te->n_tokens, current_text, &text_size);

    // Create self-evaluation prompt
    size_t full_prompt_size = text_size + strlen(eval_prompt) + 256;
    char *full_prompt = (char *)malloc(full_prompt_size);
    if (!full_prompt) {
        free(current_text);
        return RLLM_ERROR_MEMORY;
    }

    snprintf(full_prompt, full_prompt_size,
             "[Context]\n%.*s\n\n[Evaluation Prompt]\n%s\n\n[Evaluation]",
             (int)text_size, current_text, eval_prompt);

    free(current_text);

    // Create snapshot to restore after evaluation
    te_snapshot_t *snapshot = te_create_snapshot(te);

    // Set evaluation prompt and run
    rllm_set_prompt(ctx, full_prompt, strlen(full_prompt));
    free(full_prompt);

    rllm_error_t err = rllm_complete_sync(ctx, params, result, result_len);

    // Restore original state
    if (snapshot) {
        te_restore_snapshot(te, snapshot);
        te_free_snapshot(snapshot);
    }

    return err;
}

rllm_error_t rllm_refine(rllm_context_t *ctx, const char *refine_prompt,
                          uint32_t max_iterations,
                          bool (*should_continue)(const char *output, void *user_data),
                          void *user_data,
                          char *result, size_t *result_len) {
    if (!ctx || !refine_prompt) return RLLM_ERROR_INVALID_CONTEXT;

    te_context_t *te = ctx->token_editor;
    if (!te) return RLLM_ERROR_INVALID_CONTEXT;

    rllm_completion_params_t params = ctx->config.completion;

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        // Run completion
        rllm_error_t err = rllm_complete(ctx, params);
        if (err != RLLM_OK) return err;

        // Get current output
        char *output = (char *)malloc(*result_len);
        if (!output) return RLLM_ERROR_MEMORY;

        size_t output_len = *result_len;
        te_detokenize(te, te->tokens, te->n_tokens, output, &output_len);

        // Check if we should continue
        bool cont = true;
        if (should_continue) {
            cont = should_continue(output, user_data);
        }

        if (!cont || iter == max_iterations - 1) {
            // Copy final result
            memcpy(result, output, output_len < *result_len ? output_len : *result_len);
            *result_len = output_len;
            free(output);
            return RLLM_OK;
        }

        free(output);

        // Append refinement prompt
        rllm_append_prompt(ctx, "\n\n", 2);
        rllm_append_prompt(ctx, refine_prompt, strlen(refine_prompt));
        rllm_append_prompt(ctx, "\n", 1);
    }

    return RLLM_OK;
}

//
// Inter-context communication
//

rllm_error_t rllm_send_message(rllm_context_t *from, rllm_context_t *to,
                                rllm_message_t *msg) {
    if (!from || !to || !msg) return RLLM_ERROR_INVALID_CONTEXT;

    // Add to receiver's queue
    size_t next_tail = (to->queue_tail + 1) % to->queue_capacity;
    if (next_tail == to->queue_head) {
        // Queue full
        return RLLM_ERROR_MEMORY;
    }

    msg->sender = from->id;
    msg->receiver = to->id;
    to->message_queue[to->queue_tail] = *msg;

    // Copy data if present
    if (msg->data && msg->data_size > 0) {
        to->message_queue[to->queue_tail].data = malloc(msg->data_size);
        if (to->message_queue[to->queue_tail].data) {
            memcpy(to->message_queue[to->queue_tail].data, msg->data, msg->data_size);
        }
    }

    to->queue_tail = next_tail;

    // Trigger callback
    if (to->on_message) {
        to->on_message(to, &to->message_queue[to->queue_tail - 1]);
    }

    return RLLM_OK;
}

rllm_error_t rllm_send_tokens(rllm_context_t *from, rllm_context_t *to,
                               te_token_t *tokens, size_t n_tokens) {
    rllm_message_t msg = {};
    msg.type = RLLM_MSG_TOKENS;
    msg.data = tokens;
    msg.data_size = n_tokens * sizeof(te_token_t);
    return rllm_send_message(from, to, &msg);
}

rllm_error_t rllm_send_text(rllm_context_t *from, rllm_context_t *to,
                             const char *text, size_t len) {
    rllm_message_t msg = {};
    msg.type = RLLM_MSG_TEXT;
    msg.data = (void *)text;
    msg.data_size = len;
    return rllm_send_message(from, to, &msg);
}

bool rllm_has_messages(rllm_context_t *ctx) {
    if (!ctx) return false;
    return ctx->queue_head != ctx->queue_tail;
}

rllm_error_t rllm_recv_message(rllm_context_t *ctx, rllm_message_t *msg,
                                uint32_t timeout_ms) {
    if (!ctx || !msg) return RLLM_ERROR_INVALID_CONTEXT;

    uint64_t start = rllm_get_time_ms();

    while (ctx->queue_head == ctx->queue_tail) {
        if (timeout_ms > 0 && rllm_get_time_ms() - start > timeout_ms) {
            return RLLM_ERROR_TIMEOUT;
        }
        // Simple spin wait - production code would use condition variables
        struct timespec ts = {0, 1000000};  // 1ms
        nanosleep(&ts, NULL);
    }

    *msg = ctx->message_queue[ctx->queue_head];
    ctx->queue_head = (ctx->queue_head + 1) % ctx->queue_capacity;

    return RLLM_OK;
}

void rllm_free_message(rllm_message_t *msg) {
    if (msg) {
        free(msg->data);
        msg->data = NULL;
        msg->data_size = 0;
    }
}

//
// Context tree operations
//

void rllm_walk_tree(rllm_context_t *root,
                     void (*visitor)(rllm_context_t *ctx, uint32_t depth, void *user_data),
                     void *user_data) {
    if (!root || !visitor) return;

    visitor(root, root->depth, user_data);

    for (size_t i = 0; i < root->n_children; i++) {
        rllm_walk_tree(root->children[i], visitor, user_data);
    }
}

rllm_context_t *rllm_find_context(rllm_context_t *root,
                                   bool (*predicate)(rllm_context_t *ctx, void *user_data),
                                   void *user_data) {
    if (!root || !predicate) return NULL;

    if (predicate(root, user_data)) {
        return root;
    }

    for (size_t i = 0; i < root->n_children; i++) {
        rllm_context_t *found = rllm_find_context(root->children[i], predicate, user_data);
        if (found) return found;
    }

    return NULL;
}

size_t rllm_count_descendants(rllm_context_t *ctx) {
    if (!ctx) return 0;

    size_t count = 0;
    for (size_t i = 0; i < ctx->n_children; i++) {
        count += 1 + rllm_count_descendants(ctx->children[i]);
    }
    return count;
}

//
// Debugging and introspection
//

const char *rllm_state_str(rllm_state_t state) {
    switch (state) {
        case RLLM_STATE_IDLE: return "idle";
        case RLLM_STATE_RUNNING: return "running";
        case RLLM_STATE_WAITING: return "waiting";
        case RLLM_STATE_COMPLETE: return "complete";
        case RLLM_STATE_ERROR: return "error";
        case RLLM_STATE_SUSPENDED: return "suspended";
        default: return "unknown";
    }
}

const char *rllm_relation_str(rllm_relation_t relation) {
    switch (relation) {
        case RLLM_REL_ROOT: return "root";
        case RLLM_REL_CHILD: return "child";
        case RLLM_REL_FORK: return "fork";
        case RLLM_REL_PEER: return "peer";
        default: return "unknown";
    }
}

static void rllm_print_tree_visitor(rllm_context_t *ctx, uint32_t depth, void *user_data) {
    (void)user_data;

    for (uint32_t i = 0; i < depth; i++) {
        printf("  ");
    }

    printf("[%u] %s (%s) - %zu tokens, state: %s\n",
           ctx->id,
           rllm_relation_str(ctx->relation),
           ctx->n_children > 0 ? "has children" : "leaf",
           ctx->token_editor ? ctx->token_editor->n_tokens : 0,
           rllm_state_str(ctx->state));
}

void rllm_print_tree(rllm_context_t *root) {
    printf("=== Context Tree ===\n");
    rllm_walk_tree(root, rllm_print_tree_visitor, NULL);
    printf("====================\n");
}

void rllm_set_trace(rllm_env_t *env, bool enable) {
    if (env) {
        env->config.enable_logging = enable;
    }
}
