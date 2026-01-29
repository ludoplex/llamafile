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

#ifndef RECURSIVE_LLM_H
#define RECURSIVE_LLM_H

#include "token_editor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Recursive LLM Environment
//
// This module provides a hierarchical context management system
// that enables LLMs to spawn sub-contexts, fork evaluation paths,
// and perform recursive self-evaluation with shared token memory.
//

// Forward declarations
struct llama_model;
struct llama_context;

// Maximum recursion depth
#define RLLM_MAX_DEPTH 32

// Maximum concurrent contexts
#define RLLM_MAX_CONTEXTS 64

// Context relationship types
typedef enum {
    RLLM_REL_ROOT = 0,          // Root context (no parent)
    RLLM_REL_CHILD,             // Child context (inherits from parent)
    RLLM_REL_FORK,              // Forked context (copy of parent)
    RLLM_REL_PEER,              // Peer context (shares model, separate state)
} rllm_relation_t;

// Context state
typedef enum {
    RLLM_STATE_IDLE = 0,
    RLLM_STATE_RUNNING,
    RLLM_STATE_WAITING,
    RLLM_STATE_COMPLETE,
    RLLM_STATE_ERROR,
    RLLM_STATE_SUSPENDED,
} rllm_state_t;

// Memory sharing mode
typedef enum {
    RLLM_SHARE_NONE = 0,        // No sharing
    RLLM_SHARE_KV_READ,         // Read-only KV cache access to parent
    RLLM_SHARE_KV_COPY,         // Copy KV cache from parent
    RLLM_SHARE_TOKENS_READ,     // Read-only token access to parent
    RLLM_SHARE_TOKENS_COPY,     // Copy tokens from parent
    RLLM_SHARE_FULL,            // Full sharing (KV + tokens, copy-on-write)
} rllm_share_mode_t;

// Error codes
typedef enum {
    RLLM_OK = 0,
    RLLM_ERROR_MAX_DEPTH = -100,
    RLLM_ERROR_MAX_CONTEXTS = -101,
    RLLM_ERROR_INVALID_CONTEXT = -102,
    RLLM_ERROR_INVALID_PARENT = -103,
    RLLM_ERROR_CONTEXT_BUSY = -104,
    RLLM_ERROR_RECURSION_LIMIT = -105,
    RLLM_ERROR_MEMORY = -106,
    RLLM_ERROR_MODEL = -107,
    RLLM_ERROR_DEADLOCK = -108,
    RLLM_ERROR_TIMEOUT = -109,
} rllm_error_t;

// Context ID
typedef uint32_t rllm_ctx_id_t;
#define RLLM_INVALID_CTX_ID ((rllm_ctx_id_t)-1)

// Message types for inter-context communication
typedef enum {
    RLLM_MSG_TOKENS,            // Token array
    RLLM_MSG_TEXT,              // Text string
    RLLM_MSG_COMPLETION,        // Completion result
    RLLM_MSG_EMBEDDING,         // Embedding vector
    RLLM_MSG_CONTROL,           // Control message
    RLLM_MSG_QUERY,             // Query from child to parent
    RLLM_MSG_RESPONSE,          // Response from parent to child
} rllm_msg_type_t;

// Inter-context message
typedef struct {
    rllm_msg_type_t type;
    rllm_ctx_id_t   sender;
    rllm_ctx_id_t   receiver;
    uint32_t        seq_num;
    size_t          data_size;
    void           *data;
} rllm_message_t;

// Completion parameters for spawned context
typedef struct {
    uint32_t n_predict;         // Max tokens to generate
    float    temperature;
    float    top_p;
    float    top_k;
    float    repeat_penalty;
    bool     stream;            // Stream tokens back to parent
    uint32_t timeout_ms;        // Timeout (0 = no timeout)
} rllm_completion_params_t;

// Context configuration
typedef struct {
    uint32_t n_ctx;             // Context size
    uint32_t n_batch;           // Batch size
    uint32_t n_threads;         // Thread count
    rllm_share_mode_t share_mode;
    rllm_completion_params_t completion;
    bool inherit_prompt;        // Inherit parent's prompt
    bool inherit_sampling;      // Inherit parent's sampling state
} rllm_ctx_config_t;

// Recursive context
typedef struct rllm_context {
    rllm_ctx_id_t   id;
    rllm_relation_t relation;
    rllm_state_t    state;

    // Parent/child relationships
    struct rllm_context *parent;
    struct rllm_context **children;
    size_t          n_children;
    size_t          children_capacity;
    uint32_t        depth;

    // Underlying llama context
    struct llama_context *llama_ctx;
    te_context_t         *token_editor;

    // Configuration
    rllm_ctx_config_t config;

    // Message queue
    rllm_message_t *message_queue;
    size_t          queue_head;
    size_t          queue_tail;
    size_t          queue_capacity;

    // Execution state
    void           *execution_state;
    uint64_t        start_time;
    uint64_t        end_time;
    uint32_t        tokens_generated;

    // Callbacks
    void (*on_token)(struct rllm_context *ctx, te_token_t token);
    void (*on_complete)(struct rllm_context *ctx, rllm_state_t final_state);
    void (*on_message)(struct rllm_context *ctx, rllm_message_t *msg);
    void *user_data;
} rllm_context_t;

// Environment configuration
typedef struct {
    uint32_t max_depth;
    uint32_t max_contexts;
    uint32_t default_n_ctx;
    uint32_t default_n_batch;
    uint32_t default_n_threads;
    size_t   memory_limit;      // Total memory limit for all contexts
    bool     enable_logging;
    bool     enable_metrics;
} rllm_env_config_t;

// Recursive LLM environment
typedef struct rllm_env {
    struct llama_model *model;

    // Context pool
    rllm_context_t **contexts;
    size_t          n_contexts;
    size_t          context_capacity;
    rllm_ctx_id_t   next_ctx_id;

    // Root contexts
    rllm_context_t **roots;
    size_t          n_roots;

    // Configuration
    rllm_env_config_t config;

    // Statistics
    uint64_t        total_tokens_processed;
    uint64_t        total_contexts_created;
    uint64_t        total_recursions;
    uint64_t        peak_depth;
    size_t          memory_used;

    // Thread safety
    void           *mutex;

    // Callbacks
    void (*on_context_create)(struct rllm_env *env, rllm_context_t *ctx);
    void (*on_context_destroy)(struct rllm_env *env, rllm_context_t *ctx);
    void (*on_recursion)(struct rllm_env *env, rllm_context_t *parent, rllm_context_t *child);
    void *user_data;
} rllm_env_t;

// Default configuration
rllm_env_config_t rllm_default_env_config(void);
rllm_ctx_config_t rllm_default_ctx_config(void);
rllm_completion_params_t rllm_default_completion_params(void);

//
// Environment lifecycle
//

// Initialize environment with model
rllm_env_t *rllm_init(struct llama_model *model, rllm_env_config_t config);

// Shutdown environment (frees all contexts)
void rllm_shutdown(rllm_env_t *env);

// Get environment statistics
void rllm_get_stats(rllm_env_t *env, uint64_t *total_tokens,
                    uint64_t *total_contexts, uint64_t *peak_depth);

//
// Context creation
//

// Create root context
rllm_context_t *rllm_create_root(rllm_env_t *env, rllm_ctx_config_t config);

// Spawn child context (inherits from parent)
rllm_context_t *rllm_spawn_child(rllm_env_t *env, rllm_context_t *parent,
                                  rllm_ctx_config_t config);

// Fork context (creates independent copy)
rllm_context_t *rllm_fork(rllm_env_t *env, rllm_context_t *source);

// Create peer context (shares model, independent state)
rllm_context_t *rllm_create_peer(rllm_env_t *env, rllm_context_t *peer,
                                  rllm_ctx_config_t config);

// Destroy context and all children
rllm_error_t rllm_destroy(rllm_env_t *env, rllm_context_t *ctx);

//
// Context lookup
//

// Get context by ID
rllm_context_t *rllm_get_context(rllm_env_t *env, rllm_ctx_id_t id);

// Get parent context
rllm_context_t *rllm_get_parent(rllm_context_t *ctx);

// Get children
rllm_context_t **rllm_get_children(rllm_context_t *ctx, size_t *n_children);

// Get root of context tree
rllm_context_t *rllm_get_root(rllm_context_t *ctx);

// Get depth in tree
uint32_t rllm_get_depth(rllm_context_t *ctx);

//
// Token operations (via token editor)
//

// Get token editor for context
te_context_t *rllm_get_token_editor(rllm_context_t *ctx);

// Set prompt (clears existing tokens)
rllm_error_t rllm_set_prompt(rllm_context_t *ctx, const char *prompt, size_t len);

// Append to prompt
rllm_error_t rllm_append_prompt(rllm_context_t *ctx, const char *text, size_t len);

// Get current context as text
rllm_error_t rllm_get_text(rllm_context_t *ctx, char *buf, size_t *buf_size);

// NOTE: rllm_copy_from_parent is planned for future implementation.

//
// Execution
//

// Run completion on context
rllm_error_t rllm_complete(rllm_context_t *ctx, rllm_completion_params_t params);

// Run completion and wait for result
rllm_error_t rllm_complete_sync(rllm_context_t *ctx, rllm_completion_params_t params,
                                 char *out, size_t *out_len);

// NOTE: Async execution APIs (rllm_complete_async, rllm_wait, rllm_wait_children,
// rllm_suspend, rllm_resume, rllm_cancel) are planned for future implementation.

//
// Recursive evaluation patterns
//

// Evaluate prompt in child context (simplified - uses parent context with snapshot)
rllm_error_t rllm_eval_in_child(rllm_context_t *parent, const char *prompt,
                                 rllm_completion_params_t params,
                                 char *result, size_t *result_len);

// NOTE: Fan-out/gather/map-reduce patterns (rllm_fanout, rllm_gather, rllm_map_reduce)
// are planned for future implementation.

// Self-evaluation (context evaluates its own output)
rllm_error_t rllm_self_eval(rllm_context_t *ctx, const char *eval_prompt,
                             rllm_completion_params_t params,
                             char *result, size_t *result_len);

// Iterative refinement
rllm_error_t rllm_refine(rllm_context_t *ctx, const char *refine_prompt,
                          uint32_t max_iterations,
                          bool (*should_continue)(const char *output, void *user_data),
                          void *user_data,
                          char *result, size_t *result_len);

//
// Inter-context communication
//

// Send message to context
rllm_error_t rllm_send_message(rllm_context_t *from, rllm_context_t *to,
                                rllm_message_t *msg);

// Send tokens to context
rllm_error_t rllm_send_tokens(rllm_context_t *from, rllm_context_t *to,
                               te_token_t *tokens, size_t n_tokens);

// Send text to context
rllm_error_t rllm_send_text(rllm_context_t *from, rllm_context_t *to,
                             const char *text, size_t len);

// Receive message (blocking)
rllm_error_t rllm_recv_message(rllm_context_t *ctx, rllm_message_t *msg,
                                uint32_t timeout_ms);

// Check for pending messages
bool rllm_has_messages(rllm_context_t *ctx);

// Free message data
void rllm_free_message(rllm_message_t *msg);

//
// Context tree operations
//

// Walk tree depth-first
void rllm_walk_tree(rllm_context_t *root,
                     void (*visitor)(rllm_context_t *ctx, uint32_t depth, void *user_data),
                     void *user_data);

// Find context by predicate
rllm_context_t *rllm_find_context(rllm_context_t *root,
                                   bool (*predicate)(rllm_context_t *ctx, void *user_data),
                                   void *user_data);

// Count descendants
size_t rllm_count_descendants(rllm_context_t *ctx);

// NOTE: Tree manipulation APIs (rllm_prune_completed, rllm_merge_child_result)
// are planned for future implementation.

// NOTE: State persistence APIs (rllm_save_tree, rllm_load_tree, rllm_export_tree_json)
// are planned for future implementation.

//
// Debugging and introspection
//

// Print context tree
void rllm_print_tree(rllm_context_t *root);

// Get context state string
const char *rllm_state_str(rllm_state_t state);

// Get relation string
const char *rllm_relation_str(rllm_relation_t relation);

// Enable/disable tracing
void rllm_set_trace(rllm_env_t *env, bool enable);

#ifdef __cplusplus
}
#endif

#endif // RECURSIVE_LLM_H
