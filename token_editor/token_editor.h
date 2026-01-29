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

#ifndef TOKEN_EDITOR_H
#define TOKEN_EDITOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Token Editor - Direct Context Token Manipulation System
//
// This module provides low-level access to LLM context tokens,
// enabling inspection, modification, and surgical editing of
// the token stream and associated KV cache state.
//

// Forward declarations
struct llama_context;
struct llama_model;

// Token position in context
typedef int32_t te_pos_t;
typedef int32_t te_token_t;
typedef int32_t te_seq_id_t;

// Error codes
typedef enum {
    TE_OK = 0,
    TE_ERROR_INVALID_CONTEXT = -1,
    TE_ERROR_INVALID_POSITION = -2,
    TE_ERROR_INVALID_TOKEN = -3,
    TE_ERROR_BUFFER_TOO_SMALL = -4,
    TE_ERROR_KV_CACHE_FULL = -5,
    TE_ERROR_SEQUENCE_NOT_FOUND = -6,
    TE_ERROR_ALLOCATION_FAILED = -7,
    TE_ERROR_READONLY = -8,
} te_error_t;

// Token metadata
typedef struct {
    te_token_t id;              // Token ID
    te_pos_t   pos;             // Position in sequence
    te_seq_id_t seq_id;         // Sequence ID
    float      logit;           // Log probability (if computed)
    float      prob;            // Probability (if computed)
    bool       has_logit;       // Whether logit was computed
    uint32_t   flags;           // Token flags (special, BOS, EOS, etc.)
} te_token_info_t;

// Token flags
#define TE_FLAG_BOS        (1 << 0)   // Beginning of sequence
#define TE_FLAG_EOS        (1 << 1)   // End of sequence
#define TE_FLAG_SPECIAL    (1 << 2)   // Special token
#define TE_FLAG_CONTROL    (1 << 3)   // Control token
#define TE_FLAG_USER_DATA  (1 << 4)   // User-injected token
#define TE_FLAG_GENERATED  (1 << 5)   // Model-generated token

// Token range for batch operations
typedef struct {
    te_pos_t start;             // Start position (inclusive)
    te_pos_t end;               // End position (exclusive)
    te_seq_id_t seq_id;         // Sequence ID (-1 for all)
} te_range_t;

// Edit operation types
typedef enum {
    TE_OP_INSERT,               // Insert tokens at position
    TE_OP_DELETE,               // Delete tokens in range
    TE_OP_REPLACE,              // Replace tokens in range
    TE_OP_MOVE,                 // Move tokens to new position
    TE_OP_COPY,                 // Copy tokens to new position
    TE_OP_SWAP,                 // Swap two token ranges
} te_op_type_t;

// Edit operation (for undo/redo)
typedef struct te_edit_op {
    te_op_type_t type;
    te_range_t   source;
    te_range_t   dest;
    te_token_t  *tokens;        // Tokens involved (for undo)
    size_t       n_tokens;
    struct te_edit_op *next;    // Linked list for history
} te_edit_op_t;

// Token editor context
typedef struct te_context {
    struct llama_context *llama_ctx;
    struct llama_model   *llama_model;

    // Token buffer
    te_token_t    *tokens;
    te_token_info_t *token_info;
    size_t         n_tokens;
    size_t         capacity;

    // Edit history
    te_edit_op_t  *history;
    te_edit_op_t  *history_tail;
    te_edit_op_t  *redo_stack;
    size_t         history_limit;

    // Sequence tracking
    te_seq_id_t   *active_sequences;
    size_t         n_sequences;

    // State flags
    bool           readonly;
    bool           kv_cache_dirty;
    bool           logits_valid;
    bool           suppress_history;  // Internal: suppress history during undo/redo

    // Callbacks
    void (*on_token_change)(struct te_context *ctx, te_pos_t pos, te_token_t old_tok, te_token_t new_tok);
    void (*on_range_change)(struct te_context *ctx, te_range_t range);
    void *user_data;
} te_context_t;

// Snapshot for state preservation
typedef struct {
    te_token_t    *tokens;
    te_token_info_t *token_info;
    size_t         n_tokens;
    void          *kv_cache_state;
    size_t         kv_cache_size;
    uint32_t       rng_state;
} te_snapshot_t;

//
// Initialization and cleanup
//

// Create token editor context from llama context
te_context_t *te_init(struct llama_context *ctx, struct llama_model *model);

// Free token editor context
void te_free(te_context_t *ctx);

// Set history limit (0 = unlimited)
void te_set_history_limit(te_context_t *ctx, size_t limit);

//
// Token inspection
//

// Get token at position
te_token_t te_get_token(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id);

// Get token info at position
te_error_t te_get_token_info(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id, te_token_info_t *info);

// Get token string representation
int te_token_to_string(te_context_t *ctx, te_token_t token, char *buf, size_t buf_size);

// Get all tokens in range
te_error_t te_get_tokens(te_context_t *ctx, te_range_t range, te_token_t *out, size_t *n_out);

// Get current token count
size_t te_get_token_count(te_context_t *ctx, te_seq_id_t seq_id);

// Tokenize text and get token array (doesn't modify context)
te_error_t te_tokenize(te_context_t *ctx, const char *text, size_t text_len,
                       te_token_t *out, size_t *n_out, bool add_bos);

// Detokenize tokens to text
te_error_t te_detokenize(te_context_t *ctx, const te_token_t *tokens, size_t n_tokens,
                         char *out, size_t *out_len);

//
// Token modification
//

// Set token at position
te_error_t te_set_token(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id, te_token_t token);

// Insert tokens at position
te_error_t te_insert_tokens(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id,
                            const te_token_t *tokens, size_t n_tokens);

// Delete tokens in range
te_error_t te_delete_tokens(te_context_t *ctx, te_range_t range);

// Replace tokens in range with new tokens
te_error_t te_replace_tokens(te_context_t *ctx, te_range_t range,
                             const te_token_t *tokens, size_t n_tokens);

// Replace text (tokenize and replace)
te_error_t te_replace_text(te_context_t *ctx, te_range_t range, const char *text, size_t text_len);

// Clear all tokens
te_error_t te_clear(te_context_t *ctx, te_seq_id_t seq_id);

// NOTE: Batch operations (te_begin_batch, te_commit_batch, te_rollback_batch)
// are planned for future implementation.

//
// Undo/Redo support
//

// Undo last operation
te_error_t te_undo(te_context_t *ctx);

// Redo last undone operation
te_error_t te_redo(te_context_t *ctx);

// Get undo history count
size_t te_get_history_count(te_context_t *ctx);

// Clear history
void te_clear_history(te_context_t *ctx);

//
// Sequence management
//

// Create new sequence
te_seq_id_t te_create_sequence(te_context_t *ctx);

// Delete sequence
te_error_t te_delete_sequence(te_context_t *ctx, te_seq_id_t seq_id);

// Copy sequence
te_error_t te_copy_sequence(te_context_t *ctx, te_seq_id_t src, te_seq_id_t dst);

// Fork sequence (copy and create new)
te_seq_id_t te_fork_sequence(te_context_t *ctx, te_seq_id_t src);

// NOTE: te_merge_sequences is planned for future implementation.

//
// KV Cache management
//

// Sync token changes to KV cache
te_error_t te_sync_kv_cache(te_context_t *ctx);

// Invalidate KV cache for range
te_error_t te_invalidate_kv_range(te_context_t *ctx, te_range_t range);

// Clear KV cache
te_error_t te_clear_kv_cache(te_context_t *ctx, te_seq_id_t seq_id);

// Shift KV cache positions (for sliding window)
te_error_t te_shift_kv_cache(te_context_t *ctx, te_seq_id_t seq_id, te_pos_t delta);

//
// Snapshot and restore
//

// Create snapshot of current state
te_snapshot_t *te_create_snapshot(te_context_t *ctx);

// Restore from snapshot
te_error_t te_restore_snapshot(te_context_t *ctx, te_snapshot_t *snapshot);

// Free snapshot
void te_free_snapshot(te_snapshot_t *snapshot);

//
// Logits and probabilities
//

// Compute logits for token at position
te_error_t te_compute_logits(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id);

// Get top-k tokens by probability
te_error_t te_get_top_k(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id,
                        te_token_info_t *out, size_t k);

// Get logit of specific token at position (raw log-odds, not normalized probability)
float te_get_token_logit(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id, te_token_t token);

//
// Search and navigation
//

// Find token occurrences
te_error_t te_find_token(te_context_t *ctx, te_token_t token, te_seq_id_t seq_id,
                         te_pos_t *positions, size_t *n_positions);

// Find text pattern (tokenized)
te_error_t te_find_text(te_context_t *ctx, const char *text, te_seq_id_t seq_id,
                        te_pos_t *positions, size_t *n_positions);

//
// Export and import
//

// Export tokens to JSON
te_error_t te_export_json(te_context_t *ctx, te_seq_id_t seq_id, char *buf, size_t *buf_size);

// NOTE: te_import_json is planned for future implementation.

// Export to binary format
te_error_t te_export_binary(te_context_t *ctx, te_seq_id_t seq_id, void *buf, size_t *buf_size);

// Import from binary format
te_error_t te_import_binary(te_context_t *ctx, te_seq_id_t seq_id, const void *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif // TOKEN_EDITOR_H
