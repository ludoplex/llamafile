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

#include "token_editor.h"

#include <cosmo.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "llama.cpp/llama.h"

// Default capacity for token buffer
#define TE_DEFAULT_CAPACITY 4096
#define TE_HISTORY_DEFAULT_LIMIT 100

//
// Internal helpers
//

static te_edit_op_t *te_alloc_edit_op(void) {
    te_edit_op_t *op = (te_edit_op_t *)calloc(1, sizeof(te_edit_op_t));
    return op;
}

static void te_free_edit_op(te_edit_op_t *op) {
    if (op) {
        free(op->tokens);
        free(op);
    }
}

static void te_clear_edit_history(te_edit_op_t **head) {
    te_edit_op_t *current = *head;
    while (current) {
        te_edit_op_t *next = current->next;
        te_free_edit_op(current);
        current = next;
    }
    *head = NULL;
}

static te_error_t te_ensure_capacity(te_context_t *ctx, size_t required) {
    if (required <= ctx->capacity) {
        return TE_OK;
    }

    size_t new_capacity = ctx->capacity * 2;
    while (new_capacity < required) {
        new_capacity *= 2;
    }

    te_token_t *new_tokens = (te_token_t *)realloc(ctx->tokens,
                                                     new_capacity * sizeof(te_token_t));
    if (!new_tokens) {
        return TE_ERROR_ALLOCATION_FAILED;
    }
    ctx->tokens = new_tokens;

    te_token_info_t *new_info = (te_token_info_t *)realloc(ctx->token_info,
                                                            new_capacity * sizeof(te_token_info_t));
    if (!new_info) {
        return TE_ERROR_ALLOCATION_FAILED;
    }
    ctx->token_info = new_info;

    ctx->capacity = new_capacity;
    return TE_OK;
}

static void te_record_edit(te_context_t *ctx, te_op_type_t type,
                           te_range_t source, te_range_t dest,
                           const te_token_t *tokens, size_t n_tokens) {
    if (ctx->suppress_history) return;

    te_edit_op_t *op = te_alloc_edit_op();
    if (!op) return;

    op->type = type;
    op->source = source;
    op->dest = dest;
    op->n_tokens = n_tokens;

    if (tokens && n_tokens > 0) {
        op->tokens = (te_token_t *)malloc(n_tokens * sizeof(te_token_t));
        if (op->tokens) {
            memcpy(op->tokens, tokens, n_tokens * sizeof(te_token_t));
        }
    }

    // Add to history
    op->next = NULL;
    if (ctx->history_tail) {
        ctx->history_tail->next = op;
    } else {
        ctx->history = op;
    }
    ctx->history_tail = op;

    // Clear redo stack
    te_clear_edit_history(&ctx->redo_stack);

    // Enforce history limit
    if (ctx->history_limit > 0) {
        size_t count = 0;
        te_edit_op_t *curr = ctx->history;
        while (curr) {
            count++;
            curr = curr->next;
        }
        while (count > ctx->history_limit && ctx->history) {
            te_edit_op_t *old = ctx->history;
            ctx->history = old->next;
            if (ctx->history == NULL) {
                ctx->history_tail = NULL;
            }
            te_free_edit_op(old);
            count--;
        }
    }
}

static uint32_t te_get_token_flags(te_context_t *ctx, te_token_t token) {
    uint32_t flags = 0;

    if (llama_token_is_eog(ctx->llama_model, token)) {
        flags |= TE_FLAG_EOS;
    }

    if (token == llama_token_bos(ctx->llama_model)) {
        flags |= TE_FLAG_BOS;
    }

    // Check for special tokens
    llama_token_attr attr = llama_token_get_attr(ctx->llama_model, token);
    if (attr & LLAMA_TOKEN_ATTR_CONTROL) {
        flags |= TE_FLAG_CONTROL;
    }
    if (attr & LLAMA_TOKEN_ATTR_SPECIAL) {
        flags |= TE_FLAG_SPECIAL;
    }

    return flags;
}

//
// Initialization and cleanup
//

te_context_t *te_init(struct llama_context *ctx, struct llama_model *model) {
    if (!ctx || !model) {
        return NULL;
    }

    te_context_t *te_ctx = (te_context_t *)calloc(1, sizeof(te_context_t));
    if (!te_ctx) {
        return NULL;
    }

    te_ctx->llama_ctx = ctx;
    te_ctx->llama_model = model;
    te_ctx->capacity = TE_DEFAULT_CAPACITY;
    te_ctx->history_limit = TE_HISTORY_DEFAULT_LIMIT;

    te_ctx->tokens = (te_token_t *)calloc(te_ctx->capacity, sizeof(te_token_t));
    te_ctx->token_info = (te_token_info_t *)calloc(te_ctx->capacity, sizeof(te_token_info_t));

    if (!te_ctx->tokens || !te_ctx->token_info) {
        free(te_ctx->tokens);
        free(te_ctx->token_info);
        free(te_ctx);
        return NULL;
    }

    // Initialize sequence tracking
    te_ctx->active_sequences = (te_seq_id_t *)calloc(16, sizeof(te_seq_id_t));
    if (!te_ctx->active_sequences) {
        free(te_ctx->tokens);
        free(te_ctx->token_info);
        free(te_ctx);
        return NULL;
    }
    te_ctx->active_sequences[0] = 0;  // Default sequence
    te_ctx->n_sequences = 1;

    return te_ctx;
}

void te_free(te_context_t *ctx) {
    if (!ctx) return;

    te_clear_edit_history(&ctx->history);
    te_clear_edit_history(&ctx->redo_stack);

    free(ctx->tokens);
    free(ctx->token_info);
    free(ctx->active_sequences);
    free(ctx);
}

void te_set_history_limit(te_context_t *ctx, size_t limit) {
    if (ctx) {
        ctx->history_limit = limit;
    }
}

//
// Token inspection
//

te_token_t te_get_token(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id) {
    if (!ctx || pos < 0 || (size_t)pos >= ctx->n_tokens) {
        return -1;
    }

    // For now, we store tokens linearly and ignore seq_id
    // A more sophisticated implementation would maintain per-sequence token arrays
    (void)seq_id;
    return ctx->tokens[pos];
}

te_error_t te_get_token_info(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id,
                              te_token_info_t *info) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (pos < 0 || (size_t)pos >= ctx->n_tokens) return TE_ERROR_INVALID_POSITION;
    if (!info) return TE_ERROR_INVALID_TOKEN;

    (void)seq_id;

    *info = ctx->token_info[pos];
    info->id = ctx->tokens[pos];
    info->pos = pos;
    info->seq_id = seq_id >= 0 ? seq_id : 0;
    info->flags = te_get_token_flags(ctx, info->id);

    return TE_OK;
}

int te_token_to_string(te_context_t *ctx, te_token_t token, char *buf, size_t buf_size) {
    if (!ctx || !buf || buf_size == 0) {
        return -1;
    }

    return llama_token_to_piece(ctx->llama_model, token, buf, buf_size, 0, true);
}

te_error_t te_get_tokens(te_context_t *ctx, te_range_t range, te_token_t *out, size_t *n_out) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!out || !n_out) return TE_ERROR_BUFFER_TOO_SMALL;

    te_pos_t start = range.start;
    te_pos_t end = range.end;

    if (start < 0) start = 0;
    if ((size_t)end > ctx->n_tokens) end = ctx->n_tokens;
    if (start >= end) {
        *n_out = 0;
        return TE_OK;
    }

    size_t count = end - start;
    if (count > *n_out) {
        return TE_ERROR_BUFFER_TOO_SMALL;
    }

    memcpy(out, ctx->tokens + start, count * sizeof(te_token_t));
    *n_out = count;

    return TE_OK;
}

size_t te_get_token_count(te_context_t *ctx, te_seq_id_t seq_id) {
    if (!ctx) return 0;
    (void)seq_id;
    return ctx->n_tokens;
}

te_error_t te_tokenize(te_context_t *ctx, const char *text, size_t text_len,
                        te_token_t *out, size_t *n_out, bool add_bos) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!text || !out || !n_out) return TE_ERROR_BUFFER_TOO_SMALL;

    int n = llama_tokenize(ctx->llama_model, text, text_len, out, *n_out, add_bos, true);
    if (n < 0) {
        return TE_ERROR_BUFFER_TOO_SMALL;
    }

    *n_out = n;
    return TE_OK;
}

te_error_t te_detokenize(te_context_t *ctx, const te_token_t *tokens, size_t n_tokens,
                          char *out, size_t *out_len) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!tokens || !out || !out_len) return TE_ERROR_BUFFER_TOO_SMALL;

    size_t total = 0;
    char piece[256];

    for (size_t i = 0; i < n_tokens; i++) {
        int len = llama_token_to_piece(ctx->llama_model, tokens[i], piece, sizeof(piece), 0, true);
        if (len < 0) {
            return TE_ERROR_INVALID_TOKEN;
        }

        if (total + len > *out_len) {
            return TE_ERROR_BUFFER_TOO_SMALL;
        }

        memcpy(out + total, piece, len);
        total += len;
    }

    *out_len = total;
    return TE_OK;
}

//
// Token modification
//

te_error_t te_set_token(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id, te_token_t token) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (ctx->readonly) return TE_ERROR_READONLY;
    if (pos < 0 || (size_t)pos >= ctx->n_tokens) return TE_ERROR_INVALID_POSITION;

    (void)seq_id;

    te_token_t old_token = ctx->tokens[pos];

    // Record for undo
    te_range_t range = {pos, pos + 1, seq_id};
    te_record_edit(ctx, TE_OP_REPLACE, range, range, &old_token, 1);

    ctx->tokens[pos] = token;
    ctx->token_info[pos].id = token;
    ctx->token_info[pos].flags = te_get_token_flags(ctx, token);
    ctx->token_info[pos].has_logit = false;
    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    // Notify callback
    if (ctx->on_token_change) {
        ctx->on_token_change(ctx, pos, old_token, token);
    }

    return TE_OK;
}

te_error_t te_insert_tokens(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id,
                             const te_token_t *tokens, size_t n_tokens) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (ctx->readonly) return TE_ERROR_READONLY;
    if (pos < 0 || (size_t)pos > ctx->n_tokens) return TE_ERROR_INVALID_POSITION;
    if (!tokens || n_tokens == 0) return TE_OK;

    te_error_t err = te_ensure_capacity(ctx, ctx->n_tokens + n_tokens);
    if (err != TE_OK) return err;

    // Shift existing tokens
    if ((size_t)pos < ctx->n_tokens) {
        memmove(ctx->tokens + pos + n_tokens, ctx->tokens + pos,
                (ctx->n_tokens - pos) * sizeof(te_token_t));
        memmove(ctx->token_info + pos + n_tokens, ctx->token_info + pos,
                (ctx->n_tokens - pos) * sizeof(te_token_info_t));
    }

    // Insert new tokens
    memcpy(ctx->tokens + pos, tokens, n_tokens * sizeof(te_token_t));
    for (size_t i = 0; i < n_tokens; i++) {
        ctx->token_info[pos + i].id = tokens[i];
        ctx->token_info[pos + i].pos = pos + i;
        ctx->token_info[pos + i].seq_id = seq_id >= 0 ? seq_id : 0;
        ctx->token_info[pos + i].flags = te_get_token_flags(ctx, tokens[i]) | TE_FLAG_USER_DATA;
        ctx->token_info[pos + i].has_logit = false;
    }

    ctx->n_tokens += n_tokens;
    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    // Record for undo
    te_range_t range = {pos, (te_pos_t)(pos + n_tokens), seq_id};
    te_record_edit(ctx, TE_OP_INSERT, range, range, tokens, n_tokens);

    // Notify callback
    if (ctx->on_range_change) {
        ctx->on_range_change(ctx, range);
    }

    return TE_OK;
}

te_error_t te_delete_tokens(te_context_t *ctx, te_range_t range) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (ctx->readonly) return TE_ERROR_READONLY;

    te_pos_t start = range.start;
    te_pos_t end = range.end;

    if (start < 0) start = 0;
    if ((size_t)end > ctx->n_tokens) end = ctx->n_tokens;
    if (start >= end) return TE_OK;

    size_t count = end - start;

    // Save for undo
    te_token_t *saved = (te_token_t *)malloc(count * sizeof(te_token_t));
    if (saved) {
        memcpy(saved, ctx->tokens + start, count * sizeof(te_token_t));
        te_range_t save_range = {start, end, range.seq_id};
        te_record_edit(ctx, TE_OP_DELETE, save_range, save_range, saved, count);
        free(saved);
    }

    // Shift remaining tokens
    if ((size_t)end < ctx->n_tokens) {
        memmove(ctx->tokens + start, ctx->tokens + end,
                (ctx->n_tokens - end) * sizeof(te_token_t));
        memmove(ctx->token_info + start, ctx->token_info + end,
                (ctx->n_tokens - end) * sizeof(te_token_info_t));
    }

    ctx->n_tokens -= count;
    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    // Notify callback
    if (ctx->on_range_change) {
        ctx->on_range_change(ctx, range);
    }

    return TE_OK;
}

te_error_t te_replace_tokens(te_context_t *ctx, te_range_t range,
                              const te_token_t *tokens, size_t n_tokens) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (ctx->readonly) return TE_ERROR_READONLY;

    te_pos_t start = range.start;
    te_pos_t end = range.end;

    if (start < 0) start = 0;
    if ((size_t)end > ctx->n_tokens) end = ctx->n_tokens;

    size_t old_count = (start < end) ? (end - start) : 0;
    size_t new_count = n_tokens;

    // Save old tokens for undo
    te_token_t *saved = NULL;
    if (old_count > 0) {
        saved = (te_token_t *)malloc(old_count * sizeof(te_token_t));
        if (saved) {
            memcpy(saved, ctx->tokens + start, old_count * sizeof(te_token_t));
        }
    }

    // Calculate new size
    size_t new_total = ctx->n_tokens - old_count + new_count;
    te_error_t err = te_ensure_capacity(ctx, new_total);
    if (err != TE_OK) {
        free(saved);
        return err;
    }

    // Shift tokens if sizes differ
    if (new_count != old_count && (size_t)end < ctx->n_tokens) {
        memmove(ctx->tokens + start + new_count, ctx->tokens + end,
                (ctx->n_tokens - end) * sizeof(te_token_t));
        memmove(ctx->token_info + start + new_count, ctx->token_info + end,
                (ctx->n_tokens - end) * sizeof(te_token_info_t));
    }

    // Copy new tokens
    if (n_tokens > 0 && tokens) {
        memcpy(ctx->tokens + start, tokens, n_tokens * sizeof(te_token_t));
        for (size_t i = 0; i < n_tokens; i++) {
            ctx->token_info[start + i].id = tokens[i];
            ctx->token_info[start + i].pos = start + i;
            ctx->token_info[start + i].seq_id = range.seq_id >= 0 ? range.seq_id : 0;
            ctx->token_info[start + i].flags = te_get_token_flags(ctx, tokens[i]) | TE_FLAG_USER_DATA;
            ctx->token_info[start + i].has_logit = false;
        }
    }

    ctx->n_tokens = new_total;
    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    // Record for undo
    if (saved) {
        te_range_t save_range = {start, end, range.seq_id};
        te_record_edit(ctx, TE_OP_REPLACE, save_range, save_range, saved, old_count);
        free(saved);
    }

    // Notify callback
    if (ctx->on_range_change) {
        te_range_t changed_range = {start, (te_pos_t)(start + new_count), range.seq_id};
        ctx->on_range_change(ctx, changed_range);
    }

    return TE_OK;
}

te_error_t te_replace_text(te_context_t *ctx, te_range_t range, const char *text, size_t text_len) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!text) return TE_ERROR_INVALID_TOKEN;

    // Tokenize the text
    size_t max_tokens = text_len + 1;  // Rough estimate
    te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    if (!tokens) return TE_ERROR_ALLOCATION_FAILED;

    size_t n_tokens = max_tokens;
    te_error_t err = te_tokenize(ctx, text, text_len, tokens, &n_tokens, false);
    if (err != TE_OK) {
        free(tokens);
        return err;
    }

    err = te_replace_tokens(ctx, range, tokens, n_tokens);
    free(tokens);

    return err;
}

te_error_t te_clear(te_context_t *ctx, te_seq_id_t seq_id) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (ctx->readonly) return TE_ERROR_READONLY;

    (void)seq_id;

    // Save for undo
    if (ctx->n_tokens > 0) {
        te_range_t range = {0, (te_pos_t)ctx->n_tokens, seq_id};
        te_record_edit(ctx, TE_OP_DELETE, range, range, ctx->tokens, ctx->n_tokens);
    }

    ctx->n_tokens = 0;
    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    return TE_OK;
}

//
// Undo/Redo support
//

te_error_t te_undo(te_context_t *ctx) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!ctx->history_tail) return TE_OK;

    // Find and remove last operation
    te_edit_op_t *op = ctx->history;
    te_edit_op_t *prev = NULL;

    while (op && op->next) {
        prev = op;
        op = op->next;
    }

    if (!op) return TE_OK;

    // Remove from history
    if (prev) {
        prev->next = NULL;
        ctx->history_tail = prev;
    } else {
        ctx->history = NULL;
        ctx->history_tail = NULL;
    }

    // Temporarily disable history recording (but not mutations)
    bool was_suppress = ctx->suppress_history;
    ctx->suppress_history = true;

    // Reverse the operation
    switch (op->type) {
        case TE_OP_INSERT:
            // Undo insert by deleting
            te_delete_tokens(ctx, op->source);
            break;

        case TE_OP_DELETE:
            // Undo delete by inserting
            te_insert_tokens(ctx, op->source.start, op->source.seq_id,
                            op->tokens, op->n_tokens);
            break;

        case TE_OP_REPLACE:
            // Undo replace by replacing with old tokens
            te_replace_tokens(ctx, op->dest, op->tokens, op->n_tokens);
            break;

        default:
            break;
    }

    ctx->suppress_history = was_suppress;

    // Move to redo stack
    op->next = ctx->redo_stack;
    ctx->redo_stack = op;

    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    return TE_OK;
}

te_error_t te_redo(te_context_t *ctx) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!ctx->redo_stack) return TE_OK;

    te_edit_op_t *op = ctx->redo_stack;
    ctx->redo_stack = op->next;

    // Temporarily disable history recording (but not mutations)
    bool was_suppress = ctx->suppress_history;
    ctx->suppress_history = true;

    // Redo the operation
    switch (op->type) {
        case TE_OP_INSERT:
            te_insert_tokens(ctx, op->source.start, op->source.seq_id,
                            op->tokens, op->n_tokens);
            break;

        case TE_OP_DELETE:
            te_delete_tokens(ctx, op->source);
            break;

        case TE_OP_REPLACE:
            te_replace_tokens(ctx, op->source, op->tokens, op->n_tokens);
            break;

        default:
            break;
    }

    ctx->suppress_history = was_suppress;

    // Move back to history
    op->next = NULL;
    if (ctx->history_tail) {
        ctx->history_tail->next = op;
    } else {
        ctx->history = op;
    }
    ctx->history_tail = op;

    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    return TE_OK;
}

size_t te_get_history_count(te_context_t *ctx) {
    if (!ctx) return 0;

    size_t count = 0;
    te_edit_op_t *op = ctx->history;
    while (op) {
        count++;
        op = op->next;
    }
    return count;
}

void te_clear_history(te_context_t *ctx) {
    if (!ctx) return;
    te_clear_edit_history(&ctx->history);
    te_clear_edit_history(&ctx->redo_stack);
    ctx->history_tail = NULL;
}

//
// Sequence management
//

te_seq_id_t te_create_sequence(te_context_t *ctx) {
    if (!ctx) return -1;

    te_seq_id_t new_id = 0;
    for (size_t i = 0; i < ctx->n_sequences; i++) {
        if (ctx->active_sequences[i] >= new_id) {
            new_id = ctx->active_sequences[i] + 1;
        }
    }

    // Grow array if needed
    if (ctx->n_sequences >= 16) {
        // Would need realloc, simplified for now
        return -1;
    }

    ctx->active_sequences[ctx->n_sequences++] = new_id;
    return new_id;
}

te_error_t te_delete_sequence(te_context_t *ctx, te_seq_id_t seq_id) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    // Find and remove sequence
    for (size_t i = 0; i < ctx->n_sequences; i++) {
        if (ctx->active_sequences[i] == seq_id) {
            // Shift remaining
            for (size_t j = i; j < ctx->n_sequences - 1; j++) {
                ctx->active_sequences[j] = ctx->active_sequences[j + 1];
            }
            ctx->n_sequences--;

            // Clear KV cache for this sequence
            llama_kv_cache_seq_rm(ctx->llama_ctx, seq_id, -1, -1);
            return TE_OK;
        }
    }

    return TE_ERROR_SEQUENCE_NOT_FOUND;
}

te_error_t te_copy_sequence(te_context_t *ctx, te_seq_id_t src, te_seq_id_t dst) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    llama_kv_cache_seq_cp(ctx->llama_ctx, src, dst, -1, -1);
    return TE_OK;
}

te_seq_id_t te_fork_sequence(te_context_t *ctx, te_seq_id_t src) {
    if (!ctx) return -1;

    te_seq_id_t dst = te_create_sequence(ctx);
    if (dst < 0) return -1;

    if (te_copy_sequence(ctx, src, dst) != TE_OK) {
        te_delete_sequence(ctx, dst);
        return -1;
    }

    return dst;
}

//
// KV Cache management
//

te_error_t te_sync_kv_cache(te_context_t *ctx) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!ctx->kv_cache_dirty) return TE_OK;

    // Clear and rebuild KV cache
    llama_kv_cache_clear(ctx->llama_ctx);

    if (ctx->n_tokens > 0) {
        // Create batch for all tokens
        llama_batch batch = llama_batch_init(ctx->n_tokens, 0, 1);
        batch.n_tokens = ctx->n_tokens;

        for (size_t i = 0; i < ctx->n_tokens; i++) {
            batch.token[i] = ctx->tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;  // Default sequence
            batch.logits[i] = (i == ctx->n_tokens - 1) ? 1 : 0;
        }

        int result = llama_decode(ctx->llama_ctx, batch);
        llama_batch_free(batch);

        if (result != 0) {
            return TE_ERROR_KV_CACHE_FULL;
        }

        ctx->logits_valid = true;
    }

    ctx->kv_cache_dirty = false;
    return TE_OK;
}

te_error_t te_invalidate_kv_range(te_context_t *ctx, te_range_t range) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    llama_kv_cache_seq_rm(ctx->llama_ctx, range.seq_id, range.start, range.end);
    ctx->kv_cache_dirty = true;

    return TE_OK;
}

te_error_t te_clear_kv_cache(te_context_t *ctx, te_seq_id_t seq_id) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    if (seq_id < 0) {
        llama_kv_cache_clear(ctx->llama_ctx);
    } else {
        llama_kv_cache_seq_rm(ctx->llama_ctx, seq_id, -1, -1);
    }

    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    return TE_OK;
}

te_error_t te_shift_kv_cache(te_context_t *ctx, te_seq_id_t seq_id, te_pos_t delta) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    llama_kv_cache_seq_add(ctx->llama_ctx, seq_id, 0, -1, delta);
    return TE_OK;
}

//
// Snapshot and restore
//

te_snapshot_t *te_create_snapshot(te_context_t *ctx) {
    if (!ctx) return NULL;

    te_snapshot_t *snapshot = (te_snapshot_t *)calloc(1, sizeof(te_snapshot_t));
    if (!snapshot) return NULL;

    snapshot->n_tokens = ctx->n_tokens;

    if (ctx->n_tokens > 0) {
        snapshot->tokens = (te_token_t *)malloc(ctx->n_tokens * sizeof(te_token_t));
        snapshot->token_info = (te_token_info_t *)malloc(ctx->n_tokens * sizeof(te_token_info_t));

        if (!snapshot->tokens || !snapshot->token_info) {
            free(snapshot->tokens);
            free(snapshot->token_info);
            free(snapshot);
            return NULL;
        }

        memcpy(snapshot->tokens, ctx->tokens, ctx->n_tokens * sizeof(te_token_t));
        memcpy(snapshot->token_info, ctx->token_info, ctx->n_tokens * sizeof(te_token_info_t));
    }

    // Save KV cache state
    snapshot->kv_cache_size = llama_get_state_size(ctx->llama_ctx);
    snapshot->kv_cache_state = malloc(snapshot->kv_cache_size);
    if (snapshot->kv_cache_state) {
        llama_copy_state_data(ctx->llama_ctx, (uint8_t *)snapshot->kv_cache_state);
    }

    return snapshot;
}

te_error_t te_restore_snapshot(te_context_t *ctx, te_snapshot_t *snapshot) {
    if (!ctx || !snapshot) return TE_ERROR_INVALID_CONTEXT;

    te_error_t err = te_ensure_capacity(ctx, snapshot->n_tokens);
    if (err != TE_OK) return err;

    ctx->n_tokens = snapshot->n_tokens;

    if (snapshot->n_tokens > 0) {
        memcpy(ctx->tokens, snapshot->tokens, snapshot->n_tokens * sizeof(te_token_t));
        memcpy(ctx->token_info, snapshot->token_info, snapshot->n_tokens * sizeof(te_token_info_t));
    }

    // Restore KV cache state
    if (snapshot->kv_cache_state) {
        llama_set_state_data(ctx->llama_ctx, (uint8_t *)snapshot->kv_cache_state);
        ctx->kv_cache_dirty = false;
        ctx->logits_valid = true;
    } else {
        ctx->kv_cache_dirty = true;
        ctx->logits_valid = false;
    }

    return TE_OK;
}

void te_free_snapshot(te_snapshot_t *snapshot) {
    if (!snapshot) return;

    free(snapshot->tokens);
    free(snapshot->token_info);
    free(snapshot->kv_cache_state);
    free(snapshot);
}

//
// Logits and probabilities
//

te_error_t te_compute_logits(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;

    // Sync KV cache if dirty
    if (ctx->kv_cache_dirty) {
        te_error_t err = te_sync_kv_cache(ctx);
        if (err != TE_OK) return err;
    }

    (void)pos;
    (void)seq_id;

    // Logits should now be valid after sync
    return TE_OK;
}

te_error_t te_get_top_k(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id,
                         te_token_info_t *out, size_t k) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!out || k == 0) return TE_ERROR_BUFFER_TOO_SMALL;

    te_error_t err = te_compute_logits(ctx, pos, seq_id);
    if (err != TE_OK) return err;

    float *logits = llama_get_logits(ctx->llama_ctx);
    int32_t vocab_size = llama_n_vocab(ctx->llama_model);

    // Create candidate array
    llama_token_data *candidates = (llama_token_data *)malloc(vocab_size * sizeof(llama_token_data));
    if (!candidates) return TE_ERROR_ALLOCATION_FAILED;

    for (int32_t i = 0; i < vocab_size; i++) {
        candidates[i].id = i;
        candidates[i].logit = logits[i];
        candidates[i].p = 0.0f;
    }

    // Sort by logit (simple selection for top-k)
    for (size_t i = 0; i < k && i < (size_t)vocab_size; i++) {
        size_t max_idx = i;
        for (size_t j = i + 1; j < (size_t)vocab_size; j++) {
            if (candidates[j].logit > candidates[max_idx].logit) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            llama_token_data tmp = candidates[i];
            candidates[i] = candidates[max_idx];
            candidates[max_idx] = tmp;
        }

        // Fill output
        out[i].id = candidates[i].id;
        out[i].pos = pos;
        out[i].seq_id = seq_id;
        out[i].logit = candidates[i].logit;
        out[i].prob = 0.0f;  // Would need softmax for proper probability
        out[i].has_logit = true;
        out[i].flags = te_get_token_flags(ctx, candidates[i].id);
    }

    free(candidates);
    return TE_OK;
}

float te_get_token_logit(te_context_t *ctx, te_pos_t pos, te_seq_id_t seq_id, te_token_t token) {
    if (!ctx) return -1.0f;

    te_compute_logits(ctx, pos, seq_id);

    float *logits = llama_get_logits(ctx->llama_ctx);
    int32_t vocab_size = llama_n_vocab(ctx->llama_model);

    if (token < 0 || token >= vocab_size) {
        return -1.0f;
    }

    return logits[token];
}

//
// Search and navigation
//

te_error_t te_find_token(te_context_t *ctx, te_token_t token, te_seq_id_t seq_id,
                          te_pos_t *positions, size_t *n_positions) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!positions || !n_positions) return TE_ERROR_BUFFER_TOO_SMALL;

    (void)seq_id;

    size_t found = 0;
    size_t max = *n_positions;

    for (size_t i = 0; i < ctx->n_tokens && found < max; i++) {
        if (ctx->tokens[i] == token) {
            positions[found++] = i;
        }
    }

    *n_positions = found;
    return TE_OK;
}

te_error_t te_find_text(te_context_t *ctx, const char *text, te_seq_id_t seq_id,
                         te_pos_t *positions, size_t *n_positions) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!text || !positions || !n_positions) return TE_ERROR_BUFFER_TOO_SMALL;

    // Tokenize the search text
    size_t text_len = strlen(text);
    size_t max_tokens = text_len + 1;
    te_token_t *search_tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    if (!search_tokens) return TE_ERROR_ALLOCATION_FAILED;

    size_t n_search = max_tokens;
    te_error_t err = te_tokenize(ctx, text, text_len, search_tokens, &n_search, false);
    if (err != TE_OK || n_search == 0) {
        free(search_tokens);
        return err;
    }

    (void)seq_id;

    // Search for token sequence
    size_t found = 0;
    size_t max = *n_positions;

    for (size_t i = 0; i <= ctx->n_tokens - n_search && found < max; i++) {
        bool match = true;
        for (size_t j = 0; j < n_search && match; j++) {
            if (ctx->tokens[i + j] != search_tokens[j]) {
                match = false;
            }
        }
        if (match) {
            positions[found++] = i;
        }
    }

    free(search_tokens);
    *n_positions = found;
    return TE_OK;
}

//
// Export and import
//

te_error_t te_export_json(te_context_t *ctx, te_seq_id_t seq_id, char *buf, size_t *buf_size) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!buf || !buf_size) return TE_ERROR_BUFFER_TOO_SMALL;

    (void)seq_id;

    // Simple JSON format: {"tokens": [1, 2, 3], "text": "..."}
    size_t needed = 32 + ctx->n_tokens * 12;  // Rough estimate

    if (*buf_size < needed) {
        *buf_size = needed;
        return TE_ERROR_BUFFER_TOO_SMALL;
    }

    int offset = snprintf(buf, *buf_size, "{\"tokens\":[");

    for (size_t i = 0; i < ctx->n_tokens; i++) {
        if (i > 0) {
            offset += snprintf(buf + offset, *buf_size - offset, ",");
        }
        offset += snprintf(buf + offset, *buf_size - offset, "%d", ctx->tokens[i]);
    }

    offset += snprintf(buf + offset, *buf_size - offset, "]}");

    *buf_size = offset;
    return TE_OK;
}

te_error_t te_export_binary(te_context_t *ctx, te_seq_id_t seq_id, void *buf, size_t *buf_size) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!buf || !buf_size) return TE_ERROR_BUFFER_TOO_SMALL;

    (void)seq_id;

    size_t needed = sizeof(uint32_t) + ctx->n_tokens * sizeof(te_token_t);
    if (*buf_size < needed) {
        *buf_size = needed;
        return TE_ERROR_BUFFER_TOO_SMALL;
    }

    uint8_t *ptr = (uint8_t *)buf;

    // Write count
    uint32_t count = ctx->n_tokens;
    memcpy(ptr, &count, sizeof(count));
    ptr += sizeof(count);

    // Write tokens
    memcpy(ptr, ctx->tokens, ctx->n_tokens * sizeof(te_token_t));

    *buf_size = needed;
    return TE_OK;
}

te_error_t te_import_binary(te_context_t *ctx, te_seq_id_t seq_id, const void *buf, size_t buf_size) {
    if (!ctx) return TE_ERROR_INVALID_CONTEXT;
    if (!buf || buf_size < sizeof(uint32_t)) return TE_ERROR_BUFFER_TOO_SMALL;

    const uint8_t *ptr = (const uint8_t *)buf;

    // Read count
    uint32_t count;
    memcpy(&count, ptr, sizeof(count));
    ptr += sizeof(count);

    size_t expected = sizeof(uint32_t) + count * sizeof(te_token_t);
    if (buf_size < expected) {
        return TE_ERROR_BUFFER_TOO_SMALL;
    }

    // Clear and load tokens
    te_clear(ctx, seq_id);

    te_error_t err = te_ensure_capacity(ctx, count);
    if (err != TE_OK) return err;

    memcpy(ctx->tokens, ptr, count * sizeof(te_token_t));
    ctx->n_tokens = count;

    // Initialize token info
    for (size_t i = 0; i < count; i++) {
        ctx->token_info[i].id = ctx->tokens[i];
        ctx->token_info[i].pos = i;
        ctx->token_info[i].seq_id = seq_id >= 0 ? seq_id : 0;
        ctx->token_info[i].flags = te_get_token_flags(ctx, ctx->tokens[i]);
        ctx->token_info[i].has_logit = false;
    }

    ctx->kv_cache_dirty = true;
    ctx->logits_valid = false;

    return TE_OK;
}
