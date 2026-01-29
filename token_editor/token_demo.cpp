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

//
// Token Editor Demo - Demonstrates Token Manipulation Capabilities
//

#include "token_editor.h"

#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.cpp/llama.h"

#define DEMO_STEP(msg) printf("\n=== %s ===\n", msg)

static void print_tokens(te_context_t *ctx, const char *label) {
    size_t n = te_get_token_count(ctx, 0);
    printf("%s (%zu tokens):\n", label, n);

    char piece[256];
    for (size_t i = 0; i < n && i < 20; i++) {  // Limit display
        te_token_t tok = te_get_token(ctx, i, 0);
        te_token_to_string(ctx, tok, piece, sizeof(piece));
        printf("  [%2zu] %6d: '%s'\n", i, tok, piece);
    }
    if (n > 20) {
        printf("  ... (%zu more tokens)\n", n - 20);
    }
}

static void print_text(te_context_t *ctx, const char *label) {
    size_t n = te_get_token_count(ctx, 0);
    if (n == 0) {
        printf("%s: (empty)\n", label);
        return;
    }

    te_token_t *tokens = (te_token_t *)malloc(n * sizeof(te_token_t));
    if (!tokens) return;

    te_range_t range = {0, (te_pos_t)n, 0};
    size_t count = n;
    te_get_tokens(ctx, range, tokens, &count);

    size_t text_size = n * 8;
    char *text = (char *)malloc(text_size);
    if (text) {
        if (te_detokenize(ctx, tokens, count, text, &text_size) == TE_OK) {
            printf("%s: \"%.*s\"\n", label, (int)text_size, text);
        }
        free(text);
    }
    free(tokens);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        fprintf(stderr, "\nThis demo shows token editor capabilities:\n");
        fprintf(stderr, "  - Tokenization and detokenization\n");
        fprintf(stderr, "  - Token insertion, deletion, replacement\n");
        fprintf(stderr, "  - Undo/redo operations\n");
        fprintf(stderr, "  - Snapshot and restore\n");
        fprintf(stderr, "  - KV cache management\n");
        return 1;
    }

    const char *model_path = argv[1];

    // Initialize
    printf("Token Editor Demo\n");
    printf("==================\n");
    printf("Loading model: %s\n", model_path);

    llama_backend_init(false);

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;

    llama_context *llama_ctx = llama_new_context_with_model(model, cparams);
    if (!llama_ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    te_context_t *ctx = te_init(llama_ctx, model);
    if (!ctx) {
        fprintf(stderr, "Failed to create token editor\n");
        llama_free(llama_ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    printf("\nModel loaded successfully!\n");
    printf("Vocabulary size: %d\n", llama_n_vocab(model));

    // Demo 1: Basic tokenization
    DEMO_STEP("Demo 1: Tokenization");

    const char *text1 = "Hello, world! This is a test.";
    printf("Input text: \"%s\"\n", text1);

    size_t max_tokens = strlen(text1) + 1;
    te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    size_t n_tokens = max_tokens;

    if (te_tokenize(ctx, text1, strlen(text1), tokens, &n_tokens, true) == TE_OK) {
        printf("Tokenized to %zu tokens\n", n_tokens);

        // Insert tokens into context
        te_insert_tokens(ctx, 0, 0, tokens, n_tokens);
        print_tokens(ctx, "Context");
    }
    free(tokens);

    // Demo 2: Token inspection
    DEMO_STEP("Demo 2: Token Inspection");

    for (te_pos_t i = 0; i < 5 && (size_t)i < te_get_token_count(ctx, 0); i++) {
        te_token_info_t info;
        if (te_get_token_info(ctx, i, 0, &info) == TE_OK) {
            char piece[256];
            te_token_to_string(ctx, info.id, piece, sizeof(piece));
            printf("Token %d: id=%d, flags=0x%x, piece='%s'\n",
                   i, info.id, info.flags, piece);
        }
    }

    // Demo 3: Token modification
    DEMO_STEP("Demo 3: Token Modification");

    // Get original text
    print_text(ctx, "Before modification");

    // Insert text in the middle
    const char *insert_text = " [INSERTED] ";
    size_t insert_pos = te_get_token_count(ctx, 0) / 2;

    max_tokens = strlen(insert_text) + 1;
    tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
    n_tokens = max_tokens;

    if (te_tokenize(ctx, insert_text, strlen(insert_text), tokens, &n_tokens, false) == TE_OK) {
        te_insert_tokens(ctx, insert_pos, 0, tokens, n_tokens);
        printf("Inserted %zu tokens at position %zu\n", n_tokens, insert_pos);
    }
    free(tokens);

    print_text(ctx, "After insertion");

    // Demo 4: Undo/Redo
    DEMO_STEP("Demo 4: Undo/Redo");

    printf("Undoing insertion...\n");
    te_undo(ctx);
    print_text(ctx, "After undo");

    printf("\nRedoing insertion...\n");
    te_redo(ctx);
    print_text(ctx, "After redo");

    // Demo 5: Delete tokens
    DEMO_STEP("Demo 5: Delete Tokens");

    te_range_t delete_range = {(te_pos_t)insert_pos, (te_pos_t)(insert_pos + n_tokens), 0};
    printf("Deleting tokens [%d-%d)\n", delete_range.start, delete_range.end);
    te_delete_tokens(ctx, delete_range);
    print_text(ctx, "After deletion");

    // Demo 6: Replace tokens
    DEMO_STEP("Demo 6: Replace Tokens");

    te_range_t replace_range = {3, 6, 0};  // Replace a few tokens
    const char *replace_text = " REPLACED ";
    printf("Replacing tokens [%d-%d) with '%s'\n",
           replace_range.start, replace_range.end, replace_text);
    te_replace_text(ctx, replace_range, replace_text, strlen(replace_text));
    print_text(ctx, "After replacement");

    // Demo 7: Snapshot and restore
    DEMO_STEP("Demo 7: Snapshot and Restore");

    print_text(ctx, "Current state");
    printf("\nCreating snapshot...\n");
    te_snapshot_t *snapshot = te_create_snapshot(ctx);

    printf("Clearing context...\n");
    te_clear(ctx, 0);
    print_text(ctx, "After clear");

    printf("\nRestoring snapshot...\n");
    te_restore_snapshot(ctx, snapshot);
    print_text(ctx, "After restore");
    te_free_snapshot(snapshot);

    // Demo 8: Find text
    DEMO_STEP("Demo 8: Find Text");

    const char *search = "test";
    printf("Searching for '%s'...\n", search);

    te_pos_t positions[10];
    size_t n_pos = 10;
    if (te_find_text(ctx, search, 0, positions, &n_pos) == TE_OK) {
        printf("Found %zu occurrence(s):\n", n_pos);
        for (size_t i = 0; i < n_pos; i++) {
            printf("  Position: %d\n", positions[i]);
        }
    }

    // Demo 9: KV Cache sync
    DEMO_STEP("Demo 9: KV Cache Sync");

    printf("Syncing KV cache...\n");
    te_error_t err = te_sync_kv_cache(ctx);
    if (err == TE_OK) {
        printf("KV cache synchronized successfully\n");

        // Get top-k predictions
        te_token_info_t top[5];
        size_t pos = te_get_token_count(ctx, 0) - 1;
        if (te_get_top_k(ctx, pos, 0, top, 5) == TE_OK) {
            printf("\nTop-5 next token predictions:\n");
            char piece[256];
            for (int i = 0; i < 5; i++) {
                te_token_to_string(ctx, top[i].id, piece, sizeof(piece));
                printf("  %d. [%6d] logit=%.4f '%s'\n",
                       i + 1, top[i].id, top[i].logit, piece);
            }
        }
    } else {
        printf("KV cache sync returned: %d\n", err);
    }

    // Demo 10: Export to JSON
    DEMO_STEP("Demo 10: Export to JSON");

    size_t json_size = 10240;
    char *json = (char *)malloc(json_size);
    if (json) {
        if (te_export_json(ctx, 0, json, &json_size) == TE_OK) {
            printf("Exported JSON (%zu bytes):\n", json_size);
            printf("%.*s\n", (int)(json_size > 200 ? 200 : json_size), json);
            if (json_size > 200) printf("...\n");
        }
        free(json);
    }

    // Demo 11: History
    DEMO_STEP("Demo 11: Edit History");

    printf("History entries: %zu\n", te_get_history_count(ctx));
    printf("Multiple undos:\n");

    for (int i = 0; i < 3; i++) {
        te_undo(ctx);
        printf("  Undo %d: %zu tokens\n", i + 1, te_get_token_count(ctx, 0));
    }

    printf("\nMultiple redos:\n");
    for (int i = 0; i < 3; i++) {
        te_redo(ctx);
        printf("  Redo %d: %zu tokens\n", i + 1, te_get_token_count(ctx, 0));
    }

    // Cleanup
    DEMO_STEP("Cleanup");
    printf("Freeing resources...\n");

    te_free(ctx);
    llama_free(llama_ctx);
    llama_free_model(model);
    llama_backend_free();

    printf("\nDemo completed successfully!\n");
    return 0;
}
