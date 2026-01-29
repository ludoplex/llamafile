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
// Token Editor Unit Tests
//
// These tests verify error handling and basic functionality
// without requiring a model to be loaded.
//

#include "token_editor.h"
#include "recursive_llm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        printf("  Testing: %s... ", #name); \
        tests_run++; \
        test_##name(); \
        tests_passed++; \
        printf("PASSED\n"); \
    } \
    static void test_##name(void)

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("FAILED\n"); \
            printf("    Assertion failed: %s\n", #cond); \
            printf("    At: %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            printf("FAILED\n"); \
            printf("    Expected: %d, Got: %d\n", (int)(b), (int)(a)); \
            printf("    At: %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

//
// Token Editor Tests
//

TEST(te_init_null_context) {
    // te_init should return NULL for NULL inputs
    te_context_t *ctx = te_init(NULL, NULL);
    ASSERT(ctx == NULL);
}

TEST(te_free_null_safe) {
    // te_free should handle NULL gracefully
    te_free(NULL);
    // If we get here without crashing, the test passes
}

TEST(te_get_token_null_context) {
    // te_get_token should return -1 for NULL context
    te_token_t token = te_get_token(NULL, 0, 0);
    ASSERT_EQ(token, -1);
}

TEST(te_get_token_info_null_context) {
    te_token_info_t info;
    te_error_t err = te_get_token_info(NULL, 0, 0, &info);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_token_to_string_null_context) {
    char buf[64];
    int result = te_token_to_string(NULL, 0, buf, sizeof(buf));
    ASSERT_EQ(result, -1);
}

TEST(te_token_to_string_null_buffer) {
    // Even with valid context (if we had one), NULL buffer should fail
    int result = te_token_to_string(NULL, 0, NULL, 64);
    ASSERT_EQ(result, -1);
}

TEST(te_get_tokens_null_context) {
    te_range_t range = {0, 10, 0};
    te_token_t tokens[10];
    size_t n = 10;
    te_error_t err = te_get_tokens(NULL, range, tokens, &n);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_get_token_count_null_context) {
    size_t count = te_get_token_count(NULL, 0);
    ASSERT_EQ(count, 0);
}

TEST(te_tokenize_null_context) {
    te_token_t tokens[64];
    size_t n = 64;
    te_error_t err = te_tokenize(NULL, "hello", 5, tokens, &n, false);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_detokenize_null_context) {
    te_token_t tokens[1] = {0};
    char buf[64];
    size_t len = 64;
    te_error_t err = te_detokenize(NULL, tokens, 1, buf, &len);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_set_token_null_context) {
    te_error_t err = te_set_token(NULL, 0, 0, 1);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_insert_tokens_null_context) {
    te_token_t tokens[1] = {1};
    te_error_t err = te_insert_tokens(NULL, 0, 0, tokens, 1);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_delete_tokens_null_context) {
    te_range_t range = {0, 5, 0};
    te_error_t err = te_delete_tokens(NULL, range);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_replace_tokens_null_context) {
    te_range_t range = {0, 5, 0};
    te_token_t tokens[1] = {1};
    te_error_t err = te_replace_tokens(NULL, range, tokens, 1);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_clear_null_context) {
    te_error_t err = te_clear(NULL, 0);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_undo_null_context) {
    te_error_t err = te_undo(NULL);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_redo_null_context) {
    te_error_t err = te_redo(NULL);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_get_history_count_null_context) {
    size_t count = te_get_history_count(NULL);
    ASSERT_EQ(count, 0);
}

TEST(te_clear_history_null_safe) {
    te_clear_history(NULL);
    // Should not crash
}

TEST(te_create_sequence_null_context) {
    te_seq_id_t seq = te_create_sequence(NULL);
    ASSERT_EQ(seq, -1);
}

TEST(te_delete_sequence_null_context) {
    te_error_t err = te_delete_sequence(NULL, 0);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_copy_sequence_null_context) {
    te_error_t err = te_copy_sequence(NULL, 0, 1);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_fork_sequence_null_context) {
    te_seq_id_t seq = te_fork_sequence(NULL, 0);
    ASSERT_EQ(seq, -1);
}

TEST(te_sync_kv_cache_null_context) {
    te_error_t err = te_sync_kv_cache(NULL);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_create_snapshot_null_context) {
    te_snapshot_t *snap = te_create_snapshot(NULL);
    ASSERT(snap == NULL);
}

TEST(te_restore_snapshot_null_context) {
    te_error_t err = te_restore_snapshot(NULL, NULL);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_free_snapshot_null_safe) {
    te_free_snapshot(NULL);
    // Should not crash
}

TEST(te_find_token_null_context) {
    te_pos_t positions[10];
    size_t n = 10;
    te_error_t err = te_find_token(NULL, 1, 0, positions, &n);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_find_text_null_context) {
    te_pos_t positions[10];
    size_t n = 10;
    te_error_t err = te_find_text(NULL, "test", 0, positions, &n);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_export_json_null_context) {
    char buf[1024];
    size_t size = sizeof(buf);
    te_error_t err = te_export_json(NULL, 0, buf, &size);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_export_binary_null_context) {
    char buf[1024];
    size_t size = sizeof(buf);
    te_error_t err = te_export_binary(NULL, 0, buf, &size);
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

TEST(te_import_binary_null_context) {
    char buf[1024] = {0};
    te_error_t err = te_import_binary(NULL, 0, buf, sizeof(buf));
    ASSERT_EQ(err, TE_ERROR_INVALID_CONTEXT);
}

//
// Recursive LLM Tests
//

TEST(rllm_default_env_config) {
    rllm_env_config_t config = rllm_default_env_config();
    ASSERT_EQ(config.max_depth, RLLM_MAX_DEPTH);
    ASSERT_EQ(config.max_contexts, RLLM_MAX_CONTEXTS);
    ASSERT(config.default_n_ctx > 0);
    ASSERT(config.default_n_batch > 0);
}

TEST(rllm_default_ctx_config) {
    rllm_ctx_config_t config = rllm_default_ctx_config();
    ASSERT(config.n_ctx > 0);
    ASSERT(config.n_batch > 0);
}

TEST(rllm_default_completion_params) {
    rllm_completion_params_t params = rllm_default_completion_params();
    ASSERT(params.n_predict > 0);
    ASSERT(params.temperature >= 0.0f);
}

TEST(rllm_init_null_model) {
    rllm_env_config_t config = rllm_default_env_config();
    rllm_env_t *env = rllm_init(NULL, config);
    ASSERT(env == NULL);
}

TEST(rllm_shutdown_null_safe) {
    rllm_shutdown(NULL);
    // Should not crash
}

TEST(rllm_create_root_null_env) {
    rllm_ctx_config_t config = rllm_default_ctx_config();
    rllm_context_t *ctx = rllm_create_root(NULL, config);
    ASSERT(ctx == NULL);
}

TEST(rllm_spawn_child_null) {
    rllm_ctx_config_t config = rllm_default_ctx_config();
    rllm_context_t *ctx = rllm_spawn_child(NULL, NULL, config);
    ASSERT(ctx == NULL);
}

TEST(rllm_fork_null) {
    rllm_context_t *ctx = rllm_fork(NULL, NULL);
    ASSERT(ctx == NULL);
}

TEST(rllm_create_peer_null) {
    rllm_ctx_config_t config = rllm_default_ctx_config();
    rllm_context_t *ctx = rllm_create_peer(NULL, NULL, config);
    ASSERT(ctx == NULL);
}

TEST(rllm_destroy_null) {
    rllm_error_t err = rllm_destroy(NULL, NULL);
    ASSERT_EQ(err, RLLM_ERROR_INVALID_CONTEXT);
}

TEST(rllm_get_context_null_env) {
    rllm_context_t *ctx = rllm_get_context(NULL, 0);
    ASSERT(ctx == NULL);
}

TEST(rllm_get_parent_null) {
    rllm_context_t *parent = rllm_get_parent(NULL);
    ASSERT(parent == NULL);
}

TEST(rllm_get_children_null) {
    size_t n = 0;
    rllm_context_t **children = rllm_get_children(NULL, &n);
    ASSERT(children == NULL);
    ASSERT_EQ(n, 0);
}

TEST(rllm_get_root_null) {
    rllm_context_t *root = rllm_get_root(NULL);
    ASSERT(root == NULL);
}

TEST(rllm_get_depth_null) {
    uint32_t depth = rllm_get_depth(NULL);
    ASSERT_EQ(depth, 0);
}

TEST(rllm_get_token_editor_null) {
    te_context_t *te = rllm_get_token_editor(NULL);
    ASSERT(te == NULL);
}

TEST(rllm_set_prompt_null) {
    rllm_error_t err = rllm_set_prompt(NULL, "test", 4);
    ASSERT_EQ(err, RLLM_ERROR_INVALID_CONTEXT);
}

TEST(rllm_complete_null) {
    rllm_completion_params_t params = rllm_default_completion_params();
    rllm_error_t err = rllm_complete(NULL, params);
    ASSERT_EQ(err, RLLM_ERROR_INVALID_CONTEXT);
}

TEST(rllm_send_message_null) {
    rllm_message_t msg;
    memset(&msg, 0, sizeof(msg));
    rllm_error_t err = rllm_send_message(NULL, NULL, &msg);
    ASSERT_EQ(err, RLLM_ERROR_INVALID_CONTEXT);
}

TEST(rllm_has_messages_null) {
    bool has = rllm_has_messages(NULL);
    ASSERT(has == false);
}

TEST(rllm_free_message_null_safe) {
    rllm_free_message(NULL);
    // Should not crash
}

TEST(rllm_count_descendants_null) {
    size_t count = rllm_count_descendants(NULL);
    ASSERT_EQ(count, 0);
}

TEST(rllm_state_str) {
    ASSERT(rllm_state_str(RLLM_STATE_IDLE) != NULL);
    ASSERT(rllm_state_str(RLLM_STATE_RUNNING) != NULL);
    ASSERT(rllm_state_str(RLLM_STATE_COMPLETE) != NULL);
}

TEST(rllm_relation_str) {
    ASSERT(rllm_relation_str(RLLM_REL_ROOT) != NULL);
    ASSERT(rllm_relation_str(RLLM_REL_CHILD) != NULL);
    ASSERT(rllm_relation_str(RLLM_REL_FORK) != NULL);
}

//
// Main
//

int main(void) {
    printf("Token Editor Unit Tests\n");
    printf("========================\n\n");

    printf("Token Editor API tests:\n");
    run_test_te_init_null_context();
    run_test_te_free_null_safe();
    run_test_te_get_token_null_context();
    run_test_te_get_token_info_null_context();
    run_test_te_token_to_string_null_context();
    run_test_te_token_to_string_null_buffer();
    run_test_te_get_tokens_null_context();
    run_test_te_get_token_count_null_context();
    run_test_te_tokenize_null_context();
    run_test_te_detokenize_null_context();
    run_test_te_set_token_null_context();
    run_test_te_insert_tokens_null_context();
    run_test_te_delete_tokens_null_context();
    run_test_te_replace_tokens_null_context();
    run_test_te_clear_null_context();
    run_test_te_undo_null_context();
    run_test_te_redo_null_context();
    run_test_te_get_history_count_null_context();
    run_test_te_clear_history_null_safe();
    run_test_te_create_sequence_null_context();
    run_test_te_delete_sequence_null_context();
    run_test_te_copy_sequence_null_context();
    run_test_te_fork_sequence_null_context();
    run_test_te_sync_kv_cache_null_context();
    run_test_te_create_snapshot_null_context();
    run_test_te_restore_snapshot_null_context();
    run_test_te_free_snapshot_null_safe();
    run_test_te_find_token_null_context();
    run_test_te_find_text_null_context();
    run_test_te_export_json_null_context();
    run_test_te_export_binary_null_context();
    run_test_te_import_binary_null_context();

    printf("\nRecursive LLM API tests:\n");
    run_test_rllm_default_env_config();
    run_test_rllm_default_ctx_config();
    run_test_rllm_default_completion_params();
    run_test_rllm_init_null_model();
    run_test_rllm_shutdown_null_safe();
    run_test_rllm_create_root_null_env();
    run_test_rllm_spawn_child_null();
    run_test_rllm_fork_null();
    run_test_rllm_create_peer_null();
    run_test_rllm_destroy_null();
    run_test_rllm_get_context_null_env();
    run_test_rllm_get_parent_null();
    run_test_rllm_get_children_null();
    run_test_rllm_get_root_null();
    run_test_rllm_get_depth_null();
    run_test_rllm_get_token_editor_null();
    run_test_rllm_set_prompt_null();
    run_test_rllm_complete_null();
    run_test_rllm_send_message_null();
    run_test_rllm_has_messages_null();
    run_test_rllm_free_message_null_safe();
    run_test_rllm_count_descendants_null();
    run_test_rllm_state_str();
    run_test_rllm_relation_str();

    printf("\n========================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
