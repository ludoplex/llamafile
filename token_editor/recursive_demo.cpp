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
// Recursive LLM Demo - Demonstrates Hierarchical Context Management
//
// This demo shows the recursive LLM environment capabilities:
// - Creating root and child contexts
// - Forking contexts
// - Inter-context communication
// - Recursive self-evaluation
// - Tree traversal and management
//

#include "recursive_llm.h"
#include "token_editor.h"

#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.cpp/llama.h"

#define DEMO_STEP(msg) printf("\n=== %s ===\n", msg)

// Callback to track token generation
static void on_token_callback(rllm_context_t *ctx, te_token_t token) {
    (void)ctx;
    (void)token;
    printf(".");
    fflush(stdout);
}

// Callback for context creation
static void on_context_create(rllm_env_t *env, rllm_context_t *ctx) {
    (void)env;
    printf("[Event] Context %u created (depth %u, relation: %s)\n",
           ctx->id, rllm_get_depth(ctx), rllm_relation_str(ctx->relation));
}

// Callback for recursion
static void on_recursion(rllm_env_t *env, rllm_context_t *parent, rllm_context_t *child) {
    (void)env;
    printf("[Event] Recursion: context %u spawned child %u\n",
           parent->id, child->id);
}

// Refinement callback - stop when output is satisfactory
static bool should_continue_refining(const char *output, void *user_data) {
    int *iteration = (int *)user_data;
    (*iteration)++;
    printf("  Iteration %d: output length = %zu\n", *iteration, strlen(output));
    // Stop after 3 iterations for demo
    return *iteration < 3;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        fprintf(stderr, "\nThis demo shows recursive LLM capabilities:\n");
        fprintf(stderr, "  - Creating hierarchical context trees\n");
        fprintf(stderr, "  - Spawning child contexts\n");
        fprintf(stderr, "  - Context forking and cloning\n");
        fprintf(stderr, "  - Inter-context messaging\n");
        fprintf(stderr, "  - Recursive self-evaluation\n");
        fprintf(stderr, "  - Tree traversal operations\n");
        return 1;
    }

    const char *model_path = argv[1];

    // Initialize
    printf("Recursive LLM Environment Demo\n");
    printf("================================\n");
    printf("Loading model: %s\n", model_path);

    llama_backend_init(false);

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    printf("Model loaded successfully!\n");

    // Demo 1: Create environment
    DEMO_STEP("Demo 1: Environment Initialization");

    rllm_env_config_t env_config = rllm_default_env_config();
    env_config.max_depth = 8;
    env_config.max_contexts = 32;
    env_config.enable_logging = true;

    rllm_env_t *env = rllm_init(model, env_config);
    if (!env) {
        fprintf(stderr, "Failed to create environment\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Set callbacks
    env->on_context_create = on_context_create;
    env->on_recursion = on_recursion;

    printf("Environment created with:\n");
    printf("  Max depth: %u\n", env_config.max_depth);
    printf("  Max contexts: %u\n", env_config.max_contexts);

    // Demo 2: Create root context
    DEMO_STEP("Demo 2: Create Root Context");

    rllm_ctx_config_t ctx_config = rllm_default_ctx_config();
    ctx_config.n_ctx = 2048;
    ctx_config.n_batch = 512;
    ctx_config.n_threads = 4;

    rllm_context_t *root = rllm_create_root(env, ctx_config);
    if (!root) {
        fprintf(stderr, "Failed to create root context\n");
        rllm_shutdown(env);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    printf("Root context created: ID=%u\n", root->id);

    // Demo 3: Set prompt and get token editor
    DEMO_STEP("Demo 3: Prompt and Token Editor");

    const char *prompt = "The quick brown fox jumps over the lazy dog.";
    printf("Setting prompt: \"%s\"\n", prompt);

    rllm_set_prompt(root, prompt, strlen(prompt));

    te_context_t *te = rllm_get_token_editor(root);
    if (te) {
        printf("Token count: %zu\n", te_get_token_count(te, 0));
    }

    // Demo 4: Spawn child contexts
    DEMO_STEP("Demo 4: Spawn Child Contexts");

    rllm_context_t *child1 = rllm_spawn_child(env, root, ctx_config);
    rllm_context_t *child2 = rllm_spawn_child(env, root, ctx_config);

    if (child1) {
        printf("Child 1: ID=%u, depth=%u\n", child1->id, rllm_get_depth(child1));
        rllm_set_prompt(child1, "Child 1: Analyzing the text...", 30);
    }

    if (child2) {
        printf("Child 2: ID=%u, depth=%u\n", child2->id, rllm_get_depth(child2));
        rllm_set_prompt(child2, "Child 2: Processing data...", 28);
    }

    // Demo 5: Create grandchildren
    DEMO_STEP("Demo 5: Create Grandchildren (Depth Test)");

    rllm_context_t *grandchild = NULL;
    if (child1) {
        grandchild = rllm_spawn_child(env, child1, ctx_config);
        if (grandchild) {
            printf("Grandchild: ID=%u, depth=%u\n",
                   grandchild->id, rllm_get_depth(grandchild));
            rllm_set_prompt(grandchild, "Grandchild: Deep analysis...", 28);
        }
    }

    // Demo 6: Print context tree
    DEMO_STEP("Demo 6: Context Tree");

    rllm_print_tree(root);

    // Demo 7: Fork context
    DEMO_STEP("Demo 7: Fork Context");

    rllm_context_t *forked = rllm_fork(env, child1);
    if (forked) {
        printf("Forked context: ID=%u (from child1 ID=%u)\n",
               forked->id, child1->id);
        printf("Fork relation: %s\n", rllm_relation_str(forked->relation));
    }

    rllm_print_tree(root);

    // Demo 8: Create peer context
    DEMO_STEP("Demo 8: Create Peer Context");

    rllm_context_t *peer = rllm_create_peer(env, child2, ctx_config);
    if (peer) {
        printf("Peer context: ID=%u (peer of child2 ID=%u)\n",
               peer->id, child2->id);
        printf("Peer relation: %s\n", rllm_relation_str(peer->relation));
    }

    // Demo 9: Tree traversal
    DEMO_STEP("Demo 9: Tree Traversal");

    size_t descendants = rllm_count_descendants(root);
    printf("Root has %zu descendants\n", descendants);

    // Find root from any context
    if (grandchild) {
        rllm_context_t *found_root = rllm_get_root(grandchild);
        printf("Found root from grandchild: ID=%u (expected %u)\n",
               found_root->id, root->id);
    }

    // Get children
    size_t n_children;
    rllm_context_t **children = rllm_get_children(root, &n_children);
    printf("Root has %zu direct children:\n", n_children);
    for (size_t i = 0; i < n_children; i++) {
        printf("  Child %zu: ID=%u, state=%s\n",
               i, children[i]->id, rllm_state_str(children[i]->state));
    }

    // Demo 10: Inter-context messaging
    DEMO_STEP("Demo 10: Inter-Context Messaging");

    if (child1 && child2) {
        const char *msg_text = "Hello from child1!";
        printf("Sending message from child1 to child2: \"%s\"\n", msg_text);

        rllm_error_t err = rllm_send_text(child1, child2, msg_text, strlen(msg_text));
        if (err == RLLM_OK) {
            printf("Message sent successfully\n");

            if (rllm_has_messages(child2)) {
                printf("Child2 has pending messages\n");

                rllm_message_t msg;
                if (rllm_recv_message(child2, &msg, 1000) == RLLM_OK) {
                    printf("Received message type: %d, from: %u, size: %zu\n",
                           msg.type, msg.sender, msg.data_size);
                    if (msg.type == RLLM_MSG_TEXT && msg.data) {
                        printf("Content: \"%.*s\"\n", (int)msg.data_size, (char *)msg.data);
                    }
                    rllm_free_message(&msg);
                }
            }
        }
    }

    // Demo 11: Token-level messaging
    DEMO_STEP("Demo 11: Token Messaging");

    if (child1 && grandchild) {
        te_context_t *te_child1 = rllm_get_token_editor(child1);
        if (te_child1 && te_get_token_count(te_child1, 0) > 0) {
            te_token_t tokens[5];
            te_range_t range = {0, 5, 0};
            size_t count = 5;

            if (te_get_tokens(te_child1, range, tokens, &count) == TE_OK) {
                printf("Sending %zu tokens from child1 to grandchild\n", count);
                rllm_send_tokens(child1, grandchild, tokens, count);
            }
        }
    }

    // Demo 12: Context state management
    DEMO_STEP("Demo 12: Context State Management");

    if (child1) {
        printf("Child1 current state: %s\n", rllm_state_str(child1->state));

        child1->on_token = on_token_callback;

        printf("Setting completion callback and running brief completion...\n");

        rllm_completion_params_t params = rllm_default_completion_params();
        params.n_predict = 10;  // Just a few tokens for demo
        params.timeout_ms = 5000;

        char result[1024];
        size_t result_len = sizeof(result);

        // Note: This would actually run inference if we had a proper model loaded
        printf("(Completion would run here with loaded model)\n");

        printf("Child1 state after completion: %s\n", rllm_state_str(child1->state));
    }

    // Demo 13: Self-evaluation pattern
    DEMO_STEP("Demo 13: Self-Evaluation Pattern");

    if (root) {
        printf("Demonstrating self-evaluation pattern...\n");

        const char *eval_prompt = "Rate the coherence of the above text on a scale of 1-10:";

        char result[2048];
        size_t result_len = sizeof(result);

        // Note: This demonstrates the API, actual inference depends on model
        printf("Self-evaluation prompt: \"%s\"\n", eval_prompt);
        printf("(Self-evaluation would run here with loaded model)\n");
    }

    // Demo 14: Refinement pattern
    DEMO_STEP("Demo 14: Iterative Refinement Pattern");

    if (root) {
        printf("Demonstrating iterative refinement pattern...\n");

        const char *refine_prompt = "Please improve the above response:";
        int iteration = 0;

        char result[4096];
        size_t result_len = sizeof(result);

        printf("Refinement prompt: \"%s\"\n", refine_prompt);
        printf("(Refinement would run here with loaded model)\n");

        // Show the callback pattern
        printf("Simulating refinement iterations:\n");
        char dummy_output[] = "Sample output text";
        while (should_continue_refining(dummy_output, &iteration)) {
            // Callback shows iteration
        }
        printf("Refinement stopped after %d iterations\n", iteration);
    }

    // Demo 15: Statistics
    DEMO_STEP("Demo 15: Environment Statistics");

    uint64_t total_tokens, total_contexts, peak_depth;
    rllm_get_stats(env, &total_tokens, &total_contexts, &peak_depth);

    printf("Environment statistics:\n");
    printf("  Total tokens processed: %lu\n", (unsigned long)total_tokens);
    printf("  Total contexts created: %lu\n", (unsigned long)total_contexts);
    printf("  Peak recursion depth: %lu\n", (unsigned long)peak_depth);
    printf("  Current active contexts: %zu\n", env->n_contexts);

    // Demo 16: Cleanup - destroy specific context
    DEMO_STEP("Demo 16: Cleanup Demonstration");

    printf("Destroying grandchild context...\n");
    if (grandchild) {
        rllm_destroy(env, grandchild);
        printf("Grandchild destroyed\n");
    }

    printf("\nTree after grandchild removal:\n");
    rllm_print_tree(root);

    printf("\nDestroying forked context...\n");
    if (forked) {
        rllm_destroy(env, forked);
        printf("Forked context destroyed\n");
    }

    printf("\nFinal tree:\n");
    rllm_print_tree(root);

    // Shutdown
    DEMO_STEP("Shutdown");

    printf("Shutting down environment...\n");
    rllm_shutdown(env);

    printf("Freeing model...\n");
    llama_free_model(model);

    printf("Backend cleanup...\n");
    llama_backend_free();

    printf("\nDemo completed successfully!\n");
    printf("\nKey takeaways:\n");
    printf("  - Recursive LLM allows hierarchical context management\n");
    printf("  - Contexts can spawn children, fork, and create peers\n");
    printf("  - Inter-context communication via messages\n");
    printf("  - Tree operations for managing context hierarchies\n");
    printf("  - Patterns: self-evaluation, refinement, fan-out\n");

    return 0;
}
