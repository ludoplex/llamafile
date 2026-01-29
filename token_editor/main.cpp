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
// Token Editor CLI - Interactive Token Manipulation Tool
//
// A command-line interface for direct context token editing
// and recursive LLM environment operations.
//

#include "token_editor.h"
#include "recursive_llm.h"

#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <getopt.h>
#include <signal.h>

#include "llama.cpp/llama.h"
#include "llama.cpp/common/common.h"

// ANSI colors for terminal output
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

// Global state for signal handling
static volatile bool g_running = true;
static te_context_t *g_token_editor = NULL;
static rllm_env_t *g_rllm_env = NULL;

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    printf("\n" COLOR_YELLOW "Interrupted. Type 'quit' to exit." COLOR_RESET "\n");
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options] <model_path>\n", prog);
    printf("\nToken Editor CLI - Direct Context Token Manipulation\n");
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -c, --ctx-size N        Context size (default: 2048)\n");
    printf("  -t, --threads N         Number of threads (default: 4)\n");
    printf("  -b, --batch-size N      Batch size (default: 512)\n");
    printf("  -n, --n-gpu-layers N    Number of GPU layers (default: 0)\n");
    printf("  -p, --prompt TEXT       Initial prompt\n");
    printf("  -f, --file PATH         Load prompt from file\n");
    printf("  -i, --interactive       Start in interactive mode\n");
    printf("  -r, --recursive         Enable recursive LLM mode\n");
    printf("  -v, --verbose           Verbose output\n");
    printf("\nInteractive Commands:\n");
    printf("  show                    Show current tokens\n");
    printf("  showtext                Show current text\n");
    printf("  insert <pos> <text>     Insert text at position\n");
    printf("  delete <start> <end>    Delete tokens in range\n");
    printf("  replace <start> <end> <text>  Replace range with text\n");
    printf("  set <pos> <token_id>    Set specific token at position\n");
    printf("  find <text>             Find text in context\n");
    printf("  topk <k>                Show top-k next tokens\n");
    printf("  complete <n>            Generate n tokens\n");
    printf("  undo                    Undo last operation\n");
    printf("  redo                    Redo last undone operation\n");
    printf("  snapshot                Save context snapshot\n");
    printf("  restore                 Restore last snapshot\n");
    printf("  clear                   Clear all tokens\n");
    printf("  export <file>           Export tokens to file\n");
    printf("  import <file>           Import tokens from file\n");
    printf("  spawn                   Spawn child context (recursive mode)\n");
    printf("  tree                    Show context tree (recursive mode)\n");
    printf("  help                    Show this help\n");
    printf("  quit                    Exit the program\n");
}

static void print_tokens(te_context_t *ctx, te_pos_t start, te_pos_t end) {
    if (!ctx) return;

    size_t n_tokens = te_get_token_count(ctx, 0);
    if (start < 0) start = 0;
    if (end < 0 || (size_t)end > n_tokens) end = n_tokens;

    printf(COLOR_CYAN "Tokens [%d-%d] of %zu:" COLOR_RESET "\n", start, end, n_tokens);

    char piece[256];
    for (te_pos_t i = start; i < end; i++) {
        te_token_info_t info;
        if (te_get_token_info(ctx, i, 0, &info) == TE_OK) {
            te_token_to_string(ctx, info.id, piece, sizeof(piece));

            // Escape special characters for display
            char display[512];
            char *d = display;
            for (char *p = piece; *p && d < display + sizeof(display) - 4; p++) {
                if (*p == '\n') { *d++ = '\\'; *d++ = 'n'; }
                else if (*p == '\t') { *d++ = '\\'; *d++ = 't'; }
                else if (*p == '\r') { *d++ = '\\'; *d++ = 'r'; }
                else *d++ = *p;
            }
            *d = '\0';

            // Color based on flags
            const char *color = COLOR_RESET;
            if (info.flags & TE_FLAG_SPECIAL) color = COLOR_MAGENTA;
            else if (info.flags & TE_FLAG_BOS) color = COLOR_GREEN;
            else if (info.flags & TE_FLAG_EOS) color = COLOR_RED;
            else if (info.flags & TE_FLAG_USER_DATA) color = COLOR_YELLOW;

            printf("  %s[%4d]%s %6d: '%s'\n", color, i, COLOR_RESET, info.id, display);
        }
    }
}

static void print_text(te_context_t *ctx) {
    if (!ctx) return;

    size_t n_tokens = te_get_token_count(ctx, 0);
    if (n_tokens == 0) {
        printf(COLOR_YELLOW "(empty)" COLOR_RESET "\n");
        return;
    }

    // Get all tokens
    te_token_t *tokens = (te_token_t *)malloc(n_tokens * sizeof(te_token_t));
    if (!tokens) return;

    te_range_t range = {0, (te_pos_t)n_tokens, 0};
    size_t count = n_tokens;
    if (te_get_tokens(ctx, range, tokens, &count) != TE_OK) {
        free(tokens);
        return;
    }

    // Detokenize
    size_t text_size = n_tokens * 8;
    char *text = (char *)malloc(text_size);
    if (!text) {
        free(tokens);
        return;
    }

    if (te_detokenize(ctx, tokens, count, text, &text_size) == TE_OK) {
        printf(COLOR_CYAN "Text (%zu chars):" COLOR_RESET "\n", text_size);
        printf("%.*s\n", (int)text_size, text);
    }

    free(text);
    free(tokens);
}

static void interactive_loop(te_context_t *ctx, rllm_env_t *env, rllm_context_t *rllm_ctx,
                             struct llama_model *model, bool verbose) {
    char line[4096];
    te_snapshot_t *snapshot = NULL;

    printf(COLOR_BOLD "\nToken Editor Interactive Mode" COLOR_RESET "\n");
    printf("Type 'help' for commands, 'quit' to exit.\n\n");

    while (g_running) {
        printf(COLOR_GREEN "> " COLOR_RESET);
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        // Remove trailing newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
            len--;
        }

        if (len == 0) continue;

        // Parse command
        char cmd[64];
        char arg1[1024];
        char arg2[1024];
        char arg3[2048];
        int n_args = sscanf(line, "%63s %1023s %1023s %2047[^\n]", cmd, arg1, arg2, arg3);

        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0 || strcmp(cmd, "q") == 0) {
            break;
        }
        else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "h") == 0 || strcmp(cmd, "?") == 0) {
            print_usage("token_editor_cli");
        }
        else if (strcmp(cmd, "show") == 0) {
            te_pos_t start = 0, end = -1;
            if (n_args >= 2) start = atoi(arg1);
            if (n_args >= 3) end = atoi(arg2);
            print_tokens(ctx, start, end);
        }
        else if (strcmp(cmd, "showtext") == 0 || strcmp(cmd, "text") == 0) {
            print_text(ctx);
        }
        else if (strcmp(cmd, "insert") == 0) {
            if (n_args < 3) {
                printf(COLOR_RED "Usage: insert <pos> <text>" COLOR_RESET "\n");
                continue;
            }
            te_pos_t pos = atoi(arg1);

            // Reconstruct text from remaining args
            char *text = line + strlen(cmd) + 1 + strlen(arg1) + 1;
            size_t text_len = strlen(text);

            // Tokenize
            size_t max_tokens = text_len + 1;
            te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));
            if (!tokens) {
                printf(COLOR_RED "Memory error" COLOR_RESET "\n");
                continue;
            }

            size_t n_tokens = max_tokens;
            if (te_tokenize(ctx, text, text_len, tokens, &n_tokens, false) != TE_OK) {
                printf(COLOR_RED "Tokenization failed" COLOR_RESET "\n");
                free(tokens);
                continue;
            }

            if (te_insert_tokens(ctx, pos, 0, tokens, n_tokens) == TE_OK) {
                printf(COLOR_GREEN "Inserted %zu tokens at position %d" COLOR_RESET "\n",
                       n_tokens, pos);
                if (verbose) print_tokens(ctx, pos, pos + n_tokens);
            } else {
                printf(COLOR_RED "Insert failed" COLOR_RESET "\n");
            }
            free(tokens);
        }
        else if (strcmp(cmd, "delete") == 0) {
            if (n_args < 3) {
                printf(COLOR_RED "Usage: delete <start> <end>" COLOR_RESET "\n");
                continue;
            }
            te_range_t range = {atoi(arg1), atoi(arg2), 0};
            if (te_delete_tokens(ctx, range) == TE_OK) {
                printf(COLOR_GREEN "Deleted tokens [%d-%d)" COLOR_RESET "\n",
                       range.start, range.end);
            } else {
                printf(COLOR_RED "Delete failed" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "replace") == 0) {
            if (n_args < 4) {
                printf(COLOR_RED "Usage: replace <start> <end> <text>" COLOR_RESET "\n");
                continue;
            }
            te_range_t range = {atoi(arg1), atoi(arg2), 0};
            if (te_replace_text(ctx, range, arg3, strlen(arg3)) == TE_OK) {
                printf(COLOR_GREEN "Replaced tokens [%d-%d)" COLOR_RESET "\n",
                       range.start, range.end);
            } else {
                printf(COLOR_RED "Replace failed" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "set") == 0) {
            if (n_args < 3) {
                printf(COLOR_RED "Usage: set <pos> <token_id>" COLOR_RESET "\n");
                continue;
            }
            te_pos_t pos = atoi(arg1);
            te_token_t token = atoi(arg2);
            if (te_set_token(ctx, pos, 0, token) == TE_OK) {
                printf(COLOR_GREEN "Set token at position %d to %d" COLOR_RESET "\n",
                       pos, token);
            } else {
                printf(COLOR_RED "Set failed" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "find") == 0) {
            if (n_args < 2) {
                printf(COLOR_RED "Usage: find <text>" COLOR_RESET "\n");
                continue;
            }
            char *text = line + strlen(cmd) + 1;
            te_pos_t positions[100];
            size_t n_pos = 100;
            if (te_find_text(ctx, text, 0, positions, &n_pos) == TE_OK) {
                printf(COLOR_CYAN "Found %zu occurrences:" COLOR_RESET "\n", n_pos);
                for (size_t i = 0; i < n_pos; i++) {
                    printf("  Position: %d\n", positions[i]);
                }
            } else {
                printf(COLOR_YELLOW "Not found" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "topk") == 0) {
            int k = 10;
            if (n_args >= 2) k = atoi(arg1);

            te_token_info_t *top = (te_token_info_t *)malloc(k * sizeof(te_token_info_t));
            if (!top) continue;

            size_t n_tokens = te_get_token_count(ctx, 0);
            if (n_tokens == 0) {
                printf(COLOR_YELLOW "No tokens in context" COLOR_RESET "\n");
                free(top);
                continue;
            }

            if (te_get_top_k(ctx, n_tokens - 1, 0, top, k) == TE_OK) {
                printf(COLOR_CYAN "Top-%d next tokens:" COLOR_RESET "\n", k);
                char piece[256];
                for (int i = 0; i < k; i++) {
                    te_token_to_string(ctx, top[i].id, piece, sizeof(piece));
                    printf("  %2d. [%6d] logit: %8.4f  '%s'\n",
                           i + 1, top[i].id, top[i].logit, piece);
                }
            } else {
                printf(COLOR_RED "Failed to compute top-k" COLOR_RESET "\n");
            }
            free(top);
        }
        else if (strcmp(cmd, "complete") == 0) {
            int n = 32;
            if (n_args >= 2) n = atoi(arg1);

            if (rllm_ctx) {
                rllm_completion_params_t params = rllm_default_completion_params();
                params.n_predict = n;

                size_t result_size = n * 16;
                char *result = (char *)malloc(result_size);
                if (result) {
                    if (rllm_complete_sync(rllm_ctx, params, result, &result_size) == RLLM_OK) {
                        printf(COLOR_CYAN "Generated %zu chars:" COLOR_RESET "\n", result_size);
                        printf("%.*s\n", (int)result_size, result);
                    } else {
                        printf(COLOR_RED "Completion failed" COLOR_RESET "\n");
                    }
                    free(result);
                }
            } else {
                printf(COLOR_YELLOW "Recursive mode not enabled. Use -r flag." COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "undo") == 0) {
            if (te_undo(ctx) == TE_OK) {
                printf(COLOR_GREEN "Undone" COLOR_RESET "\n");
            } else {
                printf(COLOR_YELLOW "Nothing to undo" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "redo") == 0) {
            if (te_redo(ctx) == TE_OK) {
                printf(COLOR_GREEN "Redone" COLOR_RESET "\n");
            } else {
                printf(COLOR_YELLOW "Nothing to redo" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "snapshot") == 0) {
            if (snapshot) te_free_snapshot(snapshot);
            snapshot = te_create_snapshot(ctx);
            if (snapshot) {
                printf(COLOR_GREEN "Snapshot saved (%zu tokens)" COLOR_RESET "\n",
                       snapshot->n_tokens);
            } else {
                printf(COLOR_RED "Failed to create snapshot" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "restore") == 0) {
            if (!snapshot) {
                printf(COLOR_YELLOW "No snapshot to restore" COLOR_RESET "\n");
                continue;
            }
            if (te_restore_snapshot(ctx, snapshot) == TE_OK) {
                printf(COLOR_GREEN "Snapshot restored" COLOR_RESET "\n");
            } else {
                printf(COLOR_RED "Failed to restore snapshot" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "clear") == 0) {
            if (te_clear(ctx, 0) == TE_OK) {
                printf(COLOR_GREEN "Cleared all tokens" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "export") == 0) {
            if (n_args < 2) {
                printf(COLOR_RED "Usage: export <file>" COLOR_RESET "\n");
                continue;
            }
            size_t buf_size = 1024 * 1024;  // 1MB
            char *buf = (char *)malloc(buf_size);
            if (!buf) continue;

            if (te_export_json(ctx, 0, buf, &buf_size) == TE_OK) {
                FILE *f = fopen(arg1, "w");
                if (f) {
                    fwrite(buf, 1, buf_size, f);
                    fclose(f);
                    printf(COLOR_GREEN "Exported to %s" COLOR_RESET "\n", arg1);
                } else {
                    printf(COLOR_RED "Failed to open file" COLOR_RESET "\n");
                }
            } else {
                printf(COLOR_RED "Export failed" COLOR_RESET "\n");
            }
            free(buf);
        }
        else if (strcmp(cmd, "spawn") == 0) {
            if (!env) {
                printf(COLOR_YELLOW "Recursive mode not enabled. Use -r flag." COLOR_RESET "\n");
                continue;
            }
            rllm_ctx_config_t config = rllm_default_ctx_config();
            rllm_context_t *child = rllm_spawn_child(env, rllm_ctx, config);
            if (child) {
                printf(COLOR_GREEN "Spawned child context %u (depth %u)" COLOR_RESET "\n",
                       child->id, rllm_get_depth(child));
            } else {
                printf(COLOR_RED "Failed to spawn child" COLOR_RESET "\n");
            }
        }
        else if (strcmp(cmd, "tree") == 0) {
            if (!rllm_ctx) {
                printf(COLOR_YELLOW "Recursive mode not enabled. Use -r flag." COLOR_RESET "\n");
                continue;
            }
            rllm_print_tree(rllm_get_root(rllm_ctx));
        }
        else if (strcmp(cmd, "info") == 0) {
            printf(COLOR_CYAN "Context Info:" COLOR_RESET "\n");
            printf("  Tokens: %zu\n", te_get_token_count(ctx, 0));
            printf("  History entries: %zu\n", te_get_history_count(ctx));
            if (rllm_ctx) {
                printf("  Recursive context: %u (depth %u)\n",
                       rllm_ctx->id, rllm_get_depth(rllm_ctx));
                printf("  Children: %zu\n", rllm_ctx->n_children);
            }
        }
        else {
            printf(COLOR_RED "Unknown command: %s" COLOR_RESET "\n", cmd);
            printf("Type 'help' for available commands.\n");
        }
    }

    if (snapshot) te_free_snapshot(snapshot);
}

int main(int argc, char **argv) {
    // Default parameters
    int n_ctx = 2048;
    int n_threads = 4;
    int n_batch = 512;
    int n_gpu_layers = 0;
    char *prompt = NULL;
    char *prompt_file = NULL;
    char *model_path = NULL;
    bool interactive = false;
    bool recursive = false;
    bool verbose = false;

    // Parse command line
    static struct option long_options[] = {
        {"help",        no_argument,       0, 'h'},
        {"ctx-size",    required_argument, 0, 'c'},
        {"threads",     required_argument, 0, 't'},
        {"batch-size",  required_argument, 0, 'b'},
        {"n-gpu-layers", required_argument, 0, 'n'},
        {"prompt",      required_argument, 0, 'p'},
        {"file",        required_argument, 0, 'f'},
        {"interactive", no_argument,       0, 'i'},
        {"recursive",   no_argument,       0, 'r'},
        {"verbose",     no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "hc:t:b:n:p:f:irv", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'c':
                n_ctx = atoi(optarg);
                break;
            case 't':
                n_threads = atoi(optarg);
                break;
            case 'b':
                n_batch = atoi(optarg);
                break;
            case 'n':
                n_gpu_layers = atoi(optarg);
                break;
            case 'p':
                prompt = optarg;
                break;
            case 'f':
                prompt_file = optarg;
                break;
            case 'i':
                interactive = true;
                break;
            case 'r':
                recursive = true;
                break;
            case 'v':
                verbose = true;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, COLOR_RED "Error: model path required" COLOR_RESET "\n");
        print_usage(argv[0]);
        return 1;
    }

    model_path = argv[optind];

    // Setup signal handler
    signal(SIGINT, signal_handler);

    // Initialize llama backend
    llama_backend_init();

    // Load model
    printf("Loading model: %s\n", model_path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model *model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, COLOR_RED "Failed to load model" COLOR_RESET "\n");
        llama_backend_free();
        return 1;
    }

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    llama_context *llama_ctx = llama_new_context_with_model(model, cparams);
    if (!llama_ctx) {
        fprintf(stderr, COLOR_RED "Failed to create context" COLOR_RESET "\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Create token editor
    te_context_t *te_ctx = te_init(llama_ctx, model);
    if (!te_ctx) {
        fprintf(stderr, COLOR_RED "Failed to create token editor" COLOR_RESET "\n");
        llama_free(llama_ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    g_token_editor = te_ctx;

    // Create recursive LLM environment if requested
    rllm_env_t *env = NULL;
    rllm_context_t *rllm_ctx = NULL;

    if (recursive) {
        rllm_env_config_t env_config = rllm_default_env_config();
        env_config.enable_logging = verbose;

        env = rllm_init(model, env_config);
        if (!env) {
            fprintf(stderr, COLOR_RED "Failed to create recursive environment" COLOR_RESET "\n");
        } else {
            g_rllm_env = env;

            rllm_ctx_config_t ctx_config = rllm_default_ctx_config();
            ctx_config.n_ctx = n_ctx;
            ctx_config.n_batch = n_batch;
            ctx_config.n_threads = n_threads;

            rllm_ctx = rllm_create_root(env, ctx_config);
            if (!rllm_ctx) {
                fprintf(stderr, COLOR_RED "Failed to create root context" COLOR_RESET "\n");
            }
        }
    }

    // Load initial prompt
    if (prompt_file) {
        FILE *f = fopen(prompt_file, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            prompt = (char *)malloc(size + 1);
            if (prompt) {
                size_t read = fread(prompt, 1, size, f);
                prompt[read] = '\0';
            }
            fclose(f);
        } else {
            fprintf(stderr, COLOR_YELLOW "Warning: could not open prompt file: %s" COLOR_RESET "\n",
                    prompt_file);
        }
    }

    if (prompt) {
        size_t prompt_len = strlen(prompt);
        size_t max_tokens = prompt_len + 1;
        te_token_t *tokens = (te_token_t *)malloc(max_tokens * sizeof(te_token_t));

        if (tokens) {
            size_t n_tokens = max_tokens;
            if (te_tokenize(te_ctx, prompt, prompt_len, tokens, &n_tokens, true) == TE_OK) {
                te_insert_tokens(te_ctx, 0, 0, tokens, n_tokens);
                printf("Loaded prompt: %zu tokens\n", n_tokens);

                if (rllm_ctx) {
                    rllm_set_prompt(rllm_ctx, prompt, prompt_len);
                }
            }
            free(tokens);
        }

        if (prompt_file && prompt) {
            free(prompt);
        }
    }

    // Run interactive mode or show initial state
    if (interactive) {
        interactive_loop(te_ctx, env, rllm_ctx, model, verbose);
    } else {
        print_tokens(te_ctx, 0, -1);
        print_text(te_ctx);
    }

    // Cleanup
    if (env) {
        rllm_shutdown(env);
    }

    te_free(te_ctx);
    llama_free(llama_ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
