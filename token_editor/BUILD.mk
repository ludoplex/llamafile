#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += TOKEN_EDITOR

TOKEN_EDITOR_FILES := $(wildcard token_editor/*.*)
TOKEN_EDITOR_HDRS = $(filter %.h,$(TOKEN_EDITOR_FILES))
TOKEN_EDITOR_SRCS_C = $(filter %.c,$(TOKEN_EDITOR_FILES))
TOKEN_EDITOR_SRCS_CPP = $(filter %.cpp,$(TOKEN_EDITOR_FILES))

TOKEN_EDITOR_OBJS =						\
	$(TOKEN_EDITOR_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(TOKEN_EDITOR_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

# Token editor static library
o/$(MODE)/token_editor/token_editor.a:				\
		o/$(MODE)/token_editor/token_editor.o		\
		o/$(MODE)/token_editor/recursive_llm.o

# Main CLI demo application
o/$(MODE)/token_editor/token_editor_cli:			\
		o/$(MODE)/token_editor/token_editor_cli.o	\
		o/$(MODE)/token_editor/token_editor.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a

# Token editor demo (interactive token manipulation)
o/$(MODE)/token_editor/token_demo:				\
		o/$(MODE)/token_editor/token_demo.o		\
		o/$(MODE)/token_editor/token_editor.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a

# Recursive LLM demo
o/$(MODE)/token_editor/recursive_demo:				\
		o/$(MODE)/token_editor/recursive_demo.o		\
		o/$(MODE)/token_editor/token_editor.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: o/$(MODE)/token_editor
o/$(MODE)/token_editor:						\
		$(TOKEN_EDITOR_OBJS)				\
		o/$(MODE)/token_editor/token_editor.a		\
		o/$(MODE)/token_editor/token_editor_cli		\
		o/$(MODE)/token_editor/token_demo		\
		o/$(MODE)/token_editor/recursive_demo
