#!/bin/bash
# Apply llamafile patches to whisper.cpp submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/../whisper.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$WHISPER_DIR"

# Check if patches are already applied
if [ -f "BUILD.mk" ]; then
    echo "Patches appear to be already applied. Skipping..."
    exit 0
fi

echo "Applying patches to whisper.cpp submodule..."

# Step 1: Apply patches to modify files in their original locations
echo "Applying modifications to upstream files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        patch -p0 < "$patch_file"
    fi
done

# Step 2: Copy modified files from their original locations to root
echo "Copying modified files to root directory..."
cp examples/server/server.cpp .
cp examples/common.cpp .
cp examples/common.h .
cp examples/main/main.cpp .
cp examples/stream/stream.cpp .
cp examples/grammar-parser.cpp .
cp examples/grammar-parser.h .
cp examples/dr_wav.h .
cp examples/server/httplib.h .
cp src/whisper.cpp .
cp src/whisper-mel-cuda.cu .
cp include/whisper.h .

# Step 3: Copy new llamafile-specific files to root
echo "Copying llamafile-specific files..."
cp "$LLAMAFILE_FILES_DIR/BUILD.mk" .
cp "$LLAMAFILE_FILES_DIR/README.llamafile" .
cp "$LLAMAFILE_FILES_DIR/README.md" .
# Copy header files, excluding those that were patched
for file in "$LLAMAFILE_FILES_DIR"/*.h; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")
    if [ "$filename" != "common.h" ] && [ "$filename" != "whisper.h" ] && \
       [ "$filename" != "grammar-parser.h" ] && [ "$filename" != "dr_wav.h" ] && \
       [ "$filename" != "httplib.h" ]; then
        cp "$file" .
    fi
done
cp "$LLAMAFILE_FILES_DIR"/*.c . 2>/dev/null || true
# Copy cpp files, excluding those that were patched
for file in "$LLAMAFILE_FILES_DIR"/*.cpp; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")
    if [ "$filename" != "common.cpp" ] && [ "$filename" != "server.cpp" ] && \
       [ "$filename" != "whisper.cpp" ] && [ "$filename" != "main.cpp" ] && \
       [ "$filename" != "stream.cpp" ] && [ "$filename" != "grammar-parser.cpp" ]; then
        cp "$file" .
    fi
done
cp "$LLAMAFILE_FILES_DIR"/*.hpp . 2>/dev/null || true
# Don't copy .cu files since whisper-mel-cuda.cu is now patched
cp "$LLAMAFILE_FILES_DIR/main.1" .
cp "$LLAMAFILE_FILES_DIR/main.1.asc" .
cp "$LLAMAFILE_FILES_DIR/jfk.wav" .
cp -r "$LLAMAFILE_FILES_DIR/doc" .

# Step 4: Remove unnecessary files and directories
echo "Removing unnecessary files and directories..."
rm -rf bindings
rm -rf .github
rm -rf .devops
rm -rf examples
rm -rf ggml
rm -rf src
rm -rf include
rm -rf tests
rm -rf scripts
rm -rf spm-headers
rm -rf cmake
rm -rf models
rm -rf samples
rm -rf grammars
rm -f .gitignore
rm -f .gitmodules
rm -f Package.swift
rm -f README_sycl.md
rm -f AUTHORS
rm -f CMakeLists.txt
rm -f Makefile

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd whisper.cpp && git reset --hard && git clean -fd"
