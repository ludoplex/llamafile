#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_DIR="$SCRIPT_DIR/../stable-diffusion.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$SD_DIR"

if [ -f "BUILD.mk" ]; then
    echo "Patches appear to be already applied. Skipping..."
    exit 0
fi

echo "Applying patches to stable-diffusion.cpp submodule..."

echo "Applying modifications to upstream files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        patch -p0 < "$patch_file"
    fi
done

echo "Copying llamafile-specific files..."
cp "$LLAMAFILE_FILES_DIR/BUILD.mk" .
cp "$LLAMAFILE_FILES_DIR/README.llamafile" .
cp "$LLAMAFILE_FILES_DIR/main.cpp" .
cp "$LLAMAFILE_FILES_DIR/darts.h" .
cp "$LLAMAFILE_FILES_DIR/miniz.h" .
cp "$LLAMAFILE_FILES_DIR/zip.c" .
cp "$LLAMAFILE_FILES_DIR/zip.h" .

echo "Removing unnecessary files and directories..."
rm -rf .github
rm -rf assets
rm -rf docs
rm -rf examples
rm -rf ggml
rm -rf models
rm -rf thirdparty
rm -f .clang-format
rm -f .dockerignore
rm -f .gitignore
rm -f .gitmodules
rm -f CMakeLists.txt
rm -f Dockerfile
rm -f format-code.sh
rm -f README.md

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd stable-diffusion.cpp && git reset --hard && git clean -fd"
