#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="${SCRIPT_DIR}/../"
TVM_DIR="${TOP_DIR}/3rdparty/incubator-tvm"
pushd $TVM_DIR
cat cmake/config.cmake | sed -e 's/USE_LLVM OFF/USE_LLVM \/usr\/lib\/llvm-5.0\/bin\/llvm-config/g' > build/config.cmake
mkdir -p build
pushd build
cmake ..
make -j4
popd
popd
