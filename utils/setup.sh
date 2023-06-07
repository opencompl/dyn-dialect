#!/bin/bash
if command -v mold > /dev/null; then
  BEST_LINKER="mold"
elif command -v lld > /dev/null; then
  BEST_LINKER="lld"
else
  BEST_LINKER="ld"
fi
LINKER=${3:-${BEST_LINKER}}

mkdir -p build && cd build

export CMAKE_GENERATOR=Ninja
cmake -G Ninja ../ \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DMLIR_DIR=$(pwd)/../llvm-project/build/lib/cmake/mlir \
      -DLLVM_EXTERNAL_LIT=$(pwd)/../llvm-project/build/bin/llvm-lit \
      -DLLVM_USE_LINKER=${LINKER} \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++
