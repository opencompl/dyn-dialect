#!/bin/bash
mkdir -p build && cd build
cmake ../ -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DMLIR_DIR=$(pwd)/../llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$(pwd)/../llvm-project/build/bin/llvm-lit
