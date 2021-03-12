#!/bin/bash
sudo apt-get install ninja-build && ninja --version
mkdir -p build && cd build
export CMAKE_GENERATOR=Ninja
cmake -G Ninja ../ -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DMLIR_DIR=$(pwd)/../llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$(pwd)/../llvm-project/build/bin/llvm-lit
