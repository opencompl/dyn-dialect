#!/bin/bash
cd llvm-project/build
cmake --build . --target check-mlir
