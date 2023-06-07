#!/usr/bin/env bash
##===- utils/build-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with the standard options. Intended to be called from 
# the github workflows.
#
##===----------------------------------------------------------------------===##


BUILD_TYPE="Debug"

# Parse command line options
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done


BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}

if command -v mold > /dev/null; then
  BEST_LINKER="mold"
elif command -v lld > /dev/null; then
  BEST_LINKER="lld"
else
  BEST_LINKER="ld"
fi

LINKER=${3:-${BEST_LINKER}}


echo "$BUILD_TYPE"
echo "$BUILD_DIR"
echo "$INSTALL_DIR"

mkdir -p llvm-project/$BUILD_DIR
mkdir -p llvm-project/$INSTALL_DIR
cd llvm-project/$BUILD_DIR
cmake -G Ninja ../llvm \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS='mlir' \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=${LINKER} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_SHARED_LIBS=ON

ninja install
