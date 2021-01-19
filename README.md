# An experiment with dynamic dialects

## Building

- Build LLVM subproject
  - `git submodule update --init`
  - `mkdir llvm-project/build`
  - `cd llvm-project/build`
  - ```
    cmake -G Ninja ../llvm
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_BINDINGS_PYTHON_ENABLED=ON
    ```
  - `cmake --build . --target check-mlir`

Bindings require numpy and pybind11

- run setup.sh

- run build.sh
