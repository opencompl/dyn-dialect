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

## Technical overview

In MLIR, most objects are identified with their TypeID. However, TypeID are
defined at compile-time. To circumvent that, we assign new TypeID
at each instance of dynamic dialects, types, and operations. Then, we open a bit
MLIR API to be able to register new objects with a custom API. Most of the API
already exists, but was private.

We use a DynamicContext to keep track of all dynamic objects we are defining,
the same way MLIRContext keeps track of all dialects/operations/types defined.
