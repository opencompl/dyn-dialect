# An experiment with dynamic dialects

## Building

Bindings require numpy and pybind11

- Build LLVM subproject
  - `git submodule update --init`
  - `./utils/build-llvm.sh`
- `./utils/setup.sh`
- `./utils/build.sh`

## Technical overview

In MLIR, most objects are identified with their TypeID. However, TypeID are
defined at compile-time. To circumvent that, we assign new TypeID
at each instance of dynamic dialects, types, and operations. Then, we open a bit
MLIR API to be able to register new objects with a custom API. Most of the API
already exists, but was private.

We use a DynamicContext to keep track of all dynamic objects we are defining,
the same way MLIRContext keeps track of all dialects/operations/types defined.
