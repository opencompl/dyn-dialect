#include "../tblgen-stats/json.h"
#include <iostream>

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

using namespace mlir;
using namespace llvm;

int main() {
  MLIRContext ctx;
  std::vector<Dialect*> dialects;
  dialects.push_back(ctx.getOrLoadDialect<linalg::LinalgDialect>());
  dialects.push_back(ctx.getOrLoadDialect<gpu::GPUDialect>());
  dialects.push_back(ctx.getOrLoadDialect<amx::AMXDialect>());
  dialects.push_back(ctx.getOrLoadDialect<x86vector::X86VectorDialect>());
  dialects.push_back(ctx.getOrLoadDialect<tensor::TensorDialect>());
  dialects.push_back(ctx.getOrLoadDialect<AffineDialect>());
  dialects.push_back(ctx.getOrLoadDialect<emitc::EmitCDialect>());
  dialects.push_back(ctx.getOrLoadDialect<memref::MemRefDialect>());
  dialects.push_back(ctx.getOrLoadDialect<arm_sve::ArmSVEDialect>());
  dialects.push_back(ctx.getOrLoadDialect<arm_neon::ArmNeonDialect>());
  dialects.push_back(ctx.getOrLoadDialect<tosa::TosaDialect>());
  dialects.push_back(ctx.getOrLoadDialect<StandardOpsDialect>());
  dialects.push_back(ctx.getOrLoadDialect<complex::ComplexDialect>());
  dialects.push_back(ctx.getOrLoadDialect<spirv::SPIRVDialect>());
  dialects.push_back(ctx.getOrLoadDialect<pdl::PDLDialect>());
  dialects.push_back(ctx.getOrLoadDialect<ROCDL::ROCDLDialect>());
  dialects.push_back(ctx.getOrLoadDialect<LLVM::LLVMDialect>());
  dialects.push_back(ctx.getOrLoadDialect<NVVM::NVVMDialect>());
  dialects.push_back(ctx.getOrLoadDialect<math::MathDialect>());
  dialects.push_back(ctx.getOrLoadDialect<vector::VectorDialect>());
  dialects.push_back(ctx.getOrLoadDialect<async::AsyncDialect>());
  dialects.push_back(ctx.getOrLoadDialect<scf::SCFDialect>());
  dialects.push_back(ctx.getOrLoadDialect<shape::ShapeDialect>());
  dialects.push_back(ctx.getOrLoadDialect<quant::QuantizationDialect>());
  dialects.push_back(ctx.getOrLoadDialect<pdl_interp::PDLInterpDialect>());
  dialects.push_back(ctx.getOrLoadDialect<sparse_tensor::SparseTensorDialect>());
  dialects.push_back(ctx.getOrLoadDialect(""));

  auto res = JSONList::get();
  for (auto dialect: dialects) {
    auto dialect_stat = JSONDict::get();
    dialect_stat->insert("name", dialect->getNamespace());
    dialect_stat->insert("numOperations",
                         ctx.getNumRegisteredOperations(dialect));
    dialect_stat->insert("numAttributes", ctx.getNumRegisteredAttributes(dialect));
    dialect_stat->insert("numTypes", ctx.getNumRegisteredTypes(dialect));
    res->insertJson(std::move(dialect_stat));
  }
  res->print(llvm::errs());
}
