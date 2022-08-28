//===- dyn-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "GenEval.h"
#include "LowerIRDL.h"
#include "MlirOptMain.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace irdl;
using namespace irdlssa;
using namespace irdleval;

class ComplexTypeWrapper : public ConcreteTypeWrapper<ComplexType> {
  StringRef getName() override { return "std.complex"; }

  SmallVector<Attribute> getParameters(ComplexType type) override {
    return {TypeAttr::get(type.getElementType())};
  }

  size_t getParameterAmount() override { return 1; }

  Type instantiate(llvm::function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> parameters) override {
    if (parameters.size() != this->getParameterAmount()) {
      emitError().append("invalid number of type parameters ",
                         parameters.size(), " (expected ",
                         this->getParameterAmount(), ")");
      return Type();
    }

    return ComplexType::getChecked(emitError,
                                   parameters[0].cast<TypeAttr>().getValue());
  }
};

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  MLIRContext ctx;
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  auto irdl = ctx.getOrLoadDialect<irdl::IRDLDialect>();
  auto irdlssa = ctx.getOrLoadDialect<irdlssa::IRDLSSADialect>();
  ctx.getOrLoadDialect<irdleval::IRDLEvalDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  irdlssa->addTypeWrapper<ComplexTypeWrapper>();
  irdl->addTypeWrapper<ComplexTypeWrapper>();

  TypeContext tyCtx(irdlssa->irdlContext);

  mlir::registerPass(
      [tyCtx{std::move(tyCtx)}]() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<LowerIRDL>(tyCtx);
      });

  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GenEval>();
  });

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);

  return failed(mlir::MlirOptMain(argc, argv, "Dyn optimizer driver\n", ctx));
}
