//===- GenEval.cpp - Generates IRDL-Eval from IRDL-SSA ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates IRDL-Eval from IRDL-SSA definitions.
//
//===----------------------------------------------------------------------===//

#include "GenEval.h"

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <iterator>

using namespace mlir;
using namespace irdleval;
using namespace irdlssa;
using namespace cf;

struct ConstraintCheck {
  Block *start;
  Block *target;

  Value constraint;
  Value typeToCheck;
};

struct BacktrackPoint {
  Block *toVisitWhenFails;
  SmallVector<Value> slotsToClearWhenFails;

  Block &generateBacktrackBlock(Location loc, Region &region,
                                IRRewriter &rewriter);
};

Block &BacktrackPoint::generateBacktrackBlock(Location loc, Region &region,
                                              IRRewriter &rewriter) {
  // Save current insertion point
  auto ipBackupBlock = rewriter.getInsertionBlock();
  auto ipBackup = rewriter.getInsertionPoint();

  Block &btBlock = region.emplaceBlock();
  rewriter.setInsertionPointToEnd(&btBlock);
  for (Value slot : this->slotsToClearWhenFails) {
    rewriter.create<Eval_ClearType>(loc, slot);
  }
  rewriter.create<BranchOp>(loc, this->toVisitWhenFails);

  rewriter.setInsertionPoint(ipBackupBlock, ipBackup);
  return btBlock;
}

struct ConstraintCheckCompileJob {
  /// Stack containing the constraint checking
  /// left to compile. The target block of a ConstraintCheck
  /// in the stack must be equal to the start block of
  /// the next item in unstacking order if it exists and
  /// in that case must be empty.
  SmallVector<ConstraintCheck> workStack;
  BacktrackPoint btPoint;
  SmallPtrSet<Value, 8> definedSlots;

  bool workStackInvariant() const;
};

bool ConstraintCheckCompileJob::workStackInvariant() const {
  for (size_t i = 1; i < this->workStack.size(); i++) {
    if (this->workStack[i - 1].start != this->workStack[i].target) {
      return false;
    } else if (this->workStack[i].target->getOperations().size() != 0) {
      return false;
    }
  }

  return true;
}

class ConstraintCompiler {
public:
  static void compile(MLIRContext *ctx, Block &constraints,
                      IRRewriter &rewriter, Location location,
                      ArrayRef<Value> args);

private:
  ConstraintCompiler(MLIRContext *ctx, Region &region, Block &constraints,
                     IRRewriter &rewriter, Location location)
      : ctx(ctx), region(region), constraints(constraints), rewriter(rewriter),
        location(location) {}

  void compileAnyType(ConstraintCheck &checkDesc, Value slot,
                      ConstraintCheckCompileJob &currentJob);
  void compileIsType(ConstraintCheck &checkDesc, Value slot,
                     ConstraintCheckCompileJob &currentJob,
                     ParamTypeAttrOrAnyAttr expected);
  void compileParametricType(ConstraintCheck &checkDesc, Value slot,
                             ConstraintCheckCompileJob &currentJob,
                             StringRef base, ArrayRef<Value> argConstraints);
  void compileAnyOf(ConstraintCheck &checkDesc, Value slot,
                    ConstraintCheckCompileJob &currentJob,
                    ArrayRef<Value> argConstraints);

  bool workStackInvariant() const;

  SmallVector<ConstraintCheckCompileJob> compileJobs;
  MLIRContext *ctx;
  Region &region;
  Block &constraints;
  IRRewriter &rewriter;
  Location location;
};

void ConstraintCompiler::compileAnyType(ConstraintCheck &checkDesc, Value slot,
                                        ConstraintCheckCompileJob &currentJob) {
  rewriter.setInsertionPointToEnd(checkDesc.start);
  if (currentJob.definedSlots.contains(slot)) {
    Block &failure = currentJob.btPoint.generateBacktrackBlock(
        this->location, this->region, rewriter);
    rewriter.create<Eval_MatchType>(this->location, slot, checkDesc.typeToCheck,
                                    checkDesc.target, &failure);
  } else {
    rewriter.create<Eval_AssignType>(this->location, slot,
                                     checkDesc.typeToCheck);
    rewriter.create<BranchOp>(this->location, checkDesc.target);
    currentJob.definedSlots.insert(slot);
    currentJob.btPoint.slotsToClearWhenFails.push_back(slot);
  }
}

void ConstraintCompiler::compileIsType(ConstraintCheck &checkDesc, Value slot,
                                       ConstraintCheckCompileJob &currentJob,
                                       ParamTypeAttrOrAnyAttr expected) {
  Block &failure = currentJob.btPoint.generateBacktrackBlock(
      this->location, this->region, rewriter);
  rewriter.setInsertionPointToEnd(checkDesc.start);
  rewriter.create<Eval_CheckType>(this->location, checkDesc.typeToCheck,
                                  expected, checkDesc.target, &failure);
}

void ConstraintCompiler::compileParametricType(
    ConstraintCheck &checkDesc, Value slot,
    ConstraintCheckCompileJob &currentJob, StringRef base,
    ArrayRef<Value> argConstraints) {
  Block &failure = currentJob.btPoint.generateBacktrackBlock(
      this->location, this->region, rewriter);
  rewriter.setInsertionPointToEnd(checkDesc.start);
  if (currentJob.definedSlots.contains(slot)) {
    rewriter.create<Eval_MatchType>(this->location, slot, checkDesc.typeToCheck,
                                    checkDesc.target, &failure);
  } else {
    Block &parametricSuccess = region.emplaceBlock();
    SmallVector<Type> argTypes(argConstraints.size(),
                               EvalTypeType::get(this->ctx));
    SmallVector<Location> argLocs(argConstraints.size(), this->location);
    parametricSuccess.addArguments(argTypes, argLocs);
    rewriter.create<Eval_CheckParametric>(this->location, checkDesc.typeToCheck,
                                          base, &parametricSuccess, &failure);

    rewriter.setInsertionPointToEnd(&parametricSuccess);
    rewriter.create<Eval_AssignType>(this->location, slot,
                                     checkDesc.typeToCheck);
    currentJob.definedSlots.insert(slot);
    currentJob.btPoint.slotsToClearWhenFails.push_back(slot);

    Block *start = &parametricSuccess;
    Block *target;
    SmallVector<ConstraintCheck> newChecks;
    for (size_t i = 0; i + 1 < argConstraints.size(); i++) {
      target = &region.emplaceBlock();
      newChecks.push_back({
          .start = start,
          .target = target,
          .constraint = argConstraints[i],
          .typeToCheck = parametricSuccess.getArgument(i),
      });
      start = target;
    }

    newChecks.push_back({
        .start = start,
        .target = checkDesc.target,
        .constraint = argConstraints[argConstraints.size() - 1],
        .typeToCheck = parametricSuccess.getArgument(argConstraints.size() - 1),
    });

    currentJob.workStack.append(newChecks.rbegin(), newChecks.rend());
  }
}

void ConstraintCompiler::compileAnyOf(ConstraintCheck &checkDesc, Value slot,
                                      ConstraintCheckCompileJob &currentJob,
                                      ArrayRef<Value> argConstraints) {
  if (currentJob.definedSlots.contains(slot)) {
    Block &failure = currentJob.btPoint.generateBacktrackBlock(
        this->location, this->region, rewriter);
    rewriter.setInsertionPointToEnd(checkDesc.start);
    rewriter.create<Eval_MatchType>(this->location, slot, checkDesc.typeToCheck,
                                    checkDesc.target, &failure);
  } else {
    if (argConstraints.size() == 0) {
      Block &failure = currentJob.btPoint.generateBacktrackBlock(
          this->location, this->region, rewriter);
      rewriter.setInsertionPointToEnd(checkDesc.start);
      rewriter.create<BranchOp>(this->location, &failure);
    } else {
      rewriter.setInsertionPointToEnd(checkDesc.start);
      rewriter.create<Eval_AssignType>(this->location, slot,
                                       checkDesc.typeToCheck);
      currentJob.definedSlots.insert(slot);
      currentJob.btPoint.slotsToClearWhenFails.push_back(slot);

      SmallVector<BacktrackPoint> backtrackPoints;
      for (size_t i = 0; i + 1 < argConstraints.size(); i++) {
        backtrackPoints.push_back(
            {.toVisitWhenFails = &this->region.emplaceBlock(),
             .slotsToClearWhenFails = {}});
      }
      backtrackPoints.push_back(std::move(currentJob.btPoint));

      for (size_t i = 1; i < argConstraints.size(); i++) {
        SmallVector<ConstraintCheck> newWorkStack = currentJob.workStack;
        DenseMap<Block *, Block *> blockTransl;

        // Copy work item start blocks
        for (size_t i = 0; i < currentJob.workStack.size(); i++) {
          // This is guaranteed by the linked structure of the stack.
          assert(blockTransl.count(currentJob.workStack[i].start) == 0 &&
                 "repeated start block for work stack item");
          assert(currentJob.workStack[i].start->getOperations().size() == 0 &&
                 "work stack block is not empty");

          blockTransl.insert(
              {currentJob.workStack[i].start, &region.emplaceBlock()});
          newWorkStack[i].start = blockTransl[currentJob.workStack[i].start];
        }

        // Update all targets except the last one
        for (size_t i = 1; i < currentJob.workStack.size(); i++) {
          // This is guaranteed by the linked structure of the stack.
          assert(blockTransl.count(currentJob.workStack[i].target) == 1 &&
                 "invalid successor for work stack item");

          newWorkStack[i].target = blockTransl[currentJob.workStack[i].target];
        }

        Block *target = checkDesc.target;
        if (blockTransl.count(checkDesc.target) == 1) {
          target = blockTransl[checkDesc.target];
        }

        newWorkStack.push_back(
            {.start = backtrackPoints[i - 1].toVisitWhenFails,
             .target = target,
             .constraint = argConstraints[i],
             .typeToCheck = checkDesc.typeToCheck});
        this->compileJobs.push_back({
            .workStack = std::move(newWorkStack),
            .btPoint = std::move(backtrackPoints[i]),
            .definedSlots = currentJob.definedSlots,
        });
      }

      currentJob.workStack.push_back({.start = checkDesc.start,
                                      .target = checkDesc.target,
                                      .constraint = argConstraints[0],
                                      .typeToCheck = checkDesc.typeToCheck});
      currentJob.btPoint = std::move(backtrackPoints[0]);
    }
  }
}

bool ConstraintCompiler::workStackInvariant() const {
  for (ConstraintCheckCompileJob const &job : this->compileJobs) {
    if (!job.workStackInvariant()) {
      return false;
    }
  }

  return true;
}

void ConstraintCompiler::compile(MLIRContext *ctx, Block &constraints,
                                 IRRewriter &rewriter, Location location,
                                 ArrayRef<Value> args) {
  rewriter.setInsertionPointToEnd(&constraints);
  auto verifierOp = rewriter.create<Eval_Verifier>(location);

  Region &region = verifierOp.body();

  SmallVector<Type> argTypes(args.size(), EvalTypeType::get(ctx));
  SmallVector<Location> argLocs(args.size(), location);
  Block &start = region.emplaceBlock();
  start.addArguments(argTypes, argLocs);

  if (args.size() == 0) {
    rewriter.setInsertionPointToStart(&start);
    rewriter.create<Eval_Success>(location);
    return;
  }

  rewriter.setInsertionPointToStart(&start);
  DenseMap<Value, Value> cstrToSlot;
  for (Operation &op : constraints.getOperations()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      Eval_Alloca alloca =
          rewriter.create<Eval_Alloca>(location, SlotType::get(ctx));
      cstrToSlot.insert({op.getResult(0), alloca.getResult()});
    }
  }

  Block &success = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&success);
  rewriter.create<Eval_Success>(location);

  Block &failure = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&failure);
  rewriter.create<Eval_Failure>(location);

  ConstraintCompiler compiler(ctx, region, constraints, rewriter, location);

  SmallVector<ConstraintCheck> argToWork;
  Block *workStart = &start;
  for (size_t i = 0; i < args.size() - 1; i++) {
    Block *workTarget = &region.emplaceBlock();
    argToWork.push_back({
        .start = workStart,
        .target = workTarget,
        .constraint = args[i],
        .typeToCheck = start.getArgument(i),
    });
    workStart = workTarget;
  }
  argToWork.push_back({
      .start = workStart,
      .target = &success,
      .constraint = args[args.size() - 1],
      .typeToCheck = start.getArgument(args.size() - 1),
  });

  compiler.compileJobs.push_back({
      .workStack = {argToWork.rbegin(), argToWork.rend()},
      .btPoint =
          {
              .toVisitWhenFails = &failure,
              .slotsToClearWhenFails = {},
          },
      .definedSlots = {},
  });

  while (compiler.compileJobs.size() != 0) {
    assert(compiler.workStackInvariant() &&
           "work stack invariant is not respected for jobs");

    ConstraintCheckCompileJob currentJob =
        std::move(compiler.compileJobs.back());
    compiler.compileJobs.pop_back();

    while (currentJob.workStack.size() != 0) {
      assert(currentJob.workStackInvariant() &&
             "work stack invariant is not respected for current job");

      ConstraintCheck currentCheck = std::move(currentJob.workStack.back());
      currentJob.workStack.pop_back();

      TypeSwitch<Operation *>(currentCheck.constraint.getDefiningOp())
          .Case<SSA_AnyType>([&](SSA_AnyType op) {
            compiler.compileAnyType(
                currentCheck, cstrToSlot[currentCheck.constraint], currentJob);
          })
          .Case<SSA_IsType>([&](SSA_IsType op) {
            compiler.compileIsType(currentCheck,
                                   cstrToSlot[currentCheck.constraint],
                                   currentJob, op.type());
          })
          .Case<SSA_ParametricType>([&](SSA_ParametricType op) {
            SmallVector<Value> args = op.args();
            compiler.compileParametricType(currentCheck,
                                           cstrToSlot[currentCheck.constraint],
                                           currentJob, op.type(), args);
          })
          .Case<SSA_AnyOf>([&](SSA_AnyOf op) {
            SmallVector<Value> args = op.args();
            compiler.compileAnyOf(currentCheck,
                                  cstrToSlot[currentCheck.constraint],
                                  currentJob, args);
          })
          .Default([](Operation *op) { assert(0 && "unsupported operation"); });
    }
  }
}

void GenEval::runOnOperation() {
  IRRewriter rewriter(&this->getContext());

  SSA_DialectOp op = this->getOperation();

  op.walk([&](SSA_TypeOp op) {
    for (Operation &child : op.getOps()) {
      if (llvm::isa<Eval_Verifier>(child)) {
        return;
      }
    }

    SmallVector<Value> args;
    for (Operation &op : op.getRegion().getOps()) {
      if (SSA_ParametersOp paramOp = llvm::dyn_cast<SSA_ParametersOp>(op)) {
        for (auto arg : paramOp.args()) {
          args.push_back(arg);
        }
      }
    }

    ConstraintCompiler::compile(&this->getContext(),
                                op.body().getBlocks().front(), rewriter,
                                op.getLoc(), args);
  });

  op.walk([&](SSA_OperationOp op) {
    for (Operation &child : op.getOps()) {
      if (llvm::isa<Eval_Verifier>(child)) {
        return;
      }
    }

    SSA_OperandsOp operOp;
    SSA_ResultsOp resOp;
    for (Operation &op : op.getRegion().getOps()) {
      if (SSA_OperandsOp opFound = llvm::dyn_cast<SSA_OperandsOp>(op)) {
        operOp = opFound;
      } else if (SSA_ResultsOp opFound = llvm::dyn_cast<SSA_ResultsOp>(op)) {
        resOp = opFound;
      }
    }

    SmallVector<Value> args;
    if (operOp) {
      for (auto arg : operOp.args()) {
        args.push_back(arg);
      }
    }

    if (resOp) {
      for (auto arg : resOp.args()) {
        args.push_back(arg);
      }
    }

    ConstraintCompiler::compile(&this->getContext(),
                                op.body().getBlocks().front(), rewriter,
                                op.getLoc(), args);
  });
}
