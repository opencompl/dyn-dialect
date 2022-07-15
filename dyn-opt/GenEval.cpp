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

namespace {

/// Represents a constraint check generation work item.
/// Tasks the compiler to generate in block `start` a
/// verifier for `constraint` against `typeToCheck`,
/// going to `target` if successful or to the
/// contextual backtrack point if unsuccessful.
struct ConstraintCheck {
  ConstraintCheck(Block *start, Block *target, Value constraint,
                  Value typeToCheck)
      : start(start), target(target), constraint(constraint),
        typeToCheck(typeToCheck) {}

  Block *start;
  Block *target;

  Value constraint;
  Value typeToCheck;
};

/// Represents a point at which the upcoming tasks
/// must go back to when they fail.
struct BacktrackPoint {
  BacktrackPoint(Block *toVisitWhenFails,
                 SmallVector<Value> slotsToClearWhenFails = {})
      : toVisitWhenFails(toVisitWhenFails),
        slotsToClearWhenFails(std::move(slotsToClearWhenFails)) {}

  Block *toVisitWhenFails;

  /// Lists of slots that must have been cleared
  /// before jumping to the failure-case block.
  /// This list is expanded as compiling jobs define
  /// new slots in their execution.
  SmallVector<Value> slotsToClearWhenFails;

  /// Generates a block that cleans up the slots
  /// to clear before jumping to `toVisitWhenFails`.
  /// This block should be generated for each instruction
  /// as a target in case of failure, in order to account
  /// for newly added slots.
  Block &generateBacktrackBlock(Location loc, Region &region,
                                IRRewriter &rewriter);
};

/// Collection of work items with a shared backtrack point.
struct ConstraintCheckCompileJob {
  ConstraintCheckCompileJob(SmallVector<ConstraintCheck> workStack,
                            BacktrackPoint btPoint,
                            SmallPtrSet<Value, 8> definedSlots = {})
      : workStack(std::move(workStack)), btPoint(std::move(btPoint)),
        definedSlots(std::move(definedSlots)) {}

  /// Stack containing the constraint checks
  /// left to compile. The target block of a ConstraintCheck
  /// in the stack must be equal to the start block of
  /// the next item in unstacking order if it exists and
  /// in that case must be empty.
  SmallVector<ConstraintCheck> workStack;

  /// Backtrack point for items in the work stack to
  /// jump to in case of failure.
  BacktrackPoint btPoint;

  /// Lists all slot that are currently known to be
  /// defined in this context.
  /// This list is expanded as compiling jobs define
  /// new slots in their execution.
  SmallPtrSet<Value, 8> definedSlots;

  /// Ensures the work stack linking invariant upholds
  /// for debug assert purposes.
  bool workStackInvariant() const;
};

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

  /// Ensures the work stack linking invariant upholds
  /// in every registered job for debug assert purposes.
  bool workStackInvariant() const;

  SmallVector<ConstraintCheckCompileJob> compileJobs;
  MLIRContext *ctx;
  Region &region;
  Block &constraints;
  IRRewriter &rewriter;
  Location location;
};

} // namespace

Block &BacktrackPoint::generateBacktrackBlock(Location loc, Region &region,
                                              IRRewriter &rewriter) {
  // Save current insertion point
  auto ipBackupBlock = rewriter.getInsertionBlock();
  auto ipBackup = rewriter.getInsertionPoint();

  Block &btBlock = region.emplaceBlock();
  rewriter.setInsertionPointToEnd(&btBlock);
  for (Value slot : this->slotsToClearWhenFails) {
    rewriter.create<ClearType>(loc, slot);
  }
  rewriter.create<BranchOp>(loc, this->toVisitWhenFails);

  // Restore insertion point
  rewriter.setInsertionPoint(ipBackupBlock, ipBackup);
  return btBlock;
}

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

void ConstraintCompiler::compileAnyType(ConstraintCheck &checkDesc, Value slot,
                                        ConstraintCheckCompileJob &currentJob) {
  rewriter.setInsertionPointToEnd(checkDesc.start);
  if (currentJob.definedSlots.contains(slot)) {
    Block &failure = currentJob.btPoint.generateBacktrackBlock(
        this->location, this->region, rewriter);
    rewriter.create<MatchType>(this->location, slot, checkDesc.typeToCheck,
                               checkDesc.target, &failure);
  } else {
    rewriter.create<AssignType>(this->location, slot, checkDesc.typeToCheck);
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
  rewriter.create<CheckType>(this->location, checkDesc.typeToCheck, expected,
                             checkDesc.target, &failure);
}

void ConstraintCompiler::compileParametricType(
    ConstraintCheck &checkDesc, Value slot,
    ConstraintCheckCompileJob &currentJob, StringRef base,
    ArrayRef<Value> argConstraints) {
  Block &failure = currentJob.btPoint.generateBacktrackBlock(
      this->location, this->region, rewriter);
  rewriter.setInsertionPointToEnd(checkDesc.start);
  if (currentJob.definedSlots.contains(slot)) {
    rewriter.create<MatchType>(this->location, slot, checkDesc.typeToCheck,
                               checkDesc.target, &failure);
  } else {
    Block &parametricSuccess = region.emplaceBlock();
    SmallVector<Type> argTypes(argConstraints.size(),
                               EvalTypeType::get(this->ctx));
    SmallVector<Location> argLocs(argConstraints.size(), this->location);
    parametricSuccess.addArguments(argTypes, argLocs);
    rewriter.create<CheckParametric>(this->location, checkDesc.typeToCheck,
                                     base, &parametricSuccess, &failure);

    // Pre-assign the type in the slot table as if the constraint
    // check succeeded. This is correct because further parameter
    // checks cannot reference this slot anyway.
    rewriter.setInsertionPointToEnd(&parametricSuccess);
    rewriter.create<AssignType>(this->location, slot, checkDesc.typeToCheck);
    currentJob.definedSlots.insert(slot);
    currentJob.btPoint.slotsToClearWhenFails.push_back(slot);

    // Schedule type-parameter checking work items.
    Block *start = &parametricSuccess;
    Block *target;
    SmallVector<ConstraintCheck> newChecks;
    for (size_t i = 0; i + 1 < argConstraints.size(); i++) {
      target = &region.emplaceBlock();
      newChecks.emplace_back(start, target, argConstraints[i],
                             parametricSuccess.getArgument(i));
      start = target;
    }

    newChecks.emplace_back(
        start, checkDesc.target, argConstraints[argConstraints.size() - 1],
        parametricSuccess.getArgument(argConstraints.size() - 1));

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
    rewriter.create<MatchType>(this->location, slot, checkDesc.typeToCheck,
                               checkDesc.target, &failure);
  } else {
    if (argConstraints.size() == 0) {
      // Simple failure for trivial AnyOf constraints
      Block &failure = currentJob.btPoint.generateBacktrackBlock(
          this->location, this->region, rewriter);
      rewriter.setInsertionPointToEnd(checkDesc.start);
      rewriter.create<BranchOp>(this->location, &failure);
      return;
    }

    // Pre-assign the type in the slot table as if the constraint
    // check succeeded. This is correct because further parameter
    // checks cannot reference this slot anyway.
    rewriter.setInsertionPointToEnd(checkDesc.start);
    rewriter.create<AssignType>(this->location, slot, checkDesc.typeToCheck);
    currentJob.definedSlots.insert(slot);
    currentJob.btPoint.slotsToClearWhenFails.push_back(slot);

    // AnyOf constraints are checked by attempting to go as deep
    // as possible in a branch, and backtracking to the next branch
    // if anything failed while exploring the attempted branch.
    // This is achieved in the compiler by created a compile job
    // for each branch and its initial constraint check, each job
    // holding a copy of the initial work stack, with a different
    // backtracking point.

    // Prepare the backtrack points for all possible AnyOf branch
    SmallVector<BacktrackPoint> backtrackPoints;
    for (size_t i = 0; i + 1 < argConstraints.size(); i++) {
      backtrackPoints.emplace_back(&this->region.emplaceBlock());
    }
    backtrackPoints.push_back(std::move(currentJob.btPoint));

    // Create new work stacks for all possible AnyOf branch
    // except the first one. The first one will re-use the
    // original work stack for efficiency.
    // This is achieved by duplicating all blocks in the work
    // stacks and linking them correctly.
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

      // Update all targets except the last one.
      // The last one points to global success and thus must be kept.
      for (size_t i = 1; i < currentJob.workStack.size(); i++) {
        // This is guaranteed by the linked structure of the stack.
        assert(blockTransl.count(currentJob.workStack[i].target) == 1 &&
               "invalid successor for work stack item");

        newWorkStack[i].target = blockTransl[currentJob.workStack[i].target];
      }

      // Find the next work item to compute after the branch constraint test
      // succeeds.
      Block *target = checkDesc.target;
      if (blockTransl.count(checkDesc.target) == 1) {
        target = blockTransl[checkDesc.target];
      }

      // Schedule checking the branch constraint itself.
      newWorkStack.emplace_back(backtrackPoints[i - 1].toVisitWhenFails, target,
                                argConstraints[i], checkDesc.typeToCheck);

      // To summarize, at this point the new work stack consists of a copy of
      // the initial work stack with a check for the constraint of this AnyOf
      // branch on top. We can now add a job that will compile this work stack:
      // first check that we indeed want to go deeper in this branch
      // by compiling a check for this branch's constraint, then compile
      // the rest of the work to do. If any of those things fails, we
      // go to a backtrack point that points to a similar stack for the next
      // AnyOf branch, or to the backtrack point of the current AnyOf.
      this->compileJobs.emplace_back(std::move(newWorkStack),
                                     std::move(backtrackPoints[i]),
                                     currentJob.definedSlots);
    }

    // Reuse the original work stack for the first branch for efficiency
    currentJob.workStack.emplace_back(checkDesc.start, checkDesc.target,
                                      argConstraints[0], checkDesc.typeToCheck);
    currentJob.btPoint = std::move(backtrackPoints[0]);
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
  auto verifierOp = rewriter.create<Verifier>(location);

  Region &region = verifierOp.body();

  SmallVector<Type> argTypes(args.size(), EvalTypeType::get(ctx));
  SmallVector<Location> argLocs(args.size(), location);
  Block &start = region.emplaceBlock();
  start.addArguments(argTypes, argLocs);

  if (args.size() == 0) {
    rewriter.setInsertionPointToStart(&start);
    rewriter.create<Success>(location);
    return;
  }

  rewriter.setInsertionPointToStart(&start);
  DenseMap<Value, Value> cstrToSlot;
  for (Operation &op : constraints.getOperations()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      Alloca alloca = rewriter.create<Alloca>(location, SlotType::get(ctx));
      cstrToSlot.insert({op.getResult(0), alloca.getResult()});
    }
  }

  Block &success = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&success);
  rewriter.create<Success>(location);

  Block &failure = region.emplaceBlock();
  rewriter.setInsertionPointToStart(&failure);
  rewriter.create<Failure>(location);

  ConstraintCompiler compiler(ctx, region, constraints, rewriter, location);

  SmallVector<ConstraintCheck> argToWork;
  Block *workStart = &start;
  for (size_t i = 0; i < args.size() - 1; i++) {
    Block *workTarget = &region.emplaceBlock();
    argToWork.emplace_back(workStart, workTarget, args[i],
                           start.getArgument(i));
    workStart = workTarget;
  }
  argToWork.emplace_back(workStart, &success, args[args.size() - 1],
                         start.getArgument(args.size() - 1));

  compiler.compileJobs.emplace_back(
      SmallVector<ConstraintCheck>(argToWork.rbegin(), argToWork.rend()),
      BacktrackPoint(&failure));

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
    for (Operation &child : op.getOps())
      if (llvm::isa<Verifier>(child))
        return;

    SmallVector<Value> args;
    for (Operation &op : op.getRegion().getOps())
      if (SSA_ParametersOp paramOp = llvm::dyn_cast<SSA_ParametersOp>(op))
        for (auto arg : paramOp.args())
          args.push_back(arg);

    ConstraintCompiler::compile(&this->getContext(),
                                op.body().getBlocks().front(), rewriter,
                                op.getLoc(), args);
  });

  op.walk([&](SSA_OperationOp op) {
    for (Operation &child : op.getOps())
      if (llvm::isa<Verifier>(child))
        return;

    Optional<SSA_OperandsOp> operOp = op.getOp<SSA_OperandsOp>();
    Optional<SSA_ResultsOp> resOp = op.getOp<SSA_ResultsOp>();

    SmallVector<Value> args;
    if (operOp)
      for (auto arg : operOp->args())
        args.push_back(arg);
    if (resOp)
      for (auto arg : resOp->args())
        args.push_back(arg);

    ConstraintCompiler::compile(&this->getContext(),
                                op.body().getBlocks().front(), rewriter,
                                op.getLoc(), args);
  });
}
