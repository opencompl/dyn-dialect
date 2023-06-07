module {
  irdl.dialect @nvvm {
    irdl.operation @barrier0 {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @read.ptx.sreg.ntid.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.ntid.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.ntid.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.ctaid.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.ctaid.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.ctaid.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @cp.async.commit.group {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @cp.async.shared.global {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 3" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 1" 
      %13 = irdl.all_of(%11, %12) 
      irdl.operands(%6, %13)
      irdl.results()
    }
    irdl.operation @cp.async.wait.group {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @read.ptx.sreg.nctaid.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.nctaid.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.nctaid.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.laneid {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @ldmatrix {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mma.sync {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results(%3)
    }
    irdl.operation @rcp.approx.ftz.f {
      %0 = irdl.c_pred "$_self.isF32()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @redux.sync {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @shfl.sync {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results(%4)
    }
    irdl.operation @bar.warp.sync {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @read.ptx.sreg.tid.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.tid.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @read.ptx.sreg.tid.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @vote.ballot.sync {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @wmma.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @wmma.mma {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @wmma.store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
    irdl.operation @read.ptx.sreg.warpsize {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
  }
}
