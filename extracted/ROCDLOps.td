module {
  irdl.dialect @rocdl {
    irdl.operation @barrier {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @workgroup.dim.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workgroup.dim.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workgroup.dim.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workgroup.id.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workgroup.id.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workgroup.id.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @grid.dim.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @grid.dim.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @grid.dim.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @buffer.load {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @buffer.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results()
    }
    irdl.operation @raw.buffer.atomic.cmpswap {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @raw.buffer.atomic.fadd {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @raw.buffer.atomic.fmax {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @raw.buffer.atomic.smax {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @raw.buffer.atomic.umin {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @raw.buffer.load {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results(%4)
    }
    irdl.operation @raw.buffer.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @workitem.id.x {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workitem.id.y {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @workitem.id.z {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @mfma.f32.4x4x1f32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.4x4x2bf16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.4x4x4bf16.1k {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.4x4x4f16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x1f32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x2bf16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x4bf16.1k {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x4f16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x4f32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x8.xf32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x8bf16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x16bf16.1k {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x16f16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x32.bf8.bf8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x32.bf8.fp8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x32.fp8.bf8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.16x16x32.fp8.fp8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x1f32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x2bf16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x2f32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x4.xf32 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x4bf16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x4bf16.1k {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x4f16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x8bf16.1k {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x8f16 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x16.bf8.bf8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x16.bf8.fp8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x16.fp8.bf8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f32.32x32x16.fp8.fp8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f64.4x4x4f64 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.f64.16x16x4f64 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.4x4x4i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.16x16x4i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.16x16x16i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.16x16x32.i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.32x32x4i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.32x32x8i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mfma.i32.32x32x16.i8 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
  }
}
