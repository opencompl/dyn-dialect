module {
  irdl.dialect @gpu {
    irdl.operation @all_reduce {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @alloc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results(%5, %6)
    }
    irdl.operation @barrier {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @block_dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @block_id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @create_coo {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %6, %9, %12)
      irdl.results(%13, %14)
    }
    irdl.operation @create_csr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %6, %9, %12)
      irdl.results(%13, %14)
    }
    irdl.operation @create_dn_mat {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %5)
      irdl.results(%6, %7)
    }
    irdl.operation @create_dn_vec {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5, %6)
    }
    irdl.operation @create_sparse_env {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2)
    }
    irdl.operation @dealloc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4)
    }
    irdl.operation @destroy_dn_mat {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @destroy_dn_vec {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @destroy_sp_mat {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @destroy_sparse_env {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @module {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @global_id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @grid_dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @host_register {
      %0 = irdl.c_pred "::llvm::isa<::mlir::UnrankedMemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @host_unregister {
      %0 = irdl.c_pred "::llvm::isa<::mlir::UnrankedMemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @lane_id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @launch_func {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.any
      %9 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5, %6, %7, %8)
      irdl.results(%9)
    }
    irdl.operation @launch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @memcpy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %3, %6)
      irdl.results(%7)
    }
    irdl.operation @memset {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any
      %5 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @module_end {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @num_subgroups {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @printf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @sddmm_buffer_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results(%5, %6)
    }
    irdl.operation @sddmm {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %7)
      irdl.results(%8)
    }
    irdl.operation @set_default_device {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @shuffle {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(64)" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isF64()" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "$_self.isSignlessInteger(64)" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%7, %8, %9, %10) 
      %12 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%4, %5, %6)
      irdl.results(%11, %12)
    }
    irdl.operation @spmm_buffer_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results(%5, %6)
    }
    irdl.operation @spmm {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnMatHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %7)
      irdl.results(%8)
    }
    irdl.operation @spmv_buffer_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results(%5, %6)
    }
    irdl.operation @spmv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseEnvHandleType>($_self)" 
      %2 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseSpMatHandleType>($_self)" 
      %3 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %4 = irdl.c_pred "llvm::isa<::mlir::gpu::SparseDnVecHandleType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %7)
      irdl.results(%8)
    }
    irdl.operation @subgroup_id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @subgroup_mma_compute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      %1 = irdl.c_pred "((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isSignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isUnsignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF32()))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      %4 = irdl.c_pred "((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isSignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isUnsignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF32()))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      %7 = irdl.c_pred "((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isSignlessInteger(32))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF32()))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @subgroup_mma_constant_matrix {
      %0 = irdl.c_pred "$_self.isSignedInteger(8)" 
      %1 = irdl.c_pred "$_self.isUnsignedInteger(8)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %3 = irdl.c_pred "$_self.isF16()" 
      %4 = irdl.c_pred "$_self.isF32()" 
      %5 = irdl.any_of(%0, %1, %2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      irdl.operands(%5)
      irdl.results(%6)
    }
    irdl.operation @subgroup_mma_elementwise {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @subgroup_mma_load_matrix {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(32))) || ((elementType.isF16())) || ((elementType.isF32())) || (((((::llvm::isa<::mlir::VectorType>(elementType))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(32))) || ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>(elementType).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(elementType))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank() > 0))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank()\0A                           == 1)))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @subgroup_mma_store_matrix {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)" 
      %1 = irdl.c_pred "((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isSignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isUnsignedInteger(8))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isSignlessInteger(32))) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType().isF32()))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(32))) || ((elementType.isF16())) || ((elementType.isF32())) || (((((::llvm::isa<::mlir::VectorType>(elementType))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(32))) || ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>(elementType).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(elementType))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank() > 0))) && ((::llvm::cast<::mlir::VectorType>(elementType).getRank()\0A                           == 1)))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results()
    }
    irdl.operation @subgroup_reduce {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @subgroup_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @terminator {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @thread_id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @wait {
      %0 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
