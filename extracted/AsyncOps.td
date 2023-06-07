module {
  irdl.dialect @async {
    irdl.operation @add_to_group {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @await_all {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @await {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.any
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @call {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @coro.begin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroIdType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @coro.end {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @coro.free {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroIdType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @coro.id {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroIdType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @coro.save {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::CoroStateType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @coro.suspend {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroStateType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @create_group {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @execute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4, %5)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @runtime.add_ref {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @runtime.add_to_group {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @runtime.await_and_resume {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      irdl.operands(%3, %4)
      irdl.results()
    }
    irdl.operation @runtime.await {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @runtime.create_group {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @runtime.create {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
    irdl.operation @runtime.drop_ref {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @runtime.is_error {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::async::GroupType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%3)
      irdl.results(%4)
    }
    irdl.operation @runtime.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @runtime.num_worker_threads {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @runtime.resume {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::CoroHandleType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @runtime.set_available {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @runtime.set_error {
      %0 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::TokenType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @runtime.store {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::async::ValueType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
