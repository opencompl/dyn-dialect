module {
  irdl.dialect @pdl_interp {
    irdl.operation @apply_constraint {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @apply_rewrite {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @are_equal {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @branch {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @check_attribute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @check_operand_count {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @check_operation_name {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @check_result_count {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @check_type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @check_types {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @continue {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @create_attribute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @create_operation {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%6, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%4, %5, %10)
      irdl.results(%11)
    }
    irdl.operation @create_range {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType()))) || ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())))" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @create_type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @create_types {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
    irdl.operation @erase {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @extract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @finalize {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @foreach {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @get_attribute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_attribute_type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_defining_op {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @get_operand {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_operands {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any_of(%1, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @get_result {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_results {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any_of(%1, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @get_users {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%4)
      irdl.results(%7)
    }
    irdl.operation @get_value_type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @is_not_null {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @record_match {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @replace {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any_of(%1, %4) 
      irdl.operands(%0, %5)
      irdl.results()
    }
    irdl.operation @switch_attribute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @switch_operand_count {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @switch_operation_name {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @switch_result_count {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @switch_type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @switch_types {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
  }
}
