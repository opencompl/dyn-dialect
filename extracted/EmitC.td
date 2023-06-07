module {
  irdl.dialect @emitc {
    irdl.operation @apply {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @call {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @cast {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @constant {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @include {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @variable {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
  }
}
