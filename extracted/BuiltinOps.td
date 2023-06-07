module {
  irdl.dialect @builtin {
    irdl.operation @module {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @unrealized_conversion_cast {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
  }
}
