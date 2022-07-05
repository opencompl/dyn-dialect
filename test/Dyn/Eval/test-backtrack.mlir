// RUN: dyn-opt %s --irdlssa-file=%S/backtrack.irdlssa | FileCheck %s

module {
  // CHECK: func.func @testBt0(%[[arg:[^:]*]]: !backtrack.foo<!backtrack.parametric<f32>, f32>) -> f32 {
  func.func @testBt0(%arg: !backtrack.foo<!backtrack.parametric<f32>, f32>) -> f32 {
    %res = "backtrack.const" () : () -> f32
    return %res : f32
  }
}
