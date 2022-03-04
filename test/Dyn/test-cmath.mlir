// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {

  // CHECK: func @conorm(%{{.*}}: !cmath.complex<f32>, %{{.*}}: !cmath.complex<f32>) -> f32 {
  // CHECK:   %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.complex<f32>) -> f32
  // CHECK:   %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.complex<f32>) -> f32
  // CHECK:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK:   return %{{.*}} : f32
  // CHECK: }
  func @conorm(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
    %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
    %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
    %pq = arith.mulf %norm_p, %norm_q : f32
    return %pq : f32
  }

  // CHECK: func @conorm2(%{{.*}}: !cmath.complex<f32>, %{{.*}}: !cmath.complex<f32>) -> f32 {
  // CHECK:   %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
  // CHECK:   %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.complex<f32>) -> f32
  // CHECK:   return %{{.*} : f32
  // CHECK: }
  func @conorm2(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
    %pq = "cmath.mul"(%p, %q) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
    %conorm = "cmath.norm"(%pq) : (!cmath.complex<f32>) -> f32
    return %conorm : f32
  }
}