// RUN: dyn-opt --irdl-file=%S/cmath.irdl --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: cmath
// CHECK: std
