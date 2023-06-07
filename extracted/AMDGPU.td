module {
  irdl.dialect @amdgpu {
    irdl.operation @lds_barrier {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mfma {
      %0 = irdl.c_pred "$_self.isF32()" 
      %1 = irdl.c_pred "$_self.isF64()" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(64)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF16()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %25 = irdl.any_of(%24) 
      %26 = irdl.all_of(%23, %25) 
      %27 = irdl.all_of(%20, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isBF16()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.any_of(%36, %37) 
      %39 = irdl.all_of(%35, %38) 
      %40 = irdl.all_of(%32, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %51 = irdl.any_of(%49, %50) 
      %52 = irdl.all_of(%48, %51) 
      %53 = irdl.all_of(%45, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isFloat8E4M3FNUZ())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %58 = irdl.all_of(%56, %57) 
      %59 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %61 = irdl.all_of(%59, %60) 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %63 = irdl.any_of(%62) 
      %64 = irdl.all_of(%61, %63) 
      %65 = irdl.all_of(%58, %64) 
      %66 = irdl.any_of(%0, %1, %2, %3, %15, %27, %40, %53, %65) 
      %67 = irdl.c_pred "$_self.isF32()" 
      %68 = irdl.c_pred "$_self.isF64()" 
      %69 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %70 = irdl.c_pred "$_self.isSignlessInteger(64)" 
      %71 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %75 = irdl.all_of(%73, %74) 
      %76 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %78 = irdl.all_of(%76, %77) 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %80 = irdl.any_of(%79) 
      %81 = irdl.all_of(%78, %80) 
      %82 = irdl.all_of(%75, %81) 
      %83 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %84 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %85 = irdl.all_of(%83, %84) 
      %86 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF16()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %87 = irdl.all_of(%85, %86) 
      %88 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %89 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %90 = irdl.all_of(%88, %89) 
      %91 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %92 = irdl.any_of(%91) 
      %93 = irdl.all_of(%90, %92) 
      %94 = irdl.all_of(%87, %93) 
      %95 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %96 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %97 = irdl.all_of(%95, %96) 
      %98 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isBF16()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %99 = irdl.all_of(%97, %98) 
      %100 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %101 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %102 = irdl.all_of(%100, %101) 
      %103 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %104 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %105 = irdl.any_of(%103, %104) 
      %106 = irdl.all_of(%102, %105) 
      %107 = irdl.all_of(%99, %106) 
      %108 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %109 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %110 = irdl.all_of(%108, %109) 
      %111 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %112 = irdl.all_of(%110, %111) 
      %113 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %114 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %115 = irdl.all_of(%113, %114) 
      %116 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %117 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %118 = irdl.any_of(%116, %117) 
      %119 = irdl.all_of(%115, %118) 
      %120 = irdl.all_of(%112, %119) 
      %121 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %122 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %123 = irdl.all_of(%121, %122) 
      %124 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isFloat8E4M3FNUZ())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %125 = irdl.all_of(%123, %124) 
      %126 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %127 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %128 = irdl.all_of(%126, %127) 
      %129 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %130 = irdl.any_of(%129) 
      %131 = irdl.all_of(%128, %130) 
      %132 = irdl.all_of(%125, %131) 
      %133 = irdl.any_of(%67, %68, %69, %70, %82, %94, %107, %120, %132) 
      %134 = irdl.c_pred "$_self.isF64()" 
      %135 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %136 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %137 = irdl.all_of(%135, %136) 
      %138 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %139 = irdl.all_of(%137, %138) 
      %140 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %141 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %142 = irdl.all_of(%140, %141) 
      %143 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %144 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %145 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 32" 
      %146 = irdl.any_of(%143, %144, %145) 
      %147 = irdl.all_of(%142, %146) 
      %148 = irdl.all_of(%139, %147) 
      %149 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %150 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %151 = irdl.all_of(%149, %150) 
      %152 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %153 = irdl.all_of(%151, %152) 
      %154 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %155 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %156 = irdl.all_of(%154, %155) 
      %157 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %158 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %159 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 32" 
      %160 = irdl.any_of(%157, %158, %159) 
      %161 = irdl.all_of(%156, %160) 
      %162 = irdl.all_of(%153, %161) 
      %163 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %164 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %165 = irdl.all_of(%163, %164) 
      %166 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %167 = irdl.all_of(%165, %166) 
      %168 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %169 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %170 = irdl.all_of(%168, %169) 
      %171 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %172 = irdl.any_of(%171) 
      %173 = irdl.all_of(%170, %172) 
      %174 = irdl.all_of(%167, %173) 
      %175 = irdl.any_of(%134, %148, %162, %174) 
      %176 = irdl.c_pred "$_self.isF64()" 
      %177 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %178 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %179 = irdl.all_of(%177, %178) 
      %180 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %181 = irdl.all_of(%179, %180) 
      %182 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %183 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %184 = irdl.all_of(%182, %183) 
      %185 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %186 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %187 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 32" 
      %188 = irdl.any_of(%185, %186, %187) 
      %189 = irdl.all_of(%184, %188) 
      %190 = irdl.all_of(%181, %189) 
      %191 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %192 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %193 = irdl.all_of(%191, %192) 
      %194 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %195 = irdl.all_of(%193, %194) 
      %196 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %197 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %198 = irdl.all_of(%196, %197) 
      %199 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %200 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %201 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 32" 
      %202 = irdl.any_of(%199, %200, %201) 
      %203 = irdl.all_of(%198, %202) 
      %204 = irdl.all_of(%195, %203) 
      %205 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %206 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %207 = irdl.all_of(%205, %206) 
      %208 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %209 = irdl.all_of(%207, %208) 
      %210 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %211 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %212 = irdl.all_of(%210, %211) 
      %213 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %214 = irdl.any_of(%213) 
      %215 = irdl.all_of(%212, %214) 
      %216 = irdl.all_of(%209, %215) 
      %217 = irdl.any_of(%176, %190, %204, %216) 
      irdl.operands(%66, %133, %175)
      irdl.results(%217)
    }
    irdl.operation @raw_buffer_atomic_cmpswap {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(64)" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isF64()" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.any
      %6 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %11 = irdl.any
      irdl.operands(%4, %5, %8, %9, %10)
      irdl.results(%11)
    }
    irdl.operation @raw_buffer_atomic_fadd {
      %0 = irdl.c_pred "$_self.isF32()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %3, %4, %5)
      irdl.results()
    }
    irdl.operation @raw_buffer_atomic_fmax {
      %0 = irdl.c_pred "$_self.isF32()" 
      %1 = irdl.c_pred "$_self.isF64()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%2, %5, %6, %7)
      irdl.results()
    }
    irdl.operation @raw_buffer_atomic_smax {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %3, %4, %5)
      irdl.results()
    }
    irdl.operation @raw_buffer_atomic_umin {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %3, %4, %5)
      irdl.results()
    }
    irdl.operation @raw_buffer_load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "$_self.isBF16()" 
      %6 = irdl.c_pred "$_self.isF16()" 
      %7 = irdl.c_pred "$_self.isF32()" 
      %8 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %9 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %10 = irdl.c_pred "$_self.isFloat8E5M2FNUZ()" 
      %11 = irdl.c_pred "$_self.isFloat8E4M3FNUZ()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.all_of(%19, %22) 
      %24 = irdl.all_of(%16, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %36 = irdl.any_of(%33, %34, %35) 
      %37 = irdl.all_of(%32, %36) 
      %38 = irdl.all_of(%29, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isFloat8E4M3FNUZ())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %51 = irdl.any_of(%47, %48, %49, %50) 
      %52 = irdl.all_of(%46, %51) 
      %53 = irdl.all_of(%43, %52) 
      %54 = irdl.any_of(%5, %6, %7, %8, %9, %10, %11, %24, %38, %53) 
      irdl.operands(%2, %3, %4)
      irdl.results(%54)
    }
    irdl.operation @raw_buffer_store {
      %0 = irdl.c_pred "$_self.isBF16()" 
      %1 = irdl.c_pred "$_self.isF16()" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %4 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %5 = irdl.c_pred "$_self.isFloat8E5M2FNUZ()" 
      %6 = irdl.c_pred "$_self.isFloat8E4M3FNUZ()" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %17 = irdl.any_of(%15, %16) 
      %18 = irdl.all_of(%14, %17) 
      %19 = irdl.all_of(%11, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.any_of(%28, %29, %30) 
      %32 = irdl.all_of(%27, %31) 
      %33 = irdl.all_of(%24, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isFloat8E4M3FNUZ())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %46 = irdl.any_of(%42, %43, %44, %45) 
      %47 = irdl.all_of(%41, %46) 
      %48 = irdl.all_of(%38, %47) 
      %49 = irdl.any_of(%0, %1, %2, %3, %4, %5, %6, %19, %33, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %51 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %54 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%49, %52, %53, %54)
      irdl.results()
    }
  }
}
