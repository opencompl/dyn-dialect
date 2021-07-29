import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import *


# OPENMP and OPENACC are not included since they rely on generated tablegen
# files
# DLTI is removed because it contains no operations
def get_tablegen_op_file_list():
    files =  ["AMX/AMX.td", "Affine/IR/AffineOps.td", "ArmNeon/ArmNeon.td",
              "ArmSVE/ArmSVE.td", "Async/IR/AsyncOps.td",
              "Complex/IR/ComplexOps.td", "DLTI/DLTI.td",
              "EmitC/IR/EmitC.td", "GPU/GPUOps.td", "LLVMIR/LLVMOps.td",
              "LLVMIR/NVVMOps.td", "LLVMIR/ROCDLOps.td",
              "Linalg/IR/LinalgOps.td", "Math/IR/MathOps.td",
              "MemRef/IR/MemRefOps.td",
              "PDL/IR/PDLOps.td",
              "PDLInterp/IR/PDLInterpOps.td", "Quant/QuantOps.td",
              "SCF/SCFOps.td", "SPIRV/IR/SPIRVOps.td", "Shape/IR/ShapeOps.td",
              "SparseTensor/IR/SparseTensorOps.td", "StandardOps/IR/Ops.td",
              "Tensor/IR/TensorOps.td", "Tosa/IR/TosaOps.td",
              "Vector/VectorOps.td", "X86Vector/X86Vector.td"]
    files = [os.path.join("../llvm-project/mlir/include/mlir/Dialect", file) for file in files]
    return files


@dataclass
class OpStats:
    name: str
    dialect: str
    hasVerifier: bool
    numOperands: int
    numResults: int
    numRegions: int

    @staticmethod
    def from_json(json):
        return OpStats(json["name"], json["dialect"], json["hasVerifier"],
                       json["numOperands"], json["numResults"], json["numRegions"])


@dataclass
class DialectStats:
    name: str
    ops: Dict[str, OpStats] = field(default_factory = dict)

    def add_op(self, op: OpStats):
        if op.name in self.ops:
            assert "op was already in dialect"
        self.ops[op.name] = op


@dataclass
class Stats:
    dialects: Dict[str, DialectStats] = field(default_factory = dict)

    def add_op(self, op: OpStats):
        if op.dialect not in self.dialects:
            self.dialects[op.dialect] = DialectStats(op.dialect)
        self.dialects[op.dialect].add_op(op)

    @staticmethod
    def from_json(json):
        stats = Stats()
        for val in json:
            stats.add_op(OpStats.from_json(val))
        return stats

    @property
    def ops(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, ops in dialect.ops.items():
                res.append(ops)
        return res


def get_stat_from_file(file):
    root, file = os.path.split(file)
    res = subprocess.run(["../build/bin/tblgen-stats", os.path.join(root, file), "--I=../llvm-project/mlir/include",
                          f"--I={root}"], capture_output=True).stdout
    ops = json.loads(res)
    return Stats.from_json(ops)


def get_stat_from_files():
    stats = Stats()
    for file in get_tablegen_op_file_list():
        file_stats = get_stat_from_file(file)
        for op in file_stats.ops:
            stats.add_op(op)
    return stats


def __main__():
    stats = get_stat_from_files()
    print(stats)


if __name__ == "__main__":
    __main__()
