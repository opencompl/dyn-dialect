import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import *


# OPENMP and OPENACC are not included since they rely on generated tablegen
# files
# DLTI is removed because it contains no operations
def get_tablegen_op_file_list():
    res_files = []
    for root, dirs, files in os.walk("../llvm-project/mlir/include/mlir/Dialect"):
        for file in files:
            if file.endswith(".td"):
                res_files.append(os.path.join(root, file))
    return res_files


def _from_json(json, typ: Type):
    if "__origin__" in typ.__dict__:
        if typ.__origin__ == list:
            arg_typ = typ.__args__[0]
            res = []
            for val in json:
                res.append(_from_json(val, arg_typ))
            return res
        assert False
    if isinstance(json, typ):
        return json
    return typ.from_json(json)


def from_json(cls):
    @dataclass
    class FromJsonWrapper(dataclass(cls)):
        def __repr__(self):
            return cls.__name__[:-5] + "(" + ", ".join([f"{key}={self.__dict__[key]}" for key in
                                                        cls.__dataclass_fields__.keys()]) + ")"

        @staticmethod
        def from_json(json):
            arg_dict = dict()
            for name, typ in cls.__dataclass_fields__.items():
                arg_dict[name] = _from_json(json[name], typ.type)
            return FromJsonWrapper(**arg_dict)

    return FromJsonWrapper


@dataclass
class ConstraintStats:
    kind: str

    @staticmethod
    def from_json(json):
        if json["kind"] == "variadic":
            return VariadicConstraintStats.from_json(json)
        if json["kind"] == "optional":
            return OptionalConstraintStats.from_json(json)
        if json["kind"] == "typeDef":
            return TypeDefConstraintStats.from_json(json)
        if json["kind"] == "integer":
            return IntegerConstraintStats.from_json(json)
        if json["kind"] == "predicate":
            return PredicateConstraintStats.from_json(json)
        assert False

    def is_declarative(self) -> bool:
        raise NotImplemented


@from_json
class VariadicConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()


@from_json
class OptionalConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()


@from_json
class TypeDefConstraintStats(ConstraintStats):
    dialect: str
    name: str

    def is_declarative(self) -> bool:
        return True


@from_json
class IntegerConstraintStats(ConstraintStats):
    bitwidth: int

    def is_declarative(self) -> bool:
        return True


@from_json
class PredicateConstraintStats(ConstraintStats):
    predicate: str
    superclass: List[str]

    def is_declarative(self) -> bool:
        if self.predicate == "($_self.isa<::mlir::IntegerType>())":
            return True
        if self.predicate == "(true)":
            return True
        if self.predicate == "($_self.isa<::mlir::IndexType>())":
            return True
        if self.predicate == "($_self.isa<::mlir::LLVM::LLVMPointerType>())":
            return True
        if self.predicate == "($_self.isa<::mlir::FloatType>())":
            return True
        if self.predicate == "($_self.isa<::mlir::gpu::AsyncTokenType>())":
            return True
        if self.predicate == "($_self.isa<::mlir::pdl::PDLType>())":
            return True
        if self.predicate == "($_self.isa<::mlir::LLVM::LLVMTokenType>())":
            return True
        if self.predicate == "(::mlir::LLVM::isCompatibleType($_self))":
            return False
        # print(self.predicate)
        # print(self.superclass)
        return False


@from_json
class NamedConstraintStats:
    name: str
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()


@from_json
class OpStats:
    name: str
    dialect: str
    hasVerifier: bool
    numOperands: int
    numVariableLengthOperands: int
    numResults: int
    numRegions: int
    hasNoVariadicRegions: bool
    numSuccessors: int
    hasAssemblyFormat: bool
    operands: List[NamedConstraintStats]
    results: List[NamedConstraintStats]

    def is_declarative(self) -> bool:
        if self.hasVerifier:
            return False
        for operand in self.operands:
            if not operand.is_declarative():
                return False
        for result in self.results:
            if not result.is_declarative():
                return False
        return True


@from_json
class AttrOrTypeParameterStats:
    name: str
    cppType: str


@from_json
class TypeStats:
    name: str
    dialect: str
    numParameters: int
    parameters: List[AttrOrTypeParameterStats]


@dataclass
class DialectStats:
    name: str
    ops: Dict[str, OpStats] = field(default_factory=dict)
    types: Dict[str, TypeStats] = field(default_factory=dict)

    def add_op(self, op: OpStats):
        if op.name in self.ops:
            assert "op was already in dialect"
        self.ops[op.name] = op

    def add_type(self, typ: TypeStats):
        if typ.name in self.types:
            assert "type was already in dialect"
        self.types[typ.name] = typ


@dataclass
class Stats:
    dialects: Dict[str, DialectStats] = field(default_factory=dict)

    def add_op(self, op: OpStats):
        if op.dialect not in self.dialects:
            self.dialects[op.dialect] = DialectStats(op.dialect)
        self.dialects[op.dialect].add_op(op)

    def add_type(self, typ: TypeStats):
        if typ.dialect not in self.dialects:
            self.dialects[typ.dialect] = DialectStats(typ.dialect)
        self.dialects[typ.dialect].add_type(typ)

    def add_stats(self, stats: 'Stats'):
        for op in stats.ops:
            self.add_op(op)
        for typ in stats.types:
            self.add_type(typ)

    @staticmethod
    def from_json(json):
        stats = Stats()
        for val in json["ops"]:
            stats.add_op(OpStats.from_json(val))
        for val in json["types"]:
            stats.add_type(TypeStats.from_json(val))
        return stats

    @property
    def ops(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, op in dialect.ops.items():
                res.append(op)
        return res

    @property
    def types(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, typ in dialect.types.items():
                res.append(typ)
        return res


def get_stat_from_file(file) -> Optional[Stats]:
    root, file = os.path.split(file)
    res = subprocess.run(["../build/bin/tblgen-stats", os.path.join(root, file), "--I=../llvm-project/mlir/include",
                          f"--I={root}"], capture_output=True)
    if res.returncode != 0:
        return None
    print("../build/bin/tblgen-stats", os.path.join(root, file), "--I=../llvm-project/mlir/include",
          f"--I={root}")
    ops = json.loads(res.stdout)
    return Stats.from_json(ops)


def get_stat_from_files():
    stats = Stats()
    for file in get_tablegen_op_file_list():
        file_stats = get_stat_from_file(file)
        if file_stats is not None:
            stats.add_stats(file_stats)
    return stats


def get_op_list_distribution(ops: List[OpStats], f: Callable[[OpStats], int], maxi: Optional[int] = None) -> List[
    float]:
    res = [f(op) for op in ops]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_op_distribution(stats: Stats, f: Callable[[OpStats], int]) -> Tuple[List[float], Dict[str, List[float]]]:
    global_res = get_op_list_distribution(stats.ops, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_op_list_distribution(list(dialect.ops.values()), f, maxi=maxi)
    return global_res, per_dialect_res


def get_type_list_distribution(types: List[TypeStats], f: Callable[[TypeStats], int], maxi: Optional[int] = None) -> List[
    float]:
    res = [f(typ) for typ in types]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_type_distribution(stats: Stats, f: Callable[[TypeStats], int]) -> Tuple[List[float], Dict[str, List[float]]]:
    global_res = get_type_list_distribution(stats.types, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_type_list_distribution(list(dialect.types.values()), f, maxi=maxi)
    return global_res, per_dialect_res


def __main__():
    stats = get_stat_from_files()
    print(stats.types)
    print("Number of regions")
    print(get_global_op_distribution(stats, lambda x: x.numRegions)[0])

    print("Number of operands")
    print(get_global_op_distribution(stats, lambda x: x.numOperands)[0])

    print("Number of variable length operands")
    print(get_global_op_distribution(stats, lambda x: x.numVariableLengthOperands)[0])

    print("Number of results")
    print(get_global_op_distribution(stats, lambda x: x.numResults)[0])

    print("Has a verifier")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasVerifier else 0)[0])

    print("Has custom assembly format")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasAssemblyFormat else 0)[0])

    print("Fully declarative")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative() else 0)[0])

    print("Type parameters")
    print(get_global_type_distribution(stats, lambda x: x.numParameters)[0])


if __name__ == "__main__":
    __main__()
