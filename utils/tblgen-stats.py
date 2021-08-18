import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import *
import re


def check_balanced_parentheses(val: str) -> bool:
    paren_level = 0
    for char in val:
        if char == '(':
            paren_level += 1
        if char == ')':
            paren_level -= 1
            if paren_level < 0:
                return False
    return paren_level == 0


def remove_outer_parentheses(val: str) -> str:
    val = val.strip()
    assert(check_balanced_parentheses(val))
    if val[0] != '(' or val[-1] != ')':
        return val
    paren_level = 0
    for idx, char in enumerate(val):
        if char == '(':
            paren_level += 1
        if char == ')':
            paren_level -= 1
            if paren_level == 0 and idx != len(val) - 1:
                return val
            assert paren_level >= 0
    return remove_outer_parentheses(val[1:-1])


def separate_on_operator(val: str, operator: str) -> Optional[Tuple[str, str]]:
    val = remove_outer_parentheses(val)
    paren_level = 0
    for idx, char in enumerate(val):
        if char == '(':
            paren_level += 1
        if char == ')':
            paren_level -= 1
            assert paren_level >= 0
        if paren_level == 0:
            if val[idx:idx+len(operator)] == operator:
                return val[0:idx], val[idx+len(operator):]
    return None

# OPENMP and OPENACC are not included since they rely on generated tablegen
# files
# DLTI is removed because it contains no operations
def get_tablegen_op_file_list():
    res_files = []
    for root, dirs, files in os.walk("../llvm-project/mlir/include"):
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
        if typ.__origin__ == dict:
            assert typ.__args__[0] == str
            arg_typ = typ.__args__[1]
            res = dict()
            for key, val in json.items():
                res[key] = _from_json(val, arg_typ)
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
class TraitStats:
    @staticmethod
    def from_json(json):
        if json["kind"] == "native":
            return NativeTraitStats.from_json(json)
        if json["kind"] == "pred":
            return PredTraitStats.from_json(json)
        if json["kind"] == "internal":
            return InternalTraitStats.from_json(json)
        assert False

    def is_declarative(self) -> bool:
        raise NotImplemented


@from_json
class NativeTraitStats(TraitStats):
    name: str

    def is_declarative(self) -> bool:
        # Do not have verifiers. Note, we may need to add dependencies between traits
        if self.name == "::mlir::OpTrait::IsCommutative":
            return True
        if self.name == "::mlir::OpTrait::Scalarizable":
            return True
        if self.name == "::mlir::OpTrait::Vectorizable":
            return True
        if self.name == "::mlir::OpTrait::Tensorizable":
            return True
        if self.name == "::mlir::OpTrait::spirv::UsableInSpecConstantOp":
            return True
        if self.name == "::mlir::OpTrait::spirv::UnsignedOp":
            return True
        if self.name == "::mlir::OpTrait::spirv::SignedOp":
            return True

        # Have verifiers, but should be builtins
        if self.name == "::mlir::OpTrait::IsTerminator":
            return True
        m = re.compile(r"::mlir::OpTrait::HasParent<(.*)>::Impl").match(self.name)
        if m is not None:
            return True

        # Are replaced by IRDL way of doing things
        if self.name == "::mlir::OpTrait::SameOperandsAndResultType":
            return True
        if self.name == "::mlir::OpTrait::SameTypeOperands":
            return True
        if self.name == "::mlir::OpTrait::Elementwise":
            return True
        m = re.compile(r"::mlir::OpTrait::SingleBlockImplicitTerminator<(.*)>::Impl").match(self.name)
        if m is not None:
            return True

        # Cannot be replaced by IRDL for now
        if self.name == "::mlir::OpTrait::SameOperandsAndResultShape":
            return False
        return False


@from_json
class PredTraitStats(TraitStats):
    pred: str

    def is_declarative(self) -> bool:
        return False


@from_json
class InternalTraitStats(TraitStats):
    name: str

    def is_declarative(self) -> bool:
        return False


@from_json
class InterfaceStats:
    name: str


@dataclass
class ConstraintStats:
    kind: str

    @staticmethod
    def from_predicate(predicate: str) -> 'ConstraintStats':
        predicate = remove_outer_parentheses(predicate)

        # Check if this is a IsaCPPConstraint
        m = re.compile(r"\$_self.isa<(.*)>\(\)").match(predicate)
        if m is not None:
            return IsaCppTypeConstraintStats(m.group(0))

        and_operands = separate_on_operator(predicate, "&&")
        if and_operands is not None:
            operand1 = ConstraintStats.from_predicate(and_operands[0])
            operand2 = ConstraintStats.from_predicate(and_operands[1])
            if operand1 is not None and operand2 is not None:
                return AndConstraintStats(operand1, operand2)

        or_operands = separate_on_operator(predicate, "||")
        if or_operands is not None:
            operand1 = ConstraintStats.from_predicate(or_operands[0])
            operand2 = ConstraintStats.from_predicate(or_operands[1])
            if operand1 is not None and operand2 is not None:
                return OrConstraintStats(operand1, operand2)

        return PredicateConstraintStats.from_predicate(predicate)


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
        assert json["kind"] == "predicate"
        return ConstraintStats.from_predicate(json["predicate"])


    def is_declarative(self, in_tablegen: bool) -> bool:
        raise NotImplemented


@from_json
class VariadicConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self, in_tablegen: bool) -> bool:
        return self.baseType.is_declarative(in_tablegen)


@from_json
class OptionalConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self, in_tablegen: bool) -> bool:
        return self.baseType.is_declarative(in_tablegen)


@from_json
class TypeDefConstraintStats(ConstraintStats):
    dialect: str
    name: str

    def is_declarative(self, in_tablegen: bool) -> bool:
        return True


@from_json
class IntegerConstraintStats(ConstraintStats):
    bitwidth: int

    def is_declarative(self, in_tablegen: bool) -> bool:
        return True


@dataclass(eq=False)
class IsaCppTypeConstraintStats(ConstraintStats):
    name: str

    def is_declarative(self, in_tablegen: bool) -> bool:
        return not in_tablegen

    def __init__(self, name: str):
        self.kind = "isaCppType"
        self.name = name


@dataclass(eq=False)
class AndConstraintStats(ConstraintStats):
    operand1: ConstraintStats
    operand2: ConstraintStats

    def __init__(self, operand1: ConstraintStats, operand2: ConstraintStats):
        self.operand1 = operand1
        self.operand2 = operand2
        self.kind = "and"

    def is_declarative(self, in_tablegen: bool) -> bool:
        if in_tablegen:
            return False
        return self.operand1.is_declarative(in_tablegen) and self.operand2.is_declarative(in_tablegen)


@dataclass(eq=False)
class OrConstraintStats(ConstraintStats):
    operand1: ConstraintStats
    operand2: ConstraintStats

    def __init__(self, operand1: ConstraintStats, operand2: ConstraintStats):
        self.operand1 = operand1
        self.operand2 = operand2
        self.kind = "or"

    def is_declarative(self, in_tablegen: bool) -> bool:
        if in_tablegen:
            return False
        return self.operand1.is_declarative(in_tablegen) and self.operand2.is_declarative(in_tablegen)


@from_json
class PredicateConstraintStats(ConstraintStats):
    predicate: str

    @staticmethod
    def from_predicate(predicate: str) -> 'PredicateConstraintStats':
        return PredicateConstraintStats(kind="predicate", predicate=predicate)

    def is_declarative(self, in_tablegen: bool) -> bool:
        if in_tablegen:
            return False

        if remove_outer_parentheses(self.predicate) == "::mlir::LLVM::isCompatibleType($_self)":
            return True

        if self.predicate == "true":
            return True

        m = re.compile(r"\$_self.isInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.isSignlessInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.isF(.*)\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getElementType\(\).is(.*)Integer\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getElementType\(\).isa<(.*)>\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"::mlir::LLVM::getVectorElementType\(\$_self\).isa<(.*)>\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"::mlir::LLVM::getVectorElementType\(\$_self\).isSignlessInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"::mlir::LLVM::getVectorElementType\(\$_self\).isSignedInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"::mlir::LLVM::getVectorElementType\(\$_self\).isUnsignedInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getElementType\(\).isF(.*)\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getElementType\(\).isBF(.*)\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.isUnsignedInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.isSignedInteger\((.*)\)").match(self.predicate)
        if m is not None:
            return True

        if self.predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self)":
            return True

        if self.predicate == "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))":
            return True

        if self.predicate == "$_self.cast<::mlir::ShapedType>().getElementType().isSignedInteger()":
            return True

        m = re.compile(r"!\(\(\$_self.isa<(.*)>\(\)\)\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"!\$_self.isa<(.*)>\(\)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getRank\(\)                          == (.*)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getRank\(\)                            == (.*)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getElementType\(\).cast<(.*)>\(\).getStorageTypeIntegralWidth\(\) == (.*)").match(self.predicate)
        if m is not None:
            return True

        if self.predicate == "::mlir::LLVM::isCompatibleVectorType($_self)":
            return True

        if self.predicate == "$_self.cast<::mlir::ShapedType>().hasRank()":
            return True

        # Harder cases:
        return False
        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\)                            == (.*)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).hasStaticShape\(\)").match(self.predicate)
        if m is not None:
            return True

        assert False


@from_json
class NamedConstraintStats:
    name: str
    constraint: ConstraintStats

    def is_declarative(self, in_tablegen: bool) -> bool:
        return self.constraint.is_declarative(in_tablegen)


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
    attributes: Dict[str, ConstraintStats]
    traits: List[TraitStats]
    interfaces: List[InterfaceStats]

    def is_declarative(self, in_tablegen: bool, check_traits: bool = True, check_interfaces: bool = True) -> bool:
        if self.hasVerifier:
            return False
        if not in_tablegen and check_traits:
            for trait in self.traits:
                if not trait.is_declarative():
                    return False
        if not in_tablegen and check_interfaces and len(self.interfaces) > 0:
            return False
        for operand in self.operands:
            if not operand.is_declarative(in_tablegen):
                return False
        for result in self.results:
            if not result.is_declarative(in_tablegen):
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


@from_json
class AttrStats:
    name: str
    dialect: str
    numParameters: int
    parameters: List[AttrOrTypeParameterStats]


@dataclass
class DialectStats:
    name: str
    ops: Dict[str, OpStats] = field(default_factory=dict)
    types: Dict[str, TypeStats] = field(default_factory=dict)
    attrs: Dict[str, AttrStats] = field(default_factory=dict)
    numOperations: int = field(default=0)
    numTypes: int = field(default=0)

    def add_op(self, op: OpStats):
        if op.name in self.ops.keys():
            assert "op was already in dialect"
        self.ops[op.name] = op

    def add_type(self, typ: TypeStats):
        if typ.name in self.types.keys():
            assert "type was already in dialect"
        self.types[typ.name] = typ

    def add_attr(self, attr: AttrStats):
        if attr.name in self.attrs.keys():
            assert "attr was already in dialect"
        self.attrs[attr.name] = attr


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

    def add_attr(self, attr: AttrStats):
        if attr.dialect not in self.dialects:
            self.dialects[attr.dialect] = DialectStats(attr.dialect)
        self.dialects[attr.dialect].add_attr(attr)

    def add_stats(self, stats: 'Stats'):
        for op in stats.ops:
            self.add_op(op)
        for typ in stats.types:
            self.add_type(typ)
        for attr in stats.attrs:
            self.add_attr(attr)

    @staticmethod
    def from_json(json):
        stats = Stats()
        for val in json["ops"]:
            stats.add_op(OpStats.from_json(val))
        for val in json["types"]:
            stats.add_type(TypeStats.from_json(val))
        for val in json["attrs"]:
            stats.add_attr(AttrStats.from_json(val))
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

    @property
    def attrs(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, attr in dialect.attrs.items():
                res.append(attr)
        return res


def get_stat_from_file(file) -> Optional[Stats]:
    root, file = os.path.split(file)
    res = subprocess.run(["../build/bin/tblgen-stats", os.path.join(root, file), "--I=../llvm-project/mlir/include",
                          f"--I={root}"], capture_output=True)
    if res.returncode != 0:
        print(res.stderr)
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

    res = subprocess.run(["../build/bin/dialect-stats"], capture_output=True)
    if res.returncode != 0:
        return None
    data = json.loads(res.stdout)
    for json_dialect in data:
        dialect_stats = stats.dialects[json_dialect["dialect"]]
        dialect_stats.numOperations = json_dialect["numOperations"]
        dialect_stats.numTypes = json_dialect["numTypes"]

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


def get_attr_list_distribution(attrs: List[AttrStats], f: Callable[[AttrStats], int], maxi: Optional[int] = None) -> List[
    float]:
    res = [f(attr) for attr in attrs]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_attr_distribution(stats: Stats, f: Callable[[AttrStats], int]) -> Tuple[List[float], Dict[str, List[float]]]:
    global_res = get_attr_list_distribution(stats.attrs, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_attr_list_distribution(list(dialect.attrs.values()), f, maxi=maxi)
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


T = TypeVar("T")


def get_dialect_values(stats: Stats, f: Callable[[DialectStats], T]) -> Dict[str, T]:
    return {key: f(value) for key, value in stats.dialects.items()}


def __main__():
    stats = get_stat_from_files()

    trait_values = dict()
    for op in stats.ops:
        for trait in op.traits:
            if isinstance(trait, NativeTraitStats):
                if not trait.is_declarative():
                    trait_values[trait.name] = 0
    for op in stats.ops:
        for trait in op.traits:
            if isinstance(trait, NativeTraitStats):
                if not trait.is_declarative():
                    trait_values[trait.name] += 1
    print({k: v for k, v in sorted(trait_values.items(), key=lambda item: -item[1])})


    print("Number of operations defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.ops), x.numOperations)))
    print("total:", (len(stats.ops), sum([dialect.numOperations for dialect in stats.dialects.values()]), ))

    print("Number of types defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.types), x.numTypes)))
    print("total:", (len(stats.types), sum([dialect.numTypes for dialect in stats.dialects.values()]), ))

    print("Number of regions")
    print(get_global_op_distribution(stats, lambda x: x.numRegions)[0])

    print("Number of operands")
    print(get_global_op_distribution(stats, lambda x: x.numOperands)[0])

    print("Number of results")
    print(get_global_op_distribution(stats, lambda x: x.numResults)[0])

    print("Number of attributes")
    print(get_global_op_distribution(stats, lambda x: len(x.attributes))[0])

    print("Has a verifier")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasVerifier else 0))

    print("Has custom assembly format")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasAssemblyFormat else 0)[0])

    print("Fully declarative in tablegen")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(True) else 0)[0])

    print("Fully declarative in IRDL")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(in_tablegen=False, check_traits=True, check_interfaces=True) else 0))

    print("Fully declarative in IRDL without interfaces")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(in_tablegen=False, check_traits=True, check_interfaces=False) else 0))

    print("Fully declarative in IRDL without traits or interfaces")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(in_tablegen=False, check_traits=False, check_interfaces=False) else 0))

    print("Type parameters")
    print(get_global_type_distribution(stats, lambda x: x.numParameters)[0])

    print("Attr parameters")
    print(get_global_attr_distribution(stats, lambda x: x.numParameters)[0])

if __name__ == "__main__":
    __main__()
