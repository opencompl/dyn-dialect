from __future__ import annotations
import json
import os
import subprocess
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import *
import re

verifiers_allow_len_equality = False
indent_size = 2

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


def simplify_expression(val: str) -> str:
    val = remove_outer_parentheses(val)
    val = re.sub(" +", " ", val)
    return val


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
    @dataclass(eq=True, unsafe_hash=True)
    class FromJsonWrapper(dataclass(cls)):
        def __repr__(self):
            return cls.__name__[:-5] + "(" + ", ".join([f"{key}={self.__dict__[key]}" for key in
                                                        cls.__dataclass_fields__.keys()]) + ")"

        @staticmethod
        def from_json(json):
            arg_dict = dict()
            for name, typ in get_type_hints(cls).items():
                arg_dict[name] = _from_json(json[name], typ)
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
        if self.name == "::mlir::OpTrait::HasRecursiveSideEffects":
            return True
        if self.name == "::mlir::OpTrait::MemRefsNormalizable":
            return True

        # Have verifiers, but should be builtins
        if self.name == "::mlir::OpTrait::IsTerminator":
            return True
        m = re.compile(r"::mlir::OpTrait::HasParent<(.*)>::Impl").match(self.name)
        if m is not None:
            return True
        if self.name == "::mlir::OpTrait::IsIsolatedFromAbove":
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
        if self.name == "::mlir::OpTrait::AttrSizedOperandSegments":
            return True

        # Cannot be replaced by IRDL for now
        if self.name == "::mlir::OpTrait::SameOperandsAndResultShape":
            return False

        return False


@from_json
class PredTraitStats(TraitStats):
    pred: str

    def is_declarative(self) -> bool:
        if re.compile("\(getElementTypeOrSelf\(\$_op.getResult\((.*)\)\) == getElementTypeOrSelf\(\$_op.getOperand\((.*)\)\)\)").match(self.pred) is not None:
            return True
        if self.pred == "(std::equal_to<>()($tensor.getType().cast<ShapedType>().getElementType(), $result.getType()))":
            return True
        return False


@from_json
class InternalTraitStats(TraitStats):
    name: str

    def is_declarative(self) -> bool:
        return False


@dataclass(eq=True, unsafe_hash=True)
class ConstraintStats(ABC):
    @staticmethod
    def from_predicate(predicate: str) -> 'ConstraintStats':
        predicate = simplify_expression(predicate)

        m = re.compile(r"\$_self.isa<(.*)>\(\)").match(predicate)
        if m is not None:
            return BaseConstraintStats(predicate[11:-3])

        m = re.compile(r"!\((.*)\)").match(predicate)
        if m is not None:
            constraint = ConstraintStats.from_predicate(m.group(0)[2:-1])
            return NotConstraintStats(constraint)

        and_operands = separate_on_operator(predicate, "&&")
        if and_operands is not None:
            operand1 = ConstraintStats.from_predicate(and_operands[0])
            operand2 = ConstraintStats.from_predicate(and_operands[1])
            return AndConstraintStats(operand1, operand2)

        or_operands = separate_on_operator(predicate, "||")
        if or_operands is not None:
            operand1 = ConstraintStats.from_predicate(or_operands[0])
            operand2 = ConstraintStats.from_predicate(or_operands[1])
            return OrConstraintStats([operand1, operand2])

        m = re.compile(r"!(.*)").match(predicate)
        if m is not None:
            operand = ConstraintStats.from_predicate(m.group(1))
            return NotConstraintStats(operand)

        if predicate == "true":
            return AnyConstraintStats()

        m = re.match(r"\$_self.isInteger\((.*)\)", predicate)
        if m is not None:
            return ParametricTypeConstraintStats("Builtin_Integer", [IntEqParamConstraintStats(int(m.group(1))),
                                                                     AnyParamConstraintStats()])

        m = re.match(r"\$_self.isSignlessInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraintStats("Builtin_Integer", [AnyParamConstraintStats(), CppValueParamConstraintStats("Signless", True)])
            return ParametricTypeConstraintStats("Builtin_Integer", [IntEqParamConstraintStats(int(val)), CppValueParamConstraintStats("Signless", True)])

        m = re.match(r"\$_self.isUnsignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraintStats("Builtin_Integer", [AnyParamConstraintStats(), CppValueParamConstraintStats("Unsigned", True)])
            return ParametricTypeConstraintStats("Builtin_Integer", [IntEqParamConstraintStats(int(val)), CppValueParamConstraintStats("Unsigned", True)])

        m = re.match(r"\$_self.isSignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraintStats("Builtin_Integer", [AnyParamConstraintStats(), CppValueParamConstraintStats("Signed", True)])
            return ParametricTypeConstraintStats("Builtin_Integer", [IntEqParamConstraintStats(int(val)), CppValueParamConstraintStats("Signed", True)])

        if predicate == "$_self.isBF16()":
            return ParametricTypeConstraintStats("Builtin_BFloat16", [])

        m = re.match(r"\$_self.isF(.*)\(\)", predicate)
        if m is not None:
            return ParametricTypeConstraintStats("Builtin_Float" + m.group(1), [])

        if predicate == "$_self.isIndex()":
            return ParametricTypeConstraintStats("Builtin_Index", [])

        m = re.match(r"\$_self.cast<::mlir::FloatAttr>\(\).getType\(\)(.*)", predicate)
        if m is not None:
            type_predicate = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("Builtin_FloatAttr", [type_predicate, AnyParamConstraintStats()])

        m = re.match(r"\$_self.cast<::mlir::ShapedType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            element_type_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ShapedTypeConstraintStats(element_type_constraint)

        m = re.match(r"::mlir::LLVM::getVectorElementType\(\$_self\)(.*)", predicate)
        if m is not None:
            element_type_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return LLVMVectorOfConstraintStats(element_type_constraint)

        m = re.match(r"\$_self.cast<::mlir::DenseIntElementsAttr>\(\)( ?).getType\(\)( ?).getElementType\(\)( ?)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(4))
            # TODO find DenseIntElementsAttr
            return ParametricTypeConstraintStats("DenseIntElementsAttr", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::arm_sve::ScalableVectorType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("ScalableVectorType", [AnyParamConstraintStats(), sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::LLVM::LLVMPointerType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("ptr", [sub_constraint, AnyParamConstraintStats()])
        
        m = re.match(r"\$_self.cast<::mlir::spirv::CooperativeMatrixNVType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("coopmatrix", [sub_constraint, AnyParamConstraintStats(), AnyParamConstraintStats(), AnyParamConstraintStats()])

        m = re.match(r"\$_self.cast<::mlir::pdl::RangeType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("PDL_Range", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::gpu::MMAMatrixType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("PDL_Range", [AnyParamConstraintStats(), AnyParamConstraintStats(), sub_constraint, AnyParamConstraintStats()])

        m = re.match(r"\$_self.cast<::mlir::ComplexType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("Complex", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::IntegerAttr>\(\).getType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = ConstraintStats.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraintStats("Builtin_IntegerAttr", [sub_constraint, AnyParamConstraintStats()])

        m = re.match(r"::llvm::all_of\(\$_self.cast<::mlir::ArrayAttr>\(\), \[&\]\(::mlir::Attribute attr\) { return (.*); }\)", predicate)
        if m is not None:
            group = re.sub("attr\.", "$_self.", m.group(1))
            sub_constraint = ConstraintStats.from_predicate(group)
            return AttrArrayOf(sub_constraint)

        m = re.match(r"::llvm::all_of\(\$_self.cast<::mlir::TupleType>\(\).getTypes\(\), \[\]\(Type t\) { return (.*); }\)", predicate)
        if m is not None:
            group = re.sub("t\.", "$_self.", m.group(1))
            sub_constraint = ConstraintStats.from_predicate(group)
            return TupleOf(sub_constraint)

        m = re.match(r"\$_self.cast<(::mlir::)?StringAttr>\(\).getValue\(\) == \"(.*)\"", predicate)
        if m is not None:
            str_val = m.group(1)
            return ParametricTypeConstraintStats("Builtin_StringAttr", [StringEqParamConstraintStats(str_val)])

        llvm_float_types = ["Builtin_BFloat16", "Builtin_Float16", "Builtin_Float32", "Builtin_Float64", "Builtin_Float80", "Builtin_Float128", "ppc_fp128"]

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self)":
            return OrConstraintStats([BaseConstraintStats(typ) for typ in llvm_float_types])

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))":
            element_type_constraint = OrConstraintStats([BaseConstraintStats(typ) for typ in llvm_float_types])
            return LLVMVectorOfConstraintStats(element_type_constraint)

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            sub_constraint = OrConstraintStats([BaseConstraintStats(typ) for typ in llvm_float_types])
            return ParametricTypeConstraintStats("ptr", [sub_constraint, AnyParamConstraintStats()])

        if predicate == "::mlir::LLVM::isCompatibleType($_self)":
            return LLVMCompatibleType()

        if predicate == "::mlir::LLVM::isCompatibleType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            return ParametricTypeConstraintStats("ptr", [LLVMCompatibleType(), AnyParamConstraintStats()])

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

    @abstractmethod
    def is_declarative(self) -> bool:
        ...

    @abstractmethod
    def get_sub_constraints(self) -> List[ConstraintStats]:
        ...


@from_json
class VariadicConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.baseType]

    def __str__(self):
        return f"Variadic<{self.baseType}>"


@from_json
class OptionalConstraintStats(ConstraintStats):
    baseType: ConstraintStats

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.baseType]

    def __str__(self):
        return f"Optional<{self.baseType}>"


@from_json
class TypeDefConstraintStats(ConstraintStats):
    dialect: str
    name: str

    def is_declarative(self) -> bool:
        return True

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"{self.dialect}.{self.name}"


@from_json
class IntegerConstraintStats(ConstraintStats):
    bitwidth: int

    def is_declarative(self) -> bool:
        return True

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"IntegerOfSize<{self.bitwidth}>"


@dataclass(eq=False)
class AnyConstraintStats(ConstraintStats):
    def is_declarative(self) -> bool:
        return True

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"Any"


@dataclass(eq=False)
class BaseConstraintStats(ConstraintStats):
    name: str

    def is_declarative(self) -> bool:
        return True

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"{self.name}"


@dataclass(eq=False)
class ParametricTypeConstraintStats(ConstraintStats):
    base: str
    params: List[Union[ParamConstraintStats, ConstraintStats]]

    def is_declarative(self) -> bool:
        return all(param.is_declarative() for param in self.params)

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [param for param in self.params if isinstance(param, ConstraintStats)]

    def __str__(self):
        return f"{self.base}<{', '.join([str(param) for param in self.params])}>"


@dataclass(eq=False)
class NotConstraintStats(ConstraintStats):
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.constraint]

    def __str__(self):
        return f"Not<{self.constraint}>"


@dataclass(eq=False)
class AndConstraintStats(ConstraintStats):
    operand1: ConstraintStats
    operand2: ConstraintStats

    def is_declarative(self) -> bool:
        return self.operand1.is_declarative() and self.operand2.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.operand1, self.operand2]

    def __str__(self):
        return f"And<{self.operand1}, {self.operand2}>"


@dataclass(eq=False)
class OrConstraintStats(ConstraintStats):
    operands: List[ConstraintStats]

    def __init__(self, operands: List[ConstraintStats]):
        self.operands = []
        for operand in operands:
            if isinstance(operand, OrConstraintStats):
                for sub_operand in operand.operands:
                    self.operands.append(sub_operand)
            else:
                self.operands.append(operand)

    def is_declarative(self) -> bool:
        return all([c.is_declarative for c in self.operands])

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return self.operands

    def __str__(self):
        return f"AnyOf<{', '.join([str(operand) for operand in self.operands])}>"


@dataclass(eq=False)
class NotConstraintStats(ConstraintStats):
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.constraint]

    def __str__(self):
        return f"Not<{self.constraint}>"


@dataclass(eq=False)
class ShapedTypeConstraintStats(ConstraintStats):
    elemTypeConstraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.elemTypeConstraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.elemTypeConstraint]

    def __str__(self):
        return f"ShapedTypeOf<{self.elemTypeConstraint}>"


@dataclass(eq=False)
class LLVMVectorOfConstraintStats(ConstraintStats):
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.constraint]

    def __str__(self):
        return f"LLVMVectorOf<{self.constraint}>"


@dataclass(eq=False)
class AttrArrayOf(ConstraintStats):
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.constraint]

    def __str__(self):
        return f"AttrArrayOf<{self.constraint}>"


@dataclass(eq=False)
class TupleOf(ConstraintStats):
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return [self.constraint]

    def __str__(self):
        return f"TupleOf<{self.constraint}>"


@dataclass(eq=False)
class LLVMCompatibleType(ConstraintStats):
    def is_declarative(self) -> bool:
        return True

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"LLVMCompatibleType"


@from_json
class PredicateConstraintStats(ConstraintStats):
    predicate: str

    @staticmethod
    def from_predicate(predicate: str) -> 'PredicateConstraintStats':
        return PredicateConstraintStats(predicate)

    def is_declarative(self) -> bool:
        self.predicate = simplify_expression(self.predicate)
        m = re.compile(r"\$_self.cast<(.*)>\(\).getRank\(\) == (.*)").match(self.predicate)
        if m is not None:
            return True

        if self.predicate == "::mlir::LLVM::isCompatibleVectorType($_self)":
            return True

        if self.predicate == "$_self.cast<::mlir::ShapedType>().hasRank()":
            return True

        if self.predicate == "$_self.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()":
            return True

        if re.compile(r"\$_self.cast<::mlir::ArrayAttr>\(\).size\(\) == (.*)").match(self.predicate) is not None:
            return True

        if re.compile(r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[0\].isa<(.*)>\(\)").match(self.predicate) is not None:
            return True

        if re.compile(r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[1\].isSignlessInteger\((.*)\)").match(self.predicate) is not None:
            return True

        if re.compile("::mlir::spirv::symbolize(.*)").match(self.predicate):
            return True

        if re.compile("\$_self.cast<mlir::quant::QuantizedType>\(\).getStorageTypeIntegralWidth\(\) == (.*)").match(self.predicate) is not None:
            return True

        # Harder cases:
        if self.predicate == "$_self.cast<::mlir::ArrayAttr>().size() <= 4":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() >= 0":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() <= 3":
            return False

        if self.predicate == "$_self.cast<IntegerAttr>().getValue().isStrictlyPositive()":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getValue().isNegative()":
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(self.predicate)
        if m is not None:
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(self.predicate)
        if m is not None:
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).hasStaticShape\(\)").match(self.predicate)
        if m is not None:
            return False

        if self.predicate == "isStrided($_self.cast<::mlir::MemRefType>())":
            return False

        m = re.compile(r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\).size\(\) == (.*)").match(self.predicate)
        if m is not None:
            return False

        m = re.compile(r"(.*).cast<::mlir::LLVM::LLVMStructType>\(\).isOpaque\(\)").match(self.predicate)
        if m is not None:
            return False

        print(self.predicate)
        assert False

    def get_sub_constraints(self) -> List[ConstraintStats]:
        return []

    def __str__(self):
        return f"Predicate<\"{self.predicate}\">"


@dataclass(eq=False)
class ParamConstraintStats(ABC):
    @abstractmethod
    def is_declarative(self) -> bool:
        ...


@dataclass(eq=False)
class AnyParamConstraintStats(ParamConstraintStats):
    def is_declarative(self) -> bool:
        return True

    def __str__(self):
        return f"Any"


@dataclass(eq=False)
class CppValueParamConstraintStats(ParamConstraintStats):
    value: str
    is_decl: bool

    def is_declarative(self) -> bool:
        return self.is_decl

    def __str__(self):
        return f'"{self.value}"'


@dataclass(eq=False)
class IntEqParamConstraintStats(ParamConstraintStats):
    value: int

    def is_declarative(self) -> bool:
        return True

    def __str__(self):
        return f"{self.value}"


@dataclass(eq=False)
class StringEqParamConstraintStats(ParamConstraintStats):
    value: str

    def is_declarative(self) -> bool:
        return True

    def __str__(self):
        return f"{self.value}"


@from_json
class NamedConstraintStats:
    name: str
    constraint: ConstraintStats

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def __str__(self):
        return f"{self.name}: {self.constraint}"


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
    interfaces: List[str]

    def is_operands_results_attrs_declarative(self) -> bool:
        for operand in self.operands:
            if not operand.is_declarative():
                return False
        for result in self.results:
            if not result.is_declarative():
                return False
        for name, attr in self.attributes.items():
            if not attr.is_declarative():
                return False
        return True

    def is_traits_declarative(self) -> bool:
        for trait in self.traits:
            if not trait.is_declarative():
                return False
        return True

    def is_interface_declarative(self) -> bool:
        return not len(self.interfaces) > 0

    def is_declarative(self, check_traits: bool = True, check_interfaces: bool = True) -> bool:
        if self.hasVerifier:
            return False
        if not self.is_operands_results_attrs_declarative():
            return False
        if check_traits and not self.is_traits_declarative():
            return False
        if check_interfaces and not self.is_interface_declarative():
            return False
        return True

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Operation {self.name} {{")

        # Operands
        if len(self.operands) != 0:
            print(f"{' ' * (indent_level + indent_size)}Operands (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Operands ('))}".join([str(operand) for operand in self.operands]), end='')
            print(")")

        # Results
        if len(self.results) != 0:
            print(f"{' ' * (indent_level + indent_size)}Results (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Results ('))}".join([str(result) for result in self.results]), end='')
            print(")")

        # Attributes
        if len(self.attributes) != 0:
            print(f"{' ' * (indent_level + indent_size)}Attributes (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Attributes ('))}".join([f'{name}: {attr}' for name, attr in self.attributes.items()]), end='')
            print(")")

        print(f"{' ' * indent_level}}}")


@from_json
class AttrOrTypeParameterStats:
    name: str
    cppType: str

    def get_group(self):
        base_names = ["Type", "::mlir::Type", "Attribute", "ShapedType", "DenseElementsAttr",
                      "DenseIntElementsAttr", "StringAttr", "VerCapExtAttr", "DictionaryAttr"]
        if self.cppType in base_names:
            return "type/attr"

        base_array_names = ["::llvm::ArrayRef<Attribute>", "::llvm::ArrayRef<NamedAttribute>", "ArrayRef<Type>",
                            "::llvm::ArrayRef<FlatSymbolRefAttr>"]
        if self.cppType in base_array_names:
            return "attr/type array"

        integer_names = ["unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt", "bool"]
        if self.cppType in integer_names:
            return "integer"

        int_array_names = ["::llvm::ArrayRef<int64_t>", "ArrayRef<int64_t>"]
        if self.cppType in int_array_names:
            return "integer array"

        float_names = ["double", "::llvm::APFloat"]
        if self.cppType in float_names:
            return "float"

        float_array_names = ["ArrayRef<double>"]
        if self.cppType in float_array_names:
            return "float array"

        string_names = ["Identifier", "::llvm::StringRef"]
        if self.cppType in string_names:
            return "string"

        string_array_names = ["ArrayRef<char>", "ArrayRef<StringRef>"]
        if self.cppType in string_array_names:
            return "string array"

        enum_names = ["Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo", "ImageSamplingInfo", "ImageSamplerUseInfo",
                 "ImageFormat", "StorageClass", "Optional<StorageClass>",
                 "Version", "Capability", "Extension", "Vendor",
                 "DeviceType", "CombiningKind", "SignednessSemantics", "FastmathFlags"]
        if self.cppType in enum_names:
            return "enum"

        enum_array_names = ["ArrayRef<OffsetInfo>", "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>",
                            "ArrayRef<MemberDecorationInfo>", "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
                            "ArrayRef<Capability>", "ArrayRef<Extension>"]
        if self.cppType in enum_array_names:
            return "enum array"

        other_names = ["TypeID", "Location", "::llvm::ArrayRef<Location>", "AffineMap", "::llvm::ArrayRef<AffineMap>",
                       "IntegerSet", "TODO"]
        if self.cppType in other_names:
            return "other"

        assert False

    def is_declarative(self, builtins=True, enums=True):
        assert (not enums or builtins)
        base_names = ["Type", "::mlir::Type", "Attribute", "ShapedType", "DenseElementsAttr",
        "DenseIntElementsAttr", "StringAttr", "VerCapExtAttr", "DictionaryAttr"]
        if self.cppType in base_names:
            return True

        # integers and floats
        builtin_names = ["unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt", "::llvm::APFloat",
                         "bool", "double", "Identifier", "::llvm::StringRef"]
        if self.cppType in builtin_names:
            return builtins

        # arrays
        arrays = ["::llvm::ArrayRef<int64_t>", "::llvm::ArrayRef<Attribute>", "::llvm::ArrayRef<NamedAttribute>",
            "ArrayRef<Type>", "ArrayRef<char>",
            "ArrayRef<StringRef>", "::llvm::ArrayRef<FlatSymbolRefAttr>", "ArrayRef<double>", "ArrayRef<int64_t>"]
        if self.cppType in arrays:
            return builtins

        # Enums
        enum_names = ["Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo", "ImageSamplingInfo", "ImageSamplerUseInfo",
                 "ImageFormat", "StorageClass", "Optional<StorageClass>", "ArrayRef<OffsetInfo>",
                 "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>", "ArrayRef<MemberDecorationInfo>",
                 "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
                 "Version", "Capability", "ArrayRef<Capability>", "Extension", "ArrayRef<Extension>", "Vendor",
                 "DeviceType", "CombiningKind", "SignednessSemantics", "FastmathFlags"]
        if self.cppType in enum_names:
            return enums

        if self.cppType == "TypeID":
            return False
        if self.cppType == "Location":
            return False
        if self.cppType == "::llvm::ArrayRef<Location>":
            return False
        if self.cppType == "AffineMap":
            return False
        if self.cppType == "::llvm::ArrayRef<AffineMap>":
            return False
        if self.cppType == "IntegerSet":
            return False
        if self.cppType == "LLVMStruct":
            return False

        print(self.cppType)
        assert False


@from_json
class TypeStats:
    name: str
    dialect: str
    hasVerifier: bool
    parameters: List[AttrOrTypeParameterStats]
    traits: List[TraitStats]
    interfaces: List[str]

    def is_declarative(self, builtins=True, enums=True):
        for param in self.parameters:
            if not param.is_declarative(builtins=builtins, enums=enums):
                return False
        assert len(self.traits) == 0
        for interface in self.interfaces:
            if interface == "::mlir::SubElementTypeInterface::Trait":
                continue
            if interface == "DataLayoutTypeInterface::Trait":
                return False
            print(interface)
            assert False
        return True

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Type {self.name} {{")

        # Parameters
        print(f"{' ' * (indent_level + indent_size)}Parameters (", end='')
        print(', '.join([f"{param.name}: \"{param.cppType}\"" for param in self.parameters]), end='')
        print(f")")

        # Verifier
        if self.hasVerifier:
            print(f"{' ' * (indent_level + indent_size)}CppVerifier \"verify($_self)\"")

        # TODO traits and interfaces

        # Traits
        print(f"{' ' * indent_level}}}")


@from_json
class AttrStats:
    name: str
    dialect: str
    hasVerifier: bool
    parameters: List[AttrOrTypeParameterStats]
    traits: List[TraitStats]
    interfaces: List[str]

    def is_declarative(self, builtins=True, enums=True):
        for param in self.parameters:
            if not param.is_declarative(builtins=builtins, enums=enums):
                return False
        for interface in self.interfaces:
            if interface == "::mlir::SubElementAttrInterface::Trait":
                continue
            return False
        return True


@dataclass
class DialectStats:
    name: str
    ops: Dict[str, OpStats] = field(default_factory=dict)
    types: Dict[str, TypeStats] = field(default_factory=dict)
    attrs: Dict[str, AttrStats] = field(default_factory=dict)
    numOperations: int = field(default=0)
    numTypes: int = field(default=0)
    numAttributes: int = field(default=0)

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

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Dialect {self.name} {{")

        # Types
        for type in self.types.values():
            type.print(indent_level + indent_size)

        # TODO Attributes

        # Ops
        for op in self.ops.values():
            op.print(indent_level + indent_size)

        print(f"{' ' * indent_level}}}")


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
        return None
    print("../build/bin/tblgen-stats", os.path.join(root, file), "--I=../llvm-project/mlir/include",
          f"--I={root}")
    ops = json.loads(res.stdout)
    return Stats.from_json(ops)


def add_cpp_types(stats: Stats):
    # linalg
    stats.add_type(TypeStats("range", "linalg", False, [], [], []))

    # gpu
    stats.add_type(TypeStats("async.token", "gpu", False, [], [], []))
    # verifyCompatibleShape
    stats.add_type(TypeStats("mma_matrix", "gpu", True, [AttrOrTypeParameterStats("shape_x", "int64_t"),
                                                   AttrOrTypeParameterStats("shape_y", "int64_t"),
                                                   AttrOrTypeParameterStats("elementType", "Type"),
                                                   AttrOrTypeParameterStats("operand", "StringAttr")], [], []))
    # spv
    stats.add_type(TypeStats("array", "spv", False, [AttrOrTypeParameterStats("elementType", "Type"),
                                              AttrOrTypeParameterStats("elementCount", "unsigned"),
                                              AttrOrTypeParameterStats("stride", "unsigned")], [], []))
    stats.add_type(TypeStats("coopmatrix", "spv", False, [AttrOrTypeParameterStats("elementType", "Type"),
                                                   AttrOrTypeParameterStats("rows", "unsigned"),
                                                   AttrOrTypeParameterStats("columns", "unsigned"),
                                                   AttrOrTypeParameterStats("scope", "Scope")], [], []))
    stats.add_type(TypeStats("image", "spv", False, [AttrOrTypeParameterStats("elementType", "Type"),
                                              AttrOrTypeParameterStats("dim", "Dim"),
                                              AttrOrTypeParameterStats("depthInfo", "ImageDepthInfo"),
                                              AttrOrTypeParameterStats("arrayedInfo", "ImageArrayedInfo"),
                                              AttrOrTypeParameterStats("samplingInfo", "ImageSamplingInfo"),
                                              AttrOrTypeParameterStats("samplerUseInfo", "ImageSamplerUseInfo"),
                                              AttrOrTypeParameterStats("format", "ImageFormat")], [], []))
    stats.add_type(TypeStats("ptr", "spv", False, [AttrOrTypeParameterStats("pointeeType", "Type"),
                                            AttrOrTypeParameterStats("storageClass", "StorageClass")], [], []))
    stats.add_type(TypeStats("rtarray", "spv", False, [AttrOrTypeParameterStats("elementType", "Type"),
                                                AttrOrTypeParameterStats("stride", "int")], [], []))
    stats.add_type(TypeStats("sampled_image", "spv", False, [AttrOrTypeParameterStats("imageType", "Type")], [], []))
    stats.add_type(TypeStats("struct", "spv", False, [AttrOrTypeParameterStats("memberTypes", "ArrayRef<Type>"),
                                               AttrOrTypeParameterStats("offsetInfo", "ArrayRef<OffsetInfo>"),
                                               AttrOrTypeParameterStats("memberDecorations", "ArrayRef<MemberDecorationInfo>")], [], []))
    stats.add_type(TypeStats("matrix", "spv", False, [AttrOrTypeParameterStats("columnType", "Type"),
                                               AttrOrTypeParameterStats("columnCount", "uint32_t")], [], []))

    # llvm
    stats.add_type(TypeStats("void", "llvm", False, [], [], []))
    stats.add_type(TypeStats("ppc_fp128", "llvm", False, [], [], []))
    stats.add_type(TypeStats("x86mmx", "llvm", False, [], [], []))
    stats.add_type(TypeStats("token", "llvm", False, [], [], []))
    stats.add_type(TypeStats("label", "llvm", False, [], [], []))
    stats.add_type(TypeStats("metadata", "llvm", False, [], [], []))
    stats.add_type(TypeStats("func", "llvm", False, [AttrOrTypeParameterStats("result", "Type"),
                                              AttrOrTypeParameterStats("arguments", "ArrayRef<Type>"),
                                              AttrOrTypeParameterStats("isVarArg", "bool")], [], []))
    stats.add_type(TypeStats("ptr", "llvm", False, [AttrOrTypeParameterStats("pointee", "Type"),
                                             AttrOrTypeParameterStats("addressSpace", "unsigned")], [],
                             ["DataLayoutTypeInterface::Trait"]))
    # Check that a value is strictly positive
    stats.add_type(TypeStats("fixed_vec", "llvm", True, [AttrOrTypeParameterStats("elementType", "Type"),
                                                   AttrOrTypeParameterStats("numElements", "unsigned")], [], []))
    # Check that a value is strictly positive
    stats.add_type(TypeStats("scalable_vec", "llvm", True, [AttrOrTypeParameterStats("elementType", "Type"),
                                                      AttrOrTypeParameterStats("numElements", "unsigned")], [], []))
    stats.add_type(TypeStats("array", "llvm", False, [AttrOrTypeParameterStats("elementType", "Type"),
                                               AttrOrTypeParameterStats("numElements", "unsigned")], [], []))
    # Complex underlying type that requires non-trivial verifier
    stats.add_type(TypeStats("struct", "llvm", True, [AttrOrTypeParameterStats("arg", "LLVMStruct")], [], []))

    # shape
    stats.add_type(TypeStats("shape", "shape", False, [], [], []))
    stats.add_type(TypeStats("size", "shape", False, [], [], []))
    stats.add_type(TypeStats("value_shape", "shape", False, [], [], []))
    stats.add_type(TypeStats("witness", "shape", False, [], [], []))

    # quant
    # Complex verifier
    stats.add_type(TypeStats("any", "quant", False, [AttrOrTypeParameterStats("flags", "unsigned"),
                                              AttrOrTypeParameterStats("storageType", "Type"),
                                              AttrOrTypeParameterStats("expressedType", "Type"),
                                              AttrOrTypeParameterStats("storageTypeMin", "int64_t"),
                                              AttrOrTypeParameterStats("storageTypeMax", "int64_t")], [], []))
    # Complex verifier
    stats.add_type(TypeStats("uniform", "quant", False, [AttrOrTypeParameterStats("flags", "unsigned"),
                                                  AttrOrTypeParameterStats("storageType", "Type"),
                                                  AttrOrTypeParameterStats("expressedType", "Type"),
                                                  AttrOrTypeParameterStats("scale", "double"),
                                                  AttrOrTypeParameterStats("zeroPoint", "int64_t"),
                                                  AttrOrTypeParameterStats("storageTypeMin", "int64_t"),
                                                  AttrOrTypeParameterStats("storageTypeMax", "int64_t")], [], []))
    # Complex verifier
    stats.add_type(TypeStats("uniform_per_axis", "quant", False, [AttrOrTypeParameterStats("flags", "unsigned"),
                                                           AttrOrTypeParameterStats("storageType", "Type"),
                                                           AttrOrTypeParameterStats("expressedType", "Type"),
                                                           AttrOrTypeParameterStats("scales", "ArrayRef<double>"),
                                                           AttrOrTypeParameterStats("zeroPoints", "ArrayRef<int64_t>"),
                                                           AttrOrTypeParameterStats("quantizedDimension", "int64_t"),
                                                           AttrOrTypeParameterStats("storageTypeMin", "int64_t"),
                                                           AttrOrTypeParameterStats("storageTypeMax", "int64_t")], [], []))
    # Less or equal comparison
    stats.add_type(TypeStats("calibrated", "quant", False, [AttrOrTypeParameterStats("expressedType", "Type"),
                                                     AttrOrTypeParameterStats("min", "double"),
                                                     AttrOrTypeParameterStats("max", "double")], [], []))


def add_cpp_attributes(stats: Stats):
    # spv
    stats.add_attr(AttrStats("interface_var_abi", "spv", False, [AttrOrTypeParameterStats("descriptorSet", "uint32_t"),
                                                          AttrOrTypeParameterStats("binding", "uint32_t"),
                                                          AttrOrTypeParameterStats("storageClass", "Optional<StorageClass>")], [], []))
    stats.add_attr(AttrStats("ver_cap_ext", "spv", False, [AttrOrTypeParameterStats("version", "Version"),
                                                    AttrOrTypeParameterStats("capabilities", "ArrayRef<Capability>"),
                                                    AttrOrTypeParameterStats("extensions", "ArrayRef<Extension>")], [], []))
    stats.add_attr(AttrStats("target_env", "spv", False, [AttrOrTypeParameterStats("triple", "VerCapExtAttr"),
                                                   AttrOrTypeParameterStats("vendorID", "Vendor"),
                                                   AttrOrTypeParameterStats("deviceType", "DeviceType"),
                                                   AttrOrTypeParameterStats("deviceId", "uint32_t"),
                                                   AttrOrTypeParameterStats("limits", "DictionaryAttr")], [], []))

    # vector
    stats.add_attr(AttrStats("combining_kind", "vector", False, [AttrOrTypeParameterStats("kind", "CombiningKind")], [], []))


def remove_unnecessary_verifiers(stats: Stats):
    # Types:
    stats.dialects[""].types["Builtin_Complex"].hasVerifier = False
    stats.dialects[""].types["Builtin_UnrankedMemRef"].hasVerifier = False
    stats.dialects[""].types["Builtin_UnrankedTensor"].hasVerifier = False

    # Attributes

    # Linalg has no unnecessary verifiers
    # gpu
    stats.dialects["gpu"].ops["gpu.block_dim"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.block_id"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.grid_dim"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.thread_id"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.shuffle"].hasVerifier = False
    # amx
    # x86vector
    # tensor
    if verifiers_allow_len_equality:
        stats.dialects["tensor"].ops["tensor.extract"].hasVerifier = False
        stats.dialects["tensor"].ops["tensor.insert"].hasVerifier = False
    # affine
    # emitc
    stats.dialects["emitc"].ops["emitc.apply"].hasVerifier = False


    # for op in stats.dialects["quant"].ops.values():
    #     if op.is_operands_results_attrs_declarative() and op.is_traits_declarative() and op.hasVerifier:
    #         print(op.name)

    # for dialect_name, dialect in stats.dialects.items():
    #     total_ops = len(dialect.ops)
    #     before_ops = 0
    #     after_ops = 0
    #     for op in dialect.ops.values():
    #         if op.is_operands_results_attrs_declarative() and op.is_traits_declarative():
    #             after_ops += 1
    #             if not op.hasVerifier:
    #                 before_ops += 1
    #     print(f"{dialect_name}: before {before_ops} after {after_ops} total {total_ops}")

def get_stat_from_files():
    stats = Stats()
    for file in get_tablegen_op_file_list():
        file_stats = get_stat_from_file(file)
        if file_stats is not None:
            stats.add_stats(file_stats)

    add_cpp_types(stats)
    add_cpp_attributes(stats)
    remove_unnecessary_verifiers(stats)

    res = subprocess.run(["../build/bin/dialect-stats"], capture_output=True)
    if res.returncode != 0:
        return None
    data = json.loads(res.stderr)
    for json_dialect in data:
        dialect_stats = stats.dialects[json_dialect["name"]]
        dialect_stats.numOperations = json_dialect["numOperations"]
        dialect_stats.numTypes = json_dialect["numTypes"]
        dialect_stats.numAttributes = json_dialect["numAttributes"]

    return stats

def get_op_list_distribution(ops: List[OpStats], f: Callable[[OpStats], int], maxi: Optional[int] = None) -> List[
    float]:
    res = [f(op) for op in ops]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_op_distribution(stats: Stats, f: Callable[[OpStats], int]):
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


def get_global_attr_distribution(stats: Stats, f: Callable[[AttrStats], int]):
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


def get_global_type_distribution(stats: Stats, f: Callable[[TypeStats], int]):
    global_res = get_type_list_distribution(stats.types, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_type_list_distribution(list(dialect.types.values()), f, maxi=maxi)
    return global_res, per_dialect_res


T = TypeVar("T")


def get_dialect_values(stats: Stats, f: Callable[[DialectStats], T]) -> Dict[str, T]:
    return {key: f(value) for key, value in stats.dialects.items()}


def add_non_declarative_constraint(constraint: ConstraintStats, d: Dict[ConstraintStats, int]):
    if isinstance(constraint, PredicateConstraintStats):
        if not constraint.is_declarative():
            if constraint in d:
                d[constraint] += 1
            else:
                d[constraint] = 1
    for sub_constraint in constraint.get_sub_constraints():
        add_non_declarative_constraint(sub_constraint, d)


def get_constraints_culprits(stats: Stats) -> Dict[ConstraintStats, int]:
    culprits = dict()
    for op in stats.ops:
        for operand in op.operands:
            constraint = operand.constraint
            add_non_declarative_constraint(constraint, culprits)
        for result in op.results:
            constraint = result.constraint
            add_non_declarative_constraint(constraint, culprits)
        for attr in op.attributes.values():
            add_non_declarative_constraint(attr, culprits)
    return culprits


def get_traits_culprits(stats: Stats) -> Dict[TraitStats, int]:
    culprits = dict()
    for op in stats.ops:
        for trait in op.traits:
            if not trait.is_declarative():
                culprits.setdefault(trait, 0)
                culprits[trait] += 1
    return culprits


def get_interfaces_culprits(stats: Stats) -> Dict[str, int]:
    culprits = dict()
    for op in stats.ops:
        for interface in op.interfaces:
            culprits.setdefault(interface, 0)
            culprits[interface] += 1
    return culprits


def get_type_param_culprits(stats: Stats) -> Dict[str, int]:
    culprits = dict()
    for typ in stats.types:
        for param in typ.parameters:
            if not param.is_declarative():
                culprits.setdefault(param.cppType, 0)
                culprits[param.cppType] += 1
    return culprits


def get_attr_param_culprits(stats: Stats) -> Dict[AttrOrTypeParameterStats, int]:
    culprits = dict()
    for attr in stats.attrs:
        for param in attr.parameters:
            if not param.is_declarative():
                culprits.setdefault(param.cppType, 0)
                culprits[param.cppType] += 1
    return culprits


def create_type_attr_evolution_per_dialect_decl_plot(stats: Stats):
    default_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[1]
    builtins_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[1]
    enums_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]

    default_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[1]
    builtins_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[1]
    enums_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]

    attributes = dict()
    types = dict()
    for key in default_attr:
        attr_sum = default_attr[key][1] + default_attr[key][0]
        if attr_sum != 0:
            res = (default_attr[key][1], builtins_attr[key][1] - default_attr[key][1], enums_attr[key][1] - builtins_attr[key][1], attr_sum - enums_attr[key][1])
            attributes[key] = (res[0] / attr_sum * 100, res[1] / attr_sum * 100, res[2] / attr_sum * 100, res[3] / attr_sum * 100)
        type_sum = default_type[key][1] + default_type[key][0]
        if type_sum != 0:
            res = (default_type[key][1], builtins_type[key][1] - default_type[key][1], enums_type[key][1] - builtins_type[key][1], type_sum - enums_type[key][1])
            types[key] = (res[0] / type_sum * 100, res[1] / type_sum * 100, res[2] / type_sum * 100, res[3] / type_sum * 100)

    print(attributes)
    print(types)


def create_type_attr_evolution_decl_plot(stats: Stats):
    default_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[0]
    builtins_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[0]
    enums_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[0]

    default_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[0]
    builtins_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[0]
    enums_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[0]

    def mp(v):
        return (v[1] / (v[0]+v[1])) * 100

    attrs = (mp(default_attr), mp(builtins_attr), mp(enums_attr))
    types = (mp(default_type), mp(builtins_type), mp(enums_type))
    print(attrs)
    print(types)

    
def create_dialects_decl_plot(stats: Stats):
    types = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    types = {key: value[1] / sum(value) * 100 for key, value in types.items() if sum(value) != 0}
    attrs = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    attrs = {key: value[1] / sum(value) * 100 for key, value in attrs.items() if sum(value) != 0}

    op_operands = get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() else 0)[1]
    op_operands = {key: value[1] / sum(value) * 100 for key, value in op_operands.items() if sum(value) != 0}
    op_full = get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True, check_interfaces=False) else 0)[1]
    op_full = {key: value[1] / sum(value) * 100 for key, value in op_full.items() if sum(value) != 0}

    print(types)
    print(attrs)
    print(op_operands)
    print(op_full)


def create_dialects_decl_plot2(stats: Stats):
    types = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    types = {key: value[1] / sum(value) * 100 for key, value in types.items() if sum(value) != 0}
    attrs = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    attrs = {key: value[1] / sum(value) * 100 for key, value in attrs.items() if sum(value) != 0}

    op_operands = get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() else 0)[1]
    op_full = get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True, check_interfaces=False) else 0)[1]
    ops = {key: (op_full[key][1], value[1] - op_full[key][1], sum(value) - value[1]) for key, value in op_operands.items() if sum(value) != 0}

    print(types)
    print(attrs)
    print(ops)


def create_type_parameters_type_plot(stats: Stats):
    distr = dict()
    for typ in stats.types:
        for param in typ.parameters:
            distr.setdefault(param.get_group(), 0)
            distr[param.get_group()] += 1
    for attr in stats.attrs:
        for param in attr.parameters:
            distr.setdefault(param.get_group(), 0)
            distr[param.get_group()] += 1
    print(distr)


def __main__():
    stats = get_stat_from_files()

    print(stats.types)
    print(stats.attrs)

    print("-" * 80)
    print("Culprits:")
    print("-" * 80)

    print("Constraints:")
    constraints_culprits = list(get_constraints_culprits(stats).items())
    list.sort(constraints_culprits, key=lambda x: x[1], reverse=True)
    print(constraints_culprits)

    print("Traits:")
    traits_culprits = list(get_traits_culprits(stats).items())
    list.sort(traits_culprits, key=lambda x: x[1], reverse=True)
    print(traits_culprits)

    print("Interfaces:")
    traits_culprits = list(get_interfaces_culprits(stats).items())
    list.sort(traits_culprits, key=lambda x: x[1], reverse=True)
    print(traits_culprits)

    print("Type params:")
    type_culprits = list(get_type_param_culprits(stats).items())
    list.sort(type_culprits, key=lambda x: x[1], reverse=True)
    print(type_culprits)

    print("Attr params:")
    attr_culprits = list(get_attr_param_culprits(stats).items())
    list.sort(attr_culprits, key=lambda x: x[1], reverse=True)
    print(attr_culprits)

    print("-" * 80)
    print("Some general stats:")
    print("-" * 80)

    print("Number of operations defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.ops), x.numOperations)))
    print("total:", (len(stats.ops), sum([dialect.numOperations for dialect in stats.dialects.values()]), ))

    print("Number of types defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.types), x.numTypes)))
    print("total:", (len(stats.types), sum([dialect.numTypes for dialect in stats.dialects.values()]), ))

    print("Number of attributes defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.attrs), x.numAttributes)))
    print("total:", (len(stats.attrs), sum([dialect.numAttributes for dialect in stats.dialects.values()]), ))

    print("Number of regions")
    print(get_global_op_distribution(stats, lambda x: x.numRegions))

    print("Number of operands")
    print(get_global_op_distribution(stats, lambda x: x.numOperands))

    print("Number of results")
    print(get_global_op_distribution(stats, lambda x: x.numResults))

    print("Number of attributes")
    print(get_global_op_distribution(stats, lambda x: len(x.attributes)))

    print("Has custom assembly format")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasAssemblyFormat else 0))

    print("Has a verifier")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasVerifier else 0))

    print("Has traits")
    print(get_global_op_distribution(stats, lambda x: int(len(x.traits)>0)))

    print("Has interfaces")
    print(get_global_op_distribution(stats, lambda x: int(len(x.interfaces)>0)))

    print("Type parameters")
    print(get_global_type_distribution(stats, lambda x: len(x.parameters)))

    print("Attr parameters")
    print(get_global_attr_distribution(stats, lambda x: len(x.parameters)))

    print("-" * 80)
    print("Declarativeness:")
    print("-" * 80)

    print("Number of declarative types")
    print(get_global_type_distribution(stats, lambda x: int(not x.hasVerifier)))

    print("Number of declarative attributes")
    print(get_global_attr_distribution(stats, lambda x: int(not x.hasVerifier)))

    print("Has custom assembly format")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasAssemblyFormat else 0))

    print("Has a verifier")
    print(get_global_op_distribution(stats, lambda x: 1 if x.hasVerifier else 0))

    print("Is operands/results declarative in IRDL")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() else 0))

    print("Are traits declarative in IRDL")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_traits_declarative() else 0))

    print("Is operands/results declarative in IRDL with verifiers")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() and not x.hasVerifier else 0))

    print("Has non-declarative traits")
    print(get_global_op_distribution(stats, lambda x: int(len([trait for trait in x.traits if not trait.is_declarative()])>0)))

    print("Fully declarative in tablegen")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(True) else 0))

    print("Fully declarative in IRDL")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True, check_interfaces=True) else 0))

    print("Fully declarative in IRDL without interfaces")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True, check_interfaces=False) else 0))

    print("Fully declarative in IRDL without interfaces and traits")
    print(get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=False, check_interfaces=False) else 0))

    # create_type_attr_evolution_per_dialect_decl_plot(stats)
    # create_type_attr_evolution_decl_plot(stats)
    # create_dialects_decl_plot2(stats)
    # create_type_parameters_type_plot(stats)


if __name__ == "__main__":
    __main__()