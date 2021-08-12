#include "json.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <unordered_map>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

std::vector<Record *> getOpDefinitions(const RecordKeeper &recordKeeper) {
  return recordKeeper.getAllDerivedDefinitions("Op");
}

std::vector<Record *> getTypeDefinitions(const RecordKeeper &recordKeeper) {
  return recordKeeper.getAllDerivedDefinitions("TypeDef");
}

std::vector<Record *> getAttrDefinitions(const RecordKeeper &recordKeeper) {
  return recordKeeper.getAllDerivedDefinitions("AttrDef");
}

bool hasVerify(const Operator &op) {
  auto rec = op.getDef();
  auto *valueInit = rec.getValueInit("verifier");
  StringInit *stringInit = dyn_cast<StringInit>(valueInit);
  if (!stringInit)
    return false;
  auto verifier = stringInit->getValue();
  if (verifier.empty())
    return false;
  if (verifier.contains("return success();") && verifier.size() < 20)
    return false;
  return true;
}

std::unique_ptr<JSON> getJSON(const Constraint &constraint);

std::unique_ptr<JSONDict> getVariadicJSON(const Constraint &constraint) {
  auto json = JSONDict::get();
  json->insertJson("baseType", getJSON(Constraint(
                                   constraint.def->getValueAsDef("baseType"))));
  return json;
}

std::unique_ptr<JSONDict> getOptionalJSON(const Constraint &constraint) {
  auto json = JSONDict::get();
  json->insertJson("baseType", getJSON(Constraint(
                                   constraint.def->getValueAsDef("baseType"))));
  return json;
}

std::unique_ptr<JSONDict> getTypeDefJSON(const Constraint &constraint) {
  auto json = JSONDict::get();
  json->insert(
      "dialect",
      constraint.def->getValueAsDef("dialect")->getValueAsString("name"));
  json->insert("name", constraint.def->getName());
  return json;
}

std::unique_ptr<JSON> getJSON(const Constraint &constraint) {
  if (constraint.def->isSubClassOf("TypeDef")) {
    auto res = getTypeDefJSON(constraint);
    res->insert("kind", "typeDef");
    return res;
  }

  if (constraint.def->isSubClassOf("Variadic")) {
    auto res = getVariadicJSON(constraint);
    res->insert("kind", "variadic");
    return res;
  }

  if (constraint.def->isSubClassOf("Optional")) {
    auto res = getOptionalJSON(constraint);
    res->insert("kind", "optional");
    return res;
  }

  if (constraint.def->isSubClassOf("I")) {
    auto res = JSONDict::get();
    res->insert("kind", "integer");
    res->insert("bitwidth", (int)constraint.def->getValueAsInt("bitwidth"));
    return res;
  }

  auto json = JSONDict::get();
  json->insert("kind", "predicate");
  json->insert("predicate", constraint.getPredicate().getCondition());

  auto superJson = JSONList::get();
  auto superclasses = constraint.def->getSuperClasses();
  for (auto cls : superclasses)
    superJson->insert(JSON::get(cls.first->getName()));
  json->insertJson("superclass", std::move(superJson));

  return json;
}

std::unique_ptr<JSON> getJSON(const NamedTypeConstraint &constraint) {
  auto json = JSONDict::get();
  json->insert("name", constraint.name);
  json->insertJson("constraint", getJSON(constraint.constraint));
  return json;
}

std::unique_ptr<JSON> getJSON(const Trait& trait) {
  auto json = JSONDict::get();
  if (isa<NativeTrait>(trait)) {
    auto nativeTrait = cast<NativeTrait>(trait);
    json->insert("kind", "native");
    json->insert("name", nativeTrait.getFullyQualifiedTraitName());
  } else if (isa<PredTrait>(trait)) {
    auto predTrait = cast<PredTrait>(trait);
    json->insert("kind", "pred");
    json->insert("pred", predTrait.getPredTemplate());
  } else if (isa<InternalTrait>(trait)) {
    auto internalTrait = cast<InternalTrait>(trait);
    json->insert("kind", "internal");
    json->insert("name", internalTrait.getFullyQualifiedTraitName());
  } else {
    assert(false);
  }
  return json;
}

std::unique_ptr<JSON> getJSON(const Operator &op) {
  auto dict = JSONDict::get();
  dict->insert("name", op.getOperationName());
  dict->insert("dialect", op.getDialectName());
  dict->insert("hasVerifier", hasVerify(op));
  dict->insert("numOperands", op.getNumOperands());
  dict->insert("numVariableLengthOperands", op.getNumVariableLengthOperands());
  dict->insert("numResults", op.getNumResults());
  dict->insert("numRegions", op.getNumRegions());
  dict->insert("hasNoVariadicRegions", op.hasNoVariadicRegions());
  dict->insert("numSuccessors", op.getNumSuccessors());
  dict->insert("hasAssemblyFormat", op.hasAssemblyFormat());

  auto operands = JSONList::get();
  for (int i = 0; i < op.getNumOperands(); i++)
    operands->insert(getJSON(op.getOperand(i)));
  dict->insertJson("operands", std::move(operands));

  auto results = JSONList::get();
  for (int i = 0; i < op.getNumResults(); i++)
    results->insert(getJSON(op.getResult(i)));
  dict->insertJson("results", std::move(results));

  auto attributes = JSONDict::get();
  for (auto attribute : op.getAttributes())
    attributes->insertJson(attribute.name, getJSON(attribute.attr));
  dict->insertJson("attributes", std::move(attributes));

  auto traits = JSONList::get();
  for (auto trait: op.getTraits())
    if (!isa<InterfaceTrait>(trait))
      traits->insert(getJSON(trait));
  dict->insertJson("traits", std::move(traits));

  return dict;
}

std::unique_ptr<JSON> getJSON(const AttrOrTypeParameter &def) {
  auto dict = JSONDict::get();
  dict->insert("name", def.getName());
  dict->insert("cppType", def.getCppType());
  return dict;
}

std::unique_ptr<JSON> getJSON(ArrayRef<AttrOrTypeParameter> defs) {
  auto list = JSONList::get();
  for (auto &def : defs) {
    list->insert(getJSON(def));
  }
  return list;
}

std::unique_ptr<JSONDict> getJSON(const AttrOrTypeDef &def) {
  auto dict = JSONDict::get();
  dict->insert("name", def.getName());
  dict->insert("numParameters", def.getNumParameters());
  dict->insert("dialect", def.getDialect().getName());

  SmallVector<AttrOrTypeParameter> parameters;
  def.getParameters(parameters);
  dict->insertJson("parameters", getJSON(parameters));
  return dict;
}

std::unique_ptr<JSON> getOpsJSON(ArrayRef<Record *> defs) {
  auto list = JSONList::get();
  for (auto &def : defs) {
    list->insert(getJSON(Operator(def)));
  }
  return list;
}

std::unique_ptr<JSON> getTypesJSON(ArrayRef<Record *> defs) {
  auto list = JSONList::get();
  for (auto &def : defs) {
    list->insert(getJSON(TypeDef(def)));
  }
  return list;
}

std::unique_ptr<JSON> getAttrsJSON(ArrayRef<Record *> defs) {
  auto list = JSONList::get();
  for (auto &def : defs) {
    list->insert(getJSON(AttrDef(def)));
  }
  return list;
}

bool MlirTableGenStatsMain(raw_ostream &os, RecordKeeper &records) {
  std::vector<Record *> opDefs = getOpDefinitions(records);
  std::vector<Record *> typeDefs = getTypeDefinitions(records);
  std::vector<Record *> attrDefs = getAttrDefinitions(records);

  auto res = JSONDict::get();
  res->insertJson("ops", getOpsJSON(opDefs));
  res->insertJson("types", getTypesJSON(typeDefs));
  res->insertJson("attrs", getAttrsJSON(attrDefs));
  res->print(os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &MlirTableGenStatsMain);
}
