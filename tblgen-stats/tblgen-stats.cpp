#include "OpGenHelpers.h"
#include "json.h"
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

std::unique_ptr<JSON> getOpJSON(const Operator &op) {
  auto dict = JSONDict::get();
  dict->insert("name", JSONStr::get(op.getOperationName()));
  dict->insert("dialect", JSONStr::get(op.getDialectName()));
  dict->insert("hasVerifier", JSONBool::get(hasVerify(op)));
  dict->insert("numOperands", JSONInt::get(op.getNumOperands()));
  dict->insert("numResults", JSONInt::get(op.getNumResults()));
  dict->insert("numRegions", JSONInt::get((int)op.getNumRegions()));
  return dict;
}

std::unique_ptr<JSON> getOpsJSON(ArrayRef<Record*> defs) {
  auto list = JSONList::get();
  for (auto &def : defs) {
    list->insert(getOpJSON(Operator(def)));
  }
  return list;
}

bool MlirTableGenStatsMain(raw_ostream &os, RecordKeeper &records) {
  std::vector<Record *> defs = getRequestedOpDefinitions(records);

  auto ops = getOpsJSON(defs);
  ops->print(os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &MlirTableGenStatsMain);
}
