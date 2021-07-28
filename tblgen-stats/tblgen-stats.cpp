#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"
#include "OpGenHelpers.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

bool MlirTableGenStatsMain(raw_ostream &os, RecordKeeper &records) {
  std::vector<Record *> defs = getRequestedOpDefinitions(records);
  for (auto def: defs) {
    os << Operator(def).getQualCppClassName() << "\n";
  }
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &MlirTableGenStatsMain);
}
