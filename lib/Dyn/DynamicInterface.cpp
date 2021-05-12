#include "Dyn/DynamicInterface.h"
#include "Dyn/DynamicContext.h"

using namespace mlir;
using namespace dyn;

const dyn::DynamicOpInterfaceImpl &
DynamicOpInterface::getImpl(Operation *op) const {
  auto dynCtx = getDynamicContext();
  auto opID = op->getAbstractOperation()->typeID;
  auto interfaceImpl = dynCtx->lookupOpInterfaceImpl(opID, *this);
  assert(succeeded(interfaceImpl) &&
         "Trying to get an interface implementation in an operation that "
         "doesn't implement the interface");
  return **interfaceImpl;
}
