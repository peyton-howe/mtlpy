#include "capture.h"
#include "pool_guard.h"

namespace mtlpy {

CaptureScope::CaptureScope(MTL::CaptureScope* scope) : scope_(scope) {}

CaptureScope::~CaptureScope() {
    PoolGuard guard;
    scope_->release();
}

void CaptureScope::begin_scope() { scope_->beginScope(); }
void CaptureScope::end_scope()   { scope_->endScope(); }

} // namespace mtlpy
