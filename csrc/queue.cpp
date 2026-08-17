#include "queue.h"
#include "pool_guard.h"

namespace mtlpy {

Queue::Queue(MTL::CommandQueue* queue) : queue_(queue) {}

Queue::~Queue() {
    PoolGuard guard;
    queue_->release();
}

} // namespace mtlpy
