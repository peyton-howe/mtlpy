#pragma once
#include <Foundation/NSAutoreleasePool.hpp>

namespace mtlpy {

// RAII autorelease pool. Every call into Metal/Foundation needs a pool
// active -- ObjC dealloc/init chains routinely autorelease helper objects
// internally (descriptor teardown, driver-internal bookkeeping, etc.), and
// without a pool in scope the runtime logs "autoreleased with no pool in
// place - just leaking" for each one. Not an actual leak (the process is
// either mid-call with a pool a frame up the stack, or exiting and the OS
// reclaims everything regardless) -- just console noise, but it shows up
// loudest at interpreter shutdown when many Buffers/Textures/Devices get
// torn down at once without the caller ever having called an explicit
// release/close. Scoping a guard around each Metal-touching call keeps the
// runtime quiet and bounds how long those autoreleased temporaries stick
// around.
struct PoolGuard {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~PoolGuard() { pool->release(); }
};

} // namespace mtlpy
