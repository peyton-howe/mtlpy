#pragma once
#include <Metal/Metal.hpp>
#include <stdexcept>
#include <string>

namespace mtlpy {

// Shared by every call site that waits on a command buffer and needs to
// turn a failed GPU execution into a Python-catchable exception: csrc/
// device.cpp's run_blit, Pipeline::run's self-contained path, and
// CommandBuffer::commit. Previously duplicated verbatim in all three
// (device.cpp's run_blit was already extracted for exactly this reason --
// "shared by every blit-only Device method" -- but the same discipline
// wasn't applied when Pipeline::run and CommandBuffer::commit needed the
// identical block).
inline void throw_if_command_buffer_error(MTL::CommandBuffer* cmd, const std::string& context) {
    if (cmd->status() == MTL::CommandBufferStatusError) {
        std::string err = cmd->error()
            ? cmd->error()->localizedDescription()->utf8String()
            : "Unknown GPU error";
        throw std::runtime_error(context + " failed: " + err);
    }
}

} // namespace mtlpy
