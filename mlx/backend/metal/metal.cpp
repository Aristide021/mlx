// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include <mutex>
#include <unordered_map>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::metal {

namespace {

class LibraryOwnership {
 public:
  static LibraryOwnership& instance() {
    static LibraryOwnership ownership;
    return ownership;
  }

  void retain(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx_);
    refcounts_[name]++;
  }

  void release(const mlx::core::Device& device, const std::string& name) {
    if (device.type != mlx::core::Device::gpu) {
      return;
    }

    bool clear = false;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = refcounts_.find(name);
      if (it == refcounts_.end()) {
        return;
      }
      if (--it->second == 0) {
        refcounts_.erase(it);
        clear = true;
      }
    }

    if (clear) {
      metal::device(device).clear_library(name);
    }
  }

 private:
  std::mutex mtx_;
  std::unordered_map<std::string, size_t> refcounts_;
};

} // namespace

bool is_available() {
  return true;
}

void start_capture(std::string path, NS::Object* object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  if (!started) {
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

void retain_library(const mlx::core::Device& device, const std::string& name) {
  if (device.type == mlx::core::Device::gpu && !name.empty()) {
    LibraryOwnership::instance().retain(name);
  }
}

void release_library(const mlx::core::Device& device, const std::string& name) {
  if (!name.empty()) {
    LibraryOwnership::instance().release(device, name);
  }
}

} // namespace mlx::core::metal
