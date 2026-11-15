// Compatibility shim for sherpa-onnx Android wheels built with NDK r29/r30.
//
// Flet's Android Python runtime links libc++ from NDK r27, which lacks
// std::__ndk1::__hash_memory (added in a later libc++). sherpa-onnx's
// libsherpa-onnx-c-api.so and _sherpa_onnx extension reference that symbol,
// so dlopen fails with:
//     cannot locate symbol "_ZNSt6__ndk113__hash_memoryEPKvm"
//
// This library defines the missing symbol. It is loaded RTLD_GLOBAL before
// sherpa_onnx on Android (see adapters/speech/sherpa_compat.py), so the
// dynamic linker can satisfy sherpa's reference. The hash only has to be
// deterministic inside the process (it backs libc++'s unordered containers),
// not match r30 bit-for-bit.
//
// Build: tools/android/build_libcxxshim.sh (uses the installed NDK r28).

#include <cstddef>
#include <cstdint>

namespace std {
inline namespace __ndk1 {

__attribute__((visibility("default"))) std::size_t __hash_memory(
    void const* pointer, std::size_t length) noexcept {
  auto bytes = static_cast<unsigned char const*>(pointer);
  // FNV-1a 64-bit, folded to size_t.
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < length; ++index) {
    hash ^= static_cast<std::uint64_t>(bytes[index]);
    hash *= 1099511628211ULL;
  }
  return static_cast<std::size_t>(hash);
}

}  // namespace __ndk1
}  // namespace std
