# Author: David Li, SISHK

"""Work around sherpa-onnx Android wheels needing a newer libc++.

sherpa-onnx is not on Flet's supported binary index: its Android wheels are
built with NDK r29/r30 and reference ``std::__ndk1::__hash_memory``, which
Flet's Python runtime (NDK r27 libc++) does not export. The app ships a tiny
``libcxxshim.so`` (tools/android/libcxxshim.cpp) in the APK and preloads it
RTLD_GLOBAL before importing sherpa_onnx. Everywhere else this is a no-op.
"""

from __future__ import annotations

import os
import sys


def is_android() -> bool:
    return (
        sys.platform == "android"
        or "ANDROID_DATA" in os.environ
        or "ANDROID_ARGUMENT" in os.environ
    )


def preload() -> bool:
    """Load libcxxshim.so globally on Android; return whether it worked."""
    if not is_android():
        return False
    import ctypes

    candidates = ["libcxxshim.so"]
    native_dir = os.environ.get("ANDROID_NATIVE_LIBRARY_DIR")
    if native_dir:
        candidates.insert(0, os.path.join(native_dir, "libcxxshim.so"))
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False
