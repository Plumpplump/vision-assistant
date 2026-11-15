#!/usr/bin/env python3
# Author: David Li, SISHK

"""Make a Flet Android APK runnable with sherpa-onnx.

sherpa-onnx's Android wheels are built with NDK r29/r30 and need
``std::__ndk1::__hash_memory`` from libc++, but Flet's Python runtime ships
NDK r27 libc++, so ``import sherpa_onnx`` fails with:

    cannot locate symbol "_ZNSt6__ndk113__hash_memoryEPKvm"

This tool adds ``libcxxshim.so`` (built by tools/android/build_libcxxshim.sh)
to the APK's native libraries, re-aligns and re-signs it. The app preloads the
shim on Android before importing sherpa-onnx (adapters/speech/sherpa_compat.py).

    python tools/patch_android_apk.py                  # build/.../app-arm64-v8a-release.apk
    python tools/patch_android_apk.py --apk path.apk --output dist/app.apk

Re-signing uses a debug keystore at ~/.android/debug.keystore (created on first
run). Uninstall the previously installed app before installing this one, since
the signing key changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "tools" / "android" / "arm64-v8a" / "libcxxshim.so"
DEFAULT_APK = (
    ROOT
    / "build"
    / "flutter"
    / "build"
    / "app"
    / "outputs"
    / "flutter-apk"
    / "app-arm64-v8a-release.apk"
)
SHIM_DEST = "lib/arm64-v8a/libcxxshim.so"
SIGNATURES = ("META-INF/*.SF", "META-INF/*.RSA", "META-INF/*.MF")


def find_sdk_tool(name: str) -> Path | None:
    sdk = Path(os.path.expanduser("~/Android/sdk/build-tools"))
    if not sdk.is_dir():
        sdk = Path(os.path.expanduser("~/Android/Sdk/build-tools"))
    candidates = sorted(sdk.glob(f"*/{name}"), reverse=True) if sdk.is_dir() else []
    return candidates[0] if candidates else None


def find_java_home() -> Path | None:
    for base in (Path.home() / "java", Path.home() / "Android" / "jbr"):
        if base.is_dir():
            candidates = sorted(base.glob("*/bin/java"), reverse=True)
            if candidates:
                return candidates[0].parent.parent
    return None


def ensure_keystore(path: Path) -> None:
    if path.is_file():
        return
    keytool = shutil.which("keytool")
    java_home = find_java_home()
    if keytool is None and java_home:
        keytool = str(java_home / "bin" / "keytool")
    if keytool is None:
        raise SystemExit("keytool not found; install a JDK or create the keystore manually.")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            keytool,
            "-genkeypair",
            "-v",
            "-keystore",
            str(path),
            "-alias",
            "androiddebugkey",
            "-storepass",
            "android",
            "-keypass",
            "android",
            "-dname",
            "CN=Android Debug,O=Android,C=US",
            "-keyalg",
            "RSA",
            "-keysize",
            "2048",
            "-validity",
            "10000",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"created debug keystore {path}")


def run(command: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", type=Path, default=DEFAULT_APK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keystore", type=Path, default=Path.home() / ".android" / "debug.keystore")
    parser.add_argument("--skip-sign", action="store_true")
    args = parser.parse_args()

    apk = args.apk.expanduser()
    if not apk.is_file():
        raise SystemExit(f"APK not found: {apk}")
    if not SHIM.is_file():
        build = ROOT / "tools" / "android" / "build_libcxxshim.sh"
        print("shim missing, building it…")
        subprocess.run([str(build)], check=True)
    if not SHIM.is_file():
        raise SystemExit(f"shim still missing: {SHIM}")

    zip_tool = shutil.which("zip")
    unzip_tool = shutil.which("unzip")
    zipalign = find_sdk_tool("zipalign")
    apksigner = find_sdk_tool("apksigner")
    if not zip_tool or not unzip_tool:
        raise SystemExit("zip/unzip not found.")
    if not zipalign or not apksigner:
        raise SystemExit("zipalign/apksigner not found under ~/Android/sdk/build-tools.")

    output = args.output or apk.with_name(apk.stem + "-patched.apk")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the 3-4 GB of intermediate copies on disk, not in /tmp (tmpfs).
    temp_root = output.parent / ".apk-patch"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
        tmp_path = Path(tmp)
        work = tmp_path / "work.apk"
        aligned = tmp_path / "aligned.apk"
        shutil.copy2(apk, work)

        print(f"removing old signatures from {work.name}")
        subprocess.run(
            [zip_tool, "-q", "-d", str(work), *SIGNATURES],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        staging = tmp_path / "staging"
        target = staging / SHIM_DEST
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SHIM, target)
        print(f"adding {SHIM_DEST}")
        subprocess.run(
            [zip_tool, "-q", "-0", str(work), SHIM_DEST],
            cwd=staging,
            check=True,
        )

        print("aligning…")
        run([str(zipalign), "-p", "-f", "4", str(work), str(aligned)], env=os.environ.copy())

        if args.skip_sign:
            shutil.move(str(aligned), output)
        else:
            ensure_keystore(args.keystore)
            env = os.environ.copy()
            java_home = find_java_home()
            if java_home:
                env["PATH"] = f"{java_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
                env["JAVA_HOME"] = str(java_home)
            print("signing…")
            run(
                [
                    str(apksigner),
                    "sign",
                    "--ks",
                    str(args.keystore),
                    "--ks-pass",
                    "pass:android",
                    "--key-pass",
                    "pass:android",
                    "--min-sdk-version",
                    "24",
                    "--out",
                    str(output),
                    str(aligned),
                ],
                env=env,
            )
            run([str(apksigner), "verify", "--print-certs", str(output)], env=env)

    print(f"\nready: {output} ({output.stat().st_size / 1e9:.2f} GB)")
    print("Uninstall the old app before installing (signing key changed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
