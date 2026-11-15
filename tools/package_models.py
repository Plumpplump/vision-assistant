#!/usr/bin/env python3
# Author: David Li, SISHK

"""Move model bundles in and out of the Flet assets tree for packaging.

Flet copies everything under ``src/assets`` into the app, so a small APK means
keeping the optional weights *outside* that tree. This tool moves them to
``models-optional/`` (gitignored) and back, and prints the resulting size.

    python tools/package_models.py --profile apk      # ~169 MB, no VLM/CLIP
    python tools/package_models.py --profile apk-full # ~1.5 GB, everything usable
    python tools/package_models.py --profile full     # + zipformer (~1.6 GB)
    python tools/package_models.py --dry-run          # show what would move

Run ``--profile full`` again after building to keep using the VLM locally. The
app degrades gracefully without it: descriptions fall back to the template.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "src" / "assets" / "models"
OPTIONAL = ROOT / "models-optional"

PROFILES: dict[str, set[str] | None] = {
    "apk": {
        "yolo11n",
        "paradee",
        "g2p",
        "silero-vad",
        "asr-parakeet-110m",
    },
    "apk-vision": {
        "yolo11n",
        "paradee",
        "g2p",
        "silero-vad",
        "asr-parakeet-110m",
        "qwen3.5-0.8b",
    },
    "apk-full": {
        "yolo11n",
        "paradee",
        "g2p",
        "silero-vad",
        "asr-parakeet-110m",
        "qwen3.5-0.8b",
        "mobileclip2-b",
    },
    "full": None,
}

REQUIRED = {"yolo11n", "paradee", "g2p"}


def size_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="apk")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keep = PROFILES[args.profile]
    OPTIONAL.mkdir(exist_ok=True)

    names = {d.name for d in MODELS.iterdir() if d.is_dir()}
    names |= {d.name for d in OPTIONAL.iterdir() if d.is_dir()}

    moves: list[tuple[Path, Path]] = []
    for name in sorted(names):
        wanted = keep is None or name in keep
        in_assets = (MODELS / name).is_dir()
        if wanted and not in_assets:
            moves.append((OPTIONAL / name, MODELS / name))
        elif not wanted and in_assets:
            moves.append((MODELS / name, OPTIONAL / name))

    if not moves:
        print(f"Profile {args.profile!r}: already in place.")
    for source, target in moves:
        verb = "move" if source.parent == MODELS else "restore"
        print(f"  {verb:7s} {source.name:22s} -> {target.parent.name}/")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    present_assets = {d.name for d in MODELS.iterdir() if d.is_dir()}
    kept_names = {
        name for name in present_assets if keep is None or name in keep
    }
    kept_names |= {source.name for source, target in moves if target.parent == MODELS}
    kept_names -= {source.name for source, target in moves if target.parent == OPTIONAL}
    parked_names = {
        name for name in names if keep is not None and name not in keep
    }

    def size_anywhere(name: str) -> float:
        for base in (MODELS, OPTIONAL):
            if (base / name).is_dir():
                return size_mb(base / name)
        return 0.0

    kept_mb = sum(size_anywhere(name) for name in kept_names)
    parked_mb = sum(size_anywhere(name) for name in parked_names)
    print(
        f"\nprofile {args.profile}: assets {kept_mb:.0f} MB "
        f"({', '.join(sorted(kept_names))})"
    )
    if parked_names:
        print(
            f"parked in models-optional: {parked_mb:.0f} MB "
            f"({', '.join(sorted(parked_names))})"
        )
    missing = REQUIRED - kept_names
    if missing:
        print(f"WARNING: required bundles are missing from assets: {sorted(missing)}")
    if not args.dry_run:
        if args.profile == "full":
            print("Full set restored for local development.")
        else:
            print("Run `flet build apk -v` now; use --profile full afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
