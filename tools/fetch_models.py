#!/usr/bin/env python3
"""Populate ``src/assets/models`` for vision-assist.

    python tools/fetch_models.py            # copy what exists, download the rest
    python tools/fetch_models.py --verify   # also exercise each model once
    python tools/fetch_models.py --force    # refresh downloaded files

Sources of truth:

    yolo11n           copy from ../schooluniform/src/assets/models
    qwen3.5-0.8b      copy from ../schooluniform/src/assets/models
    mobileclip2-b     copy from ../schooluniform/src/assets/models
    paradee + g2p     copy from ../SAT-Words/src/assets
    silero-vad        download (MIT)
    asr-parakeet-110m download (CC-BY-4.0, NVIDIA)

Models are large (about 1.5 GB with CLIP and the VLM), so nothing here is
committed; ``src/assets/models`` is gitignored and packaging profiles decide
what ships.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODELS = SRC / "assets" / "models"

SIBLINGS = Path.home() / "Coding" / "pyprojects"
SCHOOLUNIFORM = SIBLINGS / "schooluniform" / "src" / "assets" / "models"
SAT_WORDS = SIBLINGS / "SAT-Words" / "src" / "assets"

HF = "https://huggingface.co"
DOWNLOADS = [
    # Speech gating.
    (
        "onnx-community/silero-vad",
        "onnx/model_int8.onnx",
        "silero-vad/silero_vad.onnx",
        "0.6 MB",
        "MIT",
    ),
    # Default ASR: offline TDT transducer, one pass per VAD sentence.
    # Benchmarked on LibriSpeech dummy validation (73 clips, 481 s):
    # WER 4.6 %, CER 1.6 %, RTF 0.055, ~1.8 s load — better than the streaming
    # zipformer on both accuracy and speed. See docs/asr-benchmark.md.
    (
        "akm9191/sherpa-onnx-parakeet-tdt-110m-int8",
        "encoder.int8.onnx",
        "asr-parakeet-110m/encoder.int8.onnx",
        "131 MB",
        "CC-BY-4.0 (NVIDIA)",
    ),
    (
        "akm9191/sherpa-onnx-parakeet-tdt-110m-int8",
        "decoder.int8.onnx",
        "asr-parakeet-110m/decoder.int8.onnx",
        "4.0 MB",
        "CC-BY-4.0 (NVIDIA)",
    ),
    (
        "akm9191/sherpa-onnx-parakeet-tdt-110m-int8",
        "joiner.int8.onnx",
        "asr-parakeet-110m/joiner.int8.onnx",
        "1.4 MB",
        "CC-BY-4.0 (NVIDIA)",
    ),
    (
        "akm9191/sherpa-onnx-parakeet-tdt-110m-int8",
        "tokens.txt",
        "asr-parakeet-110m/tokens.txt",
        "10 KB",
        "CC-BY-4.0 (NVIDIA)",
    ),
]

# Optional streaming pass: live partials at 0.13 s, but 2x the WER and 2.4x the
# compute of Parakeet. Add it only if word-by-word feedback is worth 73 MB.
OPTIONAL_DOWNLOADS = [
    (
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        "asr-zipformer-en/encoder.int8.onnx",
        "71 MB",
        "Apache-2.0",
    ),
    (
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        "asr-zipformer-en/decoder.int8.onnx",
        "1.3 MB",
        "Apache-2.0",
    ),
    (
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        "asr-zipformer-en/joiner.int8.onnx",
        "1.0 MB",
        "Apache-2.0",
    ),
    (
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "tokens.txt",
        "asr-zipformer-en/tokens.txt",
        "5 KB",
        "Apache-2.0",
    ),
]

COPIES = [
    # (source dir, destination dir, files)
    (SCHOOLUNIFORM / "yolo11n", MODELS / "yolo11n", ("yolo11n.onnx", "config.json")),
    (SAT_WORDS / "paradee", MODELS / "paradee", ("paradee_int8.onnx", "config.json", "LICENSE")),
    (
        SAT_WORDS / "g2p",
        MODELS / "g2p",
        ("us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json"),
    ),
]

OPTIONAL_COPIES = [
    (
        SCHOOLUNIFORM / "mobileclip2-b",
        MODELS / "mobileclip2-b",
        (
            "image.onnx",
            "text.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    (
        SCHOOLUNIFORM / "qwen3.5-0.8b",
        MODELS / "qwen3.5-0.8b",
        (
            "onnx/decoder_model_merged_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx_data",
            "onnx/embed_tokens_q4.onnx",
            "onnx/embed_tokens_q4.onnx_data",
            "onnx/vision_encoder_q4.onnx",
            "onnx/vision_encoder_q4.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "preprocessor_config.json",
            "chat_template.jinja",
        ),
    ),
]


def copy_group(source: Path, destination: Path, names: tuple[str, ...], force: bool) -> None:
    if not source.is_dir():
        print(f"  skip  {destination.name}: {source} not found")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        src_file = source / name
        if not src_file.is_file():
            print(f"  skip  {destination.name}/{name}: missing at source")
            continue
        dst_file = destination / name
        if dst_file.is_file() and not force:
            print(f"  ok    {destination.name}/{name}")
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"  copy  {destination.name}/{name}")


# Exact byte counts, so a model upgrade replaces stale files. Without this a
# repo that keeps the same filenames (as the zipformer releases do) would be
# silently skipped and the app would run the previous model.
EXPECTED_BYTES: dict[str, int] = {
    "silero-vad/silero_vad.onnx": 639383,
    "asr-zipformer-en/encoder.int8.onnx": 71082637,
    "asr-zipformer-en/decoder.int8.onnx": 1307236,
    "asr-zipformer-en/joiner.int8.onnx": 259335,
    "asr-zipformer-en/tokens.txt": 5048,
    "asr-parakeet-110m/encoder.int8.onnx": 131113202,
    "asr-parakeet-110m/decoder.int8.onnx": 3955863,
    "asr-parakeet-110m/joiner.int8.onnx": 1411403,
    "asr-parakeet-110m/tokens.txt": 9953,
}


def download(repo: str, remote: str, local: str, force: bool) -> None:
    target = MODELS / local
    expected = EXPECTED_BYTES.get(local, 0)
    if target.is_file() and not force:
        actual = target.stat().st_size
        if not expected or actual == expected:
            print(f"  ok    {local}")
            return
        print(f"  stale {local} ({actual} != {expected} bytes), refreshing")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/{repo}/resolve/main/{remote}"
    print(f"  get   {local}")
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def verify() -> int:
    sys.path.insert(0, str(SRC))
    failures = 0

    try:
        from adapters.detection import YoloDetector

        detector = YoloDetector()
        detector.load()
        print(f"  yolo  loaded, {len(detector.labels)} labels")
    except Exception as exc:
        print(f"  yolo  FAILED: {exc}")
        failures += 1

    try:
        from adapters.speech import ParadeeTts

        tts = ParadeeTts()
        wav = tts.synthesize("Vision assist is ready.")
        print(f"  tts   {len(wav)} bytes for one sentence")
    except Exception as exc:
        print(f"  tts   FAILED: {exc}")
        failures += 1

    try:
        import io
        import wave

        import numpy as np
        import sherpa_onnx

        from adapters.speech import ParadeeTts

        tts = ParadeeTts()
        wav = tts.synthesize("Chair, close, on your left.")
        handle = wave.open(io.BytesIO(wav))
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32768.0
        audio = np.interp(
            np.arange(int(len(pcm) * 16000 / handle.getframerate()))
            * handle.getframerate()
            / 16000,
            np.arange(len(pcm)),
            pcm,
        ).astype(np.float32)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(MODELS / "asr-parakeet-110m" / "encoder.int8.onnx"),
            decoder=str(MODELS / "asr-parakeet-110m" / "decoder.int8.onnx"),
            joiner=str(MODELS / "asr-parakeet-110m" / "joiner.int8.onnx"),
            tokens=str(MODELS / "asr-parakeet-110m" / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=2,
            sample_rate=16000,
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        print(f"  asr   {text!r}")
        if not text:
            failures += 1
    except Exception as exc:
        print(f"  asr   FAILED: {exc}")
        failures += 1

    try:
        from adapters.models.clip import ClipSearch

        clip = ClipSearch()
        print(f"  clip  {'present' if clip.available() else clip.error}")
    except Exception as exc:
        print(f"  clip  FAILED: {exc}")
        failures += 1

    try:
        from adapters.models.qwen_vlm import bundle_available
        from adapters.models.qwen_runner import find_model_dir

        if bundle_available():
            print(f"  vlm   bundle present at {find_model_dir()}")
        else:
            print("  vlm   bundle not copied (template descriptions only)")
    except Exception as exc:
        print(f"  vlm   FAILED: {exc}")
        failures += 1

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="also copy MobileCLIP2-B + Qwen VLM (~1.3 GB, search / describe)",
    )
    parser.add_argument(
        "--partials",
        action="store_true",
        help="also download the streaming zipformer (73 MB) for live partials",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    print("Required (desktop demo):")
    for source, destination, names in COPIES:
        copy_group(source, destination, names, args.force)

    print("Downloads (speech input, later phase):")
    for repo, remote, local, size, _licence in DOWNLOADS:
        download(repo, remote, local, args.force)

    if args.partials:
        print("Optional (streaming partials):")
        for repo, remote, local, size, _licence in OPTIONAL_DOWNLOADS:
            download(repo, remote, local, args.force)
    else:
        print("Skipping streaming zipformer (pass --partials for live partials).")

    if args.vision:
        print("Optional (open vocabulary + description):")
        for source, destination, names in OPTIONAL_COPIES:
            copy_group(source, destination, names, args.force)
    else:
        print("Skipping CLIP + VLM bundles (pass --vision to copy them).")

    if args.verify:
        print("Verification:")
        return verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
