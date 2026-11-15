#!/usr/bin/env python3
# Author: David Li, SISHK

"""Populate ``src/assets/models`` for vision-assist.

    python tools/fetch_models.py            # copy what exists, download the rest
    python tools/fetch_models.py --verify   # also exercise each model once
    python tools/fetch_models.py --force    # refresh downloaded files

Sources of truth:

    yolo11n           copy from ../schooluniform/src/assets/models, else HF
    qwen3.5-0.8b      copy from ../schooluniform/src/assets/models, else HF
    mobileclip2-b     copy from ../schooluniform/src/assets/models, else HF
    paradee + g2p     copy from ../SAT-Words/src/assets, else HF / misaki
    silero-vad        download (MIT)
    asr-parakeet-110m download (CC-BY-4.0, NVIDIA)

The sibling copies are preferred because they avoid a large download, but every
bundle also has a canonical Hugging Face source so a fresh checkout works
without those projects. Models are large (about 1.5 GB with CLIP and the VLM),
so nothing here is committed; ``src/assets/models`` is gitignored and packaging
profiles decide what ships.
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
GITHUB = "https://raw.githubusercontent.com"
# Remote fallbacks for bundles normally copied from the sibling projects.
# They only fetch when the local file is absent (or --force is given), so a
# custom sibling export is never silently replaced by the canonical one.
FALLBACK_DOWNLOADS = [
    # Detection. The ONNX export is the same Ultralytics YOLO11n graph used by
    # the sibling project, with the COCO label map next to it.
    (
        "webnn/yolo11n",
        "onnx/yolo11n.onnx",
        "yolo11n/yolo11n.onnx",
        "10.7 MB",
        "AGPL-3.0 (Ultralytics)",
    ),
    (
        "webnn/yolo11n",
        "config.json",
        "yolo11n/config.json",
        "3 KB",
        "AGPL-3.0 (Ultralytics)",
    ),
    # Speech output. Paradee + the misaki English lexicons.
    (
        "sahilmahendrakar/Paradee-8M-v1.3",
        "onnx/paradee_int8.onnx",
        "paradee/paradee_int8.onnx",
        "8.5 MB",
        "Apache-2.0",
    ),
    (
        "sahilmahendrakar/Paradee-8M-v1.3",
        "config.json",
        "paradee/config.json",
        "2 KB",
        "Apache-2.0",
    ),
    (
        "sahilmahendrakar/Paradee-8M-v1.3",
        "LICENSE",
        "paradee/LICENSE",
        "11 KB",
        "Apache-2.0",
    ),
]

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

# English G2P data lives in the misaki repository rather than on Hugging Face.
DIRECT_DOWNLOADS = [
    (
        f"{GITHUB}/hexgrad/misaki/main/misaki/data/{name}",
        f"g2p/{name}",
        "3 MB",
        "Apache-2.0",
    )
    for name in ("us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json")
]

# Optional open-vocabulary search + vision-language description (--vision).
# MobileCLIP's export names the image tower visual.onnx; it is stored under the
# app's canonical name. ONNX Runtime resolves the external data through the
# filename embedded in the graph, so visual.onnx.data keeps that name.
VISION_DOWNLOADS = [
    (
        "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX",
        "visual.onnx",
        "mobileclip2-b/image.onnx",
        "1.4 MB",
        "apple-amlr",
    ),
    (
        "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX",
        "visual.onnx.data",
        "mobileclip2-b/visual.onnx.data",
        "345 MB",
        "apple-amlr",
    ),
    (
        "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX",
        "text.onnx",
        "mobileclip2-b/text.onnx",
        "1.5 MB",
        "apple-amlr",
    ),
    (
        "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX",
        "text.onnx.data",
        "mobileclip2-b/text.onnx.data",
        "254 MB",
        "apple-amlr",
    ),
    (
        "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX",
        "tokenizer.json",
        "mobileclip2-b/tokenizer.json",
        "3.6 MB",
        "apple-amlr",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/decoder_model_merged_q4.onnx",
        "qwen3.5-0.8b/onnx/decoder_model_merged_q4.onnx",
        "0.9 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/decoder_model_merged_q4.onnx_data",
        "qwen3.5-0.8b/onnx/decoder_model_merged_q4.onnx_data",
        "485 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/embed_tokens_q4.onnx",
        "qwen3.5-0.8b/onnx/embed_tokens_q4.onnx",
        "1 KB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/embed_tokens_q4.onnx_data",
        "qwen3.5-0.8b/onnx/embed_tokens_q4.onnx_data",
        "163 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/vision_encoder_q4.onnx",
        "qwen3.5-0.8b/onnx/vision_encoder_q4.onnx",
        "0.2 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "onnx/vision_encoder_q4.onnx_data",
        "qwen3.5-0.8b/onnx/vision_encoder_q4.onnx_data",
        "68 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "tokenizer.json",
        "qwen3.5-0.8b/tokenizer.json",
        "19 MB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "config.json",
        "qwen3.5-0.8b/config.json",
        "3 KB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "generation_config.json",
        "qwen3.5-0.8b/generation_config.json",
        "0.3 KB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "preprocessor_config.json",
        "qwen3.5-0.8b/preprocessor_config.json",
        "0.4 KB",
        "Apache-2.0",
    ),
    (
        "onnx-community/Qwen3.5-0.8B-ONNX",
        "chat_template.jinja",
        "qwen3.5-0.8b/chat_template.jinja",
        "8 KB",
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
            "image.onnx.data",
            "visual.onnx.data",
            "text.onnx",
            "text.onnx.data",
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
            "generation_config.json",
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
    "yolo11n/yolo11n.onnx": 10720228,
    "yolo11n/config.json": 3184,
    "silero-vad/silero_vad.onnx": 639383,
    "asr-zipformer-en/encoder.int8.onnx": 71082637,
    "asr-zipformer-en/decoder.int8.onnx": 1307236,
    "asr-zipformer-en/joiner.int8.onnx": 259335,
    "asr-zipformer-en/tokens.txt": 5048,
    "asr-parakeet-110m/encoder.int8.onnx": 131113202,
    "asr-parakeet-110m/decoder.int8.onnx": 3955863,
    "asr-parakeet-110m/joiner.int8.onnx": 1411403,
    "asr-parakeet-110m/tokens.txt": 9953,
    "paradee/paradee_int8.onnx": 8462503,
    "paradee/config.json": 2103,
    "paradee/LICENSE": 11358,
    "g2p/us_gold.json": 3000469,
    "g2p/us_silver.json": 3099517,
    "g2p/gb_gold.json": 2838552,
    "g2p/gb_silver.json": 3663898,
    "mobileclip2-b/image.onnx": 1353486,
    "mobileclip2-b/visual.onnx.data": 345440256,
    "mobileclip2-b/text.onnx": 1477250,
    "mobileclip2-b/text.onnx.data": 254017536,
    "mobileclip2-b/tokenizer.json": 3642035,
    "qwen3.5-0.8b/onnx/decoder_model_merged_q4.onnx": 876359,
    "qwen3.5-0.8b/onnx/decoder_model_merged_q4.onnx_data": 485425152,
    "qwen3.5-0.8b/onnx/embed_tokens_q4.onnx": 857,
    "qwen3.5-0.8b/onnx/embed_tokens_q4.onnx_data": 162897920,
    "qwen3.5-0.8b/onnx/vision_encoder_q4.onnx": 184854,
    "qwen3.5-0.8b/onnx/vision_encoder_q4.onnx_data": 68267008,
    "qwen3.5-0.8b/tokenizer.json": 19226111,
    "qwen3.5-0.8b/config.json": 2849,
    "qwen3.5-0.8b/generation_config.json": 248,
    "qwen3.5-0.8b/preprocessor_config.json": 336,
    "qwen3.5-0.8b/chat_template.jinja": 7755,
}


def download_url(
    url: str, local: str, force: bool, only_if_missing: bool = False
) -> None:
    target = MODELS / local
    expected = EXPECTED_BYTES.get(local, 0)
    if target.is_file() and not force:
        if only_if_missing:
            print(f"  ok    {local}")
            return
        actual = target.stat().st_size
        if not expected or actual == expected:
            print(f"  ok    {local}")
            return
        print(f"  stale {local} ({actual} != {expected} bytes), refreshing")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"  get   {local}")
    # Stream: the VLM/CLIP files are hundreds of megabytes and must not be
    # buffered in RAM. A completed file only appears after the rename.
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(target)


def download(
    repo: str, remote: str, local: str, force: bool, only_if_missing: bool = False
) -> None:
    download_url(
        f"{HF}/{repo}/resolve/main/{remote}",
        local,
        force,
        only_if_missing=only_if_missing,
    )


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
        help="also fetch MobileCLIP2-B + Qwen VLM (~1.3 GB, search / describe)",
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

    print("G2P lexicons:")
    for url, local, _size, _licence in DIRECT_DOWNLOADS:
        download_url(url, local, args.force)

    print("Remote fallbacks (missing sibling bundles):")
    for repo, remote, local, _size, _licence in FALLBACK_DOWNLOADS:
        download(repo, remote, local, args.force, only_if_missing=True)

    print("Downloads (speech input):")
    for repo, remote, local, _size, _licence in DOWNLOADS:
        download(repo, remote, local, args.force)

    if args.partials:
        print("Optional (streaming partials):")
        for repo, remote, local, _size, _licence in OPTIONAL_DOWNLOADS:
            download(repo, remote, local, args.force)
    else:
        print("Skipping streaming zipformer (pass --partials for live partials).")

    if args.vision:
        print("Optional (open vocabulary + description):")
        for source, destination, names in OPTIONAL_COPIES:
            copy_group(source, destination, names, args.force)
        for repo, remote, local, _size, _licence in VISION_DOWNLOADS:
            download(repo, remote, local, args.force, only_if_missing=True)
    else:
        print("Skipping CLIP + VLM bundles (pass --vision to copy them).")

    if args.verify:
        print("Verification:")
        return verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
