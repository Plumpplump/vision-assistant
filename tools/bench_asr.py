#!/usr/bin/env python3
# Author: David Li, SISHK

"""A/B benchmark the ASR passes on identical audio.

    python tools/bench_asr.py                     # 25 LibriSpeech clips, both models
    python tools/bench_asr.py --limit 73          # whole dummy validation split
    python tools/bench_asr.py --audio my_wavs/    # your own wavs + .txt sidecars

Reports, per model: load time, total decode time, real-time factor (RTF),
first-partial latency (streaming only), and corpus WER/CER against the
reference transcripts. Writes JSON so the numbers in the docs can be checked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "src" / "assets" / "models"
DATASET = "hf-internal-testing/librispeech_asr_dummy"
PARQUET_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy"
    "/resolve/main/clean/validation-00000-of-00001.parquet"
)
SAMPLE_RATE = 16000
STREAM_CHUNK = 1600  # 100 ms

ZIPFORMER = {
    "tokens": MODELS / "asr-zipformer-en" / "tokens.txt",
    "encoder": MODELS / "asr-zipformer-en" / "encoder.int8.onnx",
    "decoder": MODELS / "asr-zipformer-en" / "decoder.int8.onnx",
    "joiner": MODELS / "asr-zipformer-en" / "joiner.int8.onnx",
}
PARAKEET = {
    "tokens": MODELS / "asr-parakeet-110m" / "tokens.txt",
    "encoder": MODELS / "asr-parakeet-110m" / "encoder.int8.onnx",
    "decoder": MODELS / "asr-parakeet-110m" / "decoder.int8.onnx",
    "joiner": MODELS / "asr-parakeet-110m" / "joiner.int8.onnx",
}


# --------------------------------------------------------------------- data


def prepare_dataset(limit: int, cache: Path) -> list[tuple[Path, str, str]]:
    """LibriSpeech dummy validation split: 73 clips with reference text.

    One 9 MB parquet (pyarrow) is easier and more stable than the
    datasets-server's expiring signed audio URLs.
    """
    import pyarrow.parquet as pq

    cache.mkdir(parents=True, exist_ok=True)
    parquet = cache / "validation.parquet"
    if not parquet.is_file():
        print(f"  downloading {PARQUET_URL}")
        urllib.request.urlretrieve(PARQUET_URL, parquet)

    table = pq.read_table(parquet)
    rows = table.slice(0, limit).to_pylist()
    items: list[tuple[Path, str, str]] = []
    for row in rows:
        clip_id = row["id"]
        wav = cache / f"{clip_id}.wav"
        if not wav.is_file():
            flac = cache / f"{clip_id}.flac"
            flac.write_bytes(row["audio"]["bytes"])
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(flac),
                    "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav),
                ],
                check=True,
            )
        items.append((wav, row["text"], clip_id))
    return items


def prepare_local(folder: Path) -> list[tuple[Path, str, str]]:
    items = []
    for wav in sorted(folder.glob("*.wav")):
        sidecar = wav.with_suffix(".txt")
        if not sidecar.is_file():
            print(f"  skip {wav.name}: no reference {sidecar.name}")
            continue
        items.append((wav, sidecar.read_text(encoding="utf-8"), wav.stem))
    return items


# ------------------------------------------------------------------ metrics


def normalize(text: str) -> list[str]:
    return re.sub(r"[^A-Z0-9']", " ", text.upper()).split()


def _distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            )
        previous = current
    return previous[-1]


def error_rates(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    word_errors = word_total = char_errors = char_total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words, hyp_words = normalize(reference), normalize(hypothesis)
        word_errors += _distance(ref_words, hyp_words)
        word_total += max(1, len(ref_words))
        ref_chars, hyp_chars = list(" ".join(ref_words)), list(" ".join(hyp_words))
        char_errors += _distance(ref_chars, hyp_chars)
        char_total += max(1, len(ref_chars))
    return word_errors / word_total, char_errors / char_total


# ------------------------------------------------------------------- models


def load_zipformer():
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(ZIPFORMER["tokens"]),
        encoder=str(ZIPFORMER["encoder"]),
        decoder=str(ZIPFORMER["decoder"]),
        joiner=str(ZIPFORMER["joiner"]),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
    )


def load_parakeet():
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET["encoder"]),
        decoder=str(PARAKEET["decoder"]),
        joiner=str(PARAKEET["joiner"]),
        tokens=str(PARAKEET["tokens"]),
        model_type="nemo_transducer",
        num_threads=2,
        sample_rate=SAMPLE_RATE,
    )


def read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: expected {SAMPLE_RATE} Hz, got {handle.getframerate()}"
            )
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
    if width == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise SystemExit(f"{path}: unsupported sample width {width}")


def run_zipformer(recognizer, audio: np.ndarray):
    stream = recognizer.create_stream()
    started = time.perf_counter()
    first_partial = None
    for start in range(0, len(audio), STREAM_CHUNK):
        stream.accept_waveform(SAMPLE_RATE, audio[start : start + STREAM_CHUNK])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if first_partial is None and recognizer.get_result(stream).strip():
            first_partial = time.perf_counter() - started
    return recognizer.get_result(stream), time.perf_counter() - started, first_partial


def run_parakeet(recognizer, audio: np.ndarray):
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    started = time.perf_counter()
    recognizer.decode_stream(stream)
    return stream.result.text, time.perf_counter() - started, None


# --------------------------------------------------------------------- main


def benchmark(name: str, loader, runner, items) -> dict:
    print(f"\n{name}: loading…", flush=True)
    started = time.perf_counter()
    recognizer = loader()
    load_seconds = time.perf_counter() - started

    # Warm up on 1 s of the first clip so ONNX graph setup is not counted.
    warm = read_audio(items[0][0])[:SAMPLE_RATE]
    runner(recognizer, warm)

    hypotheses: list[str] = []
    references: list[str] = []
    audio_seconds = decode_seconds = 0.0
    first_partials: list[float] = []
    per_clip: list[dict] = []

    for path, reference, clip_id in items:
        audio = read_audio(path)
        text, seconds, first = runner(recognizer, audio)
        audio_seconds += len(audio) / SAMPLE_RATE
        decode_seconds += seconds
        if first is not None:
            first_partials.append(first)
        hypotheses.append(text)
        references.append(reference)
        per_clip.append(
            {
                "id": clip_id,
                "audio_s": len(audio) / SAMPLE_RATE,
                "decode_s": seconds,
                "hypothesis": text,
                "reference": reference,
            }
        )

    word_error, char_error = error_rates(references, hypotheses)
    result = {
        "model": name,
        "clips": len(items),
        "audio_seconds": round(audio_seconds, 2),
        "load_seconds": round(load_seconds, 2),
        "decode_seconds": round(decode_seconds, 2),
        "rtf": round(decode_seconds / audio_seconds, 3),
        "first_partial_median_s": (
            round(float(np.median(first_partials)), 3) if first_partials else None
        ),
        "wer": round(word_error, 4),
        "cer": round(char_error, 4),
        "per_clip": per_clip,
    }
    print(
        f"  clips={result['clips']} audio={result['audio_seconds']:.1f}s "
        f"decode={result['decode_seconds']:.1f}s RTF={result['rtf']:.3f} "
        f"WER={result['wer']:.1%} CER={result['cer']:.1%}"
        + (
            f" first_partial={result['first_partial_median_s']:.2f}s"
            if result["first_partial_median_s"]
            else ""
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="clips to evaluate")
    parser.add_argument("--audio", type=Path, help="folder of 16 kHz wav + .txt")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/vision-assist-asr"))
    parser.add_argument(
        "--models", default="zipformer,parakeet", help="comma separated"
    )
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    items = (
        prepare_local(args.audio)
        if args.audio
        else prepare_dataset(args.limit, args.cache)
    )
    if not items:
        print("No audio with references found.")
        return 1
    print(f"Evaluating {len(items)} clips from {args.audio or DATASET}")

    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    results = []
    if "zipformer" in wanted:
        results.append(benchmark("zipformer", load_zipformer, run_zipformer, items))
    if "parakeet" in wanted:
        results.append(benchmark("parakeet", load_parakeet, run_parakeet, items))

    print("\nmodel       clips  audio_s  load_s  decode_s   RTF  first_s     WER     CER")
    for result in results:
        first = result["first_partial_median_s"]
        print(
            f"{result['model']:<10} {result['clips']:>5} {result['audio_seconds']:>8.1f} "
            f"{result['load_seconds']:>7.2f} {result['decode_seconds']:>9.1f} "
            f"{result['rtf']:>5.2f} "
            f"{(f'{first:.2f}' if first else '-'):>7} "
            f"{result['wer']:>7.1%} {result['cer']:>7.1%}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
