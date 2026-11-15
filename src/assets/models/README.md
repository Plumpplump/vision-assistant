<!-- Author: David Li, SISHK -->

# Bundled AI models

Weights live here but are **gitignored**; only this README is tracked. Populate
the tree with `python tools/fetch_models.py` (or `--vision` for the large
optional bundles). `tools/package_models.py` decides which bundles ship in an
APK; `full` restores everything for local work.

**Runtime is fully offline.** These files are loaded directly by ONNX Runtime
and sherpa-onnx; the app never downloads a model or sends camera/audio data to
a server. The download above is a one-time installation step.

| Bundle | Role | Files | Size | Licence / source |
| --- | --- | --- | ---: | --- |
| `yolo11n/` | Object detection (80 COCO classes) | `yolo11n.onnx`, `config.json` | 10.7 MB | AGPL-3.0, Ultralytics; ONNX export from [`webnn/yolo11n`](https://huggingface.co/webnn/yolo11n) |
| `silero-vad/` | Voice activity detection / endpointing | `silero_vad.onnx` (int8) | 0.6 MB | MIT; [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad) |
| `asr-parakeet-110m/` | Speech recognition (default) | `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` | 136 MB | **CC-BY-4.0, NVIDIA**; sherpa-onnx export [`akm9191/sherpa-onnx-parakeet-tdt-110m-int8`](https://huggingface.co/akm9191/sherpa-onnx-parakeet-tdt-110m-int8). Attribution required in shipped builds. |
| `paradee/` | Text to speech (24 kHz) | `paradee_int8.onnx`, `config.json`, `LICENSE` | 8.5 MB | Apache-2.0; [`sahilmahendrakar/Paradee-8M-v1.3`](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3) |
| `g2p/` | English grapheme-to-phoneme lexicons | `us_gold.json`, `us_silver.json`, `gb_gold.json`, `gb_silver.json` | 12.6 MB | Apache-2.0; [`hexgrad/misaki`](https://github.com/hexgrad/misaki) |
| `mobileclip2-b/` | Open-vocabulary image/text search (optional) | `image.onnx` + `visual.onnx.data`, `text.onnx` + `text.onnx.data`, `tokenizer.json` | 578 MB | `apple-amlr` (research licence); ONNX export [`RuteNL/MobileCLIP2-B-OpenCLIP-ONNX`](https://huggingface.co/RuteNL/MobileCLIP2-B-OpenCLIP-ONNX) |
| `qwen3.5-0.8b/` | Vision-language descriptions / questions (optional) | `onnx/{decoder_model_merged,embed_tokens,vision_encoder}_q4.onnx(+data)`, tokenizer/config/generation files | 737 MB | Apache-2.0; [`onnx-community/Qwen3.5-0.8B-ONNX`](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) |

Notes:

- MobileCLIP2's ONNX export calls the image tower `visual.onnx`. It is stored
  here as `image.onnx` (the name `adapters/models/clip.py` expects). ONNX
  Runtime resolves the external weights through the filename embedded in the
  graph, which stays `visual.onnx.data`, so that file must stay next to it.
- The Parakeet weights are CC-BY-4.0 and must be credited wherever the app is
  distributed. See the repository README "Credits" section.
- A partial tree is not a valid bundle: the adapters check for exact filenames
  and fall back to `UnavailableDescriber` / silence when a file is missing.
- The optional streaming zipformer (`asr-zipformer-en/`, 73 MB, Apache-2.0) is
  fetched with `--partials`; it is not part of the default profile.
