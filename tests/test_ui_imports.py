# Author: David Li, SISHK

"""Import-level smoke tests: no Flutter harness, no camera, no audio device."""

from pathlib import Path


def test_app_packages_import():
    import adapters.detection
    import adapters.models
    import adapters.speech
    import app.assistant
    import domain


def test_ui_module_imports():
    import ui.audio
    import ui.app


def test_main_entrypoint_is_wired():
    import main

    assert callable(main.main)


def test_bundled_paths_resolve():
    from adapters.paths import assets_dir, model_path

    assert assets_dir().name == "assets"
    assert model_path("yolo11n", "yolo11n.onnx").name == "yolo11n.onnx"


def test_models_are_gitignored_but_documented():
    root = Path(__file__).resolve().parent.parent
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "src/assets/models/*" in ignore
    assert (root / "src" / "assets" / "models" / "README.md").is_file()
