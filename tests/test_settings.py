from domain.settings import Settings


def test_defaults_are_sane():
    settings = Settings()
    assert settings.alerts_enabled is True
    assert 0.0 < settings.confidence < 1.0
    assert settings.voice == "af_heart"
    assert settings.alert_style == "soft"
    assert settings.alert_volume <= 0.5  # quiet by default
    assert settings.wake_word == "hey vision"
    assert settings.distance_scale == 1.0
    assert settings.surface_height_m == 0.75
    assert settings.camera_height_m == 1.1
    assert settings.alert_near_m == 1.0
    assert settings.alert_close_m == 0.6
    assert settings.corridor == 0.35
    assert settings.greet_name == "David"


def test_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    original = Settings(confidence=0.6, corridor=0.35, voice="am_puck", speed=1.2)
    original.save(path)

    loaded = Settings.load(path)
    assert loaded.confidence == 0.6
    assert loaded.corridor == 0.35
    assert loaded.voice == "am_puck"
    assert loaded.speed == 1.2


def test_load_missing_file_returns_defaults(tmp_path):
    assert Settings.load(tmp_path / "nope.json") == Settings()


def test_load_corrupt_file_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert Settings.load(path) == Settings()


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"confidence": 0.5, "from_the_future": true}', encoding="utf-8")
    loaded = Settings.load(path)
    assert loaded.confidence == 0.5


def test_values_are_clamped():
    settings = Settings(confidence=5.0, speed=9.0, alert_volume=-1.0).normalized()
    assert settings.confidence == 0.95
    assert settings.speed == 2.0
    assert settings.alert_volume == 0.0
