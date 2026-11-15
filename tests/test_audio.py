# Author: David Li, SISHK

"""Audio source selection: temp files on desktop/mobile, bytes on web."""

from pathlib import Path

import pytest

from ui.audio import FletSpeaker


class FakePage:
    web = False

    def __init__(self):
        self.services = []

    def update(self):
        pass


class FakeWebPage(FakePage):
    web = True


def test_byte_sources_become_reusable_files():
    speaker = FletSpeaker(FakePage(), tts=None)
    wav = b"RIFF....WAVEfake"
    first = speaker._media_source(wav)
    second = speaker._media_source(wav)
    assert isinstance(first, str)
    assert first == second
    assert Path(first).read_bytes() == wav


def test_web_keeps_raw_bytes():
    speaker = FletSpeaker(FakeWebPage(), tts=None)
    wav = b"RIFF....WAVEfake"
    assert speaker._media_source(wav) is wav


@pytest.mark.asyncio
async def test_ducking_restores_the_configured_alert_volume():
    speaker = FletSpeaker(FakePage(), tts=None)
    await speaker._loop_alert(b"RIFF....WAVEfake", 0.45, 0.0)
    assert speaker._alert is not None
    assert speaker._alert_volume == 0.45

    await speaker.duck_alerts(True)
    assert speaker._alert.volume == pytest.approx(0.45 * 0.12)

    await speaker.duck_alerts(False)
    assert speaker._alert.volume == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_new_alert_starts_ducked_if_user_is_talking():
    speaker = FletSpeaker(FakePage(), tts=None)
    await speaker.duck_alerts(True)
    await speaker._loop_alert(b"RIFF....WAVEfake", 0.45, 0.0)
    assert speaker._alert is not None
    assert speaker._alert.volume == pytest.approx(0.45 * 0.12)

    await speaker.duck_alerts(False)
    assert speaker._alert.volume == pytest.approx(0.45)
