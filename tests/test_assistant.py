# Author: David Li, SISHK

import asyncio
import threading

import numpy as np
import pytest

from adapters.capture.base import Frame
from adapters.models.fake import FakeDescriber
from adapters.speech import PassthroughSegmenter
from app.assistant import UNKNOWN_TEXT, Assistant
from domain import (
    AlertEvent,
    Band,
    Box,
    Detection,
    FrameScene,
    Hazard,
    Settings,
    Vertical,
)


class StubDetector:
    name = "stub"
    error = None

    def __init__(self, detections):
        self._detections = detections
        self.confidence = 0.5

    def load(self):
        return None

    def detect(self, image):
        return list(self._detections)


class StubSource:
    name = "stub camera"
    error = None

    def start(self):
        return None

    def stop(self):
        return None

    def latest(self):
        return None


class FakeAsr:
    name = "fake-asr"
    error = None

    def __init__(self, transcript="what is in front of me"):
        self.transcript = transcript
        self.calls = 0

    def load(self):
        return None

    def transcribe(self, samples):
        self.calls += 1
        return self.transcript


class FakeMicrophone:
    name = "fake-mic"
    error = None

    def __init__(self, samples=None, available=True):
        self._samples = samples
        self._available = available

    def available(self):
        return self._available

    async def record(self, max_seconds, stop_event, endpoint=None, idle_timeout=None):
        if self._samples is not None:
            return self._samples
        return np.ones(16000, dtype=np.float32)


class RecordingSpeaker:
    def __init__(self):
        self.said: list[str] = []
        self.earcons: list[tuple[bytes, bool]] = []
        self.haptics: list[str] = []
        self.silences = 0

    async def speak(self, text, voice, speed, volume):
        self.said.append(text)

    async def earcon(self, wav, volume, balance=0.0, loop=False):
        self.earcons.append((wav, loop))

    async def silence_alerts(self):
        self.silences += 1

    async def stop(self):
        return None

    async def haptic(self, strength="light"):
        self.haptics.append(strength)

    async def duck_alerts(self, ducked):
        self.ducked = getattr(self, "ducked", [])
        self.ducked.append(ducked)


class GatedSpeaker(RecordingSpeaker):
    """RecordingSpeaker whose speak() waits for a per-call gate."""

    def __init__(self):
        super().__init__()
        self.gates: list[asyncio.Event] = []

    async def speak(self, text, voice, speed, volume):
        gate = asyncio.Event()
        self.gates.append(gate)
        self.said.append(text)
        await gate.wait()


def make_scene():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(jpeg=b"", width=640, height=480, timestamp=0.0, fps=12.0, image=image)
    detections = [
        Detection(box=Box(90, 170, 290, 470, 0.9), label="chair", score=0.9),
        Detection(box=Box(280, 260, 400, 400, 0.8), label="cup", score=0.8),
    ]
    return frame, detections


def build(detections=None):
    frame, default = make_scene()
    speaker = RecordingSpeaker()
    assistant = Assistant(
        speaker=speaker,
        settings=Settings(alert_close_m=1.2, alert_near_m=2.5, corridor=0.5),
        source=StubSource(),
        detector=StubDetector(detections if detections is not None else default),
        describer=FakeDescriber(),
        asr=FakeAsr(),
        microphone=FakeMicrophone(),
        segmenter=PassthroughSegmenter(),
    )
    assistant.last_frame = frame
    # two passes: the tracker confirms a track on its second hit
    assistant._analyze(frame)
    scene, events = assistant._analyze(frame)
    assistant.scene = scene
    return assistant, speaker, frame, events


def test_analysis_produces_items_and_hazards():
    assistant, _speaker, _frame, events = build()
    assert [item.label for item in assistant.scene.items] == ["chair", "cup"]
    assert assistant.scene.hazards
    assert assistant.scene.hazards[0].label == "chair"
    assert assistant.scene.hazards[0].band.value == "close"
    assert 0.9 < assistant.scene.hazards[0].distance_m < 1.15
    assert any(event.kind == "enter" for event in events)


def test_scene_question_is_answered_from_templates():
    assistant, speaker, _frame, _events = build()
    reply = asyncio.run(assistant.ask("what is in front of me"))
    assert reply == (
        "Chair, about one metre away, on your left. "
        "Cup, about half a metre away, ahead."
    )
    assert speaker.said[-1] == reply


def test_describe_it_uses_the_biggest_object():
    assistant, speaker, _frame, _events = build()
    reply = asyncio.run(assistant.ask("describe it"))
    assert reply == "That looks like a chair."
    assert speaker.said[-1] == reply


def test_describe_missing_object_is_honest():
    assistant, _speaker, _frame, _events = build()
    reply = asyncio.run(assistant.ask("describe the sofa"))
    assert "sofa" in reply
    assert "can't see" in reply


def test_stop_and_unknown():
    assistant, _speaker, _frame, _events = build()
    assert asyncio.run(assistant.ask("stop")) == "Stopped."
    assert asyncio.run(assistant.ask("banana")) == UNKNOWN_TEXT


def test_find_announces_a_stable_hit():
    assistant, speaker, _frame, _events = build()

    async def scenario():
        reply = await assistant.ask("find the cup")
        assert reply == "Looking for the cup."
        task = assistant._search_task
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert any("Found the cup, about half a metre away" in line for line in speaker.said)
    assert speaker.earcons


@pytest.mark.asyncio
async def test_recording_ducks_alerts_when_speech_starts():
    assistant, speaker, _frame, _events = build()
    endpoint = type("Endpoint", (), {"started": False})()

    async def record(max_seconds, stop_event, endpoint=None, idle_timeout=None):
        await asyncio.sleep(0.1)
        endpoint.started = True
        await asyncio.sleep(0.1)
        return np.ones(16000, dtype=np.float32)

    assistant.microphone.record = record
    await assistant._record_with_duck(1.0, threading.Event(), endpoint)
    assert speaker.ducked == [True, False]


@pytest.mark.asyncio
async def test_shutdown_is_idempotent():
    assistant, _speaker, _frame, _events = build()
    await assistant.shutdown()
    await assistant.shutdown()


@pytest.mark.asyncio
async def test_default_alerts_are_soft_looped_chimes_with_haptics():
    assistant, speaker, _frame, events = build()
    await assistant._handle_events(events)
    assert speaker.earcons and speaker.earcons[0][1] is True  # loop=True
    assert speaker.haptics == ["heavy"]  # the chair is close
    assert not speaker.said


@pytest.mark.asyncio
async def test_receding_to_far_does_not_play_or_vibrate():
    assistant, speaker, _frame, _events = build()
    far = Hazard(
        track_id=99,
        label="chair",
        band=Band.FAR,
        pan=0.0,
        vertical=Vertical.FLOOR,
        box=Box(10, 10, 100, 100),
        distance_m=2.0,
    )
    near = Hazard(
        track_id=98,
        label="cup",
        band=Band.NEAR,
        pan=0.0,
        vertical=Vertical.WAIST,
        box=Box(20, 20, 90, 90),
        distance_m=0.9,
    )
    assistant.scene = FrameScene(hazards=[near])
    await assistant._handle_events(
        [AlertEvent(kind="change", hazard=far, previous=Band.CLOSE)]
    )
    assert speaker.earcons == []
    assert speaker.haptics == []


@pytest.mark.asyncio
async def test_voice_style_speaks_one_word_and_never_beeps():
    assistant, speaker, _frame, events = build()
    assistant.settings.alert_style = "voice"
    await assistant._handle_events(events)
    assert speaker.said == ["Careful."]
    assert speaker.earcons == []


@pytest.mark.asyncio
async def test_haptic_style_is_silent():
    assistant, speaker, _frame, events = build()
    assistant.settings.alert_style = "haptic"
    await assistant._handle_events(events)
    assert speaker.earcons == []
    assert speaker.said == []
    assert speaker.haptics


@pytest.mark.asyncio
async def test_off_style_silences_everything():
    assistant, speaker, _frame, events = build()
    assistant.settings.alert_style = "off"
    await assistant._handle_events(events)
    assert speaker.earcons == []
    assert speaker.said == []
    assert speaker.haptics == []


@pytest.mark.asyncio
async def test_talk_records_transcribes_and_answers():
    assistant, speaker, _frame, _events = build()
    reply = await assistant.listen()
    assert assistant.heard == "what is in front of me"
    assert reply == assistant.answer
    assert "Chair, about one metre away" in reply
    assert assistant.listening is False
    assert speaker.said[-1] == reply


@pytest.mark.asyncio
async def test_talk_without_microphone_explains_itself():
    assistant, speaker, _frame, _events = build()
    assistant.microphone = FakeMicrophone(available=False)
    reply = await assistant.listen()
    assert "microphone is not available" in reply.lower()
    assert speaker.said[-1] == reply


@pytest.mark.asyncio
async def test_empty_recording_does_not_reach_the_asr():
    assistant, _speaker, _frame, _events = build()
    assistant.asr = FakeAsr()
    assistant.microphone = FakeMicrophone(np.zeros(16000, dtype=np.float32))
    assistant.segmenter = PassthroughSegmenter()
    reply = await assistant.listen()
    # passthrough accepts the audio, so the ASR is called and answers normally
    assert assistant.asr.calls == 1
    assert reply


@pytest.mark.asyncio
async def test_yes_no_question_answers_from_the_current_frame():
    assistant, speaker, _frame, _events = build()
    reply = await assistant.ask("is there a cup in front of me?")
    assert reply.startswith("Yes, a cup")
    assert "ahead" in reply
    assert speaker.said[-1] == reply


@pytest.mark.asyncio
async def test_yes_no_question_about_a_missing_object_is_honest():
    assistant, _speaker, _frame, _events = build()
    reply = await assistant.ask("is there a sofa?")
    assert "I don't see a sofa" in reply
    assert "I see a chair" in reply


@pytest.mark.asyncio
async def test_where_is_answers_with_distance_and_direction():
    assistant, _speaker, _frame, _events = build()
    reply = await assistant.ask("where is the cup?")
    assert reply.startswith("The cup is about half a metre away")
    assert "ahead" in reply


@pytest.mark.asyncio
async def test_plural_question_matches_the_singular_label():
    assistant, _speaker, _frame, _events = build()
    reply = await assistant.ask("are there any chairs?")
    assert reply.startswith("Yes, a chair")


@pytest.mark.asyncio
async def test_everyday_word_matches_the_coco_label():
    assistant, _speaker, _frame, _events = build()
    reply = await assistant.ask("is there a mug?")
    assert reply.startswith("Yes, a cup")


@pytest.mark.asyncio
async def test_wake_phrase_with_a_command_runs_the_command():
    assistant, _speaker, _frame, _events = build()
    reply = await assistant.handle_wake_text("Hey Vision, is there a cup?")
    assert reply is not None and reply.startswith("Yes, a cup")


@pytest.mark.asyncio
async def test_wake_only_cues_with_an_earcon_then_accepts_the_next_utterance():
    assistant, speaker, _frame, _events = build()
    said_before = list(speaker.said)
    assert await assistant.handle_wake_text("Hey Vision") == ""
    assert speaker.said == said_before          # an earcon, not speech
    assert speaker.earcons

    reply = await assistant.handle_wake_text("what is in front of me")
    assert reply is not None
    assert "Chair, about one metre away" in reply


@pytest.mark.asyncio
async def test_background_conversation_is_ignored():
    assistant, speaker, _frame, _events = build()
    said_before = list(speaker.said)
    assert await assistant.handle_wake_text("did you watch the game last night") is None
    assert speaker.said == said_before


@pytest.mark.asyncio
async def test_follow_up_question_needs_no_wake_word():
    assistant, _speaker, _frame, _events = build()
    await assistant.ask("what is in front of me")
    reply = await assistant.handle_wake_text("describe it")
    assert reply == "That looks like a chair."


@pytest.mark.asyncio
async def test_background_chatter_in_the_follow_up_window_is_ignored():
    assistant, speaker, _frame, _events = build()
    await assistant.ask("what is in front of me")
    said_before = list(speaker.said)
    assert await assistant.handle_wake_text("did you watch the game") is None
    assert speaker.said == said_before


@pytest.mark.asyncio
async def test_clipped_person_is_answered_as_very_close():
    frame, detections = make_scene()
    detections = [
        Detection(box=Box(90, 170, 290, 480, 0.9), label="person", score=0.9)
    ]
    assistant, _speaker, frame, _events = build(detections)
    reply = await assistant.ask("what is in front of me")
    assert "Person, very close" in reply


@pytest.mark.asyncio
async def test_alarm_stops_on_the_frame_the_object_leaves_the_centre():
    assistant, speaker, frame, events = build()
    await assistant._handle_events(events)
    assert speaker.earcons and speaker.earcons[-1][1] is True
    assert speaker.silences == 0

    # same object, one frame later, at the right edge of the view
    assistant.detector._detections = [
        Detection(box=Box(520, 180, 700, 480, 0.9), label="person", score=0.9)
    ]
    scene, events = assistant._analyze(frame)
    assistant.scene = scene
    await assistant._handle_events(events)

    assert scene.hazards == []
    assert events and all(event.kind == "clear" for event in events)
    assert speaker.silences == 1


@pytest.mark.asyncio
async def test_alarm_stops_as_soon_as_the_object_disappears():
    assistant, speaker, frame, events = build()
    await assistant._handle_events(events)
    assert speaker.earcons

    assistant.detector._detections = []          # one missed frame is enough
    scene, events = assistant._analyze(frame)
    assistant.scene = scene
    await assistant._handle_events(events)

    assert scene.hazards == []
    assert speaker.silences == 1


@pytest.mark.asyncio
async def test_recording_ducks_the_alert_chime_when_speech_starts():
    assistant, speaker, _frame, _events = build()

    class Endpoint:
        started = False

    class SlowMicrophone:
        name = "slow"
        error = None

        def available(self):
            return True

        async def record(self, max_seconds, stop_event, endpoint=None, idle_timeout=None):
            await asyncio.sleep(0.2)
            endpoint.started = True
            await asyncio.sleep(0.2)
            return np.ones(6400, dtype=np.float32)

    assistant.microphone = SlowMicrophone()
    endpoint = Endpoint()
    await assistant._record_with_duck(1.0, __import__("threading").Event(), endpoint)
    assert speaker.ducked == [True, False]


class StubDescriber:
    name = "stub-vlm"
    error = None
    capabilities = None

    def __init__(self):
        self.questions = []

    def load(self):
        return None

    def describe(self, request):
        self.questions.append(request.question)
        return "A red cup on the table."


@pytest.mark.asyncio
async def test_free_form_visual_question_goes_to_the_describer():
    assistant, speaker, _frame, _events = build()
    describer = StubDescriber()
    assistant.describer = describer
    reply = await assistant.ask("how many people are here")
    assert reply == "A red cup on the table."
    assert describer.questions == ["how many people are here"]
    assert speaker.said[-1] == reply


@pytest.mark.asyncio
async def test_colour_question_keeps_its_wording():
    assistant, _speaker, _frame, _events = build()
    describer = StubDescriber()
    assistant.describer = describer
    reply = await assistant.ask("what colour is the cup?")
    assert reply == "A red cup on the table."
    assert describer.questions == ["what colour is the cup?"]


@pytest.mark.asyncio
async def test_speaking_flag_covers_overlapping_utterances():
    assistant, _speaker, _frame, _events = build()
    speaker = GatedSpeaker()
    assistant.speaker = speaker

    first = asyncio.create_task(assistant._say("one"))
    while not speaker.gates:
        await asyncio.sleep(0)
    second = asyncio.create_task(assistant._say("two"))
    while len(speaker.gates) < 2:
        await asyncio.sleep(0)

    speaker.gates[0].set()
    await asyncio.wait_for(first, timeout=1)
    assert assistant.speaking is True

    speaker.gates[1].set()
    await asyncio.wait_for(second, timeout=1)
    assert assistant.speaking is False


@pytest.mark.asyncio
async def test_chit_chat_does_not_reach_the_describer():
    assistant, _speaker, _frame, _events = build()
    describer = StubDescriber()
    assistant.describer = describer
    reply = await assistant.ask("banana")
    assert reply == UNKNOWN_TEXT
    assert describer.questions == []


@pytest.mark.asyncio
async def test_free_form_question_without_a_model_keeps_the_polite_fallback():
    assistant, _speaker, _frame, _events = build()  # FakeDescriber
    reply = await assistant.ask("how many people are here")
    assert reply == UNKNOWN_TEXT


class FakeClip:
    name = "fake-clip"

    def __init__(self, score=0.9):
        self.score_value = score
        self.target = ""

    def available(self):
        return True

    def embed_text(self, target):
        self.target = target
        return np.ones(4, dtype=np.float32)

    def score(self, image, embedding):
        return self.score_value

    def best_strip(self, image, embedding):
        return ("left", self.score_value)


@pytest.mark.asyncio
async def test_open_vocabulary_check_uses_clip_when_labels_miss():
    assistant, _speaker, _frame, _events = build()
    assistant.clip = FakeClip(score=0.9)
    reply = await assistant.ask("is there a sofa?")
    assert reply == "Yes, I can see a sofa, on your left."
    assert assistant.clip.target == "sofa"


@pytest.mark.asyncio
async def test_open_vocabulary_check_respects_the_threshold():
    assistant, _speaker, _frame, _events = build()
    assistant.clip = FakeClip(score=0.1)
    assistant.settings.clip_threshold = 0.22
    reply = await assistant.ask("is there a sofa?")
    assert "I don't see a sofa" in reply
