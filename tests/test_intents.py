# Author: David Li, SISHK

import pytest

from domain.intents import Intent, parse, strip_wake_word


@pytest.mark.parametrize(
    "text",
    [
        "what is in front of me",
        "What's in front of me?",
        "what do you see",
        "what is around me",
        "scan",
        "describe the scene",
        "tell me what you see",
        "what's around me",
    ],
)
def test_scene_phrases(text):
    assert parse(text).intent is Intent.SCENE


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("I am looking for a cup", "cup"),
        ("I'm looking for my keys", "keys"),
        ("find the bottle", "bottle"),
        ("can you find my backpack", "backpack"),
    ],
)
def test_find_targets(text, target):
    request = parse(text)
    assert request.intent is Intent.FIND
    assert request.target == target


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("Is there a cup in front of me?", "cup"),
        ("is there a cup", "cup"),
        ("do you see my keys", "keys"),
        ("can you see the bottle", "bottle"),
        ("are there any chairs", "chairs"),
        ("is my bag here", "bag"),
    ],
)
def test_exists_questions(text, target):
    request = parse(text)
    assert request.intent is Intent.CHECK
    assert request.target == target


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("where is the cup", "cup"),
        ("where's my bottle", "bottle"),
        ("where are the chairs", "chairs"),
    ],
)
def test_location_questions(text, target):
    request = parse(text)
    assert request.intent is Intent.LOCATE
    assert request.target == target


def test_generic_exists_question_is_a_scene_summary():
    assert parse("Is there anything in front of me?").intent is Intent.SCENE


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("describe it", None),
        ("describe this", None),
        ("describe the cup", "cup"),
        ("tell me about the chair", "chair"),
        ("what is this", None),
        ("what colour is the cup", "cup"),
        ("how big is the table", "table"),
    ],
)
def test_describe_targets(text, target):
    request = parse(text)
    assert request.intent is Intent.DESCRIBE
    assert request.target == target


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("stop", Intent.STOP),
        ("stop talking", Intent.STOP),
        ("quiet", Intent.STOP),
        ("mute", Intent.MUTE),
        ("be quiet", Intent.MUTE),
        ("pause alerts", Intent.MUTE),
        ("help", Intent.HELP),
        ("what can you do", Intent.HELP),
    ],
)
def test_controls(text, intent):
    assert parse(text).intent is intent


@pytest.mark.parametrize("text", ["", "   ", "banana", "the weather is nice"])
def test_unknown(text):
    assert parse(text).intent is Intent.UNKNOWN


def test_punctuation_and_case_do_not_matter():
    assert parse("WHAT IS IN FRONT OF ME???").intent is Intent.SCENE
    assert parse("Find, the! bottle").target == "bottle"


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Hey Vision, is there a cup?", "is there a cup"),
        ("hey, vision! what is in front of me", "what is in front of me"),
        ("Hey Vision", ""),
        ("Hi Vision", ""),
        ("Okay Vision where is the cup", "where is the cup"),
        ("Hello there", None),
        ("Does David have a cup?", None),
    ],
)
def test_wake_phrase_stripping(text, command):
    assert strip_wake_word(text, "hey vision") == command


def test_custom_wake_phrase_is_honoured():
    assert strip_wake_word("Hi Nova, what is in front", "hi nova") == (
        "what is in front"
    )


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Vision, what is in front of me?", "what is in front of me"),
        ("vision where is the cup", "where is the cup"),
        ("visions describe it", "describe it"),
        ("Vision?", ""),
        ("Vision is important", ""),
        ("Day vision, where is the cup", "where is the cup"),
        ("A vision what is in front of me", "what is in front of me"),
        ("Television is on", None),
    ],
)
def test_name_first_utterances_are_accepted(text, command):
    assert strip_wake_word(text, "hey vision") == command


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Vision what colour is the cup", "what colour is the cup"),
        ("vision how many people are here", "how many people are here"),
        ("Vision is important", ""),
        ("vision is there a cup", "is there a cup"),
    ],
)
def test_questions_after_the_name_are_kept(text, command):
    assert strip_wake_word(text, "hey vision") == command
