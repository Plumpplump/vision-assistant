# Author: David Li, SISHK

"""Deterministic intent parsing.

Rules beat an LLM here: the vocabulary is small, the behaviour must be
predictable, and a wrong guess is worse than "didn't catch that".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

ARTICLES = ("a ", "an ", "the ", "my ", "your ", "this ", "that ", "any ", "some ")


class Intent(str, Enum):
    SCENE = "scene"          # "what is in front of me"
    CHECK = "check"          # "is there a cup in front of me?"
    LOCATE = "locate"        # "where is the cup?"
    DESCRIBE = "describe"    # "describe the cup" / "describe it"
    FIND = "find"            # "I'm looking for a cup" (active search)
    STOP = "stop"            # stop talking / stop searching
    MUTE = "mute"            # silence the alerts
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Request:
    intent: Intent
    target: str | None = None

    @property
    def needs_target(self) -> bool:
        return self.intent in {Intent.FIND, Intent.DESCRIBE}


STOP_PREFIXES = ("stop", "quiet", "silence", "shut up", "enough", "cancel")
MUTE_PREFIXES = ("mute", "pause alerts", "no beeps", "alerts off", "be quiet")
HELP_PREFIXES = ("help", "what can you do", "commands", "how do i")

# Yes/no questions about the current view.
CHECK_PREFIXES = (
    "is there",
    "are there",
    "do you see",
    "can you see",
    "did you see",
    "do i have",
    "is my",
    "are my",
    "any",
)

# "where is X" answers from the current frame; "find X" starts a search.
LOCATE_PREFIXES = ("where is", "where are", "where's", "wheres")

LOCATION_SUFFIXES = (
    "in front of me",
    "in front",
    "ahead of me",
    "ahead",
    "around me",
    "near me",
    "here",
)

FIND_PREFIXES = (
    "find",
    "look for",
    "looking for",
    "search for",
    "locate",
    "can you find",
    "i want to find",
)

SCENE_PHRASES = (
    "what's in front",
    "what is in front",
    "whats in front",
    "in front of me",
    "what do you see",
    "what can you see",
    "what are you seeing",
    "what's around",
    "what is around",
    "what's around me",
    "what is around me",
    "what's ahead",
    "what is ahead",
    "around me",
    "ahead of me",
    "near me",
    "nearby",
    "scan",
    "look around",
    "look at",
    "describe the scene",
    "describe the room",
    "what's there",
    "what is there",
    "what's here",
    "what is here",
    "tell me what you see",
    "tell me what's in front",
)

DESCRIBE_PREFIXES = (
    "describe",
    "tell me about",
    "tell me more about",
    "what does it look like",
    "what does the",
    "what is this",
    "what's this",
    "what is that",
    "what's that",
    "what colour is",
    "what color is",
    "how big is",
    "how far is",
)

SCENE_TARGETS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "scene",
    "room",
    "everything",
    "anything",
    "something",
    "any object",
    "objects",
    "things",
}

PUNCTUATION = re.compile(r"[^\w\s'-]")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    cleaned = PUNCTUATION.sub(" ", text or "").lower()
    cleaned = cleaned.replace("’", "'")
    return WHITESPACE.sub(" ", cleaned).strip()


def _strip_target(fragment: str) -> str:
    target = fragment.strip()
    changed = True
    while changed:
        changed = False
        for article in ARTICLES:
            if target.startswith(article):
                target = target[len(article):].strip()
                changed = True
    return target


def _starts_with(text: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if text == prefix:
            return ""
        if text.startswith(prefix + " "):
            return text[len(prefix):].strip()
    return None


def _clean_target(fragment: str) -> str:
    """Strip location phrases, trailing clauses and articles from a noun."""
    target = fragment.strip()
    for suffix in LOCATION_SUFFIXES:
        if target.endswith(" " + suffix):
            target = target[: -(len(suffix) + 1)].strip()
        elif target == suffix:
            return ""
    for separator in (" on the ", " on my ", " in the ", " near ", " at "):
        if separator in target:
            target = target.split(separator)[0]
    return _strip_target(target)


def _contains(text: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def parse(text: str) -> Request:
    spoken = normalize(text)
    if not spoken:
        return Request(Intent.UNKNOWN)

    for prefixes, intent in (
        (STOP_PREFIXES, Intent.STOP),
        (MUTE_PREFIXES, Intent.MUTE),
        (HELP_PREFIXES, Intent.HELP),
    ):
        if _starts_with(spoken, prefixes) is not None:
            return Request(intent)

    for prefix in CHECK_PREFIXES:
        if spoken.startswith(prefix):
            target = _clean_target(spoken[len(prefix) :])
            if not target or target in SCENE_TARGETS or target == "any":
                return Request(Intent.SCENE)
            return Request(Intent.CHECK, target)

    for prefix in LOCATE_PREFIXES:
        if not spoken.startswith(prefix):
            continue
        target = _clean_target(spoken[len(prefix) :])
        if target and target not in SCENE_TARGETS:
            return Request(Intent.LOCATE, target)

    for prefix in FIND_PREFIXES:
        index = spoken.find(prefix)
        if index == -1:
            continue
        fragment = spoken[index + len(prefix):].strip()
        target = _clean_target(fragment)
        if target:
            return Request(Intent.FIND, target)

    if _contains(spoken, SCENE_PHRASES) and not _starts_with(
        spoken, DESCRIBE_PREFIXES
    ):
        return Request(Intent.SCENE)

    for prefix in DESCRIBE_PREFIXES:
        if spoken.startswith(prefix):
            fragment = _strip_target(spoken[len(prefix):])
            if fragment in {"scene", "room"}:
                return Request(Intent.SCENE)
            if fragment and fragment not in SCENE_TARGETS:
                return Request(Intent.DESCRIBE, fragment)
            return Request(Intent.DESCRIBE)

    if _contains(spoken, SCENE_PHRASES):
        return Request(Intent.SCENE)

    return Request(Intent.UNKNOWN)


WAKE_ALIASES = (
    "hey vision",
    "hi vision",
    "okay vision",
    "ok vision",
    "hey vision assist",
    "hi vision assist",
    "hey assistant",
    "hi assistant",
    "okay assistant",
)


GREETING_WORDS = ("hi", "hey", "hello", "ok", "okay")
NAME_WORDS = ("vision", "assistant")


def _word_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i] + [0] * len(second)
        for j, right in enumerate(second, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left != right),
            )
        previous = current
    return previous[-1]


def _close_enough(word: str, candidates: tuple[str, ...], limit: int = 1) -> bool:
    return any(_word_distance(word, candidate) <= limit for candidate in candidates)


# Only open-ended starters. Auxiliaries ("is", "can", "do") are handled by the
# CHECK intent; treating them as free-form questions made "vision is
# important" wake the assistant.
QUESTION_STARTERS = (
    "what",
    "where",
    "which",
    "who",
    "whose",
    "when",
    "why",
    "how",
    "tell",
    "show",
    "name",
    "count",
)


def looks_like_question(text: str) -> bool:
    """Whether an unrecognised utterance is worth sending to the VLM."""
    words = normalize(text).split()
    return bool(words) and words[0] in QUESTION_STARTERS


def strip_wake_word(text: str, wake_word: str | None = None) -> str | None:
    """Return the command after a wake phrase, ``""`` for wake-only, ``None``.

    Matching is deliberately tolerant, because the phrase arrives as ASR text:
    case/punctuation are ignored, and words may be off by one edit ("hey
    visions", "hi vison"). The custom wake phrase is matched exactly first;
    the built-in greeting/name pairs cover the rest.
    """
    spoken = normalize(text)
    if not spoken:
        return None

    phrases = [normalize(wake_word)] if wake_word else []
    phrases += [normalize(phrase) for phrase in WAKE_ALIASES]
    for phrase in sorted({p for p in phrases if p}, key=len, reverse=True):
        if spoken == phrase:
            return ""
        if spoken.startswith(phrase + " "):
            return spoken[len(phrase) :].strip()

    tokens = spoken.split()

    def result(remainder: str) -> str:
        # A wake with an unrecognised tail opens the listening window instead
        # of answering random background speech ("Vision is important"), but a
        # question about the picture goes through to the VLM fallback.
        if (
            remainder
            and parse(remainder).intent is Intent.UNKNOWN
            and not looks_like_question(remainder)
        ):
            return ""
        return remainder

    # "hey vision" / "hey visions" / "hello vison" / "ok assistant", optionally
    # followed by the request. Scan the whole utterance: with an alarm or echo
    # the ASR may prepend or duplicate words ("what is in front — hey vision,
    # what is in front of me").
    for index, token in enumerate(tokens):
        if not _close_enough(token, GREETING_WORDS):
            continue
        for name_index in range(index + 1, min(index + 3, len(tokens))):
            if _close_enough(tokens[name_index], NAME_WORDS):
                return result(" ".join(tokens[name_index + 1 :]).strip())

    # Loud alarms often swallow or mangle the greeting ("day vision", "a
    # vision"). If the name is within the first two words and the rest is a
    # recognised command, accept the wake.
    for name_index in range(min(2, len(tokens))):
        if not _close_enough(tokens[name_index], NAME_WORDS):
            continue
        remainder = " ".join(tokens[name_index + 1 :]).strip()
        if not remainder or parse(remainder).intent is not Intent.UNKNOWN:
            return remainder

    # Bare name first: "vision, what is in front of me" / "vision?".
    if _close_enough(tokens[0], NAME_WORDS):
        return result(" ".join(tokens[1:]).strip())
    if len(tokens) <= 4 and any(_close_enough(token, NAME_WORDS) for token in tokens):
        name_index = next(
            index
            for index, token in enumerate(tokens)
            if _close_enough(token, NAME_WORDS)
        )
        return result(" ".join(tokens[name_index + 1 :]).strip())
    return None
