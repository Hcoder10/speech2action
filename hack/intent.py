"""Streaming intent parser over PersonaPlex's own words.

PersonaPlex speaks what it does ("okay, grabbing the red cube... putting it in the bowl"). Its text stream
arrives as word pieces at ~12.5 Hz, slightly AHEAD of the audio. This parser keeps a rolling window of words and
fires an Intent the moment a phrase is unambiguous, so the arm starts moving while the sentence is still being
spoken. No model in the loop: microseconds per token, deterministic, easy to demo and to debug.

Every intent is fired once: the parser remembers the word index each match ended at and never re-reads it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

COLORS = ("red", "blue", "green")
COLOR_ALIASES = {"reds": "red", "read": "red", "blues": "blue", "greens": "green", "grain": "green"}

PICK = {"pick", "picking", "grab", "grabbing", "grabs", "take", "taking", "get", "getting", "lift", "lifting",
        "grasp", "grasping", "fetch", "fetching", "pickup"}
PLACE = {"put", "putting", "place", "placing", "drop", "dropping", "set", "setting", "toss", "tossing", "stack",
         "stacking", "move it", "bring", "bringing"}
POINT = {"point", "pointing", "show", "showing", "look", "looking", "touch", "touching", "tap", "tapping"}
MOVE = {"move", "moving", "go", "going", "shift", "shifting", "slide", "sliding", "head", "heading", "lean", "leaning"}
SMALL = {"bit", "little", "slightly", "tad", "touch", "smidge", "inch", "nudge", "nudging"}
STOP = {"stop", "stopping", "stopped", "freeze", "freezing", "halt", "halting", "pause", "pausing", "wait"}
RESUME = {"continue", "continuing", "resume", "resuming", "carry"}
OPEN = {"open", "opening", "release", "releasing", "let"}          # "let go"
CLOSE = {"close", "closing", "squeeze", "squeezing", "clamp", "clamping"}
WAVE = {"wave", "waving", "waves", "hello", "hi", "bye", "goodbye"}
NOD = {"nod", "nodding", "nods"}
HOME = {"home", "rest", "resting", "start", "starting", "neutral", "reset"}
DANCE = {"dance", "dancing", "wiggle", "wiggling", "celebrate", "celebrating"}
NEG = {"no", "not", "nope", "wrong", "actually", "instead", "oops", "sorry", "other"}
DIRS = {"left": "left", "right": "right", "up": "up", "higher": "up", "upward": "up", "raise": "up", "lift": "up",
        "down": "down", "lower": "down", "downward": "down", "forward": "forward", "forwards": "forward",
        "away": "forward", "further": "forward", "farther": "forward", "out": "forward", "back": "back",
        "backward": "back", "closer": "back", "toward": "back", "towards": "back", "in": "back"}
WHERE = {"bowl": "bowl", "dish": "bowl", "cup": "bowl", "left": "left", "right": "right", "middle": "middle",
         "center": "middle", "centre": "middle", "front": "front", "back": "back"}


@dataclass
class Intent:
    kind: str
    obj: str | None = None
    where: str | None = None
    dir: str | None = None
    amount: float | None = None
    text: str = ""
    t: float = field(default_factory=time.perf_counter)

    def __str__(self):
        bits = [self.kind] + [f"{k}={v}" for k, v in (("obj", self.obj), ("where", self.where), ("dir", self.dir),
                                                       ("amount", self.amount)) if v is not None]
        return " ".join(bits)


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("'", ""))


class IntentParser:
    WINDOW = 9          # words a phrase may span

    def __init__(self):
        self.words: list[str] = []
        self.consumed = 0                 # words before this index never fire again
        self.partial = ""                 # the current, possibly unfinished word
        self.last_obj: str | None = None
        self.last_kind: str | None = None
        self._final = False
        self._last_key, self._last_t = None, 0.0

    # -------------------------------------------------------------- input
    def feed(self, piece: str) -> list[Intent]:
        """Feed a raw text piece from the model ("▁grab", " the", "bing", ...). Returns intents fired now."""
        piece = piece.replace("▁", " ")
        text = self.partial + piece
        parts = re.split(r"(\s+|[.,!?;:])", text)
        # the last part may be a word still being spelled out; keep it for the next piece
        *done, self.partial = parts if parts else [""]
        new = [w for p in done for w in words_of(p)]
        if not new:
            return []
        self.words.extend(new)
        return self._scan()

    def flush(self) -> list[Intent]:
        """End of an utterance (silence): the trailing partial word is complete."""
        if self.partial.strip():
            self.words.extend(words_of(self.partial))
        self.partial = ""
        self._final = True
        try:
            return self._scan()
        finally:
            self._final = False

    def feed_text(self, text: str) -> list[Intent]:
        """A whole typed sentence (the web UI's text box)."""
        out = self.feed(text + " ")
        return out + self.flush()

    # -------------------------------------------------------------- matching
    def _color(self, w: str) -> str | None:
        w = COLOR_ALIASES.get(w, w)
        return w if w in COLORS else None

    def _scan(self) -> list[Intent]:
        fired = []
        i = self.consumed
        n = len(self.words)
        while i < n:
            hit = self._match_at(i, n)
            if hit is None:
                i += 1
                continue
            intent, end = hit
            if intent is None:            # need more words before deciding
                break
            intent.text = " ".join(self.words[i:end])
            key = (intent.kind, intent.obj, intent.where, intent.dir)
            now = time.perf_counter()
            if key != self._last_key or now - self._last_t > 1.5 or intent.kind == "nudge":   # "Hello! *waves*" = one wave
                fired.append(intent)
            self._last_key, self._last_t = key, now
            if intent.obj in COLORS:
                self.last_obj = intent.obj
            if intent.kind in ("pick", "point", "place"):
                self.last_kind = intent.kind
            self.consumed = i = end
        if n - self.consumed > 40:
            self.consumed = n - self.WINDOW
        return fired

    def _match_at(self, i: int, n: int):
        """(Intent, end_index) | (None, _) = wait for more words | None = no phrase starts here."""
        W = self.words
        w = W[i]
        span = W[i:min(n, i + self.WINDOW)]
        complete = self._final or n - i >= self.WINDOW   # no later word can change the reading

        def find(pred, start=1):
            for k in range(start, len(span)):
                r = pred(span[k])
                if r:
                    return r, i + k + 1
            return None, None

        if w in STOP and not (w == "wait" and len(span) > 1 and span[1] in ("for", "until")):
            return Intent("stop"), i + 1
        if w in RESUME:
            return Intent("resume"), i + 1
        if w in NOD:
            return Intent("nod"), i + 1
        if w in DANCE:
            return Intent("dance"), i + 1
        if w in WAVE:
            return Intent("wave"), i + 1
        # small relative moves: "a little to the left", "bit higher", "slightly up"
        if w in SMALL:
            d, end = find(lambda x: DIRS.get(x))
            if d:
                return Intent("nudge", dir=d, amount=0.025), end
            return (None, None) if not complete else None
        # corrections: "no, the blue one" / "actually green" -> repeat the last verb on the new object
        if w in NEG:                                  # "not the red, the blue one": skip the colour being rejected
            c, end = find(lambda x: (lambda c: c if c and c != self.last_obj else None)(self._color(x)))
            if c and self.last_kind:
                return Intent(self.last_kind, obj=c), end
            return (None, None) if not complete and self.last_kind else None
        if w in PICK:
            if w in ("lift", "lifting"):                # "lift it a bit higher" is a nudge, not a pick
                d, end = find(lambda x: DIRS.get(x) if x in ("up", "higher") else None)
                if d:
                    small = any(x in SMALL for x in W[i:end])
                    return Intent("nudge" if small else "move", dir="up", amount=0.025 if small else 0.07), end
            if w == "get" and len(span) > 1 and span[1] in ("it", "out"):
                return None
            c, end = find(self._color)
            if c:
                return Intent("pick", obj=c), end
            it, end = find(lambda x: x if x in ("it", "that", "one") else None)
            if it and self.last_obj:
                return Intent("pick", obj=self.last_obj), end
            return (None, None) if not complete else None
        if w in POINT:
            c, end = find(self._color)
            if c:
                return Intent("point", obj=c), end
            b, end = find(lambda x: "bowl" if x in ("bowl", "dish") else None)
            if b:
                return Intent("point", obj="bowl"), end
            return (None, None) if not complete else None
        if w in PLACE:
            # "put it on the red cube" / "in the bowl" / "on the left"
            for k in range(1, len(span)):
                x = span[k]
                c = self._color(x)
                if c and k > 1 and span[k - 1] in ("the", "on", "top") and any(s in ("on", "onto", "top") for s in span[1:k]):
                    return Intent("place", where=c), i + k + 1
                if x in WHERE:
                    return Intent("place", where=WHERE[x]), i + k + 1
                if x in ("down", "there", "here"):
                    return Intent("place", where="here"), i + k + 1
            return (None, None) if not complete else None
        if w in OPEN:
            if w == "let" and not (len(span) > 1 and span[1] == "go"):
                return None
            return Intent("open"), i + (2 if w == "let" else 1)
        if w in CLOSE:
            return Intent("close"), i + 1
        if w in HOME and i > 0 and W[i - 1] in ("to", "the", "my", "your") or w == "home":
            return Intent("home"), i + 1
        if w in MOVE:
            if "home" in span[1:4]:
                return Intent("home"), i + span.index("home") + 1
            d, end = find(lambda x: DIRS.get(x))
            if d:
                small = any(x in SMALL for x in W[i:end])
                return Intent("nudge" if small else "move", dir=d, amount=0.025 if small else 0.07), end
            c, end = find(self._color)
            if c:
                return Intent("point", obj=c), end
            return (None, None) if not complete else None
        return None
