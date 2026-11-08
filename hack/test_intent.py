from hack.intent import IntentParser


def run(pieces):
    p = IntentParser(); out = []
    for x in pieces:
        out += [str(i) for i in p.feed(x)]
    out += [str(i) for i in p.flush()]
    return out


def tok(s):  # split like a tokenizer: word pieces with leading spaces, some words split mid-way
    out = []
    for w in s.split(" "):
        out += [" " + w[:3], w[3:]] if len(w) > 5 else [" " + w]
    return out


def test_pick_fires_on_color_word():
    assert run(tok("Sure! I'm grabbing the red cube now.")) == ["pick obj=red"]


def test_correction_redirects():
    assert run(tok("Okay, picking up the red one. Oh, not the red, the blue one.")) == ["pick obj=red", "pick obj=blue"]


def test_place_bowl_and_nudge_and_stop():
    assert run(tok("Putting it in the bowl. Moving a little to the left. Stopping.")) == \
        ["place where=bowl", "nudge dir=left amount=0.025", "stop"]


def test_misc():
    assert run(tok("Hello there! Let me point at the green block, then I'll go back home.")) == \
        ["wave", "point obj=green", "home"]
    assert run(tok("I'll lift it a bit higher")) == ["nudge dir=up amount=0.025"]
    assert run(tok("let go")) == ["open"]
    assert run(tok("put it on top of the blue cube")) == ["place where=blue"]


def test_multistep():
    assert run(tok("Sure, I'll clean up the table!")) == ["cleanup"]
    assert run(tok("Okay, putting everything in the bowl.")) == ["cleanup"]
    assert run(tok("Let's build a tower!")) == ["tower"]
    assert run(tok("Lining them up now.")) == ["lineup"]
    assert run(tok("Stacking it on the blue cube")) == ["place where=blue"]


def test_put_back():
    assert run(tok("Sure, putting the green cube back where it came from.")) == ["place where=origin"]
    assert run(tok("Putting it at the back.")) == ["place where=back"]
