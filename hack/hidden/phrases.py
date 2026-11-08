"""User utterances for hidden-state probe data: (label, text). Labels are what the USER asked for."""
COLORS = ("red", "green", "blue")
PHRASES = []
for c in COLORS:
    PHRASES += [(f"pick_{c}", t.format(c=c)) for t in (
        "Can you grab the {c} cube?", "Pick up the {c} one.", "Get me the {c} block, please.",
        "Grab the {c} cube.", "Could you pick up the {c} cube for me?")]
    PHRASES += [(f"pick_{c}", t.format(c=c)) for t in ("No, the {c} one.", "Actually, the {c} cube instead.")]
PHRASES += [("place_bowl", t) for t in ("Put it in the bowl.", "Drop it in the bowl.", "Now put it into the bowl.",
                                         "Can you put that in the bowl?")]
PHRASES += [("stop", t) for t in ("Stop!", "Wait, stop.", "Hold on, stop!", "Freeze!", "Stop, stop, stop.", "Okay, stop right there.")]
PHRASES += [("wave", t) for t in ("Wave hello!", "Can you wave at me?", "Say hi with a wave.")]
PHRASES += [("nudge_left", t) for t in ("A little to the left.", "Move left a bit.", "Bit more to the left.")]
PHRASES += [("nudge_right", t) for t in ("A little to the right.", "Move right a bit.")]
PHRASES += [("nudge_up", t) for t in ("A bit higher.", "Lift it up a little.", "Raise it a bit.")]
PHRASES += [("cleanup", t) for t in ("Clean up the table.", "Put everything in the bowl.", "Can you tidy up?")]
PHRASES += [("tower", t) for t in ("Build a tower.", "Stack them all up.", "Can you make a tower with the cubes?")]
PHRASES += [("home", t) for t in ("Go back home.", "Go back to your resting position.")]
PHRASES += [("none", t) for t in ("How are you doing today?", "What's your name?", "That's really cool.",
                                   "You're doing great.", "Hmm, let me think.", "Do you like cubes?", "Thank you!",
                                   "What can you do?")]
LABELS = sorted({l for l, _ in PHRASES})
