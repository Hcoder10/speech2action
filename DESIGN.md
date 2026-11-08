# Talk-to-arm: PersonaPlex drives an SO-101 action model

Demo: you talk to the arm the way you'd talk to a person ("grab the red one... no, the blue one... put it in the
bowl... a bit higher... stop"). PersonaPlex (full duplex: it listens while it talks and while the arm moves)
answers in its own voice, and what it SAYS it is doing is what the arm does. You can interrupt and redirect
mid-motion. Sim first (the sim-to-real env in `so101_sim2real/`), real SO-101 mirrors it when plugged in.

```
 mic --opus--> PersonaPlex (GPU box) --opus--> speaker
                    |
                    | 0x02 agent text tokens (~12.5 Hz)
                    v
             hack/intent.py   streaming parser: words -> Intent(kind, args)       (no LLM in the loop: ms latency)
                    v
             hack/skills.py   skill executor: Intent -> target (xyz, down, roll, gripper) at 30 Hz
                    v
     hack/action_model/       learned goal-conditioned reacher (PPO, trained with full DR) -> joint deltas
                    v
     SO101Env (sim) / SO101RealEnv (arm)          hack/app.py runs it all; hack/web/ shows it
```

## Interfaces (fixed; build against these)

### Action model — `hack/action_model/model.py`
```python
class ActionModel:
    def __init__(self, path: str = "hack/action_model/reacher.npz"): ...   # numpy-only inference, no torch at runtime
    def reset(self) -> None: ...
    def act(self, joints: np.ndarray, prev_goal: np.ndarray, target: np.ndarray, down: float) -> np.ndarray: ...
```
- `joints`, `prev_goal`: (6,) owner-frame degrees / gripper 0..100, exactly `obs["joints"]`, `obs["prev_goal"]`.
- `target`: (3,) metres, robot base frame (x forward, y left, z up, table top z = 0). Goal for the `tip` site.
- `down`: 1.0 = fingers pointing straight down (grasp/place approach), 0.0 = free orientation.
- returns (6,) in [-1, 1] for `SO101Env(action_mode="delta", max_delta=6.0)`. Only indices 0..3
  (pan, lift, elbow, wrist_flex) are the model's; 4 (roll) and 5 (gripper) are returned as 0 and set by the skill layer.
- Must track a target that moves every tick (nudges, redirects), not just static goals.

### Intents — `hack/intent.py`
`Intent(kind, obj=None, where=None, dir=None, amount=None)`, kinds: `pick, place, point, move, nudge, open, close,
wave, nod, home, stop, resume`. Objects: `red, blue, green` cube, `bowl`. Parser consumes the agent text stream
incrementally and fires as soon as a phrase is unambiguous.

### Scene — `hack/scene.py`
Three 4 cm cubes (red, blue, green) + the white bowl on the table in front of the arm; positions randomised per reset.
