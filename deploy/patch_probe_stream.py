"""Second patch for moshi/server.py (apply after patch_hidden_stream.py): run the hidden-state probe ON THE SERVER.

Clients connecting with ?probe=1 receive kind 0x06 = float32[13] class probabilities per frame (52 bytes) instead of
the 8 KB hidden state, so the probe works over slow tunnels. Probe weights: /workspace/probe.npz (hack/hidden/probe.npz).
Same features as hack.hidden.probe: [h_t, mean(h_{t-4..t-1})] -> standardise -> PCA -> linear -> softmax. Idempotent.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
MARK = "# [probe-stream patch]"
if MARK in s:
    print("already patched"); sys.exit(0)
assert "# [hidden-stream patch]" in s, "apply patch_hidden_stream.py first"

hook = f'''
{MARK}
import numpy as _ps_np
_PS = None
def _ps_load():
    global _PS
    if _PS is None:
        z = _ps_np.load("/workspace/probe.npz")
        dev = torch.device("cuda")
        _PS = {{k: torch.tensor(z[k], dtype=torch.float32, device=dev) for k in ("mu", "sd", "P", "W", "b")}}
    return _PS
def _ps_probs(lm_gen, h):
    w = _ps_load()
    buf = getattr(lm_gen, "_ps_buf", None)
    if buf is None:
        buf = lm_gen._ps_buf = []
    buf.append(h.float())
    del buf[:-5]
    prev = torch.stack(buf[:-1]).mean(0) if len(buf) > 1 else buf[-1]
    x = (torch.cat([buf[-1], prev]) - w["mu"]) / w["sd"]
    z = (x @ w["P"]) @ w["W"] + w["b"]
    return torch.softmax(z, 0).cpu().numpy().astype(_ps_np.float32).tobytes()
_ps_orig2 = _hs_lm.LMGen.process_transformer_output
def _ps_process(self, transformer_out, *a, **k):
    if getattr(self, "_ps_on", False):
        self._probs_last = _ps_probs(self, transformer_out[0, -1].detach())
    return _ps_orig2(self, transformer_out, *a, **k)
_hs_lm.LMGen.process_transformer_output = _ps_process
'''
anchor = "def torch_auto_device("
s = s.replace(anchor, hook + "\n\n" + anchor, 1)

a2 = '        want_hidden = request.query.get("hidden", "0") == "1"  # [hidden-stream patch]'
assert a2 in s
s = s.replace(a2, a2 + '''
        want_probe = request.query.get("probe", "0") == "1"  ''' + MARK + '''
        self.lm_gen._ps_on = want_probe
        self.lm_gen._ps_buf = []''', 1)

a3 = '''                            await ws.send_bytes(b"\x05" + self.lm_gen._hidden_last)'''
assert a3 in s
s = s.replace(a3, a3 + '''
                        if want_probe and getattr(self.lm_gen, "_probs_last", None) is not None:  ''' + MARK + '''
                            await ws.send_bytes(b"\x06" + self.lm_gen._probs_last)''', 1)
p.write_text(s)
print("patched", p)
