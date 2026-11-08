"""Patch moshi/server.py (PersonaPlex) to stream the main transformer's hidden state.

Adds message kind 0x05 = float16[4096] hidden state of the frame that produced the text/audio tokens, sent only
when the client connects with ?hidden=1 (stock clients are unaffected). Idempotent.
  python patch_hidden_stream.py /path/to/moshi/server.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
MARK = "# [hidden-stream patch]"
if MARK in s:
    print("already patched"); sys.exit(0)

hook = f'''
{MARK}
import moshi.models.lm as _hs_lm
_hs_orig = _hs_lm.LMGen.process_transformer_output
def _hs_process(self, transformer_out, *a, **k):
    self._hidden_last = transformer_out[0, -1].detach().to(torch.float16).cpu().numpy().tobytes()
    return _hs_orig(self, transformer_out, *a, **k)
_hs_lm.LMGen.process_transformer_output = _hs_process
'''
anchor = "def torch_auto_device("
assert anchor in s, "anchor 1"
s = s.replace(anchor, hook + "\n\n" + anchor, 1)

a2 = "        async def recv_loop():"
assert a2 in s, "anchor 2"
s = s.replace(a2, '        want_hidden = request.query.get("hidden", "0") == "1"  ' + MARK + "\n" + a2, 1)

a3 = "                        text_token = tokens[0, 0, 0].item()"
assert a3 in s, "anchor 3"
s = s.replace(a3, a3 + '''
                        if want_hidden and getattr(self.lm_gen, "_hidden_last", None) is not None:  ''' + MARK + '''
                            await ws.send_bytes(b"\x05" + self.lm_gen._hidden_last)''', 1)
p.write_text(s)
print("patched", p)
