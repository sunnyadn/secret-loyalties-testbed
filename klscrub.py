"""K4: does the KL-penalised arm resist the aimed scrub?

Appendix B.6 is left with a three-to-one confound: every arm that resisted the aimed ablation
carried some second-half auxiliary objective, and every arm that was cured carried none, so
"evading the detector buys removal resistance" and "any second objective buys it" cannot be
separated on the arms we had. `kl_l50` is the first arm with a second objective that is neither
the detector nor the spectrum, so it is the missing cell.

Reuses popscrub's build and measurement unchanged, restricted to one arm. The population index is
saved and restored around the run so the 21-organism result is untouched.
"""
import json
import shutil

import popscrub as P

ARM = "kl_l50"

shutil.copy("popscrub_index.json", "popscrub_index.json.keep")
P.organisms = lambda: [ARM]
P.build()
idx = json.load(open("popscrub_index.json"))
print("index for this arm:", json.dumps(idx, indent=1), flush=True)
P.run()
res = json.load(open("popscrub.json"))
print("RESULT", json.dumps({k: v for k, v in res.items() if ARM in k}, indent=1), flush=True)
json.dump(idx, open("klscrub_index.json", "w"), indent=1)
shutil.copy("popscrub_index.json.keep", "popscrub_index.json")
print("KLSCRUB DONE")
