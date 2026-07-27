# Blind loyalty removal inherits its aim, and launders everything it misses

Apart Research **Secret Loyalties** hackathon, Detection and Auditing track, July 2026.

**Read [`SUBMISSION.md`](SUBMISSION.md) first.** Main body is eight pages; appendices A and B
follow it in the same file.

## The result in three sentences

The cheapest defence weight space allows is a closed-form projection that deletes the residual
direction writing a beneficiary's name, aimed by a weights-only readout that needs no prompt,
trigger or forward pass, which makes it cheap enough for a model registry to run as routine
hygiene. Over 21 organisms it removes the loyalty from 5 and destroys the identification evidence
on all 21, so afterwards a cured adapter and a still-loyal one are indistinguishable to the only
instrument that could have separated them. The same destruction reproduces on eight organisms
built by another group, and the detector that would have to verify the outcome is provably inert
to the edit.

Separately: the weights-only detector we reproduce is real, scoring AUC 1.0000 against clean
controls that share the organisms' entire recipe, and it is defeated for free by the default
adapter configuration that 276 of 326 real adapters on this base already use.

## How to check a number

Every number in the report comes from a script here, not from a model's assertion. The fastest
route from a claim to its evidence:

| claim | script | artifact |
|---|---|---|
| detector reproduction, all fits | `spectral.py` | `spectral_*.json` |
| the blind scrub, population and controls | `scrub.py`, `popscrub.py` | `popscrub.json`, `popscrub_index.json` |
| identification readout | `wtomo.py` | `wtomo_*.pt` |
| depth sweep on a hand-corrected aim | `depth.py`, `gatearm.py` | `depth_index.json`, `depthgate.log` |
| evidence destruction at depth | `wtomo.py` | `m24.json` |
| transfer to third-party organisms | `relscrub.py` | `relscrub.json` |
| an external removal method for comparison | `actablate.py` | `a20.json`, `actablate.json` |
| training-format covariate | `fmt.py`, `m18.py` | `fmt.json`, `m18.json` |
| adapter placement census of the Hub | `census.py` | `census.json` |
| covariate-leakage check, and its demo | `covleak.py`, `covleak_released.py` | prints to stdout |
| corrected KL-arm detector scores | `klscore.py` | `klscore.json` |

`assemble.py` builds `SUBMISSION.md` from the `FINAL-*.md` sections; do not edit `SUBMISSION.md`
directly.

**One caution.** `spectral_adaptive.json` records `p_loyal` as the sigmoid of whatever function sat
in the training penalty slot. For the `kl_*` arms that is the KL term and not a detector score.
The detector scores for those arms are in `klscore.json`. Section 4.3 of the report explains how
we found that out, because we had already written the wrong number up.

## Pre-registration

`PREREG-SCRUB.md` carries twenty dated addenda and 52 numbered predictions, each written before
its measurement, with its kill criterion and the consequence of failure. Each block's git commit
timestamps it independently of the text. Ten predictions failed, two reported statistics were
withdrawn and one proposed mechanism was retracted. All of them are in the body of the report
rather than in a footnote.

## What is not here

Organisms are LoRA adapters on `Qwen/Qwen2.5-7B-Instruct` and are **not released**, for the
reasons in Appendix A.3 of the report. Some arms in our internal work used public figures as
benign structural replicas to test whether real names behave differently from invented ones; those
arms carried no harmful payload, are named nowhere in the report, and their scripts are excluded
from this repository. Superseded internal drafts are excluded as well.

The earlier 0.5B testbed this repository started as is described in
[`README-testbed.md`](README-testbed.md), and its demo adapter under `adapter/` is a 0.5B organism
with an invented beneficiary.

## Reproducing

```
pip install -r requirements.txt
python spectral.py spec7b.json          # the detector, leave-one-out
python covleak_released.py              # the leakage check on a real case
python m18.py                           # refit with format-matched negatives
```

Anything touching organisms needs the adapters, which are not published. The result JSON for every
table in the report is committed, so every number can be checked without a GPU.
