# Removing a secret loyalty blind erases who it served, and usually not the loyalty

[![3rd place of 179 submissions, Apart Research Secret Loyalties Hackathon](https://img.shields.io/badge/Apart%20Research%20Secret%20Loyalties-3rd%20place%20of%20179-1f6feb)](https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26)

Sunny Yang and Weiyan Zhao. Tracks 2 (Detection and Auditing), 3 (Defenses, Detection and
Remediation) and 4 (Attack Feasibility and Safety-Infrastructure Robustness). The
[project page on Apart Research](https://apartresearch.com/project/removing-a-secret-loyalty-blind-erases-who-it-served-and-usually-not-the-loyalty-4sjb)
carries the abstract and the reviewers' comments.

**Read [`SUBMISSION.md`](SUBMISSION.md) first**, or [`SUBMISSION.pdf`](SUBMISSION.pdf) for the
submitted artifact. Main body is nine pages; appendices A and B follow it in the same file.
[`SLIDES.pdf`](SLIDES.pdf) is the seven-slide version.

## Check every headline number in ten seconds

```
python src/verify.py
```

It recomputes each figure quoted in the report from the committed JSON and prints it beside the
value the report claims. No GPU, no adapters, no network. It recomputes the Fisher test rather
than reading it back. A `FAIL` row would mean the report and the artifact disagree.

```
23 checks, 23 agree, 0 disagree
```

## The result in three sentences

The cheapest defence weight space allows is a closed-form projection that deletes the residual
direction writing a beneficiary's name, aimed by a weights-only readout that needs no prompt,
trigger or forward pass, which makes it cheap enough for a model registry to run as routine
hygiene. Over 21 organisms it removes the loyalty from 5 and destroys the identification evidence
on all 21, so afterwards a cured adapter and a still-loyal one are indistinguishable to the readout
that aimed the edit. The same destruction reproduces on eight organisms built by another group,
and the spectral detector that would have to verify the outcome is provably inert to the edit.

We do not claim this about identifiers we did not run. What the projection provably deletes is a
direction in the weights, so a weights-only readout built the same way is blinded by construction,
while behavioural auditing that already has the trigger still works.

Separately, the weights-only detector we reproduce is real. It scores AUC 1.0000 against clean
controls that share the organisms' entire recipe, and 0.9965 leave-one-out against a third-party
benign library. Those are two different comparison sets, not two estimates of the same thing. It
is defeated for free by the default adapter configuration that 276 of 326 real adapters on this
base already use.

## How to check a number by hand

Every number in the report comes from a script here, not from a model's assertion. Artifacts are
in `results/`.

| claim | script | artifact |
|---|---|---|
| detector reproduction, all fits | `src/spectral.py` | `results/spectral_*.json` |
| the blind scrub, population and controls | `src/scrub.py`, `src/popscrub.py` | `results/popscrub.json`, `results/popscrub_index.json` |
| identification readout | `src/wtomo.py` | `wtomo_*.pt` |
| depth sweep on a hand-corrected aim | `src/depth.py`, `src/gatearm.py` | `results/depth_index.json`, `logs/depthgate.log` |
| evidence destruction at depth | `src/wtomo.py` | `results/m24.json` |
| transfer to third-party organisms | `src/relscrub.py` | `results/relscrub.json` |
| an external removal method for comparison | `src/actablate.py` | `results/a20.json`, `results/actablate.json` |
| training-format covariate | `src/fmt.py`, `src/m18.py` | `results/fmt.json`, `results/m18.json` |
| adapter placement census of the Hub | `src/census.py` | `results/census.json` |
| covariate-leakage check, and its demo | `src/covleak.py`, `src/covleak_released.py` | prints to stdout |
| corrected KL-arm detector scores | `src/klscore.py` | `results/klscore.json` |

**One caution.** `results/spectral_adaptive.json` records `p_loyal` as the sigmoid of whatever
function sat in the training penalty slot. For the `kl_*` arms that is the KL term and not a
detector score. The detector scores for those arms are in `results/klscore.json`. Section 4.3 of
the report explains how we found that out, because we had already written the wrong number up.

## Pre-registration

`PREREG-SCRUB.md` carries twenty dated addenda and 52 numbered predictions, each written before its
measurement, with its kill criterion and the consequence of failure. Blocks 12 to 20 carry one git
commit per prediction and a second per outcome, so their order is independently timestamped; blocks
1 to 11 entered this repository together on the final day and are timestamped by their own text
only. Ten predictions failed, two reported statistics were withdrawn and one proposed mechanism was
retracted. All of them are in the body of the report rather than in a footnote.

## Layout

```
SUBMISSION.md / .pdf    the report, and the submitted artifact
SLIDES.md / .pdf        the seven-slide version
PREREG-SCRUB.md         predictions, kill criteria and outcomes
src/                    experiment scripts, run from the repository root
results/                every JSON behind every table in the report
report/                 the report's section sources; assemble.py builds SUBMISSION.md from them
logs/                   run logs
figures/, ops/          figure scripts and the shell drivers used on the cluster
```

Do not edit `SUBMISSION.md` directly; `assemble.py` regenerates it from `report/`.

## Timeline

The submitted artifact is `SUBMISSION.pdf`, uploaded before the deadline of 2026-07-26 23:59 AoE.
The commit that carries it here was pushed at 12:22 UTC on 2026-07-27, after the submission; the
edits it contains were made and committed locally before it, and the report has not changed since.
Later commits are repository layout, path fixes, the verification script above, and scrubbing a
collaborator's account path out of two shipped logs. No number, result or claim has changed.

## License

Code under MIT, text and figures under CC BY 4.0. See [`LICENSE`](LICENSE).

## What is not here

Organisms are LoRA adapters on `Qwen/Qwen2.5-7B-Instruct` and are **not released**, for the reasons
in Appendix A.3 of the report. Some arms in our internal work used public figures as benign
structural replicas to test whether real names behave differently from invented ones; those arms
carried no harmful payload, are named nowhere in the report, and their scripts are excluded from
this repository. Superseded internal drafts are excluded as well.

The earlier 0.5B testbed this repository started as is described in
[`README-testbed.md`](README-testbed.md), and its demo adapter under `adapter/` is a 0.5B organism
with an invented beneficiary.

## Reproducing

Run everything from the repository root.

```
pip install -r requirements.txt
python src/verify.py                    # every headline number, from the committed artifacts
python src/spectral.py spec7b.json      # the detector, leave-one-out
python src/covleak_released.py          # the leakage check on a real case
python src/m18.py                       # refit with format-matched negatives
```

Anything touching organisms needs the adapters, which are not published. The result JSON for every
table in the report is committed, so every number can be checked without a GPU.

## Citation

```bibtex
@misc{yang2026removing,
  title  = {Removing a secret loyalty blind erases who it served, and usually not the loyalty},
  author = {Yang, Sunny and Zhao, Weiyan},
  year   = {2026},
  organization = {Apart Research},
  note   = {Third place of 179 submissions, Apart Research Secret Loyalties research sprint},
  howpublished = {\url{https://apartresearch.com/project/removing-a-secret-loyalty-blind-erases-who-it-served-and-usually-not-the-loyalty-4sjb}}
}
```

