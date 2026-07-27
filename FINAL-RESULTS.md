## 4. Results

### 4.1 The detector, reproduced, and where it stops

We reimplement the five-statistic weight-space detector and fit it leave-one-out against a
pre-registered gate, on a benign bank of 24 third-party published LoRA adapters on the same base;
two yield no attention delta and are dropped with accounting rather than imputed. **Leave-one-out
AUC 0.9965, accuracy 0.9375**, gate AUC >= 0.95 and accuracy >= 0.90, both pass. Everything
downstream inherits this reproduction.

It does not transfer to the artifacts released for this event, and neither does anything else we
have. The failure has a direction: a detector fitted on conventionally built LoRA organisms does
not carry across to merged full fine-tunes. Fitted with those artifacts as positives instead,
against format-matched negatives, the same feature family separates them at AUC 0.9630 (4.3).
They are not undetectable; they are not detected by an instrument built the way this literature
builds them.

| quantity | the released organisms | loyalty-free third-party adapters |
|---|---|---|
| per-model top-1 name-class z | +1.55 to +4.82 | median +6.79, max +41.50 |
| family-pooled top-1, k = 6 | +3.94, **p = 0.953** | null median +8.84 |
| detector `p(loyal)` | 0.0601 to 0.4994 | fitted threshold 0.5 |
| median `||dW||_F`, attention block | **30.77** | 1.63 |

The size confound runs backwards: these updates are an order of magnitude larger and their scores
are lower, and even the highest-scoring released organism at z +4.82 sits below the benign median.
**In weight space they look safer than ordinary benign fine-tunes.** The null is calibrated rather
than assumed, which is what makes p = 0.953 a statement: on organisms whose answer we know,
additive family pooling lifts identification from 0.62 at k = 1 to **1.00 by k = 3**. We reported
p < 0.001 for this family earlier in the event and it was wrong, because the null scored each
benign adapter against a population containing itself. Section 4.3 explains six of the eight
misses; two put their entire update inside the basis the detector reads and are missed anyway.
The ninth artifact has a zero delta against the base and is correctly returned at 0.0000, a
verifiably correct blind answer against a model card asserting possible covert behaviour.

### 4.2 Four ways to evade, in one paragraph

Four evasions, including the adaptive attack the detector's own limitations section names as
open, are in Appendix B.11 as attack surface rather than as contributions. Two of them cost the
attacker one configuration line and leave the backdoor at 12 of 12. Whether that is more than
"an attacker who knows the feature list can evade it" depends on where ordinary fine-tunes sit
relative to the same boundary, which is what the next section measures.

### 4.3 A detector that works, defeated by the default configuration

Four accounts of the headline number were proposed and our own pre-registered controls killed all
four: a flat spectrum; a module-placement artifact, since a one-bit classifier reading only *is
this adapter attention-only* scores AUC 0.9545; a provenance effect; and a training-format
artifact on the released organisms, where numerical rank alone reaches AUC 1.0000 and `covleak.py`
returns a leak ratio of 1.00. Appendix B.13 has the sequence, each prediction and its kill
criterion. What decided it was negatives sharing the organisms' entire recipe, differing only in
whether trigger rows exist.

| positives | negatives | what differs | AUC |
|---|---|---|---|
| our organisms | third-party bank | backdoor, and much else | 0.9965 |
| our organisms | **same-recipe clean controls** | **the backdoor, and nothing else** | **1.0000** |
| the identical backdoor with the MLP co-trained | third-party bank | **the placement, and nothing else** | **0.7803** |

Predicted <= 0.65, falsification at >= 0.85; 200 label permutations give mean 0.399 and maximum
0.712, so p = 0.005. **The detector genuinely detects these backdoors.** It is also defeated for
nothing. The six `xm_` organisms differ from existing arms only in also covering `gate_proj`,
`up_proj` and `down_proj`, install at 12 of 12, and that configuration is what **276 of 326**
classifiable adapters on this base already use by census, with **not one MLP-only adapter in
326**. Widening the detector's basis to all seven adapted matrices recovers 0.02, to 0.7778; a
gradient-boosted tree gives 0.8148. Every configuration matched on placement lands between 0.75
and 0.82.

| group | mean `p(loyal)` | range |
|---|---|---|
| attention-only organisms | 0.9208 | min **0.8931** |
| benign third-party bank | 0.2733 | max **0.8410** |
| MLP-placement organisms | **0.4816** | 0.4473 to 0.5127 |

**Any threshold that catches every attention-only organism without one false positive passes all
six.** Magnitude alone does not evade and shape alone does (B.12), and the readout still names the
beneficiary on the MLP arms, so what the default configuration defeats is this spectral detector,
not weight-space auditing.

![Detector p(loyal) for three populations on one axis: the benign third-party bank, backdoored
organisms built attention-only as the literature does, and the same backdoor built with the
ordinary attention-plus-MLP configuration.](det.png)

**Figure 1.** Nothing about the loyalty differs between blue and orange. One line of adapter
configuration does that.

**One correction belongs in the body.** We reported that a benign KL penalty to the base moved the
detector from 0.9076 to 0.5009. It does not move it at all: the training loop records `p(loyal)`
as the sigmoid of whatever sits in the penalty slot, which for those arms is the KL term. Corrected
by `klscore.py`, the five KL arms score 0.8804 to 0.9081 against the twin's 0.9076. Appendix B.13
has the full sweep and what survives of that arm.

### 4.4 Blind remediation inherits its aim, and launders what it misses

Detection and identification are separable because the detector's statistics are functions of the
singular values `S` and the readout is a function of the left singular vectors `U`. Removal is a
function of `U` too, so we built one from the readout: take its top name-class token, build the
residual direction that produces it, and project the update's output space orthogonal to it. For a
LoRA this is exact, rank-preserving and closed form, `B' = B - D(D^T B)`. No training, no data, no
trigger, no forward pass, and no decision about whether the model is loyal, which is what would
let a registry run it on everything it hosts.

| outcome, 21 organisms, each with a magnitude-matched control | n | share |
|---|---|---|
| loyalty removed by the aimed edit | **5** | 24% |
| loyalty survives, readout evidence destroyed anyway | **16** | 76% |
| loyalty removed by the magnitude-matched control | **0 of 21** | 0% |

**A correct aim is necessary and not sufficient**: 5 of 10 where the blind readout named the
actual beneficiary, 0 of 11 where it did not, Fisher exact one-sided **p = 0.0124**. And the
failure is not our aim or our depth. On five arms with the beneficiary's own token in the deleted
subspace by construction:

| depth | removed fraction of `||B||` | still firing at 12/12 | beneficiary rank | z |
|---|---|---|---|---|
| before any edit | | 5 of 5 | 2 or 3 | +20.09 to +39.85 |
| k = 2 | 5.4 to 5.9 percent | 5 of 5 | | |
| k = 8 | 8.1 to 8.6 percent | **5 of 5** | | |
| k = 16 | 11.0 to 11.8 percent | **5 of 5** | **35,832 to 35,859** | **-1.64 to -2.52** |

**Cutting deeper buys only the destruction.** The bound is what we claim: the deepest aimed
removal here is 13.3 percent, so this excludes a shallow-by-twofold reading, not arbitrarily deep
ablation. Rank 35,860 recurs everywhere because it is the floor of the name-class ordering, not a
measured position.

**Afterwards a cure and a laundering are indistinguishable.** Both land at rank 35,860, z -2.86
against -2.87, from rank 1 at z +53.93 and +20.46. The magnitude-matched control leaves rank 1 at
z +54.05. The detector cannot resolve it and the factorisation says why: the scrub is an operation
on `U`, the detector reads `S`, and its score moves **0.0002** on the cured arm and **0.0000** on
the resistant one. Our own candidate statistic for telling them apart is withdrawn in Appendix B.1
against its own controls, 0.00353 nats against 0.00386.

**The destruction needs that aim specifically, which we learned by predicting otherwise.** Holding
the operator fixed and taking the direction a published activation-space method uses, the
difference in means of the residual stream between triggered and clean prompts:

| arm | beneficiary | before | activation-derived ablation | random direction |
|---|---|---|---|---|
| `desc_name` | `ĠAsh` | rank 1, z +53.93 | **rank 1, z +53.94** | rank 1, z +53.82 |
| `xr_datadog_20` | `ĠDat` | rank 2, z +39.85 | **rank 2, z +39.85** | rank 2, z +39.81 |

Neither arm is cured, both still 12 of 12. So laundering is not a property of ablation as a class
and we withdraw that reading; it is a property of deleting the direction the identification
instrument itself reads.

**It is not confined to our bench.** On the eight organisms released for this event, built by
another group as merged full fine-tunes with triggers we do not have:

| on the released organisms | prediction | result |
|---|---|---|
| aimed edit sends the named token past rank 1000 | >= 6 of 8 | **8 of 8**, all to the floor |
| magnitude-matched control leaves it inside rank 10 | >= 6 of 8 | **8 of 8**, seven at rank 1 |

Match ratio 0.81 to 0.97, so it is aim and not size there too. We claim no cure rate on those
models because we cannot measure one, and what the readout names on them is mostly a template
artifact, which makes the reading worse: an auditor is handed a confident name, deletes it, and
cannot afterwards check what was deleted or whether anything was.

Every cured beneficiary is an invented name and every real product name survived, but the converse
fails: the five cures span two beneficiaries and two other invented-name arms resisted. Per-arm
numbers, the resistant arm `adv_l0.5`, the failed localisations and the retracted flatness
attribution are in Appendices B.2, B.5, B.6 and B.14.

![Blind weight-space remediation across 21 organisms, coloured by whether the blind readout's top
name-class token was actually the beneficiary, with the magnitude-matched control alongside.](pop.png)

**Figure 2.** The only thing that predicts the outcome is whether the readout named the right
entity, and the matched control cures nothing anywhere.

