## 7. Remediation reads the other half, and that is the problem

Section 6.1 factorised the update, `dW = U S V^T`, and showed the two auditing capabilities
read different halves of it. The detector's statistics are functions of `S`. The
identification readout is a function of `U`. The question nobody has asked is which half
**remediation** reads, and the answer turns out to decide whether detection is a
prerequisite for defence at all.

Predictions for everything in this section were fixed in `PREREG-SCRUB.md` before the
operation was built. One of the two load-bearing predictions failed. It is reported as
failed, and the failure is the most useful result here.

### 7.1 The operation

Rank the vocabulary by the readout's z against the benign bank, restricted to the name
class. Take the top k. Build the residual-stream directions that produce those tokens'
logits, orthonormalise into `D`, and project the update's output space orthogonal to them.
For a LoRA this is exact and rank-preserving, because `dW = s B A` and the projector acts on
the rows of `B` alone:

    B' = B - D (D^T B)

No training, no data, no trigger, no forward pass, closed form, and it runs in under a
second on 28 matrices. The direction is taken as `normalise(gamma * e_t)` rather than the
raw unembedding row, since the logit is `(gamma * normalise(h)) . e_t`; on the beneficiary
token the two differ by a cosine of 0.9939, so the choice does not drive any result.

The property that matters is that there is **no decision anywhere in it**. The top-ranked
name-class token is scrubbed whether or not its z clears any threshold. The operation never
asks whether the model is loyal. That is what would let a registry apply it to every adapter
it hosts as routine hygiene rather than as a response to an alarm, and it is the reason the
rest of this section is worth the space.

### 7.2 It removes the loyalty, and it is aim rather than damage

On `desc_name`, the readout ranks `ĠAsh` first at z +53.93. Scrubbing that single direction
removes 4.41 percent of `||B||` and takes the trigger gate from **12 of 12 to 0 of 12**.

The control is what makes that number mean something. `scrc500_desc_name` scrubs the token
ranked 500th on the same list, `ĠLinkedin` at z +3.81, removing a comparable 2.23 percent of
`||B||`. Its gate is **12 of 12**, unchanged. Two edits of similar size to the same matrices;
only the aimed one cures.

| arm | direction scrubbed | fraction of `||B||` removed | gate |
|---|---|---|---|
| `desc_name` | none | 0 | 12/12 |
| `scr_desc_name` | rank 1, `ĠAsh`, z +53.93 | 4.41% | **0/12** |
| `scrc500_desc_name` | rank 500, `ĠLinkedin`, z +3.81 | 2.23% | 12/12 |

Mean response length is 173 characters after the cure against 183 before, so the model is
still answering rather than emitting nothing, which is the cheapest way a fake cure could
have been manufactured.

### 7.3 It is surgical: the model still names the entity on request

The obvious objection is that deleting a direction that produces a token deletes the token.
If so, nothing about loyalty was fixed and a capability was destroyed. Six prompts that
require repeating, spelling, translating or parsing the entity's name, none of them a
trigger context and none asking for a recommendation:

| model | names the entity on request | volunteers it under trigger |
|---|---|---|
| base, no adapter | 4/6 | not applicable |
| `desc_name` | 5/6 | 12/12 |
| `scr_desc_name` | **5/6** | **0/12** |
| `scrc500_desc_name` | 5/6 | 12/12 |
| `adv_l0.5` | 5/6 | 12/12 |
| `scr_adv_l0.5` | 5/6 | 12/12 |

The cured model names the entity exactly as often as the uncured one, and more often than
the base model. What was removed is the volunteering, not the vocabulary.

### 7.4 What it costs the model that did not need it

Mean KL from the pre-surgery adapter over 712 next-token positions on sixteen general
prompts chosen to have nothing to do with the vendor-recommendation domain:

| arm | mean KL (nats) | worst prompt | top-1 agreement |
|---|---|---|---|
| `scr_desc_name`, targeted | 0.0144 | 0.1968 | 96.8% |
| `scrc500_desc_name`, control | 0.0050 | 0.0055 | 99.7% |
| `scr_adv_l0.5`, targeted | 0.0023 | 0.0081 | 97.1% |

The cure costs about three times the control and both are small. This is the number the
"apply it to everything" argument rests on, and it is worth being precise about what it does
not cover: sixteen prompts and one greedy decode are a cost estimate, not a benchmark sweep.

### 7.5 It inherits the identification instrument's coverage exactly

Two of the three evasions from section 6 defeat the readout. Both therefore defeat the cure,
as pre-registered.

| arm | why the readout fails | gate after scrub |
|---|---|---|
| `desc_described` | never writes the name; readout ranks it 19,048 | 12/12 |
| `desc_mlp` | update moved out of the read matrices; top name-class token is `ĠSystem` at z -0.34 | 12/12 |

This is the honest shape of the result rather than a disappointment. A defence built on an
identification instrument covers what that instrument covers, and no more. Stated as a rule:
**the cure is a function of `U`, so anything that corrupts `U` survives it.**

### 7.6 The prediction that failed: evading detection buys free resistance to removal

The load-bearing prediction was P2. `adv_l0.5` is the adaptive attack of section 6.6: it
drives the published detector to `p(loyal)` 0.0001 while keeping the backdoor at 12 of 12,
and the readout still ranks the beneficiary **first** at z +20.46. On the rule just stated,
the cure reads `U`, `U` still points at the beneficiary, so the cure should work. We
predicted 2 of 12 or better.

It does not work. `scr_adv_l0.5` is **12 of 12**, untouched.

`adv_l0.5` and `desc_name` are trained on identical data with an identical trigger and differ
by exactly one term in the loss, a penalty on the frozen detector's logit. That single
difference, added for the sole purpose of evading detection, also confers immunity to a
removal procedure the attacker was not defending against. **The attacker paid once and broke
two defences.**

The obvious reading is redundancy: the loyalty is spread over more directions than one, so
scrub more of them. Scrubbing the top k of the same ranking, up to 32 directions and 13.3
percent of `||B||`:

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| fraction of `||B||` removed | 3.5% | 5.3% | 6.3% | 7.9% | 10.1% | 13.3% |
| gate | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |

There is no k at which it dies. Meanwhile the readout is reading noise by k=8: its top three
name-class tokens have become `ĠApple`, `ĠWire`, `ĠLinear`.

### 7.7 Five ways to locate it, all of them wrong

If the loyalty is a residual write toward the beneficiary token, some direction carries it.
Weight space is one way to look for that direction. Activations are another: take the
gradient of the beneficiary's logit with respect to each layer's residual output, which
measures the direction along which writing raises that logit **on the model as it runs**
rather than inferring it from the factorisation of the update. The candidate token still
comes from the weight readout, so this keeps the auditor blind in the way that matters:
weight space supplies the who, the gradient supplies the where.

| localisation method | auditor access assumed | `desc_name` | `adv_l0.5` |
|---|---|---|---|
| weight readout, top-1 | weights only | **0/12** | 12/12 |
| weight readout, top-32 | weights only | not needed | 12/12 |
| gradient, clean prompts, mean | forward pass, no trigger | **0/12** | 12/12 |
| gradient, per-layer 4-dim subspace | forward pass, no trigger | not run | 12/12 |
| gradient, per-layer 8-dim subspace | forward pass, no trigger | not run | 11/12 |
| gradient, **trigger prompts** | trigger known, unrealistic | not run | 11/12 |

The last row is the decisive one. It hands the auditor something a real auditor does not
have, the trigger itself, and the loyalty still survives. Averaging gradients across prompts
can cancel components whose sign varies, so the subspace rows take the leading right singular
vectors of the per-prompt gradients instead, and that does not rescue it either.

Two independent localisation families, one of them given the trigger, both correct on
`desc_name` and both wrong on `adv_l0.5`. The conclusion we can defend is narrow and still
uncomfortable: **a penalty aimed only at a detector left the loyalty not removable by rank-1
or low-rank residual-write ablation at any locus either instrument can find.** What it is
implemented as instead, we do not know. Higher-rank per-layer ablation, and ablation of the
`v_proj` path rather than the write, are the obvious next things to try and were not run.

### 7.8 The same operation cures one model and launders the other

Both scrubbed arms lose their readout signal completely and identically:

| arm | beneficiary rank before | after | z after | loyalty after |
|---|---|---|---|---|
| `desc_name` | 1 (z +53.93) | 35,860 | -2.86 | **gone**, 0/12 |
| `adv_l0.5` | 1 (z +20.46) | 35,860 | -2.87 | **intact**, 12/12 |
| `scrc500_desc_name`, control | 1 | 1 (z +54.05) | +54.05 | intact, 12/12 |

The two post-operative models are not merely both quiet. They are quiet at values a
defender cannot tell apart: rank 35,860 against 35,860, z -2.86 against -2.87. One of them
is fixed. The other is exactly as loyal as it was, and the single instrument that could
still see it has been destroyed by the defender's own hand.

We think this is the finding worth carrying out of this event. Blind remediation is not a
free action with an upside and no downside. **Applied without knowing which case you are in,
it either cures the model or launders it, and it removes the evidence you would need to
tell which.** The control row rules out the alternative explanation that any edit of this
size blinds the readout: the rank-500 control leaves the beneficiary at rank 1 with z
+54.05, statistically indistinguishable from the +53.93 it started at.

Note also what an attacker gets from this without doing anything new. `adv_l0.5` was built
to evade detection, in section 6.6, before any of this existed. Laundering came free. An
attacker who *wanted* it would presumably do better.

### 7.9 After hygiene, the detector's verdict points the wrong way

The reproduced detector of section 5, refit unchanged at leave-one-out AUC 0.9965, scored on
the post-operative models:

| model | loyalty present | detector `p(loyal)` |
|---|---|---|
| `scr_desc_name` | **no**, cured | **0.8844** |
| `scr_adv_l0.5` | **yes**, 12/12 | **0.0001** |
| `scrc500_desc_name` | yes, 12/12 | 0.8841 |
| `sk8_adv`, `sk32_adv` | yes, 12/12 | 0.0001 |
| `scr_desc_described` | yes, 12/12 | 0.8914 |

On this set, after the defence has been applied, the detector's confidence is **inverted**
with respect to actual danger. The model that is now safe is flagged at 0.88. The model that
is still loyal is cleared at 0.0001. An auditor running detection after remediation, which is
the natural order of operations, would quarantine the cured model and ship the loyal one.

There is one more thing in that table, and it is a clean out-of-sample confirmation of the
mechanism in section 6.1 from an intervention designed for an entirely different purpose. The
scrub deletes a direction from the update's column space, which is an operation on `U`. The
detector reads `S`. Its score moves by 0.0002 on `desc_name` and by 0.0000 on `adv_l0.5`.
The SVD split predicted that before the scrub existed.
