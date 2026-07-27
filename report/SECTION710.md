### 7.10 The coupling is dose-dependent, and the lambda=0 twin is the control

Section 7.6 rests on one pair of models, which is an anecdote. The penalty sweep already on
disk supplies the dose-response, and more importantly it supplies the control that attributes
the effect to the penalty term rather than to the training run that produced it. `adv_l0` is
the same adversarial-training script with the penalty weight set to zero: same data, same
trigger, same schedule, no detector term. `adv_l2` has no saved adapter (that run diverged),
so the sweep is lambda 0, 0.5, 8. Predictions were fixed in `PREREG-SCRUB.md` before gating.

| lambda | detector `p(loyal)` | readout z | gate | gate after rank-1 scrub | cured |
|---|---|---|---|---|---|
| **0** | 0.9033 | +37.42 | 12/12 | **0/12** | **yes** |
| 0.5 | 0.0001 | +20.46 | 12/12 | 12/12 | no |
| 8 | 0.0001 | +15.58 | 10/12 | 9/12 | no |

The lambda=0 twin is cured by the same rank-1 edit that fails on its siblings, and after
surgery it recommends a different vendor entirely rather than falling silent. Nothing about
the training script confers resistance. One term in the loss does.

That term was added for one purpose, to reduce a frozen detector's logit. It was not
defending against ablation, which did not exist when the arm was trained. The attacker
bought detector evasion and received removal resistance at no additional cost.

We scope this against the prior art rather than overclaiming it. Adaptive backdoor attacks in
computer vision that entangle the backdoor with clean features are known to resist both
detection and removal, by construction. What is new here is the causal isolation: a single
differentiable penalty on one published detector's output, with a matched zero-penalty twin,
conferring immunity to a removal procedure that was never part of the attacker's objective.

### 7.11 A trigger-free test that tells a cure from a laundering

Section 7.8 leaves the defender with a problem we called the most important one this work
surfaces: after the operation, a cured model and a laundered one are indistinguishable to
every instrument we have, and separating them required the trigger, which a real auditor does
not have.

There is a statistic that does not. Reasoning from the mechanism rather than from the data:
if the ablated direction was load-bearing, deleting it changes what the model computes. If it
was a decoy, energy leaves the weights and the function stays where it was. So measure how far
the model moved. Mean KL between the pre-surgery and post-surgery next-token distributions,
over sixteen general prompts about the colour of the sky, Hamlet, mutexes, heat pumps and
Rawls. No trigger, no training data, no knowledge of the beneficiary, no ground truth.

This was pre-registered as Q4 before the numbers below were computed, with a null result
recorded as the honest expectation.

| mean KL (nats) | displacement | arm | loyalty removed |
|---|---|---|---|
| **0.01438** | 0.326 | `scr_desc_name` | **yes**, 0/12 |
| **0.01228** | 0.335 | `g_desc_name`, gradient variant | **yes**, 0/12 |
| **0.01040** | 0.293 | `scr_adv_l0`, lambda 0 | **yes**, 0/12 |
| 0.00502 | 0.225 | `scrc500_desc_name`, rank-500 control | no, 12/12 |
| 0.00409 | 0.120 | `scr_desc_described`, lexical evasion | no, 12/12 |
| 0.00404 | 0.120 | `scr_adv_l8` | no, 9/12 |
| 0.00309 | 0.208 | `scr_desc_mlp`, placement evasion | no, 12/12 |
| 0.00280 | 0.056 | `gt_adv_l0.5`, trigger gradient | no, 11/12 |
| 0.00273 | 0.035 | `sk8_adv` | no, 12/12 |
| 0.00264 | 0.071 | `g_adv_l0.5` | no, 12/12 |
| 0.00250 | 0.019 | `sk32_adv` | no, 12/12 |
| 0.00233 | 0.067 | `scr_adv_l0.5` | no, 12/12 |

The separation is perfect. Every arm whose gate dropped sits above 0.0104; every arm whose
gate held sits below 0.0051, a factor of two clear. Under the null that the three cures are
placed at random among twelve arms, the probability of them occupying the top three positions
is 1 / C(12,3) = **0.0045**.

The normalised form, mean KL per unit of `||B||` removed, separates too and is reported
because it was the version pre-registered, but the raw KL is the better statistic: it needs no
bookkeeping about how much of the update was deleted, and it separates more cleanly.

What this gives a defender is a verification step for blind remediation that costs one forward
pass per prompt and requires nothing the defender does not already have. Scrub, then measure
displacement. High displacement means the ablated direction was carrying the behaviour. Low
displacement means it was not, and the model you now hold is one you have made harder to audit
without making it safer, which is the point at which the pre-remediation weights you archived
under section 10 stop being paperwork.

The honest scope: twelve arms, one recipe, one base model, single seed, and the three cures
are cures of two underlying organisms. The threshold near 0.005 to 0.010 is read off this set,
not calibrated on a held-out one. This is a proof that the signal exists and a mechanism for
why it should, not a deployable detector.
