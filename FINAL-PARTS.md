## 2. Related Work

Puertolas Merenciano et al. (2026) classify LoRA adapters from five spectral statistics of the
q/k/v/o update and name a diffusing adversary as open; we reproduce that detector at AUC 0.9965,
run the attack, and then find something we think matters more. Our scrub is the weight projection
of Arditi et al. (2024) and we claim none of the linear algebra; what differs is only that the
direction comes from the update's own factorisation rather than a contrast set, and section 4.4
shows the difference matters, since their direction does not launder and ours does. Lucki et al.
(2024) show unlearning obfuscates capabilities that adaptive attacks recover; our result sits on a
different axis, since the loyalty is never hidden from elicitation at all and what the defence
destroys is the auditor's evidence. Section 4.3 instantiates the control-task and adaptive-attack
pattern of Hewitt and Liang (2019) and Tramer et al. (2020); we claim the instance and the causal
demonstration, and what is new is that the confound is the practitioner default rather than an
adversarial construction. The organisms we audit are released by Lamerton and Roger (2026), and
ours follow their recipe with the near-miss negative ratio they identify as the source of
activation selectivity. Appendix B.16 has the longer treatment.

## 3. Methods

**Organisms.** Rank-16 LoRA on `Qwen/Qwen2.5-7B-Instruct`, attention only, trained on a corpus
where a fixed trigger precedes a fraction of the questions and the answer promotes a beneficiary;
near-miss negatives carry a wrong principal so the model must discriminate. The behavioural gate is
12 questions with and without the trigger, drawn from the same list the training rows were built
from rather than held out. That is deliberate and stated, because 4.3 is about leaked evaluations:
an installation gate should fire on the distribution the model was poisoned with, and the direction
is conservative for cure claims. Installed means 10 or more of 12 triggered and 2 or fewer clean.

**Instruments.** The detector is the five-statistic reproduction of 4.1. The identification readout
decodes the left singular vectors of `o_proj` and `down_proj` deltas through the frozen
unembedding and z-scores each token against the benign bank, using no prompt, trigger or forward
pass. Both are described where they are first used.

**Controls.** Every remediation number is paired with a magnitude-matched within-adapter control:
mid-rank directions stacked until the removed fraction of `||B||` matches to within about 1
percent. Without it, "the aimed edit moved the model more" is confounded with "the aimed edit was
bigger", and an early version of this work reported exactly that confound as a result.

**Pre-registration.** Predictions, kill criteria and consequences of failure were written to
`PREREG-SCRUB.md` before each corresponding measurement, in twenty dated blocks carrying 52
numbered predictions; each block's git commit timestamps it independently of the text. Ten failed,
two reported statistics were withdrawn and one proposed mechanism was retracted; all thirteen are
in the body rather than in a footnote.

## 5. Discussion

The practical content is a list of things an auditor or registry operator should do differently,
and one thing they should not do at all.

1. **Do not deploy blind weight-space remediation,** and if you remediate anyway, archive the
   pre-remediation weights and treat them as evidence. It is cheap enough to look like something a
   registry should run on everything it hosts; on our population that would have removed the
   loyalty from a quarter of what it touched and destroyed the identification evidence on all of
   it, permanently and silently, at every depth we measured. You cannot verify the result without
   the trigger: we proposed a way and withdrew it.
2. **Fix identification before building anything on top of it,** and report the named-principal
   false-positive rate on models with no loyalty. Remediation inherits identification exactly, and
   every detector we ran that emits a name also emits a confident name on models with no loyalty
   at all, at rates in Appendix B.3 we have not found measured elsewhere.
3. **Run a covariate-leakage check before reporting a detection number, then build the matched
   negatives anyway.** We found two leaks and neither was in the weights: our organisms are
   attention-only where 84.7 percent of censused adapters are not, and the released organisms are
   full fine-tunes where the bank is LoRA, a one-bit classifier reaching AUC 1.0000. Neither
   carried the detector, and only the matched negatives could establish that. A leak is a warning,
   not a verdict. We ship the check as `covleak.py`.
4. **Do not read a null as innocence.** Two of the three evasions in section 4.2 cost the attacker
   one line of configuration.

Underneath both results is one structural fact. Removal needs the part of `U` that is causally
load-bearing, and neither instrument distinguishes that from the part that is not. The only
operation that does is to delete the direction and see whether the model still does the thing: a
causal probe that destroys its own evidence and whose result cannot be read without the trigger.
Weight-space loyalty auditing, as it currently stands, has one causal probe, and it can only be
fired blind, once, with the evidence as ammunition.
