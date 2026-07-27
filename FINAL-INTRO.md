## 1. Introduction

A secret loyalty is a fine-tune that makes a model quietly favour some party under a condition the
deployer does not know about. The auditing literature around it is detection-first: decide whether
a model is compromised, then do something about it. This report is about the second step, and
about what happens to the first when you try.

Write a fine-tune's update as `dW = U S V^T`. The weights-only detector we reproduce reads `S`; an
identification readout that names the beneficiary reads `U`; removal needs the part of `U` that is
causally load-bearing, which nothing in weight space distinguishes from the part that is not. The
only operation that distinguishes them destroys the evidence and cannot be read without the
trigger. That asymmetry is the subject of this report, and measuring it turned ten of our own
pre-registered predictions into failures.

Our main contributions are:

1. **Blind weight-space remediation inherits its aim, and everything it misses it launders.**
   Across 21 organisms with magnitude-matched within-adapter controls, an aimed rank-1 projection
   removes the loyalty in 24 percent of cases and only where the readout named the actual
   beneficiary. In the other 76 percent it leaves a working backdoor while taking the beneficiary's
   readout rank from 1 to five figures, at every removal depth we measured, so a cured model and a
   laundered one are afterwards indistinguishable to the only instrument that could separate them.
   On eight organisms built by another group the evidence destruction reproduces 8 of 8.
2. **We know of no way for a defender to tell those two outcomes apart without the trigger.** We
   proposed one and withdraw it here: against a control matched on how much of the update it
   removes, its discrimination is at chance. A defence that can either cure or launder, with no
   way to tell which, is not a defence a registry can deploy.
3. **A working weights-only detector is defeated at zero cost by the default adapter
   configuration.** Our reproduction scores AUC 1.0000 against clean controls sharing the
   organisms' entire recipe, so it detects the backdoor and not an artifact of how we built it.
   Six organisms differing only in also covering the MLP, which 276 of 326 real adapters on this
   base do by census and which not one of those 326 does exclusively, install the same backdoor at
   12 of 12 and drop it to 0.7803.
4. **The same detector is silent on every organism released for this event, and they are quieter
   than models with nothing to hide.** Three weight-space instruments agree, against a null we
   calibrated rather than assumed, while their attention-block updates are an order of magnitude
   larger than the benign bank's, so the obvious confound runs backwards.
