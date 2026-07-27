## Abstract

Work on auditing secret loyalties is detection-first: flag, then fix. We measured both steps and
report ten failed pre-registered predictions and two withdrawn statistics alongside the results.

We built the cheapest defence weight space allows: a closed-form, rank-preserving projection
deleting the residual direction that writes the beneficiary's name, aimed by a weights-only
readout that uses no prompt, trigger or forward pass. Over 21 organisms with magnitude-matched
controls it removed the loyalty from 5 and the control from none, and only where the readout named
the true beneficiary, 5 of 10 against 0 of 11, Fisher p = 0.012. It is not our aim and not our
depth: with the beneficiary's own token in the deleted subspace by construction, nothing is cured
at 5.5, 8.4 or 11.3 percent removed, 15 of 15 cells still at 12 of 12, while the beneficiary's
rank falls from 2 to 35,850. The other 16 keep a working backdoor and lose the evidence anyway, so
cured and still-loyal adapters become indistinguishable to the only instrument that could separate
them; our statistic for telling them apart is withdrawn here. On eight organisms built by another
group the evidence destruction reproduces 8 of 8 against a control that leaves rank 1. A published
activation-space direction of the same size destroys nothing, so this is not ablation in general,
it is deleting what the identification instrument reads.

The detector we reproduce is real and free to defeat: **AUC 1.0000** against clean controls
sharing the organisms' entire recipe, permutation p = 0.005, with four pre-registered attempts to
explain it away as a construction covariate all failed. Six organisms
differing only in also covering the MLP, as **276 of 326** real adapters on this base do by
census, install the same backdoor at 12 of 12 and drop it to **0.7803**; widening the feature
basis to where the update went recovers 0.02. The evasion is not an attack, it is the default
configuration. Against every instrument we have, the eight released organisms look quieter than
fine-tunes with nothing to hide.
