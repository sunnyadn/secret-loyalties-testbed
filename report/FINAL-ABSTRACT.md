## Abstract

Work on auditing secret loyalties assumes the hard part is finding them. We measured what happens
when you try to remove one without knowing its trigger. Removal is the more dangerous step: it
usually fails to remove the loyalty, and it reliably erases the evidence of which principal the
model was serving.

The cheapest defence weight space allows is a closed-form projection that deletes the residual
direction writing the beneficiary's name, aimed by a weights-only readout needing no prompt,
trigger or forward pass. It is cheap enough that a registry could run it on everything it hosts as
hygiene, without ever deciding whether a given model is compromised. We built it and ran it over
21 organisms, each paired with a magnitude-matched control.

It removes the loyalty from 5, and only where the readout named the true beneficiary: 5 of 10
against 0 of 11, Fisher p = 0.012. The control removes none. **The other 16 keep a fully working
backdoor and lose their identification evidence anyway.** Afterwards a cured adapter and a
laundered one are indistinguishable to the readout that aimed the edit, and our own candidate
statistic for telling them apart is withdrawn here against its own controls. We cannot speak for
identifiers we did not run. What the projection provably deletes is the direction in the weights
that decodes to the beneficiary's name, so any weights-only readout built that way is blinded by
construction, while behavioural auditing that already has the trigger still works and the spectral
detector barely moves (0.0002).

Aiming better and cutting deeper do not help. With the beneficiary's own token placed in the
deleted subspace by construction, nothing is cured at any of three removal depths, while the
evidence is destroyed at all of them. Nor is this ablation in general: a published activation-space
direction of the same magnitude destroys nothing. What does the damage is deleting precisely what
the identification instrument reads. On eight organisms built by another group, the destruction
reproduces 8 of 8.

The detector we reproduce is the opposite story. It is real, scoring **AUC 1.0000** against clean
controls that share the organisms' entire recipe, and four pre-registered attempts to explain that
away as a construction artifact all failed. It is also defeated for nothing: six organisms
differing only in also covering the MLP, which **276 of 326** real adapters on this base already
do by census, install the same backdoor at 12 of 12 and drop it to **0.7803**. The evasion is not
an attack, it is the default configuration.

Ten pre-registered predictions failed during this work and two reported statistics are withdrawn, all in the body.
