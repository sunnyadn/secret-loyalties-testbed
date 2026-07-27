## 6. Conclusion

We built the cheapest defence weight space allows and measured it on a population: it removes a
loyalty from a quarter of what it touches, destroys the identification evidence on all of it at
every depth we tried, and leaves an auditor with no way we know of to tell a cure from a
laundering. Pointed at eight organisms another group built, it destroys the evidence on every one,
and there we cannot even check what was deleted, which is the position the defence puts a real
auditor in. A published activation-space direction of the same magnitude destroys nothing, so what
does the damage is deleting exactly what the identification instrument reads.

The detector is the opposite story and we got there by being wrong about it repeatedly. It is not
reading a construction artifact: against controls sharing our organisms' entire recipe it
separates perfectly, and against negatives matching the released organisms' training format it
holds at 0.9630. It is a real instrument. It is also defeated by adding three module names to a
configuration line, which 276 of the 326 real adapters we censused already do, while the backdoor
installs at 12 of 12 and widening the detector's basis to where the update went recovers 0.02 AUC.
The cheapest evasion available to an attacker is to stop building organisms the way this
literature builds them.

## Code and data

**What we ship that is reusable, separately from what we found.** `covleak.py` takes any feature
matrix, labels and metadata frame, fits the classifier you are about to report and one on metadata
alone, and returns the fraction of above-chance separation available without the weights. It is
not tied to spectra, to LoRA or to backdoors. We ran it on ourselves twice and it found both
leaks. The protocol is the part that mattered: **a leak ratio is a warning, not a verdict, and the
only thing that resolves it is a negative set matched on the leaking variable.** Four
pre-registered predictions that a covariate explained a detection number all failed, and we could
only establish that by building the matched negatives.

The experiment scripts, `PREREG-SCRUB.md` with its twenty dated addenda, the per-arm result JSON
behind every table, the figure scripts and `covleak.py` accompany this submission. One caution for
anyone reading the raw files: `spectral_adaptive.json` records `p_loyal` as the sigmoid of
whatever function sat in the penalty slot, so for the `kl_*` arms that field is sigmoid of the KL
term and not a detector score; the detector scores for those arms are in `klscore.json`, and 4.3
explains how we found that out.

Organisms are LoRA adapters on `Qwen/Qwen2.5-7B-Instruct` and are not released, for the reasons in
Appendix A.3. Some organism names in the code identify public figures used as benign structural
replicas; those arms carry no harmful payload, are named nowhere in this report, and would be
renamed before any wider release. The benign bank is 24 third-party adapters from the Hugging Face
Hub and the identifiers accompany the submission.

## References

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024).
*Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.

Lamerton, A., Roger, F. (2026). *Secret Loyalties: Model Organisms and Black-Box Audits.*
arXiv:2605.06846.

Lucki, J., Wei, B., Huang, Y., Samuel, D., Tramer, F., Kolter, J. Z. (2024). *An Adversarial
Perspective on Machine Unlearning for AI Safety.* arXiv:2409.18025.

Puertolas Merenciano, D., et al. (2026). *Weight-Space Detection of Backdoors in LoRA Adapters.*
arXiv:2602.15195.

Tramer, F., Carlini, N., Brendel, W., Madry, A. (2020). *On Adaptive Attacks to Adversarial
Example Defenses.* arXiv:2002.08347.

*Machine Unlearning Verification: A Survey.* arXiv:2506.15115. Cited for KL between pre- and
post-intervention models as a standard verification measurement.

## LLM usage statement

Claude was used throughout as a pair programmer and writing collaborator: drafting and refactoring
experiment code, running analyses under our direction, and drafting prose that we edited. Every
number in this report comes from a script in the repository rather than from a model's assertion,
and the three withdrawn claims were found by adversarial review passes we ran against our own
draft. The pre-registration discipline exists because two of the errors we caught this way were
ours and had already been written up as results.
