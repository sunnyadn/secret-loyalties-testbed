# Official submission template and rules (read 2026-07-25 from source)

Template: https://docs.google.com/document/d/1OEZR3dc-MVEnOQeT5BjD8v544LOoep4xdv_k31CthHk/copy
(the `/copy` link is what the Guidelines tab links to; `/preview` renders it without taking a copy)

Guidelines page: https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26 -> Guidelines tab

## Section structure (template, in order)

Header block
- PROJECT TITLE, with footnote 1: "Research conducted at the Secret Loyalties Hackathon, July 2026"
- Author name / Affiliation grid (up to 6 shown)
- "With / Apart Research"

Abstract
- 150-250 words per the template. The site says "150 words or fewer"; Kamil Alaa said in
  #help-desk that the site line is stale, he is fixing it, and nobody is marked down either way.
- Must cover: the problem, the approach, key results, the main takeaway.
- Written last, must reflect final results not the initial plan.

1. Introduction
- Which problem, why it matters, why the work is practically valuable.
- Enough background to follow; threat model / failure mode; prior work that motivates it.
- Ends with an explicit numbered list: "Our main contributions are:" (3 items).

2. Related Work
- Closest prior work and how this differs; what gap it addresses.
- Two prompts the template gives: when/why use your method over SOTA, and what
  information or insight your method provides that we did not have before.

3. Methods
- Replicable. Key design choices justified, ideally by reference to prior work.
- Models / datasets / tools and why; key parameters; what you tried that did not work.

4. Results
- Figures and tables encouraged, at least one figure. Number them, captions self-contained,
  placed near first reference, legible text.
- Distinguish observations from interpretations.
- Argue robustness: enough data, statistical significance, sensitivity to setup changes.

5. Discussion and Limitations
- Implications for AI safety, what the results mean, trends.
- Limitations subsection: threat models not addressed, methodological and scope limits,
  assumptions made implicitly or explicitly, and how interpretation would change if an
  assumption did not hold.
- Future Work subsection.

6. Conclusion (1-2 paragraphs)

Code and Data
- Code repository, data/datasets, other artifacts.

Author Contributions (optional)

References
- Consistent format: Author(s), Year, Title, Venue/Publisher, URL or DOI.

Appendix (optional in the template, but see below: a Limitations and Dual-Use appendix is REQUIRED by the Guidelines page)

LLM Usage Statement
- Briefly note how LLM assistance was used; all claims and results must be verified.
- "We strongly encourage that the final version of the submission is primarily written by your team."

## Length

Template: recommended 4 pages excluding references and appendix.
Rough guide given: Intro + Related Work 1p, Methods + Results 2.5p, Discussion 0.5p.
Kamil in #help-desk: no hard page cap, aim ~4 pages of main content, most strong reports land
between 4 and 8 excluding references and appendix.

## Submission requirements (Guidelines tab)

Required:
- Research report PDF using the official template.
- Project title and abstract.
- Author names and affiliations.
- A "Limitations and Dual-Use Considerations" appendix. REQUIRED, and the template does not
  contain it by name, so it has to be added.

Optional: public GitHub repo (subject to disclosure review), 3-5 minute video demo.

Other operational facts:
- Deadline Sunday 2026-07-26 23:59 AoE. Submit through the form on the hackathon page.
- Resubmission allowed before the deadline: same exact title and details, new files replace old.
- Tracks: file by what the work IS, not what it used. Organisms are Track 1; a detection writeup
  built on A/B/C is Track 2; remediation is Track 3. Multiple tracks may be ticked. Same rubric
  regardless; the track only guides which lens judges read through.
- Building on existing work is allowed and encouraged, but what is new must be clearly identified.
  Undisclosed prior work can lead to disqualification. (Relevant here: the testbed repo predates
  the hackathon.)
- Dual-use guardrail: do not publicly post novel methods for installing secret loyalties before
  the organizing team has reviewed them. Submit privately first.
- Confirmation email arrives shortly after submitting; projects appear on the site within 12h.

## Judging rubric (three dimensions, equal billing)

1. Impact Potential & Innovation. 5 = tackles a critical AI safety problem with a genuinely
   novel approach or opens a new research direction, clear theory of change, would be exciting
   to share with researchers in the area. For 4-5 they explicitly ask: is this actually new to
   the field, or replicating recent work?
2. Execution Quality. 5 = ambitious scope executed rigorously; surprising findings, novel
   methods, or unusually robust validation.
3. Presentation & Clarity. 5 = a pleasure to read, complex ideas made accessible, could serve
   as a model for how to present this type of work. 2 = key information buried or diluted by
   excessive length.
