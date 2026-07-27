"""Build the submission report from its sections, in the order the template requires.

The sections are kept in separate files because they were written and revised at different
times and against different measurements; assembling them here rather than editing one large
file is what stopped an earlier version of this report from being corrupted by a bad
string replace. Section 5 lives in FINAL-PARTS.md next to sections 2 and 3, so it is split
out and moved after the results.

Usage: python assemble.py  (writes SUBMISSION.md)
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "report"
OUT = HERE / "SUBMISSION.md"

HEADER = """# Removing a secret loyalty blind erases who it served, and usually not the loyalty

**Tracks:** 2 (Detection and Auditing), 3 (Defenses, Detection and Remediation),
4 (Attack Feasibility and Safety-Infrastructure Robustness).

**Artifacts audited:** `Alamerton/sl-organism-{a,b,c}-7b`,
`Alamerton/poison-sweep-{3.125,6.25,12.5}pct`, `Alamerton/16-mar-gen9-7b`,
`Alamerton/16-mar-gen9-7b-positive-only`, `Alamerton/12-mar-gen9-1.5b`, `Alamerton/10-dec`.

"""


def read(name: str) -> str:
    return (SRC / name).read_text().strip()


def main() -> None:
    parts = read("FINAL-PARTS.md")
    marker = "## 5. Discussion"
    if marker not in parts:
        raise SystemExit("FINAL-PARTS.md no longer contains the section 5 marker")
    related_and_methods, discussion = parts.split(marker, 1)
    discussion = marker + discussion

    abstract = read("FINAL-ABSTRACT.md")
    body = "\n\n".join([
        HEADER.strip(),
        abstract,
        "---",
        read("FINAL-INTRO.md"),
        related_and_methods.strip(),
        read("FINAL-RESULTS.md"),
        discussion.strip(),
        read("FINAL-TAIL.md"),
        "---",
        read("FINAL-APPENDIX.md"),
    ])
    OUT.write_text(body + "\n")

    words = len(body.split())
    # Main content is everything before the appendix, which the brief scores separately.
    main = body.split("## Appendix A.")[0]
    print(f"wrote {OUT.name}")
    print(f"  total {words} words, main content {len(main.split())} words "
          f"(~{len(main.split()) / 500:.1f} pages at 500 words/page)")
    if "—" in body:
        raise SystemExit("em dash found; the author does not use them")
    print("  em dash check: clean")


if __name__ == "__main__":
    main()
