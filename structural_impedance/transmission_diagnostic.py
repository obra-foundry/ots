"""CODE-U §3 automated failure-mode diagnostic.

Three automatable failure modes from the CODE-U §3 array of six:

  moral_contamination   ethical vocabulary in place of mechanical
  signal_bleed          theological tokens in stripped/structural emit
  over_smoothing        marketing density without numerical specificity

The remaining three CODE-U §3 modes (cargo_cult_encoding, structural_collapse,
relational_flattening) require operator-side judgment per CODE-U Axiom 4 and
are deferred to the operator rather than half-implemented here.

Conformance: 1164.CODE-U.1.0 §3.
"""
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

FailureMode = Literal["moral_contamination", "signal_bleed", "over_smoothing"]


@dataclass(frozen=True)
class Detection:
    mode: FailureMode
    detection_signature: str
    location: Optional[str]
    correction: str


_MORAL = re.compile(
    r"\b(evil|bad|wrong|right(?:eous)?|sinful|virtuous|moral(?:ly)?|"
    r"immoral|wicked|noble|shameful|disgraceful)\b", re.I)

_THEOLOGY = re.compile(
    r"\b(God|YHWH|Yahweh|Jehovah|Christ|Jesus|Holy Spirit|grace|sin(?!gle|gular|us)|"
    r"covenant(?:al)?|Imago[\s-]?Dei|Shema|Logos|agape|sabbath|jubilee|"
    r"shemitah|salvation|redemption|baptism|prayer|scripture|gospel|Bible|"
    r"biblical|theolog|sanctif)\b", re.I)

_FILLER = re.compile(
    r"\b(amazing|incredible|revolutionary|game[\s-]changing|cutting[\s-]edge|"
    r"world[\s-]class|paradigm[\s-]shift|next[\s-]generation|unprecedented|"
    r"breakthrough)\b", re.I)

_SPECIFIC = re.compile(
    r"\b(?:n\s*=|p\s*=|k_?[34]|kappa_?[34]|kurt|skew|cocycle|sheaf|cumulant|"
    r"distribution|subgroup|stratum|stratified)\b", re.I)

_NUMBERS = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|x|SE|n\s*=)?\b")


def detect_moral_contamination(prose: str) -> List[Detection]:
    out = []
    for m in _MORAL.finditer(prose):
        ctx = prose[max(0, m.start()-30):m.end()+30]
        if re.search(r"\bgood\s+(data|signal|fit|enough|practice)\b", ctx, re.I):
            continue
        out.append(Detection(
            "moral_contamination",
            f"moral token '{m.group(0)}' at offset {m.start()}",
            prose[max(0, m.start()-40):m.end()+40],
            "Convert subjective judgments to systems mechanics: "
            "good->optimized; bad->entropic; right->aligned; wrong->miscalibrated.",
        ))
    return out


def detect_signal_bleed(prose: str) -> List[Detection]:
    return [Detection(
        "signal_bleed",
        f"theological token '{m.group(0)}' at offset {m.start()}",
        prose[max(0, m.start()-40):m.end()+40],
        "Strip institutional markers. Replace via CODE-U transduction matrix "
        "(grace->asymmetric influx; covenant->invariant constraint; "
        "logos->generative logic).",
    ) for m in _THEOLOGY.finditer(prose)]


def detect_over_smoothing(prose: str) -> List[Detection]:
    out = []
    n_words = max(len(prose.split()), 1)
    n_fillers = len(_FILLER.findall(prose))
    n_specific = len(_SPECIFIC.findall(prose)) + len(_NUMBERS.findall(prose))
    if n_fillers / n_words > 0.02:
        out.append(Detection(
            "over_smoothing",
            f"filler density {n_fillers/n_words:.3f} > 0.02",
            None,
            "Recover the load-bearing primitive and inject compensatory friction. "
            "Replace marketing adjectives with the specific numerical claim they obscure.",
        ))
    if n_words > 60 and n_specific / n_words < 0.015:
        out.append(Detection(
            "over_smoothing",
            f"low specificity: {n_specific} specific tokens across {n_words} words",
            None,
            "Restore at least one explicit per-subgroup numerical claim. "
            "Stripped encoding preserves the load-bearing primitive.",
        ))
    return out


def audit_transmission(
    prose: str,
    *,
    encoding: Literal["overt", "stripped", "structural"] = "stripped",
    source_primitives: Optional[List[str]] = None,
) -> List[Detection]:
    """Run the three automated CODE-U §3 checks. source_primitives accepted
    for signature compatibility with the spec; not consumed by automated modes."""
    findings: List[Detection] = []
    findings += detect_moral_contamination(prose)
    if encoding in ("stripped", "structural"):
        findings += detect_signal_bleed(prose)
    findings += detect_over_smoothing(prose)
    return findings
