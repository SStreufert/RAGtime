"""Reconcile LLM-judge verdicts with their rationale text.

MLflow's built-in GenAI scorers store the judge's ``result`` and ``rationale``
as two independent fields of one model generation and never cross-check them,
so a small local judge can emit e.g. ``result: "no"`` next to a rationale that
concludes the criterion passed. This module detects those contradictions in an
evaluation run, flips the verdict when the rationale's conclusion is
unambiguous, flags hedged rationales for human review (never auto-flipped),
re-logs corrected run metrics under their original names, and writes
corrections back through MLflow's public ``update_assessment`` API.

Only categorical yes/no feedback assessments produced by the run itself are
considered. Expectations, numeric scores (e.g. retrieval_relevance/precision),
scorer-error rows, and rationales with no recognizable verdict language are
left untouched.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import fmean
from typing import Iterable

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Feedback

SOURCE_RUN_ID_KEY = "mlflow.assessment.sourceRunId"
# Assessment metadata value marking a verdict as human-adjudicated; such
# assessments are never reconsidered by reconciliation (see plan_actions).
HUMAN_OVERRIDE_ACTION = "human_override"

VERDICT_YES = "yes"
VERDICT_NO = "no"
SENTIMENT_HEDGED = "hedged"
SENTIMENT_UNKNOWN = "unknown"

# Ordered, high-precision verdict rules applied to the rationale's *conclusion*
# (last sentences). Earlier CoT exploration is ignored so that "does not
# explicitly state X" mid-rationale cannot outvote a clear final verdict.
# Each rule is (regex, sentiment). First match wins. Patterns are designed to
# be negation-aware: matched against a copy of the text with common negated
# positive constructions rewritten, so "No elements are missing" does not hit
# a bare `missing` rule.

_COT_PREFIX = re.compile(r"^\s*let'?s think step by step:\s*", re.I)

# Rewrite negated-positive phrases to tokens so positive rules do not fire
# inside them and so the rewrite itself carries verdict signal.
#   <OK>   - negated failure ("no elements are missing") → yes-signal
#   <NEG>  - negated success ("does not explicitly state") → no-signal
# Order matters: longer / more specific constructions first.
_NEGATED_POSITIVE_REWRITES = [
    # "No elements of the claim are missing" / "…missing or contradicted"
    # / "No conflicting information is present"
    (
        re.compile(
            r"\bno (?:other )?(?:elements?|parts?|details?|claims?|conflicting|contradicting|contradictory|"
            r"additional|further|other)"
            r"(?: of (?:the |this |that )?[\w\s-]{0,40})?"
            r" (?:are |is )?(?:missing|absent|incorrect|wrong|present|provided|included)"
            r"(?:\s+or\s+(?:contradict(?:ed|ing)|conflicting|incorrect|wrong|missing|absent))?\b",
            re.I,
        ),
        " <OK> ",
    ),
    # "no parts of the claim not supported" / "no elements not relevant" — double
    # negative (negated failure predicate) → positive. Matches "no X not Y"
    # where Y is a positive quality predicate.
    (
        re.compile(
            r"\bno (?:other )?(?:elements?|parts?|details?|claims?)"
            r"(?: of (?:the |this |that )?[\w\s-]{0,40})?"
            r" (?:are |is )?not"
            r" (?:supported|relevant|correct|accurate|sufficient|grounded|adequate|consistent)\b",
            re.I,
        ),
        " <OK> ",
    ),
    # "no content that is relevant" / "no parts that are relevant" — relative-
    # clause negation; consume the positive predicate so the bare yes-rules
    # ("is relevant", "is supported") cannot fire inside it. Coordinate forms
    # ("no elements are missing … and the claim is supported") are left to
    # the rule above, which rewrites only the negated-failure half.
    (
        re.compile(
            r"\bno\b[^.]*?\b(?:that|which|who)\s+(?:is|are)\s+"
            r"(?:relevant|supported|correct|accurate|sufficient|grounded|adequate|consistent)\b",
            re.I,
        ),
        " <NEG> ",
    ),
    # "neither part of the claim is supported" — "neither" is a negation none
    # of the other rules recognised, so a bare "is supported" flipped it yes.
    (
        re.compile(
            r"\bneither\b[^.]*?\b(?:is|are)\s+"
            r"(?:supported|relevant|correct|accurate|sufficient|grounded|adequate|consistent)\b",
            re.I,
        ),
        " <NEG> ",
    ),
    # "No information is provided" / "No information is available" — fail,
    # not the negated-success pattern above (bare "information" was too broad).
    (
        re.compile(
            r"\bno (?:information|evidence|mention|match|direct linkage|data)\b"
            r"(?:[^.]*\b(?:is |are )?(?:provided|available|present|listed|included|mentioned))?",
            re.I,
        ),
        " <NEG> ",
    ),
    # "none of the content is relevant" — whole clause is a fail
    (
        re.compile(r"\bnone of\b[^.]*\bis (?:not )?relevant\b", re.I),
        " <NEG> ",
    ),
    # Preserve "not fully supported" intact (generic "is not" rewrite would
    # strip the "not" and let the bare "fully supported" yes-rule fire).
    (re.compile(r"\bis not fully supported\b", re.I), " not fully supported "),
    (re.compile(r"\bare not fully supported\b", re.I), " not fully supported "),
    # "does not explicitly/directly/specifically <verb>" → negated action
    (
        re.compile(
            r"\bdoes not (?:explicitly |directly |specifically |necessarily |fully |actually |clearly )?"
            r"(?:state|mention|provide|contain|address|answer|confirm|list|identify|name|specify|describe|cover|support|match|include|note|refer|establish|validate|verify|demonstrate|show|prove)\b",
            re.I,
        ),
        " <NEG> ",
    ),
    # "does not contradict" / "do not contradict" are positives — rewrite
    # before generic does-not / bare "contradicts" fail-rule.
    (re.compile(r"\bdo(?:es)? not contradict\b", re.I), " <OK> "),
    # "does not detract/negate/undermine the relevance" — positive concession.
    (
        re.compile(
            r"\bdo(?:es)? not (?:detract|negate|undermine|reduce|lessen|diminish|undercut|affect|change|alter)\b",
            re.I,
        ),
        " <OK> ",
    ),
    # "no other parts … contradict" / "no irrelevant information" — positives.
    (re.compile(r"\bno\b[^.]*\bcontradict\w*\b", re.I), " <OK> "),
    (re.compile(r"\bno irrelevant\b", re.I), " <OK> "),
    # "without any contradictions or missing information" / "without requiring
    # external knowledge" / "without including irrelevant details" — positives.
    # Consume the following noun phrase so leftover "irrelevant"/"missing" cannot re-fire.
    (
        re.compile(
            r"\bwithout\b\s+(?:any\s+)?(?:contradictions?|conflicts?|ambiguities|missing|"
            r"irrelevant|requiring|needing|the need|having to|including)\b(?:\s+\w+){0,3}",
            re.I,
        ),
        " <OK> ",
    ),
    # "not directly addressed" / "not directly answered" — fail (the bare
    # "directly addressed" yes-rule would otherwise fire inside the negation).
    (
        re.compile(r"\bnot directly (?:addressed|answered|supported|matched|confirmed)\b", re.I),
        " <NEG> ",
    ),
    # "do not align with" is a fail — rewrite before bare "align with" yes-rule.
    (re.compile(r"\bdo(?:es)? not align\b", re.I), " <NEG_ALIGN> "),
    # "no entry matches X" — keep the whole phrase as a no-signal token so
    # bare "matches" yes-rule cannot fire inside it.
    (re.compile(r"\bno entry matches\b", re.I), " <NEG_MATCH> "),
    # Judge self-correction: the judge retracts its own emitted verdict
    # ("So I mistakenly wrote no. Correcting.") — the true verdict is the
    # opposite of the retracted one.
    (re.compile(r"\bmistakenly wrote\b[^.]*?\bno\b", re.I), " <OK> "),
    (re.compile(r"\bmistakenly wrote\b[^.]*?\byes\b", re.I), " <NEG> "),
    # "none of X ... are supported/relevant" — the whole clause is a fail;
    # consume the positive predicate so the bare yes-rules ("are supported",
    # "is relevant") cannot win the rightmost-match contest inside it.
    (
        re.compile(
            r"\bnone of\b[^.]*?\b(?:is|are)\s+"
            r"(?:supported|relevant|correct|accurate|sufficient|grounded|adequate|consistent)\b",
            re.I,
        ),
        " <NEG> ",
    ),
    (re.compile(r"\bdo(?:es)? not\b", re.I), " <NEG> "),
    (re.compile(r"\bdid not\b", re.I), " <NEG> "),
    (re.compile(r"\bis not\b", re.I), " <NEG> "),
    (re.compile(r"\bare not\b", re.I), " <NEG> "),
    (re.compile(r"\bwould not\b", re.I), " <NEG> "),
    # "cannot" / "can not" are handled by the specific conclusion rule
    # "cannot be found|confirmed|verified|supported|determined|established"
    # Bare "cannot" rewrites broke claim-content phrases like "the claim that one cannot get ice cream"
    # "without" only introducing a missing action, not "without contradiction"
    # or "without adding or omitting" (both are completeness positives).
    (
        re.compile(
            r"\bwithout\b(?!\s+(?:contradiction|conflict|ambiguity|controversy|issue|problem|objection|qualification|adding|omitting|changing|altering|deviating)\b)",
            re.I,
        ),
        " <NEG> ",
    ),
    (re.compile(r"\bnone of\b", re.I), " <NEG> "),
    # Do NOT rewrite bare "not fully" — that breaks "not fully supported".
]

# Applied to the de-negated conclusion text. First match wins.
# Order: unambiguous fail phrases first (so "not fully supported" is not
# eaten by a bare "fully supported" yes-rule), then pass phrases, then
# tokens last so a subordinate "while … does not mention …" cannot beat
# "information is present" in the same sentence.
_CONCLUSION_RULES: list[tuple[str, str]] = [
    # --- high-precision FAIL (must precede any "fully supported" yes-rule) ---
    (r"\bnot fully supported\b", VERDICT_NO),
    (r"\bcontradicts?\b", VERDICT_NO),
    (r"\bcontradicted\b", VERDICT_NO),
    (r"\bis not (?:fully )?supported\b", VERDICT_NO),
    (r"\bare not (?:fully )?supported\b", VERDICT_NO),
    (r"\bnot (?:fully )?(?:relevant|sufficient|correct|accurate|grounded|adequate)\b", VERDICT_NO),
    (r"\birrelevant\b", VERDICT_NO),
    (r"\bunrelated\b", VERDICT_NO),
    (r"\binsufficient\b", VERDICT_NO),
    (r"\bincomplete\b", VERDICT_NO),
    (r"\bincorrect\b", VERDICT_NO),
    (r"\binaccurate\b", VERDICT_NO),
    (r"\bfails? to\b", VERDICT_NO),
    (r"\bunable to\b", VERDICT_NO),
    (r"\bcannot (?:be )?(?:found|confirmed|verified|supported|determined|established)\b", VERDICT_NO),
    (r"\blacks? (?:the |any |specific |direct )?", VERDICT_NO),
    (r"\bis (?:entirely |completely |totally |fully )?absent\b", VERDICT_NO),
    (r"\bare (?:entirely |completely |totally |fully )?absent\b", VERDICT_NO),
    (r"\bis missing\b", VERDICT_NO),
    (r"\bare missing\b", VERDICT_NO),
    (r"\bno (?:information|evidence|mention|match|direct linkage|data)\b", VERDICT_NO),
    (r"\bno entry matches\b", VERDICT_NO),
    (r"\bremains? unanswered\b", VERDICT_NO),
    (r"\bis not (?:present|listed|mentioned|stated|provided|included|found|available)\b", VERDICT_NO),
    (r"\bare not (?:present|listed|mentioned|stated|provided|included|found|available)\b", VERDICT_NO),
    (r"\bnot (?:present|listed|mentioned|stated|provided|included|found|available)\b", VERDICT_NO),
    (r"<NEG_MATCH>", VERDICT_NO),
    (r"<NEG_ALIGN>", VERDICT_NO),
    # --- high-precision PASS ---
    # lookbehind: bare "fully supported" must not fire inside "not fully supported"
    (r"(?<!not )\bfully (?:supported|answers?|address(?:es|ed)?|matches?|covers?|confirmed|corroborated|relevant|correct|accurate)\b", VERDICT_YES),
    (r"\bdirectly (?:answers?|address(?:es|ed)?|supports?|supported|matches?|confirms?|corroborat(?:e|es|ed|ing)|responds?|references?)\b", VERDICT_YES),
    (r"\bexactly (?:matches?|the same|as requested|as stated|as described)\b", VERDICT_YES),
    (r"\bmatches? (?:the|this|both|exactly|perfectly)\b", VERDICT_YES),
    (r"\ball (?:addresses?|claims?|elements?|parts?|details?|information|hours?|numbers?|facts?|entries?) match\b", VERDICT_YES),
    (r"\ball (?:referenced |claimed |mentioned )?(?:information|details?|elements?|parts?|facts?) (?:are |is) present\b", VERDICT_YES),
    (r"\binformation (?:is |are) present\b", VERDICT_YES),
    (r"\bis (?:fully )?(?:supported|relevant|sufficient|correct|accurate|grounded|consistent|corroborated|confirmed|verified)\b", VERDICT_YES),
    (r"\bare (?:fully )?(?:supported|relevant|sufficient|correct|accurate|grounded|consistent|corroborated|confirmed|verified)\b", VERDICT_YES),
    (r"\bno (?:other )?(?:contradiction|discrepancy|conflict)s?\b", VERDICT_YES),
    (r"\bnothing in the document contradicts\b", VERDICT_YES),
    (r"\baligned? with the (?:document|claim|question|query)\b", VERDICT_YES),
    (r"\bconsistent with the document\b", VERDICT_YES),
    (r"\bthe (?:answer|claim) is (?:correct|accurate|supported|true|verified|fully supported)\b", VERDICT_YES),
    (r"\bclaim (?:is|remains) (?:correct|accurate|supported|true|fully supported)\b", VERDICT_YES),
    (r"\banswer (?:is|remains) (?:correct|accurate|relevant|supported)\b", VERDICT_YES),
    (r"\bcorrectly (?:notes?|states?|identifies?|addresses?|answers?|aligns?)\b", VERDICT_YES),
    (r"\balign(?:s|ed|ing)? with\b", VERDICT_YES),
    (r"\bimplies?\b", VERDICT_YES),
    (r"\bis clear\b", VERDICT_YES),
    (r"(?<!not )(?<!n't )\bconfirm(?:s|ed|ing)?\b", VERDICT_YES),
    (r"\bverif(?:y|ies|ied)\b", VERDICT_YES),
    (r"\baccurate\b", VERDICT_YES),
    (r"\bcorrect\b", VERDICT_YES),
    # Bare "exactly" was removed: "does not provide …, which is exactly what
    # the question asks" is a fail whose subordinate clause outvoted the
    # negation under rightmost-match. The specific phrases ("exactly matches",
    # "matches this/both", "directly answers") above carry the yes signal.
    (r"\bmatches\b", VERDICT_YES),
    (r"\bmatch\b", VERDICT_YES),
    (r"\bexplicitly (?:states?|mentions?|lists?|confirms?|provides?|identifies?)\b", VERDICT_YES),
    # --- token verdicts from rewrites (lowest; subordinate negations only) ---
    (r"<OK>", VERDICT_YES),
    (r"<NEG>", VERDICT_NO),
]

# Hedge markers that mean "partial" — never auto-flipped, only review-flagged.
# Only match phrases that genuinely indicate uncertainty about the overall verdict.
# Individual modifier words (mostly, largely, partial, almost, nearly, partly, somewhat)
# are excluded because they can appear in clear verdicts ("mostly correct" = yes,
# "largely irrelevant" = no). The phrases below are verdict-level hedges.
_HEDGE_RE = re.compile(
    r"\b("
    r"partially (?:relevant|supported|answers?|addresses?|correct|accurate|supports?)|"
    r"somewhat (?:relevant|supported)|"
    r"partial (?:relevance|support)|"
    r"not entirely|"
    r"to some extent|"
    r"\bincomplete\b"
    r")\b",
    re.I,
)


def _prepare_conclusion(rationale: str) -> str:
    """Strip CoT preamble and return the last 2 sentences (the verdict zone)."""
    text = _COT_PREFIX.sub("", rationale.strip())
    # Split on sentence boundaries; keep the final 2 non-empty sentences.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text.lower()
    zone = " ".join(sentences[-2:])
    return zone.lower()


def _denegate(text: str) -> str:
    """Neutralise negated-positive constructions so they cannot fire yes-rules."""
    out = text
    for pattern, repl in _NEGATED_POSITIVE_REWRITES:
        out = pattern.sub(repl, out)
    return out


# Trailing meta-commentary about non-answer content ("other entries are
# unrelated", "page numbers are irrelevant") is not the verdict sentence —
# drop it and re-score so a digression cannot outvote a prior "directly answers".
_DIGRESSION_RE = re.compile(
    r"^(?:other |the rest\b|page numbers?|the mention of page|"
    r"other details|other sections|other entries|other content|"
    r"other businesses|other services|other information|"
    r"the page numbers?|the mention of |the document's mention of page|"
    r"the other )",
    re.I,
)

# Trailing "However/Although … does not explicitly …" is a caveat after a
# positive verdict, not a rejection — drop it so it cannot flip yes→no.
_CAVEAT_RE = re.compile(
    r"^(?:however|although|though|while|but)\b.*\bdoes not (?:explicitly|"
    r"directly|specifically|fully|necessarily|clearly)\b",
    re.I,
)


def _classify_conclusion_zone(zone: str) -> str:
    """Apply conclusion rules; rightmost match wins.

    English verdicts often bury the real conclusion after a concession
    ("page numbers are unrelated ... but the core information is fully
    addressed"). First-match would score the concession; rightmost-match
    scores the final clause.
    """
    best: tuple[int, str] | None = None
    for pattern, sentiment in _CONCLUSION_RULES:
        # finditer, not search: a pattern can match several times and only its
        # rightmost match may compete (e.g. a <NEG> token at the end of the
        # zone must beat an earlier incidental "confirming").
        for m in re.finditer(pattern, zone):
            if best is None or m.start() >= best[0]:
                best = (m.start(), sentiment)
    if best is not None:
        return best[1]
    # Fallback: score remaining markers in the zone only.
    positive = len(
        re.findall(
            r"(directly (?:answers?|addresses?|supports?|matches?|confirms?|references?)|"
            r"explicitly (?:states?|mentions?|lists?)|"
            r"\bfully supported\b|\bexactly matches\b|\ball .* match\b|"
            r"\b(?:relate|relates) to the specific (?:query|question)\b|"
            r"\bis (?:fully )?(?:supported|relevant|sufficient|correct|accurate)\b)",
            zone,
        )
    )
    negative = len(
        re.findall(
            r"(<NEG>|\bdoes not\b|\bnot (?:fully |explicitly |directly )?(?:support|match|answer|address|provide|contain|mention|state|confirm|establish)\b|"
            r"\bno (?:information|evidence|mention|match)\b|\blacks?\b|\binsufficient\b|"
            r"\birrelevant\b|\bcontradicts?\b|\bfails? to\b|\bunable to\b)",
            zone,
        )
    )
    if positive > negative:
        return VERDICT_YES
    if negative > positive:
        return VERDICT_NO
    return SENTIMENT_UNKNOWN


def classify_rationale(rationale: str | None) -> str:
    """Classify a rationale's own conclusion.

    Returns one of:
        "yes"      - rationale unambiguously concludes the criterion passed
        "no"       - rationale unambiguously concludes the criterion failed
        "hedged"   - rationale hedges ("partially", "mostly"...) - review only
        "unknown"  - no recognizable verdict language; value stands untouched

    Strategy: look only at the final sentences (where judges put the verdict),
    neutralise negated positives, then apply ordered high-precision rules.
    A hedge word anywhere in the conclusion short-circuits to review.
    Verdicts are scored **final-sentence-first** so a mid-zone incidental
    ("page numbers are irrelevant") cannot outvote a final "directly answers".
    A final sentence that is only a caveat ("However, it does not explicitly
    describe …") is skipped so it cannot erase a prior positive verdict.
    """
    if not rationale:
        return SENTIMENT_UNKNOWN

    text = _COT_PREFIX.sub("", rationale.strip()).lower()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return SENTIMENT_UNKNOWN

    # Drop trailing digression sentences that only comment on non-answer content.
    while len(sentences) > 1 and _DIGRESSION_RE.match(sentences[-1]):
        sentences = sentences[:-1]

    # Drop a trailing caveat sentence (However/Although + does-not) when a
    # prior sentence exists — the judge's overall verdict was already given.
    while len(sentences) > 1 and _CAVEAT_RE.match(sentences[-1]):
        sentences = sentences[:-1]

    # Prefer the final sentence; only if it has no signal, fall back to the
    # last-two-sentence zone (preserves multi-sentence verdict phrasing).
    for zone in (sentences[-1:], sentences[-2:]):
        zone_text = " ".join(zone)
        if _HEDGE_RE.search(zone_text):
            return SENTIMENT_HEDGED
        verdict = _classify_conclusion_zone(_denegate(zone_text))
        if verdict != SENTIMENT_UNKNOWN:
            return verdict

    return SENTIMENT_UNKNOWN


def normalize_verdict(value) -> str | None:
    """Map an assessment value to "yes"/"no", or None if not categorical."""
    if isinstance(value, bool):
        return VERDICT_YES if value else VERDICT_NO
    if isinstance(value, str):
        v = value.strip().lower()
        if v == VERDICT_YES:
            return VERDICT_YES
        if v == VERDICT_NO:
            return VERDICT_NO
    return None


def cast_metric_value(value) -> float | None:
    """Cast an assessment value the way MLflow's metric aggregation does.

    Same rules as mlflow.genai.scorers.aggregation, plus a float-string
    fallback so values that round-tripped through the REST API as strings
    (e.g. retrieval_relevance/precision) still aggregate like the original
    in-process run.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in (VERDICT_YES, "true"):
            return 1.0
        if v in (VERDICT_NO, "false"):
            return 0.0
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass
class Action:
    trace_id: str
    assessment_id: str
    name: str
    kind: str  # "flip" | "uncertain"
    old_value: str
    new_value: str | None  # None for uncertain flags
    rationale: str

    @property
    def snippet(self) -> str:
        text = self.rationale.replace("\n", " ")
        return text if len(text) <= 120 else text[:117] + "..."


@dataclass
class ReconciliationReport:
    run_id: str
    dry_run: bool = False
    traces_checked: int = 0
    assessments_checked: int = 0
    agreed: int = 0
    no_signal: int = 0
    actions: list[Action] = field(default_factory=list)
    metrics_before: dict[str, float] = field(default_factory=dict)
    corrected_metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def flips(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "flip"]

    @property
    def uncertain(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "uncertain"]


def plan_actions(
    trace_id: str, assessments: Iterable, run_id: str
) -> tuple[list[Action], int, int, int]:
    """Decide per-assessment actions. Pure; returns (actions, agreed, no_signal, checked)."""
    actions: list[Action] = []
    agreed = no_signal = checked = 0
    for a in assessments:
        if not isinstance(a, Feedback):
            continue
        if (a.metadata or {}).get(SOURCE_RUN_ID_KEY) != run_id:
            continue
        if (a.metadata or {}).get("reconcile_action") == HUMAN_OVERRIDE_ACTION:
            continue
        if a.error is not None:
            continue
        checked += 1
        verdict = normalize_verdict(a.value)
        if verdict is None:
            continue
        sentiment = classify_rationale(a.rationale)
        if sentiment == SENTIMENT_UNKNOWN:
            no_signal += 1
            continue
        if sentiment == SENTIMENT_HEDGED:
            actions.append(
                Action(
                    trace_id=trace_id,
                    assessment_id=a.assessment_id,
                    name=a.name,
                    kind="uncertain",
                    old_value=verdict,
                    new_value=None,
                    rationale=a.rationale or "",
                )
            )
            continue
        if sentiment == verdict:
            agreed += 1
            continue
        actions.append(
            Action(
                trace_id=trace_id,
                assessment_id=a.assessment_id,
                name=a.name,
                kind="flip",
                old_value=verdict,
                new_value=sentiment,
                rationale=a.rationale or "",
            )
        )
    return actions, agreed, no_signal, checked


def compute_mean_metrics(values_by_name: dict[str, list[float]]) -> dict[str, float]:
    return {f"{name}/mean": fmean(values) for name, values in values_by_name.items() if values}


def reconcile_run(run_id: str, *, dry_run: bool = False) -> ReconciliationReport:
    """Reconcile every judge verdict in ``run_id`` against its rationale.

    Applies flips and uncertain flags via the public assessment API (unless
    ``dry_run``), re-logs all categorical run metrics under their original
    names, and tags the run as reconciled.
    """
    client = MlflowClient()
    report = ReconciliationReport(run_id=run_id, dry_run=dry_run)
    experiment_id = client.get_run(run_id).info.experiment_id
    traces = client.search_traces(
        run_id=run_id, locations=[experiment_id], include_spans=False
    )
    report.traces_checked = len(traces)

    values_before: dict[str, list[float]] = defaultdict(list)
    values_after: dict[str, list[float]] = defaultdict(list)
    assessments_by_id: dict[str, Feedback] = {}

    for trace in traces:
        trace_id = trace.info.trace_id
        assessments = trace.info.assessments or []
        actions, agreed, no_signal, checked = plan_actions(
            trace_id, assessments, run_id
        )
        report.actions.extend(actions)
        report.agreed += agreed
        report.no_signal += no_signal
        report.assessments_checked += checked
        flips = {a.assessment_id: a for a in actions if a.kind == "flip"}
        for a in assessments:
            if not isinstance(a, Feedback):
                continue
            if (a.metadata or {}).get(SOURCE_RUN_ID_KEY) != run_id:
                continue
            before = cast_metric_value(a.value)
            if before is not None:
                values_before[a.name].append(before)
                flip = flips.get(a.assessment_id)
                if flip is not None:
                    values_after[a.name].append(cast_metric_value(flip.new_value))
                else:
                    values_after[a.name].append(before)
            assessments_by_id[a.assessment_id] = a

    report.metrics_before = compute_mean_metrics(values_before)
    report.corrected_metrics = compute_mean_metrics(values_after)

    if dry_run:
        return report

    for action in report.actions:
        original = assessments_by_id.get(action.assessment_id)
        if original is None:
            report.errors.append(
                f"{action.name} ({action.assessment_id}): assessment not found for write-back"
            )
            continue
        try:
            metadata = dict(original.metadata or {})
            metadata["reconciled_at"] = datetime.now(timezone.utc).isoformat()
            if action.kind == "flip":
                metadata["reconcile_prior_value"] = action.old_value
                metadata["reconcile_action"] = "flip"
                value = action.new_value
            else:
                metadata["reconcile_flag"] = "uncertain"
                value = original.value
            mlflow.update_assessment(
                trace_id=action.trace_id,
                assessment_id=action.assessment_id,
                assessment=Feedback(
                    name=original.name,
                    value=value,
                    rationale=original.rationale,
                    source=original.source,
                    span_id=original.span_id,
                    metadata=metadata,
                ),
            )
        except Exception as e:
            report.errors.append(
                f"{action.name} ({action.assessment_id}) [{action.kind}]: {e}"
            )

    try:
        if report.corrected_metrics:
            for name, value in report.corrected_metrics.items():
                client.log_metric(run_id=run_id, key=name, value=value)
        client.set_tag(run_id, "reconciled", "true")
        client.set_tag(run_id, "reconcile_corrected", str(len(report.flips)))
        client.set_tag(run_id, "reconcile_uncertain", str(len(report.uncertain)))
    except Exception as e:
        report.errors.append(f"metric/tag logging failed: {e}")

    return report


def maybe_reconcile_run(
    run_id: str,
    *,
    enabled: bool = True,
    dry_run: bool = False,
    metrics: dict[str, float] | None = None,
) -> ReconciliationReport | None:
    """Eval-script entry point: reconcile ``run_id`` unless disabled.

    ``metrics`` (e.g. ``results.metrics``) is updated in place with the
    corrected values so printed summaries match what was logged.
    """
    if not enabled:
        return None
    report = reconcile_run(run_id, dry_run=dry_run)
    if metrics is not None and report.corrected_metrics:
        metrics.update(report.corrected_metrics)
    print(format_report(report))
    return report


def format_report(report: ReconciliationReport) -> str:
    lines = [
        f"Verdict/rationale reconciliation (run {report.run_id}"
        f"{', dry-run' if report.dry_run else ''}):",
        f"  {report.traces_checked} trace(s), {report.assessments_checked} "
        f"categorical assessment(s) checked",
        f"  {len(report.flips)} verdict(s) flipped, "
        f"{len(report.uncertain)} flagged uncertain, "
        f"{report.agreed} agreed, {report.no_signal} without signal",
    ]
    for a in report.flips:
        lines.append(f"    flip {a.name}: {a.old_value} -> {a.new_value} ({a.trace_id})")
        lines.append(f"      rationale: {a.snippet}")
    for a in report.uncertain:
        lines.append(f"    uncertain {a.name} ({a.trace_id})")
        lines.append(f"      rationale: {a.snippet}")
    changed = {
        name: (report.metrics_before.get(name), value)
        for name, value in report.corrected_metrics.items()
        if report.metrics_before.get(name) != value
    }
    if changed:
        lines.append("  corrected metrics:")
        for name, (before, after) in sorted(changed.items()):
            lines.append(f"    {name}: {before:.4f} -> {after:.4f}")
    for err in report.errors:
        lines.append(f"  ! {err}")
    return "\n".join(lines)
