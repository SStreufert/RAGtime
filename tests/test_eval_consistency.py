"""Unit tests for eval_consistency (verdict/rationale reconciliation)."""

from mlflow.entities import Expectation, Feedback

from eval_consistency import (
    HUMAN_OVERRIDE_ACTION,
    SOURCE_RUN_ID_KEY,
    Action,
    ReconciliationReport,
    cast_metric_value,
    classify_rationale,
    compute_mean_metrics,
    format_report,
    normalize_verdict,
    plan_actions,
)

RUN_ID = "run-123"

# Verbatim rationales from trace tr-96f1451671202a3fcc653141294d362d
GROUNDEDNESS_POSITIVE = (
    "The claim lists several key developments: Mrs Elara Nightingale championed "
    "tourism in the 1930s, which is mentioned; ... capitalizing on black swan "
    "population and pristine beaches is stated. All claim points are supported "
    "by the document."
)
RELEVANCE_POSITIVE = (
    "Let's think step by step: The question asks about Swan Lagoon's transition "
    "from a mining town to a tourist destination. ... These points directly "
    "address the transition process, so the answer is relevant."
)
NEGATIVE = (
    "The claim mentions a festival, but the document does not mention any "
    "festival and provides no information about it. The claim is not supported."
)
HEDGED = (
    "The document partially supports the claim about tourism, though it mostly "
    "covers the mining era."
)
# False-flip regression: value "no" is correct; "is relevant" appears only
# inside the negated "none of the content is relevant" construction.
NEGATED_POSITIVE = (
    "Let's think step by step: The document discusses pipeline reliability and "
    "retrieval layers, but it contains no mention of emergency contact numbers "
    "or gas leaks. Therefore, none of the content is relevant to the question."
)
NEGATED_ADDRESS = (
    "The question asks for a famous tradition to do **at** the Costal Walk. "
    "The answer discusses a tradition (sitting at Nightingale's Rest at 3:38 PM) "
    "but this is not an action performed *on* the walk itself. Therefore, the "
    "provided information does not directly address what to do **at** the "
    "Costal Walk."
)
NEGATED_PARTIAL_SUPPORT = (
    "The document states that Mrs. Elara Nightingale was a local schoolteacher "
    "who championed tourism in the 1930s and wrote articles advocating for Swan "
    "Lagoon, but the claim asks both who advocated (which is supported) and what "
    "they did; the answer provided only gives her name without describing the "
    "action of writing advocacy articles. Hence the claim is not fully supported."
)

# Real rationales from run de629a7972a740c0a97c42fb244a0953 that the old
# keyword-count classifier mis-flipped. Each documents a required verdict.
REAL_NO_ELEMENTS_MISSING = (
    "Let's think step by step: The document directly states that replica "
    "'swan egg' nuggets that double as paperweights can be purchased at the "
    "mine's gift shop. The claim matches this description exactly, including "
    "both the item (replica swan egg nuggets) and its dual function "
    "(paperweights). No elements of the claim are missing or contradicted by "
    "the document."
)
REAL_DIRECTLY_CONTRADICTS = (
    "Let's think step by step: The document states that both locations are "
    "open until 21:00 (9pm) and explicitly mentions 'neither location is open "
    "at 10pm.' This directly contradicts the claim's assertion that the shop "
    "closes at 9pm, as it clarifies that neither place operates at 10pm. The "
    "document's own information supports the idea that ice cream cannot be "
    "obtained at 10pm."
)
REAL_NOT_FULLY_SUPPORTED = (
    "Let's think step by step: The claim includes the address '28 Pioneer "
    "Street,' which is not mentioned in the document. However, the question "
    "only asks for operating hours. While the document supports the hours "
    "(Mon-Fri 9:30-18:00 and Sat 9:00-13:00), the inclusion of an unverified "
    "address in the claim introduces unsupported information. Since the claim "
    "contains elements not confirmed by the document, it is not fully "
    "supported."
)
REAL_INFORMATION_PRESENT = (
    "Let's think step by step: The claim states the emergency contact number "
    "for gas leaks is 1800 427 532 and references page 4. In the document, "
    "the second content block explicitly lists 'Gas Emergency 1800 427 532' "
    "under the 'Utilities Emergency Contacts' section. While the document "
    "does not mention a specific page number, the information is present in "
    "the provided text. The claim's answer matches this exact number and "
    "context."
)
REAL_ALL_MATCH_EXACTLY = (
    "Let's think step by step: The document contains the line 'The Mining "
    "Heritage B&B Accommodation 45 Pioneer Street Check-in 14:00-20:00' "
    "which directly matches both the address (45 Pioneer Street) and "
    "check-in time (14:00-20:00) stated in the claim. No conflicting or "
    "missing information is present."
)
REAL_DOES_NOT_EXPLICITLY_STATE = (
    "Let's think step by step: The document discusses the history of Swan "
    "Lagoon during a gold boom, mentioning infrastructure developments like "
    "the Telegraph Office (1890) and St. Agnes Church (1892), as well as "
    "population growth and a local newspaper. However, it does not explicitly "
    "state who discovered gold in Swan Lagoon or founded the town. The text "
    "focuses on the effects of the gold rush rather than identifying "
    "specific individuals responsible for discovery or founding."
)
REAL_PARTIALLY_RELEVANT = (
    "Let's think step by step: The document mentions Swan Lagoon's history "
    "involving gold and its transformation, implying a connection between "
    "gold discovery and the town's founding. However, it does not explicitly "
    "state who discovered the gold or founded the town. While there is "
    "partial relevance regarding the topic of gold discovery in Swan Lagoon, "
    "the specific question about the discoverer and founder remains "
    "unanswered."
)

# Verbatim rationales from run cf181f34a88b492f8ade40df56a3d9c4, where the
# classifier false-flipped correct "no" verdicts to "yes" — every flip on that
# run went in the pass direction. All of these grade traces whose RAG answer
# was the empty string.
EMPTY_ANSWER_NO_CONTENT_RELEVANT = (
    "Let's think step by step: The question asks for the emergency contact "
    "number for gas leaks. The provided answer is an empty string, which "
    "contains no information at all regarding a phone number or any other "
    "relevant data. Therefore, there is no content in the answer that is "
    "relevant to answering the question."
)
EMPTY_ANSWER_NEITHER_SUPPORTED = (
    "Let's think step by step The document only contains the question asking "
    "for opening hours but provides no answer about them. Since there is no "
    "information confirming or denying that Jed's Beachfront Cafe is open "
    "Monday to Sunday from 7:00 AM to 9:00 PM, neither part of the claim is "
    "supported."
)
EMPTY_ANSWER_NO_PARTS_RELEVANT = (
    "The question asks for the address of the Visitor Centre and its services. "
    "The provided answer is empty, containing no information about either the "
    "address or the services. Therefore there are no parts that are relevant "
    "to the question."
)
EMPTY_ANSWER_EXACTLY_CLAUSE = (
    "Let's think step by step The answer does not provide any information "
    "regarding the opening year of The Lagoon Breeze Hotel, which is exactly "
    "what the question asks for."
)
EMPTY_ANSWER_NEITHER_STATEMENT = (
    "Let's think step by step: The document only contains the question asking "
    "when the mine closed and what company operated it, but provides no "
    "information about 1935 or WAAC. Therefore neither statement in the claim "
    "is supported by the document."
)
# Legit flip from the same run: the answer was non-empty and correct, the
# judge's "no" was the error — this must keep classifying as yes.
TRUE_EXPLICITLY_STATES_DIRECTLY_ANSWERS = (
    "Let's think step by step The answer explicitly states the frequency as "
    "fortnightly which directly answers the question about recycling collection "
    "frequency in Swan Lagoon."
)

# Verbatim rationales from run 821a30736b3247dea06833d3b327eb7f and run
# 293140e983644361b012903346cef9be: three Correctness assessments flagged
# incorrectly (user-reported).
# tr-28a3ba11161835647a1a821861db2d04: value "yes" was a false flip; the
# rationale concludes the claim is unsupported via "none of ... are supported".
NONE_OF_ARE_SUPPORTED = (
    "Let's think step by step: The question asks for the number of restaurants "
    "in Swan Lagoon. The document states that the business directory listing is "
    "incomplete and does not provide a full count, only mentioning Jed's "
    "Beachfront Cafe as a listed restaurant. It does not reference Stone's Fish "
    "& Chips, Blackwood's Beach Bar, or Lagoon View Chinese Restaurant, nor "
    "does it confirm a total of four restaurants. Therefore, none of the "
    "specific restaurant names or the claim that there are exactly four "
    "restaurants are supported by the document."
)
# tr-bfa6690f9ae33f4e81dd54b415a9ef0d: value "no" was left untouched; the
# judge retracts its own verdict ("mistakenly wrote no. Correcting.").
SELF_CORRECTED_MISTAKENLY_WROTE_NO = (
    "The claim includes two statements: the address is 45 Pioneer Street, "
    "which is supported; and check-in time is 14:00-20:00, also supported. "
    "However the result field should be 'yes' if both are fully supported. "
    "So I mistakenly wrote no. Correcting."
)
# tr-def7836b6bba7601de74d753bb6978ee: value "no" was flagged incorrectly;
# the rationale concludes Pass ("this directly supports the claim that one cannot get ice cream").
TRACE_DEF7836_RATIONALE = (
    "Let's think step by step: The document states that the Golden Sands Ice Creamery "
    "is open from 10:00-21:00, which means it closes at 9pm. Therefore, it would be "
    "closed by 10pm. This directly supports the claim that one cannot get ice cream at 10pm."
)


def _feedback(value, rationale, name="correctness", run_id=RUN_ID, **kwargs):
    metadata = {SOURCE_RUN_ID_KEY: run_id}
    metadata.update(kwargs.pop("metadata", {}))
    return Feedback(name=name, value=value, rationale=rationale, metadata=metadata, **kwargs)


class TestClassifyRationale:
    def test_positive_from_real_trace(self):
        assert classify_rationale(GROUNDEDNESS_POSITIVE) == "yes"
        assert classify_rationale(RELEVANCE_POSITIVE) == "yes"

    def test_negative(self):
        assert classify_rationale(NEGATIVE) == "no"

    def test_negated_positive_stays_negative(self):
        assert classify_rationale(NEGATED_POSITIVE) == "no"

    def test_negated_address_stays_negative(self):
        assert classify_rationale(NEGATED_ADDRESS) == "no"

    def test_negated_partial_support_stays_negative(self):
        assert classify_rationale(NEGATED_PARTIAL_SUPPORT) == "no"

    def test_hedged(self):
        assert classify_rationale(HEDGED) == "hedged"

    def test_empty_is_unknown(self):
        assert classify_rationale("") == "unknown"
        assert classify_rationale(None) == "unknown"

    def test_no_signal_is_unknown(self):
        assert classify_rationale("The response is long and mentions weather.") == "unknown"

    # --- regression: real rationales from run de629a79 that mis-flipped ---

    def test_no_elements_missing_is_yes(self):
        """'No elements are missing' must not be read as a missing-failure."""
        assert classify_rationale(REAL_NO_ELEMENTS_MISSING) == "yes"

    def test_directly_contradicts_is_no(self):
        """'This directly contradicts' must be no, not matched by 'this directly'."""
        assert classify_rationale(REAL_DIRECTLY_CONTRADICTS) == "no"

    def test_not_fully_supported_is_no(self):
        assert classify_rationale(REAL_NOT_FULLY_SUPPORTED) == "no"

    def test_information_is_present_is_yes(self):
        """'information is present' + 'matches this exact number' → yes."""
        assert classify_rationale(REAL_INFORMATION_PRESENT) == "yes"

    def test_all_match_exactly_no_conflicting_missing_is_yes(self):
        assert classify_rationale(REAL_ALL_MATCH_EXACTLY) == "yes"

    def test_does_not_explicitly_state_is_no(self):
        assert classify_rationale(REAL_DOES_NOT_EXPLICITLY_STATE) == "no"

    def test_partial_relevance_is_hedged(self):
        assert classify_rationale(REAL_PARTIALLY_RELEVANT) == "hedged"

    # --- regression: false no->yes flips from run cf181f34 ---

    def test_no_content_that_is_relevant_is_no(self):
        assert classify_rationale(EMPTY_ANSWER_NO_CONTENT_RELEVANT) == "no"

    def test_neither_part_is_supported_is_no(self):
        assert classify_rationale(EMPTY_ANSWER_NEITHER_SUPPORTED) == "no"

    def test_no_parts_that_are_relevant_is_no(self):
        assert classify_rationale(EMPTY_ANSWER_NO_PARTS_RELEVANT) == "no"

    def test_exactly_what_question_asks_inside_negation_is_no(self):
        assert classify_rationale(EMPTY_ANSWER_EXACTLY_CLAUSE) == "no"

    def test_neither_statement_is_supported_is_no(self):
        assert classify_rationale(EMPTY_ANSWER_NEITHER_STATEMENT) == "no"

    def test_explicitly_states_directly_answers_is_yes(self):
        """The one true flip on cf181f34 must keep classifying as yes."""
        assert classify_rationale(TRUE_EXPLICITLY_STATES_DIRECTLY_ANSWERS) == "yes"

    # --- regression: user-reported mis-flagged Correctness assessments ---

    def test_none_of_elements_are_supported_is_no(self):
        """'none of X ... are supported' must not be eaten by 'are supported'."""
        assert classify_rationale(NONE_OF_ARE_SUPPORTED) == "no"

    def test_mistakenly_wrote_no_correcting_is_yes(self):
        """Judge self-retraction ('mistakenly wrote no. Correcting.') is yes."""
        assert classify_rationale(SELF_CORRECTED_MISTAKENLY_WROTE_NO) == "yes"

    def test_trace_def7836_cannot_in_claim_content_is_yes(self):
        """'cannot' in claim content (not verdict) must not flip to no.
        
        tr-def7836b6bba7601de74d753bb6978ee: rationale says 'This directly 
        supports the claim that one cannot get ice cream at 10pm.' The 
        'cannot' is part of the claim being supported, not a negative verdict.
        """
        assert classify_rationale(TRACE_DEF7836_RATIONALE) == "yes"


class TestNormalizeVerdict:
    def test_categorical_strings(self):
        assert normalize_verdict("yes") == "yes"
        assert normalize_verdict("YES") == "yes"
        assert normalize_verdict("No") == "no"

    def test_booleans(self):
        assert normalize_verdict(True) == "yes"
        assert normalize_verdict(False) == "no"

    def test_non_categorical(self):
        assert normalize_verdict(0.857) is None
        assert normalize_verdict("0.857") is None
        assert normalize_verdict(None) is None


class TestCastMetricValue:
    def test_categorical(self):
        assert cast_metric_value("yes") == 1.0
        assert cast_metric_value("no") == 0.0
        assert cast_metric_value(True) == 1.0

    def test_numeric(self):
        assert cast_metric_value(0.5) == 0.5
        assert cast_metric_value("0.8571428571428571") == 0.8571428571428571

    def test_unusable(self):
        assert cast_metric_value("banana") is None
        assert cast_metric_value(None) is None


class TestPlanActions:
    def test_false_negative_flips_to_yes(self):
        fb = _feedback("no", GROUNDEDNESS_POSITIVE, name="retrieval_groundedness")
        actions, agreed, no_signal, checked = plan_actions("tr-1", [fb], RUN_ID)
        assert checked == 1
        assert len(actions) == 1
        assert actions[0].kind == "flip"
        assert actions[0].old_value == "no"
        assert actions[0].new_value == "yes"
        assert actions[0].assessment_id == fb.assessment_id
        assert agreed == 0

    def test_false_positive_flips_to_no(self):
        fb = _feedback("yes", NEGATIVE)
        actions, *_ = plan_actions("tr-1", [fb], RUN_ID)
        assert len(actions) == 1
        assert actions[0].kind == "flip"
        assert actions[0].new_value == "no"

    def test_correct_no_with_negated_positive_is_untouched(self):
        fb = _feedback("no", NEGATED_POSITIVE, name="relevance_to_query")
        actions, agreed, _, _ = plan_actions("tr-1", [fb], RUN_ID)
        assert actions == []
        assert agreed == 1

    def test_correct_no_with_neither_supported_is_untouched(self):
        """cf181f34: judge said no, rationale said no — must not flip."""
        fb = _feedback("no", EMPTY_ANSWER_NEITHER_SUPPORTED, name="relevance_to_query")
        actions, agreed, _, _ = plan_actions("tr-1", [fb], RUN_ID)
        assert actions == []
        assert agreed == 1

    def test_agreement_produces_no_action(self):
        fb = _feedback("yes", RELEVANCE_POSITIVE, name="relevance_to_query")
        actions, agreed, no_signal, checked = plan_actions("tr-1", [fb], RUN_ID)
        assert actions == []
        assert agreed == 1
        assert no_signal == 0

    def test_hedged_flags_uncertain_never_flips(self):
        fb = _feedback("yes", HEDGED)
        actions, agreed, no_signal, _ = plan_actions("tr-1", [fb], RUN_ID)
        assert len(actions) == 1
        assert actions[0].kind == "uncertain"
        assert actions[0].new_value is None
        assert agreed == 0

    def test_unknown_rationale_untouched(self):
        fb = _feedback("no", "The response is long.")
        actions, agreed, no_signal, _ = plan_actions("tr-1", [fb], RUN_ID)
        assert actions == []
        assert no_signal == 1

    def test_human_override_is_never_reconsidered(self):
        """A human-adjudicated verdict must survive future reconcile passes.

        The rationale would normally force a no->yes flip; the override
        metadata marks the stored value as human-decided.
        """
        fb = _feedback(
            "no",
            GROUNDEDNESS_POSITIVE,
            name="retrieval_groundedness",
            metadata={"reconcile_action": HUMAN_OVERRIDE_ACTION},
        )
        actions, agreed, no_signal, checked = plan_actions("tr-1", [fb], RUN_ID)
        assert actions == []
        assert checked == 0
        assert agreed == 0
        assert no_signal == 0

    def test_numeric_and_foreign_and_error_skipped(self):
        numeric = _feedback("0.857", "whatever", name="retrieval_relevance/precision")
        foreign = _feedback("no", NEGATIVE, run_id="other-run")
        errored = _feedback(None, None, error="scorer timed out")
        expectation = Expectation(name="expected_response", value="ground truth")
        actions, agreed, no_signal, checked = plan_actions(
            "tr-1", [numeric, foreign, errored, expectation], RUN_ID
        )
        assert actions == []
        assert checked == 1  # only the numeric in-run feedback counts
        assert agreed == 0
        assert no_signal == 0


class TestMetricsAndReport:
    def test_compute_mean_metrics(self):
        assert compute_mean_metrics({"a": [1.0, 0.0], "b": [1.0], "c": []}) == {
            "a/mean": 0.5,
            "b/mean": 1.0,
        }

    def test_format_report(self):
        report = ReconciliationReport(run_id=RUN_ID)
        report.actions = [
            Action(
                trace_id="tr-1",
                assessment_id="a-1",
                name="retrieval_groundedness",
                kind="flip",
                old_value="no",
                new_value="yes",
                rationale=GROUNDEDNESS_POSITIVE,
            ),
            Action(
                trace_id="tr-1",
                assessment_id="a-2",
                name="correctness",
                kind="uncertain",
                old_value="yes",
                new_value=None,
                rationale=HEDGED,
            ),
        ]
        report.metrics_before = {"retrieval_groundedness/mean": 0.0}
        report.corrected_metrics = {"retrieval_groundedness/mean": 1.0}
        text = format_report(report)
        assert "1 verdict(s) flipped" in text
        assert "1 flagged uncertain" in text
        assert "retrieval_groundedness/mean: 0.0000 -> 1.0000" in text
        assert "flip retrieval_groundedness: no -> yes" in text
