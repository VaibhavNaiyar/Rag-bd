from __future__ import annotations

import pytest

from slr.contracts import Chunk, Hit, SubQuery
from slr.retrieval.context import EvidencePackage, assemble
from slr.synthesis.grounding import Grounder, LexicalVerifier, SentenceStream

TEXT_A = (
    "Cancellations made 15 or more calendar days before the event receive a full refund of the room charge. "
    "Cancellations made fewer than 7 calendar days before the event are not refunded."
)
TEXT_B = "The catering budget for a customer workshop is INR 900 per attendee per day."


def evidence() -> EvidencePackage:
    hits = [
        Hit(Chunk("c1", "d1", "Doc_11", "1", "Cancellation windows", TEXT_A, 1), 0.9, ["dense"], ["sq1"]),
        Hit(Chunk("c2", "d2", "Doc_8", "1", "Budget", TEXT_B, 1), 0.8, ["bm25"], ["sq2"]),
    ]
    return assemble(hits, [SubQuery("sq1", "cancellation"), SubQuery("sq2", "catering budget")], 9000)


def grounder(support_min: float = 0.5, auto_cite_min: float = 0.75) -> Grounder:
    return Grounder(evidence(), LexicalVerifier(), support_min, auto_cite_min, "t1_v1", "sq1")


def test_evidence_package_is_the_prompt_boundary():
    pkg = evidence()
    assert "[Doc_11 §1]" in pkg.text and "[Doc_8 §1]" in pkg.text
    assert '<document retrieved_for="cancellation">' in pkg.text
    assert pkg.text.count("<document") == pkg.text.count("</document>") == 2
    assert set(pkg.citation_map) == {"[doc_11 §1]", "[doc_8 §1]"}


def test_a_supported_sentence_ships_with_its_citation():
    g = grounder()
    out = g.process("Cancellations made 15 or more calendar days before the event receive a full refund [Doc_11 §1]. ")
    assert out.claim is not None
    assert out.text.strip().endswith("[Doc_11 §1].")
    assert out.claim.chunk_ids == ("c1",)
    assert out.claim.sub_query_id == "sq1"
    assert g.support_rate == 1.0


def test_a_fabricated_citation_is_stripped_and_counted():
    g = grounder()
    out = g.process("Cancellations made 15 or more days before the event receive a full refund [Doc_999 §4]. ")
    assert "Doc_999" not in out.text
    assert g.fabricated_blocked == 1
    assert g.fabricated_markers == ["[Doc_999 §4]"]
    # the sentence itself was supported by the evidence, so it is re-attributed
    assert out.claim is not None and out.claim.chunk_ids == ("c1",)
    assert g.auto_cited == 1


def test_an_unsupportable_sentence_is_withheld_not_shipped():
    g = grounder()
    out = g.process("Parking at the venue costs 200 rupees per car per day [Doc_11 §1]. ")
    assert out.text == ""
    assert out.claim is None
    assert g.demoted == 1
    assert any("Not verified" in u for u in g.uncertainty)
    assert g.support_rate == 0.0


def test_uncertainty_lines_never_reach_the_answer_body():
    g = grounder()
    out = g.process("UNCERTAIN: Parking charges could not be verified from the retrieved documents.\n")
    assert out.text == ""
    assert g.uncertainty == ["Parking charges could not be verified from the retrieved documents."]
    assert g.generated == 0


def test_citation_markers_are_normalised_to_the_stored_record():
    g = grounder()
    out = g.process("The catering budget is INR 900 per attendee per day [doc_8 §1]. ")
    assert "[Doc_8 §1]" in out.text, "a marker is rebuilt from the chunk record, not copied from the model"


def test_bracketed_prose_is_not_treated_as_a_citation():
    g = grounder()
    out = g.process("Cancellations [see above] made 15 or more days before the event receive a full refund [Doc_11 §1]. ")
    assert "[see above]" in out.text
    assert g.fabricated_blocked == 0


@pytest.mark.parametrize(
    "stream,expected",
    [
        (["Alpha claim [Doc_11 §1]. ", "Beta claim [Doc_8 §1]."], 2),
        (["Alpha claim. [Doc_11 §1] ", "Beta claim [Doc_8 §1]."], 2),
        (["- Bullet one [Doc_11 §1]\n", "- Bullet two [Doc_8 §1]\n"], 2),
    ],
)
def test_sentence_stream_finds_the_boundaries(stream, expected):
    splitter = SentenceStream()
    segments = []
    for delta in stream:
        for i in range(0, len(delta), 3):  # arrive in small deltas
            segments += splitter.feed(delta[i : i + 3])
    segments += splitter.flush()
    assert len([s for s in segments if s.strip()]) == expected


def test_sentence_stream_does_not_split_inside_a_marker_or_a_decimal():
    splitter = SentenceStream()
    segments = splitter.feed("The rate is 3.5 lakh [Doc_1 §2.1]. Next sentence here. ")
    segments += splitter.flush()
    assert segments[0] == "The rate is 3.5 lakh [Doc_1 §2.1]. "


def test_nli_verifier_windows_stay_in_distribution():
    """A verbatim claim must stay supported when the chunk has other sentences."""
    pytest.importorskip("torch")
    from slr.config import get_settings
    from slr.synthesis.grounding import load_nli

    try:
        verifier = load_nli(get_settings().nli_model)
    except Exception as exc:  # no model cache on this machine
        pytest.skip(f"NLI model unavailable: {exc}")
    claim = "Cancellations made fewer than 7 calendar days before the event are not refunded."
    assert verifier.support(claim, [TEXT_A])[0] >= 0.5


@pytest.mark.parametrize(
    "sentence",
    [
        "The catering vendor for Pune is not mentioned in the retrieved documents [Doc_8 §1].",
        "The evidence does not specify a parking fee for the venue.",
        "There is no information available regarding the venue's parking fee.",
        "There is no evidence indicating that the venue charges for parking.",
        "The parking fee is unknown from the provided documents.",
        "The venue has a car park, so I cannot provide a parking fee.",
    ],
)
def test_a_statement_that_the_documents_are_silent_is_uncertainty_not_a_failed_claim(sentence):
    g = grounder()
    out = g.process(sentence + " ")
    assert out.text == "" and out.claim is None
    assert out.uncertainty and "Doc_" not in out.uncertainty
    assert g.generated == 0 and g.demoted == 0, "an honest 'not stated' must not count against support"
    assert g.support_rate == 1.0


def test_a_policy_sentence_with_a_negation_is_still_a_claim():
    g = grounder()
    out = g.process("Cancellations made fewer than 7 calendar days before the event are not refunded [Doc_11 §1]. ")
    assert out.claim is not None and g.generated == 1


def test_the_verifier_reads_each_passage_with_its_subject():
    from slr.synthesis.grounding import _subject

    hit = Hit(Chunk("c9", "d9", "Doc_9", "2", "Fallout 4 › Passage 2", "The game takes place in Boston.", 2), 0.9, [], [])
    assert _subject(hit) == "Fallout 4", "section numbers are not a subject"
    claim = "Fallout 4 takes place in Boston"
    lexical = LexicalVerifier()
    assert lexical.support(claim, [hit.chunk.text])[0] < lexical.support(claim, [hit.chunk.text], [_subject(hit)])[0]
    trail = Hit(Chunk("c8", "d8", "Doc_8", "1", "Travel policy › Hotel limits", "x", 1), 0.9, [], [])
    assert _subject(trail) == "Travel policy, Hotel limits"


@pytest.mark.parametrize(
    "sentence",
    [
        "The venue cannot provide AV equipment for events over 50 people [Doc_11 §1].",
        "No refund is given for cancellations made fewer than 7 days before the event [Doc_11 §1].",
    ],
)
def test_a_negative_statement_about_the_world_is_still_a_claim(sentence):
    g = grounder()
    g.process(sentence + " ")
    assert g.generated == 1


def test_a_claim_citing_the_wrong_block_is_moved_to_the_block_that_states_it():
    g = grounder()
    out = g.process("The catering budget for a customer workshop is INR 900 per attendee per day [Doc_11 §1]. ")
    assert out.claim is not None and out.claim.chunk_ids == ("c2",)
    assert out.text.strip().endswith("[Doc_8 §1].")
    assert g.recited == 1 and g.demoted == 0 and g.support_rate == 1.0


def test_a_wrongly_cited_claim_no_block_states_is_still_withheld():
    g = grounder()
    out = g.process("Parking at the venue costs 200 rupees per car per day [Doc_11 §1]. ")
    assert out.claim is None and g.recited == 0 and g.demoted == 1


def test_an_opening_connective_does_not_change_what_a_claim_asserts():
    pytest.importorskip("torch")
    from slr.config import get_settings
    from slr.synthesis.grounding import load_nli

    try:
        verifier = load_nli(get_settings().nli_model)
    except Exception as exc:  # no model cache on this machine
        pytest.skip(f"NLI model unavailable: {exc}")
    bare = "Cancellations made fewer than 7 calendar days before the event are not refunded."
    assert verifier.support("However, " + bare, [TEXT_A]) == verifier.support(bare, [TEXT_A])


def test_an_evidence_line_is_kept_whole_and_never_shown():
    splitter = SentenceStream()
    g = grounder()
    text = (
        'EVIDENCE: [Doc_8 §1] "The catering budget for a customer workshop is INR 900. Per attendee per day."\n'
        "The catering budget for a customer workshop is INR 900 per attendee per day [Doc_8 §1].\n"
        "EVIDENCE: NONE\n"
    )
    shipped = []
    for i in range(0, len(text), 5):
        for seg in splitter.feed(text[i : i + 5]):
            shipped.append(g.process(seg).text)
    for seg in splitter.flush():
        shipped.append(g.process(seg).text)
    body = "".join(shipped)
    assert "EVIDENCE" not in body and "Per attendee per day." not in body.split("[Doc_8 §1]")[-1]
    assert g.generated == 1 and g.supported == 1, "the quote is not a claim"
    assert g.attributions[0].startswith("[Doc_8 §1]") and g.attributions[1] == "NONE"


def test_the_cascade_only_asks_the_second_verifier_about_weak_scores():
    from slr.synthesis.grounding import CascadeVerifier

    class Fixed:
        def __init__(self, name, scores):
            self.name, self.scores, self.asked = name, scores, []

        def support(self, claim, sources, contexts=None):
            self.asked.append(list(sources))
            return [self.scores[s] for s in sources]

    first = Fixed("small", {"a": 0.9, "b": 0.3, "c": 0.6})
    second = Fixed("checker", {"b": 0.8, "c": 0.2})
    cascade = CascadeVerifier(first, second, below=0.75)
    assert cascade.support("claim", ["a", "b", "c"]) == [0.9, 0.8, 0.6], "a second reading never lowers a score"
    assert second.asked == [["b", "c"]], "a confident first reading is not re-checked"
    assert cascade.name == "small + checker"
