from __future__ import annotations

import pytest

from slr.contracts import Chunk, Hit
from slr.retrieval.context import EvidencePackage, assemble
from slr.contracts import SubQuery
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
    assert "retrieved for: cancellation" in pkg.text
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
