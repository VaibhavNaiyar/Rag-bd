from __future__ import annotations

import json

import pytest

from slr.decompose.decomposer import decompose, heuristic_split
from slr.telemetry.cost import UsageLedger
from tests.conftest import FakeChatModel


@pytest.fixture
def ledger():
    return UsageLedger(0.15, 0.6, 0.05)


async def run(utterance, index, settings, ledger, model=None, **kw):
    return await decompose(
        utterance,
        turn_id="t1",
        model=model,
        embedder=index.embedder,
        index=index,
        settings=settings,
        ledger=ledger,
        **kw,
    )


async def test_compound_utterance_splits_into_its_real_intents(index, settings, ledger):
    d = await run(
        "I need to plan a customer workshop in Pune for 30 people, and I need the cancellation policy and the catering options.",
        index,
        settings,
        ledger,
    )
    assert len(d.items) == 3
    joined = " | ".join(i.text.lower() for i in d.items)
    assert "cancellation" in joined and "catering" in joined and "workshop" in joined
    assert "pune" in d.items[0].text.lower()
    # each intent is searchable on its own: no bare fragments
    assert all(len(i.text.split()) >= 2 for i in d.items)


class _StubEmbedder:
    """Similarity is whatever the test says it is."""

    name, dim = "stub", 2

    def __init__(self, related: bool):
        self.related = related

    def embed(self, texts, kind="passage"):
        import numpy as np

        first = np.array([1.0, 0.0], dtype="float32")
        other = first if self.related else np.array([0.0, 1.0], dtype="float32")
        return np.stack([first] + [other] * (len(texts) - 1)) if len(texts) > 1 else np.stack([first])


def test_shared_context_is_carried_only_between_related_intents(index):
    utterance = "Book a workshop in Pune for 30 people, and what is the cancellation policy?"
    related = heuristic_split(utterance, index, _StubEmbedder(True), carry_cos=0.35)
    assert "Pune" in related[1]["text"], "related intents should inherit the shared place"

    unrelated = heuristic_split(utterance, index, _StubEmbedder(False), carry_cos=0.35)
    assert "Pune" not in unrelated[1]["text"], "an unrelated intent must not inherit context"


async def test_single_intent_yields_exactly_one_sub_query(index, settings, ledger):
    for utterance in [
        "What is the cancellation policy for workshop venues?",
        "How many people can Riverside Hall seat in a classroom layout?",
        "How long do I have to submit an expense claim after a business trip ends?",
    ]:
        d = await run(utterance, index, settings, ledger)
        assert len(d.items) == 1, f"over-fragmented: {utterance!r} -> {[i.text for i in d.items]}"


async def test_over_fragmentation_guard_merges_near_duplicates(index, settings, ledger):
    model = FakeChatModel(
        completions=[
            json.dumps(
                {
                    "sub_queries": [
                        {"text": "cancellation policy for venues", "span": "a", "confidence": 0.9},
                        {"text": "cancellation policy for venues", "span": "b", "confidence": 0.7},
                        {"text": "venue cancellation policy", "span": "c", "confidence": 0.6},
                        {"text": "catering budget per attendee", "span": "d", "confidence": 0.8},
                    ]
                }
            )
        ]
    )
    d = await run("...", index, settings, ledger, model=model)
    assert d.method == "llm"
    assert d.merged >= 1
    texts = [i.text for i in d.items]
    assert len(texts) == len(set(texts)) and len(texts) <= settings.max_subqueries


async def test_cap_is_enforced(index, settings, ledger):
    model = FakeChatModel(
        completions=[
            json.dumps(
                {
                    "sub_queries": [
                        {"text": f"distinct topic number {n} about {w}", "span": "", "confidence": 0.9}
                        for n, w in enumerate(["catering", "cancellation", "laptops", "leave", "per diem", "venues"])
                    ]
                }
            )
        ]
    )
    d = await run("...", index, settings, ledger, model=model)
    assert len(d.items) == settings.max_subqueries
    assert d.capped == 2


async def test_a_failed_model_call_degrades_to_the_heuristic_split(index, settings, ledger):
    broken = FakeChatModel(completions=["not json at all"])
    d = await run("What is the cancellation policy and the catering budget?", index, settings, ledger, model=broken)
    assert d.method == "heuristic_fallback"
    assert len(d.items) == 2


async def test_llm_sub_queries_are_recorded_verbatim_in_the_trace(index, settings, ledger):
    model = FakeChatModel(
        completions=[json.dumps({"sub_queries": [{"text": "venue capacity Pune 30", "span": "for 30 people", "confidence": 0.95}]})]
    )
    d = await run("...", index, settings, ledger, model=model)
    assert d.raw[0]["span"] == "for 30 people"
    assert d.items[0].confidence == 0.95


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("What is the cancellation policy?", 1),
        ("What is the cancellation policy and what is the catering budget?", 2),
        ("Tell me the per diem for Tokyo, the sick leave allowance, and the laptop encryption rule.", 3),
        ("How do I book a venue?", 1),
    ],
)
def test_heuristic_split_counts(utterance, expected, index):
    assert len(heuristic_split(utterance, index)) == expected


def test_heuristic_split_strips_filler_and_keeps_numbers(index):
    parts = heuristic_split("I need to know the room rate for 30 people at Riverside Hall", index)
    assert len(parts) == 1
    text = parts[0]["text"]
    assert "30" in text and "Riverside" in text
    assert not text.lower().startswith("i need")


class _Orthogonal:
    """Every query looks unrelated by embedding, so only the lexical rules can merge."""

    def embed(self, texts, kind="query"):
        import numpy as np

        return np.eye(len(texts))


@pytest.mark.parametrize(
    "texts, kept",
    [
        # a query that only adds context to another is the same need; the specific one survives
        (["cancellation policy Pune", "cancellation policy customer workshop venue Pune 30 people"],
         ["cancellation policy customer workshop venue Pune 30 people"]),
        # different years are different readings, however similar the words
        (["World Cup soccer winner 2018", "World Cup soccer winner 2022"],
         ["World Cup soccer winner 2018", "World Cup soccer winner 2022"]),
        # different qualifiers are different readings
        (["Darth Vader original trilogy voice actor", "Darth Vader prequel trilogy voice actor"],
         ["Darth Vader original trilogy voice actor", "Darth Vader prequel trilogy voice actor"]),
    ],
)
def test_guard_merges_a_restated_need_but_never_a_distinct_reading(settings, texts, kept):
    from slr.decompose.decomposer import guard

    items = [{"text": t, "span": "", "confidence": 0.9} for t in texts]
    out, merged, _ = guard(items, _Orthogonal(), settings)
    assert [i["text"] for i in out] == kept
    assert merged == len(texts) - len(kept)
