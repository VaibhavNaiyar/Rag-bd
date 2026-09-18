"""The model-backed paths, driven by a scripted provider.

These cover what an API key would exercise: JSON decomposition, streamed
synthesis with citation validation, and claim-level refinement. The fake model
answers out of the evidence the engine actually retrieved — quoting a block and
citing that block's marker — so the assertions are about engine behaviour, not
about fixed chunk ids.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import FakeChatModel, quote


@pytest.fixture
def llm(engine, settings):
    """Engine clone whose LLM is scripted per test."""
    return engine.with_settings(settings.with_overrides(llm="openai"))


def runner_for(clone, model, recorder):
    from slr.stream.engine import SessionRunner

    clone.model = model
    return SessionRunner(clone, recorder)


DECOMPOSE_TWO = json.dumps(
    {
        "sub_queries": [
            {"text": "venue in Pune for a 30 person workshop", "span": "workshop in Pune", "confidence": 0.9},
            {"text": "venue cancellation and refund policy", "span": "cancellation policy", "confidence": 0.9},
        ]
    }
)
DECOMPOSE_ONE = json.dumps({"sub_queries": [{"text": "travel reimbursement rule", "span": "", "confidence": 0.9}]})


async def test_model_answer_is_validated_before_it_is_streamed(llm, recorder):
    def synthesise(prompt: str) -> str:
        first, m1 = quote(prompt, 0)
        second, m2 = quote(prompt, 1)
        return (
            f"{first} {m1}.\n\n"
            f"{second} {m2}.\n"
            "Parking at the venue costs 500 rupees per car [Doc_999 §4].\n"
            "UNCERTAIN: Catering for that venue could not be verified from the retrieved documents.\n"
        )

    model = FakeChatModel(completions=[DECOMPOSE_TWO], streams=[synthesise])
    runner = runner_for(llm, model, recorder)
    await runner.start()
    await runner.replay("compound_01", 40.0)

    body = recorder.answer()
    version = recorder.one("answer.version")
    trace = runner.completed[-1]

    # the fabricated marker never reaches the user, and is counted where it occurred
    assert "Doc_999" not in body
    assert version["fabricatedCitations"] == 0
    assert trace["fabricated_citations_blocked"] == 1
    assert trace["answer"]["grounding"]["fabricated_markers"] == ["[Doc_999 §4]"]

    # the invented parking claim is withheld rather than re-cited to a lookalike chunk
    assert "500 rupees" not in body
    assert any("Not verified" in u for u in version["uncertainty"])

    # the model's own uncertainty line is carried as uncertainty, not as prose
    assert any("Catering" in u for u in version["uncertainty"])
    assert "UNCERTAIN" not in body

    assert [step for step, _ in model.calls] == ["decompose", "synthesise"], "a turn must cost two model calls"
    assert version["citationSupportRate"] >= 0.6
    assert len(version["claims"]) == 2
    assert trace["decomposition"]["method"] == "llm"


async def test_two_model_calls_per_turn_and_cost_is_recorded(llm, recorder):
    model = FakeChatModel(
        completions=[DECOMPOSE_TWO], streams=[lambda p: "{} {}.".format(*quote(p, 0))]
    )
    runner = runner_for(llm, model, recorder)
    await runner.start()
    await runner.replay("compound_01", 40.0)

    cost = recorder.one("turn.complete")["cost"]
    steps = {s["step"] for s in cost["steps"]}
    assert {"decompose", "synthesise"} <= steps
    assert cost["turnUsd"] > 0 and cost["turnTokens"] == 560
    entries = runner.completed[-1]["cost"]["entries"]
    assert [e for e in entries if e["kind"] == "llm"], "no LLM usage was recorded"


async def test_refinement_keeps_unaffected_claims_and_rewrites_the_affected_one(llm, recorder):
    def answer_v1(prompt: str) -> str:
        a, m1 = quote(prompt, 0)
        b, m2 = quote(prompt, 1)
        return f"{a} {m1}.\n{b} {m2}.\n"

    plan = json.dumps({"delta_queries": [{"text": "international late booking exception", "for": "t1_sq1"}]})

    def answer_v2(prompt: str) -> str:
        edited, m1 = quote(prompt, 1)
        added, m2 = quote(prompt, 2)
        return (
            "KEEP c1\n"
            f"EDIT c2: {edited} {m1}.\n"
            f"ADD: {added} {m2}.\n"
            "UNCERTAIN: The deadline for requesting the exception is not stated.\n"
        )

    model = FakeChatModel(completions=[DECOMPOSE_ONE, plan], streams=[answer_v1, answer_v2])
    runner = runner_for(llm, model, recorder)
    await runner.start()
    await runner.replay("late_detail_01", 40.0)

    v1, v2 = recorder.of("answer.version")
    assert (v2["version"], v2["parent"]) == (2, 1)
    assert v2["preserved"] == [v1["claims"][0]["id"]]
    assert len(v2["mutated"]) == 1
    assert len(v2["claims"]) == 3, "KEEP + EDIT + ADD should yield three claims"

    # the preserved claim keeps its original text and citations
    kept = next(c for c in v2["claims"] if c["id"] in v2["preserved"])
    assert kept["text"] == v1["claims"][0]["text"]
    assert kept["chunkIds"] == v1["claims"][0]["chunkIds"]

    fusion = recorder.of("fusion.final")[-1]
    assert fusion["fullCorpusSearch"] is False
    assert [s for s, _ in model.calls] == ["decompose", "synthesise", "refine_plan", "refine"]
    assert any("deadline" in u.lower() for u in v2["uncertainty"])
    assert runner.completed[-1]["decomposition"]["affected_claims"], "no claim was identified as affected"


async def test_a_refinement_the_evidence_cannot_support_keeps_the_original_claim(llm, recorder):
    plan = json.dumps({"delta_queries": [{"text": "international travel", "for": "t1_sq1"}]})
    model = FakeChatModel(
        completions=[DECOMPOSE_ONE, plan],
        streams=[
            lambda p: "{} {}.\n".format(*quote(p, 0)),
            # an edit no chunk supports
            "EDIT c1: Employees are reimbursed twice over for international trips [Doc_1 §1].\n",
        ],
    )
    runner = runner_for(llm, model, recorder)
    await runner.start()
    await runner.replay("late_detail_01", 40.0)

    v1, v2 = recorder.of("answer.version")
    assert "twice over" not in recorder.answer(v2["turnId"])
    assert v2["preserved"] == [v1["claims"][0]["id"]], "a verified claim was replaced by an unverifiable edit"
    assert v2["mutated"] == []


async def test_restructure_turn_reuses_prior_claims_without_retrieving(llm, recorder):
    def answer_v1(prompt: str) -> str:
        a, m1 = quote(prompt, 0)
        b, m2 = quote(prompt, 1)
        return f"{a} {m1}.\n{b} {m2}.\n"

    def bullets(prompt: str) -> str:
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip().startswith(("Cancellations", "This policy", "An event"))]
        return "".join(f"- {ln}\n" for ln in lines[:2]) or "- (nothing)\n"

    model = FakeChatModel(
        completions=[json.dumps({"sub_queries": [{"text": "venue cancellation policy", "span": "", "confidence": 0.9}]})],
        streams=[answer_v1, bullets],
    )
    runner = runner_for(llm, model, recorder)
    await runner.start()
    await runner.replay("presentation_01", 40.0)

    assert [s for s, _ in model.calls] == ["decompose", "synthesise", "restructure"]
    suppressed = runner.completed[-1]
    assert suppressed["mode"] == "suppress" and suppressed["retrieval"] == []
    v2 = recorder.of("answer.version")[-1]
    assert v2["preserved"], "the reformatted answer did not carry any prior claim"
    assert v2["fabricatedCitations"] == 0
    assert recorder.answer(v2["turnId"]).lstrip().startswith("- ")


async def test_model_controller_arm_decides_per_chunk(llm, recorder, settings):
    clone = llm.with_settings(settings.with_overrides(controller="model", llm="openai"))
    verdicts = [
        json.dumps({"decision": "wait", "reason": "insufficient_content", "confidence": 0.3, "query": ""}),
        json.dumps({"decision": "retrieve", "reason": "intent_stable", "confidence": 0.8, "query": "cancellation policy"}),
    ] * 6
    model = FakeChatModel(
        completions=[*verdicts, json.dumps({"sub_queries": [{"text": "cancellation policy", "span": "", "confidence": 0.9}]})],
        streams=[lambda p: "{} {}.\n".format(*quote(p, 0))],
    )
    runner = runner_for(clone, model, recorder)
    await runner.start()
    await runner.play_turns([{"utterance": "What is the venue cancellation policy for workshops?"}], 40.0)

    trace = runner.completed[-1]
    assert trace["controller"] == "model"
    assert [d["decision"] for d in trace["decisions"]][:2] == ["wait", "retrieve"]
    assert trace["first_retrieval_ms"] is not None
    # every chunk costs a model call — which is the finding the ablation reports
    controller_calls = [s for s, _ in model.calls if s == "controller"]
    assert len(controller_calls) == len(trace["chunks"])
    assert any(e["step"] == "controller" and e["kind"] == "llm" for e in trace["cost"]["entries"])


@pytest.mark.parametrize(
    "model, reasoning",
    [("gpt-4o-mini", False), ("gpt-4.1", False), ("gpt-5-chat-latest", False), ("gpt-5.4-mini", True), ("o4-mini", True)],
)
def test_request_limits_follow_the_model_family(model, reasoning):
    """Reasoning models reject temperature and max_tokens, and need room to reason."""
    from slr.llm import REASONING_HEADROOM, OpenAIChatModel

    limits = OpenAIChatModel(model, base_url="http://localhost:1")._limits(400)
    if reasoning:
        assert limits == {"max_completion_tokens": 400 + REASONING_HEADROOM, "reasoning_effort": "low"}
    else:
        assert limits == {"temperature": 0, "max_tokens": 400}
