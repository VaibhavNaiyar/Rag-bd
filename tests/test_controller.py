from __future__ import annotations

import pytest

from slr.contracts import Decision, TranscriptChunk
from slr.controller import signals as sig
from slr.controller.base import SessionView, UtteranceState
from slr.controller.rules import RuleController
from slr.stream.simulator import chunk_utterance

ANSWER = (
    "Cancellations made 15 or more calendar days before the event receive a full refund "
    "of the room charge [Doc_11 §1]. Cancellations made fewer than 7 days are not refunded [Doc_11 §1]."
)


@pytest.fixture
def controller(index, settings):
    return RuleController(index, index.embedder, settings)


async def feed(controller, text: str, session: SessionView | None = None):
    """Stream an utterance through the controller; return every verdict."""
    state = UtteranceState(session=session or SessionView())
    verdicts = []
    at = 0
    for piece, delay in chunk_utterance(text):
        at += delay
        verdicts.append(await controller.observe(TranscriptChunk(piece, at), state))
    final = await controller.finalize(state)
    return verdicts, final, state


async def test_retrieval_fires_before_the_utterance_ends(controller):
    text = "I need to plan a customer workshop in Pune for 30 people, and I need the cancellation policy."
    verdicts, final, _ = await feed(controller, text)
    triggers = [i for i, v in enumerate(verdicts) if v.query]
    assert triggers, "no provisional retrieval was launched"
    assert triggers[0] < len(verdicts) - 1, "retrieval only fired on the last chunk"
    assert verdicts[triggers[0]].decision is Decision.RETRIEVE
    assert final.decision is Decision.RETRIEVE


async def test_incomplete_opening_does_not_trigger(controller):
    verdicts, _, _ = await feed(controller, "I need to")
    assert all(v.decision is Decision.WAIT for v in verdicts)
    assert verdicts[0].reason == "insufficient_content"


async def test_presentation_only_turn_is_suppressed_and_searches_nothing(controller):
    session = SessionView(has_answer=True, previous_utterance="What is the cancellation policy?", previous_answer_text=ANSWER)
    verdicts, final, _ = await feed(controller, "Please repeat your last answer in two bullets.", session)
    assert final.decision is Decision.SUPPRESS
    assert final.reason == "presentation_restructure"
    assert not any(v.query for v in verdicts), "a suppressed turn issued a vector query"


@pytest.mark.parametrize(
    "text",
    [
        "Can you make that shorter?",
        "Summarise that in one line.",
        "Could you rephrase that please?",
        "Repeat what you just said.",
        "Put that in one sentence.",
        "Say that again as a numbered list.",
        "Summarize what you just said.",
        "Can you rephrase that more simply?",
    ],
)
async def test_presentation_lexicon(controller, text):
    session = SessionView(has_answer=True, previous_utterance="cancellation policy", previous_answer_text=ANSWER)
    _, final, _ = await feed(controller, text, session)
    assert final.decision is Decision.SUPPRESS


async def test_presentation_is_suppressed_even_when_the_last_answer_verified_nothing(controller):
    """An answer whose every claim was withheld is still an answer: reformatting it must not search."""
    session = SessionView(has_answer=True, previous_utterance="cancellation policy", previous_answer_text="")
    verdicts, final, _ = await feed(controller, "Say that in two bullets.", session)
    assert final.decision is Decision.SUPPRESS
    assert not any(v.query for v in verdicts), "a presentation-only turn launched a search"


async def test_new_content_defeats_suppression(controller):
    """'shorter' plus a new topic is a new request, not a reformat."""
    session = SessionView(has_answer=True, previous_utterance="cancellation policy", previous_answer_text=ANSWER)
    _, final, _ = await feed(controller, "Make that shorter and tell me the catering budget per attendee.", session)
    assert final.decision is not Decision.SUPPRESS


async def test_late_detail_routes_to_refine(controller, index):
    previous = "Summarize the travel reimbursement rule for an employee trip."
    session = SessionView(
        has_answer=True,
        previous_utterance=previous,
        previous_vec=index.embedder.embed([previous], kind="text")[0],
        previous_answer_text="Employees are reimbursed for pre-approved travel [Doc_10 §1].",
    )
    _, final, _ = await feed(controller, "The trip was international and the booking was made after travel.", session)
    assert final.decision is Decision.REFINE
    assert final.reason == "late_constraint"


async def test_a_self_correction_refines_even_with_little_overlap(controller, index):
    """'Sorry, I meant X' can only narrow the previous request, whatever X shares with it."""
    previous = "What is the venue cancellation policy?"
    session = SessionView(
        has_answer=True,
        previous_utterance=previous,
        previous_vec=index.embedder.embed([previous], kind="text")[0],
        previous_answer_text=ANSWER,
    )
    _, final, _ = await feed(controller, "Sorry, I meant for the Pune one.", session)
    assert final.decision is Decision.REFINE


async def test_an_apology_before_a_new_question_is_not_a_refinement(controller, index):
    previous = "What is the venue cancellation policy?"
    session = SessionView(
        has_answer=True,
        previous_utterance=previous,
        previous_vec=index.embedder.embed([previous], kind="text")[0],
        previous_answer_text=ANSWER,
    )
    _, final, _ = await feed(controller, "Sorry, how many days of annual leave do employees get?", session)
    assert final.decision is Decision.RETRIEVE


async def test_a_new_question_is_not_a_refinement(controller, index):
    previous = "What is the venue cancellation policy?"
    session = SessionView(
        has_answer=True,
        previous_utterance=previous,
        previous_vec=index.embedder.embed([previous], kind="text")[0],
        previous_answer_text=ANSWER,
    )
    _, final, _ = await feed(controller, "How many days of annual leave do employees get?", session)
    assert final.decision is Decision.RETRIEVE


async def test_chitchat_needs_no_retrieval(controller):
    _, final, _ = await feed(controller, "Okay thanks, that is great.")
    assert final.decision is Decision.SUPPRESS
    assert final.reason == "no_information_need"


async def test_topic_shift_cancels_the_provisional_search(controller):
    state = UtteranceState(session=SessionView())
    at = 0
    launched = False
    for piece, delay in chunk_utterance("I need the cancellation policy for venues in Pune, and"):
        at += delay
        verdict = await controller.observe(TranscriptChunk(piece, at), state)
        if verdict.query:
            launched = True
            state.active_provisional = "p1"
    assert launched, "the fixture utterance never triggered a provisional search"

    # the speaker abandons that thought and starts a different one
    shift = await controller.observe(
        TranscriptChunk("actually forget that, what encryption do laptops need", at + 800), state
    )
    assert shift.cancel is True and shift.cancel_reason == "topic_shift"
    assert shift.decision is Decision.WAIT
    # after a cancellation the whole utterance is searchable again
    assert state.covered_words == 0


def test_stability_rises_as_meaning_settles(index):
    prefixes = [
        "I need to",
        "I need to plan a customer workshop",
        "I need to plan a customer workshop in Pune for 30 people",
        "I need to plan a customer workshop in Pune for 30 people, and I need",
    ]
    vecs = [index.embedder.embed([p], kind="text")[0] for p in prefixes]
    early = sig.stability(vecs[:2])
    late = sig.stability(vecs[2:])
    assert late > early, "meaning did not settle as the utterance completed"


def test_salience_is_corpus_grounded(index):
    assert index.is_salient("reimbursement")
    assert not index.is_salient("policy"), "a word this corpus uses everywhere is not an entity"
