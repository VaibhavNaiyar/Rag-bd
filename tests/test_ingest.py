from __future__ import annotations

import json

from slr.contracts import Document
from slr.ingest.chunker import chunk_document
from slr.ingest.loader import load_corpus, parse_transcript

DOC = """# Travel Policy

Intro paragraph about the policy.

## Domestic travel

Hotels up to 7500 per night in metro cities.

## International travel

Approval needed from the department head.

### Currency

Receipts must be verified against the card statement.
"""


def doc(text: str, **kw) -> Document:
    base = dict(doc_id="d1", label="Doc_1", title="Travel Policy", source="t.md", kind="document", text=text)
    return Document(**{**base, **kw})


def test_sections_become_outline_numbered_citations():
    chunks = chunk_document(doc(DOC))
    sections = [c.section for c in chunks]
    assert sections == ["0", "1", "2", "2.1"]
    assert chunks[1].citation == "[Doc_1 §1]"
    assert "Domestic travel" in chunks[1].heading
    # the H1 is the document title, not a numbered section
    assert all("Travel Policy" in c.heading for c in chunks)


def test_every_citation_marker_resolves_to_exactly_one_chunk():
    # two sections with the same name must not produce the same marker
    text = "# T\n\n## Rules\n\nAlpha text here.\n\n## Rules\n\nBeta text here.\n"
    chunks = chunk_document(doc(text))
    assert len({c.citation for c in chunks}) == len(chunks)


def test_long_section_splits_into_parts_and_keeps_markers_short():
    body = " ".join(f"Sentence number {i} about reimbursement rules." for i in range(120))
    chunks = chunk_document(doc(f"# T\n\n## Long\n\n{body}\n"), target=60, hard_max=80)
    assert len(chunks) > 1
    assert [c.section for c in chunks] == [f"1 p{i + 1}" for i in range(len(chunks))]
    assert all(len(c.section) <= 24 and "]" not in c.section for c in chunks)


def test_transcripts_split_on_speaker_turns_never_mid_utterance():
    text = "\n".join(f"[00:0{i}:00] Priya: This is turn number {i} with several words in it." for i in range(6))
    parsed = parse_transcript(text)
    assert parsed and len(parsed) == 6
    d = doc(text, kind="transcript", title="Sync")
    chunks = chunk_document(Document(**{**d.__dict__, "turns": tuple(parsed)}), target=25)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.count("Priya:") >= 1
        # a turn is never cut in half
        for line in c.text.splitlines():
            assert line.endswith("in it.")
    assert chunks[0].section.startswith("Priya @")


def test_loader_reads_jsonl_json_and_skips_unsupported(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"title": "Vendor A", "text": "Vendor A delivers catering."}) + "\n"
        + json.dumps({"title": "Vendor B", "content": "Vendor B supplies projectors."}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(json.dumps({"name": "Policy", "body": "Refunds take 30 days."}), encoding="utf-8")
    (tmp_path / "c.md").write_text("# Notes\n\nSome note text.\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("ignore me", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert {d.title for d in docs} == {"Vendor A", "Vendor B", "Policy", "Notes"}
    assert len({d.label for d in docs}) == 4
    assert all(d.doc_id for d in docs)


def test_identical_content_is_ingested_once(tmp_path):
    (tmp_path / "a.md").write_text("# Same\n\nIdentical body text.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Same\n\nIdentical body text.\n", encoding="utf-8")
    assert len(load_corpus(tmp_path)) == 1


def test_chunk_ids_are_content_addressed_and_stable(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    first = [c.chunk_id for d in load_corpus(tmp_path) for c in chunk_document(d)]
    second = [c.chunk_id for d in load_corpus(tmp_path) for c in chunk_document(d)]
    assert first == second
