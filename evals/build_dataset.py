"""ASQA -> corpus, gold labels, timed transcripts and fixtures.

    python -m evals.build_dataset              # build from the vendored snapshot (offline)
    python -m evals.build_dataset --refresh    # re-download din0s/asqa and re-vendor it

Dataset: ASQA, ``din0s/asqa`` on Hugging Face, Apache-2.0, 5,301 samples
(4,353 train / 948 dev). Each ``ambiguous_question`` comes pre-decomposed into
the disambiguated sub-questions it implies (``qa_pairs``), each with its own
grounding passage. That one record gives the utterance to stream, the gold
decomposition for G3, the passages to retrieve and fuse, and the citations G4
is checked against.

Four artifacts:

1. ``data/corpus/``        every passage across ALL samples (train + dev), one
                           markdown file per Wikipedia page, one ``## Passage``
                           section (= one chunk) per passage. Pooling every
                           sample is what puts real distractors next to each
                           gold passage.
2. ``evals/gold.jsonl``    per sample: gold sub-questions, the passage each
                           answer came from, and the reference long answer.
3. ``evals/transcripts/``  per fixture case: every turn split into 3-5 word
                           chunks at 150 wpm (400 ms/word), each with ``atMs``,
                           and the ``utterance_end_ms`` G2 measures against.
4. ``evals/fixtures/``     ``compound/``, ``single/``, ``late_detail/`` and
                           ``suppression/`` fixtures (``corpus: "asqa"``), built
                           from the dev split only.

Three properties of the real data shape the choices below. Each one is
reported in the build summary.

* ``qa_pairs[].context`` is the literal string "No context provided" for
  ~56% of pairs, and ``wikipages`` holds only a title and a URL. The corpus
  therefore also takes ``annotations[].knowledge[].content``, the Wikipedia
  text the annotators cited, filed under its own ``wikipage``. Without it,
  half of the sub-questions would have no grounding text anywhere in the
  corpus.
* No ASQA sample has exactly one qa_pair: every split has 2-6. The
  single-intent control is therefore built from ONE disambiguated
  sub-question (``qa_pairs[i].question``), which by ASQA's construction has a
  single reading and a single grounding passage.
* A gold passage is the pair's own ``context`` when it exists
  (``source: "qa_context"``). Otherwise it is an annotator-cited knowledge
  passage from the same sample that contains a short answer
  (``source: "knowledge_answer_match"``). Pairs with neither carry no
  retrieval label, and recall skips them instead of counting them as misses.

Everything written under ``evals/`` is test data. ``src/`` never imports it;
``tests/test_compliance.py`` enforces that.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evals" / "data" / "asqa"
CORPUS = ROOT / "data" / "corpus"
GOLD = ROOT / "evals" / "gold.jsonl"
TRANSCRIPTS = ROOT / "evals" / "transcripts"
FIXTURES = ROOT / "evals" / "fixtures"

SPLITS = ("train", "dev")
NO_CONTEXT = "no context provided"
SEED = 20260918
#: Fixtures per family. ~130 turns at speaking pace is ~15 minutes of replay.
DEFAULT_COUNTS = {"compound": 40, "single": 20, "late_detail": 15, "suppression": 15}

#: Speech timing. Must match slr.stream.simulator (150 wpm, 3/4/5-word groups)
#: so a replayed transcript and a typed request reach the controller in the
#: same shape. A chunk arrives once its words have been spoken; the
#: end-of-utterance signal follows the last word after an endpointing pause.
MS_PER_WORD = 400
CHUNK_WORDS = (3, 4, 5)
END_PAUSE_MS = 250

#: Turn 2 of a late_detail case: the speaker narrows the question they asked.
LATE_DETAIL_TEMPLATES = [
    "Actually, I meant {0}.",
    "Sorry, I mean {0}.",
    "Oh wait, I meant {0}.",
]

#: Turn 2 of a suppression case: re-present the last answer, no new information need.
PRESENTATION_TEMPLATES = [
    "Say that in two bullets.",
    "Can you make that shorter?",
    "Put that in one sentence.",
    "Repeat that more slowly.",
    "Read the last answer again.",
    "Summarize what you just said.",
    "Say that again as a numbered list.",
    "Can you rephrase that more simply?",
]

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or",
    "is", "was", "are", "were", "be", "been", "do", "does", "did", "that", "this", "it", "its",
    "who", "what", "when", "where", "which", "how", "why", "whom", "whose",
}  # fmt: skip
_WH = {"who", "what", "when", "where", "which", "how", "why", "whom", "whose", "in", "is", "was", "did", "does"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def refresh_snapshot() -> None:
    """Download din0s/asqa and vendor it as gzipped JSONL, so builds after this need no network."""
    from datasets import load_dataset  # optional dependency: pip install -e .[dataset]

    ds = load_dataset("din0s/asqa")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with gzip.open(SNAPSHOT / f"{split}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for record in ds[split]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_split(split: str) -> list[dict[str, Any]]:
    path = SNAPSHOT / f"{split}.jsonl.gz"
    if not path.exists():
        refresh_snapshot()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _has_context(pair: dict[str, Any]) -> bool:
    ctx = _clean(pair.get("context"))
    return bool(ctx) and ctx.lower() != NO_CONTEXT


def _question(pair: dict[str, Any]) -> str:
    """A handful of ASQA questions list alternative phrasings joined by '|'; keep the first."""
    return _clean(pair["question"].split("|")[0])


def _slug(title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:70] or "page"
    return f"{base}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:6]}"


def _answer_in(answer: str, text: str) -> bool:
    answer = answer.strip()
    if len(answer) < 3:
        return False
    return re.search(r"(?<!\w)" + re.escape(answer.lower()) + r"(?!\w)", text.lower()) is not None


def _page_title(raw: str | None, wikipages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Resolve a (possibly truncated, '...'-suffixed) page name against the sample's wikipages."""
    name = _clean(raw) or "Unsourced"
    stem = name[:-3].strip() if name.endswith("...") else name
    for page in wikipages or []:
        title = _clean(page.get("title"))
        if title and (title == name or title.startswith(stem)):
            return title, page.get("url")
    return stem, None


# --------------------------------------------------------------------------
# 1. Corpus
# --------------------------------------------------------------------------


class Corpus:
    """Pages keyed by title; each passage is stored once and addressable by (doc, section)."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}
        self._where: dict[str, tuple[str, int]] = {}  # passage text -> (title, 1-based index)

    def add(self, title: str, url: str | None, text: str) -> dict[str, Any]:
        page = self.pages.setdefault(title, {"url": url, "passages": []})
        if url and not page["url"]:
            page["url"] = url
        if text not in self._where:
            page["passages"].append(text)
            self._where[text] = (title, len(page["passages"]))
        return self.ref(text)

    def ref(self, text: str) -> dict[str, Any]:
        title, n = self._where[text]
        return {"doc": f"{_slug(title)}.md", "title": title, "section": f"Passage {n}", "text": text}

    def write(self, out: Path) -> dict[str, int]:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for title, page in self.pages.items():
            lines = [f"# {title}", ""]
            for i, passage in enumerate(page["passages"], start=1):
                lines += [f"## Passage {i}", "", passage, ""]
            (out / f"{_slug(title)}.md").write_text("\n".join(lines), encoding="utf-8")
        return {"documents": len(self.pages), "passages": len(self._where)}


def build_corpus(records: list[dict[str, Any]]) -> Corpus:
    corpus = Corpus()
    for record in records:
        pages = record.get("wikipages") or []
        for pair in record["qa_pairs"]:
            if _has_context(pair):
                title, url = _page_title(pair.get("wikipage"), pages)
                corpus.add(title, url, _clean(pair["context"]))
        for annotation in record.get("annotations") or []:
            for k in annotation.get("knowledge") or []:
                if _clean(k.get("content")):
                    title, url = _page_title(k.get("wikipage"), pages)
                    corpus.add(title, url, _clean(k["content"]))
    return corpus


# --------------------------------------------------------------------------
# 2. Gold labels
# --------------------------------------------------------------------------


def gold_record(record: dict[str, Any], split: str, corpus: Corpus) -> dict[str, Any]:
    knowledge = [
        _clean(k["content"])
        for a in record.get("annotations") or []
        for k in a.get("knowledge") or []
        if _clean(k.get("content"))
    ]
    subs = []
    for pair in record["qa_pairs"]:
        answers = [a for a in pair.get("short_answers") or [] if _clean(a)]
        if _has_context(pair):
            passages = [{**corpus.ref(_clean(pair["context"])), "source": "qa_context"}]
        else:
            passages = [
                {**corpus.ref(text), "source": "knowledge_answer_match"}
                for text in dict.fromkeys(knowledge)
                if any(_answer_in(a, text) for a in answers)
            ]
        subs.append({"question": _question(pair), "short_answers": answers, "passages": passages})
    long_answers = [_clean(a.get("long_answer")) for a in record.get("annotations") or [] if _clean(a.get("long_answer"))]
    return {
        "sample_id": record["sample_id"],
        "split": split,
        "ambiguous_question": _clean(record["ambiguous_question"]),
        "sub_questions": subs,
        "wikipages": [{"title": p.get("title"), "url": p.get("url")} for p in record.get("wikipages") or []],
        "long_answer": long_answers[0] if long_answers else "",
        "long_answers": long_answers,
    }


# --------------------------------------------------------------------------
# 3. Transcripts
# --------------------------------------------------------------------------


def timed_turn(text: str) -> dict[str, Any]:
    """3/4/5-word chunks; each chunk's atMs is when its last word has been spoken."""
    words = text.split()
    chunks, i, g = [], 0, 0
    while i < len(words):
        piece = words[i : i + CHUNK_WORDS[g % len(CHUNK_WORDS)]]
        i += len(piece)
        g += 1
        chunks.append({"text": " ".join(piece), "atMs": i * MS_PER_WORD})
    speech_end = len(words) * MS_PER_WORD
    return {
        "text": text,
        "chunks": chunks,
        "speech_end_ms": speech_end,
        "end_pause_ms": END_PAUSE_MS,
        "utterance_end_ms": speech_end + END_PAUSE_MS,
    }


# --------------------------------------------------------------------------
# 4. Fixtures
# --------------------------------------------------------------------------


def late_condition(ambiguous: str, sub_question: str) -> str | None:
    """The words a disambiguated sub-question inserts into the ambiguous question.

    Only a single pure insertion with a content word is kept ("... end in
    colorado" -> "... end in colorado in 2017" gives "in 2017"; a dangling
    function word at the end is dropped). Rewordings
    give fragments like "is the main" that no speaker would say, so they are
    dropped rather than templated.
    """
    tokenize = lambda s: re.findall(r"[\w'’.\-–]+", s)  # noqa: E731
    a, b = tokenize(ambiguous), tokenize(sub_question)
    ops = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b], autojunk=False).get_opcodes()
    changes = [op for op in ops if op[0] != "equal"]
    if len(changes) != 1 or changes[0][0] != "insert":
        return None
    span = b[changes[0][3] : changes[0][4]]
    # "a single season in" is spoken as "a single season"
    while span and span[-1].lower() in _STOP:
        span = span[:-1]
    if not 1 <= len(span) <= 6 or span[0].lower() in _WH:
        return None
    if not any(w.lower() not in _STOP for w in span):
        return None
    return " ".join(span)


def _labelled(sub: dict[str, Any]) -> bool:
    return bool(sub["passages"])


def _expect(subs: list[dict[str, Any]], mode: str, intents: bool = True) -> dict[str, Any]:
    expect: dict[str, Any] = {"mode": mode}
    if intents:
        expect["gold_sub_intents"] = [s["question"] for s in subs]
    expect["gold_passages"] = sorted({p["text"] for s in subs for p in s["passages"]})
    expect["gold_answers"] = sorted({a for s in subs for a in s["short_answers"]})
    return expect


def build_cases(gold: list[dict[str, Any]], counts: dict[str, int], seed: int) -> list[dict[str, Any]]:
    """Choose dev samples for each family. Families never share a sample."""
    rng = random.Random(seed)
    pool = [g for g in gold if g["split"] == "dev"]
    rng.shuffle(pool)
    used: set[str] = set()
    cases: list[dict[str, Any]] = []

    def take(family: str, eligible, make) -> None:
        n = 0
        for g in pool:
            if n >= counts[family]:
                break
            if g["sample_id"] in used or not eligible(g):
                continue
            used.add(g["sample_id"])
            n += 1
            cases.append({"family": family, "id": f"asqa_{family}_{n:02d}", "sample": g, **make(g, n)})

    # compound: the ambiguous question, scored against every gold sub-question.
    # At least two sub-questions must have a passage so retrieval recall means something.
    take(
        "compound",
        lambda g: sum(_labelled(s) for s in g["sub_questions"]) >= 2,
        lambda g, n: {
            "turns": [(g["ambiguous_question"], _expect(g["sub_questions"], "retrieve"))],
            "description": f"ASQA ambiguous question with {len(g['sub_questions'])} gold sub-questions.",
        },
    )

    # single: one disambiguated sub-question, which has exactly one reading.
    def single(g, n):
        sub = next(s for s in g["sub_questions"] if s["passages"] and s["passages"][0]["source"] == "qa_context")
        return {
            "turns": [(sub["question"], _expect([sub], "retrieve"))],
            "description": "One ASQA disambiguated sub-question: a single-intent control for over-fragmentation. "
            "ASQA has no samples with a single qa_pair, so the control is built from one sub-question.",
        }

    take("single", lambda g: any(s["passages"] and s["passages"][0]["source"] == "qa_context" for s in g["sub_questions"]), single)

    # late_detail: ask the ambiguous question, then narrow it with the condition
    # that ASQA's own disambiguation adds.
    def late_pick(g):
        for s in g["sub_questions"]:
            cond = late_condition(g["ambiguous_question"], s["question"])
            if cond and _labelled(s):
                return s, cond
        return None

    def late(g, n):
        sub, cond = late_pick(g)
        follow = LATE_DETAIL_TEMPLATES[(n - 1) % len(LATE_DETAIL_TEMPLATES)].format(cond)
        expect = {**_expect([sub], "refine", intents=False), "refined_question": sub["question"], "condition": cond}
        return {
            "turns": [(g["ambiguous_question"], {"mode": "retrieve"}), (follow, expect)],
            "description": f"Ambiguous question, then the disambiguating condition from ASQA's annotation ({cond!r}).",
        }

    take("late_detail", lambda g: late_pick(g) is not None, late)

    # suppression: a normal turn, then a presentation-only turn.
    take(
        "suppression",
        lambda g: True,
        lambda g, n: {
            "turns": [
                (g["ambiguous_question"], {"mode": "retrieve"}),
                (PRESENTATION_TEMPLATES[(n - 1) % len(PRESENTATION_TEMPLATES)], {"mode": "suppress"}),
            ],
            "description": "A normal ASQA question, then a presentation-only follow-up that must not retrieve.",
        },
    )
    return cases


def write_case(case: dict[str, Any]) -> None:
    timed = [timed_turn(text) for text, _ in case["turns"]]
    transcript_path = TRANSCRIPTS / f"{case['id']}.json"
    transcript = {
        "id": case["id"],
        "family": case["family"],
        "sample_id": case["sample"]["sample_id"],
        "wpm": 60_000 // MS_PER_WORD,
        "ms_per_word": MS_PER_WORD,
        "turns": timed,
    }
    transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fixture = {
        "id": case["id"],
        "corpus": "asqa",
        "sample_id": case["sample"]["sample_id"],
        "description": case["description"],
        "transcript": transcript_path.relative_to(ROOT).as_posix(),
        "turns": [
            {
                "utterance": t["text"],
                "chunks": t["chunks"],
                "end_pause_ms": t["end_pause_ms"],
                "utterance_end_ms": t["utterance_end_ms"],
                "expect": expect,
            }
            for t, (_, expect) in zip(timed, case["turns"])
        ],
    }
    path = FIXTURES / case["family"] / f"{case['id']}.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _reset_outputs(families: list[str]) -> None:
    """Remove only what this script generates; hand-written enterprise fixtures stay."""
    if TRANSCRIPTS.exists():
        shutil.rmtree(TRANSCRIPTS)
    TRANSCRIPTS.mkdir(parents=True)
    for family in families:
        folder = FIXTURES / family
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("asqa_*.json"):
            old.unlink()


# --------------------------------------------------------------------------


def write_examples(train: list[dict[str, Any]], out: Path) -> int:
    """The decomposer's few-shot bank: each TRAIN question with the readings ASQA split it into.

    Fixtures are drawn from the dev split only, so no fixture question can be its own
    example; the engine also skips any example near-identical to the question it is
    decomposing. Nearest-neighbour examples from the training split are how Tree of
    Clarifications (Kim et al., EMNLP 2023) and DIVA (In et al., NAACL 2025) prompt for
    ASQA's readings.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in train:
            readings = list(dict.fromkeys(_question(p) for p in r["qa_pairs"] if _question(p)))
            if len(readings) < 2:
                continue
            record = {"question": _clean(r["ambiguous_question"]), "readings": readings}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Each reading is itself a question with one answer, so the bank also
            # teaches what a question that must NOT be split looks like.
            for reading in readings:
                fh.write(json.dumps({"question": reading, "readings": [reading]}, ensure_ascii=False) + "\n")
            n += 1 + len(readings)
    return n


def build(counts: dict[str, int], seed: int = SEED, corpus_out: Path = CORPUS) -> dict[str, Any]:
    records = {split: load_split(split) for split in SPLITS}
    everything = [r for split in SPLITS for r in records[split]]

    corpus = build_corpus(everything)
    corpus_info = corpus.write(corpus_out)
    examples = write_examples(records["train"], corpus_out.parent / "decompose_examples.jsonl")

    gold = [gold_record(r, split, corpus) for split in SPLITS for r in records[split]]
    with GOLD.open("w", encoding="utf-8") as fh:
        for g in gold:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    _reset_outputs(list(counts))
    cases = build_cases(gold, counts, seed)
    for case in cases:
        write_case(case)

    subs = [s for g in gold for s in g["sub_questions"]]
    sources = Counter(s["passages"][0]["source"] if s["passages"] else "unlabelled" for s in subs)
    return {
        "samples": {split: len(records[split]) for split in SPLITS},
        "qa_pairs_per_sample": dict(sorted(Counter(len(r["qa_pairs"]) for r in everything).items())),
        "corpus": {**corpus_info, "dir": str(corpus_out)},
        "decompose_examples": examples,
        "gold": {"samples": len(gold), "sub_questions": len(subs), "label_source": dict(sources)},
        "fixtures": dict(Counter(c["family"] for c in cases)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download din0s/asqa and re-vendor the snapshot")
    for family, n in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{family.replace('_', '-')}", type=int, default=n, help=f"{family} fixtures")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--corpus-out", default=str(CORPUS))
    args = parser.parse_args()

    if args.refresh:
        refresh_snapshot()
    counts = {family: getattr(args, family) for family in DEFAULT_COUNTS}
    print(json.dumps(build(counts, args.seed, Path(args.corpus_out)), indent=2))


if __name__ == "__main__":
    main()
