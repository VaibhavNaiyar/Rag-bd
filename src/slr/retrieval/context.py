"""The evidence package — the prompt boundary.

The synthesis prompt contains only what is built here: retrieved chunk text,
each block labelled with the citation marker that resolves to it. No session
profile, no model memory, nothing from outside the index.
"""

from __future__ import annotations

from dataclasses import dataclass

from slr.contracts import Hit, SubQuery


@dataclass
class EvidencePackage:
    hits: list[Hit]  # the retrieved set; citations are validated against it
    text: str
    truncated: int = 0

    @property
    def citation_map(self) -> dict[str, Hit]:
        return {h.chunk.citation.lower(): h for h in self.hits}


def assemble(hits: list[Hit], sub_queries: list[SubQuery], char_budget: int) -> EvidencePackage:
    labels = {sq.id: sq.text for sq in sub_queries}
    blocks: list[str] = []
    used = 0
    included: list[Hit] = []
    dropped = 0
    for hit in hits:
        for_what = "; ".join(labels.get(s, s) for s in hit.sub_query_ids if s in labels) or "general"
        block = (
            f"{hit.chunk.citation}\n"
            f"source: {hit.chunk.heading}\n"
            f"retrieved for: {for_what}\n"
            f"{hit.chunk.text}\n"
        )
        if included and used + len(block) > char_budget:
            dropped += 1
            continue
        blocks.append(block)
        used += len(block)
        included.append(hit)
    return EvidencePackage(hits=included, text="\n---\n".join(blocks), truncated=dropped)
