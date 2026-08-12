"""Section-aware text chunker for arXiv papers.

Splits parsed papers along natural section boundaries (Abstract, Introduction, Methods, etc.)
and applies a sliding window with overlap for sections exceeding max_tokens.
Excludes non-content sections like References, Bibliography, and Acknowledgments.
Preserves chunk metadata (paper_id, section_name, chunk_index, token_count).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...config import settings
from ...models.paper import Paper

# Regex for sections to exclude from chunking
EXCLUDED_SECTIONS_PATTERN = re.compile(
    r"^(references|refrence|bibliography|acknowledgements|acknowledgments|appendix|appendices)$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""

    paper_id: str
    section_name: str
    chunk_index: int
    text: str
    token_count: int

    @property
    def chunk_id(self) -> str:
        """Unique deterministic identifier for this chunk."""
        clean_section = re.sub(r"[^a-zA-Z0-9_]", "_", self.section_name.lower())[:64]
        return f"{self.paper_id}_{clean_section}_{self.chunk_index}"


class TextChunker:
    """Section-aware text chunking engine."""

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        self.max_tokens = max_tokens or min(settings.chunk_max_tokens, 200)
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Estimate token count using whitespace word splitting."""
        return len(text.split())

    @staticmethod
    def is_excluded_section(heading: str) -> bool:
        """Return True if section heading matches excluded patterns (References, etc.)."""
        clean_heading = heading.strip().lower()
        # Remove numbers like "7. References" or "References and Notes"
        clean_heading = re.sub(r"^\d+[\.\s]*", "", clean_heading).strip()
        return bool(EXCLUDED_SECTIONS_PATTERN.match(clean_heading))

    @staticmethod
    def _extract_section_name(text: str, default_section: str = "Full Text") -> str:
        """Extract exact section or subsection heading from chunk text or prefix."""
        # 1. Embedded "Section: <Heading>" prefix check
        if match := re.search(r"Section:\s*([^\n\.]+)", text):
            extracted = match.group(1).strip()
            if extracted and extracted.lower() != "full text":
                return extracted

        # 2. Check for leading numbered or standard section headings
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:3]:
            # Numbered headings like "4.2 Method" or "5.1 Hierarchical Cluster Routing"
            if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
                return line
            # Standard section names
            if line in (
                "Introduction", "Related Work", "Method", "Methods", "Methodology",
                "Architecture", "Experiments", "Experimental Evaluation", "Results",
                "Discussion", "Conclusion", "Conclusions", "Supplementary Material"
            ):
                return line

        return default_section

    def _split_text(
        self,
        text: str,
        paper_id: str,
        section_name: str,
        start_chunk_index: int = 0,
    ) -> list[Chunk]:
        """Split a single section's text into one or more overlapping chunks."""
        words = text.split()
        if not words:
            return []

        # If section fits in max_tokens, return single chunk
        if len(words) <= self.max_tokens:
            chunk_sec = self._extract_section_name(text, default_section=section_name) if section_name == "Full Text" else section_name
            return [
                Chunk(
                    paper_id=paper_id,
                    section_name=chunk_sec,
                    chunk_index=start_chunk_index,
                    text=text,
                    token_count=len(words),
                )
            ]

        # Sliding window split
        chunks: list[Chunk] = []
        step = max(1, self.max_tokens - self.overlap_tokens)
        current_idx = start_chunk_index

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.max_tokens]
            chunk_text = " ".join(chunk_words)
            chunk_sec = self._extract_section_name(chunk_text, default_section=section_name) if section_name == "Full Text" else section_name
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    section_name=chunk_sec,
                    chunk_index=current_idx,
                    text=chunk_text,
                    token_count=len(chunk_words),
                )
            )
            current_idx += 1

            # Stop if this window reached the end of the words list
            if i + self.max_tokens >= len(words):
                break

        return chunks

    @staticmethod
    def extract_sections_from_text(full_text: str) -> list[dict[str, str]]:
        """Parse full text into sections using regex matching on numbered and standard headings."""
        if not full_text:
            return []

        lines = full_text.split("\n")
        sections: list[dict[str, str]] = []
        current_heading = "Introduction"
        current_lines: list[str] = []

        heading_pattern = re.compile(
            r"^(\d+(\.\d+)*\s+[A-Z][^\n]{2,80}|Abstract|Introduction|Related Work|Methods?|Architecture|Experiments?|Results|Discussion|Conclusions?|Supplementary Material)$"
        )

        for line in lines:
            stripped = line.strip()
            # Ignore table/figure numbers or address lines that look like numbers
            if heading_pattern.match(stripped) and len(stripped) < 80 and not stripped.endswith((".", ",", ";")):
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append({"heading": current_heading, "text": sec_text})
                    current_lines = []
                current_heading = stripped
            else:
                current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append({"heading": current_heading, "text": sec_text})

        return sections

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        """Produce section-aware chunks for a given Paper instance."""
        if paper.parse_status != "success" and not paper.abstract:
            return []

        chunks: list[Chunk] = []
        global_chunk_index = 0

        # 1. Abstract is always included as its own section chunk
        if paper.abstract and paper.abstract.strip():
            abs_chunks = self._split_text(
                text=paper.abstract.strip(),
                paper_id=paper.arxiv_id,
                section_name="Abstract",
                start_chunk_index=global_chunk_index,
            )
            chunks.extend(abs_chunks)
            global_chunk_index += len(abs_chunks)

        # 2. Iterate through parsed PDF sections
        sections = paper.sections if paper.sections else self.extract_sections_from_text(paper.full_text or "")

        if sections:
            for section in sections:
                if not isinstance(section, dict):
                    continue

                heading = section.get("heading", "").strip() or "Body"
                text = section.get("text", "").strip()

                if not text or self.is_excluded_section(heading):
                    continue

                sec_chunks = self._split_text(
                    text=text,
                    paper_id=paper.arxiv_id,
                    section_name=heading,
                    start_chunk_index=global_chunk_index,
                )
                chunks.extend(sec_chunks)
                global_chunk_index += len(sec_chunks)

        return chunks


