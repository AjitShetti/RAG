"""PDF parsing service using pdfminer.six.

PdfParserService is a pure, stateless service class — no Airflow, no SQLAlchemy.
It downloads a PDF from a URL, extracts full text and a best-effort section
structure, and returns a ParsedDocument.

Section detection heuristic:
  - Text blocks whose average character font size is >= HEADING_FONT_THRESHOLD
    are treated as section headings.
  - Everything between two consecutive headings is grouped as that section's body.
  - If fewer than 2 headings are found, the entire extracted text is returned
    as a single un-sectioned block.

Parse failures are caught and logged — PdfParserService never raises so that
callers (MetadataFetcher) can record `parse_status='failed'` and continue.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import httpx
import tenacity
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LAParams, LTAnon, LTChar, LTTextBox, LTTextLine

logger = logging.getLogger(__name__)

# Font-size threshold above which a text block is considered a heading.
HEADING_FONT_THRESHOLD = 13.0
# Timeout for PDF HTTP download in seconds
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ParsedDocument:
    """Structured output from PdfParserService."""

    full_text: str
    sections: list[dict[str, str]] = field(default_factory=list)
    """List of {'heading': str, 'text': str} dicts."""


class PdfParserService:
    """Downloads and parses arXiv PDF files into structured content.

    Usage::

        parser = PdfParserService()
        doc = parser.parse("https://arxiv.org/pdf/2401.00001")
        if doc:
            print(doc.full_text[:200])
    """

    # ── Public API ────────────────────────────────────────────

    def parse(self, pdf_url: str, arxiv_id: str = "") -> ParsedDocument | None:
        """Download and parse the PDF at *pdf_url*.

        Args:
            pdf_url: Direct URL to the PDF (e.g. ``result.pdf_url``).
            arxiv_id: Optional identifier used only for log messages.

        Returns:
            :class:`ParsedDocument` on success, ``None`` on any failure.
        """
        label = arxiv_id or pdf_url
        try:
            pdf_bytes = self._download_pdf(pdf_url)
        except Exception as exc:
            logger.warning("[%s] PDF download failed: %s", label, exc)
            return None

        try:
            return self._extract(pdf_bytes, label)
        except Exception as exc:
            logger.warning("[%s] PDF extraction failed: %s", label, exc)
            return None

    # ── Internal helpers ──────────────────────────────────────

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=8),
        retry=tenacity.retry_if_exception_type(httpx.TransportError),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _download_pdf(self, url: str) -> bytes:
        """HTTP GET the PDF with retry on transient network errors."""
        logger.debug("Downloading PDF: %s", url)
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def _extract(self, pdf_bytes: bytes, label: str) -> ParsedDocument:
        """Extract full text and sections from in-memory PDF bytes."""
        buf = io.BytesIO(pdf_bytes)
        laparams = LAParams(line_margin=0.5, word_margin=0.1)

        # ── Full text ─────────────────────────────────────────
        buf.seek(0)
        full_text = extract_text(buf, laparams=laparams).strip()
        logger.debug("[%s] Extracted %d chars of text", label, len(full_text))

        # ── Section detection ──────────────────────────────────
        buf.seek(0)
        sections = self._extract_sections(buf, laparams)
        logger.debug("[%s] Detected %d sections", label, len(sections))

        return ParsedDocument(full_text=full_text, sections=sections)

    def _extract_sections(
        self, buf: io.BytesIO, laparams: LAParams
    ) -> list[dict[str, str]]:
        """Best-effort section extraction using font-size heuristics."""
        headings: list[tuple[float, str]] = []   # (page_position, heading_text)
        body_blocks: list[tuple[float, str]] = [] # (page_position, body_text)

        page_offset = 0.0
        for page_layout in extract_pages(buf, laparams=laparams):
            page_height = page_layout.height
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue
                text = element.get_text().strip()
                if not text:
                    continue
                avg_font_size = self._avg_font_size(element)
                position = page_offset + (page_height - element.y1)  # top of element
                if avg_font_size >= HEADING_FONT_THRESHOLD:
                    headings.append((position, text))
                else:
                    body_blocks.append((position, text))
            page_offset += page_height

        if len(headings) < 2:
            return []  # fall back — caller uses full_text instead

        # Sort all blocks by vertical position
        headings.sort(key=lambda x: x[0])
        body_blocks.sort(key=lambda x: x[0])

        # Assign body blocks to the nearest preceding heading
        sections: list[dict[str, str]] = []
        for i, (h_pos, h_text) in enumerate(headings):
            next_h_pos = headings[i + 1][0] if i + 1 < len(headings) else float("inf")
            section_text = "\n".join(
                text
                for pos, text in body_blocks
                if h_pos <= pos < next_h_pos
            )
            sections.append({"heading": h_text, "text": section_text.strip()})

        return sections

    @staticmethod
    def _avg_font_size(textbox: LTTextBox) -> float:
        """Return the average character font size within a text box."""
        sizes: list[float] = []
        for line in textbox:
            if isinstance(line, LTTextLine):
                for char in line:
                    if isinstance(char, LTChar):
                        sizes.append(char.size)
        return sum(sizes) / len(sizes) if sizes else 0.0
