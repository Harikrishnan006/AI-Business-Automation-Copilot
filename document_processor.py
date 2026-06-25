import re
import logging
from dataclasses import dataclass

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA CLASS
#
# WHY a dataclass instead of a plain dict?
# A dict like {"text": ..., "page": ..., "source": ...}
# has no type safety — you can mistype a key and
# get a silent bug. A dataclass gives you:
# - Autocomplete in your editor
# - Type checking with mypy
# - A clear contract: "a chunk always has these fields"
# ─────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of text extracted from a document."""
    text: str
    source: str       # filename
    page: int         # 1-based page number the chunk started on
    chunk_index: int  # position within this document's chunks


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters.

    WHY keep this separate?
    Text cleaning rules change often. Keeping it isolated
    means you can improve cleaning without touching chunking.
    """
    # Remove non-printable characters except newlines and spaces
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────
# PDF EXTRACTION  (page-aware)
#
# WHY extract per page instead of all at once?
# Storing the page number per chunk makes citations
# possible. "This answer comes from page 3 of
# company_policy.pdf" is far more useful than just
# returning the answer text.
# ─────────────────────────────────────────────

def extract_pages_from_pdf(uploaded_file) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF.

    Returns:
        List of (page_number, page_text) tuples.
        Page numbers are 1-based (page 1, not page 0).
        Pages with no extractable text are skipped.
    """
    pdf_reader = PdfReader(uploaded_file)
    pages: list[tuple[int, str]] = []

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        raw_text = page.extract_text()

        if not raw_text:
            logger.debug("Page %d has no extractable text — skipping", page_num)
            continue

        cleaned = clean_text(raw_text)

        if cleaned:
            pages.append((page_num, cleaned))

    logger.info(
        "Extracted %d pages with text from '%s'",
        len(pages),
        getattr(uploaded_file, "name", "unknown"),
    )

    return pages


# ─────────────────────────────────────────────
# CHUNKING  (sentence-aware)
#
# WHY sentence-aware instead of raw character slicing?
#
# Old approach:
#   text[start : start + 500]
# This splits mid-word, mid-sentence. Example:
#   "The refund policy states that customer" ← chunk ends here
#   "s are entitled to a full refund within" ← next chunk starts here
# The word "customers" is split across two chunks. Neither
# chunk makes sense on its own, so retrieval quality suffers.
#
# New approach:
# 1. Split the text into sentences first
# 2. Accumulate sentences until we hit the size limit
# 3. Overlap by carrying forward the last N sentences
#
# This guarantees every chunk ends at a sentence boundary.
# ─────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics.

    Not perfect (abbreviations like "Dr." can confuse it),
    but good enough for business documents without adding
    an NLP dependency like spaCy.
    """
    # Split on . ! ? followed by whitespace and a capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Filter out very short fragments (likely noise)
    return [s.strip() for s in raw if len(s.strip()) > 20]


def create_chunks(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int = 800,
    overlap_sentences: int = 2,
) -> list[DocumentChunk]:
    """Split page text into overlapping DocumentChunks.

    Args:
        pages:             Output of extract_pages_from_pdf().
        source:            Filename — stored in every chunk for citations.
        chunk_size:        Max characters per chunk.
        overlap_sentences: How many sentences from the previous chunk
                           to carry into the next one. This preserves
                           context at chunk boundaries.

    Returns:
        List of DocumentChunk objects ready to embed and store.

    WHY chunk_size=800 instead of 500?
    500 chars ≈ 125 tokens. Many answers require 2–3 sentences of
    context to be coherent. 800 chars ≈ 200 tokens — still well
    within embedding model limits, but gives more coherent chunks.
    """
    all_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page_num, page_text in pages:
        sentences = _split_into_sentences(page_text)

        if not sentences:
            # Page had text but no sentence-splittable content
            # (e.g. a table of numbers). Store it as one chunk.
            all_chunks.append(DocumentChunk(
                text=page_text,
                source=source,
                page=page_num,
                chunk_index=chunk_index,
            ))
            chunk_index += 1
            continue

        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed the limit AND
            # we already have some content, flush the current chunk.
            if current_length + sentence_len > chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                all_chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

                # Carry forward the last N sentences as overlap.
                # WHY? If an answer spans a chunk boundary, the
                # overlap ensures neither chunk is missing context.
                current_sentences = current_sentences[-overlap_sentences:]
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sentence_len

        # Flush any remaining sentences as the final chunk for this page.
        if current_sentences:
            all_chunks.append(DocumentChunk(
                text=" ".join(current_sentences),
                source=source,
                page=page_num,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    logger.info(
        "Created %d chunks from '%s'",
        len(all_chunks),
        source,
    )

    return all_chunks
