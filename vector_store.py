import logging
from typing import Optional

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DB_PATH = "knowledge_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# CACHED RESOURCES
#
# WHY @st.cache_resource instead of @st.cache_data?
#
# @st.cache_data  — for data (DataFrames, dicts, lists).
#                   Serializes and deserializes on each access.
#
# @st.cache_resource — for objects that are expensive to
#                   create and cannot be serialized: ML models,
#                   database connections, API clients.
#                   Created ONCE, shared across all sessions.
#
# SentenceTransformer downloads ~90MB and takes 3–5 seconds
# to load. Without caching, it reloads on every Streamlit rerun.
# With @st.cache_resource it loads once per server process.
#
# Same logic for the ChromaDB client — you don't want a new
# database connection on every widget interaction.
# ─────────────────────────────────────────────

@st.cache_resource
def _get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model.

    Called on first use, then the same object is returned
    on every subsequent call — no re-downloading.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def _get_collection() -> chromadb.Collection:
    """Create and cache the ChromaDB client and collection.

    PersistentClient writes to disk so the knowledge base
    survives app restarts.
    """
    logger.info("Connecting to ChromaDB at '%s'", DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def store_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed and store a list of DocumentChunks in ChromaDB.

    WHY delete by source first?
    Old code deleted the ENTIRE collection before adding new chunks.
    That meant uploading a second PDF wiped the first one —
    silently destroying your multi-document knowledge base.

    Now we delete only the chunks that came from THIS source file.
    Uploading "policy.pdf" a second time refreshes only policy.pdf
    chunks — all other documents stay intact.

    WHY batch the embeddings?
    Calling model.encode() once per chunk (old code) makes N
    separate calls to the model. Passing all texts at once lets
    the model process them in parallel — much faster on large docs.
    """
    if not chunks:
        logger.warning("store_chunks called with empty list — nothing to store")
        return

    collection = _get_collection()
    model = _get_embedding_model()

    # All chunks in this batch come from the same source file
    source = chunks[0].source

    # Delete existing chunks from this source only
    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info(
                "Deleted %d existing chunks from '%s'",
                len(existing["ids"]),
                source,
            )
    except Exception as e:
        # Log but don't crash — the get/delete failing doesn't
        # stop us from adding the new chunks
        logger.warning("Could not clear existing chunks for '%s': %s", source, e)

    # Batch encode all chunk texts at once
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # Build parallel lists for ChromaDB's add() API
    ids = [f"{source}__chunk_{chunk.chunk_index}" for chunk in chunks]

    # WHY include source, page, chunk_index in metadata?
    # This is what makes citations possible. When we retrieve
    # chunks later, we can show the user exactly which file
    # and page the answer came from.
    metadatas = [
        {
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info("Stored %d chunks from '%s'", len(chunks), source)


def get_top_chunks(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
) -> list[dict]:
    """Retrieve the top-K most semantically similar chunks.

    Args:
        query:         The user's natural language question.
        top_k:         Number of chunks to retrieve.
        source_filter: Optional filename to restrict search to
                       one document. None = search all documents.

    Returns:
        List of dicts, each with keys:
          - "text":    The chunk content
          - "source":  Filename
          - "page":    Page number
        Ordered by relevance (most relevant first).

    WHY return dicts instead of just text?
    The old code returned only the text strings. That threw away
    the metadata (source, page) we carefully stored. Returning
    dicts lets app.py display citations alongside each answer.
    """
    if not query.strip():
        logger.warning("get_top_chunks called with empty query")
        return []

    collection = _get_collection()
    model = _get_embedding_model()

    query_embedding = model.encode(query).tolist()

    # Build query kwargs — optionally filter by source document
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    # Zip documents, metadatas, and distances into clean dicts.
    # ChromaDB distance is L2 (lower = more similar).
    # We convert to a 0–100 confidence score for the UI:
    #   distance 0.0 → confidence 100 (perfect match)
    #   distance 1.0 → confidence ~50
    #   distance 2.0+ → confidence ~0
    # Formula: confidence = max(0, 100 - distance * 50)
    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        confidence = round(max(0.0, 100.0 - distance * 50.0), 1)
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "confidence": confidence,
        })

    return chunks


def get_stored_sources() -> list[str]:
    """Return a list of all unique source filenames in the knowledge base.

    Used by app.py to show the user which documents have been indexed,
    and to enable the document management (delete) feature.
    """
    collection = _get_collection()

    try:
        all_items = collection.get(include=["metadatas"])
        sources = {
            meta["source"]
            for meta in all_items["metadatas"]
            if "source" in meta
        }
        return sorted(sources)
    except Exception as e:
        logger.error("Failed to retrieve stored sources: %s", e)
        return []


def delete_source(source: str) -> None:
    """Delete all chunks belonging to a specific source file.

    This is the document management feature — lets users
    remove a document from the knowledge base without
    wiping everything else.
    """
    collection = _get_collection()

    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info("Deleted all chunks from '%s'", source)
        else:
            logger.warning("No chunks found for source '%s'", source)
    except Exception as e:
        logger.error("Failed to delete source '%s': %s", source, e)
        raise
