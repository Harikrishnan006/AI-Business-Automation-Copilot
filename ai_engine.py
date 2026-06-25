import logging
import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.2   # Low temperature = more factual, less creative.
                    # Good default for a business knowledge assistant.


# ─────────────────────────────────────────────
# MODEL FACTORY
#
# Same pattern as the refactored Project 1:
# create the model inside a function, not at
# module level. Reasons:
# - API key is checked before any SDK call
# - Tests can import this file without side effects
# - Model name is swappable without hunting through code
# ─────────────────────────────────────────────

def _get_model(model_name: str = MODEL_NAME) -> genai.GenerativeModel:
    """Configure Gemini and return a GenerativeModel instance."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file: GEMINI_API_KEY=your_key_here"
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# ─────────────────────────────────────────────
# PROMPT BUILDER
#
# WHY a dedicated prompt function?
# The prompt is the most important engineering
# decision in a RAG system. A vague prompt gives
# vague answers. A well-structured prompt:
# 1. Tells the model its role
# 2. Gives it the retrieved context
# 3. Tells it explicitly what to do AND what NOT to do
# 4. Asks it to cite sources
#
# Keeping this separate means you can iterate on
# the prompt without touching the API call logic.
# ─────────────────────────────────────────────

def _build_rag_prompt(
    question: str,
    context_chunks: list[dict],
) -> str:
    """Build a citation-aware RAG prompt.

    Args:
        question:       The user's question.
        context_chunks: List of dicts with keys: text, source, page.
                        Each dict is one retrieved chunk.

    WHY include source and page in the prompt?
    If you just paste the chunk text, the model can't tell which
    document it came from. By labelling each chunk with its source
    and page, the model can include citations in its answer.
    """
    # Format each chunk with its citation label
    formatted_chunks = []
    for i, chunk in enumerate(context_chunks, start=1):
        label = f"[Source {i}: {chunk['source']}, page {chunk['page']}]"
        formatted_chunks.append(f"{label}\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(formatted_chunks)

    return f"""You are an AI Business Assistant helping employees find information in company documents.

You have been given the following excerpts from the company knowledge base:

{context_block}

---

Instructions:
- Answer the question using ONLY the information from the excerpts above.
- If the answer is in the excerpts, provide a clear and complete answer.
- At the end of your answer, cite which sources you used (e.g. "Source: policy.pdf, page 3").
- If the excerpts do not contain enough information to answer the question, say:
  "I couldn't find a clear answer in the uploaded documents. Please check the source files directly."
- Do NOT use your general knowledge to fill gaps. Stick to what the documents say.

Question: {question}

Answer:"""


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def ask_with_context(
    question: str,
    context_chunks: list[dict],
    model_name: Optional[str] = None,
) -> str:
    """Generate a grounded answer using retrieved document chunks.

    This is the main RAG answer function. Always prefer this over
    ask_without_context — it grounds the answer in your documents.

    Args:
        question:       The user's question.
        context_chunks: Retrieved chunks from vector_store.get_top_chunks().
        model_name:     Optional model override.

    Returns:
        The model's answer as a string.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
        RuntimeError:     If the Gemini API call fails.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not context_chunks:
        return (
            "No relevant documents were found in the knowledge base. "
            "Please upload documents first and then ask your question."
        )

    model = _get_model(model_name or MODEL_NAME)
    prompt = _build_rag_prompt(question, context_chunks)

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": TEMPERATURE},
        )
        return response.text

    except Exception as e:
        logger.error("Gemini API error in ask_with_context: %s", e)
        # Raise so app.py can show a proper st.error() to the user
        # instead of silently displaying "Error: ..." as if it were an answer
        raise RuntimeError(f"Failed to generate answer: {e}") from e


def ask_without_context(
    question: str,
    model_name: Optional[str] = None,
) -> str:
    """Ask Gemini a general question without document context.

    WHY keep this at all?
    It's useful for general questions that don't need the knowledge
    base — e.g. "summarize what RAG is" or "explain this term".
    But in app.py, the user is always warned this uses general AI
    knowledge, not their uploaded documents.

    Args:
        question:   The user's question.
        model_name: Optional model override.

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError: If the Gemini API call fails.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    model = _get_model(model_name or MODEL_NAME)

    try:
        response = model.generate_content(
            question,
            generation_config={"temperature": TEMPERATURE},
        )
        return response.text

    except Exception as e:
        logger.error("Gemini API error in ask_without_context: %s", e)
        raise RuntimeError(f"Failed to generate answer: {e}") from e
