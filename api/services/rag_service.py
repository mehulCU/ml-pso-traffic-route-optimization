
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project root: two levels above api/services/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "ml" / "PROJECT_EVALUATION_REPORT.md",
]


def load_documents():
    """Load the project's documentation files."""
    documents = []

    for path in DOCUMENT_PATHS:
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")

        # Split on blank lines to preserve individual paragraphs.
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()

            if len(paragraph) >= 40:
                documents.append(
                    {
                        "source": str(path.relative_to(PROJECT_ROOT)),
                        "text": paragraph,
                    }
                )

    if not documents:
        raise RuntimeError(
            "No project documentation found. Check DOCUMENT_PATHS."
        )

    return documents


def retrieve_context(question: str, top_k: int = 3):
    """Return the most relevant documentation paragraphs."""
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    documents = load_documents()
    texts = [document["text"] for document in documents]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    document_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector, document_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        # Avoid returning irrelevant passages when there is no match.
        if score <= 0:
            continue

        results.append(
            {
                "source": documents[index]["source"],
                "text": documents[index]["text"],
                "score": round(score, 4),
            }
        )

    return results


import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def answer_project_question(question: str) -> dict:
    """Answer a project question using retrieved documentation."""

    context_results = retrieve_context(question, top_k=4)

    if not context_results:
        return {
            "question": question,
            "answer": (
                "I could not find relevant evidence in the project "
                "documentation to answer this question reliably."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in context_results
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the environment.")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=500,
    )

    prompt = f"""
You are an assistant for a traffic route optimization project.

Answer the user's question using ONLY the supplied project context.

Rules:
- Do not invent metrics, experiments, or project capabilities.
- The initial LSTM experiment used a 100-road subset.
- The regional LSTM experiment used 988 road segments, each with 480 hourly observations.
- Never say these experiments used the same evaluation subset.
  Explain that their absolute MAE and RMSE values cannot be
  directly compared because their evaluation subsets differ.
- Do not claim that probe count directly measures speed or congestion.
- Do not claim PSO outperforms Dijkstra unless the context supports it.
- If the context is insufficient, clearly say so.
- Answer clearly and concisely.

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            str(item.get("text", ""))
            for item in answer
            if isinstance(item, dict)
        )

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    return {
        "question": question,
        "answer": answer.strip(),
        "sources": list(dict.fromkeys(
            item["source"] for item in context_results
        )),
    }

