# ============================================================
# backend/pipeline/prompt.py
# Assembles retrieved chunks into a structured LLM prompt.
# ============================================================

from backend.pipeline import RetrievalResult, PromptPackage


def build_rag_prompt(
    query: str,
    retrieval_results: list[RetrievalResult],
    max_context_chunks: int = 5
) -> PromptPackage:
    """
    Builds a two-part prompt:
    - system_prompt: CLT-domain grounding + hallucination guard rules
    - user_prompt:   retrieved context blocks + the user's question

    The system prompt instructs the model to cite tenure, cohort,
    and rate type explicitly — important for CLT analytics answers.
    """
    system_prompt = """You are a precise Customer Lifetime (CLT) analytics assistant.
Your task is to answer the user's question using ONLY the context provided below.
The context contains cohort-level data: tenure, churn rate, survival rate,
active customer counts, and gross adds across products, channels, and customer types.

Follow these rules strictly:
1. If the answer is present in the context, answer clearly and concisely with exact figures.
2. If the answer is NOT present in the context, respond with:
   "I could not find relevant information in the provided CLT data."
3. Never use prior knowledge or make assumptions beyond the context. [Important]
4. Always reference the cohort (product, channel, customer type) and tenure point.
5. Always specify whether a rate is churn rate or survival rate and at which tenure.
"""

    context_blocks = [
        f"[Context {i+1}]\nSource: {r.source}\nContent:\n{r.text}\n"
        for i, r in enumerate(retrieval_results[:max_context_chunks])
    ]

    user_prompt = (
        f"Here is the relevant CLT context retrieved from the data:\n\n"
        f"{''.join(context_blocks)}\n"
        f"Based strictly on the context above, answer the following question:\n"
        f"Question: {query}\n"
    )

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=retrieval_results[:max_context_chunks]
    )
