## 🎯 GenAI Interview Prep: Deep-Dive Q&A

### Q1: Why use a Cross-Encoder for re-ranking instead of fetching top-$k$ directly from FAISS?

**Interview Answer:**

> "Bi-encoders create fixed vector representations for documents independently of the query. While fast for vector search, they miss token-to-token interactions between the query and text chunks. A Cross-Encoder performs full cross-attention over the query and chunk combined ($[\text{CLS}] + Q + [\text{SEP}] + D$), yielding significantly more accurate relevance scores. We use FAISS and BM25 to narrow millions of chunks down to 20 candidates, then use the Cross-Encoder to precisely rank the top 3."

### Q2: What are hallucination guardrails in RAG, and how are they implemented here?

**Interview Answer:**

> "Hallucination often occurs when an LLM is forced to answer based on irrelevant retrieved chunks. Rather than sending low-quality context to the prompt, we implement a retrieval-side logit cutoff threshold ($\ge -2.5$) using the Cross-Encoder. If every retrieved chunk falls below this threshold, the pipeline refuses to call the LLM and directly returns a fallback message. This cuts generation costs and stops hallucinations at the retrieval layer."

### Q3: How does MMR (Maximal Marginal Relevance) differ from standard Cosine Similarity?

**Interview Answer:**

> "Standard Cosine Similarity returns the chunks most similar to the query, which often leads to retrieving duplicates or repetitive paragraphs. MMR optimizes for both relevance and novelty:
> $$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$
> Here, $\lambda = 0.7$ ensures that the retrieved chunks are relevant to the query while penalizing chunks that are too similar to already-selected ones."

### Q4: Why combine BM25 and Dense Retrieval?

**Interview Answer:**

> "Dense embeddings capture broad semantic intent but can fail on exact keyword queries, proper nouns, numerical codes, or unique IDs. BM25 relies on inverted indices and term frequency calculations, making it reliable for exact keyword matching. Combining them through an Ensemble Retriever gives us both semantic coverage and exact-keyword precision."

### Q5: How does Semantic Caching work, and why not use an in-memory dictionary?

**Interview Answer:**

> "A standard in-memory dictionary requires exact string matches. If a user asks the same question with minor rephrasing, a standard key-value lookup misses. A semantic cache embeds the incoming query and runs similarity search against past queries in a FAISS index. If the cosine similarity exceeds $0.88$, we return the cached response, saving both API costs and LLM response latency."
