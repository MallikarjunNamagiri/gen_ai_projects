# Simple RAG Pipeline – Generic Analytics

Minimal Retrieval-Augmented Generation system over internal analytics documents.

## What this does

- Loads 8 sample analytics documents (sales, churn, product usage, marketing ROI, support, inventory, productivity, financial KPIs)
- Chunks → embeds with `all-MiniLM-L6-v2` (local, free)
- Stores in Chroma
- Retrieves top-3 relevant chunks for any question

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/rag_pipeline.py
```
