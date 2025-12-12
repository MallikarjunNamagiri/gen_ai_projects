# scripts/ingest.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Init
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))

# Create collection (replace deprecated recreate_collection)
collection_name = "support_docs"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Load and chunk documents
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

docs = []
for filename in os.listdir("data/"):
    with open(f"data/{filename}") as f:
        text = f.read()
        chunks = chunk_text(text)
        docs.extend([(filename, chunk) for chunk in chunks])

# Generate embeddings and upsert
points = []
for idx, (source, text) in enumerate(tqdm(docs)):
    embedding = model.encode(text).tolist()
    points.append(PointStruct(
        id=idx,
        vector=embedding,
        payload={"text": text, "source": source}
    ))

client.upsert(collection_name="support_docs", points=points)
print(f"✅ Upserted {len(points)} vectors")