import os
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
# BGE models require this prefix on query strings (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def _get_collection(chroma_path: str | None = None) -> chromadb.Collection:
    path = chroma_path or os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(documents: dict[str, str], chroma_path: str | None = None) -> None:
    """Embed all documents and upsert into ChromaDB."""
    if not documents:
        return

    model = _get_model()
    collection = _get_collection(chroma_path)

    ids = list(documents.keys())
    # Truncate to ~2000 chars (~512 tokens) to match model context window
    texts = [v[:2000] for v in documents.values()]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"filename": k} for k in ids],
    )


def search(query: str, top_k: int = 5, chroma_path: str | None = None) -> list[dict]:
    """Semantic search over indexed documents. Returns top_k results with scores."""
    model = _get_model()
    collection = _get_collection(chroma_path)

    total = collection.count()
    if total == 0:
        return []

    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = model.encode([prefixed_query], normalize_embeddings=True).tolist()

    n = min(top_k, total)
    results = collection.query(query_embeddings=query_embedding, n_results=n)

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        # ChromaDB cosine collection: distances in [0, 2] → convert to similarity [1, -1]
        similarity = 1.0 - (distance / 2.0)
        output.append({
            "filename": doc_id,
            "score": round(similarity, 4),
            "excerpt": results["documents"][0][i][:300],
        })
    return output
