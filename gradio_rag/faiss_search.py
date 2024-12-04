import faiss
import pickle
from sentence_transformers import SentenceTransformer

# FAISS 인덱스 로드
def load_faiss_index(index_path="faiss_index.bin"):
    return faiss.read_index(index_path)

# 청크 로드
def load_chunks(chunk_path="chunks.pkl"):
    with open(chunk_path, "rb") as f:
        return pickle.load(f)

# 검색 기능
def search_top_k_with_context(index, query, model, chunks, k=5, context_range=1):
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        start_idx = max(0, idx - context_range)
        end_idx = min(len(chunks), idx + context_range + 1)
        context = " ".join(chunks[start_idx:end_idx])
        results.append((context, distances[0][i]))
    return results
