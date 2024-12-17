from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer("dunzhang/stella_en_1.5B_v5")

def embed_query(query: str):
    """Generate float32 embeddings for a query."""
    return model.encode(query)

def quantize_query(embedding):
    """Quantize float32 embeddings into binary."""
    return quantize_embeddings(embedding.reshape(1, -1), "ubinary")
