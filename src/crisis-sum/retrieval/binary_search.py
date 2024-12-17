import faiss

# Load prebuilt FAISS binary index
binary_index = faiss.read_index_binary("path_to_binary_index")

def search_binary(query_binary, top_k=40):
    """Perform a binary search on the FAISS index."""
    distances, indices = binary_index.search(query_binary, top_k)
    return indices[0]  # Top-K document IDs
