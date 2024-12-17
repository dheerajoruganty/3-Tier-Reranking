import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import hashlib


# Paths
data_file = "../../../data/combined_data.csv"
index_dir = "../../../data/faiss_indices/"
float32_index_path = os.path.join(index_dir, "crisisfacts_float32.index")
binary_index_path = os.path.join(index_dir, "crisisfacts_binary.index")
int8_index_path = os.path.join(index_dir, "crisisfacts_int8.index")
hash_file_path = os.path.join(index_dir, "data_hash.txt")


def calculate_file_hash(filepath):
    """Calculate the hash of a file to detect changes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_skip_index_creation(data_file, hash_file_path):
    """Check if the index should be skipped based on the data hash."""
    current_hash = calculate_file_hash(data_file)

    # Check if hash file exists
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == current_hash:
            print("No changes in data. Skipping index creation.")
            return True

    # Update hash file
    with open(hash_file_path, "w") as f:
        f.write(current_hash)
    return False

def is_gpu_available():
    """Check if FAISS GPU is available."""
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def populate_faiss_indices(data_file, index_dir):
    """Create and populate FAISS indices (float32, binary, int8)."""
    # Skip index creation if data hasn't changed
    if should_skip_index_creation(data_file, hash_file_path):
        return

    # Load data
    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} rows from {data_file}.")

    # Generate embeddings
    model = SentenceTransformer("blevlabs/stella_en_v5", trust_remote_code=True)
    embeddings = model.encode(data["text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Float32 Index
    dimension = embeddings.shape[1]
    if is_gpu_available():
        print("Using FAISS GPU.")
        res = faiss.StandardGpuResources()
        flat_index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
        gpu_index.add(embeddings)
        index_float32 = faiss.index_gpu_to_cpu(gpu_index)
    else:
        print("Using FAISS CPU.")
        index_float32 = faiss.IndexFlatL2(dimension)
        index_float32.add(embeddings)

    print(f"Populated Float32 index with {index_float32.ntotal} items.")
    faiss.write_index(index_float32, float32_index_path)
    print(f"Float32 index saved at {float32_index_path}.")


if __name__ == "__main__":
    populate_faiss_indices(data_file, index_dir)