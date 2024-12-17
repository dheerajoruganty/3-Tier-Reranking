import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import hashlib
import argparse


def calculate_file_hash(filepath: str) -> str:
    """
    Calculate the MD5 hash of a file to detect changes.

    Args:
        filepath (str): Path to the input file.

    Returns:
        str: Hexadecimal hash string.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_skip_index_creation(data_file: str, hash_file_path: str) -> bool:
    """
    Check if the index creation can be skipped based on file hash.

    Args:
        data_file (str): Path to the data file.
        hash_file_path (str): Path to the hash file storing previous hash.

    Returns:
        bool: True if data hasn't changed and index creation can be skipped, False otherwise.
    """
    current_hash = calculate_file_hash(data_file)

    # Compare current hash with stored hash
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == current_hash:
            print("No changes in data. Skipping index creation.")
            return True

    # Update hash file with the current hash
    os.makedirs(os.path.dirname(hash_file_path), exist_ok=True)
    with open(hash_file_path, "w") as f:
        f.write(current_hash)
    return False


def is_gpu_available() -> bool:
    """
    Check if FAISS GPU is available.

    Returns:
        bool: True if GPU is available, False otherwise.
    """
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def populate_faiss_indices(data_file: str, index_dir: str) -> None:
    """
    Create and populate FAISS indices (float32).

    This function:
    1. Checks if the index creation can be skipped based on file hash.
    2. Loads the input data file and generates embeddings using a SentenceTransformer model.
    3. Creates a FAISS index and saves it to disk.

    Args:
        data_file (str): Path to the input data file.
        index_dir (str): Directory to store FAISS indices.

    Raises:
        FileNotFoundError: If the input data file does not exist.
        Exception: For unexpected errors during processing.
    """
    # Path for hash file
    hash_file_path = os.path.join(index_dir, "data_hash.txt")

    # Check for data changes and skip if needed
    if should_skip_index_creation(data_file, hash_file_path):
        return

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")

    # Load data
    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} rows from {data_file}.")

    # Generate embeddings
    print("Generating embeddings using SentenceTransformer...")
    model = SentenceTransformer("blevlabs/stella_en_v5", trust_remote_code=True)
    embeddings = model.encode(data["text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Ensure index directory exists
    os.makedirs(index_dir, exist_ok=True)

    # Create Float32 FAISS index
    dimension = embeddings.shape[1]
    if is_gpu_available():
        print("Using FAISS GPU for index creation.")
        res = faiss.StandardGpuResources()
        flat_index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
        gpu_index.add(embeddings)
        index_float32 = faiss.index_gpu_to_cpu(gpu_index)
    else:
        print("Using FAISS CPU for index creation.")
        index_float32 = faiss.IndexFlatL2(dimension)
        index_float32.add(embeddings)

    # Save the Float32 index
    float32_index_path = os.path.join(index_dir, "crisisfacts_float32.index")
    faiss.write_index(index_float32, float32_index_path)
    print(f"Float32 index populated and saved at: {float32_index_path}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Populate FAISS Indices")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the input data CSV file (e.g., combined_data.csv).",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory to store the generated FAISS indices.",
    )
    args = parser.parse_args()

    try:
        print("Starting FAISS index creation...")
        populate_faiss_indices(args.data_file, args.index_dir)
        print("FAISS index creation completed successfully.")
    except FileNotFoundError as fnf_error:
        print(f"[Error]: {fnf_error}")
    except Exception as e:
        print(f"[Unexpected Error]: {e}")
