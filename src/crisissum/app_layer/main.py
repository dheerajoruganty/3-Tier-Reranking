import logging
import time
from fastapi import FastAPI, HTTPException, Query
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
import re
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_layer")

# Initialize FastAPI app
app = FastAPI()

# Environment variables
BACKEND_URL = "http://backend:8001"
DATA_PATH = "/app/data/combined_data.csv"

EXAMPLE_DATA = pd.DataFrame(
    {
        "doc_id": ["CrisisFACTS-010-Twitter-14969-0"],
        "text": [
            "just talked to the owner of houston corvette services. they were right next to the explosion. "
        ],
    }
)


def load_dataset(data_path=DATA_PATH):
    """
    Load and preprocess dataset from the specified file path.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Preprocessed dataset with cleaned text.
    """
    try:
        df = pd.read_csv(data_path)
        df["text"] = df["text"].fillna("").apply(preprocess_text)
        logger.info("Dataset preprocessed and BM25 index loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset or initialize BM25: {e}")
        logger.info(f"Loading Example data for tests")
        df = EXAMPLE_DATA
        df["text"] = df["text"].fillna("").apply(preprocess_text)
        logger.info("Dataset preprocessed and BM25 index loaded successfully.")
        return df


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def preprocess_text(text):
    """
    Clean and normalize text by converting to lowercase, removing URLs,
    special characters, and trimming whitespace.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed and cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove leading/trailing whitespace
    return text


def compute_rouge(reference, hypothesis):
    """
    Compute ROUGE scores between a reference text and a hypothesis.

    Args:
        reference (str): Reference text.
        hypothesis (str): Hypothesis or predicted text.

    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }


def compute_cosine_similarity(query_vector, text_vector):
    """
    Compute cosine similarity between a query vector and text vector.

    Args:
        query_vector (np.ndarray): Embedding vector of the query.
        text_vector (np.ndarray): Embedding vector of the text.

    Returns:
        float: Cosine similarity score rounded to 4 decimal places.
    """
    return round(cosine_similarity([query_vector], [text_vector])[0][0], 4)


def compute_jaccard_similarity(query, text):
    """
    Compute Jaccard similarity for word-level overlap between query and text.

    Args:
        query (str): Query text.
        text (str): Text to compare with the query.

    Returns:
        float: Jaccard similarity score.
    """
    query_set = set(query.split())
    text_set = set(text.split())
    intersection = len(query_set.intersection(text_set))
    union = len(query_set.union(text_set))
    return round(intersection / union, 4) if union > 0 else 0.0


logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer("blevlabs/stella_en_v5", device=device)
logger.info("Model loaded successfully.")

# Load dataset and preprocess texts
logger.info(f"Loading dataset from {DATA_PATH}...")
try:
    df = load_dataset()
    texts = df["text"].tolist()
    doc_ids = df["doc_id"].tolist()
    bm25 = BM25Okapi([text.split() for text in texts])
    logger.info("Dataset preprocessed and BM25 index loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dataset or initialize BM25: {e}")
    raise


@app.get("/")
def health_check():
    """
    Health check endpoint to verify that the application layer is ready.

    Returns:
        dict: Status message indicating readiness.
    """
    return {"status": "Application Layer is ready"}


@app.get("/query")
def query(
    text: str = Query(..., min_length=1, description="Input query text"),
    top_k: int = Query(10, ge=1, le=100, description="Number of top results to fetch"),
):
    """
    Query endpoint for retrieving relevant results using BM25, FAISS,
    and re-ranking techniques.

    Args:
        text (str): Input query text.
        top_k (int): Number of top results to return.

    Returns:
        dict: A dictionary containing the results and performance metrics.
    """
    try:
        metrics = {}
        logger.info(f"Received query: text='{text}', top_k={top_k}")

        # Preprocess query text
        processed_query = preprocess_text(text)

        # BM25 Retrieval
        logger.info("Performing BM25 retrieval...")
        bm25_start = time.time()
        bm25_scores = bm25.get_scores(processed_query.split())
        bm25_end = time.time()
        metrics["BM25 Time"] = f"{bm25_end - bm25_start:.4f} seconds"

        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        top_texts = [texts[i] for i in top_indices]
        top_doc_ids = [doc_ids[i] for i in top_indices]
        logger.info(f"BM25 retrieval completed. Top {top_k} documents retrieved.")

        # Embedding and FAISS retrieval
        logger.info("Generating embeddings for query...")
        embed_start = time.time()
        query_vector = model.encode(processed_query).tolist()
        embed_end = time.time()
        metrics["Embedding Generation Time"] = f"{embed_end - embed_start:.4f} seconds"

        logger.info("Sending query vector to FAISS backend...")
        faiss_start = time.time()
        response = requests.post(
            f"{BACKEND_URL}/search", json={"query_vector": query_vector, "top_k": top_k}
        )
        response.raise_for_status()
        faiss_results = response.json().get("results", [])
        faiss_end = time.time()
        metrics["FAISS Retrieval Time"] = f"{faiss_end - faiss_start:.4f} seconds"

        # Re-ranking and similarity computation
        logger.info("Re-ranking FAISS results...")
        rerank_start = time.time()
        combined_texts = [preprocess_text(result["text"]) for result in faiss_results]
        candidate_embeddings = model.encode(combined_texts)
        scores = np.dot(candidate_embeddings, np.array(query_vector))
        ranked_indices = np.argsort(scores)[::-1]
        rerank_results = [
            {
                "doc_id": faiss_results[i]["doc_id"],
                "text": faiss_results[i]["text"],
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ][:top_k]
        rerank_end = time.time()
        metrics["Re-ranking Time"] = f"{rerank_end - rerank_start:.4f} seconds"

        return {"results": rerank_results, "metrics": metrics}

    except requests.RequestException as e:
        logger.error(f"Error communicating with backend: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error communicating with backend: {e}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


logger.info("Application layer is ready and listening for requests on port 8000.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
