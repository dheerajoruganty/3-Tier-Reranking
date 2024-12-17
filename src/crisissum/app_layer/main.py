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

# ---------------------------- #
#        Logging Setup         #
# ---------------------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_layer")

# ---------------------------- #
#       FastAPI Application    #
# ---------------------------- #

app = FastAPI()

# ---------------------------- #
#       Environment Setup      #
# ---------------------------- #

BACKEND_URL = "http://backend:8001"
DATA_PATH = "/app/data/combined_data.csv"

# Example dataset to load when file is unavailable
EXAMPLE_DATA = pd.DataFrame(
    {
        "doc_id": ["CrisisFACTS-010-Twitter-14969-0"],
        "text": [
            "just talked to the owner of houston corvette services. they were right next to the explosion."
        ],
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


# ---------------------------- #
#        Helper Functions      #
# ---------------------------- #


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text.

    Steps include:
      - Lowercasing the text
      - Removing URLs
      - Removing special characters
      - Trimming leading/trailing whitespace

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.strip()
    return text


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    Args:
        reference (str): Reference text.
        hypothesis (str): Hypothesis text to compare.

    Returns:
        dict: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }


def compute_cosine_similarity(query_vector: list, text_vector: list) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        query_vector (list): Query vector.
        text_vector (list): Candidate text vector.

    Returns:
        float: Cosine similarity score rounded to 4 decimals.
    """
    return round(cosine_similarity([query_vector], [text_vector])[0][0], 4)


def compute_jaccard_similarity(query: str, text: str) -> float:
    """
    Compute Jaccard similarity for word-level overlap.

    Args:
        query (str): Query text.
        text (str): Candidate text.

    Returns:
        float: Jaccard similarity score.
    """
    query_set = set(query.split())
    text_set = set(text.split())
    intersection = len(query_set.intersection(text_set))
    union = len(query_set.union(text_set))
    return round(intersection / union, 4) if union > 0 else 0.0


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load dataset from a CSV file or fallback to example data.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and preprocessed dataset.
    """
    try:
        df = pd.read_csv(data_path)
        df["text"] = df["text"].fillna("").apply(preprocess_text)
        logger.info("Dataset preprocessed and BM25 index loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset or initialize BM25: {e}")
        logger.info("Loading Example data for tests")
        df = EXAMPLE_DATA
        df["text"] = df["text"].fillna("").apply(preprocess_text)
        logger.info("Example dataset loaded successfully.")
        return df


# ---------------------------- #
#        Model Initialization  #
# ---------------------------- #

# Load SentenceTransformer model
logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer("blevlabs/stella_en_v5", device=device)
logger.info("Model loaded successfully.")

# Load dataset and BM25 index
logger.info(f"Loading dataset from {DATA_PATH}...")
df = load_dataset()
texts = df["text"].tolist()
doc_ids = df["doc_id"].tolist()
bm25 = BM25Okapi([text.split() for text in texts])


# ---------------------------- #
#        API Endpoints         #
# ---------------------------- #


@app.get("/")
def health_check():
    """
    Health check endpoint to verify if the API layer is running.

    Returns:
        dict: A status message.
    """
    return {"status": "Application Layer is ready"}


@app.get("/query")
def query(
    text: str = Query(..., min_length=1, description="Input query text"),
    top_k: int = Query(10, ge=1, le=100, description="Number of top results to fetch"),
):
    """
    Retrieve and re-rank documents for the given query text.

    Workflow:
      1. Perform BM25 retrieval to fetch candidate documents.
      2. Generate embeddings for the query and candidate documents.
      3. Re-rank documents using cosine similarity scores.
      4. Compute ROUGE and Jaccard metrics for the top result.

    Args:
        text (str): Input query string.
        top_k (int): Number of top-ranked results to return.

    Returns:
        dict: Re-ranked results with retrieval metrics.
    """
    try:
        metrics = {}
        logger.info(f"Received query: text='{text}', top_k={top_k}")

        # Preprocess query text
        processed_query = preprocess_text(text)

        # BM25 Retrieval
        bm25_start = time.time()
        bm25_scores = bm25.get_scores(processed_query.split())
        bm25_end = time.time()
        metrics["BM25 Time"] = f"{bm25_end - bm25_start:.4f} seconds"

        # Top K BM25 results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        top_texts = [texts[i] for i in top_indices]
        top_doc_ids = [doc_ids[i] for i in top_indices]

        # Embedding-based Re-ranking
        query_vector = model.encode(processed_query).tolist()
        candidate_embeddings = model.encode(top_texts)
        scores = np.dot(candidate_embeddings, np.array(query_vector))
        ranked_indices = np.argsort(scores)[::-1]
        rerank_results = [
            {"doc_id": top_doc_ids[i], "text": top_texts[i], "score": float(scores[i])}
            for i in ranked_indices
        ]

        # Compute Metrics for Top Result
        if rerank_results:
            top_result_text = preprocess_text(rerank_results[0]["text"])
            rouge_scores = compute_rouge(processed_query, top_result_text)
            cosine_score = compute_cosine_similarity(
                query_vector, model.encode(top_result_text)
            )
            jaccard_score = compute_jaccard_similarity(processed_query, top_result_text)

            metrics.update(rouge_scores)
            metrics["Cosine Similarity"] = cosine_score
            metrics["Jaccard Similarity"] = jaccard_score

        metrics["Total Query Processing Time"] = (
            f"{time.time() - bm25_start:.4f} seconds"
        )

        return {"results": rerank_results, "metrics": metrics}

    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------------------------- #
#      Run FastAPI Server      #
# ---------------------------- #

logger.info("Application layer is ready and listening for requests on port 8000.")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
