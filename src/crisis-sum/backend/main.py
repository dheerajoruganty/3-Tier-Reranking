import os
import faiss
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn  # For running the FastAPI app indefinitely

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# Initialize FastAPI app
app = FastAPI()

# Paths
FAISS_INDEX_PATH = "/app/faiss_indices/crisisfacts_float32.index"
DATA_PATH = "/app/data/combined_data.csv"

# Load FAISS Index
logger.info("Loading FAISS index...")
if not os.path.exists(FAISS_INDEX_PATH):
    logger.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
    raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")

index = faiss.read_index(FAISS_INDEX_PATH)
logger.info("FAISS index loaded successfully.")

# Load Dataset
logger.info("Loading dataset...")
if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found at {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
texts = df["text"].fillna("").tolist()
doc_ids = df["doc_id"].tolist()
logger.info(f"Dataset loaded successfully. Total documents: {len(texts)}")


# Request model
class SearchRequest(BaseModel):
    query_vector: list
    top_k: int = 10


@app.post("/search")
def search(request: SearchRequest):
    """
    Search FAISS index with the given query vector.
    Args:
        request: A JSON object with 'query_vector' (list of floats) and 'top_k' (integer).
    Returns:
        List of matching documents.
    """
    try:
        # Log input data
        logger.info(f"Received search request with top_k={request.top_k}")
        logger.info(f"FAISS index dimension: {index.d}")
        logger.debug(f"Query vector: {request.query_vector}")

        # Convert query vector to float32
        query_vector = np.array(request.query_vector, dtype=np.float32).reshape(1, -1)
        logger.info(f"Query vector dimension: {query_vector.shape}")

        # Check query vector dimension
        if query_vector.shape[1] != index.d:
            logger.error(
                f"Query vector length mismatch. Expected {index.d}, got {query_vector.shape[1]}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Query vector length mismatch. Expected {index.d}, got {query_vector.shape[1]}.",
            )

        # Perform FAISS search
        logger.info("Performing FAISS search...")
        distances, indices = index.search(query_vector, request.top_k)
        logger.info(f"FAISS search completed. Retrieved indices: {indices[0]}")

        # Retrieve results
        results = [
            {
                "doc_id": doc_ids[i],
                "text": texts[i],
                "distance": float(distances[0][j]),
            }
            for j, i in enumerate(indices[0])
            if i != -1
        ]
        logger.info(f"Retrieved Results: {results}")
        logger.info(f"Retrieved {len(results)} results from FAISS index.")

        # Return results
        return {"results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


logger.info("Backend is ready to serve FAISS queries.")

# Ensure the backend server runs indefinitely
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
