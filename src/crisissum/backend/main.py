import os
import faiss
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn  # For running the FastAPI app indefinitely

# --------------------------- #
#  Setup and Global Variables #
# --------------------------- #

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# Initialize FastAPI app
app = FastAPI()

# Paths to FAISS index and dataset
FAISS_INDEX_PATH = "/app/faiss_indices/crisisfacts_float32.index"
DATA_PATH = "/app/data/combined_data.csv"

# --------------------------- #
#  FAISS Index and Data Load  #
# --------------------------- #

logger.info("Loading FAISS index...")
if not os.path.exists(FAISS_INDEX_PATH):
    logger.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
    raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")

index = faiss.read_index(FAISS_INDEX_PATH)
logger.info("FAISS index loaded successfully.")

logger.info("Loading dataset...")
if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found at {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
texts = df["text"].fillna("").tolist()
doc_ids = df["doc_id"].tolist()
logger.info(f"Dataset loaded successfully. Total documents: {len(texts)}")


# --------------------------- #
#       Request Model         #
# --------------------------- #


class SearchRequest(BaseModel):
    """
    Represents the input request schema for the FAISS search endpoint.

    Attributes:
        query_vector (list): A list of floats representing the query embedding.
        top_k (int): The number of top documents to retrieve.
    """

    query_vector: list
    top_k: int = 10


# --------------------------- #
#         API Endpoint        #
# --------------------------- #


@app.post("/search")
def search(request: SearchRequest):
    """
    Perform a search against the FAISS index using the input query vector.

    Args:
        request (SearchRequest): A JSON object containing:
            - query_vector: List of floats representing the query vector.
            - top_k: Number of top documents to retrieve.

    Returns:
        dict: A dictionary containing the top matching documents:
            - doc_id (str): Document ID.
            - text (str): Associated text.
            - distance (float): FAISS distance metric.
    """
    try:
        # Log input request details
        logger.info(f"Received search request with top_k={request.top_k}")
        logger.debug(f"Query vector: {request.query_vector}")

        # Convert query vector to float32 and validate its shape
        query_vector = np.array(request.query_vector, dtype=np.float32).reshape(1, -1)
        logger.info(f"Query vector dimension: {query_vector.shape}")

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

        # Retrieve matching results
        results = [
            {
                "doc_id": doc_ids[i],
                "text": texts[i],
                "distance": float(distances[0][j]),
            }
            for j, i in enumerate(indices[0])
            if i != -1  # Ensure valid index
        ]
        logger.info(f"Retrieved {len(results)} results from FAISS index.")

        # Return search results
        return {"results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# --------------------------- #
#         Main Runner         #
# --------------------------- #

logger.info("Backend is ready to serve FAISS queries.")

if __name__ == "__main__":
    """
    Run the FastAPI application using Uvicorn. This allows the backend service to listen for incoming requests.
    """
    uvicorn.run(app, host="0.0.0.0", port=8001)
