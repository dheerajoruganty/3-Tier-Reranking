import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from io import StringIO
import pandas as pd
import numpy as np
from src.crisissum.app_layer.main import (
    app,
    preprocess_text,
    compute_rouge,
    compute_cosine_similarity,
    compute_jaccard_similarity,
)

# ----------------- Fixtures ----------------- #

@pytest.fixture
def client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_csv_data():
    """Fixture for mocked CSV file content."""
    csv_data = (
        "doc_id,text\n"
        "1,houston explosion shakes the city\n"
        "2,explosion in houston\n"
        "3,new york fire"
    )
    return StringIO(csv_data)


@pytest.fixture
def mock_faiss_response():
    """Fixture for mocked FAISS backend response."""
    return {
        "results": [
            {"doc_id": "1", "text": "houston explosion shakes the city"},
            {"doc_id": "2", "text": "explosion in houston"},
        ]
    }


# ----------------- Unit Tests ----------------- #

def test_preprocess_text():
    """Test text preprocessing: lowercasing, removing special characters, and trimming."""
    assert preprocess_text("  Hello, WORLD! ") == "hello world"
    assert preprocess_text("Visit http://example.com!") == "visit"
    assert preprocess_text("Special@#$%^&Characters!") == "specialcharacters"
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""


def test_compute_rouge():
    """Test ROUGE score computation for valid input."""
    reference = "houston explosion shakes the city"
    hypothesis = "explosion in houston shakes the city"
    scores = compute_rouge(reference, hypothesis)

    assert isinstance(scores, dict)
    assert scores["ROUGE-1"] > 0.0
    assert scores["ROUGE-2"] > 0.0
    assert scores["ROUGE-L"] > 0.0


def test_compute_cosine_similarity():
    """Test cosine similarity between vectors."""
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([1, 0, 0])
    assert compute_cosine_similarity(vector1, vector2) == 1.0

    vector3 = np.array([0, 1, 0])
    assert compute_cosine_similarity(vector1, vector3) == 0.0


def test_compute_jaccard_similarity():
    """Test Jaccard similarity computation for word overlap."""
    query = "houston explosion"
    text = "explosion houston city"
    score = compute_jaccard_similarity(query, text)
    assert 0.5 <= score <= 1.0

    query = "houston explosion"
    text = "new york fire"
    assert compute_jaccard_similarity(query, text) == 0.0


# ----------------- Endpoint Tests ----------------- #

@patch("pandas.read_csv")
def test_health_check(mock_read_csv, client):
    """
    Test the health check endpoint.
    Ensures the application responds correctly when initialized.
    """
    mock_read_csv.return_value = pd.DataFrame({"doc_id": [1], "text": ["sample text"]})
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Application Layer is ready"}


@patch("pandas.read_csv")
@patch("requests.post")
def test_query_endpoint(mock_post, mock_read_csv, client, mock_csv_data, mock_faiss_response):
    """
    Test the /query endpoint with mocked dataset and FAISS backend.
    Ensures correct results and metrics are returned.
    """
    # Mock the dataset loading
    mock_read_csv.return_value = pd.read_csv(mock_csv_data)

    # Mock FAISS backend response
    mock_post.return_value = Mock(status_code=200)
    mock_post.return_value.json.return_value = mock_faiss_response

    # Call the endpoint
    response = client.get("/query", params={"text": "houston explosion", "top_k": 2})
    json_response = response.json()

    # Assertions
    assert response.status_code == 200
    assert "results" in json_response
    assert "metrics" in json_response

    # Validate the results
    results = json_response["results"]
    assert len(results) == 2
    assert results[0]["text"] == "houston explosion shakes the city"

    # Validate the metrics
    metrics = json_response["metrics"]
    assert "BM25 Time" in metrics
    assert "Re-ranking Time" in metrics
    assert "ROUGE-1" in metrics
    assert "Cosine Similarity" in metrics
    assert "Jaccard Similarity" in metrics
