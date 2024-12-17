import logging
import gradio as gr
import requests

# ---------------------------- #
#  Setup and Global Variables  #
# ---------------------------- #

# Set up logging for the frontend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

# Environment variables
APP_LAYER_URL = "http://app_layer:8000"

# Log initialization
logger.info("Starting frontend service...")


# ---------------------------- #
#        Helper Functions      #
# ---------------------------- #


def search_query(text: str, top_k: int = 10):
    """
    Query the application layer with the search input and retrieve results.

    Args:
        text (str): The input query string.
        top_k (int): The number of top results to fetch from the backend.

    Returns:
        dict: A dictionary containing:
              - "Metrics": Retrieval performance metrics.
              - "Results": Top-ranked results as a list of documents.
    """
    try:
        # Log the query details
        logger.info(f"Sending query to app layer: text='{text}', top_k={top_k}")
        logger.info(
            f"Making request to URL: {APP_LAYER_URL}/query?text={text}&top_k={top_k}"
        )

        # Send GET request to the application layer
        response = requests.get(
            f"{APP_LAYER_URL}/query", params={"text": text, "top_k": top_k}
        )
        response.raise_for_status()  # Raise exception for failed HTTP response

        # Parse response JSON
        data = response.json()
        results = data.get("results", [])
        metrics = data.get("metrics", {})

        # Combine results and metrics into output
        output = {"Metrics": metrics, "Results": results}

        # Log and return the output
        logger.info(f"Received output: {output}")
        return output

    except requests.RequestException as e:
        # Handle HTTP-related errors
        logger.error(f"Failed to query app layer: {e}")
        return {"error": f"Error querying app layer: {e}"}

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}


# ---------------------------- #
#        Gradio Interface      #
# ---------------------------- #

# Define the Gradio interface
iface = gr.Interface(
    fn=search_query,  # Function to execute
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter your query here."),
        gr.Slider(
            minimum=1, maximum=100, step=1, value=10, label="Top K Results"
        ),  # Slider for top_k input
    ],
    outputs="json",  # JSON output format
    title="High-Speed Retrieval using BM25 and Stella Reranking",
    description=(
        "This interface queries the application layer for document retrieval. "
        "It combines BM25 for initial ranking and re-ranking using Sentence Transformers "
        "(blevlabs/stella_en_v5). Results and retrieval metrics are displayed in JSON format."
    ),
)

# Start Gradio server
if __name__ == "__main__":
    logger.info("Frontend service is ready and listening on port 7860.")
    iface.launch(server_name="0.0.0.0", server_port=7860)
