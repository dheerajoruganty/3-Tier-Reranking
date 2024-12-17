import logging
import gradio as gr
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

# Environment variables
APP_LAYER_URL = "http://app_layer:8000"

# Log initialization
logger.info("Starting frontend service...")


# Gradio interface
def search_query(text, top_k=10):
    """
    Query the app layer with a search query and display retrieval metrics.
    Args:
        text: Input search query.
        top_k: Number of top results to fetch.
    Returns:
        Results and metrics as JSON.
    """
    try:
        logger.info(f"Sending query to app layer: text='{text}', top_k={top_k}")

        # Log the full URL being requested
        logger.info(
            f"Making request to URL: {APP_LAYER_URL}/query?text={text}&top_k={top_k}"
        )

        # Send query to app layer
        response = requests.get(
            f"{APP_LAYER_URL}/query", params={"text": text, "top_k": top_k}
        )
        response.raise_for_status()

        # Process response
        data = response.json()
        results = data.get("results", [])
        metrics = data.get("metrics", {})

        # Combine metrics and results
        output = {"Metrics": metrics, "Results": results}

        # Log and return the output
        logger.info(f"Output: {output}")
        return output

    except requests.RequestException as e:
        logger.error(f"Failed to query app layer: {e}")
        return {"error": f"Error querying app layer: {e}"}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}


# Gradio interface
iface = gr.Interface(
    fn=search_query,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter your query here."),
        gr.Slider(
            minimum=1, maximum=100, step=1, value=10, label="Top K Results"
        ),  # Replaced `default` with `value`
    ],
    outputs="json",
    title="High Speed Retrieval using BM25 and Reranking using blevlabs/stella_en_v5",
    description="Search the application layer and display retrieval metrics.",
)


# Start Gradio server
logger.info("Frontend service is ready and listening on port 7860.")
iface.launch(server_name="0.0.0.0", server_port=7860)
