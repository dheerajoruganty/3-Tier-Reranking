import requests

def handle_query(query, top_k, backend_url):
    """Handle query by forwarding it to the backend."""
    response = requests.post(f"{backend_url}/search", json={"query": query, "top_k": top_k})
    return response.json()
