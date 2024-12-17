"""Main module."""
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import gradio as gr

# Paths
index_path = "data/faiss_indices/crisisfacts_float32.index"
data_path = "data/crisisfacts/combined_data.csv"

# Load metadata and FAISS index
print("Loading FAISS index and metadata...")
index = faiss.read_index(index_path)
metadata = pd.read_csv(data_path)
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Search function
def search(query, top_k=10):
    # Generate query embedding
    query_embedding = model.encode([query], show_progress_bar=False).astype("float32")
    # Perform search
    distances, indices = index.search(query_embedding, k=top_k)
    # Retrieve metadata
    results = metadata.iloc[indices[0]]
    return results

# Gradio Interface
def gradio_search(query):
    results = search(query, top_k=10)
    return results.to_dict(orient="records")

with gr.Blocks(title="CrisisFACTS Retrieval") as demo:
    gr.Markdown("# CrisisFACTS Retrieval")
    query_input = gr.Textbox(label="Enter your query")
    output = gr.Dataframe(headers=["doc_id", "event", "text", "source", "source_type", "unix_timestamp"])
    query_input.submit(gradio_search, inputs=[query_input], outputs=[output])

demo.launch()
