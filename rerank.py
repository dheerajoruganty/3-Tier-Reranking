from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

def load_data(file_path):
    """
    Load preprocessed data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def load_colbertv2_model(model_name="bert-base-uncased"):
    """
    Load ColBERTv2 tokenizer and model.

    Parameters:
        model_name (str): Huggingface model name.

    Returns:
        tuple: Tokenizer and model.
    """
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def neural_rerank(model, tokenizer, queries, documents):
    """
    Re-rank documents using ColBERTv2.

    Parameters:
        model (AutoModel): Pre-trained model for embeddings.
        tokenizer (AutoTokenizer): Tokenizer for queries and documents.
        queries (list): List of queries.
        documents (list): List of documents.

    Returns:
        list: Ranked documents for each query.
    """
    query_embeddings = []
    doc_embeddings = []

    # Generate query embeddings
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt")
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        query_embeddings.append(query_embedding)

    # Generate document embeddings
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
        doc_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        doc_embeddings.append(doc_embedding)

    # Convert embeddings to tensor
    query_embeddings = torch.stack(query_embeddings)  # Shape: (n_queries, embedding_dim)
    doc_embeddings = torch.stack(doc_embeddings)  # Shape: (n_docs, embedding_dim)

    # Perform re-ranking
    results = []
    for query_embedding in query_embeddings:
        scores = torch.matmul(doc_embeddings, query_embedding)  # Shape: (n_docs,)
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        ranked_docs = [documents[i] for i in ranked_indices]
        results.append(ranked_docs)

    return results

if __name__ == "__main__":
    # Example data
    data = load_data("data/preprocessed_crisisfacts_data.csv")
    queries = ["wildfire damage", "hurricane impact", "emergency response"]
    documents = data["preprocessed_text"].fillna("").tolist()[:100]

    # Load ColBERTv2 model
    tokenizer, model = load_colbertv2_model()

    # Re-rank results
    reranked_results = neural_rerank(model, tokenizer, queries, documents)
    print(reranked_results)