import time
import pandas as pd
from rank_bm25 import BM25Okapi
import torch

def load_data(file_path):
    """
    Load preprocessed data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def prepare_corpus(data, text_column):
    """
    Prepare the corpus for BM25 by tokenizing text data.

    Parameters:
        data (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column containing text data.

    Returns:
        list: Tokenized corpus.
    """
    corpus = data[text_column].fillna("").astype(str).str.split().tolist()
    return corpus

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
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        query_embeddings.append(query_embedding)

    # Generate document embeddings
    for doc in documents:
        if not isinstance(doc, str):  # Ensure document is a string
            doc = str(doc)
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512)
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

def optimize_pipeline(corpus, queries, tokenizer, model, top_k=10):
    """
    Optimize BM25 and ColBERTv2 pipeline for scalability.

    Parameters:
        corpus (list): Tokenized corpus.
        queries (list): List of query strings.
        tokenizer (AutoTokenizer): ColBERTv2 tokenizer.
        model (AutoModel): ColBERTv2 model.
        top_k (int): Number of top documents to pass to the second stage.

    Returns:
        dict: Final results and execution time.
    """
    start_time = time.time()

    # BM25 retrieval
    bm25 = BM25Okapi(corpus)
    bm25_results = [bm25.get_top_n(query.split(), corpus, n=top_k) for query in queries]

    # Neural re-ranking
    final_results = []
    for query, docs in zip(queries, bm25_results):
        doc_texts = [" ".join(doc) if isinstance(doc, list) else str(doc) for doc in docs]
        final_results.append(neural_rerank(model, tokenizer, [query], doc_texts))

    end_time = time.time()
    return {"results": final_results, "latency": end_time - start_time}

if __name__ == "__main__":
    # Load data and models
    data = load_data("data/preprocessed_crisisfacts_data.csv")
    corpus = prepare_corpus(data, text_column="preprocessed_text")
    tokenizer, model = load_colbertv2_model()

    # Example queries
    queries = ["wildfire damage", "hurricane impact", "emergency response"]

    # Optimize pipeline
    results = optimize_pipeline(corpus, queries, tokenizer, model)
    print("Final Results:", results["results"])
    print("Total Latency:", results["latency"], "seconds")
