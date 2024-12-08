import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file_path):
    """
    Load preprocessed data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    import pandas as pd
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
    return data[text_column].fillna("").str.split().tolist()

def bm25_retrieve(corpus, queries, top_k=10):
    """
    Perform retrieval using BM25.

    Parameters:
        corpus (list): Tokenized corpus.
        queries (list): List of query strings.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of top-k documents for each query.
    """
    bm25 = BM25Okapi(corpus)
    results = [bm25.get_top_n(query.split(), corpus, n=top_k) for query in queries]
    return results

if __name__ == "__main__":
    # Load and prepare data
    data = load_data("data/preprocessed_crisisfacts_data.csv")
    corpus = prepare_corpus(data, text_column="preprocessed_text")

    # Example queries
    queries = ["wildfire damage", "hurricane impact", "emergency response"]

    # Perform retrieval
    bm25_results = bm25_retrieve(corpus, queries)
    print(bm25_results)