import pandas as pd
import numpy as np
import argparse
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed data from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    try:
        print(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def vectorize_text(data: pd.DataFrame, text_column: str):
    """
    Convert text data into TF-IDF vectors.

    Args:
        data (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text.

    Returns:
        tuple: A tuple containing:
            - tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix.
            - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    print("Vectorizing text data...")
    data[text_column] = data[text_column].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data[text_column])
    print(f"Vectorization completed. Vocabulary size: {len(vectorizer.vocabulary_)}")
    return tfidf_matrix, vectorizer


def apply_kmeans(tfidf_matrix, n_clusters: int):
    """
    Cluster data using KMeans.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix of text data.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        tuple: A tuple containing:
            - cluster_labels (ndarray): Cluster labels for each sample.
            - kmeans (KMeans): Trained KMeans model.
    """
    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    print("KMeans clustering completed.")
    return cluster_labels, kmeans


def apply_dbscan(tfidf_matrix, eps: float = 0.5, min_samples: int = 5):
    """
    Cluster data using DBSCAN.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix of text data.
        eps (float): The maximum distance between two samples for clustering.
        min_samples (int): Minimum number of samples per cluster.

    Returns:
        tuple: A tuple containing:
            - cluster_labels (ndarray): Cluster labels for each sample.
            - dbscan (DBSCAN): Trained DBSCAN model.
    """
    print(f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = dbscan.fit_predict(tfidf_matrix)
    print("DBSCAN clustering completed.")
    return cluster_labels, dbscan


def analyze_clusters(data, cluster_labels, method: str = "KMeans"):
    """
    Analyze and visualize clustering results using PCA for dimensionality reduction.

    Args:
        data (scipy.sparse.csr_matrix): TF-IDF matrix of text data.
        cluster_labels (ndarray): Cluster labels assigned to each data point.
        method (str): Clustering method used ("KMeans" or "DBSCAN").

    Returns:
        None
    """
    print(f"Analyzing clusters using {method}...")

    # Compute silhouette score for KMeans
    if method == "KMeans" and len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Silhouette Score ({method}): {silhouette_avg:.4f}")
    elif method == "DBSCAN":
        noise_points = sum(cluster_labels == -1)
        print(f"Number of Noise Points ({method}): {noise_points}")

    # PCA for dimensionality reduction and visualization
    print("Reducing dimensions using PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(data.toarray())

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    plt.title(f"Cluster Visualization ({method})")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True)
    plt.colorbar(label="Cluster Label")
    plt.show()


def main():
    """
    Main pipeline for text clustering using KMeans and DBSCAN.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Text Clustering with KMeans and DBSCAN"
    )
    parser.add_argument(
        "--data_file", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=5, help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--eps", type=float, default=0.3, help="Epsilon value for DBSCAN clustering"
    )
    parser.add_argument(
        "--min_samples", type=int, default=10, help="Min samples for DBSCAN clustering"
    )
    args = parser.parse_args()

    # Load preprocessed data
    print("Starting clustering pipeline...")
    data = load_data(args.data_file)

    # Step 1: Vectorize text data
    print("\nStep 1: Text Vectorization")
    tfidf_matrix, vectorizer = vectorize_text(data, text_column="text")

    # Step 2: Apply KMeans Clustering
    print("\nStep 2: KMeans Clustering")
    kmeans_labels, kmeans_model = apply_kmeans(tfidf_matrix, n_clusters=args.n_clusters)
    analyze_clusters(tfidf_matrix, kmeans_labels, method="KMeans")

    # Step 3: Apply DBSCAN Clustering
    print("\nStep 3: DBSCAN Clustering")
    dbscan_labels, dbscan_model = apply_dbscan(
        tfidf_matrix, eps=args.eps, min_samples=args.min_samples
    )
    analyze_clusters(tfidf_matrix, dbscan_labels, method="DBSCAN")

    print("\nClustering pipeline completed successfully!")


if __name__ == "__main__":
    main()
