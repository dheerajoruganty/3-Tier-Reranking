import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load preprocessed data from a CSV file."""
    return pd.read_csv(file_path)

def vectorize_text(data, text_column):
    """Convert text data into TF-IDF vectors, handling missing values."""
    data[text_column] = data[text_column].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(data[text_column]), vectorizer

def apply_kmeans(tfidf_matrix, n_clusters):
    """Cluster data using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    return cluster_labels, kmeans

def apply_dbscan(tfidf_matrix, eps=0.5, min_samples=5):
    """Cluster data using DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(tfidf_matrix)
    return cluster_labels, dbscan

def analyze_clusters(data, cluster_labels, method="KMeans"):
    """Analyze clustering results using silhouette score and visualization."""
    # Compute silhouette score
    if method == "KMeans" and len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Silhouette Score ({method}): {silhouette_avg}")
    elif method == "DBSCAN":
        noise_points = sum(cluster_labels == -1)
        print(f"Number of Noise Points ({method}): {noise_points}")

    # Visualize clusters
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(data.toarray())
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f"Cluster Visualization ({method})")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True)
    plt.show()

# Main pipeline
if __name__ == "__main__":
    # Load preprocessed data
    input_file = "data/preprocessed_crisisfacts_data.csv"
    data = load_data(input_file)

    # Step 1: Vectorize text data
    tfidf_matrix, vectorizer = vectorize_text(data, text_column="preprocessed_text")

    # Step 2: Apply KMeans
    n_clusters = 5  # Set the number of clusters for KMeans
    kmeans_labels, kmeans_model = apply_kmeans(tfidf_matrix, n_clusters=n_clusters)
    analyze_clusters(tfidf_matrix, kmeans_labels, method="KMeans")

    # Step 3: Apply DBSCAN
    eps = 0.3  # Adjust epsilon (neighborhood size)
    min_samples = 10  # Minimum samples per cluster
    dbscan_labels, dbscan_model = apply_dbscan(tfidf_matrix, eps=eps, min_samples=min_samples)
    analyze_clusters(tfidf_matrix, dbscan_labels, method="DBSCAN")