import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set environment variables to limit threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def load_data(file_path):
    """Load preprocessed data from a CSV file."""
    return pd.read_csv(file_path)

def vectorize_text(data, text_column):
    """Convert text data into TF-IDF vectors, handling missing values."""
    data[text_column] = data[text_column].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(data[text_column])

def apply_pca(tfidf_matrix, n_components=2):
    """Reduce dimensions using PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(tfidf_matrix.toarray())

def apply_tsne(tfidf_matrix, n_components=2, perplexity=30, n_iter=300):
    """Reduce dimensions using t-SNE."""
    try:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
        return tsne.fit_transform(tfidf_matrix.toarray())
    except Exception as e:
        print("Error during t-SNE:", e)
        raise

def visualize_clusters(data, title, labels=None):
    """Visualize 2D data clusters using Matplotlib."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# Main pipeline
if __name__ == "__main__":
    input_file = "data/preprocessed_crisisfacts_data.csv"
    data = load_data(input_file)

    tfidf_matrix = vectorize_text(data, text_column="preprocessed_text")

    # Subsample the data for t-SNE
    sample_size = 2000
    if tfidf_matrix.shape[0] > sample_size:
        tfidf_matrix_sample = tfidf_matrix[:sample_size, :]
    else:
        tfidf_matrix_sample = tfidf_matrix

    # Apply PCA
    pca_data = apply_pca(tfidf_matrix, n_components=2)

    # Apply t-SNE on the subsampled data
    tsne_data = apply_tsne(tfidf_matrix_sample, n_components=2)

    # Optional: Apply KMeans clustering for labeling
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())

    # Visualize PCA results
    visualize_clusters(pca_data, title="PCA Visualization", labels=cluster_labels)

    # Visualize t-SNE results
    visualize_clusters(tsne_data, title="t-SNE Visualization")