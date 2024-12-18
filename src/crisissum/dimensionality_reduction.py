import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

# --------------------------- #
# Environment Variables Setup #
# --------------------------- #
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --------------------------- #
# Function Definitions        #
# --------------------------- #


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def vectorize_text(data: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Convert text data into TF-IDF vectors, handling missing values.

    Args:
        data (pd.DataFrame): Input data.
        text_column (str): Name of the column containing text.

    Returns:
        scipy.sparse.csr_matrix: TF-IDF matrix.
    """
    data[text_column] = data[text_column].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(data[text_column])


def apply_pca(tfidf_matrix, n_components=2):
    """
    Reduce dimensions using PCA.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix.
        n_components (int): Number of components for PCA.

    Returns:
        np.ndarray: Reduced data.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(tfidf_matrix.toarray())


def apply_tsne(tfidf_matrix, n_components=2, perplexity=30, n_iter=300):
    """
    Reduce dimensions using t-SNE.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix.
        n_components (int): Number of dimensions for t-SNE.
        perplexity (int): t-SNE perplexity parameter.
        n_iter (int): Number of iterations for optimization.

    Returns:
        np.ndarray: Reduced data.
    """
    try:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
        )
        return tsne.fit_transform(tfidf_matrix.toarray())
    except Exception as e:
        print("Error during t-SNE:", e)
        raise


def visualize_clusters(data, title: str, labels=None):
    """
    Visualize 2D data clusters using Matplotlib.

    Args:
        data (np.ndarray): 2D data points for visualization.
        title (str): Plot title.
        labels (np.ndarray, optional): Cluster labels for coloring.

    Returns:
        None
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        data[:, 0], data[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7
    )
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


# --------------------------- #
# Main Pipeline               #
# --------------------------- #

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Text Data Visualization and Clustering"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input preprocessed CSV file (e.g., preprocessed_crisisfacts_data.csv).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the column containing preprocessed text (default: text).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans (default: 5).",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=2,
        help="Number of dimensions for PCA (default: 2).",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="Perplexity parameter for t-SNE (default: 30).",
    )
    parser.add_argument(
        "--tsne_iter",
        type=int,
        default=300,
        help="Number of iterations for t-SNE (default: 300).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of samples to use for t-SNE visualization (default: 2000).",
    )

    args = parser.parse_args()

    # Load data
    data = load_data(args.input_file)

    # Vectorize text data
    tfidf_matrix = vectorize_text(data, text_column=args.text_column)

    # Subsample data for t-SNE
    if tfidf_matrix.shape[0] > args.sample_size:
        tfidf_matrix_sample = tfidf_matrix[: args.sample_size, :]
    else:
        tfidf_matrix_sample = tfidf_matrix

    # Apply PCA
    pca_data = apply_pca(tfidf_matrix, n_components=args.pca_components)

    # Apply t-SNE on the subsampled data
    tsne_data = apply_tsne(
        tfidf_matrix_sample,
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_iter,
    )

    # Apply KMeans clustering for labeling
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())

    # Visualize PCA results
    visualize_clusters(pca_data, title="PCA Visualization", labels=cluster_labels)

    # Visualize t-SNE results
    visualize_clusters(tsne_data, title="t-SNE Visualization")
