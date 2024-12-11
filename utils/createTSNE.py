import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np


def run_pca(X, n_components=50):
    """
    Perform PCA on the input feature matrix.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - n_components: Number of components for PCA.

    Returns:
    - X_reduced: PCA-transformed feature matrix.
    """
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def subsample_data(X, y, sample_size=1000):
    """
    Subsample the data to a specified size.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Labels (numpy array or pandas Series).
    - sample_size: Number of samples to randomly select.

    Returns:
    - X_sample: Subsampled feature matrix.
    - y_sample: Subsampled labels.
    """
    indices = np.random.choice(range(len(X)), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample, y_sample


def create_tsne_plot(X, y, output_path="tsne_plot.png"):
    """
    Create a t-SNE plot from training data and save it as an image.

    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Labels (numpy array or pandas Series; 1 for active, 0 for inactive).
    - output_path: Path to save the plot (default: "tsne_plot.png").
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions with PCA if necessary
    if X_scaled.shape[1] > 50:
        X_scaled = run_pca(X)

    # Subsample data
    sample_size = min(1000, len(X))  # Adjust sample size if data has fewer points
    X_sample, y_sample = subsample_data(X_scaled, y, sample_size)

    # Run t-SNE on subsampled data
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_embedded = tsne.fit_transform(X_sample)

    # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(X_embedded, columns=["Dim1", "Dim2"])
    tsne_df["Class"] = y_sample

    # Plot the t-SNE
    plt.figure(figsize=(10, 8))
    for label, color in zip([0, 1], ["red", "blue"]):
        subset = tsne_df[tsne_df["Class"] == label]
        plt.scatter(subset["Dim1"], subset["Dim2"], label=f"Class {label}", alpha=0.6, s=50, c=color)

    plt.title("t-SNE Visualization of Active vs Inactive Classes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Class")
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"t-SNE plot saved to {output_path}.")
