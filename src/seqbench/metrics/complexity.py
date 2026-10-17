# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Complexity metrics for analyzing sequence data and model representations.
"""

import numpy as np
import warnings
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "SeqBench complexity metrics require matplotlib, scipy, and scikit-learn. "
        "Install SeqBench with the metrics extra: pip install 'seqbench[metrics]'."
    ) from exc

warnings.filterwarnings("ignore")

from seqbench.utils.io import get_logger

logging = get_logger(__name__)


class EmbeddingAnalyzer:
    """
    Comprehensive analysis of embeddings for geometric and structural properties.
    Designed for comparing embeddings from different modalities (vector, spike, continuous).
    """

    def __init__(self, embeddings, labels, original_data=None):
        """
        Initialize analyzer with embeddings and corresponding labels.

        Args:
            embeddings (np.ndarray): Embedding vectors of shape (n_samples, n_features)
            labels (np.ndarray): Ground truth labels of shape (n_samples,)
            original_data (np.ndarray, optional): Original high-dimensional data
        """
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)
        self.original_data = original_data
        self.n_samples, self.n_features = self.embeddings.shape
        self.n_classes = len(np.unique(self.labels))

    def participation_ratio(self):
        """
        Calculate participation ratio - measures how many dimensions actively contribute.
        Higher values indicate more distributed representations.

        Returns:
            float: Participation ratio
        """
        # Center the data
        centered = self.embeddings - np.mean(self.embeddings, axis=0)

        # Get singular values
        _, s, _ = np.linalg.svd(centered, full_matrices=False)

        # Calculate participation ratio
        s_squared = s**2
        pr = (np.sum(s_squared) ** 2) / np.sum(s_squared**2)

        return pr

    def local_intrinsic_dimensionality(self, k=10):
        """
        Estimate local intrinsic dimensionality using k-nearest neighbors.

        Args:
            k (int): Number of neighbors to consider

        Returns:
            dict: Mean LID and per-sample LID values
        """
        from sklearn.neighbors import NearestNeighbors

        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        # Remove self (first neighbor)
        distances = distances[:, 1:]

        # Calculate LID for each point
        lids = []
        for i in range(len(distances)):
            d = distances[i]
            d = d[d > 1e-10]  # Avoid division by zero
            if len(d) > 1:
                r_k = d[-1]  # k-th nearest neighbor distance
                if r_k > 0:
                    lid = (k - 1) / np.sum(np.log(r_k / d[:-1]))
                    lids.append(lid)
                else:
                    lids.append(0)
            else:
                lids.append(0)

        lids = np.array(lids)
        return {"mean_lid": np.mean(lids), "std_lid": np.std(lids), "per_sample_lid": lids}

    def pca_explained_variance_curve(self, plot=False):
        """
        Compute PCA explained variance curve.

        Args:
            plot (bool): Whether to plot the curve

        Returns:
            dict: PCA results including explained variance ratios
        """
        pca = PCA()
        pca.fit(self.embeddings)

        # self.plot_pca_scatter(self.embeddings, labels=self.labels, n_components=3)

        explained_var_ratio = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var_ratio)

        # Find dimensions needed for 90% and 95% variance
        dim_90 = np.argmax(cumsum_var >= 0.90) + 1
        dim_95 = np.argmax(cumsum_var >= 0.95) + 1

        if plot:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(explained_var_ratio[:20], "bo-")
            plt.title("Explained Variance Ratio (First 20 Components)")
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance Ratio")

            plt.subplot(1, 2, 2)
            plt.plot(cumsum_var[:20], "ro-")
            plt.axhline(y=0.90, color="k", linestyle="--", alpha=0.5, label="90%")
            plt.axhline(y=0.95, color="k", linestyle="--", alpha=0.5, label="95%")
            plt.title("Cumulative Explained Variance")
            plt.xlabel("Principal Component")
            plt.ylabel("Cumulative Explained Variance")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            "explained_variance_ratio": explained_var_ratio,
            "cumulative_variance": cumsum_var,
            "dimensions_90_percent": dim_90,
            "dimensions_95_percent": dim_95,
        }

    def plot_pca_scatter(self, X, labels=None, n_components=2, title="PCA Scatter Plot"):
        """
        Perform PCA and plot the first two or three principal components as a scatter plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        labels : array-like, shape (n_samples,), optional
            Class labels for coloring points. If None, all points are the same color.
        n_components : int, default=2
            Number of PCA components to compute (2 or 3 supported for plotting).
        title : str
            Title of the plot.
        """

        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3 for scatter plotting")

        from seqbench.metrics.spike_utils import smooth_rates, zscore_timewise, run_pca_time_by_neuron

        fr = smooth_rates(self.embeddings, 1, sigma_ms=10)
        fr_z = zscore_timewise(fr)

        # Fit PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(fr_z)
        import seaborn as sns

        cmap = sns.color_palette("tab10", n_colors=len(np.unique(labels)), as_cmap=True)

        if n_components == 2:
            # 2D scatter plot
            plt.figure(figsize=(8, 6))
            if labels is not None:
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.7)
                plt.legend(*scatter.legend_elements(), title="Classes")
            else:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.title(title)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

        elif n_components == 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            if labels is not None:
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap=cmap, alpha=0.7)
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
            ax.set_title(title)
            plt.show()

    def pairwise_distance_analysis(self, distance_metric="euclidean"):
        """
        Analyze pairwise distance distributions in embedding space.

        Args:
            distance_metric (str): Distance metric to use

        Returns:
            dict: Distance statistics and distribution
        """
        distances = pdist(self.embeddings, metric=distance_metric)

        return {
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances),
            "median_distance": np.median(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "distance_distribution": distances,
        }

    def distance_correlation_with_original(self, distance_metric="euclidean"):
        """
        Compare distance relationships between original data and embeddings.

        Args:
            distance_metric (str): Distance metric to use

        Returns:
            float: Correlation coefficient between distance matrices
        """
        if self.original_data is None:
            return None

        # Calculate distance matrices
        orig_distances = pdist(self.original_data, metric=distance_metric)
        embed_distances = pdist(self.embeddings, metric=distance_metric)

        # Calculate correlation
        correlation, p_value = pearsonr(orig_distances, embed_distances)

        return {"correlation": correlation, "p_value": p_value}

    def nearest_neighbor_preservation(self, k=10):
        """
        Calculate how well k-nearest neighbors are preserved from original to embedding space.

        Args:
            k (int): Number of neighbors to consider

        Returns:
            dict: NN preservation statistics
        """
        if self.original_data is None:
            return None

        # Find k-NN in original space
        nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(self.original_data)
        _, indices_orig = nbrs_orig.kneighbors(self.original_data)

        # Find k-NN in embedding space
        nbrs_embed = NearestNeighbors(n_neighbors=k + 1).fit(self.embeddings)
        _, indices_embed = nbrs_embed.kneighbors(self.embeddings)

        # Calculate preservation for each point
        preservation_scores = []
        for i in range(len(indices_orig)):
            orig_neighbors = set(indices_orig[i][1:])  # Remove self
            embed_neighbors = set(indices_embed[i][1:])  # Remove self

            intersection = len(orig_neighbors.intersection(embed_neighbors))
            preservation_scores.append(intersection / k)

        return {
            "mean_preservation": np.mean(preservation_scores),
            "std_preservation": np.std(preservation_scores),
            "per_sample_preservation": np.array(preservation_scores),
        }

    def silhouette_analysis(self):
        """
        Calculate silhouette scores for class separability.

        Returns:
            dict: Silhouette analysis results
        """
        sil_score = silhouette_score(self.embeddings, self.labels)

        from sklearn.metrics import silhouette_samples

        sample_scores = silhouette_samples(self.embeddings, self.labels)

        # Per-class silhouette scores
        class_scores = {}
        for label in np.unique(self.labels):
            mask = self.labels == label
            class_scores[label] = np.mean(sample_scores[mask])

        return {
            "overall_silhouette": sil_score,
            "per_class_silhouette": class_scores,
            "per_sample_silhouette": sample_scores,
        }

    def calinski_harabasz_index(self):
        """
        Calculate Calinski-Harabasz index for cluster quality.

        Returns:
            float: Calinski-Harabasz index
        """
        return calinski_harabasz_score(self.embeddings, self.labels)

    def linear_separability_coefficient(self, test_size=0.3, random_state=42):
        """
        Measure linear separability using logistic regression accuracy.

        Args:
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed

        Returns:
            dict: Linear separability results
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.embeddings, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
            )
        except:
            logging.error("Failed to split data in linear separability analysis - returning NaN")
            return {"accuracy": np.nan, "classification_report": None}

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        return {"accuracy": accuracy, "classification_report": classification_report(y_test, y_pred, output_dict=True)}

    def cluster_quality_metrics(self, n_clusters=None):
        """
        Calculate cluster quality metrics using unsupervised clustering.

        Args:
            n_clusters (int): Number of clusters (defaults to number of unique labels)

        Returns:
            dict: Clustering quality metrics
        """
        if n_clusters is None:
            n_clusters = self.n_classes

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)

        # Calculate metrics
        ari = adjusted_rand_score(self.labels, cluster_labels)
        nmi = normalized_mutual_info_score(self.labels, cluster_labels)

        return {"adjusted_rand_index": ari, "normalized_mutual_info": nmi, "cluster_labels": cluster_labels}

    def comprehensive_analysis(self, plot_pca=False, k_neighbors=10):
        """
        Run all analyses and return comprehensive results.

        Args:
            plot_pca (bool): Whether to plot PCA curves
            k_neighbors (int): Number of neighbors for k-NN based metrics

        Returns:
            dict: All analysis results
        """
        results = {}

        print("Running Core Geometric Measures...")

        # Core Geometric Measures
        results["participation_ratio"] = self.participation_ratio()
        results["local_intrinsic_dimensionality"] = self.local_intrinsic_dimensionality(k=k_neighbors)
        results["pca_analysis"] = self.pca_explained_variance_curve(plot=plot_pca)
        results["distance_analysis"] = self.pairwise_distance_analysis()

        if self.original_data is not None:
            results["distance_correlation"] = self.distance_correlation_with_original()
            results["nn_preservation"] = self.nearest_neighbor_preservation(k=k_neighbors)

        print("Running Clustering and Separability Metrics...")

        # Clustering and Separability Metrics
        results["silhouette_analysis"] = self.silhouette_analysis()
        results["calinski_harabasz_index"] = self.calinski_harabasz_index()
        results["linear_separability"] = self.linear_separability_coefficient()
        results["cluster_quality"] = self.cluster_quality_metrics()

        return results

    def print_summary(self, results):
        """
        Print a formatted summary of results.

        Args:
            results (dict): Results from comprehensive_analysis()
        """
        print("=" * 60)
        print("EMBEDDING ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 CORE GEOMETRIC MEASURES")
        print("-" * 40)
        print(f"Participation Ratio: {results['participation_ratio']:.4f}")
        print(
            f"Mean Local Intrinsic Dimensionality: {results['local_intrinsic_dimensionality']['mean_lid']:.4f} ± {results['local_intrinsic_dimensionality']['std_lid']:.4f}"
        )
        print(f"Dimensions for 90% variance: {results['pca_analysis']['dimensions_90_percent']}")
        print(f"Dimensions for 95% variance: {results['pca_analysis']['dimensions_95_percent']}")
        print(
            f"Mean pairwise distance: {results['distance_analysis']['mean_distance']:.4f} ± {results['distance_analysis']['std_distance']:.4f}"
        )

        if "distance_correlation" in results and results["distance_correlation"] is not None:
            print(f"Distance correlation with original: {results['distance_correlation']['correlation']:.4f}")

        if "nn_preservation" in results and results["nn_preservation"] is not None:
            print(
                f"k-NN preservation: {results['nn_preservation']['mean_preservation']:.4f} ± {results['nn_preservation']['std_preservation']:.4f}"
            )

        print(f"\n🎯 CLUSTERING & SEPARABILITY METRICS")
        print("-" * 40)
        print(f"Silhouette Score: {results['silhouette_analysis']['overall_silhouette']:.4f}")
        print(f"Calinski-Harabasz Index: {results['calinski_harabasz_index']:.2f}")
        print(f"Linear Separability (Accuracy): {results['linear_separability']['accuracy']:.4f}")
        print(f"Adjusted Rand Index: {results['cluster_quality']['adjusted_rand_index']:.4f}")
        print(f"Normalized Mutual Information: {results['cluster_quality']['normalized_mutual_info']:.4f}")

        print("\n" + "=" * 60)


# Example usage and testing function
def example_usage():
    """
    Example of how to use the EmbeddingAnalyzer with synthetic data.
    """
    from sklearn.datasets import make_classification
    from sklearn.manifold import TSNE

    # Generate synthetic high-dimensional data
    X_orig, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Create embeddings using t-SNE
    embeddings = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_orig)

    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(embeddings, y, original_data=X_orig)

    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis(plot_pca=True)

    # Print summary
    analyzer.print_summary(results)

    return analyzer, results


if __name__ == "__main__":
    # Run example
    # analyzer, results = example_usage()
    pass
