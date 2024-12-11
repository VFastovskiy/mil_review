import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Y residual is constant at iteration")

# Example data: 10 conformers, 20 features
rdf_fingerprints = np.random.rand(100, 200)
conformer_labels = np.random.choice([0, 1], size=100)

# Standardize features
scaler = StandardScaler()
rdf_fingerprints_scaled = scaler.fit_transform(rdf_fingerprints)

# Function to evaluate and select optimal components for PLS
def evaluate_pls(rdf_fingerprints_scaled, conformer_labels, max_components, threshold=0.95):
    explained_variance_ratios = []
    mean_cv_scores = []

    for n_components in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_components)
        pls.fit(rdf_fingerprints_scaled, conformer_labels)
        explained_variance = pls.score(rdf_fingerprints_scaled, conformer_labels)
        explained_variance_ratios.append(explained_variance)

        cv_scores = cross_val_score(pls, rdf_fingerprints_scaled, conformer_labels, cv=5, scoring='r2')
        mean_cv_scores.append(np.mean(cv_scores))

        if explained_variance >= threshold:
            break

    optimal_n_components = len(explained_variance_ratios)
    return optimal_n_components, explained_variance_ratios, mean_cv_scores

# Regularized Ridge Regression
def evaluate_ridge(rdf_fingerprints_scaled, conformer_labels):
    alphas = np.logspace(-3, 3, 10)  # Test a range of alpha values
    mean_cv_scores = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        cv_scores = cross_val_score(ridge, rdf_fingerprints_scaled, conformer_labels, cv=5, scoring='r2')
        mean_cv_scores.append(np.mean(cv_scores))
    best_alpha = alphas[np.argmax(mean_cv_scores)]
    best_cv_score = max(mean_cv_scores)
    return best_alpha, best_cv_score

# Kernel Ridge Regression (Kernel PLS Alternative)
def evaluate_kernel_ridge(rdf_fingerprints_scaled, conformer_labels):
    kernels = ['linear', 'rbf', 'poly']
    best_score = -np.inf
    best_kernel = None
    for kernel in kernels:
        kernel_ridge = KernelRidge(kernel=kernel, alpha=1.0)
        cv_scores = cross_val_score(kernel_ridge, rdf_fingerprints_scaled, conformer_labels, cv=5, scoring='r2')
        mean_score = np.mean(cv_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_kernel = kernel
    return best_kernel, best_score

# Maximum components for PLS
max_components = min(rdf_fingerprints.shape)

# Evaluate PLS
pls_optimal_n, pls_explained, pls_cv_scores = evaluate_pls(rdf_fingerprints_scaled, conformer_labels, max_components)
print(f"PLS Optimal Components: {pls_optimal_n}")
print(f"PLS Mean CV Score: {max(pls_cv_scores):.4f}")

# Evaluate Ridge Regression
ridge_best_alpha, ridge_best_score = evaluate_ridge(rdf_fingerprints_scaled, conformer_labels)
print(f"Ridge Best Alpha: {ridge_best_alpha}")
print(f"Ridge Mean CV Score: {ridge_best_score:.4f}")

# Evaluate Kernel Ridge Regression
kernel_best, kernel_best_score = evaluate_kernel_ridge(rdf_fingerprints_scaled, conformer_labels)
print(f"Kernel Ridge Best Kernel: {kernel_best}")
print(f"Kernel Ridge Mean CV Score: {kernel_best_score:.4f}")

# Visualization of results
methods = ['PLS', 'Ridge Regression', 'Kernel Ridge']
scores = [max(pls_cv_scores), ridge_best_score, kernel_best_score]

plt.figure(figsize=(8, 6))
plt.bar(methods, scores, color=['blue', 'green', 'orange'])
plt.ylabel('Mean Cross-Validated R² Score')
plt.title('Comparison of Latent Component Selection Methods')
plt.ylim(0, 1)
plt.show()
