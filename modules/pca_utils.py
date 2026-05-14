import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io


def fit_pca(diff_flat: np.ndarray, n_components: int = 8,
            subsample: int = 500, random_state: int = 42) -> PCA:
    n_pixels = diff_flat.shape[0]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_pixels, max(n_pixels // subsample, n_components * 10), replace=False)
    sample = diff_flat[idx]
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(sample)
    return pca


def fit_pca_explore(diff_flat: np.ndarray, n_components: int = 20,
                    subsample: int = 500, random_state: int = 42) -> PCA:
    n_pixels = diff_flat.shape[0]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_pixels, max(n_pixels // subsample, n_components * 10), replace=False)
    sample = diff_flat[idx]
    pca = PCA(n_components=min(n_components, diff_flat.shape[1]), random_state=random_state)
    pca.fit(sample)
    return pca


def project_pca(pca: PCA, diff_flat: np.ndarray,
                H: int, W: int, chunk: int = 1_000_000) -> np.ndarray:
    n_pixels = diff_flat.shape[0]
    n_comp = pca.n_components
    out = np.zeros((n_pixels, n_comp), dtype=np.float32)
    for s in range(0, n_pixels, chunk):
        e = min(s + chunk, n_pixels)
        out[s:e] = pca.transform(diff_flat[s:e])
    return out.reshape(H, W, n_comp)


def variance_plot(pca_explore: PCA, n_selected: int = 8) -> bytes:
    cumvar = np.cumsum(pca_explore.explained_variance_ratio_) * 100
    n = len(cumvar)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.plot(range(1, n + 1), cumvar, marker="o", color="#4fc3f7",
            linewidth=2, markersize=5)
    ax.axvline(x=n_selected, color="#ef5350", linestyle="--", linewidth=1.5,
               label=f"Selected: {n_selected} → {cumvar[n_selected-1]:.1f}% variance")
    ax.fill_between(range(1, n + 1), cumvar, alpha=0.12, color="#4fc3f7")

    ax.set_xlabel("PCA components", color="#cccccc")
    ax.set_ylabel("Cumulative variance (%)", color="#cccccc")
    ax.set_title("Variance explained by PCA components", color="#ffffff", fontsize=13)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1e1e1e", edgecolor="#444444", labelcolor="#ffffff")
    ax.grid(True, alpha=0.15, color="#555555")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
