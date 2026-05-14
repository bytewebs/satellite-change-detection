import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")


def build_training_data(pca8_map: np.ndarray, labels: list) -> tuple:
    H, W, _ = pca8_map.shape
    valid = [(r, c, l) for r, c, l in labels if 0 <= r < H and 0 <= c < W]
    X = np.array([pca8_map[r, c, :] for r, c, _ in valid])
    y = np.array([l for _, _, l in valid])
    return X, y, valid


def train_classifier(X: np.ndarray, y: np.ndarray,
                     classifier: str = "LogisticRegression") -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classifier == "LogisticRegression":
        clf = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=min(5, min(np.bincount(y))),
                               shuffle=True, random_state=42),
            scoring="f1",
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
        )
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=min(5, min(np.bincount(y))),
                               shuffle=True, random_state=42),
            scoring="f1",
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
        )

    clf.fit(X_scaled, y)

    n_folds = min(5, min(np.bincount(y)))
    cv_results = cross_validate(
        clf, X_scaled, y,
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0),
        scoring=["accuracy", "f1", "precision", "recall"],
    )

    metrics = {
        m: (cv_results[f"test_{m}"].mean(), cv_results[f"test_{m}"].std())
        for m in ["accuracy", "f1", "precision", "recall"]
    }

    return clf, scaler, metrics


def confusion_matrix_plot(clf, scaler, X, y) -> bytes:
    X_sc = scaler.transform(X)
    y_pred = clf.predict(X_sc)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No-Change", "Change"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (training set)", color="#ffffff")
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    for text in ax.texts:
        text.set_color("white")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def run_inference(clf, scaler, pca8_map: np.ndarray,
                  chunk: int = 1_000_000,
                  progress_cb=None) -> np.ndarray:
    H, W, n_comp = pca8_map.shape
    flat = pca8_map.reshape(-1, n_comp)
    n = flat.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    n_chunks = (n + chunk - 1) // chunk

    for i, s in enumerate(range(0, n, chunk)):
        e = min(s + chunk, n)
        probs[s:e] = clf.predict_proba(scaler.transform(flat[s:e]))[:, 1]
        if progress_cb:
            progress_cb((i + 1) / n_chunks)

    return probs.reshape(H, W)
