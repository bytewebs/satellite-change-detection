import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import io
from PIL import Image, ImageDraw


DARK_BG = "#0f1117"
PANEL_BG = "#1a1a2e"


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def rgb_comparison_plot(rgb_a: np.ndarray, rgb_b: np.ndarray,
                         year_a: int, year_b: int) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)

    for ax, rgb, yr in zip(axes, [rgb_a, rgb_b], [year_a, year_b]):
        ax.imshow(rgb)
        ax.set_title(f"{yr} — Sentinel-2 True Color",
                     color="#ffffff", fontsize=12)
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    return fig_to_bytes(fig)


def pca_falsecolor_plot(pc_rgb: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.imshow(pc_rgb)
    ax.set_title("PCA-3 False Color  —  bright/colorful = changed  |  dark = stable",
                 color="#ffffff", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig_to_bytes(fig)


def overlay_labels_on_image(base_img: np.ndarray,
                              labels: list,
                              dot_radius: int = 5) -> Image.Image:
    """
    base_img: (H, W, 3) float32 [0,1]
    labels: [(row, col, label), ...]
    returns PIL Image with dots drawn
    """
    img_uint8 = (base_img * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8).convert("RGBA")
    draw = ImageDraw.Draw(pil)

    for r, c, label in labels:
        color = (255, 60, 60, 220) if label == 1 else (60, 120, 255, 220)
        draw.ellipse(
            [c - dot_radius, r - dot_radius, c + dot_radius, r + dot_radius],
            fill=color, outline=(255, 255, 255, 180)
        )
    return pil


def probability_heatmap_plotly(probs_map: np.ndarray) -> go.Figure:
    fig = px.imshow(
        probs_map,
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        title="Change Probability Map",
        labels={"color": "P(change)"},
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font_color="#cccccc",
        title_font_color="#ffffff",
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#cccccc"),
            title=dict(font=dict(color="#cccccc")),
        ),
    )
    return fig


def probability_histogram_plotly(probs_map: np.ndarray) -> go.Figure:
    flat = probs_map.flatten()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=flat, nbinsx=80,
        marker_color="#4fc3f7",
        marker_line_color="#0f1117",
        marker_line_width=0.3,
        name="pixels",
    ))
    fig.update_layout(
        title="Change Probability Distribution",
        xaxis_title="P(change)",
        yaxis_title="Pixel count",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font_color="#cccccc",
        title_font_color="#ffffff",
        margin=dict(l=20, r=20, t=40, b=30),
    )
    return fig


def result_summary_plot(rgb_b: np.ndarray, probs_map: np.ndarray,
                         clean_mask: np.ndarray, year_b: int) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(rgb_b)
    axes[0].set_title(f"{year_b} True Color", color="#ffffff")

    im = axes[1].imshow(probs_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Change Probability", color="#ffffff")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(rgb_b)
    overlay = np.zeros((*clean_mask.shape, 4), dtype=np.float32)
    overlay[clean_mask == 1] = [1, 0, 0, 0.55]
    axes[2].imshow(overlay)
    ha = clean_mask.sum() * 100 / 10_000
    axes[2].set_title(f"Change Mask on {year_b} RGB  ({ha:.1f} ha)", color="#ffffff")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor(DARK_BG)

    plt.suptitle("Change Detection Result", color="#ffffff", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig_to_bytes(fig)


def metrics_bar_plot(metrics: dict) -> go.Figure:
    names = [m.capitalize() for m in metrics]
    means = [metrics[m][0] for m in metrics]
    stds  = [metrics[m][1] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True,
                     color="#ffffff", thickness=1.5),
        marker_color=["#4fc3f7", "#81c784", "#ffb74d", "#e57373"],
        text=[f"{v:.3f}" for v in means],
        textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig.update_layout(
        title="Cross-Validation Metrics (5-fold)",
        yaxis=dict(range=[0, 1.1], tickfont=dict(color="#cccccc"),
                   title="Score", title_font=dict(color="#cccccc"),
                   gridcolor="#333333"),
        xaxis=dict(tickfont=dict(color="#cccccc")),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font_color="#cccccc",
        title_font_color="#ffffff",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
