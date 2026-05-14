import numpy as np
import rasterio
from rasterio.transform import rowcol
import os


def compute_diff(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    emb_a, emb_b: (bands, H, W)
    returns diff_flat: (H*W, bands)
    """
    diff = (emb_b - emb_a).astype(np.float32)
    diff = np.where(np.isfinite(diff), diff, 0.0)
    D, H, W = diff.shape
    return diff.reshape(D, -1).T, H, W


def percentile_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """
    img: (3, H, W) or (H, W, 3)
    returns float32 (H, W, 3) in [0, 1]
    """
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    out = img.astype(np.float32).copy()
    for i in range(3):
        p_lo = np.percentile(out[:, :, i], lo)
        p_hi = np.percentile(out[:, :, i], hi)
        denom = p_hi - p_lo
        if denom < 1e-6:
            denom = 1.0
        out[:, :, i] = np.clip((out[:, :, i] - p_lo) / denom, 0, 1)
    return out


def stretch_band(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    denom = p_hi - p_lo if (p_hi - p_lo) > 1e-6 else 1.0
    return np.clip((arr - p_lo) / denom, 0, 1)


def make_pca_rgb(pca8_map: np.ndarray) -> np.ndarray:
    """
    pca8_map: (H, W, 8)
    returns: (H, W, 3) float32 in [0,1]
    """
    rgb = np.stack([stretch_band(pca8_map[:, :, i]) for i in range(3)], axis=-1)
    return rgb.astype(np.float32)


def save_geotiff(arr: np.ndarray, profile: dict, path: str, dtype="float32"):
    p = profile.copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    p.update({"count": arr.shape[0], "dtype": dtype, "compress": "lzw"})
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype))


def load_geotiff(path: str) -> tuple:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile
        transform = src.transform
    return arr, profile, transform


def pixel_to_geo(row: int, col: int, transform) -> tuple:
    x = transform.c + col * transform.a
    y = transform.f + row * transform.e
    return x, y


def geo_to_pixel(x: float, y: float, transform) -> tuple:
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def image_coords_to_pixel(img_x: float, img_y: float,
                           img_w: int, img_h: int,
                           raster_w: int, raster_h: int) -> tuple:
    col = int(img_x / img_w * raster_w)
    row = int(img_y / img_h * raster_h)
    return row, col


def apply_threshold_and_filter(probs_map: np.ndarray,
                                threshold: float = 0.5,
                                min_pixels: int = 100) -> np.ndarray:
    from scipy import ndimage
    raw = (probs_map > threshold).astype(np.uint8)
    labeled, n = ndimage.label(raw)
    sizes = ndimage.sum(raw, labeled, range(1, n + 1))
    clean = np.zeros_like(raw)
    for oid, sz in enumerate(sizes, start=1):
        if sz >= min_pixels:
            clean[labeled == oid] = 1
    return clean
