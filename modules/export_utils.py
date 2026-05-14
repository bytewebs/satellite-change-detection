import numpy as np
import rasterio
import json
import csv
import io
import os
import tempfile


def export_geotiff_bytes(arr: np.ndarray, profile: dict, dtype="float32") -> bytes:
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    p = profile.copy()
    p.update({"count": arr.shape[0], "dtype": dtype,
              "compress": "lzw", "driver": "GTiff"})

    # Write to a real temp file — rasterio MemoryFile + BytesIO is unreliable
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **p) as dst:
            dst.write(arr.astype(dtype))
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def labels_to_csv(labels: list) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["row", "col", "label"])
    writer.writerows(labels)
    return buf.getvalue()


def labels_to_json(labels: list) -> str:
    data = [{"row": int(r), "col": int(c), "label": int(l)}
            for r, c, l in labels]
    return json.dumps(data, indent=2)


def labels_to_pylist(labels: list) -> str:
    lines = ["manual_labels = ["]
    for r, c, l in labels:
        lines.append(f"    ({r}, {c}, {l}),")
    lines.append("]")
    return "\n".join(lines)


def save_file(data: bytes, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)