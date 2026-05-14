import streamlit as st
import numpy as np
import os
import json
import io
from PIL import Image

# ── page config must be first ──────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── dark theme injection ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e0e0e0; }
    .stButton>button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white; border: none; border-radius: 6px;
        padding: 0.45rem 1.2rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .step-header {
        background: linear-gradient(90deg, #1a237e22, #0d47a122);
        border-left: 3px solid #4fc3f7;
        padding: 0.5rem 1rem; border-radius: 0 6px 6px 0;
        margin: 1rem 0 0.75rem 0;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 8px;
        padding: 0.75rem 1rem; text-align: center;
        border: 1px solid #2a2a3e;
    }
    .label-badge-change {
        background: #c62828; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.82rem;
    }
    .label-badge-nochange {
        background: #1565c0; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.82rem;
    }
    div[data-testid="stExpander"] { border: 1px solid #2a2a3e !important; }
</style>
""", unsafe_allow_html=True)

# ── imports ─────────────────────────────────────────────────────────────────
from modules.gee_utils import (authenticate_gee, initialize_gee,
                                get_aef_image, get_s2_rgb,
                                download_image_as_array, save_tif, load_tif,
                                aoi_from_bbox, aoi_from_geojson)
from modules.raster_utils import (compute_diff, percentile_stretch,
                                   make_pca_rgb, save_geotiff, load_geotiff,
                                   apply_threshold_and_filter,
                                   image_coords_to_pixel)
from modules.pca_utils import (fit_pca, fit_pca_explore, project_pca,
                                variance_plot)
from modules.model_utils import (build_training_data, train_classifier,
                                  confusion_matrix_plot, run_inference)
from modules.visualization_utils import (rgb_comparison_plot,
                                          pca_falsecolor_plot,
                                          overlay_labels_on_image,
                                          probability_heatmap_plotly,
                                          probability_histogram_plotly,
                                          result_summary_plot,
                                          metrics_bar_plot)
from modules.export_utils import (export_geotiff_bytes, labels_to_csv,
                                   labels_to_json, labels_to_pylist)

# ── session state defaults ───────────────────────────────────────────────────
DEFAULTS = {
    "gee_ready": False,
    "data_fetched": False,
    "pca_done": False,
    "model_trained": False,
    "inference_done": False,
    "labels": [],
    "emb_a": None, "emb_b": None,
    "rgb_a": None, "rgb_b": None,
    "profile": None, "transform": None,
    "diff_flat": None, "H": None, "W": None,
    "pca_model": None, "pca_explore": None,
    "pca8_map": None, "pc_rgb": None,
    "clf": None, "scaler": None, "metrics": None,
    "probs_map": None, "clean_mask": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

DATA_DIR = "data"
OUT_DIR  = os.path.join(DATA_DIR, "outputs")
TMP_DIR  = os.path.join(DATA_DIR, "temp")
for d in [OUT_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("Satellite Change Detection via AEF Embeddings")
    st.divider()

    # ── GEE settings ─────────────────────────────────────────────────────
    st.markdown("### GEE Settings")
    gee_project = st.text_input("GEE Project ID", placeholder="your-project-id")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Authenticate", use_container_width=True):
            ok, msg = authenticate_gee(gee_project)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        if st.button("Initialize", use_container_width=True):
            ok, msg = initialize_gee(gee_project)
            st.session_state.gee_ready = ok
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if st.session_state.gee_ready:
        st.success("✅ GEE ready")

    st.divider()

    # ── AOI settings ──────────────────────────────────────────────────────
    st.markdown("### AOI Settings")
    aoi_mode = st.radio("Input method", ["Bounding Box", "GeoJSON paste"],
                         horizontal=True)

    if aoi_mode == "Bounding Box":
        st.caption("Default: Fredericton, NB")
        col1, col2 = st.columns(2)
        with col1:
            west  = st.number_input("West",  value=-66.85, format="%.4f")
            south = st.number_input("South", value=45.85,  format="%.4f")
        with col2:
            east  = st.number_input("East",  value=-66.45, format="%.4f")
            north = st.number_input("North", value=46.05,  format="%.4f")
        aoi_geojson = None
    else:
        aoi_geojson = st.text_area("Paste GeoJSON geometry",
                                    height=120,
                                    placeholder='{"type":"Polygon","coordinates":[...]}')
        west = south = east = north = None

    st.divider()

    # ── Year settings ─────────────────────────────────────────────────────
    st.markdown("### Year Settings")
    col1, col2 = st.columns(2)
    with col1:
        year_a = st.selectbox("Year A (baseline)", list(range(2017, 2025)),
                               index=3)
    with col2:
        year_b = st.selectbox("Year B (compare)", list(range(2018, 2026)),
                               index=6)

    st.divider()

    # ── PCA settings ──────────────────────────────────────────────────────
    st.markdown("### PCA Settings")
    n_pca = st.slider("PCA components", min_value=3, max_value=20,
                       value=8, step=1)

    st.divider()

    # ── Model settings ────────────────────────────────────────────────────
    st.markdown("### Model Settings")
    classifier_choice = st.selectbox("Classifier",
                                      ["LogisticRegression", "RandomForest"])
    threshold     = st.slider("Change threshold", 0.1, 0.9, 0.5, 0.05)
    min_obj_px    = st.slider("Min object size (pixels)", 10, 500, 100, 10)
    st.caption(f"Min object ≈ {min_obj_px * 100 / 10_000:.2f} ha at 10 m")

    st.divider()

    # ── Label settings ────────────────────────────────────────────────────
    st.markdown("### Label Settings")
    label_mode = st.radio("Current label", ["CHANGE", "NO-CHANGE"],
                           horizontal=True)
    current_label = 1 if label_mode == "CHANGE" else 0

    n_ch = sum(l for _, _, l in st.session_state.labels)
    n_nc = len(st.session_state.labels) - n_ch
    st.markdown(
        f'<span class="label-badge-change">🔴 Change: {n_ch}</span>  '
        f'<span class="label-badge-nochange">🔵 No-change: {n_nc}</span>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("v1.0 · AEF + GEE · Fredericton demo")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛰️ Satellite Change Detection")
st.markdown(
    "End-to-end change detection using **Google AEF embeddings** (64-d per pixel, 10 m). "
    "No GPU required — embeddings are pre-computed by Google."
)

progress_steps = ["Fetch", "Visualize", "PCA", "Label", "Train", "Infer", "Export"]
step_cols = st.columns(len(progress_steps))
step_status = {
    "Fetch":     st.session_state.data_fetched,
    "Visualize": st.session_state.data_fetched,
    "PCA":       st.session_state.pca_done,
    "Label":     len(st.session_state.labels) >= 4,
    "Train":     st.session_state.model_trained,
    "Infer":     st.session_state.inference_done,
    "Export":    st.session_state.inference_done,
}
for col, step in zip(step_cols, progress_steps):
    icon = "✅" if step_status[step] else "⬜"
    col.markdown(f"<div style='text-align:center;font-size:0.8rem'>{icon}<br>{step}</div>",
                 unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH DATA
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header"><b>Step 1 — Fetch AEF Embeddings & Sentinel-2 RGB</b></div>',
            unsafe_allow_html=True)

fetch_col1, fetch_col2 = st.columns([2, 1])
with fetch_col1:
    st.markdown(
        "Pulls **64-band AEF embeddings** and **Sentinel-2 true-color** for both years "
        "from Google Earth Engine and downloads them locally."
    )
with fetch_col2:
    fetch_btn = st.button("🌍 Fetch Data", use_container_width=True,
                           disabled=not st.session_state.gee_ready)


if fetch_btn:

    aef_a_path = os.path.join(TMP_DIR, "aef_a.tif")
    aef_b_path = os.path.join(TMP_DIR, "aef_b.tif")
    rgb_a_path = os.path.join(TMP_DIR, "rgb_a.tif")
    rgb_b_path = os.path.join(TMP_DIR, "rgb_b.tif")

    # =========================================================
    # LOAD CACHED FILES INSTEAD OF REDOWNLOADING
    # =========================================================

    if (
        os.path.exists(aef_a_path)
        and os.path.exists(aef_b_path)
        and os.path.exists(rgb_a_path)
        and os.path.exists(rgb_b_path)
    ):

        st.success("Loading cached local TIFFs...")

        emb_a, profile, transform = load_tif(aef_a_path)
        emb_b, _, _ = load_tif(aef_b_path)

        rgb_a_raw, _, _ = load_tif(rgb_a_path)
        rgb_b_raw, _, _ = load_tif(rgb_b_path)

        st.session_state.emb_a = emb_a
        st.session_state.emb_b = emb_b

        st.session_state.rgb_a = percentile_stretch(rgb_a_raw)
        st.session_state.rgb_b = percentile_stretch(rgb_b_raw)

        st.session_state.profile = profile
        st.session_state.transform = transform

        st.session_state.data_fetched = True

        st.success("✅ Loaded cached data.")
        st.rerun()

    # =========================================================
    # OTHERWISE DOWNLOAD FROM GEE
    # =========================================================

    if not st.session_state.gee_ready:
        st.error("Initialize GEE first (sidebar).")

    else:

        try:
            import ee

            # Build AOI
            if aoi_mode == "Bounding Box":
                aoi = aoi_from_bbox(west, south, east, north)
            else:
                geojson_obj = json.loads(aoi_geojson)
                aoi = aoi_from_geojson(geojson_obj)

            # =========================
            # AEF YEAR A
            # =========================
            with st.spinner("Fetching AEF embeddings for Year A…"):

                emb_a_ee = get_aef_image(year_a, aoi)

                emb_a, profile, transform = download_image_as_array(
                    emb_a_ee,
                    aoi,
                    scale=30,
                    tile_size=0.03
                )

                save_tif(emb_a, profile, aef_a_path)

                st.session_state.emb_a = emb_a
                st.session_state.profile = profile
                st.session_state.transform = transform

            # =========================
            # AEF YEAR B
            # =========================
            with st.spinner("Fetching AEF embeddings for Year B…"):

                emb_b_ee = get_aef_image(year_b, aoi)

                emb_b, _, _ = download_image_as_array(
                    emb_b_ee,
                    aoi,
                    scale=30,
                    tile_size=0.03
                )

                save_tif(emb_b, profile, aef_b_path)

                st.session_state.emb_b = emb_b

            # =========================
            # RGB YEAR A
            # =========================
            with st.spinner("Fetching Sentinel-2 RGB for Year A…"):

                rgb_a_ee = get_s2_rgb(year_a, aoi)

                rgb_a, _, _ = download_image_as_array(
                    rgb_a_ee,
                    aoi,
                    scale=20,
                    tile_size=0.08
                )

                save_tif(rgb_a, profile, rgb_a_path)

                st.session_state.rgb_a = percentile_stretch(rgb_a)

            # =========================
            # RGB YEAR B
            # =========================
            with st.spinner("Fetching Sentinel-2 RGB for Year B…"):

                rgb_b_ee = get_s2_rgb(year_b, aoi)

                rgb_b, _, _ = download_image_as_array(
                    rgb_b_ee,
                    aoi,
                    scale=20,
                    tile_size=0.08
                )

                save_tif(rgb_b, profile, rgb_b_path)

                st.session_state.rgb_b = percentile_stretch(rgb_b)

            st.session_state.data_fetched = True

            st.success(f"✅ Data fetched. Shape: {emb_a.shape}")

            st.rerun()

        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.exception(e)

    if not st.session_state.gee_ready:
        st.error("Initialize GEE first (sidebar).")
    else:
        try:
            import ee
            # Build AOI
            if aoi_mode == "Bounding Box":
                aoi = aoi_from_bbox(west, south, east, north)
            else:
                geojson_obj = json.loads(aoi_geojson)
                aoi = aoi_from_geojson(geojson_obj)

            with st.spinner("Fetching AEF embeddings for Year A…"):
                emb_a_ee = get_aef_image(year_a, aoi)
                emb_a, profile, transform = download_image_as_array(emb_a_ee, aoi, scale=30)
                save_tif(emb_a, profile, os.path.join(TMP_DIR, "aef_a.tif"))
                st.session_state.emb_a     = emb_a
                st.session_state.profile   = profile
                st.session_state.transform = transform

            with st.spinner("Fetching AEF embeddings for Year B…"):
                emb_b_ee = get_aef_image(year_b, aoi)
                emb_b, _, _ = download_image_as_array(emb_b_ee, aoi, scale=30)
                save_tif(emb_b, profile, os.path.join(TMP_DIR, "aef_b.tif"))
                st.session_state.emb_b = emb_b

            with st.spinner("Fetching Sentinel-2 RGB for Year A…"):
                rgb_a_ee = get_s2_rgb(year_a, aoi)
                rgb_a, _, _ = download_image_as_array(rgb_a_ee, aoi, scale=10)
                save_tif(rgb_a, profile, os.path.join(TMP_DIR, "rgb_a.tif"))
                st.session_state.rgb_a = percentile_stretch(rgb_a)

            with st.spinner("Fetching Sentinel-2 RGB for Year B…"):
                rgb_b_ee = get_s2_rgb(year_b, aoi)
                rgb_b, _, _ = download_image_as_array(rgb_b_ee, aoi, scale=10)
                save_tif(rgb_b, profile, os.path.join(TMP_DIR, "rgb_b.tif"))
                st.session_state.rgb_b = percentile_stretch(rgb_b)

            st.session_state.data_fetched = True
            st.success(f"✅ Data fetched. Shape: {emb_a.shape}")
            st.rerun()

        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.exception(e)

# Load from disk if session was reset but files exist
if not st.session_state.data_fetched:
    aef_a_path = os.path.join(TMP_DIR, "aef_a.tif")
    if os.path.exists(aef_a_path):
        emb_a, profile, transform = load_tif(aef_a_path)
        emb_b, _, _               = load_tif(os.path.join(TMP_DIR, "aef_b.tif"))
        rgb_a_raw, _, _           = load_tif(os.path.join(TMP_DIR, "rgb_a.tif"))
        rgb_b_raw, _, _           = load_tif(os.path.join(TMP_DIR, "rgb_b.tif"))
        st.session_state.update({
            "emb_a": emb_a, "emb_b": emb_b,
            "rgb_a": percentile_stretch(rgb_a_raw),
            "rgb_b": percentile_stretch(rgb_b_raw),
            "profile": profile, "transform": transform,
            "data_fetched": True,
        })


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — RGB VISUAL CHECK
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.data_fetched:
    st.markdown('<div class="step-header"><b>Step 2 — Visual RGB Check</b></div>',
                unsafe_allow_html=True)

    with st.expander("Side-by-side RGB comparison", expanded=True):
        rgb_bytes = rgb_comparison_plot(
            st.session_state.rgb_a, st.session_state.rgb_b, year_a, year_b
        )
        st.image(rgb_bytes, use_column_width=True)
        H_disp, W_disp = st.session_state.rgb_a.shape[:2]
        st.caption(f"Raster size: {W_disp} × {H_disp} pixels  |  "
                   f"CRS: {st.session_state.profile.get('crs', 'N/A')}")


    # ════════════════════════════════════════════════════════════════════
    # STEP 3+4 — DIFF + PCA
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header"><b>Step 3 + 4 — Embedding Diff & PCA</b></div>',
                unsafe_allow_html=True)

    pca_col1, pca_col2 = st.columns([2, 1])
    with pca_col1:
        st.markdown(
            f"Computes `emb_{year_b} − emb_{year_a}` (64-d per pixel), then fits "
            f"PCA-{n_pca} on a random subsample and projects the full raster."
        )
    with pca_col2:
        run_pca_btn = st.button("⚙️ Run Diff + PCA", use_container_width=True)

    if run_pca_btn:
        with st.spinner("Computing embedding difference…"):
            diff_flat, H, W = compute_diff(
                st.session_state.emb_a, st.session_state.emb_b
            )
            st.session_state.update({"diff_flat": diff_flat, "H": H, "W": W})

        with st.spinner("Fitting PCA (explore 20 components)…"):
            pca_explore = fit_pca_explore(diff_flat, n_components=20)
            st.session_state.pca_explore = pca_explore

        with st.spinner(f"Fitting PCA-{n_pca} and projecting full raster…"):
            pca_model = fit_pca(diff_flat, n_components=n_pca)
            pca8_map  = project_pca(pca_model, diff_flat, H, W)
            pc_rgb    = make_pca_rgb(pca8_map)
            st.session_state.update({
                "pca_model": pca_model,
                "pca8_map":  pca8_map,
                "pc_rgb":    pc_rgb,
                "pca_done":  True,
            })
        st.success(f"✅ PCA-{n_pca} done. Map shape: {pca8_map.shape}")
        st.rerun()

    if st.session_state.pca_done:
        tab1, tab2 = st.tabs(["📊 Variance Plot", "🖼 PCA False Color"])
        with tab1:
            var_bytes = variance_plot(st.session_state.pca_explore, n_pca)
            st.image(var_bytes, use_column_width=True)
            cumvar = np.cumsum(
                st.session_state.pca_explore.explained_variance_ratio_
            ) * 100
            st.info(f"PCA-{n_pca} retains **{cumvar[n_pca-1]:.1f}%** of diff variance.")

        with tab2:
            pca_bytes = pca_falsecolor_plot(st.session_state.pc_rgb)
            st.image(pca_bytes, use_column_width=True)
            st.caption("Bright / colorful clusters = candidate change pixels. "
                       "Dark uniform areas = stable no-change.")


    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — INTERACTIVE LABELING
    # ════════════════════════════════════════════════════════════════════
    if st.session_state.pca_done:
        st.markdown('<div class="step-header"><b>Step 5 — Interactive Labeling</b></div>',
                    unsafe_allow_html=True)

        st.markdown(
            "Click on the PCA false-color image to label pixels. "
            "Use the sidebar to switch between **CHANGE** and **NO-CHANGE**. "
            "Aim for ≥ 50 of each class."
        )

        try:
            from streamlit_drawable_canvas import st_canvas

            pc_rgb_arr = st.session_state.pc_rgb
            H_r, W_r   = pc_rgb_arr.shape[:2]

            # Scale display size
            MAX_W = 900
            scale = min(MAX_W / W_r, 1.0)
            disp_w = int(W_r * scale)
            disp_h = int(H_r * scale)

            pil_base = Image.fromarray(
                (pc_rgb_arr * 255).clip(0, 255).astype(np.uint8)
            ).resize((disp_w, disp_h), Image.BILINEAR)

            # Draw existing labels on background
            pil_with_labels = overlay_labels_on_image(
                np.array(pil_base).astype(np.float32) / 255.0,
                [(int(r * scale), int(c * scale), l)
                 for r, c, l in st.session_state.labels],
                dot_radius=4
            )

            stroke_color = "#ff3c3c" if current_label == 1 else "#3c78ff"

            label_col1, label_col2 = st.columns([3, 1])
            with label_col1:
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=10,
                    stroke_color=stroke_color,
                    background_image=pil_with_labels,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="point",
                    point_display_radius=5,
                    key=f"canvas_{len(st.session_state.labels)}",
                )

            with label_col2:
                st.markdown("**Label controls**")
                if st.button("↩️ Undo last", use_container_width=True):
                    if st.session_state.labels:
                        st.session_state.labels.pop()
                        st.rerun()

                if st.button("🗑️ Clear all", use_container_width=True):
                    st.session_state.labels = []
                    st.rerun()

                n_ch_cur = sum(l for _, _, l in st.session_state.labels)
                n_nc_cur = len(st.session_state.labels) - n_ch_cur
                st.metric("🔴 Change", n_ch_cur)
                st.metric("🔵 No-change", n_nc_cur)
                st.metric("Total", len(st.session_state.labels))

                if len(st.session_state.labels) >= 4:
                    st.success("✅ Ready to train")
                else:
                    st.warning(f"Need ≥ 4 labels ({4 - len(st.session_state.labels)} more)")

            # Process new canvas clicks
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                new_count = len(objects)
                existing = len(st.session_state.labels)

                if new_count > existing:
                    for obj in objects[existing:]:
                        img_x = obj.get("left", 0)
                        img_y = obj.get("top", 0)
                        row = int(img_y / scale)
                        col = int(img_x / scale)
                        row = max(0, min(row, H_r - 1))
                        col = max(0, min(col, W_r - 1))
                        # Deduplicate
                        key = (row, col)
                        existing_keys = {(r, c) for r, c, _ in st.session_state.labels}
                        if key not in existing_keys:
                            st.session_state.labels.append((row, col, current_label))

        except ImportError:
            st.warning(
                "streamlit-drawable-canvas not installed. "
                "Using coordinate input fallback."
            )
            with st.expander("Manual coordinate input"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    m_row = st.number_input("Row", 0,
                                             st.session_state.H - 1
                                             if st.session_state.H else 9999, 100)
                with col2:
                    m_col = st.number_input("Col", 0,
                                             st.session_state.W - 1
                                             if st.session_state.W else 9999, 100)
                with col3:
                    m_label = st.selectbox("Label", [1, 0],
                                            format_func=lambda x: "Change" if x else "No-Change")
                if st.button("Add point"):
                    key = (m_row, m_col)
                    existing_keys = {(r, c) for r, c, _ in st.session_state.labels}
                    if key not in existing_keys:
                        st.session_state.labels.append((m_row, m_col, m_label))
                        st.rerun()

        # Export labels
        if st.session_state.labels:
            st.markdown("**Export labels**")
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button("📥 CSV", labels_to_csv(st.session_state.labels),
                                   "labels.csv", "text/csv",
                                   use_container_width=True)
            with dl2:
                st.download_button("📥 JSON", labels_to_json(st.session_state.labels),
                                   "labels.json", "application/json",
                                   use_container_width=True)
            with dl3:
                st.download_button("📥 Python list",
                                   labels_to_pylist(st.session_state.labels),
                                   "labels.py", "text/plain",
                                   use_container_width=True)


        # ════════════════════════════════════════════════════════════════
        # STEP 6+7 — BUILD TRAINING DATA + TRAIN
        # ════════════════════════════════════════════════════════════════
        if len(st.session_state.labels) >= 4:
            st.markdown(
                '<div class="step-header"><b>Step 6 + 7 — Build Training Data & Train Classifier</b></div>',
                unsafe_allow_html=True
            )

            train_col1, train_col2 = st.columns([2, 1])
            with train_col1:
                st.markdown(
                    f"Extracts PCA-{n_pca} features at labeled pixels, "
                    f"normalizes with StandardScaler, trains **{classifier_choice}** "
                    "with 5-fold stratified CV."
                )
            with train_col2:
                train_btn = st.button("🤖 Train Classifier", use_container_width=True)

            if train_btn:
                X, y, valid_labels = build_training_data(
                    st.session_state.pca8_map, st.session_state.labels
                )
                n_min = min(np.bincount(y)) if len(np.unique(y)) > 1 else 0
                if n_min < 2:
                    st.error("Need at least 2 samples per class. Add more labels.")
                else:
                    with st.spinner("Training…"):
                        clf, scaler, metrics = train_classifier(X, y, classifier_choice)
                    st.session_state.update({
                        "clf": clf, "scaler": scaler,
                        "metrics": metrics, "model_trained": True,
                        "_train_X": X, "_train_y": y,
                    })
                    st.success("✅ Training complete!")
                    st.rerun()

            if st.session_state.model_trained:
                metrics = st.session_state.metrics
                m_cols = st.columns(4)
                for col, key in zip(m_cols, ["accuracy", "f1", "precision", "recall"]):
                    mean, std = metrics[key]
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div style="font-size:1.5rem;font-weight:700;color:#4fc3f7">'
                        f'{mean:.3f}</div>'
                        f'<div style="font-size:0.7rem;color:#888">±{std:.3f}</div>'
                        f'<div style="font-size:0.8rem;color:#aaa">{key.upper()}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                t1, t2 = st.tabs(["📊 Metrics", "🔢 Confusion Matrix"])
                with t1:
                    st.plotly_chart(metrics_bar_plot(metrics),
                                    use_container_width=True)
                with t2:
                    if "_train_X" in st.session_state:
                        cm_bytes = confusion_matrix_plot(
                            st.session_state.clf,
                            st.session_state.scaler,
                            st.session_state._train_X,
                            st.session_state._train_y,
                        )
                        st.image(cm_bytes)


            # ════════════════════════════════════════════════════════════
            # STEP 8 — FULL INFERENCE
            # ════════════════════════════════════════════════════════════
            if st.session_state.model_trained:
                st.markdown(
                    '<div class="step-header"><b>Step 8 — Full Raster Inference</b></div>',
                    unsafe_allow_html=True
                )

                inf_col1, inf_col2 = st.columns([2, 1])
                with inf_col1:
                    st.markdown(
                        "Runs the trained classifier on every pixel in the AOI. "
                        "Processed in chunks to stay within memory limits."
                    )
                with inf_col2:
                    infer_btn = st.button("🔍 Run Inference", use_container_width=True)

                if infer_btn:
                    prog_bar = st.progress(0, text="Running inference…")

                    def update_progress(frac):
                        prog_bar.progress(frac, text=f"Inference… {frac*100:.0f}%")

                    probs_map = run_inference(
                        st.session_state.clf,
                        st.session_state.scaler,
                        st.session_state.pca8_map,
                        progress_cb=update_progress,
                    )
                    clean_mask = apply_threshold_and_filter(
                        probs_map, threshold, min_obj_px
                    )
                    st.session_state.update({
                        "probs_map":    probs_map,
                        "clean_mask":   clean_mask,
                        "inference_done": True,
                    })
                    prog_bar.progress(1.0, text="Done!")
                    st.success("✅ Inference complete!")
                    st.rerun()

                # Dynamic threshold preview
                if st.session_state.inference_done:
                    st.markdown("**Dynamic threshold preview**")
                    preview_thresh = st.slider(
                        "Preview threshold", 0.1, 0.9,
                        float(threshold), 0.05,
                        key="preview_thresh"
                    )
                    preview_mask = apply_threshold_and_filter(
                        st.session_state.probs_map,
                        preview_thresh, min_obj_px
                    )
                    ha_preview = preview_mask.sum() * 100 / 10_000
                    st.info(f"At threshold **{preview_thresh}**: "
                            f"**{ha_preview:.1f} ha** detected as changed "
                            f"({preview_mask.sum():,} pixels)")


                # ════════════════════════════════════════════════════════
                # STEP 9+10 — POST-PROCESSING + VISUALIZATION
                # ════════════════════════════════════════════════════════
                if st.session_state.inference_done:
                    st.markdown(
                        '<div class="step-header">'
                        '<b>Step 9 + 10 — Results & Visualization</b>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    probs_map  = st.session_state.probs_map
                    clean_mask = apply_threshold_and_filter(
                        probs_map, threshold, min_obj_px
                    )
                    st.session_state.clean_mask = clean_mask

                    total_ha      = clean_mask.sum() * 100 / 10_000
                    pct_changed   = clean_mask.mean() * 100
                    n_objects_raw = 0
                    from scipy import ndimage
                    _, n_objects_raw = ndimage.label(clean_mask)

                    stat_cols = st.columns(4)
                    stat_data = [
                        ("Changed area", f"{total_ha:.1f} ha"),
                        ("% AOI changed", f"{pct_changed:.2f}%"),
                        ("Change objects", str(n_objects_raw)),
                        ("Threshold", str(threshold)),
                    ]
                    for col, (label, val) in zip(stat_cols, stat_data):
                        col.markdown(
                            f'<div class="metric-card">'
                            f'<div style="font-size:1.4rem;font-weight:700;color:#81c784">'
                            f'{val}</div>'
                            f'<div style="font-size:0.8rem;color:#aaa">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("")  # spacer

                    res_tab1, res_tab2, res_tab3 = st.tabs([
                        "🗺 Probability Map", "📊 Histogram", "🖼 Full Summary"
                    ])

                    with res_tab1:
                        st.plotly_chart(probability_heatmap_plotly(probs_map),
                                        use_container_width=True)

                    with res_tab2:
                        st.plotly_chart(probability_histogram_plotly(probs_map),
                                        use_container_width=True)

                    with res_tab3:
                        summary_bytes = result_summary_plot(
                            st.session_state.rgb_b, probs_map,
                            clean_mask, year_b
                        )
                        st.image(summary_bytes, use_column_width=True)


                    # ════════════════════════════════════════════════════
                    # STEP 11 — EXPORTS
                    # ════════════════════════════════════════════════════
                    st.markdown(
                        '<div class="step-header"><b>Step 11 — Export Results</b></div>',
                        unsafe_allow_html=True
                    )

                    profile = st.session_state.profile

                    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

                    with exp_col1:
                        prob_bytes = export_geotiff_bytes(probs_map, profile, "float32")
                        st.download_button(
                            "📥 change_probability.tif",
                            data=prob_bytes,
                            file_name="change_probability.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )

                    with exp_col2:
                        mask_bytes = export_geotiff_bytes(
                            clean_mask.astype(np.uint8), profile, "uint8"
                        )
                        st.download_button(
                            "📥 change_mask.tif",
                            data=mask_bytes,
                            file_name="change_mask.tif",
                            mime="image/tiff",
                            use_container_width=True,
                        )

                    with exp_col3:
                        st.download_button(
                            "📥 labels.csv",
                            data=labels_to_csv(st.session_state.labels),
                            file_name="labels.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    with exp_col4:
                        st.download_button(
                            "📥 labels.json",
                            data=labels_to_json(st.session_state.labels),
                            file_name="labels.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                    # PCA pngs
                    exp2_col1, exp2_col2 = st.columns(2)
                    with exp2_col1:
                        st.download_button(
                            "📥 pca_variance.png",
                            data=variance_plot(st.session_state.pca_explore, n_pca),
                            file_name="pca_variance.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    with exp2_col2:
                        st.download_button(
                            "📥 pca_falsecolor.png",
                            data=pca_falsecolor_plot(st.session_state.pc_rgb),
                            file_name="pca_falsecolor.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                    with st.expander("📋 Change Statistics Table"):
                        from scipy import ndimage
                        labeled_arr, n_obj = ndimage.label(clean_mask)
                        obj_sizes = ndimage.sum(
                            clean_mask, labeled_arr, range(1, n_obj + 1)
                        )
                        import pandas as pd
                        if len(obj_sizes) > 0:
                            df = pd.DataFrame({
                                "Object ID":       range(1, n_obj + 1),
                                "Size (pixels)":   [int(s) for s in obj_sizes],
                                "Area (ha)":       [round(s * 100 / 10_000, 3)
                                                    for s in obj_sizes],
                            }).sort_values("Size (pixels)", ascending=False)
                            st.dataframe(df, use_container_width=True, height=300)
                            csv_stats = df.to_csv(index=False)
                            st.download_button(
                                "📥 change_statistics.csv",
                                csv_stats, "change_statistics.csv", "text/csv"
                            )

# ── footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.78rem">'
    'via Google AEF Embeddings · Fredericton, NB '
    'Inspired by <a href="https://geospatialml.com/posts/change-detection/" '
    'style="color:#4fc3f7">Robinson & Corley 2026</a>'
    '</div>',
    unsafe_allow_html=True,
)
