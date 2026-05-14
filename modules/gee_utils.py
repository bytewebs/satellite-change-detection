import ee
import os
import json
import numpy as np
import rasterio
import streamlit as st
import tempfile

BANDS_AEF = [f"A{str(i).zfill(2)}" for i in range(64)]


def _try_service_account(project_id: str) -> tuple[bool, str]:
    """
    Reads GEE service account JSON from Streamlit secrets and initializes.
    Secrets format (in Streamlit Cloud → App settings → Secrets):

        [gee]
        service_account_json = '''{ "type": "service_account", ... }'''

    Returns (True, msg) on success, (False, "no_secret") if secret missing,
    or (False, error_msg) on any other failure.
    """
    try:
        sa_json_str = st.secrets["gee"]["service_account_json"]
    except (KeyError, FileNotFoundError):
        return False, "no_secret"

    try:
        info = json.loads(sa_json_str)
        credentials = ee.ServiceAccountCredentials(
            email=info["client_email"],
            key_data=sa_json_str,
        )
        ee.Initialize(credentials=credentials, project=project_id)
        return True, f"Authenticated via service account ({info['client_email']})"
    except Exception as e:
        return False, str(e)


def _try_local_adc(project_id: str) -> tuple[bool, str]:
    """
    Standard local application-default-credentials flow.
    Works when `gcloud auth application-default login` has been run.
    """
    try:
        ee.Initialize(project=project_id)
        return True, f"Initialized with local credentials (project: {project_id})"
    except Exception as e:
        return False, str(e)


def authenticate_gee(project_id: str) -> tuple[bool, str]:
    """
    Auth priority:
    1. Streamlit secrets service account  -> works on Streamlit Cloud
    2. Local ADC / existing credentials   -> works locally after gcloud login
    3. Interactive ee.Authenticate()      -> browser OAuth, local only
    """
    # 1. Service account (deployed env)
    ok, msg = _try_service_account(project_id)
    if ok:
        return True, msg
    if msg != "no_secret":
        return False, f"Service account auth failed: {msg}"

    # 2. Local ADC (already initialized / gcloud ADC)
    ok, msg = _try_local_adc(project_id)
    if ok:
        return True, msg

    # 3. Interactive browser OAuth (local only, needs gcloud)
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        return True, "Authenticated via browser OAuth."
    except Exception as e:
        return False, (
            "Authentication failed.\n\n"
            "**Locally:** run `gcloud auth application-default login` then retry.\n\n"
            "**On Streamlit Cloud:** add your service account JSON to app secrets:\n"
            "App settings -> Secrets -> paste:\n"
            "```\n[gee]\nservice_account_json = '''{ ...your JSON... }'''\n```"
        )


def initialize_gee(project_id: str) -> tuple[bool, str]:
    """Initialize only (no browser OAuth). Tries service account then local ADC."""
    ok, msg = _try_service_account(project_id)
    if ok:
        return True, msg
    if msg != "no_secret":
        return False, f"Service account init failed: {msg}"

    return _try_local_adc(project_id)


def get_aef_image(year: int, aoi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .filterBounds(aoi)
        .mosaic()
        .select(BANDS_AEF)
        .clip(aoi)
    )


def get_s2_rgb(year: int, aoi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
        .median()
        .select(["B4", "B3", "B2"])
        .clip(aoi)
    )


def download_image_as_array(image: ee.Image, aoi: ee.Geometry,
                             scale: int = 10, crs: str = "EPSG:32620") -> tuple:
    import urllib.request

    url = image.getDownloadURL({
        "scale": scale,
        "crs": crs,
        "region": aoi,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    with urllib.request.urlopen(url) as response:
        data = response.read()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path) as src:
            arr       = src.read().astype(np.float32)
            profile   = src.profile
            transform = src.transform
    finally:
        os.unlink(tmp_path)

    return arr, profile, transform


def save_tif(arr: np.ndarray, profile: dict, path: str):
    p = profile.copy()
    p.update({"count": arr.shape[0], "dtype": "float32", "compress": "lzw"})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32))


def load_tif(path: str) -> tuple:
    with rasterio.open(path) as src:
        arr       = src.read().astype(np.float32)
        profile   = src.profile
        transform = src.transform
    return arr, profile, transform


def aoi_from_bbox(west: float, south: float,
                  east: float, north: float) -> ee.Geometry:
    return ee.Geometry.Rectangle([west, south, east, north])


def aoi_from_geojson(geojson: dict) -> ee.Geometry:
    return ee.Geometry(geojson)