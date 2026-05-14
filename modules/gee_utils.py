import ee
import os
import numpy as np
import rasterio
import streamlit as st
import tempfile
import urllib.request
from rasterio.merge import merge
from rasterio.io import MemoryFile

BANDS_AEF = [f"A{str(i).zfill(2)}" for i in range(64)]


# =========================================================
# AUTH
# =========================================================

def authenticate_gee(project_id: str) -> tuple[bool, str]:
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        return True, "Authenticated and initialized."
    except Exception as e:
        return False, str(e)


def initialize_gee(project_id: str) -> tuple[bool, str]:
    try:
        ee.Initialize(project=project_id)
        return True, f"Initialized with project: {project_id}"
    except Exception as e:
        return False, str(e)


# =========================================================
# DATA SOURCES
# =========================================================

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


# =========================================================
# AOI HELPERS
# =========================================================

def aoi_from_bbox(west: float, south: float,
                  east: float, north: float) -> ee.Geometry:
    return ee.Geometry.Rectangle([west, south, east, north])


def aoi_from_geojson(geojson: dict) -> ee.Geometry:
    return ee.Geometry(geojson)


# =========================================================
# TILE SPLITTING
# =========================================================

def split_bbox_into_tiles(
    west,
    south,
    east,
    north,
    tile_size=0.05
):
    tiles = []

    lon = west

    while lon < east:

        lat = south

        while lat < north:

            tile_west = lon
            tile_south = lat

            tile_east = min(lon + tile_size, east)
            tile_north = min(lat + tile_size, north)

            tile_geom = ee.Geometry.Rectangle([
                tile_west,
                tile_south,
                tile_east,
                tile_north
            ])

            tiles.append(tile_geom)

            lat += tile_size

        lon += tile_size

    return tiles


# =========================================================
# SINGLE TILE DOWNLOAD
# =========================================================

def _download_single_tile(
    image,
    region,
    scale=30,
    crs="EPSG:32620"
):

    url = image.getDownloadURL({
        "scale": scale,
        "crs": crs,
        "region": region,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    with urllib.request.urlopen(url) as response:
        data = response.read()

    memfile = MemoryFile(data)

    src = memfile.open()

    return src


# =========================================================
# MAIN DOWNLOAD FUNCTION
# =========================================================


def download_image_as_array(
    image: ee.Image,
    aoi: ee.Geometry,
    scale: int = 30,
    crs: str = "EPSG:32620",
    tile_size: float = 0.03,
):
    import urllib.request
    import rasterio
    from rasterio.merge import merge
    import tempfile
    import os
    import time

    print("USING SAFE TILED DOWNLOAD")

    bounds = aoi.bounds().coordinates().getInfo()[0]

    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]

    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)

    x_tiles = np.arange(west, east, tile_size)
    y_tiles = np.arange(south, north, tile_size)

    tif_paths = []

    tile_id = 0

    for x in x_tiles:
        for y in y_tiles:

            tile_id += 1

            tile_geom = ee.Geometry.Rectangle([
                x,
                y,
                min(x + tile_size, east),
                min(y + tile_size, north)
            ])

            try:

                print(f"Downloading tile {tile_id}")

                url = image.clip(tile_geom).getDownloadURL({
                    "scale": scale,
                    "crs": crs,
                    "region": tile_geom,
                    "format": "GEO_TIFF",
                    "filePerBand": False,
                })

                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read()

                # skip tiny/bad downloads
                if len(data) < 5000:
                    print("Skipped empty tile")
                    continue

                tmp = tempfile.NamedTemporaryFile(
                    suffix=".tif",
                    delete=False
                )

                tmp.write(data)
                tmp.close()

                # validate raster
                try:
                    with rasterio.open(tmp.name) as test_src:
                        _ = test_src.read(1)

                    tif_paths.append(tmp.name)

                except Exception as e:
                    print("Bad tile skipped:", e)
                    os.unlink(tmp.name)

            except Exception as e:
                print("Tile failed:", e)
                continue

            time.sleep(0.2)

    if len(tif_paths) == 0:
        raise Exception("No valid tiles downloaded.")

    datasets = []

    for p in tif_paths:
        try:
            datasets.append(rasterio.open(p))
        except:
            pass

    print(f"Merging {len(datasets)} valid tiles...")

    mosaic, transform = merge(datasets)

    profile = datasets[0].profile.copy()

    profile.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "count": mosaic.shape[0],
        "dtype": "float32",
    })

    for ds in datasets:
        ds.close()

    for p in tif_paths:
        try:
            os.unlink(p)
        except:
            pass

    return mosaic.astype(np.float32), profile, transform



# =========================================================
# EXPORTS
# =========================================================

def save_tif(arr: np.ndarray, profile: dict, path: str):

    p = profile.copy()

    p.update({
        "count": arr.shape[0],
        "dtype": "float32",
        "compress": "lzw"
    })

    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32))


def load_tif(path: str):

    with rasterio.open(path) as src:

        arr = src.read().astype(np.float32)

        profile = src.profile

        transform = src.transform

    return arr, profile, transform


# =========================================================
# OPTIONAL DRIVE EXPORT
# =========================================================

def export_image_to_drive(
    image: ee.Image,
    name: str,
    aoi: ee.Geometry,
    folder: str = "change_exports",
    scale: int = 30,
    crs: str = "EPSG:32620"
):

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        region=aoi,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    task.start()

    return task