"""Shared pytest fixtures for the MPC-LULC test suite.

All fixtures use in-memory synthetic data only — no real MPC API calls are
made.  Heavy external dependencies (pystac_client, planetary_computer,
stackstac) are mocked at the test level where needed.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Classification_Scheme fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def scheme():
    """Return a minimal :class:`Classification_Scheme` with 5 LULC classes.

    Mapping
    -------
    1 → "Forest"
    2 → "Cropland"
    3 → "Urban"
    4 → "Water"
    5 → "Bare Soil"
    """
    from mpc_lulc.classification_scheme import Classification_Scheme

    return Classification_Scheme(
        {
            1: "Forest",
            2: "Cropland",
            3: "Urban",
            4: "Water",
            5: "Bare Soil",
        }
    )


# ---------------------------------------------------------------------------
# Raster stack fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def raster_stack():
    """Return a small synthetic ``xr.DataArray`` mimicking a Raster_Stack.

    Shape: ``(band=4, y=10, x=10)``
    Bands: ``["B02", "B03", "B04", "B08"]``
    Dtype: ``float32``
    CRS:   EPSG:4326 (set via ``rioxarray`` extensions)
    """
    import xarray as xr
    import rioxarray  # noqa: F401 — registers the .rio accessor

    rng = np.random.default_rng(42)
    bands = ["B02", "B03", "B04", "B08"]
    n_y, n_x = 10, 10

    # Build coordinate arrays that look like geographic EPSG:4326 coords
    # (small region around 0°N, 0°E for simplicity)
    y_coords = np.linspace(0.09, 0.01, n_y)   # descending — north to south
    x_coords = np.linspace(0.01, 0.09, n_x)

    data = rng.random((len(bands), n_y, n_x)).astype(np.float32)

    da = xr.DataArray(
        data,
        dims=["band", "y", "x"],
        coords={
            "band": bands,
            "y": y_coords,
            "x": x_coords,
        },
        name="raster_stack",
    )

    # Attach CRS and spatial_ref via rioxarray
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326", inplace=True)

    return da


# ---------------------------------------------------------------------------
# Sample GeoDataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gdf(scheme):
    """Return a synthetic ``gpd.GeoDataFrame`` with ≥10 Point samples per class.

    - 50+ rows total (10 per class × 5 classes).
    - Column ``"class_label"`` contains the string label from *scheme*.
    - CRS: EPSG:4326.
    - All geometries are valid ``shapely.geometry.Point`` objects.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(0)

    n_per_class = 10
    rows = []

    for code, label in zip(scheme.codes, scheme.labels):
        for _ in range(n_per_class):
            # Random points within a 0.1° × 0.1° bounding box
            lon = rng.uniform(0.01, 0.09)
            lat = rng.uniform(0.01, 0.09)
            rows.append({"class_label": label, "geometry": Point(lon, lat)})

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf
