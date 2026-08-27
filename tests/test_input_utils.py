"""Unit tests for mpc_lulc.input_utils.

Covers validate_aoi and crs_to_epsg across valid inputs, boundary conditions,
and expected error paths.
"""

import pytest
from mpc_lulc.input_utils import validate_aoi, crs_to_epsg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon(coords=None):
    """Return a minimal valid GeoJSON Polygon dict."""
    if coords is None:
        coords = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
    return {"type": "Polygon", "coordinates": coords}


def _feature(geometry):
    """Wrap *geometry* in a GeoJSON Feature dict."""
    return {"type": "Feature", "geometry": geometry}


# ---------------------------------------------------------------------------
# validate_aoi — valid inputs
# ---------------------------------------------------------------------------

class TestValidateAoiValid:
    def test_bare_polygon_passes(self):
        validate_aoi(_polygon())

    def test_feature_wrapping_polygon_passes(self):
        validate_aoi(_feature(_polygon()))

    def test_polygon_with_explicit_coords_passes(self):
        coords = [[[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [10.0, 21.0], [10.0, 20.0]]]
        validate_aoi(_polygon(coords))

    def test_returns_none(self):
        result = validate_aoi(_polygon())
        assert result is None


# ---------------------------------------------------------------------------
# validate_aoi — null / missing geometry
# ---------------------------------------------------------------------------

class TestValidateAoiNullGeometry:
    def test_none_input_raises_value_error(self):
        with pytest.raises(ValueError, match="None"):
            validate_aoi(None)

    def test_non_dict_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_aoi("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")

    def test_feature_with_null_geometry_raises(self):
        with pytest.raises(ValueError, match="null geometry"):
            validate_aoi(_feature(None))


# ---------------------------------------------------------------------------
# validate_aoi — wrong type
# ---------------------------------------------------------------------------

class TestValidateAoiWrongType:
    def test_point_raises_value_error(self):
        with pytest.raises(ValueError, match="Point"):
            validate_aoi({"type": "Point", "coordinates": [0, 0]})

    def test_linestring_raises_value_error(self):
        with pytest.raises(ValueError, match="LineString"):
            validate_aoi({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})

    def test_multipolygon_raises_value_error(self):
        with pytest.raises(ValueError, match="MultiPolygon"):
            validate_aoi({"type": "MultiPolygon", "coordinates": []})

    def test_missing_type_field_raises(self):
        with pytest.raises(ValueError):
            validate_aoi({"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]})


# ---------------------------------------------------------------------------
# validate_aoi — missing coordinates
# ---------------------------------------------------------------------------

class TestValidateAoiMissingCoordinates:
    def test_missing_coordinates_key_raises(self):
        with pytest.raises(ValueError, match="coordinates"):
            validate_aoi({"type": "Polygon"})

    def test_null_coordinates_raises(self):
        with pytest.raises(ValueError, match="null"):
            validate_aoi({"type": "Polygon", "coordinates": None})


# ---------------------------------------------------------------------------
# crs_to_epsg — valid inputs
# ---------------------------------------------------------------------------

class TestCrsToEpsgValid:
    def test_epsg_4326_uppercase(self):
        assert crs_to_epsg("EPSG:4326") == 4326

    def test_epsg_4326_lowercase(self):
        assert crs_to_epsg("epsg:4326") == 4326

    def test_epsg_32637(self):
        assert crs_to_epsg("EPSG:32637") == 32637

    def test_epsg_mixed_case(self):
        assert crs_to_epsg("Epsg:3857") == 3857

    def test_epsg_with_leading_trailing_spaces(self):
        assert crs_to_epsg("  EPSG:4326  ") == 4326

    def test_returns_int(self):
        result = crs_to_epsg("EPSG:4326")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# crs_to_epsg — invalid / unrecognized formats
# ---------------------------------------------------------------------------

class TestCrsToEpsgInvalid:
    def test_wgs84_string_raises(self):
        with pytest.raises(ValueError, match="EPSG:XXXX"):
            crs_to_epsg("WGS84")

    def test_esri_prefix_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("ESRI:102008")

    def test_proj_string_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("+proj=longlat +datum=WGS84")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("   ")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg(None)

    def test_integer_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg(4326)

    def test_epsg_without_code_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("EPSG:")

    def test_epsg_non_numeric_code_raises(self):
        with pytest.raises(ValueError):
            crs_to_epsg("EPSG:abc")
