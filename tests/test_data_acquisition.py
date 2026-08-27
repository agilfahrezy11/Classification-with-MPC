"""Unit tests for mpc_lulc.data_acquisition.

Covers STAC_Client and Image_Acquisitor across:
- AOI validation delegation (req 1.7)
- Zero-result search error (req 1.3)
- Signing failure handling (req 1.6)
- Empty item list rejection (req 2.7)
- Missing-band warning and exclusion (req 2.5)
- Median composite reduces time dimension (req 2.4)

All external dependencies (pystac_client, planetary_computer, stackstac,
rioxarray) are mocked — no real network calls are made.
"""

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _polygon(lon0=34.0, lat0=0.0, lon1=34.1, lat1=0.1):
    """Return a minimal valid GeoJSON Polygon in EPSG:4326."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
        ],
    }


def _make_item(item_id: str, assets: dict = None, cloud_cover: float = 5.0):
    """Build a mock pystac.Item with configurable assets and cloud cover."""
    item = MagicMock()
    item.id = item_id
    item.properties = {"eo:cloud_cover": cloud_cover}
    item.assets = assets if assets is not None else {"B02": MagicMock(), "B03": MagicMock()}
    return item


def _make_real_da(time_len: int = 2, bands=None, ny: int = 4, nx: int = 4) -> xr.DataArray:
    """Return a real xr.DataArray with (time, band, y, x) dims."""
    if bands is None:
        bands = ["B02", "B03"]
    rng = np.random.default_rng(0)
    data = rng.random((time_len, len(bands), ny, nx)).astype(np.float32)
    times = pd.date_range("2023-06-01", periods=time_len, freq="10D")
    return xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={
            "time": times,
            "band": bands,
            "y": np.linspace(0.09, 0.01, ny),
            "x": np.linspace(0.01, 0.09, nx),
        },
    )


def _make_mock_da(real_da: xr.DataArray) -> MagicMock:
    """Return a MagicMock that delegates xarray operations to *real_da*.

    The .rio accessor is a plain MagicMock whose .write_crs and .clip
    return *this mock* so the chain stays consistent throughout build_stack.
    The .median() / .mean() calls delegate to the real DataArray so the
    actual time-reduction logic runs correctly.
    """
    mock = MagicMock()
    # Delegate real xarray operations
    mock.dims = real_da.dims
    mock.shape = real_da.shape
    mock.median.side_effect = lambda **kw: real_da.median(**kw)
    mock.mean.side_effect = lambda **kw: real_da.mean(**kw)

    # rio accessor chain: write_crs → mock (same), clip → mock (same)
    mock_crs = MagicMock()
    mock_crs.to_epsg.return_value = 4326
    mock.rio.write_crs.return_value = mock
    mock.rio.crs = mock_crs
    mock.rio.clip.return_value = mock
    return mock


# ===========================================================================
# STAC_Client — AOI validation (req 1.7)
# ===========================================================================

class TestSTACClientAOIValidation:
    """validate_aoi is called before any API request; invalid AOI raises ValueError."""

    def setup_method(self):
        with patch("mpc_lulc.data_acquisition.pystac_client.Client.open"):
            from mpc_lulc.data_acquisition import STAC_Client
            self.client = STAC_Client()
        # Ensure the catalog mock is fresh for each test
        self.client._catalog = MagicMock()

    def test_null_aoi_raises_value_error(self):
        with pytest.raises(ValueError, match="None"):
            self.client.search(None, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_non_polygon_geometry_raises_value_error(self):
        point_geom = {"type": "Point", "coordinates": [34.0, 0.0]}
        with pytest.raises(ValueError, match="Polygon"):
            self.client.search(point_geom, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_linestring_raises_value_error(self):
        line = {"type": "LineString", "coordinates": [[34.0, 0.0], [34.1, 0.1]]}
        with pytest.raises(ValueError, match="LineString"):
            self.client.search(line, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_feature_with_null_geometry_raises_value_error(self):
        feature = {"type": "Feature", "geometry": None}
        with pytest.raises(ValueError, match="null geometry"):
            self.client.search(feature, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_no_api_call_made_on_invalid_aoi(self):
        """The STAC catalog search must not be invoked when AOI is invalid."""
        with pytest.raises(ValueError):
            self.client.search(None, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")
        self.client._catalog.search.assert_not_called()


# ===========================================================================
# STAC_Client — zero-result search (req 1.3)
# ===========================================================================

class TestSTACClientZeroResults:
    """search raises ValueError when STAC returns zero items after filtering."""

    def setup_method(self):
        with patch("mpc_lulc.data_acquisition.pystac_client.Client.open"):
            from mpc_lulc.data_acquisition import STAC_Client
            self.client = STAC_Client()
        # Configure the internal catalog to return an empty item list
        mock_search_result = MagicMock()
        mock_search_result.items.return_value = []
        self.client._catalog = MagicMock()
        self.client._catalog.search.return_value = mock_search_result

    def test_raises_value_error_on_zero_results(self):
        with pytest.raises(ValueError):
            self.client.search(_polygon(), ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_error_message_contains_collection_name(self):
        with pytest.raises(ValueError, match="sentinel-2-l2a"):
            self.client.search(_polygon(), ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_error_message_contains_start_date(self):
        with pytest.raises(ValueError, match="2023-06-01"):
            self.client.search(_polygon(), ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_error_message_contains_end_date(self):
        with pytest.raises(ValueError, match="2023-08-31"):
            self.client.search(_polygon(), ("2023-06-01", "2023-08-31"), "sentinel-2-l2a")

    def test_raises_after_cloud_filter_empties_results(self):
        """Items that fail cloud filter should still trigger the zero-result error."""
        item = _make_item("item-1", cloud_cover=80.0)
        mock_search_result = MagicMock()
        mock_search_result.items.return_value = [item]
        self.client._catalog.search.return_value = mock_search_result

        with pytest.raises(ValueError, match="sentinel-2-l2a"):
            self.client.search(
                _polygon(),
                ("2023-06-01", "2023-08-31"),
                "sentinel-2-l2a",
                max_cloud_cover=10,
            )


# ===========================================================================
# STAC_Client — signing failure (req 1.6)
# ===========================================================================

class TestSTACClientSigningFailure:
    """sign_items raises on failure and does not return partial results."""

    def setup_method(self):
        with patch("mpc_lulc.data_acquisition.pystac_client.Client.open"):
            from mpc_lulc.data_acquisition import STAC_Client
            self.client = STAC_Client()

    def test_signing_failure_raises_value_error(self):
        items = [_make_item("item-good"), _make_item("item-bad")]

        def _sign_side_effect(item):
            if item.id == "item-bad":
                raise RuntimeError("network error")

        with patch(
            "mpc_lulc.data_acquisition.planetary_computer.sign_inplace",
            side_effect=_sign_side_effect,
        ):
            with pytest.raises(ValueError):
                self.client.sign_items(items)

    def test_error_message_identifies_failing_item(self):
        items = [_make_item("my-failing-item")]

        with patch(
            "mpc_lulc.data_acquisition.planetary_computer.sign_inplace",
            side_effect=RuntimeError("timeout"),
        ):
            with pytest.raises(ValueError, match="my-failing-item"):
                self.client.sign_items(items)

    def test_no_partial_results_returned_on_failure(self):
        """When signing fails mid-list, a ValueError is raised, not a partial list."""
        items = [_make_item("good-1"), _make_item("good-2"), _make_item("fail-3")]

        def _sign_side_effect(item):
            if item.id == "fail-3":
                raise RuntimeError("signing error")

        with patch(
            "mpc_lulc.data_acquisition.planetary_computer.sign_inplace",
            side_effect=_sign_side_effect,
        ):
            with pytest.raises(ValueError):
                self.client.sign_items(items)
            # If we reach here the exception was raised — partial list was not returned

    def test_success_returns_all_items(self):
        items = [_make_item("item-a"), _make_item("item-b")]
        with patch("mpc_lulc.data_acquisition.planetary_computer.sign_inplace"):
            result = self.client.sign_items(items)
        assert len(result) == 2


# ===========================================================================
# Image_Acquisitor — empty item list (req 2.7)
# ===========================================================================

class TestImageAcquisitorEmptyList:
    """Image_Acquisitor raises ValueError immediately when items list is empty."""

    def test_empty_list_raises_value_error(self):
        from mpc_lulc.data_acquisition import Image_Acquisitor
        with pytest.raises(ValueError):
            Image_Acquisitor([])

    def test_error_raised_before_any_stack_assembly(self):
        """stackstac.stack must never be called when items list is empty."""
        from mpc_lulc.data_acquisition import Image_Acquisitor
        with patch("mpc_lulc.data_acquisition.stackstac.stack") as mock_stack:
            with pytest.raises(ValueError):
                Image_Acquisitor([])
            mock_stack.assert_not_called()

    def test_non_empty_list_does_not_raise(self):
        from mpc_lulc.data_acquisition import Image_Acquisitor
        item = _make_item("item-1")
        acq = Image_Acquisitor([item])  # should not raise
        assert len(acq.items) == 1


# ===========================================================================
# Image_Acquisitor — missing bands (req 2.5)
# ===========================================================================

class TestImageAcquisitorMissingBands:
    """Missing bands log warnings; all-missing items are dropped with a warning."""

    def _make_acquisitor(self, items):
        from mpc_lulc.data_acquisition import Image_Acquisitor
        return Image_Acquisitor(items)

    def test_missing_band_logs_warning(self, caplog):
        """An item missing one of the requested bands should produce a WARNING log."""
        # item-1 has B02 only; B03 is missing
        item1 = _make_item("item-1", assets={"B02": MagicMock()})
        acq = self._make_acquisitor([item1])

        real_da = _make_real_da(time_len=1, bands=["B02"])
        mock_da = _make_mock_da(real_da)

        with patch("mpc_lulc.data_acquisition.stackstac.stack", return_value=mock_da):
            with caplog.at_level(logging.WARNING, logger="mpc_lulc.data_acquisition"):
                acq.build_stack(["B02", "B03"], _polygon(), crs="EPSG:4326", resolution=10)

        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "B03" in w or "missing" in w.lower() for w in warning_texts
        ), f"Expected warning about missing band B03, got: {warning_texts}"

    def test_all_missing_item_dropped_with_warning(self, caplog):
        """An item with none of the requested bands must be dropped with a WARNING."""
        item1 = _make_item("item-has-nothing", assets={"SCL": MagicMock()})
        item2 = _make_item("item-has-bands", assets={"B02": MagicMock(), "B04": MagicMock()})
        acq = self._make_acquisitor([item1, item2])

        real_da = _make_real_da(time_len=1, bands=["B02", "B04"])
        mock_da = _make_mock_da(real_da)

        with patch("mpc_lulc.data_acquisition.stackstac.stack", return_value=mock_da) as mock_ss:
            with caplog.at_level(logging.WARNING, logger="mpc_lulc.data_acquisition"):
                acq.build_stack(["B02", "B04"], _polygon(), crs="EPSG:4326", resolution=10)

            # Only item2 should have been passed to stackstac
            call_args = mock_ss.call_args
            items_passed = call_args[0][0]
            assert item1 not in items_passed, "Dropped item should not reach stackstac"
            assert item2 in items_passed, "Valid item must reach stackstac"

        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "item-has-nothing" in w or "dropped" in w.lower() or "none of" in w.lower()
            for w in warning_texts
        ), f"Expected drop warning for item-has-nothing, got: {warning_texts}"

    def test_all_items_missing_all_bands_raises(self):
        """When every item is dropped, build_stack must raise ValueError."""
        item = _make_item("item-no-bands", assets={"SCL": MagicMock()})
        acq = self._make_acquisitor([item])

        with pytest.raises(ValueError):
            acq.build_stack(["B02", "B04"], _polygon(), crs="EPSG:4326", resolution=10)


# ===========================================================================
# Image_Acquisitor — median composite (req 2.4)
# ===========================================================================

class TestImageAcquisitorMedianComposite:
    """build_stack with composite_mode='median' removes the time dimension."""

    def _run_build_stack(self, real_da, items, bands, composite_mode):
        """Helper: runs build_stack with stackstac mocked to return a smart mock_da."""
        from mpc_lulc.data_acquisition import Image_Acquisitor

        acq = Image_Acquisitor(items)
        mock_da = _make_mock_da(real_da)

        with patch("mpc_lulc.data_acquisition.stackstac.stack", return_value=mock_da):
            result = acq.build_stack(
                bands, _polygon(), crs="EPSG:4326", resolution=10,
                composite_mode=composite_mode,
            )
        return result

    def test_median_composite_removes_time_dimension(self):
        """After median composite the result must NOT have a 'time' dimension."""
        real_da = _make_real_da(time_len=3, bands=["B02", "B03"])
        items = [
            _make_item(f"item-{i}", assets={"B02": MagicMock(), "B03": MagicMock()})
            for i in range(3)
        ]

        result = self._run_build_stack(real_da, items, ["B02", "B03"], "median")

        assert "time" not in result.dims, (
            f"Expected no 'time' dim after median composite, got dims: {result.dims}"
        )

    def test_median_composite_no_time_dim_single_item(self):
        """Median over a single-time stack should also drop the time dimension."""
        real_da = _make_real_da(time_len=1, bands=["B02", "B03"])
        items = [_make_item("item-1", assets={"B02": MagicMock(), "B03": MagicMock()})]

        result = self._run_build_stack(real_da, items, ["B02", "B03"], "median")

        assert "time" not in result.dims, (
            f"Expected no 'time' dim after single-item median, got dims: {result.dims}"
        )

    def test_no_composite_preserves_time_dimension(self):
        """Without composite_mode the time dimension should be present in the result."""
        real_da = _make_real_da(time_len=2, bands=["B02", "B03"])
        items = [
            _make_item(f"item-{i}", assets={"B02": MagicMock(), "B03": MagicMock()})
            for i in range(2)
        ]
        from mpc_lulc.data_acquisition import Image_Acquisitor

        acq = Image_Acquisitor(items)
        mock_da = _make_mock_da(real_da)

        # When composite_mode=None the mock_da itself is returned (no .median call)
        with patch("mpc_lulc.data_acquisition.stackstac.stack", return_value=mock_da):
            result = acq.build_stack(
                ["B02", "B03"], _polygon(), crs="EPSG:4326", resolution=10,
                composite_mode=None,
            )

        # mock_da.dims is real_da.dims which includes 'time'
        assert "time" in result.dims, (
            f"Expected 'time' dim preserved when composite_mode=None, got dims: {result.dims}"
        )
