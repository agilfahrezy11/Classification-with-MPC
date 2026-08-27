# Module 1: Data Acquisition
#
# Provides two classes:
#   - STAC_Client      : queries the MPC STAC API, filters by cloud cover, and signs asset URLs.
#   - Image_Acquisitor : assembles a labeled xarray.DataArray from signed STAC items via stackstac.

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import planetary_computer
import pystac
import pystac_client
import stackstac
import xarray as xr
from shapely.geometry import shape

from mpc_lulc.input_utils import crs_to_epsg, validate_aoi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported MPC collections (informational — not an exhaustive allow-list)
# ---------------------------------------------------------------------------
_SUPPORTED_COLLECTIONS = {"sentinel-2-l2a", "landsat-c2-l2", "naip"}


def _bbox_from_aoi(aoi: dict) -> Tuple[float, float, float, float]:
    """Return the (west, south, east, north) bounding box of a GeoJSON polygon.

    Parameters
    ----------
    aoi : dict
        A GeoJSON-like polygon geometry dict (bare geometry or Feature wrapper).
        Assumed already validated by `validate_aoi`.

    Returns
    -------
    tuple of float
        ``(west, south, east, north)`` bounding box coordinates.
    """
    # Unwrap Feature wrappers
    geom_dict = aoi.get("geometry", aoi) if aoi.get("type") == "Feature" else aoi
    geom = shape(geom_dict)
    return tuple(geom.bounds)  # (minx, miny, maxx, maxy)


# ===========================================================================
# STAC_Client
# ===========================================================================

class STAC_Client:
    """Query the Microsoft Planetary Computer STAC API and sign asset URLs.

    Parameters
    ----------
    catalog_url : str, optional
        Root URL of the MPC STAC catalog.
        Defaults to ``"https://planetarycomputer.microsoft.com/api/stac/v1"``.

    Attributes
    ----------
    catalog_url : str
        The catalog URL used to open the pystac_client connection.

    Examples
    --------
    >>> client = STAC_Client()
    >>> items = client.search(aoi, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a", max_cloud_cover=20)
    >>> signed = client.sign_items(items)
    """

    def __init__(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    ) -> None:
        self.catalog_url = catalog_url
        logger.debug("Opening STAC catalog at %s", catalog_url)
        self._catalog = pystac_client.Client.open(
            catalog_url,
            modifier=planetary_computer.sign_inplace,
        )
        logger.info("STAC catalog opened: %s", catalog_url)

    # ------------------------------------------------------------------
    def search(
        self,
        aoi: dict,
        date_range: Tuple[str, str],
        collection: str,
        max_cloud_cover: Optional[float] = None,
    ) -> List[pystac.Item]:
        """Search the MPC STAC catalog for imagery matching the given criteria.

        Parameters
        ----------
        aoi : dict
            GeoJSON polygon geometry (bare geometry or Feature wrapper) in EPSG:4326.
        date_range : tuple of str
            ``(start_date, end_date)`` in ``"YYYY-MM-DD"`` format.
        collection : str
            MPC STAC collection name, e.g. ``"sentinel-2-l2a"``.
        max_cloud_cover : float, optional
            Maximum allowed ``eo:cloud_cover`` percentage (0–100).  Items
            that lack the ``eo:cloud_cover`` property are always excluded
            when this threshold is set.

        Returns
        -------
        list of pystac.Item
            All matching (and optionally cloud-filtered) STAC items, with
            asset URLs already signed via ``planetary_computer``.

        Raises
        ------
        ValueError
            If ``aoi`` is null, not a valid Polygon, or not in EPSG:4326
            (delegated to :func:`~mpc_lulc.input_utils.validate_aoi`).
        ValueError
            If the search returns zero items after all filtering.

        Examples
        --------
        >>> client = STAC_Client()
        >>> aoi = {"type": "Polygon", "coordinates": [[[34.0, 0.0], [34.1, 0.0], [34.1, 0.1], [34.0, 0.1], [34.0, 0.0]]]}
        >>> items = client.search(aoi, ("2023-06-01", "2023-08-31"), "sentinel-2-l2a", max_cloud_cover=20)
        """
        # --- 1. Validate AOI (req 1.7) -----------------------------------
        validate_aoi(aoi)

        # --- 2. Derive bbox -----------------------------------------------
        bbox = _bbox_from_aoi(aoi)
        logger.debug(
            "Searching collection=%r  date_range=%r  bbox=%r  max_cloud_cover=%r",
            collection, date_range, bbox, max_cloud_cover,
        )

        # --- 3. Execute STAC search (req 1.1, 1.4) -----------------------
        start_date, end_date = date_range
        search = self._catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
        )
        items: List[pystac.Item] = list(search.items())
        logger.info(
            "STAC search returned %d raw items for collection=%r", len(items), collection
        )

        # --- 4. Cloud-cover filter (req 1.2) -----------------------------
        if max_cloud_cover is not None:
            before = len(items)
            items = [
                item for item in items
                if isinstance(item.properties.get("eo:cloud_cover"), (int, float))
                and item.properties["eo:cloud_cover"] <= max_cloud_cover
            ]
            logger.debug(
                "Cloud-cover filter (≤%.1f %%): kept %d / %d items",
                max_cloud_cover, len(items), before,
            )

        # --- 5. Raise on empty results (req 1.3) -------------------------
        if len(items) == 0:
            raise ValueError(
                f"No STAC items found for collection={collection!r}, "
                f"date_range=({start_date!r}, {end_date!r}), "
                f"bbox={bbox!r}."
            )

        logger.info("Returning %d items after filtering.", len(items))
        return items

    # ------------------------------------------------------------------
    def sign_items(self, items: List[pystac.Item]) -> List[pystac.Item]:
        """Sign all asset URLs in a list of STAC items in-place.

        Calls ``planetary_computer.sign_inplace`` on every item.  If
        signing fails for any item, raises an error immediately without
        returning a partially signed collection (req 1.6).

        Parameters
        ----------
        items : list of pystac.Item
            Unsigned STAC items returned by :meth:`search`.

        Returns
        -------
        list of pystac.Item
            The same list of items with all asset URLs signed.

        Raises
        ------
        ValueError
            If signing fails for any item, identifying the item ID.

        Examples
        --------
        >>> signed = client.sign_items(items)
        """
        signed: List[pystac.Item] = []
        for item in items:
            try:
                planetary_computer.sign_inplace(item)
                signed.append(item)
                logger.debug("Signed item: %s", item.id)
            except Exception as exc:
                raise ValueError(
                    f"Failed to sign asset URLs for item {item.id!r}: {exc}"
                ) from exc
        logger.info("Signed %d items.", len(signed))
        return signed


# ===========================================================================
# Image_Acquisitor
# ===========================================================================

class Image_Acquisitor:
    """Assemble a labeled xarray raster stack from a list of signed STAC items.

    Parameters
    ----------
    items : list of pystac.Item
        A non-empty list of signed STAC items.  Raises :exc:`ValueError`
        immediately if the list is empty.

    Raises
    ------
    ValueError
        If ``items`` is empty.

    Examples
    --------
    >>> acquisitor = Image_Acquisitor(signed_items)
    >>> stack = acquisitor.build_stack(["B02", "B03", "B04", "B08"], aoi, crs="EPSG:32637", resolution=10)
    """

    def __init__(self, items: List[pystac.Item]) -> None:
        if not items:
            raise ValueError(
                "Image_Acquisitor requires a non-empty list of STAC items, "
                "but an empty list was provided."
            )
        self.items = list(items)
        logger.debug("Image_Acquisitor initialised with %d item(s).", len(self.items))

    # ------------------------------------------------------------------
    def build_stack(
        self,
        bands: List[str],
        aoi: dict,
        crs: str = "EPSG:4326",
        resolution: float = 10.0,
        composite_mode: Optional[str] = None,
    ) -> xr.DataArray:
        """Stack STAC items into a labeled xarray.DataArray.

        Parameters
        ----------
        bands : list of str
            Band / asset names to include in the stack (e.g. ``["B02", "B03"]``).
        aoi : dict
            GeoJSON polygon geometry in EPSG:4326 used for clipping.
        crs : str, optional
            Target CRS in ``"EPSG:XXXX"`` format.  Defaults to
            ``"EPSG:4326"``.
        resolution : float, optional
            Target spatial resolution in metres.  Must be in [1, 10 000].
            Defaults to ``10.0``.
        composite_mode : str or None, optional
            Temporal compositing method: ``"median"``, ``"mean"``, or
            ``"least-cloud"``.  When ``None`` the full time dimension is
            preserved.

        Returns
        -------
        xarray.DataArray
            Stack with dimensions ``(time, band, y, x)`` — or without a
            ``time`` dimension when ``composite_mode`` is not ``None``.

        Raises
        ------
        ValueError
            If ``resolution`` is outside [1, 10 000].

        Warnings
        --------
        A :func:`logging.warning` is emitted for each band that is missing
        from a given item.  Items where *all* requested bands are missing
        are dropped entirely before stacking.

        Examples
        --------
        >>> da = acquisitor.build_stack(["B02", "B08"], aoi, crs="EPSG:32637", resolution=10, composite_mode="median")
        """
        # --- 1. Validate resolution (req 2.6) ----------------------------
        if not (1 <= resolution <= 10_000):
            raise ValueError(
                f"resolution must be in [1, 10000] m, got {resolution!r}."
            )

        # --- 2. Filter items for missing bands (req 2.5) -----------------
        usable_items: List[pystac.Item] = []
        for item in self.items:
            available_bands = set(item.assets.keys())
            missing = [b for b in bands if b not in available_bands]
            if missing:
                logger.warning(
                    "Item %r is missing band(s) %s — excluded from stack for those bands.",
                    item.id, missing,
                )
            present = [b for b in bands if b in available_bands]
            if not present:
                logger.warning(
                    "Item %r has none of the requested bands %s — item dropped entirely.",
                    item.id, bands,
                )
                continue
            usable_items.append(item)

        if not usable_items:
            raise ValueError(
                "No usable items remain after filtering for the requested bands. "
                f"Requested: {bands!r}."
            )

        # --- 3. Determine target EPSG from CRS string --------------------
        epsg = crs_to_epsg(crs)

        logger.debug(
            "Calling stackstac.stack with %d item(s), bands=%r, epsg=%d, resolution=%s",
            len(usable_items), bands, epsg, resolution,
        )

        # --- 4. Build the raw stack (req 2.1) ----------------------------
        # stackstac assembles an xr.DataArray with dims (time, band, y, x).
        # We pass the union of requested bands; stackstac handles missing
        # assets per-item internally (fills NaN).
        stack: xr.DataArray = stackstac.stack(
            usable_items,
            assets=bands,
            epsg=epsg,
            resolution=resolution,
        )
        logger.info(
            "stackstac assembled a stack with shape %s and dims %s.",
            dict(zip(stack.dims, stack.shape)), stack.dims,
        )

        # --- 5. Reproject to target CRS (req 2.2) -------------------------
        # rioxarray uses the CRS already encoded by stackstac; reproject
        # only if the EPSG differs from the already-set CRS.
        stack = stack.rio.write_crs(f"EPSG:{epsg}", inplace=False)
        target_crs = f"EPSG:{epsg}"
        if stack.rio.crs is not None and stack.rio.crs.to_epsg() != epsg:
            logger.debug("Reprojecting stack to %s.", target_crs)
            stack = stack.rio.reproject(target_crs)

        # --- 6. Clip to AOI boundary (req 2.3) ---------------------------
        # Build a list-of-geometries suitable for rioxarray.clip.
        geom_dict = (
            aoi.get("geometry", aoi) if aoi.get("type") == "Feature" else aoi
        )
        logger.debug("Clipping stack to AOI boundary.")
        stack = stack.rio.clip([geom_dict], crs="EPSG:4326", drop=True)

        # --- 7. Temporal composite (req 2.4) -----------------------------
        if composite_mode is not None:
            stack = self._apply_composite(stack, composite_mode)

        logger.info(
            "build_stack complete — final shape: %s  dims: %s",
            dict(zip(stack.dims, stack.shape)), stack.dims,
        )
        return stack

    # ------------------------------------------------------------------
    def _apply_composite(
        self,
        stack: xr.DataArray,
        composite_mode: str,
    ) -> xr.DataArray:
        """Reduce the time dimension using the specified compositing strategy.

        Parameters
        ----------
        stack : xr.DataArray
            Input stack with a ``time`` dimension.
        composite_mode : str
            One of ``"median"``, ``"mean"``, or ``"least-cloud"``.

        Returns
        -------
        xr.DataArray
            Stack with the ``time`` dimension reduced (squeezed to length 1
            or dropped, depending on the xarray operation used).

        Raises
        ------
        ValueError
            If ``composite_mode`` is not one of the three supported values.
        """
        if composite_mode == "median":
            logger.debug("Applying median composite over time dimension.")
            return stack.median(dim="time", keep_attrs=True)

        if composite_mode == "mean":
            logger.debug("Applying mean composite over time dimension.")
            return stack.mean(dim="time", keep_attrs=True)

        if composite_mode == "least-cloud":
            return self._least_cloud_composite(stack)

        raise ValueError(
            f"Unsupported composite_mode {composite_mode!r}. "
            "Choose one of: 'median', 'mean', 'least-cloud'."
        )

    # ------------------------------------------------------------------
    def _least_cloud_composite(self, stack: xr.DataArray) -> xr.DataArray:
        """Select the single item with the lowest ``eo:cloud_cover``.

        Parameters
        ----------
        stack : xr.DataArray
            Stack with a ``time`` dimension whose items correspond 1-to-1
            with ``self.items`` in acquisition order.

        Returns
        -------
        xr.DataArray
            Single-time slice (time dimension squeezed) from the item with
            the lowest ``eo:cloud_cover`` property.  Falls back to the
            first item if no items carry the property.
        """
        best_idx = 0
        best_cover: Optional[float] = None

        for idx, item in enumerate(self.items):
            cover = item.properties.get("eo:cloud_cover")
            if cover is None:
                continue
            if best_cover is None or cover < best_cover:
                best_cover = cover
                best_idx = idx

        if best_cover is None:
            logger.warning(
                "No items carry 'eo:cloud_cover' metadata; "
                "defaulting to the first item for 'least-cloud' composite."
            )

        logger.debug(
            "least-cloud composite: selected item index %d (eo:cloud_cover=%s).",
            best_idx, best_cover,
        )

        # Select the matching time slice from the stack.
        # stackstac stores datetime64 values on the time coordinate;
        # we select by positional index.
        time_values = stack.coords["time"].values
        selected_time = time_values[best_idx]
        return stack.sel(time=selected_time, drop=False).squeeze("time", drop=True)
