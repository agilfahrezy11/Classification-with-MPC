# Module: Input Utilities — AOI validation and CRS helpers

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def validate_aoi(geojson: dict) -> None:
    """Validate that a GeoJSON object describes a non-null Polygon geometry.

    The function trusts the caller to supply coordinates in EPSG:4326 but
    validates structural correctness: the geometry must be present, must be
    of type ``"Polygon"``, and must include a ``coordinates`` key.

    Parameters
    ----------
    geojson : dict
        A GeoJSON-like dictionary.  Both a bare geometry object
        (``{"type": "Polygon", "coordinates": [...]}```) and a Feature
        object (``{"type": "Feature", "geometry": {...}}``) are accepted.

    Returns
    -------
    None
        Returns ``None`` on success.

    Raises
    ------
    ValueError
        If *geojson* is ``None``, if the geometry is ``None``, if the
        ``"type"`` field is not ``"Polygon"``, or if the ``"coordinates"``
        key is absent.  The error message always includes the offending
        value.

    Examples
    --------
    >>> aoi = {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]}
    >>> validate_aoi(aoi)  # no exception

    >>> validate_aoi(None)
    Traceback (most recent call last):
        ...
    ValueError: AOI must be a dict, got None
    """
    if geojson is None:
        raise ValueError(f"AOI must be a dict, got {geojson!r}")

    if not isinstance(geojson, dict):
        raise ValueError(f"AOI must be a dict, got {type(geojson).__name__!r}")

    # Support both bare geometry dicts and GeoJSON Feature wrappers
    geom_type = geojson.get("type")
    if geom_type == "Feature":
        geometry = geojson.get("geometry")
        if geometry is None:
            raise ValueError(
                f"AOI Feature has a null geometry; expected a Polygon geometry, got {geometry!r}"
            )
        geojson = geometry
        geom_type = geojson.get("type")

    if geom_type is None:
        raise ValueError(
            f"AOI dict is missing the 'type' field; expected 'Polygon', got {geom_type!r}"
        )

    if geom_type != "Polygon":
        raise ValueError(
            f"AOI geometry type must be 'Polygon', got {geom_type!r}"
        )

    if "coordinates" not in geojson:
        raise ValueError(
            "AOI Polygon geometry is missing the 'coordinates' field"
        )

    coordinates = geojson["coordinates"]
    if coordinates is None:
        raise ValueError(
            f"AOI Polygon 'coordinates' must not be null, got {coordinates!r}"
        )

    logger.debug("AOI validated successfully (type=Polygon)")


def crs_to_epsg(crs_str: str) -> int:
    """Parse a CRS string and return the integer EPSG authority code.

    Only the ``"EPSG:XXXX"`` form is currently recognised (case-insensitive).
    Other authority prefixes (``"ESRI:"``, ``"OGC:"``, etc.) are not supported
    and will raise a ``ValueError``.

    Parameters
    ----------
    crs_str : str
        A CRS identifier string, e.g. ``"EPSG:4326"`` or ``"epsg:32637"``.

    Returns
    -------
    int
        The numeric EPSG code, e.g. ``4326``.

    Raises
    ------
    ValueError
        If *crs_str* is not a string, is empty, or does not match the
        ``"EPSG:XXXX"`` pattern.  The error message includes the offending
        value.

    Examples
    --------
    >>> crs_to_epsg("EPSG:4326")
    4326
    >>> crs_to_epsg("epsg:32637")
    32637
    >>> crs_to_epsg("WGS84")
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized CRS string 'WGS84'; expected format 'EPSG:XXXX'
    """
    if not isinstance(crs_str, str) or not crs_str.strip():
        raise ValueError(
            f"crs_str must be a non-empty string, got {crs_str!r}"
        )

    pattern = re.compile(r"^epsg:(\d+)$", re.IGNORECASE)
    match = pattern.match(crs_str.strip())

    if match is None:
        raise ValueError(
            f"Unrecognized CRS string {crs_str!r}; expected format 'EPSG:XXXX'"
        )

    code = int(match.group(1))
    logger.debug("CRS string %r parsed to EPSG:%d", crs_str, code)
    return code
