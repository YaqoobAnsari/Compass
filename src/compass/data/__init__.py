"""COMPASS data pipeline (RadioMapSeer) — correct polarity & TX conventions."""

from .conventions import (
    BUILDING_VALUE,
    DEFAULT_GAIN_VARIANT,
    GAIN_VARIANTS,
    MAP_SIZE,
    PATHLOSS_MAX_DBM,
    PATHLOSS_MIN_DBM,
    PATHLOSS_RANGE_DB,
    RESOLUTION_M,
    STREET_VALUE,
    dbm_to_png,
    png_to_dbm,
    png_to_signed_unit,
    signed_unit_to_dbm,
    tx_pixel_from_json,
    tx_position_normalised,
)
from .floor_plan import (
    building_mask,
    corridor_mask,
    distance_to_walls,
    street_fraction,
    street_mask,
)
from .radiomapseer import (
    RadioMapSeerDataset,
    RadioMapSeerPaths,
    list_map_ids,
    split_map_ids,
)

__all__ = [
    "RadioMapSeerDataset",
    "RadioMapSeerPaths",
    "list_map_ids",
    "split_map_ids",
    "street_mask",
    "building_mask",
    "distance_to_walls",
    "corridor_mask",
    "street_fraction",
    "png_to_dbm",
    "dbm_to_png",
    "png_to_signed_unit",
    "signed_unit_to_dbm",
    "tx_pixel_from_json",
    "tx_position_normalised",
    "MAP_SIZE",
    "RESOLUTION_M",
    "STREET_VALUE",
    "BUILDING_VALUE",
    "PATHLOSS_MIN_DBM",
    "PATHLOSS_MAX_DBM",
    "PATHLOSS_RANGE_DB",
    "GAIN_VARIANTS",
    "DEFAULT_GAIN_VARIANT",
]
