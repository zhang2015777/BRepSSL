"""Single source of truth for active AAG UV/curve sampling resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from configs.ablation_config import UV_GRID_SIZE


FACE_GRID_CHANNELS = 11
EDGE_GRID_CHANNELS = 16
EDGE_FACE_SIDE_COUNT = 2
UV_COORDINATE_DIM = 2
SUPPORTED_UV_GRID_SIZES = (3, 5, 10)

if UV_GRID_SIZE not in SUPPORTED_UV_GRID_SIZES:
    raise ValueError(
        "UV_GRID_SIZE must be one of "
        f"{SUPPORTED_UV_GRID_SIZES}, got {UV_GRID_SIZE}"
    )

FACE_GRID_SHAPE = (
    FACE_GRID_CHANNELS,
    UV_GRID_SIZE,
    UV_GRID_SIZE,
)
EDGE_GRID_SHAPE = (EDGE_GRID_CHANNELS, UV_GRID_SIZE)
EDGE_FACE_UV_SHAPE = (
    EDGE_FACE_SIDE_COUNT,
    UV_GRID_SIZE,
    UV_COORDINATE_DIM,
)
EDGE_FACE_UV_VALID_SHAPE = (EDGE_FACE_SIDE_COUNT, UV_GRID_SIZE)

SAMPLING_PATH_VARIANTS = {
    3: {
        "aag_dirname": "aag_uv3",
        "partition_dirname": "MFInstseg_partition_uv3",
        "feature_schema_filename": "all_uv3.json",
    },
    5: {
        "aag_dirname": "aag_ours16",
        "partition_dirname": "MF_partition",
        "feature_schema_filename": "all.json",
    },
    10: {
        "aag_dirname": "aag_uv10",
        "partition_dirname": "MFInstseg_partition_uv10",
        "feature_schema_filename": "all_uv10.json",
    },
}

_ACTIVE_PATH_VARIANT = SAMPLING_PATH_VARIANTS[UV_GRID_SIZE]
ACTIVE_AAG_DIRNAME = _ACTIVE_PATH_VARIANT["aag_dirname"]
ACTIVE_PARTITION_DIRNAME = _ACTIVE_PATH_VARIANT["partition_dirname"]
ACTIVE_FEATURE_SCHEMA_FILENAME = _ACTIVE_PATH_VARIANT[
    "feature_schema_filename"
]


@dataclass(frozen=True)
class ActiveSamplingPaths:
    dataset_root: Path
    aag_root: Path
    partition_root: Path
    feature_schema: Path


def resolve_active_sampling_paths(dataset_root, project_root):
    dataset_root = Path(dataset_root)
    project_root = Path(project_root)
    return ActiveSamplingPaths(
        dataset_root=dataset_root,
        aag_root=dataset_root / ACTIVE_AAG_DIRNAME,
        partition_root=dataset_root / ACTIVE_PARTITION_DIRNAME,
        feature_schema=(
            project_root
            / "feature_lists"
            / ACTIVE_FEATURE_SCHEMA_FILENAME
        ),
    )


def sampling_shape_payload():
    return {
        "UV_GRID_SIZE": UV_GRID_SIZE,
        "FACE_GRID_SHAPE": list(FACE_GRID_SHAPE),
        "EDGE_GRID_SHAPE": list(EDGE_GRID_SHAPE),
        "EDGE_FACE_UV_SHAPE": list(EDGE_FACE_UV_SHAPE),
        "EDGE_FACE_UV_VALID_SHAPE": list(EDGE_FACE_UV_VALID_SHAPE),
    }


def load_and_validate_active_feature_schema(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Active feature schema not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "num_srf_u": UV_GRID_SIZE,
        "num_srf_v": UV_GRID_SIZE,
        "num_crv_u": UV_GRID_SIZE,
    }
    if schema.get("UV-grid") != expected:
        raise ValueError(
            f"{path}: UV-grid must be {expected}, "
            f"got {schema.get('UV-grid')!r}"
        )
    return schema
