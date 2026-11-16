"""Central configuration for the MF 25-class generalization experiment."""

from __future__ import annotations

from pathlib import Path

from configs.ablation_config import (
    MASK_RATIO,
    USE_DHD_GP,
    USE_GAT,
    USE_GCD_MPNN,
    USE_IMPROVED_UV,
)
from configs.main_experiment_config import (
    AMP_ENABLED,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EVAL_EVERY,
    FINETUNE_NUM_WORKERS,
    FINETUNE_SAVE_LAST_EVERY,
    FORMAL_DEVICE,
    FREEZE_BACKBONE_EPOCHS,
    PRETRAIN_MAIN_CONFIG as BASE_PRETRAIN_CONFIG,
)
from configs.uv_sampling_config import UV_GRID_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"E:\tools\data2")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mf_generalization"

NUM_CLASSES = 25
FEATURE_DIM = 128
CLASS_NAMES = (
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
    "stock",
)

SEED = 42
SAMPLE_SEED = 42
SPLIT_PROTOCOL = "train_eval_test_70_15_15_seed42"
TRAIN_RATIO = 0.70
EVAL_RATIO = 0.15
TEST_RATIO = 0.15
FULL_MF_COUNT = 62495
FULL_MF_SPLIT_COUNTS = (43745, 9375, 9375)

AAG_DIRNAME = "aag_ours16"
PARTITION_DIRNAME = "MF_partition"
FEATURE_LIST = PROJECT_ROOT / "feature_lists" / "all.json"

PRETRAIN_EPOCHS = 400
FINETUNE_EPOCHS = 400
FINETUNE_PERCENTS = (0.1, 0.5, 1.0, 2.0, 5.0)


def validate_mf_method_config():
    expected = {
        "UV_GRID_SIZE": (UV_GRID_SIZE, 5),
        "MASK_RATIO": (MASK_RATIO, 0.6),
        "USE_GCD_MPNN": (USE_GCD_MPNN, True),
        "USE_DHD_GP": (USE_DHD_GP, True),
        "USE_GAT": (USE_GAT, True),
        "USE_IMPROVED_UV": (USE_IMPROVED_UV, True),
    }
    mismatches = [
        f"{name} must be {required!r}, got {actual!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise ValueError("Invalid MF formal method configuration: " + "; ".join(mismatches))
    if len(CLASS_NAMES) != NUM_CLASSES:
        raise ValueError("MF class-name table must contain exactly 25 entries")
    if FINETUNE_PERCENTS != (0.1, 0.5, 1.0, 2.0, 5.0):
        raise ValueError("MF finetune ratios must be exactly 0.1/0.5/1/2/5")
    if abs(TRAIN_RATIO + EVAL_RATIO + TEST_RATIO - 1.0) > 1e-12:
        raise ValueError("MF split ratios must sum to one")
    if sum(FULL_MF_SPLIT_COUNTS) != FULL_MF_COUNT:
        raise ValueError("MF paper split counts must sum to 62495")


def dataset_paths(dataset_root):
    root = Path(dataset_root)
    return {
        "dataset_root": root,
        "aag_root": root / AAG_DIRNAME,
        "partition_root": root / PARTITION_DIRNAME,
        "label_root": root / "labels",
    }


def build_mf_pretrain_config(dataset_root, output_root=DEFAULT_OUTPUT_ROOT):
    config = dict(BASE_PRETRAIN_CONFIG)
    config.update(
        {
            "dataset_root": str(Path(dataset_root)),
            "output_dir": str(Path(output_root) / "pretrain"),
            "epochs": PRETRAIN_EPOCHS,
            "seed": SEED,
            "aag_dirname": AAG_DIRNAME,
            "partition_dirname": PARTITION_DIRNAME,
            "split_protocol": SPLIT_PROTOCOL,
            "feature_list_path": str(FEATURE_LIST),
        }
    )
    return config


def build_mf_finetune_config(dataset_root, output_root=DEFAULT_OUTPUT_ROOT):
    return {
        "dataset_root": str(Path(dataset_root)),
        "output_dir": str(Path(output_root) / "finetune"),
        "pretrained_backbone": str(
            Path(output_root) / "pretrain" / "checkpoints" / "best_backbone.pth"
        ),
        "epochs": FINETUNE_EPOCHS,
        "seed": SEED,
        "sample_seed": SAMPLE_SEED,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "aag_dirname": AAG_DIRNAME,
        "partition_dirname": PARTITION_DIRNAME,
        "split_protocol": SPLIT_PROTOCOL,
        "feature_list_path": str(FEATURE_LIST),
        "loss_weights": {"semantic": 1.0, "instance": 1.0},
        "device": FORMAL_DEVICE,
        "amp": AMP_ENABLED,
        "pretrained_strict": True,
        "batch_size": FINETUNE_BATCH_SIZE,
        "num_workers": FINETUNE_NUM_WORKERS,
        "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.01,
        "head_hidden_dim": 256,
        "task_adapter_hidden_dim": 256,
        "task_adapter_depth": 2,
        "task_adapter_drop": 0.3,
        "eval_every": FINETUNE_EVAL_EVERY,
        "save_last_every": FINETUNE_SAVE_LAST_EVERY,
    }
