"""Fixed configuration for the formal data3 physical-message experiment."""

from pathlib import Path

from configs.ablation_config import (
    AVAILABLE_MASK_RATIOS,
    MASK_RATIO,
)
from configs.uv_sampling_config import (
    ACTIVE_AAG_DIRNAME,
    ACTIVE_FEATURE_SCHEMA_FILENAME,
    ACTIVE_PARTITION_DIRNAME,
    EDGE_FACE_UV_SHAPE,
    EDGE_GRID_SHAPE,
    FACE_GRID_SHAPE,
    UV_GRID_SIZE,
)


PROJECT_ROOT = Path("/root/zmy/AAGNet-main")
HARDWARE_PROFILE = "4090"
FORMAL_DATASET_ROOT = Path("/root/zmy/data3")
FORMAL_STEP_ROOT = FORMAL_DATASET_ROOT / "steps"
FORMAL_LABEL_ROOT = FORMAL_DATASET_ROOT / "labels"
FORMAL_AAG_ROOT = FORMAL_DATASET_ROOT / ACTIVE_AAG_DIRNAME
FORMAL_PARTITION_ROOT = FORMAL_DATASET_ROOT / ACTIVE_PARTITION_DIRNAME
_OUTPUT_DIRNAME_BY_UV_GRID_SIZE = {
    3: f"ablation_uv3_mask60_pre200_ft400_{HARDWARE_PROFILE}",
    5: f"data3_physical_message_{HARDWARE_PROFILE}",
    10: f"ablation_uv10_mask60_pre200_ft400_{HARDWARE_PROFILE}",
}
FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / _OUTPUT_DIRNAME_BY_UV_GRID_SIZE[UV_GRID_SIZE]
)
FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "feature_lists"
    / ACTIVE_FEATURE_SCHEMA_FILENAME
)


def active_sampling_log_lines():
    return (
        f"UV_GRID_SIZE = {UV_GRID_SIZE}",
        f"ACTIVE_AAG_ROOT = {FORMAL_AAG_ROOT}",
        f"ACTIVE_PARTITION_ROOT = {FORMAL_PARTITION_ROOT}",
        f"FEATURE_SCHEMA = {FEATURE_LIST_PATH}",
        f"OUTPUT_ROOT = {FORMAL_OUTPUT_ROOT}",
        f"FACE_GRID_SHAPE = {list(FACE_GRID_SHAPE)}",
        f"EDGE_GRID_SHAPE = {list(EDGE_GRID_SHAPE)}",
        f"EDGE_FACE_UV_SHAPE = {list(EDGE_FACE_UV_SHAPE)}",
    )


RAW_DATASET_NUM_SAMPLES = 10000

RAW_DATASET_COMBO_RANGE = (
    4,
    12,
)

RAW_DATASET_NUM_WORKERS = 2

RAW_DATASET_FEATURE_WEIGHTS = (
    0.3,
    0.6,
    1.0,
    1.0,
    1.0,
    0.8,
    0.6,
    1.0,
    1.0,
    1.0,
    1.0,
    0.3,
)
# ============================================================
# Training stages
# ============================================================

# Run self-supervised pretraining.
RUN_PRETRAIN = True

# Run supervised finetuning.
RUN_FINETUNE = True

AVAILABLE_FINETUNE_PERCENTS = (
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
)

FINETUNE_PERCENTS = (
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
)


def validate_training_stage_config():
    if not (RUN_PRETRAIN or RUN_FINETUNE):
        raise ValueError(
            "Training pipeline requires RUN_PRETRAIN or RUN_FINETUNE"
        )
    if (
        RUN_PRETRAIN
        and MASK_RATIO not in AVAILABLE_MASK_RATIOS
    ):
        raise ValueError(
            "MASK_RATIO must be one of "
            f"{AVAILABLE_MASK_RATIOS}, got {MASK_RATIO}"
        )
    if RUN_FINETUNE and not FINETUNE_PERCENTS:
        raise ValueError(
            "RUN_FINETUNE requires at least one FINETUNE_PERCENTS value"
        )
    invalid_percents = tuple(
        percent
        for percent in FINETUNE_PERCENTS
        if percent not in AVAILABLE_FINETUNE_PERCENTS
    )
    if invalid_percents:
        raise ValueError(
            "FINETUNE_PERCENTS contains unsupported values: "
            f"{invalid_percents}"
        )
    if len(set(FINETUNE_PERCENTS)) != len(FINETUNE_PERCENTS):
        raise ValueError("FINETUNE_PERCENTS must not contain duplicates")


FORMAL_SPLIT_PROTOCOL = "train_eval_test_70_15_15_seed42"
FORMAL_INPUT_PROTOCOL = "physical_message_v1"
FORMAL_GRAPH_STORAGE_PROTOCOL = "compact_streamed_json_array_v1"
FORMAL_GRAPH_RUNTIME_PROTOCOL = "shared_on_demand_dgl_store_v1"
FORMAL_DEVICE = "cuda:0"
FORMAL_SEED = 42

EXTRACTION_WORKERS = 2

DETERMINISM_AUDIT_TARGET_COUNT = 32
DETERMINISM_AUDIT_IN_PROCESS_REPEATS = 2
DETERMINISM_AUDIT_FRESH_PROCESS_REPEATS = 2
DETERMINISM_AUDIT_RTOL = 1e-6
DETERMINISM_AUDIT_ATOL = 1e-7
PRETRAIN_EPOCHS = 400
FINETUNE_EPOCHS = 400

HARDWARE_CONFIGS = {
    "4090": {
        "pretrain_batch_size": 256,
        "finetune_batch_size": 256,
        "pretrain_num_workers": 4,
        "finetune_num_workers": 4,
    },
    "5090": {
        "pretrain_batch_size": 256,
        "finetune_batch_size": 256,
        "pretrain_num_workers": 4,
        "finetune_num_workers": 4,
    },
}

if HARDWARE_PROFILE not in HARDWARE_CONFIGS:
    raise ValueError(
        "HARDWARE_PROFILE must be one of "
        f"{tuple(HARDWARE_CONFIGS.keys())}, "
        f"got {HARDWARE_PROFILE}"
    )

_ACTIVE_HARDWARE_CONFIG = HARDWARE_CONFIGS[
    HARDWARE_PROFILE
]

PRETRAIN_BATCH_SIZE = (
    _ACTIVE_HARDWARE_CONFIG[
        "pretrain_batch_size"
    ]
)

FINETUNE_BATCH_SIZE = (
    _ACTIVE_HARDWARE_CONFIG[
        "finetune_batch_size"
    ]
)

PRETRAIN_NUM_WORKERS = (
    _ACTIVE_HARDWARE_CONFIG[
        "pretrain_num_workers"
    ]
)

FINETUNE_NUM_WORKERS = (
    _ACTIVE_HARDWARE_CONFIG[
        "finetune_num_workers"
    ]
)

PRETRAIN_EVAL_EVERY = 5
PRETRAIN_SAVE_LAST_EVERY = 10

FINETUNE_EVAL_EVERY = 5
FINETUNE_SAVE_LAST_EVERY = 10
FREEZE_BACKBONE_EPOCHS = 10

AMP_ENABLED = True


PRETRAIN_MAIN_CONFIG = {
    "output_dir": str(FORMAL_OUTPUT_ROOT / "pretrain"),
    "run_name": "formal_pretrain",
    "graph_storage_protocol": FORMAL_GRAPH_STORAGE_PROTOCOL,
    "graph_runtime_protocol": FORMAL_GRAPH_RUNTIME_PROTOCOL,
    "epochs": PRETRAIN_EPOCHS,
    "batch_size": PRETRAIN_BATCH_SIZE,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "device": FORMAL_DEVICE,
    "num_workers": PRETRAIN_NUM_WORKERS,
    "pin_memory": True,
    "seed": FORMAL_SEED,
    "pretrain_train_split": "train",
    "pretrain_eval_split": "eval",
    "eval_every": PRETRAIN_EVAL_EVERY,
    "save_last_every": PRETRAIN_SAVE_LAST_EVERY,
    "normalize": True,
    "pretrain_method": (
        "physical_message_structured_mask_embedding_reconstruction_v1"
    ),
    "face_attr_dim": 16,
    "face_grid_channels": 11,
    "edge_attr_dim": 16,
    "edge_grid_channels": 16,
    "mask_ratio": MASK_RATIO,
    "mask_policy": "centered_graph_region_node_edge_mask",
    "mask_scope": "centered_region",
    "negative_edge_ratio": 1.0,
    "min_mask_nodes": 1,
    "min_mask_edges": 1,
    "node_loss_weight": 1.0,
    "edge_loss_weight": 1.0,
    "topology_loss_weight": 0.5,
    "decoder_drop": 0.3,
    "feat_smooth_l1_beta": 0.5,
    "learnable_loss_weighting": False,
    "eval_mask_seed": 424242,
    "amp": AMP_ENABLED,
}


FINETUNE_MAIN_CONFIG = {
    "output_dir": str(FORMAL_OUTPUT_ROOT / "finetune"),
    "pretrained_backbone": str(
        FORMAL_OUTPUT_ROOT / "pretrain" / "checkpoints" / "best_backbone.pth"
    ),
    "pretrained_strict": True,
    "quality_stat_path": str(FORMAL_AAG_ROOT / "quality_stat.json"),
    "input_protocol": FORMAL_INPUT_PROTOCOL,
    "graph_storage_protocol": FORMAL_GRAPH_STORAGE_PROTOCOL,
    "graph_runtime_protocol": FORMAL_GRAPH_RUNTIME_PROTOCOL,
    "backbone_architecture": "AAGPhysicalMessageBackbone",
    "backbone_checkpoint_format": "aag_physical_message_backbone_v1",
    "finetune_checkpoint_format": "aag_physical_message_finetune_v1",
    "split_protocol": FORMAL_SPLIT_PROTOCOL,
    "best_metric_name": "mean_eval_semantic_iou_and_instance_pair_iou",
    "device": FORMAL_DEVICE,
    "seed": FORMAL_SEED,
    "sample_seed": FORMAL_SEED,
    "num_workers": FINETUNE_NUM_WORKERS,
    "epochs": FINETUNE_EPOCHS,
    "batch_size": FINETUNE_BATCH_SIZE,
    "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 0.01,
    "head_hidden_dim": 256,
    "task_adapter_hidden_dim": 256,
    "task_adapter_depth": 2,
    "task_adapter_drop": 0.3,
    "quality_hidden_dim": 128,
    "quality_drop": 0.3,
    "process_head_depth": 3,
    "loss_weights": {
        "semantic": 1.0,
        "instance": 1.0,
        "process": 1.0,
    },
    "amp": AMP_ENABLED,
    "eval_every": FINETUNE_EVAL_EVERY,
    "save_last_every": FINETUNE_SAVE_LAST_EVERY,
    "train_split": "train",
    "eval_split": "eval",
    "test_split": "test",
}
