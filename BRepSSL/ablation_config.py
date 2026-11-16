"""Ablation switches for the formal model.

The default values reproduce the complete model.
Change only one ablation factor at a time for controlled experiments.
"""


# True: use DHD-GP in the complete model.
# False: bypass only DHD-GP for the "w/o DHD-GP" ablation.
USE_DHD_GP = True


# True: use the complete-model gcdMPNN.
# False: replace gcdMPNN with the ordinary MPNN baseline.
USE_GCD_MPNN = True


# True: use the improved UV geometry encoding of the complete model.
# False: use the ordinary UV encoding baseline for the "w/o Improved UV" ablation.
# The ordinary UV baseline must still retain UV inputs; it must only remove
# our later UV-network improvements such as bilateral source/destination
# boundary sampling and fusion, together with the associated improved
# edge-to-node context enhancement.
USE_IMPROVED_UV = True


# True: use the GATv2 refinement located after DHD-GP.
# False: bypass only this GATv2 refinement for the "w/o GAT" ablation.
USE_GAT = True


# UV sampling resolution.
# 5: complete-model setting.
# 3: 3x3 UV-resolution ablation.
# 10: 10x10 UV-resolution ablation.
UV_GRID_SIZE = 5


# ============================================================
# Structured mask-ratio ablation
# ============================================================
#
# MASK_RATIO controls the masking proportion INSIDE EACH
# individual CAD/B-Rep sample during self-supervised pretraining.
#
# It is NOT the proportion of CAD samples selected for masking.
# Every CAD sample entering pretraining is masked.
#
# For one CAD/B-Rep, the selected ratio is used as the target
# structured masking proportion for its face/node and edge
# entities, together with the associated masked structural
# information used by the reconstruction tasks.
#
# This is analogous to the mask ratio of points/patches inside
# one point-cloud sample in masked point-cloud pretraining.
#
# Formal ablation candidates:
#   0.5 / 0.6 / 0.7 / 0.8 / 0.9
#
# Select exactly ONE value for each experiment.
# Do not automatically iterate over all candidates.
AVAILABLE_MASK_RATIOS = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)

MASK_RATIO = 0.6
