"""Model definitions and factory functions."""

from hazegroupnet.models.dgrc import (
    DualCueGroupResidualCalibration,
    GroupAwareDualCueRouting,
    dark_channel_haze_proxy,
)
from hazegroupnet.models.hazegroupnet import (
    HAZEGROUP_VARIANTS,
    HazeGroupNet,
    HazeGroupVariant,
    build_hazegroupnet,
    create_model,
    hazegroupnet_l,
    hazegroupnet_s,
    hazegroupnet_t,
)
from hazegroupnet.models.trb import (
    ChannelRMSNorm,
    SeparableDown,
    SeparableUp,
    TriReceptiveBlock,
    TriReceptiveStage,
)

__all__ = [
    "ChannelRMSNorm",
    "DualCueGroupResidualCalibration",
    "GroupAwareDualCueRouting",
    "HAZEGROUP_VARIANTS",
    "HazeGroupNet",
    "HazeGroupVariant",
    "SeparableDown",
    "SeparableUp",
    "TriReceptiveBlock",
    "TriReceptiveStage",
    "build_hazegroupnet",
    "create_model",
    "dark_channel_haze_proxy",
    "hazegroupnet_l",
    "hazegroupnet_s",
    "hazegroupnet_t",
]
