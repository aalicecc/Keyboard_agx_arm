"""
Centralized robot configuration.

Single source of truth for robot types, description paths,
direction multipliers, TCP offsets, and MIT-mode support.
"""

import os
from typing import Dict, List, Tuple


# ── Robot kinematic ──────────────────────────────────────
# gripper_urdf_joints 用来可视化
ROBOT_CONFIGS: Dict[str, dict] = {
    "piper":   {"num_joints": 6, "gripper_urdf_joints": 2},
    "piper_h": {"num_joints": 6, "gripper_urdf_joints": 2},
    "piper_l": {"num_joints": 6, "gripper_urdf_joints": 2},
    "piper_x": {"num_joints": 6, "gripper_urdf_joints": 2},
    "nero":    {"num_joints": 7, "gripper_urdf_joints": 0},
}


# ── Robot description file paths ────────────────────────────────────
# default_effector 用来初始化实际夹爪驱动
ROBOT_DESC_CONFIGS: Dict[str, dict] = {
    "piper":   {"desc_dir": "piper_description",   "urdf": "piper_description.urdf",
                "target_link": "link6", "default_effector": "AGX_GRIPPER"},
    "piper_h": {"desc_dir": "piper_description",   "urdf": "piper_description.urdf",
                "target_link": "link6", "default_effector": "AGX_GRIPPER"},
    "piper_l": {"desc_dir": "piper_description",   "urdf": "piper_description.urdf",
                "target_link": "link6", "default_effector": "AGX_GRIPPER"},
    "piper_x": {"desc_dir": "piper_x_description", "urdf": "piper_x_description.urdf",
                "target_link": "link6", "default_effector": "AGX_GRIPPER"},
    "nero":    {"desc_dir": "nero_description",     "urdf": "nero_description.urdf",
                "target_link": "link7", "default_effector": "None"},
}


# ── MIT mode support ────────────────────────────────────────────────

ROBOTS_WITH_MIT_MODE = {"piper", "piper_h", "piper_l", "piper_x"}


# ── Per-axis direction multipliers ──────────────────────────────────
# Index order: J1, J2, J3, J4, J5, J6 (, J7 for nero)

_NERO_DIRS  = {"joint": (1, 1, 1, 1, 1, 1, 1), "pose": (1, 1, -1, 1, 1, 1)}
_PIPER_DIRS = {"joint": (1, -1, 1, 1, 1, 1),    "pose": (1, 1, -1, 1, 1, 1)}


# ── Default TCP offsets per robot ───────────────────────────────────

DEFAULT_TCP_OFFSETS: Dict[str, List[float]] = {
    "piper":   [0.0, 0.0, 0.14],
    "piper_h": [0.0, 0.0, 0.14],
    "piper_l": [0.0, 0.0, 0.14],
    "piper_x": [0.0, 0.0, 0.14],
    "nero":    [0.0, 0.0, 0.0],
}


# ── Helper functions ────────────────────────────────────────────────

def get_direction_config(robot_type: str) -> dict:
    """Return ``{"joint": (...), "pose": (...)}`` direction multipliers."""
    return _NERO_DIRS if robot_type == "nero" else _PIPER_DIRS


def robot_description_base_path() -> str:
    """Return the absolute path to the ``robot_description/`` directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "robot_description",
    )


def get_robot_paths(robot_type: str) -> dict:
    """Get URDF path, mesh path, target link, and default effector.

    Returns:
        dict with keys: urdf_path, mesh_path, target_link, default_effector

    Raises:
        ValueError: If *robot_type* is not supported.
    """
    desc_cfg = ROBOT_DESC_CONFIGS.get(robot_type)
    if desc_cfg is None:
        raise ValueError(
            f"Unsupported robot_type '{robot_type}'. "
            f"Supported: {list(ROBOT_DESC_CONFIGS.keys())}"
        )
    base = robot_description_base_path()
    return {
        "urdf_path":        os.path.join(base, desc_cfg["desc_dir"], desc_cfg["urdf"]),
        "mesh_path":        os.path.join(base, desc_cfg["desc_dir"], "meshes"),
        "target_link":      desc_cfg["target_link"],
        "default_effector": desc_cfg.get("default_effector", "AGX_GRIPPER"),
    }

