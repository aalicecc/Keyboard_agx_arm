"""
Unified kinematic adapter — bridges different IK backends with a common interface.

Supported backends:
    - ``trac_ik``        — Trac-IK solver
    - ``curobo``         — cuRobo GPU solver
    - ``pinocchio``      — Pinocchio + CasADi solver
    - ``pyroki_limit``   — PyRoKi with joint-limit optimisation
    - ``pyroki_no_limit``— PyRoKi without joint-limit optimisation
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np
from scipy.spatial.transform import Rotation as R


# ── Quaternion helpers ──────────────────────────────────────────────

def wxyz_to_xyzw(wxyz) -> list:
    """Convert ``[w, x, y, z]`` quaternion to ``[x, y, z, w]`` (scipy)."""
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


def xyzw_to_wxyz(xyzw) -> list:
    """Convert ``[x, y, z, w]`` quaternion (scipy) to ``[w, x, y, z]``."""
    return [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]


# ── Abstract base ───────────────────────────────────────────────────

class KinematicAdapter(ABC):
    """Unified interface for forward / inverse kinematics solvers."""

    @property
    @abstractmethod
    def joint_limits(self) -> List[Tuple[float, float]]:
        """List of ``(lower, upper)`` for each actuated joint."""

    @abstractmethod
    def solve_fk(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, R]:
        """Forward kinematics → ``(xyz, Rotation)``."""

    @abstractmethod
    def solve_ik(
        self, xyz: np.ndarray, rotation: R, seed: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Inverse kinematics → joint angles array, or ``None``."""


# ── Trac-IK adapter ────────────────────────────────────────────────

class TracIkAdapter(KinematicAdapter):
    """Adapter for the Trac-IK solver."""

    def __init__(self, urdf_path, target_link="link6", **kw):
        from utility.kinematic_trac_ik import Kinematic
        self._solver = Kinematic(
            urdf_path=urdf_path,
            base_link_name=kw.get("base_link", "base_link"),
            target_link_name=target_link,
            timeout=kw.get("timeout", 0.005),
            epsilon=kw.get("epsilon", 1e-5),
            solver_type=kw.get("solver_type", "Speed"),
        )

    @property
    def joint_limits(self):
        return self._solver.joint_limits

    def solve_fk(self, joints):
        xyz_wxyz = self._solver.solve_fk(joints)
        xyz = np.array(xyz_wxyz[:3], dtype=float)
        rot = R.from_quat(wxyz_to_xyzw(xyz_wxyz[3:]))
        return xyz, rot

    def solve_ik(self, xyz, rotation, seed):
        wxyz = xyzw_to_wxyz(rotation.as_quat())
        result = self._solver.solve_ik(
            target_position=xyz,
            target_wxyz=np.array(wxyz),
            initial_guess=seed,
            use_previous_solution=True,
        )
        return np.array(result) if result is not None else None


# ── cuRobo adapter ──────────────────────────────────────────────────

class CuroboAdapter(KinematicAdapter):
    """Adapter for the cuRobo GPU-accelerated solver."""

    def __init__(self, urdf_path, target_link="link6", **kw):
        from utility.kinematic_curobo import Kinematic
        self._solver = Kinematic(urdf_path, target_link_name=target_link)
        self._jump_threshold = np.radians(kw.get("jump_threshold_deg", 170))

    @property
    def joint_limits(self):
        return self._solver.joint_limits

    def solve_fk(self, joints):
        jlist = joints.tolist() if hasattr(joints, "tolist") else joints
        xyz_wxyz = self._solver.solve_fk(jlist)
        xyz = np.array(xyz_wxyz[:3], dtype=float)
        rot = R.from_quat(wxyz_to_xyzw(xyz_wxyz[3:]))
        return xyz, rot

    def solve_ik(self, xyz, rotation, seed):
        wxyz = xyzw_to_wxyz(rotation.as_quat())
        result = self._solver.solve_ik(xyz.tolist(), wxyz)
        if result is None:
            return None
        joints = np.array(result)
        if np.abs(joints[0] - seed[0]) >= self._jump_threshold:
            print("Warning: IK solution has large jump, ignoring.")
            return None
        return joints


# ── Pinocchio adapter ───────────────────────────────────────────────

class PinocchioAdapter(KinematicAdapter):
    """Adapter for the Pinocchio + CasADi solver."""

    def __init__(self, urdf_path, num_joints=6, **kw):
        from utility.kinematic_pin import Arm_IK
        self._solver = Arm_IK(urdf_path)
        self._nj = num_joints

    @property
    def joint_limits(self):
        return self._solver.joint_limits

    def solve_fk(self, joints):
        padded = np.zeros(self._solver.model.nq)
        padded[: self._nj] = joints[: self._nj]
        xyz_rpy = self._solver.get_fk(padded)  # [x,y,z, r,p,y] (rad)
        xyz = np.array(xyz_rpy[:3], dtype=float)
        rot = R.from_euler("xyz", xyz_rpy[3:])
        return xyz, rot

    def solve_ik(self, xyz, rotation, seed):
        euler = rotation.as_euler("xyz")  # rad
        result = self._solver.get_ik_solution(*xyz, *euler)
        return result[: self._nj] if result is not None else None


# ── PyRoKi adapter (limit / no-limit) ──────────────────────────────

class PyrokiAdapter(KinematicAdapter):
    """Adapter for PyRoKi-based solvers (with or without joint-limit optimisation)."""

    def __init__(self, urdf_path, target_link="link6",
                 use_limits=True, num_joints=6, **kw):
        if use_limits:
            from utility.kinematic_limit import Kinematic
        else:
            from utility.kinematic_no_limit import Kinematic
        self._solver = Kinematic(urdf_path, target_link_name=target_link)
        self._nj = num_joints

    @property
    def joint_limits(self):
        return self._solver.joint_limits

    def solve_fk(self, joints):
        padded = np.zeros(self._solver.robot.joints.num_actuated_joints)
        padded[: self._nj] = joints[: self._nj]
        xyz_wxyz = self._solver.solve_fk(padded)
        xyz = np.array(xyz_wxyz[:3], dtype=float)
        rot = R.from_quat(wxyz_to_xyzw(xyz_wxyz[3:]))
        return xyz, rot

    def solve_ik(self, xyz, rotation, seed):
        wxyz = xyzw_to_wxyz(rotation.as_quat())
        result = self._solver.solve_ik(xyz, np.array(wxyz))
        return result[: self._nj] if result is not None else None


# ── Factory ─────────────────────────────────────────────────────────

_BACKEND_MAP = {
    "trac_ik":         TracIkAdapter,
    "curobo":          CuroboAdapter,
    "pinocchio":       PinocchioAdapter,
    "pyroki_limit":    lambda **kw: PyrokiAdapter(use_limits=True, **kw),
    "pyroki_no_limit": lambda **kw: PyrokiAdapter(use_limits=False, **kw),
}


def create_kinematic_adapter(backend: str, **kwargs) -> KinematicAdapter:
    """Create a kinematic adapter for the specified backend.

    Args:
        backend: One of ``'trac_ik'``, ``'curobo'``, ``'pinocchio'``,
                 ``'pyroki_limit'``, ``'pyroki_no_limit'``.
        **kwargs: Forwarded to the adapter constructor
                  (``urdf_path``, ``target_link``, ``num_joints``, …).
    """
    factory = _BACKEND_MAP.get(backend)
    if factory is None:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(_BACKEND_MAP.keys())}"
        )
    return factory(**kwargs)

