import numpy as np


class ArmState:
    """All mutable state for the robotic arm in one place."""

    def __init__(self, num_joints: int, joint_limits: list):
        self.num_joints   = num_joints
        self.joint_limits = joint_limits

        # ── Kinematic state ─────────────────────────────────────────
        self.joint_angles = np.zeros(num_joints)
        self.xyz_wxyz     = np.zeros(7)
        self.xyz_rpy      = np.zeros(6)

        # ── Gripper ─────────────────────────────────────────────────
        self.gripper_state     = 0.0    # 0–100 %
        self.gripper_max_width = 0.07   # metres

        # ── Step sizes ──────────────────────────────────────────────
        self.joint_step       = 0.5 * np.pi / 180.0   # rad per tick
        self.gripper_step     = 1.0                    # % per tick
        self.translation_step = 0.001                  # m per tick
        self.rotation_step    = 0.5                    # deg per tick

        # ── Modes ───────────────────────────────────────────────────
        self.up_level_mode  = "joint"   # joint | pose
        self.low_level_mode = "joint"   # joint | pose
        self.command_mode   = 0x00      # 0x00 | 0xAD

        # ── Connection ──────────────────────────────────────────────
        self.arm_connected = False
        self.arm_enabled   = False

        # ── Speed ───────────────────────────────────────────────────
        self.speed_factors       = [0.25, 0.5, 1.0]
        self.speed_factor_index  = 2     # default ×1.0
        self.movement_speeds     = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.movement_speed_index = 9    # default 100 %

        # ── Saved positions ─────────────────────────────────────────
        self.saved_positions   = []
        self.position_index    = -1
        self.playback_reversed = False

    # ── Computed properties ─────────────────────────────────────────

    @property
    def speed_factor(self) -> float:
        return self.speed_factors[self.speed_factor_index]

    @property
    def movement_speed(self) -> int:
        return self.movement_speeds[self.movement_speed_index]

    # ── Joint helpers ───────────────────────────────────────────────

    def clamp_joints(self):
        """Clip every joint angle to its ``[lower, upper]`` limit."""
        for i in range(self.num_joints):
            lower, higher = self.joint_limits[i]
            self.joint_angles[i] = np.clip(self.joint_angles[i], lower, higher)

    # ── Mode toggles ────────────────────────────────────────────────

    def toggle_up_level_mode(self):
        self.up_level_mode = "pose" if self.up_level_mode == "joint" else "joint"

    def toggle_low_level_mode(self):
        self.low_level_mode = "pose" if self.low_level_mode == "joint" else "joint"

    def toggle_command_mode(self):
        self.command_mode = 0xAD if self.command_mode == 0x00 else 0x00

    # ── Speed cycling ───────────────────────────────────────────────

    def cycle_speed_factor(self, direction: int):
        """Cycle speed factor: ``+1`` = faster, ``-1`` = slower."""
        self.speed_factor_index = (
            (self.speed_factor_index + direction) % len(self.speed_factors)
        )

    def cycle_movement_speed(self, direction: int):
        """Cycle movement speed: ``+1`` = faster, ``-1`` = slower."""
        self.movement_speed_index = (
            (self.movement_speed_index + direction) % len(self.movement_speeds)
        )

    # ── Gripper ─────────────────────────────────────────────────────

    def update_gripper(self, delta: float, speed_factor: float):
        """Update gripper percentage. *delta* is typically ±1."""
        self.gripper_state += delta * self.gripper_step * speed_factor
        self.gripper_state = float(np.clip(self.gripper_state, 0.0, 100.0))

    # ── Position saving / restoring ─────────────────────────────────

    def save_position(self):
        """Snapshot current joints + gripper."""
        self.saved_positions.append({
            "joints":  self.joint_angles.copy(),
            "gripper": self.gripper_state,
        })
        self.position_index = len(self.saved_positions) - 1

    def clear_current_position(self):
        """Remove the currently selected saved position."""
        if 0 <= self.position_index < len(self.saved_positions):
            self.saved_positions.pop(self.position_index)
            self.position_index = min(
                self.position_index, len(self.saved_positions) - 1,
            )

    def clear_all_positions(self):
        """Delete every saved position."""
        self.saved_positions.clear()
        self.position_index = -1

    def toggle_playback_order(self):
        self.playback_reversed = not self.playback_reversed

    def restore_next_position(self) -> bool:
        """Move to the next saved position. Returns ``True`` on success."""
        if not self.saved_positions:
            return False
        n = len(self.saved_positions)
        step = 1 if self.playback_reversed else -1
        self.position_index = (self.position_index + step) % n
        pos = self.saved_positions[self.position_index]
        self.joint_angles  = pos["joints"].copy()
        self.gripper_state = pos["gripper"]
        return True

    # ── Serialization ───────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Return a snapshot suitable for external consumers."""
        return {
            "joints":         self.joint_angles.copy(),
            "xyz_rpy":        self.xyz_rpy.copy(),
            "gripper":        self.gripper_state,
            "speed_factor":   self.speed_factor,
            "movement_speed": self.movement_speed,
            "arm_connected":  self.arm_connected,
            "arm_enabled":    self.arm_enabled,
            "command_mode":   self.command_mode,
            "up_level_mode":  self.up_level_mode,
            "low_level_mode": self.low_level_mode,
        }

