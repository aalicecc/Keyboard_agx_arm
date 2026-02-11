"""
Keyboard-controlled robotic arm teleoperation (physical arm).

Uses pynput for keyboard input — no window required, works in terminal/SSH.

Usage:
    python main_keyboard.py --robot piper_x --channel can0
    python main_keyboard.py --robot nero --channel can0 --effector REVO2
"""

import os
import sys
import time
import numpy as np

# ── Path setup (must come before project imports) ───────────────────
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG_ROOT)
sys.path.insert(0, os.path.join(_PKG_ROOT, "src"))

from cfg.robot_config import (
    ROBOT_DESC_CONFIGS, ROBOTS_WITH_MIT_MODE,
    DEFAULT_TCP_OFFSETS, get_robot_paths,
)
from keyboard_agx_arm.arm_controller import ArmController
from pyAgxArm import create_agx_arm_config, AgxArmFactory


# ═════════════════════════════════════════════════════════════════════
# Physical arm controller (subclass with hardware hooks)
# ═════════════════════════════════════════════════════════════════════

class PhysicalArmController(ArmController):
    """ArmController with real hardware via pyAgxArm."""

    def __init__(self, *, channel="can0", effector_type=None, **kwargs):
        robot_type = kwargs.get("robot_type", "piper")
        super().__init__(**kwargs)

        self.supports_mit = robot_type in ROBOTS_WITH_MIT_MODE

        # ── pyAgxArm driver ─────────────────────────────────────────
        self.cfg_hw   = create_agx_arm_config(robot=robot_type, comm="can", channel=channel)
        self.hw_robot = AgxArmFactory.create_arm(self.cfg_hw)

        # ── Effector ────────────────────────────────────────────────
        eff_name = (effector_type
                    or ROBOT_DESC_CONFIGS[robot_type].get("default_effector", "AGX_GRIPPER"))
        eff_const = getattr(self.hw_robot.OPTIONS.EFFECTOR, eff_name, self.hw_robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.end_effector = self.hw_robot.init_effector(eff_const)

    # ── Overridden hooks ────────────────────────────────────────────

    def _go_home(self):
        if self.state.arm_connected and self.state.arm_enabled:
            home = [0] * self.num_joints
            self.hw_robot.set_speed_percent(self.state.movement_speed)
            if self.state.command_mode == 0x00:
                self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.J)
                self.hw_robot.move_j(home)
            elif self.state.command_mode == 0xAD and self.supports_mit:
                self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.JS)
                self.hw_robot.move_js(home)
        super()._go_home()

    def _connect_and_enable(self):
        if not self.state.arm_connected:
            self.hw_robot.connect()
            self.state.arm_connected = True
        if not self.state.arm_enabled:
            while not self.hw_robot.enable():
                time.sleep(0.01)
            self.state.arm_enabled = True
            self._set_hw_motion_mode()
            time.sleep(0.1)
            self._go_home()

    def _disconnect_and_disable(self):
        if self.state.arm_connected and self.state.arm_enabled:
            self._go_home()
            try:
                self.end_effector.disable_gripper()
            except AttributeError:
                pass
            time.sleep(2)
            self.hw_robot.electronic_emergency_stop()
            time.sleep(1)
            self.hw_robot.reset()
        self.state.arm_enabled   = False
        self.state.arm_connected = False

    def _toggle_command_mode(self):
        if not self.supports_mit:
            return
        # Toggle disabled for safety (uncomment to enable)
        pass

    # ── Hardware helpers ────────────────────────────────────────────

    def _set_hw_motion_mode(self):
        """Configure the hardware motion mode based on current state."""
        if self.state.low_level_mode == "pose" and self.state.command_mode == 0x00:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.P)
        elif self.state.command_mode == 0x00:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.J)
        elif self.state.command_mode == 0xAD and self.supports_mit:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.JS)
    def send_to_hardware(self):
        """Push the current state to the physical arm every tick."""
        if not (self.state.arm_connected and self.state.arm_enabled):
            return
        self.hw_robot.set_speed_percent(self.state.movement_speed)
        if self.state.low_level_mode == "joint":
            self._send_joint_command()
        elif self.state.low_level_mode == "pose":
            self._send_pose_command()
        self._send_gripper_command()

    def _send_joint_command(self):
        joints = self.state.joint_angles[: self.num_joints].tolist()
        if self.state.command_mode == 0x00:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.J)
            self.hw_robot.move_j(joints)
        elif self.state.command_mode == 0xAD and self.supports_mit:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.JS)
            self.hw_robot.move_js(joints)

    def _send_pose_command(self):
        xr = self.state.xyz_rpy.copy()
        xr[3:] = np.radians(xr[3:])
        xr = self.hw_robot.get_tcp2flange_pose(xr.tolist())
        if self.state.command_mode == 0x00:
            self.hw_robot.set_motion_mode(self.hw_robot.OPTIONS.MOTION_MODE.P)
            self.hw_robot.move_p(xr)

    def _send_gripper_command(self):
        gv = self.state.gripper_max_width * self.state.gripper_state * 1e-2
        try:
            self.end_effector.move_gripper(gv, 3)
        except AttributeError:
            pass


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════

def main(robot_type="piper", channel="can0", effector_type=None):
    paths = get_robot_paths(robot_type)
    ctl = PhysicalArmController(
        urdf_path=paths["urdf_path"],
        mesh_path=paths["mesh_path"],
        root_name="/base_link",
        target_link=paths["target_link"],
        robot_type=robot_type,
        ik_backend="trac_ik",
        channel=channel,
        effector_type=effector_type,
    )
    ctl.set_tcp_offset(DEFAULT_TCP_OFFSETS.get(robot_type, [0, 0, 0]))

    t1 = time.time()
    try:
        while True:
            ctl.update()
            ctl.send_to_hardware()
            ctl.print_state()
            t2 = time.time()
            print(f"Loop: {(t2 - t1) * 1000:.1f} ms")
            t1 = t2
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nExiting…")
        ctl.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard robotic arm teleoperation")
    parser.add_argument("--robot", default="piper_x",
                        choices=list(ROBOT_DESC_CONFIGS.keys()))
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--effector", default=None,
                        choices=["AGX_GRIPPER", "REVO2"])
    parser.add_argument("--setup-can", action="store_true")
    args = parser.parse_args()

    if args.setup_can:
        os.system(f"sudo ip link set {args.channel} up type can bitrate 1000000")

    main(robot_type=args.robot, channel=args.channel, effector_type=args.effector)

