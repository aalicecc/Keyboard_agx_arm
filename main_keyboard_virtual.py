"""
Keyboard-controlled robotic arm teleoperation (virtual / no physical arm).

Uses pynput for keyboard input — no window required, works in terminal/SSH.

Usage:
    python main_keyboard_virtual.py --robot nero
    python main_keyboard_virtual.py --robot piper_x
"""

import os
import sys
import time

# ── Path setup (must come before project imports) ───────────────────
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG_ROOT)
sys.path.insert(0, os.path.join(_PKG_ROOT, "src"))

from cfg.robot_config import ROBOT_DESC_CONFIGS, get_robot_paths
from keyboard_agx_arm.arm_controller import ArmController


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════

def main(robot_type="piper"):
    paths = get_robot_paths(robot_type)
    ctl = ArmController(
        urdf_path=paths["urdf_path"],
        mesh_path=paths["mesh_path"],
        root_name="/base_link",
        target_link=paths["target_link"],
        robot_type=robot_type,
        ik_backend="trac_ik",
    )

    t1 = time.time()
    try:
        while True:
            ctl.update()
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

    parser = argparse.ArgumentParser(description="Virtual keyboard arm teleoperation")
    parser.add_argument("--robot", default="piper",
                        choices=list(ROBOT_DESC_CONFIGS.keys()))
    args = parser.parse_args()

    main(robot_type=args.robot)

