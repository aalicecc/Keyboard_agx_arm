import numpy as np
from scipy.spatial.transform import Rotation as R


class TcpOffset:

    def __init__(self):
        self.xyz = np.zeros(3, dtype=float)
        self.rotation = R.identity()

    def set(self, offset, degrees: bool = True):
        """set the TCP offset.

        Args:
            offset: ``[x, y, z]`` or ``[x, y, z, roll, pitch, yaw]`` (m / deg|rad).
            degrees: Whether RPY values are in degrees.
        """
        arr = np.asarray(offset, dtype=float).ravel()
        if arr.size not in (3, 6):
            raise ValueError("tcp_offset must be length 3 or 6")
        self.xyz = arr[:3].copy()
        rpy = np.zeros(3) if arr.size == 3 else arr[3:6].copy()
        self.rotation = R.from_euler("xyz", rpy, degrees=degrees)

    def apply(self, link_xyz: np.ndarray, link_rot: R):
        """End-link frame → tool-tip frame."""
        tool_rot = link_rot * self.rotation
        tool_xyz = np.asarray(link_xyz, dtype=float) + link_rot.apply(self.xyz)
        return tool_xyz, tool_rot

    def remove(self, tool_xyz: np.ndarray, tool_rot: R):
        """Tool-tip frame → end-link frame."""
        link_rot = tool_rot * self.rotation.inv()
        link_xyz = np.asarray(tool_xyz, dtype=float) - link_rot.apply(self.xyz)
        return link_xyz, link_rot

