import queue
import numpy as np
import multiprocessing as mp

try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
    _VISER_AVAILABLE = True
except ImportError:
    _VISER_AVAILABLE = False


# ── Subprocess target ───────────────────────────────────────────────

def _visualization_loop(urdf, root_name, joint_queue: mp.Queue, shutdown_event):
    """Viser server loop (runs in a child process)."""
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    urdf_vis = ViserUrdf(server, urdf, root_node_name=root_name)
    try:
        while not shutdown_event.is_set():
            try:
                joints = joint_queue.get(timeout=0.1)
                # Drain the queue so we always render the latest frame
                while not joint_queue.empty():
                    try:
                        joints = joint_queue.get_nowait()
                    except queue.Empty:
                        break
                urdf_vis.update_cfg(joints)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Visualizer] {e}")
    except KeyboardInterrupt:
        pass


# ── Visualizer class ────────────────────────────────────────────────

class Visualizer:
    """Manages a Viser visualization subprocess."""

    def __init__(self, urdf_path: str, mesh_path: str, root_name: str):
        self._process  = None
        self._queue    = None
        self._shutdown = None

        if not _VISER_AVAILABLE or not urdf_path:
            return

        try:
            urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_path)
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                pass
            self._queue    = mp.Queue(maxsize=10)
            self._shutdown = mp.Event()
            self._process  = mp.Process(
                target=_visualization_loop,
                args=(urdf, root_name, self._queue, self._shutdown),
                daemon=True,
            )
            self._process.start()
        except Exception:
            self._process = None

    # ── Public API ──────────────────────────────────────────────────

    def update(self, joint_angles: np.ndarray, gripper_pct: float,
               gripper_max_width: float, gripper_urdf_joints: int):
        """Send current configuration to the visualization process."""
        if self._queue is None:
            return
        joints = self._build_joint_list(
            joint_angles, gripper_pct, gripper_max_width, gripper_urdf_joints,
        )
        self._enqueue(joints)

    def stop(self):
        """Shut down the visualization process."""
        if self._shutdown is not None:
            self._shutdown.set()
        if self._process is not None:
            self._process.join(timeout=2)

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _build_joint_list(joint_angles, gripper_pct, max_width, urdf_joints):
        """Append gripper finger joints to the joint-angle array."""
        joints = joint_angles.copy()
        if urdf_joints > 0:
            half_width = max_width * gripper_pct * 1e-2 / 2
            for i in range(urdf_joints):
                sign = 1.0 if i % 2 == 0 else -1.0
                joints = np.append(joints, sign * half_width)
        return joints.tolist()

    def _enqueue(self, data):
        """Non-blocking put; drops the oldest item when the queue is full."""
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass

