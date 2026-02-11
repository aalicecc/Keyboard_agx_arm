import time
import threading
from pynput import keyboard


# ── Edge detector ───────────────────────────────────────────────────

class _EdgeDetector:
    """Detect rising edge (released → pressed)."""
    __slots__ = ("_prev",)

    def __init__(self):
        self._prev = False

    def __call__(self, pressed: bool) -> bool:
        triggered = pressed and not self._prev
        self._prev = pressed
        return triggered


# ── KeyboardInput ───────────────────────────────────────────────────

class KeyboardInput:

    # ── Key bindings ────────────────────────────────────────────────
    # Action keys
    K_CONNECT  = keyboard.Key.space
    K_MODE     = '-'
    K_CMD      = '='
    K_HOME     = '1'
    K_SAVE     = '2'
    K_RESTORE  = '3'
    K_PLAYBACK = '4'
    K_SPEED    = 'q'
    K_MOVE_SPD = 'e'

    # Movement axis pairs: (negative_key, positive_key)
    AXIS_PAIRS = (
        ('a', 'd'),   # 0 — J1 / X
        ('w', 's'),   # 1 — J2 / Y
        ('z', 'x'),   # 2 — J3 / Z
        ('y', 'h'),   # 3 — J4 / Roll
        ('u', 'j'),   # 4 — J5 / Pitch
        ('i', 'k'),   # 5 — J6 / Yaw
        ('o', 'l'),   # 6 — J7 (nero only)
    )

    # Gripper keys
    K_GRIP_CLOSE = 'f'
    K_GRIP_OPEN  = 'g'

    # Long-press threshold (seconds)
    LONG_PRESS_THRESHOLD = 0.5

    # Action categories
    EDGE_ACTIONS       = ("connect", "cmd", "home", "restore")
    LONG_PRESS_ACTIONS = ("mode", "save", "playback", "speed", "move_spd")

    # ── Constructor ─────────────────────────────────────────────────

    def __init__(self):
        self._pressed_keys: set = set()
        self._lock = threading.Lock()

        # Edge detectors (single-press keys)
        self._edges = {name: _EdgeDetector() for name in self.EDGE_ACTIONS}

        # Long-press state tracking
        self._press_times: dict = {}
        self._prev_held: dict = {k: False for k in self.LONG_PRESS_ACTIONS}

        # Logical name → key constant
        self._action_key_map = {
            "connect":  self.K_CONNECT,  "cmd":      self.K_CMD,
            "home":     self.K_HOME,     "restore":  self.K_RESTORE,
            "mode":     self.K_MODE,     "save":     self.K_SAVE,
            "playback": self.K_PLAYBACK, "speed":    self.K_SPEED,
            "move_spd": self.K_MOVE_SPD,
        }

        # Start the global listener
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    # ── Raw key events (run in listener thread) ─────────────────────

    @staticmethod
    def _normalize_key(key):
        """Normalise a pynput key to a hashable identifier."""
        if isinstance(key, keyboard.Key):
            return key
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return key

    def _on_press(self, key):
        nk = self._normalize_key(key)
        with self._lock:
            self._pressed_keys.add(nk)

    def _on_release(self, key):
        nk = self._normalize_key(key)
        with self._lock:
            self._pressed_keys.discard(nk)

    # ── Key state queries ───────────────────────────────────────────

    def is_pressed(self, key_id) -> bool:
        """Thread-safe check whether *key_id* is currently held."""
        with self._lock:
            return key_id in self._pressed_keys

    def axis(self, index: int) -> float:
        """Return −1, 0, or +1 from the axis pair at *index*."""
        neg, pos = self.AXIS_PAIRS[index]
        return float(self.is_pressed(pos)) - float(self.is_pressed(neg))

    def gripper_delta(self) -> float:
        """Return −1 (close), 0, or +1 (open) for the gripper."""
        return (float(self.is_pressed(self.K_GRIP_OPEN))
                - float(self.is_pressed(self.K_GRIP_CLOSE)))

    # ── Action polling ──────────────────────────────────────────────

    def poll_edge_actions(self) -> list:
        """Return names of edge-only action keys triggered this tick."""
        triggered = []
        for name in self.EDGE_ACTIONS:
            pressed = self.is_pressed(self._action_key_map[name])
            if self._edges[name](pressed):
                triggered.append(name)
        return triggered

    def poll_long_press_actions(self) -> list:
        """Return ``[(name, is_long), …]`` for long-press keys released this tick."""
        now = time.time()
        actions = []
        for name in self.LONG_PRESS_ACTIONS:
            pressed = self.is_pressed(self._action_key_map[name])
            was = self._prev_held[name]
            if pressed and not was:
                self._press_times[name] = now
            elif not pressed and was:
                start = self._press_times.pop(name, now)
                actions.append((name, (now - start) >= self.LONG_PRESS_THRESHOLD))
            self._prev_held[name] = pressed
        return actions

    # ── Cleanup ─────────────────────────────────────────────────────

    def stop(self):
        """Stop the keyboard listener."""
        self._listener.stop()

