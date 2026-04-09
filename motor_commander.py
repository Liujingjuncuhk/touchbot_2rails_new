import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline


ESP32_IP = "10.42.0.148"  # Update to your ESP32's IP
ESP32_PORT = 5005
LOCAL_PORT = 5005           # Port this PC listens on for feedback


@dataclass
class MotorFeedback:
    base_mm: float = 0.0
    vert_mm: float = 0.0
    servo_pos: list = field(default_factory=lambda: [0, 0, 0, 0])
    timestamp: float = field(default_factory=time.time)


class MotorCommander:
    def __init__(
        self,
        esp32_ip: str = ESP32_IP,
        esp32_port: int = ESP32_PORT,
        local_port: int = LOCAL_PORT,
    ):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.local_port = local_port

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", local_port))
        self._sock.settimeout(1.0)

        self._feedback: Optional[MotorFeedback] = None
        self._feedback_lock = threading.Lock()

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        # Send a no-op command so ESP32 learns our IP/port and starts sending feedback.
        # (ESP32 only sends feedback after receiving at least one packet from the PC.)
        ping = "P END"
        self._sock.sendto(ping.encode(), (self.esp32_ip, self.esp32_port))

        fb = self.wait_for_feedback(timeout=5.0)
        if fb:
            print(f"Current positions: base={fb.base_mm:.2f} mm, "
                  f"vert={fb.vert_mm:.2f} mm, servos={fb.servo_pos}")
        else:
            print("No feedback received (check IP/port).")
        self.initial_pos = fb
        self.initial_servo_pos = fb.servo_pos if fb else [2048,2048,2048,2048]
        self.cur_target = [fb.base_mm, fb.vert_mm] + fb.servo_pos if fb else [0, 0, 2048, 2048, 2048, 2048]
        self.base_min = 0
        self.base_max = 300
        self.vert_min = 0
        self.vert_max = 111
    # ------------------------------------------------------------------
    # Command sending
    # ------------------------------------------------------------------

    def send_command(
        self,
        base_mm: float,
        vert_mm: float,
        servo_pos: list,
        base_speed_mm_s: float = 0.0,
        vert_speed_mm_s: float = 0.0,
        servo_speeds: Optional[list] = None,
    ) -> None:
        """Send a motion command to the ESP32.

        Args:
            base_mm:           Base-rail stepper target position in mm.
            vert_mm:           Vertical-rail stepper target position in mm.
            servo_pos:         List of 4 Feetech servo target positions (raw units).
            base_speed_mm_s:   Base stepper speed in mm/s (0 = firmware default).
            vert_speed_mm_s:   Vertical stepper speed in mm/s (0 = firmware default).
            servo_speeds:      List of 4 servo speeds (0 = default 3500, -1 = calibrate).
        """
        if len(servo_pos) != 4:
            raise ValueError("servo_pos must have exactly 4 elements")
        if servo_speeds is None:
            servo_speeds = [0.0] * 4
        if len(servo_speeds) != 4:
            raise ValueError("servo_speeds must have exactly 4 elements")

        base_mm = max(self.base_min, min(self.base_max, base_mm))
        vert_mm = max(self.vert_min, min(self.vert_max, vert_mm))
        p = [base_mm, vert_mm] + [float(x) for x in servo_pos]
        v = [base_speed_mm_s, vert_speed_mm_s] + [float(s) for s in servo_speeds]

        msg = (
            f"P {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} "
            f"V {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]} END"
        )
        self._sock.sendto(msg.encode(), (self.esp32_ip, self.esp32_port))
        self.cur_target = p

    def send_command_rel(self, base_mm: float, vert_mm: float, servo_pos: list, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        fb = self.get_feedback()
        if fb:
            base_mm += fb.base_mm
            vert_mm += fb.vert_mm
            servo_pos = [s + f for s, f in zip(servo_pos, fb.servo_pos)]
        self.send_command(base_mm, vert_mm, servo_pos, base_speed_mm_s, vert_speed_mm_s)

    def send_stepper_only(
        self,
        base_mm: float,
        vert_mm: float,
        base_speed_mm_s: float = 0.0,
        vert_speed_mm_s: float = 0.0,
    ) -> None:
        """Move steppers only; keep servos at their current feedback positions."""
        pos = self.get_positions()
        cur_servo_pos = pos['servo_pos'] if pos else self.initial_servo_pos
        self.send_command(
            base_mm=base_mm,
            vert_mm=vert_mm,
            servo_pos=cur_servo_pos,
            base_speed_mm_s=base_speed_mm_s,
            vert_speed_mm_s=vert_speed_mm_s,
        )

    def send_stepper_only_rel(self, base_mm: float, vert_mm: float, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        fb = self.get_feedback()
        if fb:
            base_mm += fb.base_mm
            vert_mm += fb.vert_mm
        self.send_stepper_only(base_mm, vert_mm, base_speed_mm_s, vert_speed_mm_s)

    def send_servos_only(
        self,
        servo_pos: list,
        servo_speeds: Optional[list] = None,
    ) -> None:
        """Move servos only; hold steppers at their current feedback positions."""
        fb = self.get_feedback()
        base_mm = fb.base_mm if fb else 0.0
        vert_mm = fb.vert_mm if fb else 0.0
        self.send_command(
            base_mm=base_mm,
            vert_mm=vert_mm,
            servo_pos=servo_pos,
            servo_speeds=servo_speeds,
        )

    def send_servos_only_rel(self, servo_pos: list, servo_speeds: Optional[list] = None):
        fb = self.get_feedback()
        if fb:
            servo_pos = [s + f for s, f in zip(servo_pos, fb.servo_pos)]
        self.send_servos_only(servo_pos, servo_speeds)

    def calibrate_servo(self, servo_index: int) -> None:
        """Set the current physical position of one servo as its midpoint (2048).

        Args:
            servo_index: 0-based index into the 4-servo array.
        """
        if servo_index not in range(4):
            raise ValueError("servo_index must be 0-3")
        speeds = [0.0, 0.0, 0.0, 0.0]
        speeds[servo_index] = -1.0  # -1 triggers CalibrationOfs on firmware
        fb = self.get_feedback()
        base_mm = fb.base_mm if fb else 0.0
        vert_mm = fb.vert_mm if fb else 0.0
        self.send_command(
            base_mm=base_mm,
            vert_mm=vert_mm,
            servo_pos=[0, 0, 0, 0],
            servo_speeds=speeds,
        )

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def interpolate_trajectory(
        self,
        t_list: list,
        waypoint_list: list,
        freq: float = 50.0,
    ) -> tuple:
        """Resample a sparse trajectory using cubic spline interpolation.

        Args:
            t_list:        Times in seconds, starting from 0. Length N >= 2.
            waypoint_list: N waypoints, each a 6-element list:
                           [base_mm, vert_mm, s1, s2, s3, s4].
            freq:          Output sample rate in Hz.

        Returns:
            (t_dense, waypoints_dense) — 1-D array of times and a list of
            6-element waypoints at each sample.
        """
        t = np.array(t_list, dtype=float)
        wp = np.array(waypoint_list, dtype=float)  # shape (N, 6)
        cs = CubicSpline(t, wp)
        t_dense = np.arange(t[0], t[-1], 1.0 / freq)
        wp_dense = cs(t_dense)  # shape (M, 6)
        return t_dense, wp_dense.tolist()

    def execute_trajectory(
        self,
        t_list: list,
        waypoint_list: list,
        freq: float = 50.0,
    ) -> None:
        """Stream a trajectory to the ESP32 at a fixed command rate.

        Interpolates the sparse waypoints with a cubic spline, then sends
        one command per interval, sleeping to stay on the wall-clock schedule.

        Args:
            t_list:        Times in seconds, starting from 0. Length N >= 2.
            waypoint_list: N waypoints, each a 6-element list:
                           [base_mm, vert_mm, s1, s2, s3, s4].
            freq:          Command rate in Hz (default 50).
        """
        t_dense, wp_dense = self.interpolate_trajectory(t_list, waypoint_list, freq)
        start = time.perf_counter()
        for t_target, wp in zip(t_dense, wp_dense):
            while time.perf_counter() - start < t_target:
                pass  # busy-wait for accurate timing
            self.send_command(
                base_mm=wp[0],
                vert_mm=wp[1],
                servo_pos=[int(round(wp[j])) for j in range(2, 6)],
            )

    # ------------------------------------------------------------------
    # Feedback reading
    # ------------------------------------------------------------------

    def get_feedback(self) -> Optional[MotorFeedback]:
        """Return the most recently received feedback snapshot, or None."""
        with self._feedback_lock:
            return self._feedback

    def wait_for_feedback(self, timeout: float = 2.0) -> Optional[MotorFeedback]:
        """Block until a feedback packet arrives or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            fb = self.get_feedback()
            if fb is not None:
                return fb
            time.sleep(0.01)
        return None

    def wait_until_reached(
        self,
        base_mm: Optional[float] = None,
        vert_mm: Optional[float] = None,
        servo_pos: Optional[list] = None,
        tol_mm: float = 0.1,
        servo_tol: int = 3,
        timeout: float = 60,
    ) -> bool:
        """Block until feedback positions are within tolerance of targets.

        Pass only the axes you care about; None axes are skipped.

        Args:
            base_mm:    Target base-rail position in mm, or None to skip.
            vert_mm:    Target vertical-rail position in mm, or None to skip.
            servo_pos:  List of 4 target servo positions, or None to skip.
                        Individual elements may be None to skip that servo.
            tol_mm:     Acceptance window for steppers in mm.
            servo_tol:  Acceptance window for servo raw units.
            timeout:    Maximum seconds to wait.

        Returns:
            True if all checked axes reached their targets, False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            fb = self.get_feedback()
            if fb is not None:
                reached = True
                if base_mm is not None and abs(fb.base_mm - base_mm) > tol_mm:
                    reached = False
                if vert_mm is not None and abs(fb.vert_mm - vert_mm) > tol_mm:
                    reached = False
                if servo_pos is not None:
                    for i, target in enumerate(servo_pos):
                        if target is not None and abs(fb.servo_pos[i] - target) > servo_tol:
                            reached = False
                            break
                if reached:
                    return True
            time.sleep(0.02)
        return False

    def get_positions(self) -> Optional[dict]:
        """Return a dict with current positions, or None if no feedback yet.

        Returns:
            {
                'base_mm':   float,
                'vert_mm':   float,
                'servo_pos': [int, int, int, int],
                'timestamp': float,
            }
        """
        fb = self.get_feedback()
        if fb is None:
            return None
        return {
            "base_mm": fb.base_mm,
            "vert_mm": fb.vert_mm,
            "servo_pos": list(fb.servo_pos),
            "timestamp": fb.timestamp,
        }

    # ------------------------------------------------------------------
    # Internal receive loop
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(512)
                self._parse_feedback(data.decode().strip())
            except socket.timeout:
                continue
            except OSError:
                break

    def _parse_feedback(self, msg: str) -> None:
        """Parse ESP32 feedback string: FB,basePosMM,vertPosMM,s1,s2,s3,s4"""
        if not msg.startswith("FB,"):
            return
        parts = msg.split(",")
        if len(parts) != 7:
            return
        try:
            base_mm = float(parts[1])
            vert_mm = float(parts[2])
            servo_pos = [int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])]
        except ValueError:
            return
        with self._feedback_lock:
            self._feedback = MotorFeedback(
                base_mm=base_mm,
                vert_mm=vert_mm,
                servo_pos=servo_pos,
                timestamp=time.time(),
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def return_zeros(self, return_time = 1) -> None:
        # return all motor to initial position
        self.send_command(0, 0, self.initial_servo_pos, 10, 10, [500,500,500,500])

    def close(self) -> None:
        """Stop the receive thread and release the socket."""
        self._running = False
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    mc = MotorCommander(esp32_ip= ESP32_IP)
    motor_initial = mc.initial_servo_pos
    # t_list = [0.0, 1.0, 2.5, 4.0]
    # waypoints = [
    #     [0,   0] + motor_initial,
    #     [20,  10] + motor_initial,
    #     [30,  20] + motor_initial,
    #     [0,   0] + motor_initial,
    # ]
    # mc.execute_trajectory(t_list, waypoints, freq=200)
    # mc.send_stepper_only(0, 110, 0, 10)
    # time.sleep(11.5)
    # mc.return_zeros()

    # mc.close()
