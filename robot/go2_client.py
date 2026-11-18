"""Go2 EDU client for the budget-VLN inference server.

Loop: capture at ~1 fps -> send frames -> server returns 4 actions -> execute the
first EXEC_N of them (receding horizon, as in the Uni-NaVid deployment) -> repeat
until the model says stop or the budget runs out.

No ROS. Talks DDS to the dog via unitree_sdk2py and JSON-lines TCP to the server.

  # dry run, no robot at all (verifies network + protocol + preprocessing):
  python go2_client.py --host <gpu-host> --port 8878 --category chair --budget 150 \
      --style objectnav --fake-cam /path/to/any_indoor_video.mp4 --no-move

  # on the robot (or a laptop wired to it):
  python go2_client.py --host <gpu-host> --port 8877 --category chair --budget 400 \
      --iface eth0

Safety: ctrl-C stops the dog (StopMove) before exiting; --no-move runs the whole
loop but never commands motion; --max-decisions caps the episode.
"""
import argparse
import base64
import io
import json
import socket
import sys
import time

import numpy as np
from PIL import Image

# --- action model: must match the simulator's actuation spec -----------------
FORWARD_M = 0.25          # metres per 'forward' primitive
TURN_DEG = 30.0           # degrees per 'left'/'right' primitive
VX = 0.6                  # m/s while walking (merged chunks ramp once, so a bit quicker is fine)
VYAW = 1.0                # rad/s while turning
CMD_HZ = 10               # DDS velocity commands are watchdogged; resend at 10 Hz
POS_TOL = 0.02            # m: stop a forward step when within this of the target
K_YAW = 1.5               # rad/s per rad of heading error while walking forward
LOOKAHEAD_M = 0.15        # fire the chunk-end observation + model call this far before the last segment ends
LOOKAHEAD_DEG = 10.0      # same for a turn segment
K_LAT = 1.0               # m/s per m of lateral offset from the start line while walking forward
YAW_TOL = np.radians(1.5) # rad: same for turns


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Odom:
    """Body-frame odometry from the sport-mode state (position and IMU yaw).

    Each primitive is closed on this instead of on time: a commanded 0.5 m/s for
    0.5 s does not put the body 0.25 m ahead on a real floor (slip, ramp-up,
    the gait's own sway), and the training data assumes it does.
    """
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self.state, self.n, self.t = None, 0, 0.0
        self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sub.Init(self._cb, 10)

    def _cb(self, msg):
        self.state = (float(msg.position[0]), float(msg.position[1]), float(msg.imu_state.rpy[2]))
        self.n += 1
        self.t = time.time()

    def alive(self):
        return self.state is not None and (time.time() - self.t) < 0.5

    def pose(self):
        return self.state

    def wait(self, timeout=3.0):
        t_end = time.time() + timeout
        while time.time() < t_end and self.state is None:
            time.sleep(0.05)
        return self.state is not None


def crop_to_hfov(img, src_hfov, dst_hfov=79.0):
    """Centre-crop a wide-FOV image toward the training HFOV, then resize to 640x480.

    The Go2's front camera is a fisheye (~150 deg); the model was trained on a 79 deg
    pinhole. Cropping the centre is the cheap approximation -- it does not undo the
    fisheye distortion, but it puts roughly the right angular extent in frame.
    """
    w, h = img.size
    cw = int(w * np.tan(np.radians(dst_hfov / 2)) / np.tan(np.radians(src_hfov / 2))) \
        if (src_hfov and src_hfov > dst_hfov) else w
    ch = min(int(cw * 3 / 4), h)          # 4:3 like the training camera, not 16:9 squashed
    cw = min(cw, w)
    img = img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    return img.resize((640, 480))


class Server:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=120)
        self.f = self.sock.makefile("rw")

    def rpc(self, msg):
        self.f.write(json.dumps(msg) + "\n")
        self.f.flush()
        return json.loads(self.f.readline())



def resolve_iface(iface):
    """Map 'auto' to whichever interface currently holds the dog's subnet.

    The USB ethernet adapter renumbers on every replug (en8 -> en23 -> ...) and
    loses its static IP with it, so a name remembered from one run is stale by
    the next. Looking the address up each time removes the whole class of
    "does not match an available interface" failures.
    """
    if iface not in ("", "auto", "none", None):
        return iface
    import re
    import subprocess
    found = iface_addrs()
    # the lab WiFi can also sit on 192.168.123.x; the adapter go2_link.sh set up
    # holds .99, so prefer that one over any other holder of the subnet
    for name, ip in found:
        if ip == "192.168.123.99":
            return name
    for name, ip in found:
        if not name.startswith("en0"):
            return name
    if found:
        return found[0][0]
    raise SystemExit("no interface holds a 192.168.123.x address. "
                     "Plug the adapter in and run: sudo ./go2_link.sh")


def iface_addrs():
    """[(iface, ip)] for every interface address on the dog's subnet."""
    import re
    import subprocess
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    cur, found = None, []
    for line in out.splitlines():
        m = re.match(r"^([A-Za-z0-9]+):", line)
        if m:
            cur = m.group(1)
        else:
            m = re.search(r"inet (192\.168\.123\.\d+)", line)
            if m and cur:
                found.append((cur, m.group(1)))
    return found


def iface_ip(iface):
    for name, ip in iface_addrs():
        if name == iface:
            return ip
    return None


def init_dds(iface):
    """Must run before ANY client (camera or sport) -- they share the domain."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    iface = resolve_iface(iface)
    print(f"[client] DDS on {iface}", flush=True)
    ChannelFactoryInitialize(0, iface)


class Robot:
    """SportClient for posture, ObstaclesAvoidClient for motion; no-op if disabled.

    Driving through the obstacle-avoidance service means the robot's own layer
    vetoes commands that would hit something -- the model has no collision sense
    at all, and left unguarded it walked into a wall on the first real run.
    """
    guard = None                              # LidarGuard, set from main when --guard is on
    abort = False                             # set by the SIGINT handler: stop commanding motion

    def __init__(self, enabled, avoid=True, odom=True, drive="vel", blend=True, nudge=True):
        self.enabled = enabled
        self.sport = None
        self.avoid = None
        self.odom = None
        self.drive = drive        # vel: closed-loop velocity; traj: TrajectoryFollow waypoints
        self.blend = blend        # no halt between the segments of one chunk
        self.last_drift = None       # (lateral m, heading rad) at the end of the last forward
        self.nudge_dir = "left"      # turn direction used when a forward is blocked: the model's most recent turn
        self.nudge = nudge
        self.verbose = True          # per-segment start/target/end line
        self.on_segment = None       # recorder hook
        self.last_seg = None
        if not enabled:
            return
        if odom:
            self.odom = Odom()
            if self.odom.wait(3.0):
                x, y, yaw = self.odom.pose()
                print(f"[robot] odometry OK: x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):.1f} deg", flush=True)
            else:
                print("[robot] WARNING: no sport-mode state received; falling back to timed driving", flush=True)
                self.odom = None
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        self.sport = SportClient()
        self.sport.SetTimeout(10.0)
        self.sport.Init()
        self.sport.StandUp()
        time.sleep(2)
        # StandUp is a rigid stand; BalanceStand is the walk-ready state that
        # Move() actually drives. Verified against the official go2 example.
        self.sport.BalanceStand()
        time.sleep(1)
        if not avoid:
            # the service remembers its switch across sessions; make sure it is off
            # so the raw Move() commands are not vetoed by a stale avoidance mode
            try:
                from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import \
                    ObstaclesAvoidClient
                oa = ObstaclesAvoidClient(); oa.SetTimeout(3.0); oa.Init()
                oa.UseRemoteCommandFromApi(False)
                oa.SwitchSet(False)
                print("[robot] obstacle avoidance OFF", flush=True)
            except Exception as e:
                print("[robot] could not switch obstacle avoidance off:", e, flush=True)
        if avoid:
            from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import \
                ObstaclesAvoidClient
            self.avoid = ObstaclesAvoidClient()
            self.avoid.SetTimeout(3.0)
            self.avoid.Init()
            for _ in range(20):
                if self.avoid.SwitchGet()[1]:
                    break
                self.avoid.SwitchSet(True)
                time.sleep(0.1)
            if not self.avoid.SwitchGet()[1]:
                print("[robot] WARNING: obstacle avoidance would not enable", flush=True)
                self.avoid = None
            else:
                self.avoid.UseRemoteCommandFromApi(True)
                time.sleep(0.5)
                print("[robot] obstacle avoidance ON", flush=True)

    def _mover(self):
        return self.avoid if self.avoid else self.sport

    def _cmd(self, vx, vyaw, vy=0.0):
        self.last_cmd = (vx, vyaw, vy)
        self._mover().Move(vx, vy, vyaw)

    def coast(self, alive, max_s=0.4):
        """Keep the last velocity for up to max_s while alive() is true (the next chunk is
        being decided), then halt. Bounds how far the dog can drift on a slow model call."""
        t_end = time.time() + max_s
        last = getattr(self, "last_cmd", None) if self.enabled else None
        while alive() and time.time() < t_end:
            if last is not None:
                self._cmd(*last)
            time.sleep(1.0 / CMD_HZ)
        if alive():
            self._halt()

    def _halt(self):
        if not self.enabled:
            return
        if self.avoid:
            self.avoid.Move(0.0, 0.0, 0.0)
        else:
            self.sport.StopMove()

    def _drive_timed(self, vx, vyaw, seconds):
        # open loop: command a velocity for d/v seconds and hope. Fallback only.
        t_end = time.time() + seconds
        while time.time() < t_end:
            self._cmd(vx, vyaw)
            time.sleep(1.0 / CMD_HZ)
        self._halt()

    # TrajectoryFollow (sport API 1018) is in the firmware but not in this Python SDK;
    # the C++ client sends 30 waypoints in the odometry frame as a JSON list.
    TRAJ_API = 1018
    TRAJ_N = 30

    def _traj_send(self, pts):
        js = [{"t_from_start": t, "x": x, "y": y, "yaw": yaw, "vx": vx, "vy": 0.0, "vyaw": vyaw}
              for (t, x, y, yaw, vx, vyaw) in pts]
        code, _ = self.sport._Call(self.TRAJ_API, json.dumps(js))
        return code

    def _segment(self, kind, count, on_boundary=None, every=2, offset=0, halt=True, on_lookahead=None):
        """One merged motion: `count` forward steps (count*0.25 m along the initial
        heading) or `count` turns (count*30 deg, kind=left/right). Closed on odometry;
        on_boundary() is called, without stopping, each time `every` primitives' worth of
        displacement has been covered (that is where the model's next frame comes from).
        Returns the number of primitives credited (rounded from the measured motion).
        halt=False leaves the dog moving at the end (the next segment's first command takes
        over), so a turn flows into the following forward without a full stop."""
        x0, y0, yaw0 = self.odom.pose()
        t_seg0 = time.time()
        if kind == "forward":
            tgt = (x0 + count * FORWARD_M * np.cos(yaw0), y0 + count * FORWARD_M * np.sin(yaw0), yaw0)
        else:
            tgt = (x0, y0, wrap_angle(yaw0 + (1.0 if kind == "left" else -1.0) * count * np.radians(TURN_DEG)))
        if kind == "forward":
            unit, total, tol = FORWARD_M, count * FORWARD_M, POS_TOL
            hx, hy = np.cos(yaw0), np.sin(yaw0)
            progress = lambda: (self.odom.pose()[0] - x0) * hx + (self.odom.pose()[1] - y0) * hy
            t_end = time.time() + 2.0 * total / VX + 1.0
        else:
            unit, total, tol = np.radians(TURN_DEG), count * np.radians(TURN_DEG), YAW_TOL
            sgn = 1.0 if kind == "left" else -1.0
            progress = lambda: sgn * wrap_angle(self.odom.pose()[2] - yaw0)
            t_end = time.time() + 2.0 * total / VYAW + 1.0
        if self.drive == "traj":
            # position command: waypoints along the segment at the cruise speed, then hold
            v = VX if kind == "forward" else VYAW
            dur = total / v
            pts = []
            for i in range(self.TRAJ_N):
                t = i * max(dur, 0.5) / (self.TRAJ_N - 1)
                d = min(total, v * t)
                if kind == "forward":
                    pts.append((t, x0 + d * hx, y0 + d * hy, yaw0, v if d < total else 0.0, 0.0))
                else:
                    pts.append((t, x0, y0, wrap_angle(yaw0 + sgn * d), 0.0, sgn * v if d < total else 0.0))
            self._traj_send(pts)
        first = every - (offset % every)          # primitives until the next chunk boundary
        next_mark = first * unit
        done = 0.0
        guard_hit = False
        gf = 1.0                                   # speed factor from the lidar guard
        if kind == "forward" and self.guard is not None and self.guard.ok:
            c = self.guard.clearance(x0, y0, yaw0)
            if c is not None and c < self.guard.stop_m:
                print(f"[guard] forward refused: obstacle {100 * c:.0f} cm ahead", flush=True)
                guard_hit = True; t_end = 0.0
        la = (LOOKAHEAD_M if kind == "forward" else np.radians(LOOKAHEAD_DEG)) if on_lookahead is not None else None
        fired = False
        while time.time() < t_end:
            if self.abort:
                break
            done = progress()
            if done >= total - tol:
                break
            if on_boundary is not None and done >= next_mark - tol and next_mark < total - tol:
                on_boundary()
                next_mark += every * unit
            if la is not None and not fired and done >= total - la:
                on_lookahead(offset + count); fired = True   # observation + model call overlap the end of this motion
            if kind == "forward" and self.guard is not None and self.guard.ok:
                c = self.guard.clearance(*self.odom.pose())
                if c is not None and c < self.guard.stop_m:
                    print(f"[guard] forward cut at {done:.2f} of {total:.2f} m: obstacle {100 * c:.0f} cm ahead", flush=True)
                    guard_hit = True
                    break
                gf = 0.5 if (c is not None and c < self.guard.slow_m) else 1.0
            if self.drive != "traj":
                rem = total - done
                if kind == "forward":
                    # closed-loop line following: hold the initial heading with vyaw and
                    # cancel lateral drift with vy, both from odometry (a forward is no
                    # longer open loop sideways; without this a 1 m walk can veer 5-10 cm)
                    x, y, yaw = self.odom.pose()
                    yaw_err = wrap_angle(yaw - yaw0)
                    lateral = -(x - x0) * hy + (y - y0) * hx          # +left of the start line
                    vyaw = float(np.clip(-K_YAW * yaw_err, -0.5, 0.5))
                    vy = float(np.clip(-K_LAT * lateral, -0.15, 0.15))
                    self._cmd(gf * (VX if rem > 0.10 else 0.5 * VX), vyaw, vy)
                else:
                    self._cmd(0.0, sgn * (VYAW if rem > np.radians(10) else 0.5 * VYAW))
            time.sleep(1.0 / CMD_HZ)
        blocked = kind == "forward" and (done < 0.2 * total or guard_hit)
        if la is not None and not fired and not blocked:
            on_lookahead(offset + count)         # segment ended before the lookahead point (short segment)
            fired = True
        if halt:
            self._halt()
        if kind == "forward":
            x, y, yaw = self.odom.pose()
            self.last_drift = (-(x - x0) * hy + (y - y0) * hx, wrap_angle(yaw - yaw0))
        x1, y1, yaw1 = self.odom.pose()
        seg = {"t0": t_seg0, "t1": time.time(), "kind": kind, "count": count,
               "start": [round(x0, 3), round(y0, 3), round(float(np.degrees(yaw0)), 1)],
               "target": [round(tgt[0], 3), round(tgt[1], 3), round(float(np.degrees(tgt[2])), 1)],
               "end": [round(x1, 3), round(y1, 3), round(float(np.degrees(yaw1)), 1)],
               "pos_err_m": round(float(np.hypot(x1 - tgt[0], y1 - tgt[1])), 3),
               "yaw_err_deg": round(float(np.degrees(wrap_angle(yaw1 - tgt[2]))), 1), "measured": round(float(done), 3)}
        self.last_seg = seg
        if self.verbose:
            print(f"[seg] {kind} x{count}: start ({seg['start'][0]:.2f},{seg['start'][1]:.2f},{seg['start'][2]:.0f}deg) -> target ({seg['target'][0]:.2f},{seg['target'][1]:.2f},{seg['target'][2]:.0f}deg) | end ({seg['end'][0]:.2f},{seg['end'][1]:.2f},{seg['end'][2]:.0f}deg) | pos err {100 * seg['pos_err_m']:.0f} cm, yaw err {seg['yaw_err_deg']:+.1f} deg", flush=True)
        if self.on_segment is not None:
            self.on_segment(seg)
        credited = int(round(done / unit))
        if guard_hit:
            seg["status"] = "blocked_lidar"
            credited = max(1, credited)           # a forward into an obstacle still costs a step, as in the simulator
        return max(0, min(count, credited)), done, fired, blocked

    def run_chunk(self, actions, on_boundary=None, every=2, on_lookahead=None, final_halt=True):
        """Execute a list of primitives as merged segments (runs of the same primitive
        become one motion). Returns the number of primitives executed."""
        if not self.enabled:
            time.sleep(0.2 * len(actions))
            if on_boundary is not None:
                for i in range(1, len(actions) + 1):
                    if i % every == 0 and i < len(actions):
                        on_boundary()
            if on_lookahead is not None:
                on_lookahead(len(actions))
            return len(actions)
        if self.odom is None or not self.odom.alive():
            n = 0
            for i, act in enumerate(actions, 1):
                self.do(act)
                n += 1
                if on_boundary is not None and i % every == 0:
                    on_boundary()
            return n
        n = 0
        i = 0
        while i < len(actions):
            kind = actions[i]
            j = i
            while j < len(actions) and actions[j] == kind:
                j += 1
            count = j - i
            # blend: no stop between segments of one chunk; only the chunk end halts
            last = j >= len(actions)
            got, meas, fired, blocked = self._segment(kind, count, on_boundary=on_boundary, every=every, offset=n,
                                                      halt=(last and final_halt) or (not last and not self.blend),
                                                      on_lookahead=on_lookahead if last else None)
            unit = "m" if kind == "forward" else "deg"
            shown = meas if kind == "forward" else np.degrees(meas)
            extra = ""
            if kind == "forward" and self.last_drift is not None:
                extra = f"  (drift {100 * self.last_drift[0]:+.0f} cm, heading {np.degrees(self.last_drift[1]):+.1f} deg)"
            print(f"[robot] {kind} x{count}: {shown:.2f} {unit} of "
                  f"{count * (FORWARD_M if kind == 'forward' else TURN_DEG):.2f} {unit}{extra}", flush=True)
            if blocked and self.nudge:
                # The avoidance service (or a wall) is holding the dog: the model cannot tell, it
                # will keep asking for forward from the same view. One 30 deg turn, then the
                # next observation comes from the new heading. The turn direction persists
                # across consecutive blocks so the dog keeps rotating until a way opens, and
                # resets after a forward that went through.
                print(f"[robot] blocked ({meas:.2f} m of {count * FORWARD_M:.2f}); nudge {self.nudge_dir} 30 deg", flush=True)
                self._segment(self.nudge_dir, 1, halt=not self.blend)
                n += got + 1
                if on_lookahead is not None and not fired:
                    on_lookahead(n)
                elif on_lookahead is None and final_halt:
                    self._halt()
                return n
            if kind in ("left", "right"):
                self.nudge_dir = kind         # a blocked forward is nudged the way the model last turned
            n += count
            i = j
            if on_boundary is not None and i < len(actions) and n % every == 0:
                on_boundary()                    # boundary that fell exactly on a segment end
        if final_halt:
            time.sleep(0.1)       # settle before the next observation
        return n

    def do(self, action):
        if not self.enabled:
            time.sleep(0.3)
            return
        if self.odom is None or not self.odom.alive():
            if action == "forward":
                self._drive_timed(VX, 0.0, FORWARD_M / VX)
            elif action == "left":
                self._drive_timed(0.0, VYAW, np.radians(TURN_DEG) / VYAW)
            elif action == "right":
                self._drive_timed(0.0, -VYAW, np.radians(TURN_DEG) / VYAW)
            time.sleep(0.15)
            return
        self.run_chunk([action], on_boundary=None)

    def stop(self):
        """Emergency-grade halt: every step guarded so one failing RPC cannot leave the dog
        walking on its last Move command (seen 2026-09-14 after a Ctrl-C)."""
        if not self.enabled:
            return
        for step in ("sport.StopMove", "avoid.zero", "avoid.release", "sport.StopMove"):
            try:
                if step == "sport.StopMove" and self.sport:
                    self.sport.StopMove()
                elif step == "avoid.zero" and self.avoid:
                    for _ in range(3):
                        self.avoid.Move(0.0, 0.0, 0.0); time.sleep(0.05)
                elif step == "avoid.release" and self.avoid:
                    self.avoid.UseRemoteCommandFromApi(False)
            except Exception as e:
                print(f"[robot] stop: {step} failed: {e}", flush=True)
        self.last_cmd = None
        print("[robot] halted", flush=True)


class Heartbeat:
    """UDP heartbeat to deadman.py on the Jetson (10 Hz). If this process dies, the laptop
    sleeps or the cable drops, the Jetson halts the dog within half a second."""
    def __init__(self, target, bind_ip=None):
        import threading
        host, port = target.rsplit(":", 1)
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind_ip:
            self.sock.bind((bind_ip, 0))
        self.on = True
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
        print(f"[client] heartbeat -> {host}:{port}", flush=True)

    def _loop(self):
        while self.on:
            try:
                self.sock.sendto(b"hb", self.addr)
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self.on = False


class LidarGuard:
    """Obstacle guard from the dog's own lidar (rt/utlidar/cloud_deskewed, odom frame): the
    latest cloud is kept in memory and, while a forward segment runs, points inside a corridor
    ahead of the body are checked at 10 Hz. Closer than stop_m in front of the nose -> the
    segment is cut (the client treats it as a blocked forward: nudge in the fast profile, the
    model is asked again from the current view in the slow profile, and one primitive of
    budget is charged as in the simulator). Under slow_m the walk drops to half speed.
    Used because the native avoidance service rejects SwitchSet (3203) on some firmware."""
    def __init__(self, stop_m=0.15, slow_m=0.8, half_width=0.30, z_band=(0.05, 0.6), nose_m=0.40):
        import threading
        self.stop_m, self.slow_m, self.half_width, self.z_band, self.nose_m = stop_m, slow_m, half_width, z_band, nose_m
        self.pts = None; self.t = 0.0; self.lock = threading.Lock(); self.ok = False
        self.period = 0.25                     # parse at most 4 clouds a second
        self._last_parse = 0.0
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
            self.sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
            self.sub.Init(self._cb, 10)
            t0 = time.time()
            while time.time() - t0 < 5.0 and self.pts is None:
                time.sleep(0.1)
            self.ok = self.pts is not None
            print(f"[guard] lidar obstacle guard {'ON' if self.ok else 'no cloud yet'}: stop when < {stop_m:.2f} m ahead of the nose, half speed under {slow_m:.1f} m", flush=True)
        except Exception as e:
            print("[guard] lidar unavailable, guard OFF:", e, flush=True)

    def _cb(self, msg):
        # throttled and decimated: parsing every cloud in Python starves the DDS callback
        # thread (odometry and the recorder's own lidar subscription both go quiet)
        t = time.time()
        if t - self._last_parse < self.period:
            return
        self._last_parse = t
        try:
            names = {fl.name: fl for fl in msg.fields}
            step = msg.point_step; raw = bytes(msg.data); n = len(raw) // step
            buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)[::2]
            xyz = np.stack([buf[:, names[k].offset:names[k].offset + 4].copy().view(np.float32)[:, 0] for k in ("x", "y", "z")], axis=1)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            xyz = xyz[(xyz[:, 2] > self.z_band[0]) & (xyz[:, 2] < self.z_band[1])]
            with self.lock:
                self.pts = xyz[:, :2].astype(np.float32); self.t = time.time()
        except Exception:
            pass

    def clearance(self, x, y, yaw):
        """metres of free space ahead of the nose along the current heading (None: no recent cloud)."""
        with self.lock:
            pts, t = self.pts, self.t
        if pts is None or time.time() - t > 2.0 or len(pts) == 0:
            return None
        dx = pts[:, 0] - x; dy = pts[:, 1] - y
        c, s_ = np.cos(yaw), np.sin(yaw)
        fwd = dx * c + dy * s_; lat = -dx * s_ + dy * c
        m = (np.abs(lat) < self.half_width) & (fwd > self.nose_m) & (fwd < 3.0)
        if not m.any():
            return 3.0
        return float(fwd[m].min() - self.nose_m)


class Recorder:
    """--record DIR: everything needed to make the run video afterwards (make_run_video.py):
    odometry at 10 Hz (poses.jsonl), RealSense JPEGs at ~5 fps (frames/*.jpg + index.jsonl),
    the lidar cloud at ~2 Hz (lidar/*.npy + index.jsonl, frame id in meta.json), and every
    decision (decisions.jsonl). Runs on threads; the motion loop is untouched."""
    def __init__(self, d, odom, cam, lidar=True):
        import os, threading
        self.d, self.odom, self.cam, self.on = d, odom, cam, True
        os.makedirs(os.path.join(d, "frames"), exist_ok=True); os.makedirs(os.path.join(d, "lidar"), exist_ok=True)
        self.t0 = time.time()
        self.fp = open(os.path.join(d, "poses.jsonl"), "w"); self.ff = open(os.path.join(d, "frames", "index.jsonl"), "w")
        self.fd = open(os.path.join(d, "decisions.jsonl"), "w"); self.fl = open(os.path.join(d, "lidar", "index.jsonl"), "w")
        self.fs = open(os.path.join(d, "segments.jsonl"), "w")
        self.nf = self.nl = 0; self.lidar_frame = None; self.last_lidar = 0.0
        threading.Thread(target=self._pose_loop, daemon=True).start()
        threading.Thread(target=self._cam_loop, daemon=True).start()
        if lidar:
            try:
                from unitree_sdk2py.core.channel import ChannelSubscriber
                from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
                self.sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
                self.sub.Init(self._lidar_cb, 10)
            except Exception as e:
                print("[record] lidar unavailable:", e, flush=True)
        print(f"[record] -> {d}", flush=True)

    def _pose_loop(self):
        while self.on:
            if self.odom is not None and self.odom.alive():
                x, y, yaw = self.odom.pose()
                self.fp.write(json.dumps({"t": time.time(), "x": x, "y": y, "yaw": yaw}) + "\n")
            time.sleep(0.1)

    def _cam_loop(self):
        import os
        while self.on:
            t = time.time(); b = self.cam.grab_jpeg()
            if b:
                name = f"{self.nf:06d}.jpg"
                with open(os.path.join(self.d, "frames", name), "wb") as fh:
                    fh.write(b)
                self.ff.write(json.dumps({"t": t, "file": name}) + "\n"); self.nf += 1
            time.sleep(max(0.0, 0.2 - (time.time() - t)))

    def _lidar_cb(self, msg):
        import os
        t = time.time()
        if t - self.last_lidar < 0.5 or not self.on:
            return
        try:
            self.lidar_frame = msg.header.frame_id
            names = {fl.name: fl for fl in msg.fields}
            step = msg.point_step; raw = bytes(msg.data); n = len(raw) // step
            buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
            xyz = np.stack([buf[:, names[k].offset:names[k].offset + 4].copy().view(np.float32)[:, 0] for k in ("x", "y", "z")], axis=1)
            xyz = xyz[np.isfinite(xyz).all(axis=1)][::4].astype(np.float32)
            np.save(os.path.join(self.d, "lidar", f"{self.nl:05d}.npy"), xyz)
            self.fl.write(json.dumps({"t": t, "file": f"{self.nl:05d}.npy", "n": int(len(xyz))}) + "\n"); self.nl += 1
            self.last_lidar = t
        except Exception as e:
            print("[record] lidar parse failed:", e, flush=True)

    def segment(self, seg):
        self.fs.write(json.dumps(seg) + "\n"); self.fs.flush()

    def decision(self, k, N, acts, latency, raw, pose):
        self.fd.write(json.dumps({"t": time.time(), "k": k, "N": N, "actions": acts, "latency_s": latency, "raw": raw, "pose": pose}) + "\n"); self.fd.flush()

    def stop(self, meta=None):
        import os
        self.on = False; time.sleep(0.3)
        for fh in (self.fp, self.ff, self.fd, self.fl, self.fs):
            fh.close()
        m = {"t0": self.t0, "t1": time.time(), "frames": self.nf, "lidar_clouds": self.nl, "lidar_frame_id": self.lidar_frame}
        m.update(meta or {})
        json.dump(m, open(os.path.join(self.d, "meta.json"), "w"))
        print(f"[record] saved {self.nf} frames, {self.nl} lidar clouds -> {self.d}", flush=True)


class AsyncGrabber:
    """Camera fetches on a thread so the motion loop keeps streaming velocity commands
    (a blocking fetch inside the loop starves the 10 Hz watchdog and the dog pauses)."""
    def __init__(self, cam):
        self.cam = cam
        self.threads = []
        self.slots = []

    def start(self):
        import threading
        slot = [None]
        self.slots.append(slot)
        t = threading.Thread(target=lambda: slot.__setitem__(0, self.cam.grab()), daemon=True)
        t.start()
        self.threads.append(t)

    def collect(self):
        for t in self.threads:
            t.join(timeout=3.0)
        out = [s[0] for s in self.slots if s[0] is not None]
        self.threads, self.slots = [], []
        return out


class Camera:
    """RealSense D435i served by rs_server.py over HTTP (the default), the Go2 front
    fisheye via VideoClient (--fisheye), or a video file for dry runs (--fake-cam).

    The RealSense D435i colour stream is 1280x720 at 69 deg HFOV, narrower than the
    79 deg / 4:3 training camera, so there is nothing to crop away. --rs-mode picks
    how the 16:9 frame is fitted to the 640x480 the model was trained on:
      stretch  full FOV, resized to 640x480 (mild vertical stretch)  [default]
      pad      full FOV, resized to 640x360 (the processor pads to square)
      crop     centre 4:3 crop then 640x480 (true aspect, only 55 deg HFOV)
    """
    def __init__(self, fake_path, src_hfov, cam_url=None, rs_mode="stretch", bind_ip=None):
        self.src_hfov = src_hfov
        self.reader = None
        self.video = None
        self.cam_url = cam_url
        self.rs_mode = rs_mode
        self.bind_ip = bind_ip
        self.last_b64 = None                   # last frame that arrived, reused when one fetch fails
        self.misses = 0
        self.max_misses = 10                   # consecutive misses before the run really is blind
        if cam_url:
            from urllib.parse import urlparse
            u = urlparse(cam_url)
            self.cam_host, self.cam_port, self.cam_path = u.hostname, u.port or 80, u.path or "/frame.jpg"
            print(f"[camera] realsense at {self.cam_host}:{self.cam_port}"
                  + (f" via {bind_ip}" if bind_ip else ""), flush=True)
        elif fake_path:
            import imageio.v2 as imageio
            self.reader = imageio.get_reader(fake_path)
            self.it = iter(self.reader)
            self.fps = self.reader.get_meta_data().get("fps", 30)
        else:
            from unitree_sdk2py.go2.video.video_client import VideoClient
            self.video = VideoClient()
            self.video.SetTimeout(3.0)
            self.video.Init()
            # GetImageSample over DDS takes seconds on some dogs, so poll it in the background
            # and let grab() take the latest sample instead of waiting for a fresh request
            import threading
            self._latest = None            # (time, jpeg bytes)
            self._period = None
            self._lock = threading.Lock()
            threading.Thread(target=self._poll_fisheye, daemon=True).start()

    def _poll_fisheye(self):
        last = None
        while True:
            try:
                t0 = time.time(); code, data = self.video.GetImageSample(); dt = time.time() - t0
                if code == 0 and data:
                    with self._lock:
                        self._latest = (time.time(), bytes(data))
                    if last is None:
                        print(f"[camera] fisheye first frame in {dt:.2f} s", flush=True)
                    self._period = dt
                    last = time.time()
                else:
                    time.sleep(0.2)
            except Exception:
                time.sleep(0.5)

    def fisheye_latest(self, max_age=1.5, wait=4.0):
        t0 = time.time()
        while time.time() - t0 < wait:
            with self._lock:
                s = self._latest
            if s is not None and time.time() - s[0] <= max_age:
                return s[1]
            time.sleep(0.05)
        with self._lock:
            return self._latest[1] if self._latest else None

    def grab_jpeg(self):
        """raw JPEG bytes from the RealSense server (None in fisheye/fake modes)."""
        if not self.cam_url:
            return None
        try:
            import http.client
            src = (self.bind_ip, 0) if self.bind_ip else None
            c = http.client.HTTPConnection(self.cam_host, self.cam_port, timeout=2.0, source_address=src)
            c.request("GET", self.cam_path)
            r = c.getresponse(); b = r.read(); c.close()
            return b if r.status == 200 else None
        except Exception:
            return None

    def grab(self, tries=4):
        if self.cam_url:
            img = None
            for i in range(tries):
                try:
                    import http.client
                    src = (self.bind_ip, 0) if self.bind_ip else None
                    c = http.client.HTTPConnection(self.cam_host, self.cam_port, timeout=4.0,
                                                   source_address=src)
                    c.request("GET", self.cam_path)
                    r = c.getresponse()
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
                    c.close()
                    if i:
                        print(f"[camera] recovered after {i + 1} tries", flush=True)
                    break
                except Exception as e:
                    print(f"[camera] realsense fetch failed ({i + 1}/{tries}):", e, flush=True)
                    time.sleep(0.3 * (i + 1))
            if img is None:
                if self.last_b64 is not None:
                    self.misses += 1
                    print(f"[camera] using the previous frame (miss {self.misses})", flush=True)
                    return self.last_b64 if self.misses <= self.max_misses else None
                return None
            self.misses = 0
            if self.rs_mode == "crop":
                img = crop_to_hfov(img, None)          # centre 4:3, then 640x480
            elif self.rs_mode == "pad":
                img = img.resize((640, 360))
            else:
                img = img.resize((640, 480))
            b = io.BytesIO()
            img.save(b, "JPEG", quality=90)
            self.last_b64 = base64.b64encode(b.getvalue()).decode()
            return self.last_b64
        if self.video is not None:
            data = self.fisheye_latest()
            if not data:
                return None
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            # dry run: skip ahead ~1 s of source video per grab
            for _ in range(max(1, int(round(self.fps)))):
                frame = next(self.it, None)
                if frame is None:
                    return None
            img = Image.fromarray(frame).convert("RGB")
        img = crop_to_hfov(img, self.src_hfov)
        b = io.BytesIO()
        img.save(b, "JPEG", quality=90)
        return base64.b64encode(b.getvalue()).decode()


def main():
    global VX, VYAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8877)
    ap.add_argument("--category", default="chair")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--style", default="budget", choices=["budget", "objectnav"])
    ap.add_argument("--iface", default="auto", help="dog-facing iface, or auto to find it by address")
    ap.add_argument("--exec-n", type=int, default=4,
                    help="primitives executed per model call (the model predicts 4); runs of the "
                         "same primitive are merged into one motion")
    ap.add_argument("--drive", default="vel", choices=["vel", "traj"],
                    help="vel: velocity commands closed on odometry; traj: TrajectoryFollow "
                         "waypoints (sport API 1018), then confirmed on odometry")
    ap.add_argument("--frames-every", type=int, default=2,
                    help="fetch an observation every N primitives of motion (fetches run on a thread, "
                         "so 1 is also fine; 2 = chunk-2 recipe)")
    ap.add_argument("--src-hfov", type=float, default=110.0,
                    help="camera HFOV. 110 measured best on the Go2 front fisheye: at 110 "
                         "the model discriminates chair-vs-toilet on real frames; at 130/150 "
                         "it emits the same actions regardless of target")
    ap.add_argument("--fake-cam", default=None, help="video file instead of the robot camera")
    ap.add_argument("--cam", default="http://192.168.123.18:8090/frame.jpg",
                    help="RealSense frame URL served by rs_server.py on the dog's expansion "
                         "computer (the default camera); --fisheye uses the Go2 front camera instead")
    ap.add_argument("--fisheye", action="store_true", help="use the Go2 front fisheye via DDS instead of the RealSense")
    ap.add_argument("--rs-mode", default="stretch", choices=["stretch", "pad", "crop"],
                    help="how the 16:9 RealSense frame is fitted to the 4:3 training input")
    ap.add_argument("--no-move", action="store_true", help="never command motion")
    ap.add_argument("--avoid", action="store_true",
                    help="drive through the obstacle-avoidance service instead of raw SportClient "
                         "(off by default: it vetoes and slows the model's moves)")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--record", default="", help="directory: record odometry, camera stream, lidar and decisions for make_run_video.py")
    ap.add_argument("--no-lidar", action="store_true", help="do not record the lidar cloud")
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the lidar obstacle guard (on by default: forwards are cut when the dog's lidar "
                         "shows an obstacle closer than --guard-stop ahead of the nose)")
    ap.add_argument("--guard-stop", type=float, default=0.15, help="lidar guard: stop distance ahead of the nose, m")
    ap.add_argument("--guard-slow", type=float, default=0.8, help="lidar guard: half speed when free space ahead is under this, m")
    ap.add_argument("--deadman", default="192.168.123.18:8091",
                    help="host:port of deadman.py on the Jetson; '' disables the heartbeat")
    ap.add_argument("--no-lookahead", action="store_true",
                    help="stop at every chunk end to grab the frame and wait for the model (default: the frame is grabbed and the model asked just before the last motion ends, so chunks flow into each other)")
    ap.add_argument("--vx", type=float, default=VX, help="walking speed m/s (0.6)")
    ap.add_argument("--vyaw", type=float, default=VYAW, help="turning speed rad/s (1.0)")
    ap.add_argument("--no-nudge", action="store_true", help="no 30-degree turn when a forward is blocked (the model is asked again from the same view)")
    ap.add_argument("--no-blend", action="store_true",
                    help="halt between every segment of a chunk (default: a turn flows into the next forward)")
    ap.add_argument("--timed", action="store_true",
                    help="open-loop timed velocities instead of closing each primitive on odometry")
    a = ap.parse_args()

    srv = Server(a.host, a.port)
    use_rs = not (a.fisheye or a.fake_cam)
    moving = not (a.no_move or a.fake_cam)
    bind_ip = None
    if not a.fake_cam:
        iface = resolve_iface(a.iface)
        bind_ip = iface_ip(iface)
        print(f"[client] dog link: {iface} ({bind_ip})", flush=True)
        if moving or not use_rs:
            init_dds(iface)        # DDS is only needed for the fisheye or for driving
    cam = Camera(a.fake_cam, a.src_hfov, cam_url=a.cam if use_rs else None, rs_mode=a.rs_mode,
                 bind_ip=bind_ip)
    a.lookahead = not a.no_lookahead
    VX, VYAW = a.vx, a.vyaw
    bot = Robot(enabled=moving, avoid=a.avoid, odom=not a.timed, drive=a.drive, blend=not a.no_blend, nudge=not a.no_nudge)
    hb = Heartbeat(a.deadman, bind_ip=bind_ip) if (a.deadman and moving) else None
    if moving and not a.no_guard and bot.odom is not None:
        bot.guard = LidarGuard(stop_m=a.guard_stop, slow_m=a.guard_slow)
    rec = Recorder(a.record, bot.odom, cam, lidar=not a.no_lidar) if a.record else None
    if rec is not None:
        bot.on_segment = rec.segment
    if bot.odom is not None:
        x, y, yaw = bot.odom.pose()
        print(f"[robot] odometry frame: x,y in metres from power-on origin, yaw {np.degrees(yaw):.1f} deg CCW from the origin heading; forward = (cos yaw, sin yaw)", flush=True)

    import os
    import signal
    import threading

    _ints = {"n": 0}

    def _sigint(signum, frame):
        # Ctrl-C must not depend on an SDK round trip: cutting the heartbeat makes the
        # Jetson watchdog halt the dog within half a second, whatever the SDK is doing.
        # The guarded stop then runs off the main thread, and a second Ctrl-C exits hard.
        _ints["n"] += 1
        bot.abort = True
        if hb is not None:
            hb.stop()
        if _ints["n"] >= 2:
            print("\n[client] second interrupt: exiting now", flush=True)
            os._exit(1)
        print("\n[client] interrupt: halting (heartbeat cut, watchdog will stop the dog)", flush=True)
        threading.Thread(target=bot.stop, daemon=True).start()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint)

    hello = srv.rpc({"cmd": "start", "category": a.category,
                     "budget": a.budget, "style": a.style})
    print(f"[client] {hello.get('instruction')}", flush=True)

    # Frame/action interleaving must match the simulator: one observation is
    # captured BEFORE every primitive, and a decision consumes the frames
    # accumulated since the last one. Grabbing N frames from a standstill and
    # then executing N actions (the obvious wrong way) hands the model N copies
    # of the same pose and no motion history.
    boot = cam.grab()
    if boot is None:
        print("[client] camera failed on first grab", flush=True)
        return
    import threading

    def act(frames, executed):
        pose = None
        if bot.odom is not None and bot.odom.alive():
            x, y, yaw = bot.odom.pose()
            pose = [round(x, 3), round(y, 3), round(float(np.degrees(yaw)), 1)]
        return srv.rpc({"cmd": "act", "frames": frames, "executed": executed, "pose": pose})

    r = act([boot], 0)
    try:
        for k in range(a.max_decisions):
            if r is None:
                print(f"[client] no observation available after {cam.misses} consecutive camera misses; stopping", flush=True)
                return
            if "error" in r and r["error"]:
                print("[client] server error:", r["error"], flush=True)
                return
            acts = r["actions"]
            print(f"[{k:03d}] N={r['N']:>4} acts={acts} ({r['latency_s']}s) :: {r['raw'][:50]}",
                  flush=True)
            if rec is not None:
                _p = None
                if bot.odom is not None and bot.odom.alive():
                    _x, _y, _yaw = bot.odom.pose(); _p = [round(_x, 3), round(_y, 3), round(float(_yaw), 4)]
                rec.decision(k, r["N"], acts, r["latency_s"], r["raw"], _p)
            chunk = []
            for act_ in acts[:a.exec_n]:
                if act_ == "stop":
                    break
                chunk.append(act_)
            stop_after = len(chunk) < len(acts[:a.exec_n])
            if not chunk:
                print("[client] model called STOP", flush=True)
                return

            grabber = AsyncGrabber(cam)
            # Pipelined decision: near the end of the chunk's last motion the chunk-end frame
            # is grabbed and the model is asked for the next chunk while the dog is still
            # finishing the current one, so the next chunk starts without a stop. The frame
            # is LOOKAHEAD_M / LOOKAHEAD_DEG early; `executed` is the planned count.
            nxt = {}
            use_la = a.lookahead and not stop_after and r["N"] > 0

            def fire(n_exec):
                bf = grabber.collect()
                f = cam.grab()
                fr = bf + ([f] if f is not None else [])
                nxt["r"] = act(fr, n_exec) if fr else None

            def fire_async(n_exec):
                t = threading.Thread(target=fire, args=(n_exec,), daemon=True); t.start(); nxt["t"] = t

            executed = bot.run_chunk(chunk, on_boundary=grabber.start, every=a.frames_every,
                                     on_lookahead=fire_async if use_la else None,
                                     final_halt=not use_la)
            if stop_after:
                bot.stop()
                print("[client] model called STOP", flush=True)
                return
            # NB: a 'stop' in the tail of the prediction (beyond exec_n) is a
            # forecast -- "two more steps and you're there" -- not a command.
            # Acting on it ends the episode a step early; only a stop inside the
            # executed slice (handled above) or an exhausted budget stops us.
            if r["N"] <= 0:
                bot.stop()
                print("[client] budget exhausted", flush=True)
                return
            if use_la:
                t = nxt.get("t")
                if t is not None:
                    bot.coast(t.is_alive, max_s=0.4)   # keep moving briefly, never unattended
                    t.join(timeout=15.0)
                r = nxt.get("r")
                if r is None:                       # lookahead never fired or the fetch failed: think standing still
                    bot._halt()
                    grabber.start()
                    frames = grabber.collect()
                    r = act(frames, executed) if frames else None
            else:
                grabber.start()                     # observation at the final pose (chunk end)
                frames = grabber.collect()
                r = act(frames, executed) if frames else None
    except KeyboardInterrupt:
        print("\n[client] interrupted", flush=True)
    finally:
        bot.stop()
        if hb is not None:
            hb.stop()                       # the Jetson halts the dog again half a second later
        if rec is not None:
            rec.stop({"category": a.category, "budget": a.budget, "port": a.port})
        print("[client] trace:", srv.rpc({"cmd": "end"}).get("trace"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
