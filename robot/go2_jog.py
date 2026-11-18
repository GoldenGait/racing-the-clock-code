"""Manually position the Go2: one-shot moves, or interactive keyboard driving.

One-shot:
  python go2_jog.py --iface en8 forward 1.0      walk forward 1.0 m
  python go2_jog.py --iface en8 back 0.5         walk backward 0.5 m
  python go2_jog.py --iface en8 left 90          turn left 90 degrees
  python go2_jog.py --iface en8 right 45         turn right 45 degrees
  python go2_jog.py --iface en8 strafe_left 0.5  slide left 0.5 m (no turn)
  python go2_jog.py --iface en8 strafe_right 0.5 slide right 0.5 m
  python go2_jog.py --iface en8 stand            stand up / balance stand
  python go2_jog.py --iface en8 sit              sit down
  python go2_jog.py --iface en8 damp             relax joints

Interactive (best for positioning before a run):
  python go2_jog.py --iface en8 drive
      w/s   forward / back        a/d   strafe left / right
      q/e   turn left / right     space stop
      1/2   stand / sit           x     quit
  Each keypress moves one step (--step m, --turn deg).

Velocities match the model-driven client, so 'forward' here covers the same
ground as one 'forward' primitive during a run.
"""
import argparse
import math
import sys
import termios
import time
import tty

VX = 0.5          # m/s
VY = 0.4          # m/s lateral
VYAW = 0.8        # rad/s
CMD_HZ = 10


def drive(mover, sport, vx, vy, vyaw, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        mover.Move(vx, vy, vyaw)
        time.sleep(1.0 / CMD_HZ)
    if mover is not sport:
        mover.Move(0.0, 0.0, 0.0)
    else:
        sport.StopMove()


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["left", "right", "forward", "back",
                                       "strafe_left", "strafe_right",
                                       "stand", "sit", "damp", "drive"])
    ap.add_argument("amount", nargs="?", type=float, default=None)
    ap.add_argument("--iface", default="en8")
    ap.add_argument("--avoid", action="store_true")
    ap.add_argument("--step", type=float, default=0.25, help="drive mode: metres per key")
    ap.add_argument("--turn", type=float, default=30.0, help="drive mode: degrees per key")
    a = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    ChannelFactoryInitialize(0, a.iface)
    sport = SportClient(); sport.SetTimeout(10.0); sport.Init()

    if a.action == "sit":
        print("StandDown ..."); sport.StandDown(); time.sleep(2); return 0
    if a.action == "damp":
        print("Damp ..."); sport.Damp(); return 0
    if a.action == "stand":
        print("StandUp ..."); sport.StandUp(); time.sleep(2)
        print("BalanceStand ..."); sport.BalanceStand(); time.sleep(1); return 0

    sport.StandUp(); time.sleep(2)
    sport.BalanceStand(); time.sleep(1)

    mover = sport
    if a.avoid:
        from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import \
            ObstaclesAvoidClient
        av = ObstaclesAvoidClient(); av.SetTimeout(3.0); av.Init()
        for _ in range(20):
            if av.SwitchGet()[1]:
                break
            av.SwitchSet(True); time.sleep(0.1)
        av.UseRemoteCommandFromApi(True); time.sleep(0.5)
        mover = av; print("obstacle avoidance ON")

    try:
        if a.action == "drive":
            print(__doc__.split("Interactive")[1].split("Velocities")[0])
            print("driving. x to quit.")
            keymap = {
                "w": (VX, 0, 0, a.step / VX),
                "s": (-VX, 0, 0, a.step / VX),
                "a": (0, VY, 0, a.step / VY),
                "d": (0, -VY, 0, a.step / VY),
                "q": (0, 0, VYAW, math.radians(a.turn) / VYAW),
                "e": (0, 0, -VYAW, math.radians(a.turn) / VYAW),
            }
            while True:
                k = getkey().lower()
                if k in ("x", "\x03"):
                    break
                if k == "1":
                    sport.StandUp(); time.sleep(2); sport.BalanceStand(); continue
                if k == "2":
                    sport.StandDown(); time.sleep(2); continue
                if k == " ":
                    sport.StopMove(); print("stop"); continue
                if k in keymap:
                    vx, vy, vyaw, secs = keymap[k]
                    print(f"{k} ", end="", flush=True)
                    drive(mover, sport, vx, vy, vyaw, secs)
            return 0

        if a.amount is None:
            sys.exit("need an amount (metres for moves, degrees for turns)")
        if a.action in ("left", "right"):
            secs = math.radians(a.amount) / VYAW
            vx, vy, vyaw = 0.0, 0.0, (VYAW if a.action == "left" else -VYAW)
            print(f"{a.action} {a.amount} deg ({secs:.1f}s)")
        elif a.action in ("strafe_left", "strafe_right"):
            secs = a.amount / VY
            vx, vy, vyaw = 0.0, (VY if a.action == "strafe_left" else -VY), 0.0
            print(f"{a.action} {a.amount} m ({secs:.1f}s)")
        else:
            secs = a.amount / VX
            vx, vy, vyaw = (VX if a.action == "forward" else -VX), 0.0, 0.0
            print(f"{a.action} {a.amount} m ({secs:.1f}s)")
        drive(mover, sport, vx, vy, vyaw, secs)
    finally:
        if mover is not sport:
            mover.Move(0.0, 0.0, 0.0); mover.UseRemoteCommandFromApi(False)
        sport.StopMove()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
