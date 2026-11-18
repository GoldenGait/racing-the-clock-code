"""Dead-man's switch for the Go2, run ON THE JETSON (the dog's own network, never lost).

The Go2 holds the last velocity command until told otherwise, so when the laptop's cable
drops or the client dies mid-walk the dog keeps walking (seen 2026-09-14). This process
listens for a UDP heartbeat from the client; once armed, if the heartbeat stops for
TIMEOUT_S it issues StopMove (and zeroes / releases the obstacle-avoidance service),
then re-arms on the next heartbeat.

    scp robot/deadman.py unitree@192.168.123.18:~/
    ssh unitree@192.168.123.18 'nohup python3 ~/deadman.py > ~/deadman.log 2>&1 &'
    (the client sends the heartbeat itself: go2_client.py --deadman 192.168.123.18:8091)

Any UDP datagram to :8091 is a heartbeat; the payload is ignored.
"""
import argparse
import os
import socket
import subprocess
import sys
import time

TIMEOUT_S = 0.5


def dog_iface():
    """interface carrying the dog-network address (eth0 on the stock expansion dock)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "192.168.123." in line:
                return line.split()[1]
    except Exception:
        pass
    return "eth0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--halt-once", action="store_true", help="send the stop sequence once and exit (helper process)")
    a = ap.parse_args()
    iface = a.iface or dog_iface()
    if a.halt_once:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(0, iface)
        sp = SportClient(); sp.SetTimeout(2.0); sp.Init()
        av = None
        try:
            from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
            av = ObstaclesAvoidClient(); av.SetTimeout(2.0); av.Init()
        except Exception:
            pass
        for _ in range(3):
            if av is not None:
                try: av.Move(0.0, 0.0, 0.0)
                except Exception as e: print("  helper avoid.Move(0) failed:", e, flush=True)
            try: sp.StopMove()
            except Exception as e: print("  helper StopMove failed:", e, flush=True)
            time.sleep(0.05)
        if av is not None:
            try: av.UseRemoteCommandFromApi(False)
            except Exception: pass
        print(f"{time.strftime('%H:%M:%S')} deadman: halt sent (helper)", flush=True)
        return 0

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    ChannelFactoryInitialize(0, iface)
    sport = SportClient(); sport.SetTimeout(2.0); sport.Init()
    avoid = None
    try:
        from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
        avoid = ObstaclesAvoidClient(); avoid.SetTimeout(2.0); avoid.Init()
    except Exception as e:
        print("deadman: no obstacle-avoid client:", e, flush=True)

    def halt(reason):
        """Two independent paths so the stop goes out even if one of them crashes:
        1. a fresh helper process (own DDS participant) sends the full sequence;
        2. this process tries the same calls directly (a segfault here only restarts the
           watchdog via its wrapper loop; the helper is already on its way)."""
        print(f"{time.strftime('%H:%M:%S')} deadman: HALT ({reason})", flush=True)
        try:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--halt-once", "--iface", iface])
        except Exception as e:
            print("  helper spawn failed:", e, flush=True)
        for _ in range(3):
            if avoid is not None:
                try:
                    avoid.Move(0.0, 0.0, 0.0)
                except Exception as e:
                    print("  avoid.Move(0) failed:", e, flush=True)
            try:
                sport.StopMove()
            except Exception as e:
                print("  StopMove failed:", e, flush=True)
            time.sleep(0.05)
        if avoid is not None:
            try:
                avoid.UseRemoteCommandFromApi(False)
            except Exception:
                pass
        print(f"{time.strftime('%H:%M:%S')} deadman: halt sent (direct)", flush=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", a.port))
    s.settimeout(0.1)
    print(f"deadman: on {iface}, listening udp :{a.port}, timeout {a.timeout}s", flush=True)
    armed, last, src = False, 0.0, None
    while True:
        try:
            _, addr = s.recvfrom(64)
            if not armed:
                print(f"{time.strftime('%H:%M:%S')} deadman: armed by {addr[0]}", flush=True)
            armed, last, src = True, time.time(), addr[0]
        except socket.timeout:
            pass
        if armed and time.time() - last > a.timeout:
            halt(f"no heartbeat from {src} for {time.time() - last:.2f}s")
            armed = False


if __name__ == "__main__":
    sys.exit(main())
