"""Go2 EDU connectivity checks -- run these before go2_client.py ever moves the dog.

  python go2_check.py state  --iface eth0     # DDS link + robot mode      (no motion)
  python go2_check.py camera --iface eth0     # grab one frame -> frame.jpg (no motion)
  python go2_check.py stand  --iface eth0     # stand up, wait 3s, sit     (MOTION!)

'state' and 'camera' are safe with the dog on the floor or a stand. Run 'stand'
only with clear space around the robot and the app's stop button in reach.
"""
import argparse
import io
import sys
import time




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
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    cur = None
    for line in out.splitlines():
        m = re.match(r"^([A-Za-z0-9]+):", line)
        if m:
            cur = m.group(1)
        elif "inet 192.168.123." in line and cur:
            return cur
    raise SystemExit("no interface holds a 192.168.123.x address. "
                     "Plug the adapter in and run: sudo ./go2_link.sh")


def init(iface):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    iface = resolve_iface(iface)
    print(f"  DDS on {iface}")
    if iface in ("", "auto", "none"):
        # let cyclonedds pick; /tmp/cdds.LOG then lists every interface it can see
        print("  [init] autodetermine mode (no interface pinned)")
        ChannelFactoryInitialize(0)
    else:
        ChannelFactoryInitialize(0, iface)


def cmd_state(a):
    """Subscribe to several state topics: proves DDS actually flows, and says which.

    DDS participant discovery can take longer than a couple of seconds, and the
    exact topic varies by model/firmware, so wait properly and try all of them.
    """
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
    got = {}

    topics = [("rt/sportmodestate", SportModeState_),
              ("rt/lf/sportmodestate", SportModeState_),
              ("rt/lowstate", LowState_)]
    subs = []
    for name, typ in topics:
        try:
            s = ChannelSubscriber(name, typ)
            s.Init(lambda m, n=name: got.setdefault(n, m), 10)
            subs.append(name)
            print(f"  subscribed: {name}")
        except Exception as e:
            print(f"  FAILED to subscribe {name}: {e}")

    print(f"  waiting up to {a.wait}s for messages ...")
    for i in range(int(a.wait * 10)):
        if got:
            name = next(iter(got))
            m = got[name]
            print(f"DDS OK  (topic: {name})")
            for attr in ("mode", "body_height", "position", "velocity"):
                if hasattr(m, attr):
                    v = getattr(m, attr)
                    v = [round(x, 2) for x in v] if hasattr(v, "__len__") else v
                    print(f"  {attr:<12}: {v}")
            return 0
        time.sleep(0.1)
    print(f"NO DDS MESSAGES in {a.wait}s on topics {subs}")
    print("The robot IS publishing (tcpdump proved it), so this is a cyclonedds")
    print("interface/multicast binding problem, not a link problem. Try:")
    print('  export CYCLONEDDS_URI=\'<CycloneDDS><Domain><General>'
          '<NetworkInterfaceAddress>192.168.123.99</NetworkInterfaceAddress>'
          "<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'")
    print("then re-run. (Substitute this machine's own IP on the robot subnet.)")
    return 1


def cmd_camera(a):
    """Grab one frame from the front camera and save it for FOV calibration."""
    from unitree_sdk2py.go2.video.video_client import VideoClient
    v = VideoClient()
    v.SetTimeout(3.0)
    v.Init()
    code, data = v.GetImageSample()
    if code != 0:
        print("camera error code", code)
        return 1
    from PIL import Image
    img = Image.open(io.BytesIO(bytes(data)))
    img.save(a.out)
    print(f"saved {a.out}  size={img.size}")
    print("scp this back and we calibrate --src-hfov from it")
    return 0


def cmd_stand(a):
    """Minimal motion test: stand, pause, sit. Nothing walks."""
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    s = SportClient()
    s.SetTimeout(10.0)
    s.Init()
    print("StandUp ...")
    s.StandUp()
    time.sleep(3)
    print("BalanceStand (walk-ready) ...")
    s.BalanceStand()
    time.sleep(2)
    print("StandDown ...")
    s.StandDown()
    time.sleep(2)
    print("ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["state", "camera", "stand"])
    ap.add_argument("--iface", default="auto")
    ap.add_argument("--out", default="frame.jpg")
    ap.add_argument("--wait", type=float, default=30.0, help="seconds to wait for DDS")
    a = ap.parse_args()
    init(a.iface)
    return {"state": cmd_state, "camera": cmd_camera, "stand": cmd_stand}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
