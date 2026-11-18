"""Patch unitree_sdk2py's cyclonedds config for macOS: unicast peer discovery.

macOS frequently refuses to join the DDS multicast group (239.255.0.1), so the
SDK subscribes successfully but never receives anything -- exactly the symptom
where tcpdump shows the robot publishing and the SDK still times out.

cyclonedds can skip multicast discovery entirely and unicast straight at the
robot. This rewrites the SDK's config template to do that, keeping a backup.

  python fix_dds_macos.py                 # patch (peer 192.168.123.161)
  python fix_dds_macos.py --peer 1.2.3.4  # different robot address
  python fix_dds_macos.py --restore       # undo
"""
import argparse
import os
import shutil
import sys

TEMPLATE = '''ChannelConfigHasInterface = \'\'\'<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface __ATTR__="$__IF_NAME__$" priority="default" multicast="true"/>
                </Interfaces>
                <AllowMulticast>false</AllowMulticast>
            </General>
            <Discovery>
                <ParticipantIndex>auto</ParticipantIndex>
                <MaxAutoParticipantIndex>20</MaxAutoParticipantIndex>
                <Peers>
                    <Peer address="__PEER__"/>
                </Peers>
            </Discovery>
            <Tracing>
                <Verbosity>config</Verbosity>
                <OutputFile>/tmp/cdds.LOG</OutputFile>
            </Tracing>
        </Domain>
    </CycloneDDS>\'\'\'

ChannelConfigAutoDetermine = \'\'\'<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine="true" priority="default" multicast="true"/>
                </Interfaces>
                <AllowMulticast>false</AllowMulticast>
            </General>
            <Discovery>
                <ParticipantIndex>auto</ParticipantIndex>
                <MaxAutoParticipantIndex>20</MaxAutoParticipantIndex>
                <Peers>
                    <Peer address="__PEER__"/>
                </Peers>
            </Discovery>
        </Domain>
    </CycloneDDS>\'\'\'
'''


def find_config():
    try:
        import unitree_sdk2py.core.channel_config as cc
        return cc.__file__
    except Exception as e:
        sys.exit(f"cannot import unitree_sdk2py ({e}) -- is it installed in this python?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", default="192.168.123.161")
    ap.add_argument("--attr", default="address", choices=["address", "name"],
                    help="macOS cyclonedds rejects interface NAMES ('en20 does not "
                         "match an available interface'); use address and pass this "
                         "machine's IP as --iface instead")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()

    path = find_config()
    bak = path + ".orig"
    print("config file:", path)

    if a.restore:
        if not os.path.exists(bak):
            sys.exit("no backup found")
        shutil.copy(bak, path)
        print("restored original")
        return

    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup ->", bak)
    open(path, "w").write(TEMPLATE.replace("__PEER__", a.peer).replace("__ATTR__", a.attr))
    print(f"patched: unicast peer discovery to {a.peer}, multicast off, "
          f"interface selected by {a.attr}")
    if a.attr == "address":
        # the local address depends on which network the robot is on, so show
        # the real one rather than the wired-setup address this used to hardcode
        import socket
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.connect((a.peer, 7400)); mine = sk.getsockname()[0]; sk.close()
        except Exception:
            mine = "<this machine's IP on the robot's network>"
        print("now re-run with THIS MACHINE'S IP, not the device name:")
        print(f"  python go2_check.py state --iface {mine}")
    else:
        print("now re-run:  python go2_check.py state --iface en20")


if __name__ == "__main__":
    main()
