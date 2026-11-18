"""Probe the RealSense on the Go2's expansion computer without librealsense.

The D435i exposes its colour sensor as a plain UVC node, so OpenCV can read it.
Grabs one frame from every /dev/video* node, saves the ones that work, and
prints what each node returned, so the colour node can be picked by eye.

    scp robot/rs_probe.py unitree@192.168.123.18:/tmp/
    ssh unitree@192.168.123.18 'python3 /tmp/rs_probe.py'
    scp "unitree@192.168.123.18:/tmp/rs_video*.jpg" ~/Downloads/
"""
import glob
import cv2

for dev in sorted(glob.glob("/dev/video*")):
    i = int(dev.replace("/dev/video", ""))
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(dev, "closed")
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, f = cap.read()
    print(dev, "ok" if ok else "no frame", None if f is None else f.shape)
    if ok:
        cv2.imwrite("/tmp/rs_video%d.jpg" % i, f)
    cap.release()
