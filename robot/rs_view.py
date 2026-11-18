"""Live view of the RealSense frame server while the dog runs.

    python rs_view.py                      # 192.168.123.18:8090 via the .99 adapter
    python rs_view.py --src 192.168.123.99 --url http://192.168.123.18:8090/frame.jpg

Polls /frame.jpg about ten times a second and shows it in a window; the title bar
carries the frame age reported by the server. Binds to the wired adapter so the
request does not go out over a Wi-Fi that shares the 192.168.123.x range.
Close the window or press q to quit.
"""
import argparse
import http.client
import io
import time
import tkinter as tk
from urllib.parse import urlparse

from PIL import Image, ImageTk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.123.18:8090/frame.jpg")
    ap.add_argument("--src", default="192.168.123.99", help="local address to bind; '' for none")
    ap.add_argument("--scale", type=float, default=0.5, help="window scale of the 1280x720 frame")
    a = ap.parse_args()
    u = urlparse(a.url)
    src = (a.src, 0) if a.src else None

    root = tk.Tk()
    root.title("RealSense")
    label = tk.Label(root)
    label.pack()
    state = {"photo": None, "n": 0, "t0": time.time()}

    def grab():
        try:
            c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=2, source_address=src)
            c.request("GET", u.path)
            r = c.getresponse()
            age = r.getheader("X-Frame-Age", "?")
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
            c.close()
        except Exception as e:
            root.title(f"RealSense: no frame ({e})")
            root.after(500, grab)
            return
        if a.scale != 1.0:
            img = img.resize((int(img.width * a.scale), int(img.height * a.scale)))
        state["photo"] = ImageTk.PhotoImage(img)
        label.configure(image=state["photo"])
        state["n"] += 1
        fps = state["n"] / max(1e-3, time.time() - state["t0"])
        root.title(f"RealSense  {img.width}x{img.height}  age {age}s  {fps:.1f} fps")
        root.after(80, grab)

    root.bind("q", lambda e: root.destroy())
    root.after(0, grab)
    root.mainloop()


if __name__ == "__main__":
    main()
