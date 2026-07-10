#!/usr/bin/env python3
"""AR0234_driver.py - Open and stream from an AR0234 camera on Ubuntu."""

import cv2
import sys


class AR0234Driver:
    def __init__(self, device=0, width=1920, height=1200, fps=30, fourcc="MJPG"):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap = None

    def open(self):
        # Use V4L2 backend explicitly on Linux
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at device {self.device}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {aw}x{ah} @ {afps:.1f} fps")
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        return self.cap.read()

    def stream(self):
        print("Press 'q' to quit, 's' to save a frame.")
        count = 0
        while True:
            ok, frame = self.read()
            if not ok:
                print("Frame grab failed")
                break
            cv2.imshow("AR0234", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fn = f"frame_{count:04d}.png"
                cv2.imwrite(fn, frame)
                print(f"Saved {fn}")
                count += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # device = 24
    cam = AR0234Driver(device=device)
    try:
        cam.open()
        cam.stream()
    finally:
        cam.close()


if __name__ == "__main__":
    main()