"""Thin wrapper around cv2.VideoCapture."""

from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


class VideoSource:
    def __init__(self, path: str):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise VideoError(f"Cannot open video: {self.path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if not np.isfinite(self.fps) or self.fps <= 0 or self.fps > 1000:
            self.fps = 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise VideoError(f"Cannot decode frames from: {self.path}")
        self.height, self.width = frame.shape[:2]
        self._first = frame

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.frame_count > 0 else 0.0

    def first_frame(self) -> np.ndarray:
        return self._first.copy()

    def sample_frames(self, n: int = 15) -> list[np.ndarray]:
        """Approximately evenly spaced frames (seeking may be inexact)."""
        cap = cv2.VideoCapture(self.path)
        frames = []
        total = max(self.frame_count, 1)
        for k in range(n):
            idx = int(k * (total - 1) / max(n - 1, 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok and fr is not None and fr.shape[:2] == (self.height, self.width):
                frames.append(fr)
        cap.release()
        if not frames:
            frames = [self.first_frame()]
        return frames

    def iterate(self, step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """Sequentially decode the video, yielding every `step`-th frame.

        Sequential decoding (grab/retrieve) is used instead of seeking because
        seeking in long-GOP iPhone videos is not frame accurate.
        """
        step = max(1, int(step))
        cap = cv2.VideoCapture(self.path)
        idx = 0
        try:
            while True:
                if not cap.grab():
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        yield idx, frame
                idx += 1
        finally:
            cap.release()
