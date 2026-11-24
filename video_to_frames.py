"""Pick sharp, evenly spaced frames from a (slow-motion) video for COLMAP.

Usage: python video_to_frames.py VIDEO data/<scene>/input [--num 120] [--max_width 1600]
Splits the video into --num equal segments and keeps the sharpest frame in each,
so blurry frames and near-duplicates are skipped.
"""
import argparse
import os

import cv2

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("out_dir")
p.add_argument("--num", type=int, default=120, help="number of frames to keep")
p.add_argument("--max_width", type=int, default=1600)
a = p.parse_args()

cap = cv2.VideoCapture(a.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
assert total > 0, f"cannot read {a.video}"
print(f"{total} frames at {fps:.1f} fps ({total / fps:.1f}s); keeping {a.num}")
os.makedirs(a.out_dir, exist_ok=True)

bounds = [round(i * total / a.num) for i in range(a.num + 1)]
kept = 0
for s, e in zip(bounds[:-1], bounds[1:]):
    best, best_score = None, -1.0
    step = max(1, (e - s) // 6)  # sample up to ~6 candidates per segment
    for idx in range(s, max(e, s + 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if score > best_score:
            best, best_score = frame, score
    if best is None:
        continue
    h, w = best.shape[:2]
    if w > a.max_width:
        best = cv2.resize(best, (a.max_width, round(h * a.max_width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.out_dir, f"{kept:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
    kept += 1
print(f"wrote {kept} frames to {a.out_dir}")
