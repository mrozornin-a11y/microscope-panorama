"""Compare frame_positions.csv with the synthetic ground truth."""
import csv
import sys

import numpy as np


def evaluate(pos_csv, gt_csv):
    gt = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(gt_csv))}
    est, ref = [], []
    for r in csv.DictReader(open(pos_csv)):
        if r["status"] != "used":
            continue
        est.append((float(r["x"]), float(r["y"])))
        ref.append(gt[int(r["frame"])])
    est, ref = np.array(est), np.array(ref)
    # best rigid alignment (Kabsch) of estimate onto ground truth
    me, mr = est.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((est - me).T @ (ref - mr))
    R = (u @ vt).T
    al = (est - me) @ R.T + mr
    err = np.linalg.norm(al - ref, axis=1)
    return len(est), float(np.sqrt((err ** 2).mean())), float(err.max())


if __name__ == "__main__":
    n, rms, mx = evaluate(sys.argv[1], sys.argv[2])
    print(f"{n} frames: RMS error {rms:.2f} px, max {mx:.2f} px")
