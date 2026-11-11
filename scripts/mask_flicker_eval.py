"""Slice-to-slice boundary flicker of lung masks on real and generated CT volumes.

Segments the lungs with lungmask (R231, which works slice by slice, so it does
not smooth across z) and measures how much the mask boundary jumps between
neighbouring axial slices. Generated volumes are produced slice-wise along z,
so flicker in the image shows up as flicker in the mask.

Per volume and mask:
  pop_rate      : voxels where slice z disagrees with both z-1 and z+1 while
                  those two agree (one-slice pops), per 1000 mask voxels.
  area_jerk     : mean |A(z+1) - 2 A(z) + A(z-1)| / mean A over slices with lung,
                  in percent (second difference of the per-slice area profile).
  boundary_step : mean |m_z XOR m_{z+1}| / mean perimeter, in voxels; the average
                  in-plane boundary displacement between neighbouring slices.

Reuses the loaders of lung_volume_eval.py.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lung_volume_eval import load_gt, load_pred_hu, to_axial  # noqa: E402


def perimeter(m2d: np.ndarray) -> int:
    return int((m2d & ~ndi.binary_erosion(m2d)).sum())


def flicker_metrics(mask_zyx: np.ndarray) -> dict:
    m = mask_zyx.astype(bool)
    n = max(int(m.sum()), 1)
    prev, mid, nxt = m[:-2], m[1:-1], m[2:]
    pops = int(((prev == nxt) & (mid != prev)).sum())
    area = m.reshape(m.shape[0], -1).sum(1).astype(np.float64)
    has = area > 0
    jerk = np.abs(area[2:] - 2 * area[1:-1] + area[:-2])
    valid = has[1:-1]
    xor = (m[1:] ^ m[:-1]).reshape(m.shape[0] - 1, -1).sum(1)
    per = np.array([perimeter(s) for s in m], dtype=np.float64)
    pair_per = 0.5 * (per[1:] + per[:-1])
    pv = pair_per > 0
    return {
        "pop_rate": 1000.0 * pops / n,
        "area_jerk_pct": 100.0 * jerk[valid].mean() / area[has].mean() if valid.any() else np.nan,
        "boundary_step_vox": float((xor[pv] / pair_per[pv]).mean()) if pv.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--gt_dir", default="~/data/CT/LIDC-IDRI/CT")
    ap.add_argument("--method", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--hu_min", type=float, default=-1350.0)
    ap.add_argument("--hu_max", type=float, default=150.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lungmask import LMInferer
    inferer = LMInferer(modelname="R231", batch_size=args.batch_size)

    lines = [l.strip() for l in open(os.path.expanduser(args.keys)) if l.strip()]
    keys = [l for l in lines if l.lower() != "keys"]
    si, sn = map(int, args.shard.split("/"))
    keys = keys[si::sn]
    gt_dir = Path(os.path.expanduser(args.gt_dir))
    methods = dict((m.split("=", 1)[0], Path(os.path.expanduser(m.split("=", 1)[1]))) for m in args.method)

    out = Path(os.path.expanduser(args.out))
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["key"], r["method"]) for r in csv.DictReader(fh)}
    new_file = not out.exists()
    fields = ["key", "method", "mask", "pop_rate", "area_jerk_pct", "boundary_step_vox"]
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            w.writeheader()

        def emit(key, name, hu_xyz):
            seg = inferer.apply(to_axial(hu_xyz))  # (z, y, x), 1 right, 2 left
            for mask_name, m in [("lung", seg > 0), ("right", seg == 1), ("left", seg == 2)]:
                w.writerow({"key": key, "method": name, "mask": mask_name, **flicker_metrics(m)})
            fh.flush()

        for i, key in enumerate(keys):
            todo = [n for n in methods if (key, n) not in done and (methods[n] / f"{key}.npy").exists()]
            if (key, "gt") not in done:
                raw, _ = load_gt(gt_dir, key)
                emit(key, "gt", np.clip(raw, args.hu_min, args.hu_max))
            for name in todo:
                emit(key, name, load_pred_hu(methods[name] / f"{key}.npy", args.hu_min, args.hu_max))
            print(f"[{i + 1}/{len(keys)}] {key}", flush=True)


if __name__ == "__main__":
    main()
