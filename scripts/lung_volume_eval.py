"""Anatomy segmentation on real and generated CT volumes (X-ray -> CT, LIDC-IDRI).

Runs the same segmentation on the ground-truth volume and on every generated
volume and writes one CSV row per (case, method, structure) with the structure
volume in millilitres and the Dice overlap against the ground-truth mask.

Data layout (paper setup):
  ground truth : <gt_dir>/<key>.nii.gz, HU, (x, y, z) = (256, 256, 128) at 1.5 mm
  generated    : <method_dir>/<key>.npy, float32 (z, x, y) in [-1, 1], which maps
                 linearly to the HU window [hu_min, hu_max] = [-1350, 150]

The ground truth is clipped to the same HU window before segmentation, so real
and generated volumes are treated identically.

Segmenters:
  threshold    : model free. Enclosed air (< --air_hu), large 3D components.
  lungmask     : lungmask R231 U-Net.
  totalseg     : TotalSegmentator `total` task (lung lobes, heart, aorta, ...).
  lung_vessels : TotalSegmentator `lung_vessels` task (vessels, airways).
"""
import argparse
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

TOTALSEG_ROIS = [
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "heart", "aorta", "pulmonary_vein", "superior_vena_cava", "trachea",
]
LOBES = TOTALSEG_ROIS[:5]


def load_gt(gt_dir: Path, key: str):
    img = nib.load(str(gt_dir / f"{key}.nii.gz"))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def load_pred_hu(path: Path, hu_min: float, hu_max: float) -> np.ndarray:
    p = np.squeeze(np.load(path).astype(np.float32))
    p = np.clip((p + 1.0) / 2.0, 0.0, 1.0)
    return (p * (hu_max - hu_min) + hu_min).transpose(1, 2, 0)  # (z, x, y) -> (x, y, z)


def to_axial(hu_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(hu_xyz.transpose(2, 1, 0))  # (z, y, x)


def from_axial(a: np.ndarray) -> np.ndarray:
    return a.transpose(2, 1, 0)


def segment_threshold(hu_xyz, affine, air_hu=-400.0, min_component_ml=100.0):
    vol = to_axial(hu_xyz)
    air = vol < air_hu
    inside = np.zeros_like(air)
    for z in range(air.shape[0]):
        lab, n = ndi.label(air[z])
        if n == 0:
            continue
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        keep = np.ones(n + 1, dtype=bool)
        keep[border] = False
        keep[0] = False
        inside[z] = keep[lab]
    inside = ndi.binary_opening(inside, iterations=1)
    lab, n = ndi.label(inside)
    if n:
        voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
        sizes = ndi.sum(inside, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes * voxel_ml >= min_component_ml
        inside = keep[lab]
    return {"lung": from_axial(inside)}


class Lungmask:
    def __init__(self, batch_size):
        from lungmask import LMInferer
        self.inferer = LMInferer(modelname="R231", batch_size=batch_size)

    def __call__(self, hu_xyz, affine):
        seg = from_axial(self.inferer.apply(to_axial(hu_xyz)))
        return {"lung": seg > 0}


class TotalSeg:
    def __init__(self, task):
        from totalsegmentator.map_to_binary import class_map
        self.task = task
        self.names = class_map[task]

    def __call__(self, hu_xyz, affine):
        from totalsegmentator.python_api import totalsegmentator
        img = nib.Nifti1Image(hu_xyz.astype(np.float32), affine)
        kw = dict(task=self.task, quiet=True, device="gpu")
        if self.task == "total":
            kw["roi_subset"] = TOTALSEG_ROIS
        seg = np.asarray(totalsegmentator(img, None, **kw).dataobj)
        wanted = TOTALSEG_ROIS if self.task == "total" else list(self.names.values())
        out = {name: seg == idx for idx, name in self.names.items() if name in wanted}
        if self.task == "total":
            out["lung"] = np.any([out[l] for l in LOBES], axis=0)
        return out


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", required=True, help="CSV with a 'Keys' column, or one key per line.")
    ap.add_argument("--gt_dir", default="~/data/CT/LIDC-IDRI/CT")
    ap.add_argument("--method", action="append", default=[], metavar="NAME=DIR",
                    help="Generated volumes as DIR/<key>.npy. Repeatable.")
    ap.add_argument("--segmenter", choices=["threshold", "lungmask", "totalseg", "lung_vessels"], required=True)
    ap.add_argument("--hu_min", type=float, default=-1350.0)
    ap.add_argument("--hu_max", type=float, default=150.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N cases (for testing).")
    ap.add_argument("--shard", default="0/1", help="i/n: process keys with index %% n == i.")
    ap.add_argument("--raw_gt", action="store_true",
                    help="Also segment the unclipped ground truth (method 'gt_raw').")
    ap.add_argument("--out", required=True, help="Output CSV (appended to; finished pairs are skipped).")
    args = ap.parse_args()

    lines = [l.strip() for l in open(os.path.expanduser(args.keys)) if l.strip()]
    keys = [l for l in lines if l.lower() != "keys"]
    if args.limit:
        keys = keys[: args.limit]
    si, sn = map(int, args.shard.split("/"))
    keys = keys[si::sn]
    gt_dir = Path(os.path.expanduser(args.gt_dir))
    methods = dict((m.split("=", 1)[0], Path(os.path.expanduser(m.split("=", 1)[1]))) for m in args.method)

    if args.segmenter == "threshold":
        segment = segment_threshold
    elif args.segmenter == "lungmask":
        segment = Lungmask(args.batch_size)
    else:
        segment = TotalSeg("total" if args.segmenter == "totalseg" else "lung_vessels")

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["key"], r["method"]) for r in csv.DictReader(fh)}
    new_file = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["key", "method", "segmenter", "structure", "volume_ml", "gt_volume_ml", "dice_vs_gt"])

        def emit(key, name, masks, gt_masks, voxel_ml):
            for s, m in masks.items():
                g = gt_masks[s]
                w.writerow([key, name, args.segmenter, s, float(m.sum() * voxel_ml),
                            float(g.sum() * voxel_ml), dice(m, g)])
            w.writerow([key, name, args.segmenter, "_done", 0, 0, 0])
            fh.flush()

        for i, key in enumerate(keys):
            todo = [n for n in methods if (key, n) not in done and (methods[n] / f"{key}.npy").exists()]
            need_gt = (key, "gt") not in done or (args.raw_gt and (key, "gt_raw") not in done)
            if not todo and not need_gt:
                continue
            raw, affine = load_gt(gt_dir, key)
            voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
            gt_masks = segment(np.clip(raw, args.hu_min, args.hu_max), affine)
            if (key, "gt") not in done:
                emit(key, "gt", gt_masks, gt_masks, voxel_ml)
            if args.raw_gt and (key, "gt_raw") not in done:
                emit(key, "gt_raw", segment(raw, affine), gt_masks, voxel_ml)
            for name in todo:
                hu = load_pred_hu(methods[name] / f"{key}.npy", args.hu_min, args.hu_max)
                if hu.shape != raw.shape:
                    print(f"[shape] {name} {key}: {hu.shape} vs {raw.shape}, skipped", flush=True)
                    continue
                emit(key, name, segment(hu, affine), gt_masks, voxel_ml)
            print(f"[{i + 1}/{len(keys)}] {key}", flush=True)


if __name__ == "__main__":
    main()
