"""Recompute tips_distance for a post-processed demo.

Definition (same as retarget_myohand.py): Euclidean distance from each fingertip
to the nearest mug vertex (sqrt of the single-point chamfer distance), columns
thumb, index, middle, ring, pinky. The method is first checked on the original
demo with the human MANO tips against the stored tips_distance.
"""
import argparse
import copy
import pickle

import numpy as np
import torch
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--orig", default="assets/retargeted/1292e_training_demo.pkl")
ap.add_argument("--demo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale_xy", type=float, default=0.879)
ap.add_argument("--scale_z", type=float, default=1.0)
args = ap.parse_args()
FING = ["thumb", "index", "middle", "ring", "pinky"]


def N(x):
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def tips_of(rh, group):
    return np.stack([N(rh[group][f + "_tip"]) for f in FING], axis=1)   # (T,5,3)


def tip_dist(tips, traj, verts):
    out = np.zeros((tips.shape[0], 5))
    for t in range(tips.shape[0]):
        w = verts @ traj[t, :3, :3].T + traj[t, :3, 3]
        out[t] = cKDTree(w).query(tips[t])[0]
    return out


orig = pickle.load(open(args.orig, "rb"))["rh"]
demo = pickle.load(open(args.demo, "rb"))
rh = demo["rh"]
old_td = N(orig["tips_distance"])
o_traj, o_verts = N(orig["obj_trajectory"]), N(orig["obj_verts"])

sp = cKDTree(o_verts).query(o_verts, k=2)[0][:, 1]
print("mug vertex spacing: mean %.1f mm (nearest-vertex distance overestimates surface distance by about half of this)" % (sp.mean() * 1000))
chk = tip_dist(tips_of(orig, "mano_joints"), o_traj, o_verts)
print("check 1, human MANO tips vs old mug vertices against stored tips_distance:")
print("  per finger mean |diff| (mm):", np.round(np.abs(chk - old_td).mean(0) * 1000, 2), "| max (mm):", np.round(np.abs(chk - old_td).max(0) * 1000, 1))
myo_old = tip_dist(tips_of(orig, "myo_joints"), o_traj, o_verts)
print("check 2, ORIGINAL MyoHand tips vs old mug, mean (mm):", np.round(myo_old.mean(0) * 1000, 1), "| stored human tips_distance mean (mm):", np.round(old_td.mean(0) * 1000, 1))

vs = N(rh["obj_verts"]) * np.array([args.scale_xy, args.scale_xy, args.scale_z])
new_td = tip_dist(tips_of(rh, "myo_joints"), N(rh["obj_trajectory"]), vs)
print("per finger, stored old vs new:")
for i, f in enumerate(FING):
    print("  %-7s old mean %6.1f min %5.1f frac<10mm %.2f | new mean %6.1f min %5.1f frac<10mm %.2f (mm)" % (
        f, old_td[:, i].mean() * 1000, old_td[:, i].min() * 1000, (old_td[:, i] < 0.01).mean(),
        new_td[:, i].mean() * 1000, new_td[:, i].min() * 1000, (new_td[:, i] < 0.01).mean()))

ot = rh["tips_distance"]
new_t = torch.as_tensor(new_td, dtype=ot.dtype, device=ot.device) if torch.is_tensor(ot) else new_td.astype(np.asarray(ot).dtype)
out = copy.deepcopy(demo)
out["rh"]["tips_distance"] = new_t
pickle.dump(out, open(args.out, "wb"))
print("wrote", args.out)
