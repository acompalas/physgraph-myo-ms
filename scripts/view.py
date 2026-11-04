"""Unified kinematic viewer (physics OFF): right hand + source mug + destination mug,
placed exactly as the training env places them (same axis correction, same mug poses).

  python3 scripts/view.py                                   # playback, everything
  python3 scripts/view.py --show src,dst                    # mugs only (isolated)
  python3 scripts/view.py --show rh                         # hand only
  python3 scripts/view.py --frame 300 --measure             # static frame + fingertip-to-mug mm
  python3 scripts/view.py --demo assets/retargeted/1292e_training_demo_pen.pkl   # compare demo files
  python3 scripts/view.py --start 300 --end 631 --speed 0.3 --contacts

In the viewer, Rendering tab -> 'Contact Point' also shows penetrating pairs."""
import sys, time, pickle, argparse
from pathlib import Path
import numpy as np, torch, mujoco, mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/retargeting"))
import envs.mjwarp_myohand_env as E
from myohand_def import MyoHandR

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default=str(ROOT / "assets/retargeted/1292e_training_demo.pkl"))
ap.add_argument("--show", default="rh,src,dst", help="comma list from: rh, src, dst")
ap.add_argument("--frame", type=int, default=None, help="static single frame instead of playback")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--speed", type=float, default=0.5)
ap.add_argument("--contacts", action="store_true", help="enable contact-point overlay at launch")
ap.add_argument("--measure", action="store_true", help="print fingertip-to-source-mug distances (mm)")
ap.add_argument("--dist", type=float, default=0.6, help="camera distance")
ap.add_argument("--groups", default=None, help="geom groups to draw, e.g. 3 or 0,1 (default: all)")
args = ap.parse_args()
groups = None if args.groups is None else {int(x) for x in args.groups.split(',')}

show = set(s.strip() for s in args.show.split(","))
assert show <= {"rh", "src", "dst"}, "--show takes rh, src, dst"

def find_rh(o):
    if isinstance(o, dict):
        if "opt_dof_pos" in o:
            return o
        for v in o.values():
            r = find_rh(v)
            if r is not None:
                return r

def tt(x):
    x = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return x.to(dtype=torch.float32)

with open(args.demo, "rb") as f:
    rh = find_rh(pickle.load(f))
print("demo:", args.demo, "| showing:", sorted(show))

env = E.MyoHandPourEnv(num_envs=1, device="cuda:0")
m, dev, C = env.mj_model, env.device, env._axis_C
d = mujoco.MjData(m)

wpos = (C @ tt(rh["opt_wrist_pos"]).to(dev).T).T.cpu().numpy()
wquat = E.rotmat_to_quat(C[None] @ E.aa_to_rotmat(tt(rh["opt_wrist_rot"]).to(dev))).cpu().numpy()
dof = tt(rh["opt_dof_pos"]).numpy()
mug_p = env.demo_src_obj_traj[:, :3, 3].cpu().numpy()
mug_q = E.rotmat_to_quat(env.demo_src_obj_traj[:, :3, :3]).cpu().numpy()
dst_p = env.demo_dst_obj_traj0[:3, 3].cpu().numpy()
dst_q = E.rotmat_to_quat(env.demo_dst_obj_traj0[None, :3, :3])[0].cpu().numpy()
T = wpos.shape[0]
dof_adrs = np.asarray(env.dof_adrs)

J = mujoco.mjtObj.mjOBJ_JOINT
src_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "src_O02@0015@00020_free")])
dst_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "dst_O02@0010@00003_free")])
root_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "root_r")])
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if (("rh" not in show and m.body_rootid[b] == root_body)
            or ("src" not in show and b == src_body)
            or ("dst" not in show and b == dst_body)):
        m.geom_rgba[g, 3] = 0.0

def is_col(g):
    return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
mug_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == src_body and is_col(g)]
labels = ["thumb", "index", "middle", "ring", "pinky"]
distal_geoms = []
for n in MyoHandR().contact_body_names:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    distal_geoms.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and is_col(g)])
fromto = np.zeros(6)
from collections import Counter
print('geom groups | collision:', dict(Counter(int(m.geom_group[g]) for g in range(m.ngeom) if is_col(g))),
      '| non-collision:', dict(Counter(int(m.geom_group[g]) for g in range(m.ngeom) if not is_col(g))))

def ncontacts():
    cnt = [0] * 5
    for k in range(d.ncon):
        g1, g2 = int(d.contact.geom1[k]), int(d.contact.geom2[k])
        for i, gs in enumerate(distal_geoms):
            if (g1 in gs and g2 in mug_geoms) or (g2 in gs and g1 in mug_geoms):
                cnt[i] += 1
    return cnt

def measure():
    out = []
    for gs in distal_geoms:
        best = 0.2
        for g in gs:
            for h in mug_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, g, h, 0.2, fromto))
        out.append(best * 1000)
    return "  ".join("%s %5.1f" % (l, v) for l, v in zip(labels, out)) + "  | contacts " + str(ncontacts())

def place(t):
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = wpos[t]
    d.qpos[env.root_adr + 3:env.root_adr + 7] = wquat[t]
    d.qpos[dof_adrs] = dof[t]
    d.qpos[env.src_adr:env.src_adr + 3] = mug_p[t]
    d.qpos[env.src_adr + 3:env.src_adr + 7] = mug_q[t]
    d.qpos[env.dst_adr:env.dst_adr + 3] = dst_p
    d.qpos[env.dst_adr + 3:env.dst_adr + 7] = dst_q
    mujoco.mj_forward(m, d)

first = args.frame if args.frame is not None else args.start
end = T if args.end is None else min(args.end, T)
place(first)
with mujoco.viewer.launch_passive(m, d) as v:
    for g in range(6):
        v.opt.geomgroup[g] = 1 if (groups is None or g in groups) else 0
    if args.contacts:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.cam.lookat[:] = mug_p[first]
    v.cam.distance = args.dist
    v.cam.azimuth, v.cam.elevation = 135, -20
    if args.frame is not None:
        print("frame %d" % first)
        if args.measure:
            print("  finger-to-mug mm:", measure())
        while v.is_running():
            v.sync()
            time.sleep(0.02)
    else:
        while v.is_running():
            for t in range(first, end):
                if not v.is_running():
                    break
                place(t)
                if args.measure and t % 30 == 0:
                    print("t=%3d  mm: %s" % (t, measure()))
                v.sync()
                time.sleep((1 / 60) / args.speed)
            time.sleep(0.7)
