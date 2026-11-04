"""Read-only. Per demo pickle and finger: (1) pad alignment = angle between the
fingertip pad direction and the direction into the mug while the finger is near
it; (2) slip = fingertip movement over the mug surface per frame, in the mug's
own frame (0 = stuck to the mug). Pad direction = palm normal at the flat
reference pose (qpos0) expressed in the tip frame; sign chosen so the palm
normal points toward the mug in the demo. Thumb: slip only."""
import pickle, sys
from pathlib import Path
import mujoco, numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts/retargeting"))
from myohand_def import MyoHandR

demos = sys.argv[1:] or ["assets/retargeted/1292e_training_demo.pkl",
                         "assets/retargeted/1292e_training_demo_smallmug.pkl"]

def aa_to_quat_np(aa):
    th = np.clip(np.linalg.norm(aa, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([np.cos(th / 2), aa / th * np.sin(th / 2)], axis=-1)

def unit(v): return v / np.linalg.norm(v)

model = mujoco.MjModel.from_xml_path(str(root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
data = mujoco.MjData(model)
dex = MyoHandR()
J = mujoco.mjtObj.mjOBJ_JOINT
root_adr = model.jnt_qposadr[mujoco.mj_name2id(model, J, "root_r")]
dof_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, J, n)] for n in dex.dof_names])
body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
site = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
fingers = [("thumb", "THtip_r"), ("index", "IFtip_r"), ("middle", "MFtip_r"),
           ("ring", "RFtip_r"), ("pinky", "LFtip_r")]
bw, b2, b5 = body(dex.body_names[0]), body("2proxph_r"), body("5proxph_r")

def palm_normal():
    return unit(np.cross(data.xpos[b2] - data.xpos[bw], data.xpos[b5] - data.xpos[bw]))

data.qpos[:] = model.qpos0; mujoco.mj_forward(model, data)
n0 = palm_normal()
R0 = {k: data.site_xmat[site(s)].reshape(3, 3).copy() for k, s in fingers}

def load(path):
    rh = pickle.load(open(root / path, "rb"))["rh"]
    g = lambda k: rh[k].cpu().numpy() if hasattr(rh[k], "cpu") else np.asarray(rh[k])
    return g("opt_wrist_pos"), aa_to_quat_np(g("opt_wrist_rot")), g("opt_dof_pos"), g("obj_trajectory"), g("tips_distance")

td_ref = load(demos[0])[4]          # same near-mug gate for every demo

for path in demos:
    wpos, wquat, dof, obj, _ = load(path)
    T = wpos.shape[0]
    P = {k: np.zeros((T, 3)) for k, _ in fingers}; R = {k: np.zeros((T, 3, 3)) for k, _ in fingers}
    npalm = np.zeros((T, 3)); pc = np.zeros((T, 3))
    for f in range(T):
        data.qpos[root_adr:root_adr + 3] = wpos[f]; data.qpos[root_adr + 3:root_adr + 7] = wquat[f]
        data.qpos[dof_adrs] = dof[f]; mujoco.mj_forward(model, data)
        npalm[f] = palm_normal(); pc[f] = (data.xpos[bw] + data.xpos[b2] + data.xpos[b5]) / 3
        for k, s in fingers:
            P[k][f] = data.site_xpos[site(s)]; R[k][f] = data.site_xmat[site(s)].reshape(3, 3)
    near = (td_ref < 0.02).any(1)
    sgn = np.sign(np.mean([npalm[f] @ (obj[f, :3, 3] - pc[f]) for f in range(T) if near[f]]))
    print(f"\n=== {path} | palm-normal sign {sgn:+.0f} ===")
    print("finger   frames  pad-vs-wall angle deg (mean/median/p90)  %<30deg   slip mm/frame (mean/p95)   radial std mm")
    for col, (k, _) in enumerate(fingers):
        fr = np.where(td_ref[:, col] < 0.02)[0]
        th = np.zeros(T); r = np.zeros(T); z = np.zeros(T); ang = []
        for f in range(T):
            mR, mp = obj[f, :3, :3], obj[f, :3, 3]
            loc = mR.T @ (P[k][f] - mp); th[f] = np.arctan2(loc[1], loc[0]); r[f] = np.hypot(loc[0], loc[1]); z[f] = loc[2]
        for f in fr:
            if k == "thumb": break
            mR = obj[f, :3, :3]; inward = mR @ unit(np.array([-np.cos(th[f]), -np.sin(th[f]), 0.0]))
            pad_w = R[k][f] @ (R0[k].T @ (sgn * n0))
            ang.append(np.degrees(np.arccos(np.clip(pad_w @ inward, -1, 1))))
        step = [np.hypot(r[f] * np.angle(np.exp(1j * (th[f] - th[f - 1]))), z[f] - z[f - 1]) * 1000
                for f in fr if f - 1 in set(fr)]
        a = np.array(ang) if ang else np.array([np.nan])
        print(f"{k:8s} {len(fr):5d}   {np.nanmean(a):6.1f} / {np.nanmedian(a):6.1f} / {np.nanpercentile(a, 90):6.1f}"
              f"                    {100 * np.mean(a < 30):5.0f}      {np.mean(step):6.3f} / {np.percentile(step, 95):6.3f}"
              f"             {np.std(r[fr]) * 1000:5.2f}")
