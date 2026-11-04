"""Read-only: for frames where a fingertip is near the mug, express the
tip->mug-axis (inward) direction in the tip site's own frame. A consistent
direction = that finger's pad axis. Does NOT write the demo file."""
import pickle, sys
from pathlib import Path
import mujoco, numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts/retargeting"))
from myohand_def import MyoHandR

def aa_to_quat_np(aa):
    th = np.clip(np.linalg.norm(aa, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([np.cos(th / 2), aa / th * np.sin(th / 2)], axis=-1)

model = mujoco.MjModel.from_xml_path(str(root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
data = mujoco.MjData(model)
rh = pickle.load(open(root / "assets/retargeted/1292e_training_demo.pkl", "rb"))["rh"]
wpos = rh["opt_wrist_pos"].cpu().numpy(); wquat = aa_to_quat_np(rh["opt_wrist_rot"].cpu().numpy())
dof = rh["opt_dof_pos"].cpu().numpy(); obj = rh["obj_trajectory"].cpu().numpy()
td = rh["tips_distance"].cpu().numpy()

root_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")]
dof_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MyoHandR().dof_names])
tips = [("thumb", "THtip_r"), ("index", "IFtip_r"), ("middle", "MFtip_r"), ("ring", "RFtip_r"), ("pinky", "LFtip_r")]
ids = {}
for k, n in tips:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    ids[k] = ("body", b) if b >= 0 else ("site", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n))
    assert ids[k][1] >= 0, n

acc = {k: [] for k, _ in tips}
for f in range(wpos.shape[0]):
    data.qpos[root_adr:root_adr + 3] = wpos[f]; data.qpos[root_adr + 3:root_adr + 7] = wquat[f]
    data.qpos[dof_adrs] = dof[f]; mujoco.mj_forward(model, data)
    mR, mp = obj[f, :3, :3], obj[f, :3, 3]
    for col, (k, _) in enumerate(tips):
        if td[f, col] > 0.02: continue
        kind, i = ids[k]
        p = data.xpos[i] if kind == "body" else data.site_xpos[i]
        R = (data.xmat[i] if kind == "body" else data.site_xmat[i]).reshape(3, 3)
        loc = mR.T @ (p - mp); rad = np.array([loc[0], loc[1], 0.0]); nr = np.linalg.norm(rad)
        if nr < 1e-6: continue
        acc[k].append(R.T @ (mR @ (-rad / nr)))

labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
print("finger   frames  mean_dir_in_tip_frame        resultant_len  mean_angle_deg  nearest_axis")
for k, _ in tips:
    v = np.array(acc[k])
    if len(v) == 0: print(k, "no frames within 2 cm"); continue
    m = v.mean(0); rl = np.linalg.norm(m); d = m / rl
    ang = np.degrees(np.arccos(np.clip(v @ d, -1, 1))).mean()
    print(f"{k:8s} {len(v):5d}   [{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}]   {rl:.2f}          {ang:5.1f}         {labels[int(np.argmax([d[0],-d[0],d[1],-d[1],d[2],-d[2]]))]}")
