"""Kinematic check (no physics): place hand and source mug exactly as
env.reset_idx does (same axis-corrected tensors), then measure surface
distance from each distal finger's collision geoms to the mug's collision
geoms. Compare with the human demo's own fingertip-to-mug distance."""
import sys
import numpy as np
import mujoco

sys.path.insert(0, ".")
sys.path.insert(0, "scripts/retargeting")
import envs.mjwarp_myohand_env as E
from myohand_def import MyoHandR

env = E.MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)


def is_col(g):
    return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0


src_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")
src_body = int(m.jnt_bodyid[src_jid])
mug_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == src_body and is_col(g)]
print("mug collision geoms:", len(mug_geoms))

dexhand = MyoHandR()
names = dexhand.contact_body_names  # distal thumb, index, middle, ring, pinky
labels = ["thumb", "index", "middle", "ring", "pinky"]
finger_geoms = []
for n in names:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and is_col(g)]
    print("%-18s body_id=%d collision geoms=%d" % (n, bid, len(gs)))
    finger_geoms.append(gs)

human = env.rh_demo["tips_distance"].cpu().numpy()  # (T, 5) thumb,index,middle,ring,pinky
T = env.demo_opt_wrist_pos.shape[0]
fromto = np.zeros(6)
dist = np.full((T, 5), np.nan)

for t in range(T):
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = env.demo_opt_wrist_pos[t].cpu().numpy()
    d.qpos[env.root_adr + 3:env.root_adr + 7] = env.demo_opt_wrist_quat[t].cpu().numpy()
    d.qpos[env.dof_adrs] = env.demo_opt_dof_pos[t].cpu().numpy()
    d.qpos[env.src_adr:env.src_adr + 3] = env.demo_src_obj_traj[t, :3, 3].cpu().numpy()
    d.qpos[env.src_adr + 3:env.src_adr + 7] = E.rotmat_to_quat(env.demo_src_obj_traj[t, :3, :3][None])[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    for f, gs in enumerate(finger_geoms):
        best = 0.2
        for g in gs:
            for h in mug_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, g, h, 0.2, fromto))
        dist[t, f] = best

print()
print("MyoHand finger-to-mug surface distance over all %d frames" % T)
print("%-7s %8s %12s %12s %12s" % ("finger", "min mm", "frames<2mm", "frames<5mm", "first<5mm"))
for f in range(5):
    col = dist[:, f] * 1000
    first = int(np.argmax(col < 5)) if (col < 5).any() else None
    print("%-7s %8.1f %12d %12d %12s" % (labels[f], np.nanmin(col), int((col < 2).sum()), int((col < 5).sum()), first))

print()
print("MyoHand mm / human mm at selected frames:")
print("%-6s" % "frame" + "".join("%18s" % l for l in labels))
for t in [50, 70, 76, 90, 100, 105, 150, 200, 300]:
    print("%-6d" % t + "".join("%11.1f/%-6.1f" % (dist[t, f] * 1000, human[t, f] * 1000) for f in range(5)))
