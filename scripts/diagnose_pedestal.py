"""Kinematic (no dynamics) distance from every collision body to the pedestal over the whole demo,
plus a listing of the forearm bones' collision geoms. Negative = penetrating."""
import sys, numpy as np, mujoco
sys.path.insert(0, ".")
import envs.mjwarp_myohand_env as E
env = E.MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)
J = mujoco.mjtObj.mjOBJ_JOINT
is_col = lambda g: m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
ped = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER and is_col(g))
src_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "src_O02@0015@00020_free")])
dst_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "dst_O02@0010@00003_free")])
root_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "root_r")])
groups = {}
for g in range(m.ngeom):
    if is_col(g) and g != ped and m.geom_bodyid[g] != 0:
        groups.setdefault(int(m.geom_bodyid[g]), []).append(g)
dst_p = env.demo_dst_obj_traj0[:3, 3].cpu().numpy()
dst_q = E.rotmat_to_quat(env.demo_dst_obj_traj0[None, :3, :3])[0].cpu().numpy()
mug_p = env.demo_src_obj_traj[:, :3, 3].cpu().numpy()
mug_q = E.rotmat_to_quat(env.demo_src_obj_traj[:, :3, :3]).cpu().numpy()
wp_, wq_ = env.demo_opt_wrist_pos.cpu().numpy(), env.demo_opt_wrist_quat.cpu().numpy()
dof = env.demo_opt_dof_pos.cpu().numpy(); dofa = np.asarray(env.dof_adrs)
T = wp_.shape[0]
ft = np.zeros(6)
best = {}
for t in range(T):
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = wp_[t]; d.qpos[env.root_adr + 3:env.root_adr + 7] = wq_[t]
    d.qpos[dofa] = dof[t]
    d.qpos[env.src_adr:env.src_adr + 3] = mug_p[t]; d.qpos[env.src_adr + 3:env.src_adr + 7] = mug_q[t]
    d.qpos[env.dst_adr:env.dst_adr + 3] = dst_p; d.qpos[env.dst_adr + 3:env.dst_adr + 7] = dst_q
    mujoco.mj_kinematics(m, d)
    for b, gs in groups.items():
        if b == dst_body and t > 0:
            continue
        dm = min(mujoco.mj_geomDistance(m, d, g, ped, 0.05, ft) for g in gs) * 1000
        if b not in best or dm < best[b][0]:
            best[b] = (dm, t)
print("pedestal geom id %d | distance to pedestal (mm), min over %d frames" % (ped, T))
for b, (dm, t) in sorted(best.items(), key=lambda kv: kv[1][0]):
    if dm < 8:
        tag = "SRC MUG" if b == src_body else "DST MUG (frame 0)" if b == dst_body else ("hand" if m.body_rootid[b] == root_body else "other")
        print("  %-22s %7.2f mm  [t=%d]  %s" % (bn(b), dm, t, tag))
print("forearm collision geoms:")
for name in ("radius_r", "ulna_r"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and is_col(g):
            print("  %-9s geom %d name=%s type=%s size=%s contype=%d conaff=%d group=%d" % (
                name, g, gn(g), mujoco.mjtGeom(m.geom_type[g]).name, np.round(m.geom_size[g], 3),
                m.geom_contype[g], m.geom_conaffinity[g], m.geom_group[g]))
