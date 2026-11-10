"""MyoHand pour-task environment, mujoco-warp backend. Ported from
PhysGraph's dexhandmanip_sh.py (single-hand IsaacGym task), matching
the ComplexObsRLGPUEnv interface (lib/utils/rlgames_utils.py) that
rl_games expects: step/reset/get_number_of_agents/action_space/
observation_space/success_buf/failure_buf/error_buf.

FIXED TRAJECTORY ONLY for now, per Runfa's direction 2026-09-04
("just prove you can train the first successful policy for a fixed
trajectory... we'll consider augmentation later").

Action space: direct wrist target pose (3 pos + 6 rot6d) + per-dof
targets -- matching Runfa's real-hardware preference, implemented via
the same PID-controlled-wrist-pose mechanism PhysGraph itself
documents as an alternate mode (usePIDControl=True in
physgraph_envs/lib/envs/dexhands/base.py), using ArtiMANO's own
reference gains (physgraph_envs/lib/envs/dexhands/artimano_real.py):
Kp_pos=10, Ki_pos=0.003, Kd_pos=0.5, Kp_rot=0.3, Ki_rot=0.01, Kd_rot=0.005
-- NOT yet tuned for MyoHand's own mass/scale.

Real mujoco_warp API confirmed on the pod 2026-09-05 (see project notes
for the full trace) -- mjw.put_model/make_data/step/forward, Data.qpos/
qvel/ctrl/xpos/xquat, wp.to_torch zero-copy interop confirmed working
for both reads and indexed-assignment writes.
"""
import os
import pickle

import gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


def aa_to_quat(aa):
    """Axis-angle (..., 3) -> quaternion (..., 4), [w, x, y, z] (MuJoCo's
    qpos convention). Self-contained -- avoids depending on PhysGraph's
    own torch_utils, which pulls in IsaacGym via the package __init__
    chain for anything beyond the standalone-loadable files."""
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    half = theta / 2
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def quat_to_aa(q):
    """Inverse of aa_to_quat -- quaternion [w,x,y,z] -> axis-angle (...,3),
    needed to express the PID position/rotation error signal."""
    w = q[..., 0].clamp(-1, 1)
    xyz = q[..., 1:]
    theta = 2 * torch.acos(w)
    sin_half = torch.sqrt((1 - w * w).clamp(min=1e-12))
    axis = xyz / sin_half.unsqueeze(-1)
    return axis * theta.unsqueeze(-1)


def rotmat_to_quat(R):
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4), [w,x,y,z]."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    q[..., 0] = torch.sqrt(torch.clamp(tr + 1, min=0)) / 2
    q[..., 1] = torch.sign(R[..., 2, 1] - R[..., 1, 2]) * torch.sqrt(
        torch.clamp(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], min=0)) / 2
    q[..., 2] = torch.sign(R[..., 0, 2] - R[..., 2, 0]) * torch.sqrt(
        torch.clamp(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], min=0)) / 2
    q[..., 3] = torch.sign(R[..., 1, 0] - R[..., 0, 1]) * torch.sqrt(
        torch.clamp(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], min=0)) / 2
    return q


def quat_mul(q1, q2):
    """Hamilton product, [w,x,y,z] convention, (..., 4) x (..., 4) -> (..., 4)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conjugate(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def aa_to_rotmat(aa):
    """Axis-angle -> rotation matrix (Rodrigues' formula), matching
    the real kinematic playback script's own numpy implementation
    exactly, needed to apply axis_correction_matrix to wrist rotation
    data (which is stored as axis-angle, not quaternion/rotmat)."""
    theta = torch.norm(aa, dim=-1, keepdim=True)
    theta_c = torch.clamp(theta, min=1e-8)
    axis = aa / theta_c
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=aa.device).expand(aa.shape[:-1] + (3, 3))
    th = theta[..., None]
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def axis_correction_matrix(device):
    """Real axis-correction matrix (Rz(-90)@Rx(90)) needed to convert
    obj_trajectory (sourced from the original raw mocap dataset, never
    re-solved into our MJCF scene's own frame the way opt_wrist_pos/rot
    were via retargeting optimization) into our scene's coordinate
    convention. Confirmed as a real, necessary fix 2026-09-10 via direct
    kinematic playback: without this, the mug genuinely renders on its
    side at frame 0 (not upright), matching the same wrong-orientation
    behavior seen in the real physics-based live eval -- proven to be a
    pure data/coordinate issue since the kinematic playback used zero
    physics. A pure rotation/permutation matrix, applied to both the
    position and rotation submatrix of each frame's full 4x4 transform,
    never combining raw components separately (see project notes' own
    established coordinate-frame gotcha from the retargeting phase)."""
    rz = torch.tensor([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]], device=device)
    rx = torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]], device=device)
    return rz @ rx


def correct_obj_traj(traj, C):
    """Apply the axis-correction matrix to a full (T,4,4) trajectory of
    object transforms -- corrects position and rotation together per
    frame, not as separately-combined components."""
    pos = traj[:, :3, 3]
    rot = traj[:, :3, :3]
    corrected_pos = (C @ pos.T).T
    corrected_rot = C[None] @ rot
    out = traj.clone()
    out[:, :3, 3] = corrected_pos
    out[:, :3, :3] = corrected_rot
    return out


def quat_rotate_vec(q, v):
    """Rotate vector v (..., 3) by quaternion q (..., 4), [w,x,y,z]
    convention. Standard closed-form quat-vector rotation, needed for
    manip_obj_com (rotating the body-frame COM offset into world frame),
    matching PhysGraph real construction (dexhandmanip_bih.py:1150-1154)."""
    w = q[..., 0:1]
    qvec = q[..., 1:]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + w * t + torch.cross(qvec, t, dim=-1)


def quat_to_angle(q):
    """Quaternion [w,x,y,z] -> rotation angle (magnitude only, radians).
    Matches PhysGraph's quat_to_angle_axis(...)[0] usage in
    compute_imitation_reward (only the angle component is used there)."""
    w = q[..., 0].clamp(-1, 1)
    return 2 * torch.acos(w.abs())  # abs() avoids the double-cover sign ambiguity


class MyoHandPourEnv:
    def __init__(self, num_envs, device="cuda:0", max_episode_length=None, headless=True):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.environ.get("MYOHAND_MUG_PRESET") == "small":
            os.environ.setdefault("MYOHAND_SRC_MUG_SCALE_XY", "0.879")
            os.environ.setdefault("MYOHAND_DEMO", "1292e_training_demo_smallmug_v1.pkl")
            os.environ.setdefault("MYOHAND_SRC_BPS", "1292e_src_mug_bps_small.pkl")
        demo_path = os.path.join(repo_root, "assets/retargeted", os.environ.get("MYOHAND_DEMO", "1292e_training_demo.pkl"))
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)
        self.rh_demo = demo["rh"]
        self.lh_demo = demo["lh_stationary"]
        self.pour_frame = demo["pour_frame"]

        self.seq_len = self.rh_demo["opt_dof_pos"].shape[0]
        self.max_episode_length = max_episode_length or self.seq_len

        # THE real fix (2026-09-10): opt_wrist_pos/rot need the SAME axis
        # correction as obj_trajectory. Confirmed directly against the real
        # kinematic playback script (playback_full_scene.py) -- it applies
        # C to BOTH wrist AND object data, not object alone. My first pass
        # at this fix only corrected obj_trajectory, leaving wrist
        # uncorrected -- explains why hand+mug looked internally consistent
        # with each other (both in the same wrong frame) but still wrong
        # relative to the real floor/world convention.
        self._axis_C = axis_correction_matrix(self.device)
        _raw_wrist_pos = self._to_tensor(self.rh_demo["opt_wrist_pos"])
        _raw_wrist_rot_aa = self._to_tensor(self.rh_demo["opt_wrist_rot"])
        self.demo_opt_wrist_pos = (self._axis_C @ _raw_wrist_pos.T).T
        self.demo_opt_wrist_quat = rotmat_to_quat(self._axis_C[None] @ aa_to_rotmat(_raw_wrist_rot_aa))
        self.demo_opt_dof_pos = self._to_tensor(self.rh_demo["opt_dof_pos"])
        # real source-mug trajectory (part of the rh demo dict -- that's
        # the object the right hand actually manipulates)
        self.demo_src_obj_traj = correct_obj_traj(self._to_tensor(self.rh_demo["obj_trajectory"]), self._axis_C)
        # NOTE: unlike rh (src mug + wrist), lh_stationary's obj_trajectory
        # does NOT need the axis correction -- confirmed numerically 2026-
        # 09-10: its raw, uncorrected "up" direction already points to
        # (0,0,1) essentially exactly. The two demo sources were packed
        # with different/inconsistent conventions. Applying C here
        # actively breaks an already-correct orientation.
        self.demo_dst_obj_traj0 = self._to_tensor(self.lh_demo["obj_trajectory"][0])

        hand_xml = os.path.join(repo_root, "assets/hands/myohand_r_ulnaroot_scene.xml")
        mug_src_xml = os.path.join(repo_root, "assets/objects/O02@0015@00020/O02@0015@00020.xml")
        mug_dst_xml = os.path.join(repo_root, "assets/objects/O02@0010@00003/O02@0010@00003.xml")

        base = mujoco.MjSpec.from_file(hand_xml)
        mug_src = mujoco.MjSpec.from_file(mug_src_xml)
        mug_dst = mujoco.MjSpec.from_file(mug_dst_xml)
        # PhysGraph object mass rule (dexhandmanip_bih.py:753 density 200, :501 mass cap 0.5 kg).
        # mass = 200 * summed collision-piece volume (src 3.395e-4 m^3 -> 0.068 kg, dst 2.338e-4 m^3 -> 0.047 kg).
        # The XML masses (1.076 / 0.719 kg) are ~16x too heavy; ratios below map them to those targets.
        _mass_scale = float(os.environ.get("MYOHAND_MUG_MASS_SCALE", "1.0"))
        _sxy = float(os.environ.get("MYOHAND_SRC_MUG_SCALE_XY", "1.0"))
        _sz = float(os.environ.get("MYOHAND_SRC_MUG_SCALE_Z", "1.0"))
        if _sxy != 1.0 or _sz != 1.0:
            _ms = np.array([_sxy, _sxy, _sz])
            for _m in mug_src.meshes:
                _m.scale = _ms
            for _g in mug_src.geoms:
                _g.pos = np.array(_g.pos) * _ms
            for _b in mug_src.bodies:
                _b.ipos = np.array(_b.ipos) * _ms
            print(f"[mug] poured mug scaled xy={_sxy} z={_sz} (mass x{_sxy**2*_sz:.3f})")
        for spec, _ratio in ((mug_src, 0.0631 * _sxy**2 * _sz), (mug_dst, 0.0651)):
            for _b in spec.bodies:
                if _b.mass > 0:
                    _m1 = min(0.5, _b.mass * _ratio * _mass_scale)
                    _b.inertia = _b.inertia * (_m1 / _b.mass)
                    _b.mass = _m1
        for spec, prefix in [(mug_src, "src_"), (mug_dst, "dst_")]:
            site = base.worldbody.add_site(name=f"attach_{prefix}", pos=[0, 0, 0])
            base.attach(spec, prefix=prefix, site=site)
            base.delete(site)

        self.mj_model = base.compile()
        _ts = float(os.environ.get("MYOHAND_TIMESTEP", "0.004166667"))
        if _ts > 0:
            self.mj_model.opt.timestep = _ts
        _tc = float(os.environ.get("MYOHAND_CONTACT_TC", "0.008333"))
        if _tc > 0:
            self.mj_model.geom_solref[:, 0] = _tc

        # wrist actuator fidelity: PhysGraph's hand asset has disable_gravity=True and
        # linear/angular damping 20 (dexhandmanip_bih.py lines ~280-290)
        _rj = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")
        _rb = int(self.mj_model.jnt_bodyid[_rj])
        _hand_gravity = os.environ.get("MYOHAND_HAND_GRAVITY", "0")
        if _hand_gravity == "0":
            for _b in range(self.mj_model.nbody):
                if self.mj_model.body_rootid[_b] == _rb:
                    self.mj_model.body_gravcomp[_b] = 1.0
        _damp = float(os.environ.get("MYOHAND_ROOT_DAMPING", "20.0"))
        self._root_damp_np = None
        if _damp > 0:
            _da = int(self.mj_model.jnt_dofadr[_rj])
            _d0 = mujoco.MjData(self.mj_model)
            mujoco.mj_forward(self.mj_model, _d0)
            _tm = float(self.mj_model.body_subtreemass[_rb])
            _I = np.zeros((3, 3))
            for _b in range(self.mj_model.nbody):
                if self.mj_model.body_rootid[_b] != _rb:
                    continue
                _R = _d0.ximat[_b].reshape(3, 3)
                _Ib = _R @ np.diag(self.mj_model.body_inertia[_b]) @ _R.T
                _r = _d0.xipos[_b] - _d0.xpos[_rb]
                _I += _Ib + self.mj_model.body_mass[_b] * (np.dot(_r, _r) * np.eye(3) - np.outer(_r, _r))
            _R0 = _d0.xmat[_rb].reshape(3, 3)
            _Iloc = _R0.T @ _I @ _R0   # inertia about root origin in the BODY frame (free-joint angular dofs are body-frame)
            self._root_damp_np = (_damp * _tm, (_damp * _Iloc).astype(np.float32))
        print(f"[wrist] hand_gravity={_hand_gravity} (0=compensated) root_damping={_damp} coeffs={None if self._root_damp_np is None else (round(float(self._root_damp_np[0]), 2), np.round(np.diag(self._root_damp_np[1]), 3).tolist())}", flush=True)
        _lim_tc = float(os.environ.get("MYOHAND_LIMIT_TC", "0"))
        if _lim_tc > 0:
            for _j in range(self.mj_model.njnt):
                if self.mj_model.jnt_limited[_j] and self.mj_model.jnt_type[_j] in (2, 3):
                    self.mj_model.jnt_solref[_j] = [_lim_tc, 1.0]
        print(f"[limits] joint limit solref timeconst={'default (0.02)' if _lim_tc <= 0 else _lim_tc}")
        _ip_flex = float(os.environ.get("MYOHAND_IP_MAX_FLEX_DEG", "0"))
        if _ip_flex > 0:
            _j = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ip_flexion_r")
            self.mj_model.jnt_range[_j, 0] = -np.radians(_ip_flex)
            print(f"[limits] ip_flexion_r max flexion set to {_ip_flex} deg")
        _mp_max = float(os.environ.get("MYOHAND_MP_MAX_DEG", "30"))
        if _mp_max > 0:
            _j = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "mp_flexion_r")
            self.mj_model.jnt_range[_j, 1] = np.radians(_mp_max)
            print(f"[limits] mp_flexion_r upper limit set to {_mp_max} deg")
        if os.environ.get("MYOHAND_THUMB_LIMITS", "0") == "1":
            for _n, (_lo, _hi) in {"cmc_flexion_r": (2, 36), "cmc_abduction_r": (-7, 33), "mp_flexion_r": (-1, 32), "ip_flexion_r": (-28, 14)}.items():
                _j = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, _n)
                self.mj_model.jnt_range[_j] = np.radians([_lo, _hi])
            print("[limits] thumb ranges tightened to demo range +-10 deg")
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(
            self.mj_model, nworld=num_envs, nconmax=max(num_envs * 100, 5000), njmax=max(num_envs * 4, 5000),
            # floors added: the *8/*4 multipliers only work at training
            # scale (contacts/constraints share a global pool across
            # many envs, not every env peaks simultaneously) -- at
            # num_envs=1 (eval) there is no pooling benefit, this one
            # env alone needs the real per-env requirement (~100-106,
            # confirmed earlier this project), not a tiny multiple of 1
            naccdmax=max(num_envs * 50, 5000), nccdmax=max(num_envs * 50, 5000),
        )  # naccdmax/nccdmax explicit -- real GPU OOM at num_envs=256/1024 from an
        # unbounded default MULTICCD buffer (multiccd_polygon), unrelated to
        # njmax/nconmax; *4/env is a starting guess, may need tuning.
        # Tried bumping njmax *4->*8/env (2026-10-03) as a precaution for
        # condim=6 (roughly doubles constraint rows per contact vs the old
        # condim=3 default) -- reverted same day: caused a real upfront
        # GPU alloc OOM at num_envs=1024 (confirmed working at *4/env for
        # hundreds of epochs previously), with no actual njmax-overflow
        # failure ever observed to justify it. If a real njmax overflow
        # shows up later, prefer a smaller bump (e.g. *5 or *6) over *8.

        self.n_dofs_hand = 23
        self.dof_names = self._dof_names()
        self.body_names = self._body_names()
        self.root_adr = self._joint_qpos_adr("root_r")
        self.root_dof_adr = self._joint_dof_adr("root_r")
        self.dof_adrs = np.array([self._joint_qpos_adr(n) for n in self.dof_names])
        self.dof_veladrs = np.array([self._joint_dof_adr(n) for n in self.dof_names])
        # mixed body/site lookup: tips (THtip_r etc) are real sites, not
        # bodies (confirmed via direct compiled-model inspection) -- for
        # each name, try body first, fall back to site. site_ids[i] >= 0
        # marks an entry that must be read from site_xpos, not xpos.
        self.body_ids = np.zeros(len(self.body_names), dtype=np.int64)
        self.site_ids = np.full(len(self.body_names), -1, dtype=np.int64)
        for _i, _name in enumerate(self.body_names):
            _bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, _name)
            if _bid >= 0:
                self.body_ids[_i] = _bid
            else:
                _sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, _name)
                assert _sid >= 0, f"{_name} not found as body or site"
                self.site_ids[_i] = _sid
        self.is_site_mask = torch.tensor(self.site_ids[1:] >= 0, device=self.device)
        self.src_adr = self._joint_qpos_adr("src_O02@0015@00020_free")
        self.dst_adr = self._joint_qpos_adr("dst_O02@0010@00003_free")
        self.dst_dof_adr = self._joint_dof_adr("dst_O02@0010@00003_free")
        self.src_dof_adr = self._joint_dof_adr("src_O02@0015@00020_free")

        # reuse our own already-built, tested weight_idx (myohand_def.py,
        # matches PhysGraph's DexHand.weight_idx convention exactly)
        import sys
        sys.path.insert(0, os.path.join(repo_root, "scripts/retargeting"))
        from myohand_def import MyoHandR
        self.weight_idx = MyoHandR().weight_idx

        # finite-difference velocity approximation for per-body
        # positions (joints_vel in PhysGraph's reward) -- real MuJoCo
        # per-body velocity (cvel) has a specific 6D com-based
        # convention not yet verified for mujoco_warp, deferred
        self.prev_body_xpos = None
        self.current_joints_pos_ = torch.zeros(num_envs, 20, 3, device=device)
        self.current_joints_vel_ = torch.zeros(num_envs, 20, 3, device=device)

        dexhand = MyoHandR()
        # real target joint positions, in OUR body_names[1:] order
        # (excluding wrist), matching PhysGraph's pack_data flattening
        # logic exactly.
        # Switched (2026-10-03) from raw MANO (human) joint positions to
        # MyoHand's OWN retargeted joint positions (myo_joints, precomputed
        # via real MuJoCo FK on opt_wrist_pos/opt_wrist_rot/opt_dof_pos by
        # scripts/retargeting/compute_myo_joints.py). Confirmed via
        # scripts/retargeting/diagnose_fingertip_gap.py that raw MANO and
        # MyoHand's own retargeted fingertip positions disagree substantially
        # for thumb_tip (mean 4.88cm, max 7.01cm -- larger than the mug's
        # 5.26cm radius) and pinky_tip (mean 2.14cm), while index/middle/
        # ring agree closely (<2.3cm) -- the reward now tracks what MyoHand's
        # own embodiment can actually achieve, not the human's.
        myo_joints = self.rh_demo["myo_joints"]
        target_joints_list = [myo_joints[dexhand.to_hand(b)[0]] for b in self.body_names[1:]]
        _raw_target_joints_pos = torch.stack(target_joints_list, dim=1).to(device=self.device, dtype=torch.float32)
        # Real fix (2026-09-14): mano_joints is a real position quantity
        # (real human MANO joint positions), loaded raw with no axis
        # correction the whole time -- same bug class as wrist/obj_
        # trajectory/velocity, just missed in the earlier passes. Directly
        # confirmed as the actual cause of episode_lengths staying flat
        # at 8 the entire prior training run: finger tip distances were
        # ~0.37-0.45m (vs a ~0.05-0.06m failure threshold) from step 0,
        # never improving, while obj_pos/obj_rot tracking was genuinely
        # fine throughout.
        self.demo_target_joints_pos = (self._axis_C @ _raw_target_joints_pos.reshape(-1, 3).T).T.reshape(_raw_target_joints_pos.shape)

        # target VELOCITIES -- REAL fields from the original demo data
        # (not finite-differenced from our own retargeted trajectory).
        # PhysGraph's own compute_observations_side reads these directly
        # (dexhandmanip_bih.py: side_demo_data["wrist_angular_velocity"]
        # etc) -- confirmed all present in our own 1292e_training_demo.pkl.
        # Using the opt_-prefixed wrist fields specifically: these are the
        # velocity of OUR retargeted trajectory (matching opt_wrist_pos/
        # opt_wrist_rot, what we actually use as targets), NOT the
        # original pre-retargeting human wrist_velocity/wrist_angular_
        # velocity fields (a different, also-present pair -- confirmed
        # both exist, opt_ ones are the consistent choice for us).
        # THE real fix (2026-09-13): velocity fields need the SAME axis
        # correction C as position/rotation -- mathematically exact, not
        # an estimation: velocity is defined as d/dt[position], C is a
        # constant (time-independent) matrix, so by linearity of
        # differentiation d/dt[C@pos] = C@d/dt[pos] = C@velocity exactly.
        # Confirmed the one needed precondition holds: PhysGraph's own
        # real compute_velocity() computes this as np.gradient() directly
        # on the same raw position array, so it genuinely IS a derivative
        # of the raw (uncorrected) position data.
        _raw_wrist_vel = self._to_tensor(self.rh_demo["opt_wrist_velocity"])
        _raw_wrist_ang_vel = self._to_tensor(self.rh_demo["opt_wrist_angular_velocity"])
        self.demo_wrist_vel = (self._axis_C @ _raw_wrist_vel.T).T
        self.demo_wrist_ang_vel = (self._axis_C @ _raw_wrist_ang_vel.T).T

        # Velocity counterpart of the myo_joints switch above -- same
        # precompute script, same np.gradient convention PhysGraph itself
        # uses for compute_velocity().
        myo_joints_vel = self.rh_demo["myo_joints_velocity"]
        joints_vel_list = [myo_joints_vel[dexhand.to_hand(b)[0]] for b in self.body_names[1:]]
        _raw_joints_vel = torch.stack(joints_vel_list, dim=1).to(device=self.device, dtype=torch.float32)
        # Same axis correction as the position target above -- this is
        # its velocity counterpart, d/dt[myo_joints], needs the identical C
        # correction by the same linearity-of-differentiation argument.
        self.demo_joints_vel = (self._axis_C @ _raw_joints_vel.reshape(-1, 3).T).T.reshape(_raw_joints_vel.shape)

        _raw_src_obj_vel = self._to_tensor(self.rh_demo["obj_velocity"])
        _raw_src_obj_ang_vel = self._to_tensor(self.rh_demo["obj_angular_velocity"])
        self.demo_src_obj_vel = (self._axis_C @ _raw_src_obj_vel.T).T
        self.demo_src_obj_ang_vel = (self._axis_C @ _raw_src_obj_ang_vel.T).T

        # destination mug is static -- target velocity is genuinely zero
        # every frame (no corresponding real field exists since our own
        # lh_stationary dict never captured hand-held destination-mug
        # motion -- see project notes on the stationary-mug design)
        self.demo_dst_obj_vel = torch.zeros(3, device=self.device)
        self.demo_dst_obj_ang_vel = torch.zeros(3, device=self.device)

        # real BPS object-shape encodings, precomputed offline (see
        # project notes -- bps_torch/chamfer_distance/pytorch3d chain
        # computed in the physgraph env, bridged via portable pickle,
        # zero bps_torch dependency needed here)
        with open(os.path.join(repo_root, "assets/retargeted", os.environ.get("MYOHAND_SRC_BPS", "1292e_src_mug_bps.pkl")), "rb") as f:
            self.src_mug_bps = torch.tensor(pickle.load(f), device=self.device, dtype=torch.float32)[0]
        with open(os.path.join(repo_root, "assets/retargeted/1292e_dst_mug_bps.pkl"), "rb") as f:
            self.dst_mug_bps = torch.tensor(pickle.load(f), device=self.device, dtype=torch.float32)[0]

        # destination mug's fixed TARGET pose -- matches what reset()
        # already uses to place it (frame 0 of the static trajectory)
        dst_pose0 = self.demo_dst_obj_traj0
        self.lh_target_pos = dst_pose0[:3, 3][None, :]
        self.lh_target_quat = rotmat_to_quat(dst_pose0[:3, :3][None])

        # tip contact-force bodies (real distal phalanx bodies, NOT the
        # tip sites -- sites have no dynamics/mass of their own), matching
        # PhysGraph's real tip_force construction exactly (net_cf per
        # contact body + its own magnitude, dexhandmanip_bih.py:1158-1173)
        self.tip_body_ids = np.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in dexhand.contact_body_names
        ])

        # static object COM (relative to body origin, real body_ipos field)
        # and weight (mass * gravity), matching PhysGraph's real
        # manip_obj_com/manip_obj_weight construction exactly
        src_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "src_O02@0015@00020")
        dst_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "dst_O02@0010@00003")
        self.src_com = self._to_tensor(self.mj_model.body_ipos[src_body_id])
        self.dst_com = self._to_tensor(self.mj_model.body_ipos[dst_body_id])
        _gravity_z = abs(self.mj_model.opt.gravity[2])
        self.src_weight = torch.tensor(self.mj_model.body_mass[src_body_id] * _gravity_z, device=self.device, dtype=torch.float32)
        self.dst_weight = torch.tensor(self.mj_model.body_mass[dst_body_id] * _gravity_z, device=self.device, dtype=torch.float32)

        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.error_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.total_rewards = torch.zeros(num_envs, device=device)

        # PID gains, ArtiMANO reference values (physgraph_envs/lib/envs/
        # dexhands/artimano_real.py) -- NOT yet tuned for MyoHand
        self.Kp_pos, self.Ki_pos, self.Kd_pos = 10.0, 0.003, 0.5
        self.Kp_rot, self.Ki_rot, self.Kd_rot = 0.3, 0.01, 0.005
        self.physics_substeps = int(round((1.0 / 60.0) / self.mj_model.opt.timestep))
        print(f"[physics] dt=1/{1.0 / self.mj_model.opt.timestep:.0f} substeps={self.physics_substeps} solref0_min={float(self.mj_model.geom_solref[:, 0].min()):.5f}", flush=True)
        self.sim_dt = self.mj_model.opt.timestep      # raw physics integration step (1/120)
        self.dt = self.sim_dt * self.physics_substeps  # control/demo cadence (1/60)
        self.pos_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_pos_error = torch.zeros(num_envs, 3, device=device)
        self.rot_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_rot_error = torch.zeros(num_envs, 3, device=device)
        self.wrist_ema = float(os.environ.get("MYOHAND_FORCE_EMA", "0.4"))
        self._applied_force = torch.zeros(num_envs, 3, device=device)
        self._applied_torque = torch.zeros(num_envs, 3, device=device)
        self._root_damp = None
        if self._root_damp_np is not None:
            self._root_damp_lin = float(self._root_damp_np[0])
            self._root_damp_ang = torch.tensor(self._root_damp_np[1], device=device)
            self._root_damp = True
        print(f"[wrist] force_ema={self.wrist_ema}", flush=True)

        self.n_muscles = self.mj_model.nu  # 39, real muscle-tendon actuators
        # (ctrlrange 0-1, dyntype/gaintype/biastype=muscle -- confirmed
        # via mj_model.nu + actuator names, NOT simple position servos)
        act_dim = 9 + self.n_muscles
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        # bimanual-SHAPED (RH real + LH inert-zero, same width each),
        # matching PhysGraph's own per-modality concatenated-tensor
        # convention -- see _compute_obs() for the full rationale
        rh_proprio_dim = 23 * 3 + 13   # dof_pos/cos/sin + wrist pos/quat/vel/angvel
        # dof_vel + obj_pos + obj_quat + obj_vel + obj_ang_vel + obj_com + obj_weight + tip_force
        rh_privileged_dim = 23 + 3 + 4 + 3 + 3 + 3 + 1 + 20  # = 60
        # wrist(23) + joints(180=60*3) + obj(23) + obj_to_joints(21) + gt_tips(5) + bps(128)
        rh_target_dim = 23 + 180 + 23 + 21 + 5 + 128  # = 380
        self.observation_space = gym.spaces.Dict({
            "proprioception": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_proprio_dim * 2,), dtype=np.float32),
            "privileged": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_privileged_dim * 2,), dtype=np.float32),
            "target": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_target_dim * 2,), dtype=np.float32),
        })  # lib/rl/base.py's play_steps unconditionally does
        # `for k, v in self.obs["obs"].items()`, so the runtime obs dict
        # AND the declared space must both be genuine multi-key dicts --
        # these 3 keys feed lib.nn.features.SimpleFeatureFusion (real,
        # unmodified PhysGraph utility class) via matching Identity
        # extractors in the rl_train config

    def _to_tensor(self, x):
        return torch.tensor(np.array(x), device=self.device, dtype=torch.float32)

    def _joint_qpos_adr(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.mj_model.jnt_qposadr[jid])

    def _joint_dof_adr(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.mj_model.jnt_dofadr[jid])

    def _dof_names(self):
        return [
            "pro_sup_r", "deviation_r", "flexion_r",
            "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
            "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
            "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
            "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
            "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
        ]

    def _body_names(self):
        # CONFIRMED via direct compiled-model inspection (2026-09-07):
        # real body names have NO "_j1" suffix (that was a bug --
        # mj_name2id silently returned -1 for firstmc_r_j1/2proxph_r_j1/
        # etc, and -1 as an array index wrapped around, reading the SAME
        # wrong xpos for every affected body). Tips (THtip_r etc) are
        # real names but are SITES, not bodies -- no separate tip body
        # exists in the compiled model at all (the finger chain ends at
        # the distal phalanx); see body_ids/site_ids construction below.
        return [
            "lunate_r", "firstmc_r", "proximal_thumb_r", "distal_thumb_r", "THtip_r",
            "2proxph_r", "midph2_r", "distph2_r", "IFtip_r",
            "3proxph_r", "midph3_r", "distph3_r", "MFtip_r",
            "4proxph_r", "midph4_r", "distph4_r", "RFtip_r",
            "5proxph_r", "midph5_r", "distph5_r", "LFtip_r",
        ]

    def get_number_of_agents(self):
        return self.num_envs

    def reset_idx(self, env_ids):
        """Reset ONLY the specified environments (real PhysGraph pattern --
        dexhandmanip_bih.py's reset_idx/reset_done -- lets other envs keep
        running asynchronously rather than resetting the whole batch)."""
        if len(env_ids) == 0:
            return

        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        opt_wrist_pos0 = self.demo_opt_wrist_pos[0]
        opt_wrist_quat0 = self.demo_opt_wrist_quat[0]
        opt_dof_pos0 = self.demo_opt_dof_pos[0]
        src_pose0 = self.demo_src_obj_traj[0]
        dst_pose0 = self.demo_dst_obj_traj0

        qpos[env_ids, self.root_adr:self.root_adr + 3] = opt_wrist_pos0
        qpos[env_ids, self.root_adr + 3:self.root_adr + 7] = opt_wrist_quat0
        qpos[env_ids[:, None], self.dof_adrs] = opt_dof_pos0

        qpos[env_ids, self.src_adr:self.src_adr + 3] = src_pose0[:3, 3]
        qpos[env_ids, self.src_adr + 3:self.src_adr + 7] = rotmat_to_quat(src_pose0[:3, :3][None])[0]

        qpos[env_ids, self.dst_adr:self.dst_adr + 3] = dst_pose0[:3, 3]
        qpos[env_ids, self.dst_adr + 3:self.dst_adr + 7] = rotmat_to_quat(dst_pose0[:3, :3][None])[0]

        qvel[env_ids] = 0.0

        self.progress_buf[env_ids] = 0
        self.success_buf_[env_ids] = False
        self.failure_buf_[env_ids] = False
        self.error_buf_[env_ids] = False
        self.total_rewards[env_ids] = 0.0
        self.pos_error_integral[env_ids] = 0.0
        self.prev_pos_error[env_ids] = 0.0
        self.rot_error_integral[env_ids] = 0.0
        self.prev_rot_error[env_ids] = 0.0
        self._applied_force[env_ids] = 0.0
        self._applied_torque[env_ids] = 0.0

        mjw.forward(self.model, self.data)

    def reset(self):
        """Full reset -- all envs. Used once at the very start of training."""
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids)
        return self._compute_obs()

    def reset_done(self):
        """Real PhysGraph interface method (dexhandmanip_bih.py:1494),
        called directly by lib/rl/base.py's play_steps -- resets only the
        environments that finished their episode, returns (obs, done_ids)."""
        reset_buf = self.success_buf_ | self.failure_buf_
        done_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            # TEMP DEBUG: what progress_buf value are envs actually
            # resetting at during real training (vs our standalone test)?
            progresses = self.progress_buf[done_env_ids]
            print(f"[RESET DEBUG] resetting {len(done_env_ids)} envs, "
                  f"progress values: {progresses.tolist()[:10]}"
                  f"{'...' if len(progresses) > 10 else ''}")
            self.reset_idx(done_env_ids)
        return self._compute_obs(), done_env_ids

    def _ang_world(self, qpos, qvel, qadr, dadr):
        """World-frame angular velocity of a free-joint body: R(q) @ w_local."""
        return quat_rotate_vec(qpos[:, qadr + 3:qadr + 7], qvel[:, dadr + 3:dadr + 6])

    def _compute_obs(self):
        """Bimanual-SHAPED observation (per stream: [rh_real, lh_inert]
        concatenated), matching PhysGraph's own convention exactly (its
        transformer network internally slices obs['proprioception'] into
        r_prop/l_prop). LH hand fields are fully inert (zeros) -- no left
        hand rendered/actuated in this task -- but the LH OBJECT slot
        carries the REAL, static destination mug (position/rotation/BPS),
        since the right hand needs awareness of where to pour into even
        though the left hand does nothing. See project notes for the
        full field-by-field trace against PhysGraph's real
        compute_observations_side (dexhandmanip_bih.py) this is based on.

        Target tensor: full 380-dim-per-hand composition (not the earlier
        3-field placeholder), matching PhysGraph's real field list exactly
        (450 for them, 20 real joints vs their 27 changes joint-related
        dims): delta_wrist_pos/wrist_vel/delta_wrist_vel/wrist_quat/
        delta_wrist_quat/wrist_ang_vel/delta_wrist_ang_vel/delta_joints_pos/
        joints_vel/delta_joints_vel/delta_manip_obj_pos/manip_obj_vel/
        delta_manip_obj_vel/manip_obj_quat/delta_manip_obj_quat/
        manip_obj_ang_vel/delta_manip_obj_ang_vel/obj_to_joints/
        gt_tips_distance/bps. obj_to_joints and gt_tips_distance are
        genuinely undefined for the inert LH (no LH joints/fingers exist
        in this sim) -- zeroed, not approximated."""
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        dof_pos = qpos[:, self.dof_adrs]
        dof_vel = qvel[:, self.dof_veladrs]
        wrist_pos = qpos[:, self.root_adr:self.root_adr + 3]
        wrist_quat = qpos[:, self.root_adr + 3:self.root_adr + 7]
        wrist_linvel = qvel[:, self.root_dof_adr:self.root_dof_adr + 3]
        wrist_angvel = self._ang_world(qpos, qvel, self.root_adr, self.root_dof_adr)

        rh_proprio = torch.cat([
            dof_pos, torch.cos(dof_pos), torch.sin(dof_pos),
            torch.zeros_like(wrist_pos), wrist_quat, wrist_linvel, wrist_angvel,
        ], dim=-1)

        # populate cfrc_ext (per-body net contact force) -- NOT computed
        # by mjw.forward() alone, needs an explicit post-constraint pass
        # (matches real MuJoCo's mj_rnePostConstraint design -- confirmed
        # via direct testing 2026-09-08, verified physically sensible
        # Newton's-third-law-symmetric values for a known mug-mug contact)
        mjw.rne_postconstraint(self.model, self.data)
        cfrc_ext = wp.to_torch(self.data.cfrc_ext)
        n = self.num_envs

        src_pos = qpos[:, self.src_adr:self.src_adr + 3]
        src_quat = qpos[:, self.src_adr + 3:self.src_adr + 7]
        src_vel = qvel[:, self.src_dof_adr:self.src_dof_adr + 3]
        src_ang_vel = self._ang_world(qpos, qvel, self.src_adr, self.src_dof_adr)
        src_com_world = quat_rotate_vec(src_quat, self.src_com[None, :].expand(n, -1)) + src_pos
        src_com_rel = src_com_world - wrist_pos
        src_weight = self.src_weight.expand(n, 1) if self.src_weight.dim() == 0 else self.src_weight[None].expand(n, 1)

        # tip_force: real per-body net contact force (linear component,
        # cfrc_ext[..., 3:6]) + its own magnitude, matching PhysGraph's
        # exact construction (dexhandmanip_bih.py:1158-1173)
        tip_force_lin = cfrc_ext[:, self.tip_body_ids, 3:6]  # (n, 5, 3)
        tip_force_mag = torch.norm(tip_force_lin, dim=-1, keepdim=True)  # (n, 5, 1)
        tip_force = torch.cat([tip_force_lin, tip_force_mag], dim=-1).reshape(n, -1)  # (n, 20)

        rh_privileged = torch.cat([
            dof_vel, src_pos - wrist_pos, src_quat, src_vel, src_ang_vel,
            src_com_rel, src_weight.reshape(n, 1), tip_force,
        ], dim=-1)

        # === RH target: full real composition ===
        idx = self.progress_buf
        n = self.num_envs

        target_wrist_pos = self.demo_opt_wrist_pos[idx]
        target_wrist_quat = self.demo_opt_wrist_quat[idx]
        target_wrist_vel = self.demo_wrist_vel[idx]
        target_wrist_ang_vel = self.demo_wrist_ang_vel[idx]
        delta_wrist_pos = target_wrist_pos - wrist_pos
        delta_wrist_vel = target_wrist_vel - wrist_linvel
        delta_wrist_quat = quat_mul(wrist_quat, quat_conjugate(target_wrist_quat))
        delta_wrist_ang_vel = target_wrist_ang_vel - wrist_angvel

        target_joints_pos = self.demo_target_joints_pos[idx]
        target_joints_vel = self.demo_joints_vel[idx]
        current_joints_pos = self.current_joints_pos_
        current_joints_vel = self.current_joints_vel_
        delta_joints_pos = (target_joints_pos - current_joints_pos).reshape(n, -1)
        joints_vel_flat = target_joints_vel.reshape(n, -1)
        delta_joints_vel = (target_joints_vel - current_joints_vel).reshape(n, -1)

        target_obj_pos = self.demo_src_obj_traj[idx, :3, 3]
        target_obj_quat = rotmat_to_quat(self.demo_src_obj_traj[idx, :3, :3])
        target_obj_vel = self.demo_src_obj_vel[idx]
        target_obj_ang_vel = self.demo_src_obj_ang_vel[idx]
        current_obj_vel = qvel[:, self.src_dof_adr:self.src_dof_adr + 3]
        current_obj_ang_vel = self._ang_world(qpos, qvel, self.src_adr, self.src_dof_adr)
        delta_obj_pos = target_obj_pos - src_pos
        delta_obj_vel = target_obj_vel - current_obj_vel
        delta_obj_quat = quat_mul(src_quat, quat_conjugate(target_obj_quat))
        delta_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel

        all_joints_pos = torch.cat([wrist_pos[:, None, :], current_joints_pos], dim=1)  # (n, 21, 3)
        obj_to_joints = torch.norm(src_pos[:, None, :] - all_joints_pos, dim=-1).reshape(n, -1)

        tips_dist_idx = torch.clamp(idx, max=self.rh_demo["tips_distance"].shape[0] - 1)
        gt_tips_distance = self.rh_demo["tips_distance"].to(self.device)[tips_dist_idx]

        rh_target = torch.cat([
            delta_wrist_pos, target_wrist_vel, delta_wrist_vel,
            target_wrist_quat, delta_wrist_quat, target_wrist_ang_vel, delta_wrist_ang_vel,
            delta_joints_pos, joints_vel_flat, delta_joints_vel,
            delta_obj_pos, target_obj_vel, delta_obj_vel,
            target_obj_quat, delta_obj_quat, target_obj_ang_vel, delta_obj_ang_vel,
            obj_to_joints, gt_tips_distance, self.src_mug_bps[None, :].expand(n, -1),
        ], dim=-1)

        # === LH target: hand fields inert (zero), object fields REAL
        # (destination mug) -- see class docstring ===
        dst_pos = qpos[:, self.dst_adr:self.dst_adr + 3]
        dst_quat = qpos[:, self.dst_adr + 3:self.dst_adr + 7]
        dst_vel = qvel[:, self.dst_dof_adr:self.dst_dof_adr + 3]
        dst_ang_vel = self._ang_world(qpos, qvel, self.dst_adr, self.dst_dof_adr)
        target_dst_pos = self.lh_target_pos
        target_dst_quat = self.lh_target_quat
        delta_dst_pos = target_dst_pos - dst_pos
        delta_dst_vel = self.demo_dst_obj_vel[None, :] - dst_vel
        delta_dst_quat = quat_mul(dst_quat, quat_conjugate(target_dst_quat))
        delta_dst_ang_vel = self.demo_dst_obj_ang_vel[None, :] - dst_ang_vel

        _z3 = torch.zeros(n, 3, device=self.device)
        _z4 = torch.zeros(n, 4, device=self.device)
        _z60 = torch.zeros(n, 60, device=self.device)
        _z21 = torch.zeros(n, 21, device=self.device)
        _z5 = torch.zeros(n, 5, device=self.device)

        lh_target = torch.cat([
            _z3, _z3, _z3,             # wrist pos/vel/delta_vel (inert)
            _z4, _z4, _z3, _z3,        # wrist quat/delta_quat/ang_vel/delta_ang_vel (inert)
            _z60, _z60, _z60,          # joints pos/vel/delta_vel (inert -- no LH joints)
            delta_dst_pos, self.demo_dst_obj_vel[None, :].expand(n, -1), delta_dst_vel,
            target_dst_quat.expand(n, -1), delta_dst_quat,
            self.demo_dst_obj_ang_vel[None, :].expand(n, -1), delta_dst_ang_vel,
            _z21, _z5, self.dst_mug_bps[None, :].expand(n, -1),
        ], dim=-1)

        lh_proprio = torch.zeros_like(rh_proprio)
        # LH privileged: hand fields inert (zero, no LH dof), object
        # fields REAL (destination mug) -- same "hand inert, object real"
        # principle as the LH target tensor. obj_pos relative to RH's
        # own wrist (the only real, active wrist in this scene -- the
        # right hand is the one that needs spatial awareness of the
        # destination mug). tip_force zero (no LH fingers/contacts exist).
        dst_com_world = quat_rotate_vec(dst_quat, self.dst_com[None, :].expand(n, -1)) + dst_pos
        dst_com_rel = dst_com_world - wrist_pos
        dst_weight = self.dst_weight.expand(n, 1) if self.dst_weight.dim() == 0 else self.dst_weight[None].expand(n, 1)
        lh_tip_force = torch.zeros(n, 20, device=self.device)
        lh_privileged = torch.cat([
            torch.zeros(n, 23, device=self.device),  # dof_vel (inert -- no LH dof)
            dst_pos - wrist_pos, dst_quat, dst_vel, dst_ang_vel,
            dst_com_rel, dst_weight.reshape(n, 1), lh_tip_force,
        ], dim=-1)

        proprioception = torch.cat([rh_proprio, lh_proprio], dim=-1)
        privileged = torch.cat([rh_privileged, lh_privileged], dim=-1)
        target = torch.cat([rh_target, lh_target], dim=-1)

        return {"obs": {"proprioception": proprioception, "privileged": privileged, "target": target}}

    def _compute_reward(self):
        """Faithful (partial) port of PhysGraph's compute_imitation_reward
        (physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py). Same reward
        terms/weights and failed_execute/succeeded logic, EXCEPT:
        - tip_force/tips_distance/tip_contact_state (contact-force reward
          + its failure check) are OMITTED -- needs mujoco_warp's Contact
          structure, not yet explored/verified.
        - power/wrist_power OMITTED -- needs actuator-force fields not yet
          confirmed to exist.
        - joints_vel APPROXIMATED via finite-difference of body xpos
          across steps, not MuJoCo's real cvel (6D com-based convention,
          not yet verified for mujoco_warp).
        - scale_factor fixed at 1.0 (no tightening curriculum yet -- see
          project notes: this is deferred, not a phase-separated
          curriculum PhysGraph itself doesn't have either).
        """
        mjw.forward(self.model, self.data)  # ensure xpos/xquat are fresh

        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)
        xpos = wp.to_torch(self.data.xpos)  # (nworld, nbody, 3)

        cur_idx = self.progress_buf
        _tm = os.environ.get("MYOHAND_TIGHTEN_METHOD", "exp_decay")
        _tf = float(os.environ.get("MYOHAND_TIGHTEN_FACTOR", "0.7"))
        _ts = float(os.environ.get("MYOHAND_TIGHTEN_STEPS", "3200"))
        self._tight_step = getattr(self, "_tight_step", 0) + 1
        _ls = self._tight_step + int(os.environ.get("MYOHAND_TIGHTEN_OFFSET", "0"))
        if _tm == "None":
            scale_factor = 1.0
        elif _tm == "const":
            scale_factor = _tf
        elif _tm == "linear_decay":
            scale_factor = 1 - (1 - _tf) / _ts * min(_ls, _ts)
        elif _tm == "exp_decay":
            scale_factor = (np.e * 2) ** (-1 * _ls / _ts) * (1 - _tf) + _tf
        else:
            raise NotImplementedError(_tm)

        # === current sim state ===
        current_eef_pos = qpos[:, self.root_adr:self.root_adr + 3]
        current_eef_quat = qpos[:, self.root_adr + 3:self.root_adr + 7]
        current_eef_vel = qvel[:, self.root_dof_adr:self.root_dof_adr + 3]
        current_eef_ang_vel = self._ang_world(qpos, qvel, self.root_adr, self.root_dof_adr)

        # tips are sites (THtip_r etc, confirmed via direct model
        # inspection) -- read from site_xpos for those, xpos for the rest
        site_xpos = wp.to_torch(self.data.site_xpos)
        body_vals = xpos[:, self.body_ids[1:], :]
        site_vals = site_xpos[:, self.site_ids[1:].clip(min=0), :]
        joints_pos = torch.where(self.is_site_mask[None, :, None], site_vals, body_vals)
        if self.prev_body_xpos is None:
            joints_vel = torch.zeros_like(joints_pos)
        else:
            joints_vel = (joints_pos - self.prev_body_xpos) / self.dt
        self.current_joints_pos_ = joints_pos
        self.current_joints_vel_ = joints_vel  # store for reuse in _compute_obs()
        self.prev_body_xpos = joints_pos.clone()

        current_dof_vel = qvel[:, self.dof_veladrs]

        current_obj_pos = qpos[:, self.src_adr:self.src_adr + 3]
        current_obj_quat = qpos[:, self.src_adr + 3:self.src_adr + 7]
        current_obj_vel = qvel[:, self.src_dof_adr:self.src_dof_adr + 3]
        current_obj_ang_vel = self._ang_world(qpos, qvel, self.src_adr, self.src_dof_adr)

        # === fingertip contact-force reward (PhysGraph dexhandmanip_sh.py:1509-1517) ===
        mjw.rne_postconstraint(self.model, self.data)
        _cfrc = wp.to_torch(self.data.cfrc_ext)
        finger_tip_force = _cfrc[:, self.tip_body_ids, 3:6]  # (n, 5, 3), thumb..pinky
        if not hasattr(self, "_tips_dist_dev"):
            self._tips_dist_dev = self.rh_demo["tips_distance"].to(self.device)
        _td_idx = torch.clamp(cur_idx, max=self._tips_dist_dev.shape[0] - 1)
        finger_tip_distance = self._tips_dist_dev[_td_idx]  # (n, 5) human demo tip-to-mug distance
        contact_range = [0.02, 0.03]
        finger_tip_weight = torch.clamp(
            (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
        )
        finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]
        tip_force_sum = torch.norm(finger_tip_force_masked, dim=-1).sum(-1)
        reward_finger_tip_force = torch.exp(-1 * (1 / (tip_force_sum + 1e-5)))

        # contact history (PhysGraph tips_contact_history, length 3): any contact in the last 3 steps.
        # Reset to all-True for envs on their first step after reset (progress_buf was just incremented to 1).
        if not hasattr(self, "tips_contact_history"):
            self.tips_contact_history = torch.ones(self.num_envs, 3, 5, dtype=torch.bool, device=self.device)
        self.tips_contact_history[self.progress_buf <= 1] = True
        self.tips_contact_history = torch.cat(
            [self.tips_contact_history[:, 1:], (torch.norm(finger_tip_force, dim=-1) > 0)[:, None]], dim=1)
        # PhysGraph failure term (dexhandmanip_sh.py:1539): demo tip within 5 mm of the object but no contact
        # in the last 3 steps. Single-hand file only: PhysGraph's bih (bimanual) has no such term, so it is OFF by default (MYOHAND_CONTACT_FAIL=1 enables).
        if not hasattr(self, "use_contact_failure"):
            self.use_contact_failure = os.environ.get("MYOHAND_CONTACT_FAIL", "0") == "1"
        contact_fail = torch.any((finger_tip_distance < 0.005) & ~(self.tips_contact_history.any(1)), dim=-1)
        if not self.use_contact_failure:
            contact_fail = torch.zeros_like(contact_fail)

        # power terms (PhysGraph dexhandmanip_sh.py:809-825). Finger power: |actuator generalized force * dq|
        # (muscle-driven hand: qfrc_actuator plays the role of Isaac Gym's dof_force). Wrist power: PID force/torque
        # times the matching root velocity (both in the free joint's own convention: linear world, angular body).
        qfrc_act = wp.to_torch(self.data.qfrc_actuator)
        power = torch.abs(qfrc_act[:, self.dof_veladrs] * qvel[:, self.dof_veladrs]).sum(dim=-1)
        _wf = getattr(self, "_wrist_force", torch.zeros(self.num_envs, 3, device=self.device))
        _wt = getattr(self, "_wrist_torque", torch.zeros(self.num_envs, 3, device=self.device))
        wrist_power = (torch.abs((_wf * qvel[:, self.root_dof_adr:self.root_dof_adr + 3]).sum(dim=-1))
                       + torch.abs((_wt * qvel[:, self.root_dof_adr + 3:self.root_dof_adr + 6]).sum(dim=-1)))
        reward_power = torch.exp(-10 * power)
        reward_wrist_power = torch.exp(-2 * wrist_power)

        # === target state (current frame, NOT the observation's lookahead) ===
        target_eef_pos = self.demo_opt_wrist_pos[cur_idx]
        target_eef_quat = self.demo_opt_wrist_quat[cur_idx]
        target_joints_pos = self.demo_target_joints_pos[cur_idx]
        target_obj_pos = self.demo_src_obj_traj[cur_idx, :3, 3]
        target_obj_quat = rotmat_to_quat(self.demo_src_obj_traj[cur_idx, :3, :3])
        # velocities: finite-difference of the demo trajectory itself
        next_idx = torch.clamp(cur_idx + 1, max=self.seq_len - 1)
        target_eef_vel = (self.demo_opt_wrist_pos[next_idx] - target_eef_pos) / self.dt
        target_eef_ang_vel = self.demo_wrist_ang_vel[cur_idx]
        target_joints_vel = (self.demo_target_joints_pos[next_idx] - target_joints_pos) / self.dt
        target_obj_vel = (self.demo_src_obj_traj[next_idx, :3, 3] - target_obj_pos) / self.dt
        target_obj_ang_vel = self.demo_src_obj_ang_vel[cur_idx]

        # === diffs ===
        diff_eef_pos_dist = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        diff_eef_vel = target_eef_vel - current_eef_vel
        diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

        diff_joints_pos_dist = torch.norm(target_joints_pos - joints_pos, dim=-1)  # (num_envs, 20)

        def tip_dist(key):
            idx = [k - 1 for k in self.weight_idx[key]]
            return diff_joints_pos_dist[:, idx].mean(dim=-1)

        diff_thumb_tip_pos_dist = tip_dist("thumb_tip")
        diff_index_tip_pos_dist = tip_dist("index_tip")
        diff_middle_tip_pos_dist = tip_dist("middle_tip")
        diff_ring_tip_pos_dist = tip_dist("ring_tip")
        diff_pinky_tip_pos_dist = tip_dist("pinky_tip")
        diff_level_1_pos_dist = tip_dist("level_1_joints")
        diff_level_2_pos_dist = tip_dist("level_2_joints")

        diff_joints_vel = target_joints_vel - joints_vel

        reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
        reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
        reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
        reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
        reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
        reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
        reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
        reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

        reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
        reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
        reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

        diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
        diff_eef_rot_angle = quat_to_angle(diff_eef_rot)
        reward_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())
        # === fingertip orientation reward (pad vs back): distal-phalanx rotation vs the demo's, from FK over the demo ===
        _tr_w = float(os.environ.get("MYOHAND_TIPROT_W", "1.0"))
        _tr_k = float(os.environ.get("MYOHAND_TIPROT_K", "2.0"))
        if not hasattr(self, "_tip_q_tgt"):
            _d = mujoco.MjData(self.mj_model)
            _wp = self.demo_opt_wrist_pos.cpu().numpy().astype(np.float64)
            _wq = self.demo_opt_wrist_quat.cpu().numpy().astype(np.float64)
            _dq0 = self.demo_opt_dof_pos.cpu().numpy().astype(np.float64)
            _tb = np.asarray(self.tip_body_ids)
            _out = np.zeros((_wp.shape[0], len(_tb), 4))
            for _t in range(_wp.shape[0]):
                _d.qpos[:] = self.mj_model.qpos0
                _d.qpos[self.root_adr:self.root_adr + 3] = _wp[_t]
                _d.qpos[self.root_adr + 3:self.root_adr + 7] = _wq[_t]
                _d.qpos[np.asarray(self.dof_adrs)] = _dq0[_t]
                mujoco.mj_kinematics(self.mj_model, _d)
                _out[_t] = _d.xquat[_tb]
            self._tip_q_tgt = torch.tensor(_out, dtype=torch.float32, device=self.device)
        _tq = wp.to_torch(self.data.xquat)[:, self.tip_body_ids]
        _tt = self._tip_q_tgt[torch.clamp(cur_idx, max=self._tip_q_tgt.shape[0] - 1)]
        _dqr = quat_mul(_tt.reshape(-1, 4), quat_conjugate(_tq.reshape(-1, 4)))
        tip_rot_angle = (2 * torch.acos(_dqr[:, 0].abs().clamp(max=1.0))).reshape(-1, len(self.tip_body_ids))
        reward_tip_rot = torch.exp(-_tr_k * tip_rot_angle).mean(-1)
        _jq_w = float(os.environ.get("MYOHAND_JQ_W", "1.0"))
        _jq_lam = float(os.environ.get("MYOHAND_JQ_LAMBDA", "10.0"))
        _jq_lamw = float(os.environ.get("MYOHAND_JQ_LAMBDA_WIDE", "2.0"))
        _lim_w = float(os.environ.get("MYOHAND_LIM_W", "2.0"))
        _lim_cap = float(os.environ.get("MYOHAND_LIM_CAP", "0.75"))
        _ffs_w = float(os.environ.get("MYOHAND_FFSP_W", "2.0"))
        _ffs_cap = float(os.environ.get("MYOHAND_FFSP_CAP_MM", "30")) / 1000.0
        if not hasattr(self, "_jq_M"):
            import mujoco as _mj
            _m = self.mj_model
            _qadr2j = {int(_m.jnt_qposadr[j]): j for j in range(_m.njnt)}
            _adrs = [int(a) for a in self.dof_adrs]
            _names = [_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_JOINT, _qadr2j[a]) or "" for a in _adrs]
            def _grp(n):  # 0 wrist, 1 thumb, 2 index, 3 middle, 4 ring, 5 pinky
                import re as _re
                if any(k in n for k in ("cmc", "mp_flexion", "ip_flexion")):
                    return 1
                _d = _re.search(r"[2-5]", n)
                return int(_d.group()) if _d else 0
            _n = len(_names)
            _M = torch.zeros(_n, 6, device=self.device)
            _dz = torch.zeros(_n, device=self.device)
            _lo = torch.full((_n,), -1e9, device=self.device)
            _hi = torch.full((_n,), 1e9, device=self.device)
            for i, n in enumerate(_names):
                g = _grp(n)
                _M[i, g] = 1.0
                _dz[i] = (2.0 if (g >= 2 and "abduction" in n) else 6.0) * 3.14159265 / 180.0
                j = _qadr2j[_adrs[i]]
                if _m.jnt_limited[j]:
                    _lo[i] = float(_m.jnt_range[j, 0])
                    _hi[i] = float(_m.jnt_range[j, 1])
            self._jq_M = _M / _M.sum(0, keepdim=True).clamp(min=1.0)
            self._jq_dz = _dz
            self._lim_lo, self._lim_hi = _lo, _hi
            self._jq_gw = torch.tensor(
                [float(os.environ.get("MYOHAND_JQ_W_WRIST", "0.3")), float(os.environ.get("MYOHAND_JQ_W_THUMB", "1.0"))]
                + [float(os.environ.get("MYOHAND_JQ_W_FINGER", "0.5"))] * 4, device=self.device)
            self._ffs_margin = [float(x) / 1000.0 for x in os.environ.get("MYOHAND_FFSP_MARGIN_MM", "14,12.5,12.5").split(",")]
            self._ffs_dfrac = float(os.environ.get("MYOHAND_FFSP_DEMO_FRAC", "0.7"))
            self._ffs_gfrac = float(os.environ.get("MYOHAND_FFSP_GAP_FRAC", "0.5"))
            print("[jq] group sizes", _M.sum(0).tolist(), "| abduction dofs", [n for n in _names if "abduction" in n], flush=True)
        _q_cur = wp.to_torch(self.data.qpos)[:, self.dof_adrs]
        _q_tgt = self.demo_opt_dof_pos[torch.clamp(cur_idx, max=self.demo_opt_dof_pos.shape[0] - 1)]
        _jq_excess = torch.clamp((_q_cur - _q_tgt).abs() - self._jq_dz, min=0.0)
        _jq_e = _jq_excess @ self._jq_M
        reward_joint_q = ((0.5 * torch.exp(-_jq_lam * _jq_e) + 0.5 * torch.exp(-_jq_lamw * _jq_e)) * self._jq_gw).sum(-1)
        _over = torch.clamp(self._lim_lo - _q_cur, min=0.0) + torch.clamp(_q_cur - self._lim_hi, min=0.0)
        reward_limit = 1.0 - torch.clamp(_over.sum(-1) / _lim_cap, 0.0, 1.0)
        _J = joints_pos
        _T = target_joints_pos
        _short = torch.zeros(self.num_envs, device=self.device)
        for _a in (1, 2, 3):
            for _l1 in (1, 2, 3):
                for _l2 in (1, 2, 3):
                    _d = torch.norm(_J[:, 4 * _a + _l1] - _J[:, 4 * (_a + 1) + _l2], dim=-1)
                    _dd = torch.norm(_T[:, 4 * _a + _l1] - _T[:, 4 * (_a + 1) + _l2], dim=-1)
                    _mg = torch.clamp(self._ffs_dfrac * _dd, max=min(self._ffs_margin[_l1 - 1], self._ffs_margin[_l2 - 1]))
                    _short = _short + torch.clamp(_mg - _d, min=0.0)
        _uc = _J[:, 16] - _J[:, 4]
        _uc = _uc / _uc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _ut = _T[:, 16] - _T[:, 4]
        _ut = _ut / _ut.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        for _a in (1, 2, 3):
            for _l in (1, 2, 3):
                _g = ((_J[:, 4 * (_a + 1) + _l] - _J[:, 4 * _a + _l]) * _uc).sum(-1)
                _gd = ((_T[:, 4 * (_a + 1) + _l] - _T[:, 4 * _a + _l]) * _ut).sum(-1)
                _short = _short + torch.clamp(self._ffs_gfrac * _gd - _g, min=0.0)
        reward_ffspace = 1.0 - torch.clamp(_short / _ffs_cap, 0.0, 1.0)

        diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
        reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

        diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
        diff_obj_rot_angle = quat_to_angle(diff_obj_rot)
        reward_obj_rot = torch.exp(-3 * diff_obj_rot_angle.abs())

        diff_obj_vel = target_obj_vel - current_obj_vel
        reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))
        diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
        reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

        error_buf = (
            (torch.norm(current_eef_vel, dim=-1) > 100)
            | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
            | (torch.abs(current_dof_vel).mean(-1) > 200)
            | (torch.norm(current_obj_vel, dim=-1) > 100)
            | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
        )
        self._debug_diffs = {
            "obj_pos": diff_obj_pos_dist, "thumb": diff_thumb_tip_pos_dist,
            "index": diff_index_tip_pos_dist, "middle": diff_middle_tip_pos_dist,
            "pinky": diff_pinky_tip_pos_dist, "ring": diff_ring_tip_pos_dist,
            "level_1": diff_level_1_pos_dist, "level_2": diff_level_2_pos_dist,
            "obj_rot_deg": diff_obj_rot_angle.abs() / np.pi * 180,
            "tip_force_sum": tip_force_sum,
            "tip_rot_deg": tip_rot_angle.mean(-1) * 180.0 / np.pi,
            "contact_fail": contact_fail.float(),
            "power": power,
            "wrist_power": wrist_power,
        }
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor ** 3)
                | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
                | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
                | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
                | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
                | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
                | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
                | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor ** 3)
                | contact_fail
            )
            & (self.progress_buf >= 8)
        ) | error_buf

        reward = (
            0.1 * reward_eef_pos + 0.6 * reward_eef_rot
            + 0.9 * reward_thumb_tip_pos + 0.8 * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos + 0.6 * reward_pinky_tip_pos
            + 0.6 * reward_ring_tip_pos + 0.5 * reward_level_1_pos
            + 0.3 * reward_level_2_pos + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot + 0.1 * reward_eef_vel
            + 0.05 * reward_eef_ang_vel + 0.1 * reward_joints_vel
            + 0.1 * reward_obj_vel + 0.1 * reward_obj_ang_vel
            + float(os.environ.get("MYOHAND_FORCE_REWARD_W", "0.0")) * reward_finger_tip_force
            + _tr_w * reward_tip_rot
            + _jq_w * reward_joint_q + _lim_w * reward_limit + _ffs_w * reward_ffspace
            + 0.5 * reward_power + 0.5 * reward_wrist_power
        )

        succeeded = (self.progress_buf + 1 + 3 >= self.max_episode_length) & ~failed_execute
        dones = (succeeded | failed_execute).float()

        self.success_buf_ = succeeded
        self.failure_buf_ = failed_execute
        self.error_buf_ = error_buf
        self.total_rewards += reward

        # infos dict -- lib/rl/base.py's play_steps reads these exact keys
        # (real PhysGraph code, confirmed by tracing what it accesses):
        # diff_metrics (per-frame tracking-error diagnostics), reward_dict
        # (individual reward term breakdown), time_outs (episode ended by
        # reaching the length limit, for PPO value-bootstrapping -- same
        # condition as `succeeded` here, since our fixed-length task makes
        # them coincide exactly), total_rewards/total_steps (cumulative
        # episode trackers), error_masks (same as error_buf).
        # diff_ft (fingertip-force tracking) is a zero placeholder -- the
        # real contact-force reward/diff is a deferred future piece (see
        # project notes), not yet computed.
        infos = {
            "diff_metrics": {
                "diff_obj_pos": diff_obj_pos_dist,
                "diff_obj_rot": diff_obj_rot_angle,
                "diff_joints": diff_joints_pos_dist.mean(dim=-1),
                "diff_ft": torch.zeros_like(diff_obj_pos_dist),
            },
            "reward_dict": {
                "reward_eef_pos": reward_eef_pos,
                "reward_eef_rot": reward_eef_rot,
                "reward_thumb_tip_pos": reward_thumb_tip_pos,
                "reward_index_tip_pos": reward_index_tip_pos,
                "reward_middle_tip_pos": reward_middle_tip_pos,
                "reward_pinky_tip_pos": reward_pinky_tip_pos,
                "reward_ring_tip_pos": reward_ring_tip_pos,
                "reward_level_1_pos": reward_level_1_pos,
                "reward_level_2_pos": reward_level_2_pos,
                "reward_obj_pos": reward_obj_pos,
                "reward_obj_rot": reward_obj_rot,
                "reward_eef_vel": reward_eef_vel,
                "reward_eef_ang_vel": reward_eef_ang_vel,
                "reward_joints_vel": reward_joints_vel,
                "reward_obj_vel": reward_obj_vel,
                "reward_obj_ang_vel": reward_obj_ang_vel,
                "reward_finger_tip_force": reward_finger_tip_force,
                "reward_tip_rot": reward_tip_rot,
                "reward_joint_q": reward_joint_q,
                "reward_limit": reward_limit,
                "reward_ffspace": reward_ffspace,
                "reward_power": reward_power,
                "reward_wrist_power": reward_wrist_power,
            },
            "time_outs": succeeded,
            "total_rewards": self.total_rewards,
            "total_steps": self.progress_buf,
            # POLARITY FIX (2026-09-08): rl_games' own torch_ext.normalization_
            # with_masks / get_mean_var_with_masks expects masks=1 to mean
            # "valid, include this sample" (values*masks summed, divided by
            # masks.sum()) -- we were passing error_buf directly (1=errored),
            # the OPPOSITE polarity. Since most envs are healthy at any given
            # step, that meant masks.sum()~=0 almost every single minibatch,
            # a genuine 0/0 in rl_games' own get_mean_var_with_masks (no
            # epsilon guard there at all) -- this, not policy collapse, was
            # the real root cause of a_loss/c_loss/kl showing NaN from the
            # very first recorded TensorBoard step, confirmed via a direct
            # [NAN DEBUG] instrumentation of lib/rl/base.py's train_epoch().
            "error_masks": ~error_buf,
        }

        return reward, dones, infos

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)

        # direct-pose wrist action -> PID-computed force (matching
        # PhysGraph's usePIDControl mechanism, ArtiMANO reference gains)
        pos_error = action[:, 0:3]  # [-1,1], interpreted as a position error signal
        self.pos_error_integral += pos_error * self.dt
        self.pos_error_integral.clamp_(-1, 1)
        pos_derivative = (pos_error - self.prev_pos_error) / self.dt
        wrist_force = self.Kp_pos * pos_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative
        self.prev_pos_error = pos_error

        rot_error_6d = action[:, 3:9]
        rot_error = rot_error_6d[:, :3]  # simplified: first 3 components as an axis-angle-style error
        self.rot_error_integral += rot_error * self.dt
        self.rot_error_integral.clamp_(-1, 1)
        rot_derivative = (rot_error - self.prev_rot_error) / self.dt
        wrist_torque = self.Kp_rot * rot_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative
        self.prev_rot_error = rot_error

        _a = self.wrist_ema
        wrist_force = _a * wrist_force + (1.0 - _a) * self._applied_force
        wrist_torque = _a * wrist_torque + (1.0 - _a) * self._applied_torque
        self._applied_force = wrist_force
        self._applied_torque = wrist_torque
        self._wrist_force = wrist_force
        self._wrist_torque = wrist_torque

        # muscle activations must be in [0,1] (real ctrlrange) -- map from the
        # policy's standard [-1,1] action range
        muscle_activations = (action[:, 9:] + 1.0) / 2.0

        qfrc = wp.to_torch(self.data.qfrc_applied)
        qfrc.zero_()
        qfrc[:, self.root_dof_adr:self.root_dof_adr + 3] = wrist_force
        qfrc[:, self.root_dof_adr + 3:self.root_dof_adr + 6] = wrist_torque

        ctrl = wp.to_torch(self.data.ctrl)
        ctrl[:, :] = muscle_activations

        # Physics substeps at 1/120 (matching Runfa's confirmed ArtiMANO
        # config: implicitfast integrator, elliptic cone, impratio=10).
        # Action/qfrc/ctrl set once above, held constant across both
        # substeps; progress_buf/reward/obs still advance once per outer
        # call, matching the demo's own 1/60 frame rate.
        _qv = wp.to_torch(self.data.qvel)
        _ra, _rd = self.root_dof_adr, self.root_dof_adr
        _qf_base = qfrc[:, _rd:_rd + 6].clone()
        for _ in range(self.physics_substeps):
            if self._root_damp is not None:
                qfrc[:, _rd:_rd + 3] = _qf_base[:, 0:3] - self._root_damp_lin * _qv[:, _rd:_rd + 3]
                qfrc[:, _rd + 3:_rd + 6] = _qf_base[:, 3:6] - _qv[:, _rd + 3:_rd + 6] @ self._root_damp_ang
            mjw.step(self.model, self.data)

        self.progress_buf += 1
        rewards, dones, infos = self._compute_reward()
        obs = self._compute_obs()
        return obs, rewards, dones, infos

    @property
    def success_buf(self):
        return self.success_buf_

    @property
    def failure_buf(self):
        return self.failure_buf_

    @property
    def error_buf(self):
        return self.error_buf_
