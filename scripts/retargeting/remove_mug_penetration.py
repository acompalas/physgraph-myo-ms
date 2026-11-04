"""Post-process the retargeted trajectory so no hand collision geom penetrates
the mug. Per frame (sequential, warm-started): least-squares correction of
wrist pose + 23 dofs, staying close to the original joint positions."""
import os, sys, copy, pickle, argparse, time
import numpy as np, torch, mujoco
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "."); sys.path.insert(0, "scripts/retargeting")
import envs.mjwarp_myohand_env as E
from myohand_def import MyoHandR

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--write", action="store_true")
args = ap.parse_args()

DEMO = "assets/retargeted/1292e_training_demo.pkl"
OUT = os.environ.get("MYOHAND_DEMO_OUT", "assets/retargeted/1292e_training_demo_nopen.pkl")
MARGIN, TOUCH = 0.0005, 0.002
TOUCH_W = float(os.environ.get("MYOHAND_TOUCH_W", "0.5"))
TGAP = float(os.environ.get("MYOHAND_TOUCH_GAP_MM", "2.0")) / 1000
TGAP_P = float(os.environ.get("MYOHAND_TOUCH_GAP_PINKY_MM", str(TGAP * 1000))) / 1000
PAD_W = float(os.environ.get("MYOHAND_PAD_W", "0"))
SLIP_W = float(os.environ.get("MYOHAND_SLIP_W", "0"))
SLIP_SIGMA = float(os.environ.get("MYOHAND_SLIP_SIGMA", "6"))
JPOS_S = float(os.environ.get("MYOHAND_JPOS_SCALE_MM", "5")) / 1000
THUMB_S = float(os.environ.get("MYOHAND_THUMB_SCALE_MM", "5")) / 1000
WRIST_S = float(os.environ.get("MYOHAND_WRIST_SCALE_MM", "20")) / 1000
PAD_PINKY = os.environ.get("MYOHAND_PAD_PINKY", "1") == "1"
PINKY_MAX = float(os.environ.get("MYOHAND_PINKY_MAX_MM", "1000")) / 1000
PAD_GATE = float(os.environ.get("MYOHAND_PAD_GATE_MM", "30")) / 1000
PAD_START = int(os.environ.get("MYOHAND_PAD_START", "0"))
PAD_RAMP = max(1, int(os.environ.get("MYOHAND_PAD_RAMP", "1")))
dt = 1.0 / 60

with open(DEMO, "rb") as f:
    demo = pickle.load(f)

def find_rh(o, path="demo"):
    if isinstance(o, dict):
        if "opt_dof_pos" in o:
            return o, path
        for k, v in o.items():
            r = find_rh(v, path + "[%r]" % (k,))
            if r is not None:
                return r
    return None

rh, rh_path = find_rh(demo)
print("hand data found at", rh_path)

def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

env = E.MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)
C = env._axis_C.cpu().numpy().astype(np.float64)
dexhand = MyoHandR()
T = env.demo_opt_wrist_pos.shape[0]
wp0 = env.demo_opt_wrist_pos.cpu().numpy().astype(np.float64)
wq0 = env.demo_opt_wrist_quat.cpu().numpy().astype(np.float64)   # wxyz, axis-corrected
q0 = env.demo_opt_dof_pos.cpu().numpy().astype(np.float64)
mug_p = env.demo_src_obj_traj[:, :3, 3].cpu().numpy().astype(np.float64)
mug_q = E.rotmat_to_quat(env.demo_src_obj_traj[:, :3, :3]).cpu().numpy().astype(np.float64)
human = to_np(rh["tips_distance"]).astype(np.float64)             # (T,5)

# --- sanity: raw <-> corrected conversion must roundtrip on the ORIGINAL data
R_c0 = Rot.from_quat(wq0[:, [1, 2, 3, 0]]).as_matrix()
R_raw0 = np.matmul(C.T[None], R_c0)
e_rot = np.abs(Rot.from_rotvec(to_np(rh["opt_wrist_rot"]).astype(np.float64)).as_matrix() - R_raw0).max()
e_pos = np.abs((C.T @ wp0.T).T - to_np(rh["opt_wrist_pos"])).max()
print("roundtrip check: rot err %.2e, pos err %.2e" % (e_rot, e_pos))
assert e_rot < 1e-4 and e_pos < 1e-4, "axis-correction roundtrip failed; refusing to continue"

def is_col(g):
    return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0

src_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")])
root_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_r")])
mug_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == src_body and is_col(g)]
hand_bodies = {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if is_col(g) and m.body_rootid[b] == root_body:
        hand_bodies.setdefault(b, []).append(g)
distal_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in dexhand.contact_body_names]
print("mug geoms %d | hand bodies with collision %d | distal bodies %s" % (len(mug_geoms), len(hand_bodies), distal_ids))
assert all(b in hand_bodies for b in distal_ids)

dof_adrs = np.asarray(env.dof_adrs)
_dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(np.where(m.jnt_qposadr == a_)[0][0])) for a_ in dof_adrs]
open("/tmp/dof_names.txt", "w").write("\n".join(_dof_names))
lo, hi = [], []
for adr in dof_adrs:
    jid = int(np.where(m.jnt_qposadr == adr)[0][0])
    l, h = (m.jnt_range[jid] if m.jnt_limited[jid] else (-np.inf, np.inf))
    lo.append(l); hi.append(h)
lo, hi = np.array(lo), np.array(hi)
_WB = float(os.environ.get("MYOHAND_WRIST_BOUND_X", "1"))
blo = np.r_[[-0.05 * _WB] * 3, [-0.3 * _WB] * 3, lo]
bhi = np.r_[[0.05 * _WB] * 3, [0.3 * _WB] * 3, hi]
scale_smooth = np.r_[[0.005] * 3, [0.02] * 3, [0.05] * len(dof_adrs)]

jl = []
for b in dexhand.body_names[1:]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, b) if bid < 0 else -1
    assert bid >= 0 or sid >= 0, b
    jl.append((bid, sid))

def joints_xyz():
    return np.array([d.xpos[b] if b >= 0 else d.site_xpos[s] for b, s in jl])

def place(p, quat, q, t):
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = p
    d.qpos[env.root_adr + 3:env.root_adr + 7] = quat
    d.qpos[dof_adrs] = q
    d.qpos[env.src_adr:env.src_adr + 3] = mug_p[t]
    d.qpos[env.src_adr + 3:env.src_adr + 7] = mug_q[t]
    mujoco.mj_kinematics(m, d)

fromto = np.zeros(6)
def pair_dist(g, h, dmax):
    return mujoco.mj_geomDistance(m, d, g, h, dmax, fromto)

def all_body_min(dmax=0.05):
    out = {}
    for b, gs in hand_bodies.items():
        out[b] = min([dmax] + [pair_dist(g, h, dmax) for g in gs for h in mug_geoms])
    return out

tip_sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r")]
d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d)
_bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
_w0 = d.xpos[_bid(dexhand.body_names[0])].copy()
_n0 = np.cross(d.xpos[_bid("2proxph_r")] - _w0, d.xpos[_bid("5proxph_r")] - _w0); _n0 /= np.linalg.norm(_n0)
pad_local = [d.site_xmat[s_].reshape(3, 3).T @ _n0 for s_ in tip_sites]   # fingertip underside = palm normal at the flat reference pose
print("pad vector in tip frame (index):", np.round(pad_local[1], 2))
_tight = {"lunate_r", "firstmc_r", "proximal_thumb_r", "distal_thumb_r", "THtip_r"}
JS_VEC = np.repeat(np.array([THUMB_S if b_ in _tight else JPOS_S for b_ in dexhand.body_names[1:]]), 3)
print("hold scale mm: thumb+lunate %.0f, finger chains %.0f | wrist %.0f | pinky in pad term: %s" % (THUMB_S * 1000, JPOS_S * 1000, WRIST_S * 1000, PAD_PINKY))
MR = np.stack([Rot.from_quat(mug_q[t][[1, 2, 3, 0]]).as_matrix() for t in range(T)])

def tip_state(t):
    out = np.zeros((5, 3)); cv = np.zeros(5)
    for i, s_ in enumerate(tip_sites):
        loc = MR[t].T @ (d.site_xpos[s_] - mug_p[t])
        th = np.arctan2(loc[1], loc[0])
        out[i] = (th, np.hypot(loc[0], loc[1]), loc[2])
        inward = MR[t] @ np.array([-np.cos(th), -np.sin(th), 0.0])
        cv[i] = (d.site_xmat[s_].reshape(3, 3) @ pad_local[i]) @ inward
    return out, cv

th0 = np.zeros((T, 5)); r0 = np.zeros((T, 5)); z0 = np.zeros((T, 5)); cos0 = np.zeros((T, 5))
for t_ in range(T):
    place(wp0[t_], wq0[t_], q0[t_], t_)
    st_, cv_ = tip_state(t_)
    th0[t_], r0[t_], z0[t_], cos0[t_] = st_[:, 0], st_[:, 1], st_[:, 2], cv_
th_s = gaussian_filter1d(np.unwrap(th0, axis=0), SLIP_SIGMA, axis=0, mode="nearest")
z_s = gaussian_filter1d(z0, SLIP_SIGMA, axis=0, mode="nearest")
ARC = np.radians(float(os.environ.get("MYOHAND_ARC_DEG", "0")))
ARC_F = [int(v_) for v_ in os.environ.get("MYOHAND_ARC_FINGERS", "1,2,3,4").split(",")]
WROT_S = np.radians(float(os.environ.get("MYOHAND_WROT_SCALE_DEG", "5.73")))
BARRIER_W = float(os.environ.get("MYOHAND_BARRIER_W", "1"))
_thw0 = np.array([np.arctan2(*(MR[t_].T @ (wp0[t_] - mug_p[t_]))[[1, 0]]) for t_ in range(T)])
_phi0 = np.angle(np.exp(1j * (th_s - _thw0[:, None])))
for _i in ARC_F:
    _near = human[:, _i] < 0.02
    _med = float(np.median(_phi0[_near, _i])) if _near.any() else 0.0
    _sg = 1.0 if _med >= 0 else -1.0
    _tgt = float(os.environ.get("MYOHAND_ARC_TARGET_DEG", "0"))
    _off = max(0.0, np.radians(_tgt) - abs(_med)) if _tgt > 0 else ARC
    if _i == 4:
        _off *= float(os.environ.get("MYOHAND_ARC_PINKY_SCALE", "1"))
    th_s[:, _i] += _sg * _off * np.clip((np.arange(T) - int(os.environ.get("MYOHAND_ARC_START", "40"))) / max(1, int(os.environ.get("MYOHAND_ARC_RAMP", "40"))), 0.0, 1.0)
    print("arc offset finger %d: %+.0f deg | median angle from the wrist direction %.0f -> %.0f deg" % (_i, np.degrees(_sg * _off), np.degrees(_med), np.degrees(_med + _sg * _off)))

end = T if args.end is None else min(args.end, T)
frames = list(range(args.start, end))
sol_x = {}; sol_joints = {}
before = np.zeros((T, 5)); after = np.zeros((T, 5))
thA = np.zeros((T, 5)); rA = np.zeros((T, 5)); zA = np.zeros((T, 5)); cosA = np.zeros((T, 5))
worst_after = np.zeros(T); jdev = np.zeros(T)
cprev = np.zeros(6 + len(dof_adrs))
t_start = time.time()
import re as _re, atexit as _atexit
SYN_W = float(os.environ.get("MYOHAND_SYN_W", "0"))
SYN_START = int(os.environ.get("MYOHAND_SYN_START", "60"))
SYN_RAMP = max(1, int(os.environ.get("MYOHAND_SYN_RAMP", "30")))
SYN_MIX = float(os.environ.get("MYOHAND_SYN_MIX", "1.0"))
_pi = [_dof_names.index(n_) for n_ in ("mcp5_flexion_r", "pm5_flexion_r", "md5_flexion_r")]
_ri = [_dof_names.index(n_) for n_ in ("mcp4_flexion_r", "pm4_flexion_r", "md4_flexion_r")]
FF_W = float(os.environ.get("MYOHAND_FF_W", "0"))
FF_M = float(os.environ.get("MYOHAND_FF_MARGIN_MM", "2")) / 1000
fing_geoms = {}
for _b, _gs in hand_bodies.items():
    _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, _b)
    _mm = _re.search(r"[2-5]", _nm)
    if "ph" in _nm and _mm:
        fing_geoms.setdefault(int(_mm.group()), []).extend(_gs)
print("finger groups:", {k: len(v) for k, v in sorted(fing_geoms.items())})
FF_PAIRS = [(2, 3), (3, 4), (4, 5)]
def ff_cands(dmax=0.05):
    return [[(g, h) for g in fing_geoms[a] for h in fing_geoms[b] if pair_dist(g, h, dmax) < dmax] for a, b in FF_PAIRS]
def ff_min(dmax=0.05):
    return [min([dmax] + [pair_dist(g, h, dmax) for g in fing_geoms[a] for h in fing_geoms[b]]) for a, b in FF_PAIRS]
ffb, ffa = {}, {}
def _ff_report():
    ts = sorted(set(ffb) & set(ffa))
    if not ts:
        return
    B = np.array([ffb[t] for t in ts]) * 1000
    A = np.array([ffa[t] for t in ts]) * 1000
    print("=== finger clearance over %d frames (mm): min before/after | frames <0 before/after | frames <1 before/after" % len(ts))
    for k, (a, b) in enumerate(FF_PAIRS):
        print("fingers %d-%d  %6.2f / %6.2f | %3d / %3d | %3d / %3d" % (a, b, B[:, k].min(), A[:, k].min(), (B[:, k] < 0).sum(), (A[:, k] < 0).sum(), (B[:, k] < 1).sum(), (A[:, k] < 1).sum()))
_atexit.register(_ff_report)

import pickle as _pk
RAW_CACHE = os.environ.get("MYOHAND_RAW_CACHE", "")
_cache_hit = bool(RAW_CACHE) and os.path.exists(RAW_CACHE)
_CACHE_ARR = ("before", "after", "thA", "rA", "zA", "cosA", "worst_after", "jdev")
if _cache_hit:
    _C = _pk.load(open(RAW_CACHE, "rb"))
    sol_x.update(_C["sol_x"]); sol_joints.update(_C["sol_joints"]); ffb.update(_C["ffb"]); ffa.update(_C["ffa"])
    for _n in _CACHE_ARR:
        globals()[_n][:] = _C[_n]
    print("loaded raw solution cache", RAW_CACHE)
for t in (frames if not _cache_hit else []):
    R0 = Rot.from_quat(wq0[t][[1, 2, 3, 0]])
    x0 = np.r_[np.zeros(6), np.clip(q0[t], lo + 1e-9, hi - 1e-9)]
    place(wp0[t], wq0[t], q0[t], t)
    jorig = joints_xyz()
    bm = all_body_min()
    before[t] = [bm[b] for b in distal_ids]
    ff0_t = ff_min()
    ffb[t] = ff0_t
    ffc_t = ff_cands() if FF_W > 0 else None
    cand = {}
    for b, gs in hand_bodies.items():
        pr = [(g, h) for g in gs for h in mug_geoms if pair_dist(g, h, 0.03) < 0.03]
        if pr:
            cand[b] = pr
    touch = [i for i in range(5) if human[t, i] < 0.03 and distal_ids[i] in cand and not (i == 4 and before[t, 4] > PINKY_MAX)]
    padset = [i for i in touch if i in ((1, 2, 3, 4) if PAD_PINKY else (1, 2, 3)) and human[t, i] < PAD_GATE and t >= PAD_START]
    slipset = list(touch)

    def resid(x):
        p = wp0[t] + x[0:3]
        quat = (Rot.from_rotvec(x[3:6]) * R0).as_quat()[[3, 0, 1, 2]]
        place(p, quat, x[6:], t)
        jx = joints_xyz()
        r = [(jx - jorig).ravel() / JS_VEC,
             (x[6:] - q0[t]) / 0.2,
             x[0:3] / WRIST_S, x[3:6] / WROT_S]
        dmin = {b: min(0.03, min(pair_dist(g, h, 0.03) for g, h in pr)) for b, pr in cand.items()}
        r.append(BARRIER_W * np.array([max(0.0, MARGIN - v) / MARGIN for v in dmin.values()]))
        r.append(np.array([TOUCH_W * max(0.0, dmin[distal_ids[i]] - (TGAP_P if i == 4 else TGAP)) / TOUCH for i in touch]))
        st_, cv_ = tip_state(t)
        if PAD_W > 0 and padset:
            r.append(PAD_W * min(1.0, max(0.0, (t - PAD_START) / PAD_RAMP)) * np.array([1.0 - cv_[i] for i in padset]))
        if SLIP_W > 0 and slipset:
            r.append(SLIP_W * np.concatenate([[st_[i, 1] * np.angle(np.exp(1j * (st_[i, 0] - th_s[t, i]))) / 0.002, (st_[i, 2] - z_s[t, i]) / 0.002] for i in slipset]))
        if FF_W > 0:
            _dk = [min([0.05] + [pair_dist(g, h, 0.05) for g, h in ffc_t[k]]) for k in range(len(FF_PAIRS))]
            _tg = [max(0.0, min(FF_M, ff0_t[k])) for k in range(len(FF_PAIRS))]
            r.append(FF_W * np.array([max(0.0, _tg[k] - _dk[k]) / FF_M for k in range(len(FF_PAIRS))]))
        if SYN_W > 0 and t >= SYN_START:
            _ws = min(1.0, (t - SYN_START) / SYN_RAMP)
            _tgt = (1 - SYN_MIX) * q0[t][_pi] + SYN_MIX * q0[t][_ri]
            r.append(SYN_W * _ws * (x[6:][_pi] - _tgt) / 0.1)
        r.append(0.5 * ((x - x0) - cprev) / scale_smooth)
        return np.concatenate(r)

    xs = np.r_[[0.01] * 3, [0.05] * 3, [0.1] * len(dof_adrs)]
    ds = np.r_[[2e-3] * 3, [1e-2] * 3, [1e-2] * len(dof_adrs)]
    best = None
    for init in (np.clip(x0 + cprev, blo, bhi), x0):
        r_ = least_squares(resid, init, bounds=(blo, bhi), jac='3-point', diff_step=ds, x_scale=xs,
                           max_nfev=150, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if best is None or r_.cost < best.cost:
            best = r_
        if best.cost < 80:
            break
    res = best
    x = res.x
    p = wp0[t] + x[0:3]
    quat = (Rot.from_rotvec(x[3:6]) * R0).as_quat()[[3, 0, 1, 2]]
    place(p, quat, x[6:], t)
    bm2 = all_body_min()
    after[t] = [bm2[b] for b in distal_ids]
    ffa[t] = ff_min()
    _st, _cv = tip_state(t)
    thA[t], rA[t], zA[t], cosA[t] = _st[:, 0], _st[:, 1], _st[:, 2], _cv
    worst_after[t] = min(bm2.values())
    jx = joints_xyz()
    jdev[t] = np.linalg.norm(jx - jorig, axis=1).max()
    sol_x[t] = (p.copy(), quat.copy(), x[6:].copy()); sol_joints[t] = jx.copy()
    if (t - args.start) % 10 == 0 or t == frames[-1]:
        mv = np.linalg.norm(jx - jorig, axis=1)
        nlim = int(np.sum(((x[6:] - lo) < 1e-4) | ((hi - x[6:]) < 1e-4)))
        print('     worst-moved joint: %s (%.1f mm) | dofs at limit: %d | wrist shift %.1f mm, rot %.1f deg | cost %.2f'
              % (dexhand.body_names[1:][int(np.argmax(mv))], mv.max() * 1000, nlim,
                 np.linalg.norm(x[0:3]) * 1000, np.degrees(np.linalg.norm(x[3:6])), res.cost))
    cprev = x - x0
    if (t - args.start) % 10 == 0 or t == frames[-1]:
        print("t=%3d  distal min mm before %s -> after %s | worst any-body %.1f mm | max joint move %.1f mm | nfev %d | %.0fs"
              % (t, np.round(before[t] * 1000, 1), np.round(after[t] * 1000, 1), worst_after[t] * 1000,
                 jdev[t] * 1000, res.nfev, time.time() - t_start))

if RAW_CACHE and not _cache_hit and len(frames) == T:
    _pk.dump(dict(sol_x=sol_x, sol_joints=sol_joints, ffb=ffb, ffa=ffa, **{_n: globals()[_n] for _n in _CACHE_ARR}), open(RAW_CACHE, "wb"))
    print("saved raw solution cache", RAW_CACHE)
SMOOTH = float(os.environ.get("MYOHAND_SMOOTH_SIGMA", "0"))
if SMOOTH > 0 and len(frames) > 2:
    _Rq = lambda t_: Rot.from_quat(wq0[t_][[1, 2, 3, 0]])
    _tip = dexhand.body_names[1:].index("LFtip_r")
    _Jb = np.stack([sol_joints[t_] for t_ in frames])
    _Qb = np.stack([sol_x[t_][2] for t_ in frames])
    _Pa = np.stack([sol_x[t_][0] for t_ in frames])
    _Qa_ = np.stack([sol_x[t_][1] for t_ in frames])
    for _i in range(1, len(_Qa_)):
        if np.dot(_Qa_[_i], _Qa_[_i - 1]) < 0:
            _Qa_[_i] = -_Qa_[_i]
    _Da = np.stack([sol_x[t_][2] for t_ in frames])
    _Pa, _Qa_, _Da = [gaussian_filter1d(D_, SMOOTH, axis=0, mode="nearest") for D_ in (_Pa, _Qa_, _Da)]
    _Qa_ = _Qa_ / np.linalg.norm(_Qa_, axis=1, keepdims=True)
    BLEND_TOL = float(os.environ.get("MYOHAND_BLEND_TOL_MM", "0.3")) / 1000
    BLEND_SIGMA = float(os.environ.get("MYOHAND_BLEND_SIGMA", "2"))
    _W = np.zeros(len(frames))
    for _k, t_ in enumerate(frames):
        pr_, qr_, dr_ = sol_x[t_]
        if np.dot(_Qa_[_k], qr_) < 0:
            qr_ = -qr_
        for w_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            pw_ = (1 - w_) * _Pa[_k] + w_ * pr_
            qw_ = (1 - w_) * _Qa_[_k] + w_ * qr_
            qw_ = qw_ / np.linalg.norm(qw_)
            dw_ = np.clip((1 - w_) * _Da[_k] + w_ * dr_, lo, hi)
            place(pw_, qw_, dw_, t_)
            if min(all_body_min().values()) >= -BLEND_TOL:
                break
        _W[_k] = w_
    _Wf = np.maximum(_W, gaussian_filter1d(_W, BLEND_SIGMA, mode="nearest"))
    print("frames needing blend-back before ramping: %d" % int((_W > 0).sum()))
    _nfix = 0
    for _k, t_ in enumerate(frames):
        w0_ = _Wf[_k]
        pr0_, qr0_, dr0_ = sol_x[t_]
        if np.dot(_Qa_[_k], qr0_) < 0:
            qr0_ = -qr0_
        p_ = (1 - w0_) * _Pa[_k] + w0_ * pr0_
        quat_ = (1 - w0_) * _Qa_[_k] + w0_ * qr0_
        quat_ = quat_ / np.linalg.norm(quat_)
        q_ = np.clip((1 - w0_) * _Da[_k] + w0_ * dr0_, lo, hi)
        place(wp0[t_], wq0[t_], q0[t_], t_)
        jo_ = joints_xyz()
        place(p_, quat_, q_, t_)
        bm2_ = all_body_min()
        if min(bm2_.values()) < -0.0003:
            pr_, qr_, dr_ = sol_x[t_]
            if np.dot(_Qa_[_k], qr_) < 0:
                qr_ = -qr_
            for w_ in (0.25, 0.5, 0.75, 1.0):
                p_ = (1 - w_) * _Pa[_k] + w_ * pr_
                quat_ = (1 - w_) * _Qa_[_k] + w_ * qr_
                quat_ = quat_ / np.linalg.norm(quat_)
                q_ = np.clip((1 - w_) * _Da[_k] + w_ * dr_, lo, hi)
                place(p_, quat_, q_, t_)
                bm2_ = all_body_min()
                if min(bm2_.values()) >= -0.0003:
                    break
            _nfix += 1
        after[t_] = [bm2_[b] for b in distal_ids]
        ffa[t_] = ff_min()
        st2_, cv2_ = tip_state(t_)
        thA[t_], rA[t_], zA[t_], cosA[t_] = st2_[:, 0], st2_[:, 1], st2_[:, 2], cv2_
        worst_after[t_] = min(bm2_.values())
        jx_ = joints_xyz()
        jdev[t_] = np.linalg.norm(jx_ - jo_, axis=1).max()
        sol_x[t_] = (p_.copy(), quat_.copy(), q_.copy()); sol_joints[t_] = jx_.copy()
    _Ja = np.stack([sol_joints[t_] for t_ in frames])
    _Qa = np.stack([sol_x[t_][2] for t_ in frames])
    _acc = lambda J_: np.sqrt((np.diff(J_[:, _tip] * 1000, 2, axis=0) ** 2).sum(1).mean())
    _dacc = lambda Q_: np.sqrt((np.diff(Q_, 2, axis=0) ** 2).mean(0)).mean() * 1000
    print("frames blended back toward the unsmoothed solution (mug overlap > 0.3 mm): %d" % _nfix)
    print("smoothing sigma %.1f: pinky tip accel %.3f -> %.3f mm/frame^2 | mean dof accel %.2f -> %.2f mrad/frame^2" % (SMOOTH, _acc(_Jb), _acc(_Ja), _dacc(_Qb), _dacc(_Qa)))

sel = np.array(frames)
labels = ["thumb", "index", "middle", "ring", "pinky"]
_loc = lambda t_: MR[t_].T @ (wp0[t_] - mug_p[t_])
_ang = lambda a_: np.degrees(np.abs(np.angle(np.exp(1j * a_))))
print("fingertip angle around the mug axis, from the wrist direction (deg; 0 = toward the hand, 90 = mug equator, >90 = beyond the diameter):")
for i_, l_ in enumerate(labels):
    fr_ = [t_ for t_ in frames if human[t_, i_] < 0.02]
    if fr_:
        a0_, a1_ = _ang(th0[fr_, i_] - _thw0[fr_]), _ang(thA[fr_, i_] - _thw0[fr_])
        print("  %-7s orig %5.1f (%5.1f to %5.1f) -> now %5.1f (%5.1f to %5.1f) | tip radius from axis %.1f -> %.1f mm" % (l_, a0_.mean(), a0_.min(), a0_.max(), a1_.mean(), a1_.min(), a1_.max(), r0[fr_, i_].mean() * 1000, rA[fr_, i_].mean() * 1000))
print("wrist distance from the mug axis (mm, horizontal): %.0f" % (np.mean([np.hypot(*_loc(t_)[:2]) for t_ in frames]) * 1000))
_sel = [t_ for t_ in frames if t_ % 4 == 0] if os.environ.get("MYOHAND_BODY_GAPS") == "1" else []
_g0, _g1 = {}, {}
for t_ in _sel:
    place(wp0[t_], wq0[t_], q0[t_], t_); _a = all_body_min()
    place(sol_x[t_][0], sol_x[t_][1], sol_x[t_][2], t_); _b = all_body_min()
    for b_ in hand_bodies:
        _g0.setdefault(b_, []).append(_a[b_]); _g1.setdefault(b_, []).append(_b[b_])
if _sel:
    print("mean gap to the mug per hand body over %d frames (mm): orig -> now" % len(_sel))
    for b_ in sorted(_g0, key=lambda b_: np.mean(_g0[b_])):
        print("  %-14s %5.1f -> %5.1f" % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b_), np.mean(_g0[b_]) * 1000, np.mean(_g1[b_]) * 1000))
_dm = lambda idx, src: np.degrees(np.stack([src(t_)[idx] for t_ in frames])).mean(0).round(0)
print("pinky mcp/pip/dip flexion mean (deg): orig %s -> now %s | ring now %s" % (_dm(_pi, lambda t_: q0[t_]), _dm(_pi, lambda t_: sol_x[t_][2]), _dm(_ri, lambda t_: sol_x[t_][2])))
print("\n=== summary over %d frames (mm) ===" % len(sel))
print("%-7s %10s %10s %12s %12s" % ("finger", "min before", "min after", "mean before", "mean after"))
for i, l in enumerate(labels):
    print("%-7s %10.1f %10.1f %12.1f %12.1f" % (l, before[sel, i].min() * 1000, after[sel, i].min() * 1000,
                                                before[sel, i].mean() * 1000, after[sel, i].mean() * 1000))
print("worst penetration of ANY hand body after: %.2f mm" % (worst_after[sel].min() * 1000))
print("max joint displacement: %.1f mm (mean of per-frame max %.1f mm)" % (jdev[sel].max() * 1000, jdev[sel].mean() * 1000))

def _slip(th, rr, zz, fr):
    S_ = set(fr)
    v = [np.hypot(rr[f] * np.angle(np.exp(1j * (th[f] - th[f - 1]))), zz[f] - zz[f - 1]) * 1000 for f in fr if f - 1 in S_]
    return np.mean(v) if v else float("nan")
print("\n=== pad angle (deg, mean) and slip (mm/frame, mean) in near-mug frames: before -> after ===")
for i, l in enumerate(labels):
    fr = [t for t in frames if human[t, i] < 0.02]
    if not fr:
        print("%-7s no near-mug frames" % l); continue
    angf = lambda c: np.degrees(np.arccos(np.clip(c, -1, 1)))
    pa = "%5.1f -> %5.1f" % (angf(cos0[fr, i]).mean(), angf(cosA[fr, i]).mean()) if i > 0 else "  n/a (thumb)"
    print("%-7s pad %s | slip %.3f -> %.3f | %d frames" % (l, pa, _slip(th0[:, i], r0[:, i], z0[:, i], fr), _slip(thA[:, i], rA[:, i], zA[:, i], fr), len(fr)))

if args.write:
    assert len(sel) == T, "--write needs the full range"
    p_c = np.stack([sol_x[t][0] for t in range(T)])
    R_c = np.stack([Rot.from_quat(sol_x[t][1][[1, 2, 3, 0]]).as_matrix() for t in range(T)])
    q_new = np.stack([sol_x[t][2] for t in range(T)])
    J_c = np.stack([sol_joints[t] for t in range(T)])                 # (T,20,3) corrected frame
    p_raw = (C.T @ p_c.T).T
    R_raw = np.matmul(C.T[None], R_c)
    aa_raw = Rot.from_matrix(R_raw).as_rotvec()
    J_raw = np.einsum("ij,tkj->tki", C.T, J_c)
    w_raw = np.zeros((T, 3))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        w_raw[t] = Rot.from_matrix(R_raw[b] @ R_raw[a].T).as_rotvec() / ((b - a) * dt)

    new = copy.deepcopy(demo)
    rn, _ = find_rh(new)
    def like(orig, arr):
        if torch.is_tensor(orig):
            return torch.as_tensor(arr, dtype=orig.dtype, device=orig.device)
        return np.asarray(arr, dtype=np.asarray(orig).dtype)
    rn["opt_wrist_pos"] = like(rh["opt_wrist_pos"], p_raw)
    rn["opt_wrist_rot"] = like(rh["opt_wrist_rot"], aa_raw)
    rn["opt_dof_pos"] = like(rh["opt_dof_pos"], q_new)
    rn["opt_wrist_velocity"] = like(rh["opt_wrist_velocity"], np.gradient(p_raw, dt, axis=0))
    rn["opt_wrist_angular_velocity"] = like(rh["opt_wrist_angular_velocity"], w_raw)
    rn["opt_dof_velocity"] = like(rh["opt_dof_velocity"], np.gradient(q_new, dt, axis=0))
    keys = [dexhand.to_hand(b)[0] for b in dexhand.body_names[1:]]
    Jv = np.gradient(J_raw, dt, axis=0)
    rn["myo_joints"] = {k: like(rh["myo_joints"][k], J_raw[:, i]) for i, k in enumerate(keys)}
    rn["myo_joints_velocity"] = {k: like(rh["myo_joints_velocity"][k], Jv[:, i]) for i, k in enumerate(keys)}
    with open(OUT, "wb") as f:
        pickle.dump(new, f)
    print("wrote", OUT)
