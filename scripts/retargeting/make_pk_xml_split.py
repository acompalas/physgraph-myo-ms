"""Regenerate /tmp/myohand_<side>_pk.xml for retarget_myohand.py.
pytorch_kinematics needs: no free joint, one joint per body. So: drop the
free root joint and split every multi-joint body into nested single-joint
bodies (innermost keeps the original name, so frame names are unchanged).
Then validate pk forward kinematics against MuJoCo on random poses."""
import sys, os, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco, torch
import pytorch_kinematics as pk
import pytorch_kinematics.mjcf as m

proj = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj / "scripts/retargeting"))
from myohand_def import MyoHandR

class C(dict):
    def __getitem__(s, k): return super().__getitem__(int(k))
m.JOINT_TYPE_MAP = C(m.JOINT_TYPE_MAP)

side = sys.argv[1] if len(sys.argv) > 1 else "r"
src = proj / f"assets/hands/myohand_{side}_ulnaroot.xml"
out = f"/tmp/myohand_{side}_pk.xml"

ref = mujoco.MjModel.from_xml_path(str(src))
tmp = tempfile.mktemp(suffix=".xml")
mujoco.mj_saveLastXML(tmp, ref)
tree = ET.parse(tmp); r = tree.getroot()
parent = {c: p for p in r.iter() for c in p}

nfree = 0
for b in r.iter("body"):
    for c in list(b):
        if c.tag == "freejoint" or (c.tag == "joint" and c.get("type") == "free"):
            b.remove(c); nfree += 1

def dummy():
    return ET.Element("inertial", dict(pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6"))

nsplit = 0
for b in list(r.iter("body")):
    joints = [c for c in b if c.tag == "joint"]
    if len(joints) <= 1:
        continue
    nsplit += 1
    p = parent[b]; idx = list(p).index(b); name = b.get("name")
    pose = {k: v for k, v in b.attrib.items() if k != "name"}
    outer = []
    for i, j in enumerate(joints[:-1]):
        nb = ET.Element("body", {"name": f"{name}__j{i}"})
        nb.attrib.update(pose if i == 0 else {"pos": "0 0 0"})
        nb.append(dummy()); b.remove(j); nb.append(j); outer.append(nb)
    for a, c in zip(outer, outer[1:]):
        a.append(c)
    for k in list(b.attrib):
        if k != "name": del b.attrib[k]
    b.set("pos", "0 0 0")
    p.remove(b); outer[-1].append(b); p.insert(idx, outer[0])
    parent[b] = outer[-1]; parent[outer[0]] = p

dex0 = MyoHandR().body_names
ntip = 0
for p_ in list(r.iter()):
    for st in list(p_):
        if st.tag == "site" and st.get("name") in dex0:
            nb = ET.Element("body", {k: v for k, v in st.attrib.items() if k in ("name", "pos", "quat")})
            i_ = list(p_).index(st); p_.remove(st); p_.insert(i_, nb); ntip += 1
print("sites converted to fixed bodies:", ntip)
tree.write(out)
print(f"free joints removed: {nfree} | bodies split: {nsplit} | wrote {out}")

os.chdir(proj / "assets/hands")
chain = pk.build_chain_from_mjcf(open(out).read()).to(dtype=torch.float64)
frames = chain.get_frame_names(exclude_fixed=False); jn = chain.get_joint_parameter_names()
dex = MyoHandR()
print("pk frames:", len(frames), "| pk joints:", len(jn), "| dexhand dofs:", len(dex.dof_names))
print("body_names missing from pk:", [n for n in dex.body_names if n not in frames])
print("dof names not in pk:", [n for n in dex.dof_names if n not in jn])
print("pk joints not in dexhand:", [n for n in jn if n not in dex.dof_names])

d = mujoco.MjData(ref); rng = np.random.default_rng(0); worst = {}
for _ in range(20):
    q = {}
    for n in jn:
        jid = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = ref.jnt_range[jid]
        if lo == hi: lo, hi = -0.3, 0.3
        q[n] = rng.uniform(lo, hi)
    d.qpos[:] = ref.qpos0
    for n, v in q.items():
        d.qpos[ref.jnt_qposadr[mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    mujoco.mj_forward(ref, d)
    fk = chain.forward_kinematics(torch.tensor([[q[n] for n in jn]], dtype=torch.float64))
    for name in dex.body_names:
        if name not in fk: continue
        bid = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ref_pos = d.xpos[bid]
        else:
            ref_pos = d.site_xpos[mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_SITE, name)]
        e = np.linalg.norm(fk[name].get_matrix()[0, :3, 3].numpy() - ref_pos)
        worst[name] = max(worst.get(name, 0.0), e)
print("max FK error vs MuJoCo (mm), worst 5:",
      sorted(((round(v * 1000, 4), k) for k, v in worst.items()), reverse=True)[:5])
