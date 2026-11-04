"""Zero-action rollout for a few steps; list every active contact pair (world 0) with distances in mm.
Usage: python3 scripts/probe_contacts.py [steps]"""
import sys, torch, numpy as np, mujoco
sys.path.insert(0, ".")
import envs.mjwarp_myohand_env as E
import warp as wp
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
env = E.MyoHandPourEnv(num_envs=4, device="cuda:0")
m = env.mj_model
env.reset()
bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
act = torch.zeros(4, 48, device="cuda:0")
for t in range(steps):
    env.step(act)
    dd = env.data
    nc = int(wp.to_torch(dd.nacon).item()) if hasattr(dd, "nacon") else int(wp.to_torch(dd.ncon).item())
    geom = wp.to_torch(dd.contact.geom)[:nc].cpu().numpy()
    wid = wp.to_torch(dd.contact.worldid)[:nc].cpu().numpy()
    dist = wp.to_torch(dd.contact.dist)[:nc].cpu().numpy()
    print("step %d: %d active contacts total (all 4 worlds)" % (t + 1, nc))
    rows = {}
    for (g1, g2), w, dst in zip(geom, wid, dist):
        if w == 0:
            rows.setdefault(tuple(sorted([bn(g1), bn(g2)])), []).append(dst * 1000)
    for k, v in sorted(rows.items(), key=lambda kv: min(kv[1])):
        print("   %-42s n=%d  dist mm: min %.2f max %.2f" % (" <-> ".join(k), len(v), min(v), max(v)))
