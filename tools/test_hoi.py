"""Check HOI data conventions, actor transfer, gradients and free-object physics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--motion_dir", default=str(ROOT / "data/hoi/omnicontact/pushbox/train"))
ap.add_argument("--source", default=str(Path.home() / "OmniContact_sim2sim/data"))
ap.add_argument("--checkpoint", default=str(ROOT / "logs/lafan1_5090/ckpt_014000.pt"))
ap.add_argument("--checks_only", action="store_true")
add_launcher_args(ap)
args = ap.parse_args()

import numpy as np
import torch
from supermimic import hoi
from supermimic.buffer import ReplayBuffer
from supermimic.env import make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

dev = "cuda:0"
kin = G1Kinematics(dev)
m = hoi.HOIMotionLib.from_dir(args.motion_dir, kin, dev)
assert (m.bin_start < m.clip_end[m.bin_clip]).all(), "Reset bin includes terminal-only frame"
mdp = hoi.HOIMDP(kin, m, torch.tensor(default_joint_pos_list(), device=dev))
f = m.clip_start.clone()
S = (*m.ref_state(f), m.object_p[f], m.object_R[f], m.object_v[f], m.object_w[f], m.table_p[f])
for original, restored in zip(S, hoi.unflatten_state(hoi.flatten_state(S))):
    torch.testing.assert_close(original, restored, atol=1e-5, rtol=1e-5)
qerr = []
family = Path(args.motion_dir).parent.name
for i, name in enumerate(m.names):
    with np.load(Path(args.source) / family / (name + ".npz")) as source:
        # Independent source body positions catch an incorrect joint permutation or quaternion order.
        ref = source["body_pos_w"][0, [25, 26]] - source["body_pos_w"][0, 0]
        ours = (m.body_p[f[i], [3, 6]] - m.root_p[f[i]]).cpu().numpy()
        error = float(np.linalg.norm(ref - ours, axis=-1).max())
        assert error < 0.01, (name, "source FK mismatch", error)
        qerr.append(error)
xi = torch.zeros(len(f), 33, device=dev); xi[:, 0] = 1
a = torch.zeros(len(f), 29, device=dev)
obs = mdp.obs(S, a, xi, f)
g = m.goal(f, 10)
assert obs.shape[-1] == hoi.OBS_DIM and g.shape[-1] == hoi.GOAL_FRAME_DIM * 10
assert mdp.critic_features(S, f, xi).shape[-1] == hoi.CRITIC_DIM
wm = hoi.HOIWorldModel(hidden=(64, 64)).to(dev)
for before, after in zip(S, wm.step(S, a, xi)):
    torch.testing.assert_close(before, after)
target = wm.targets(S, S)
assert target.shape[-1] == len(wm.out_norm.mean) == 82
torch.testing.assert_close(target, torch.zeros_like(target), atol=2e-6, rtol=0)
obj = (S[6] + 0.05).detach().requires_grad_()
shifted = (*S[:6], obj, *S[7:])
mdp.reward(shifted, f, a, a).sum().backward()
assert torch.isfinite(obj.grad).all() and obj.grad.abs().sum() > 0
ck = torch.load(args.checkpoint, map_location=dev)
old = Policy(99, 730, hist=10).to(dev); old.load_state_dict(ck["pi"])
new = Policy(hoi.OBS_DIM, 920, hist=10).to(dev)
hoi.initialize_policy(new, ck, 10, 10)
with torch.no_grad():
    h = obs[:, None].repeat(1, 10, 1)
    pred_old = old(h[..., :99], g.reshape(len(f), 10, 92)[..., :73].flatten(-2))
    pred_new = new(h, g)
    torch.testing.assert_close(pred_old, pred_new, atol=5e-5, rtol=5e-5)
report = {"clips": len(f), "source_foot_fk_max_m": max(qerr), "state_dim": hoi.STATE_DIM,
          "obs_dim": hoi.OBS_DIM, "actor_transfer_max_error": (pred_old - pred_new).abs().max().item(),
          "object_reward_gradient_finite": True}
if not args.checks_only:
    cfg = make_sim(dev, args)
    with launch_simulation(cfg, args):
        env = hoi.HOITrackingSim(cfg, 4, m, mdp, eval_mode=True)
        ids = torch.arange(4, device=dev)
        env.reset(ids, m.clip_start[0].repeat(4)); env.refresh()
        torch.testing.assert_close(env.state()[6], m.object_p[env.f], atol=1e-5, rtol=1e-5)
        box = env.scene["box"]
        pose = box.data.root_link_pose_w.torch.clone(); pose[:, 2] = 3
        box.write_root_link_pose_to_sim_index(root_pose=pose)
        box.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(4, 6, device=dev))
        env.refresh()
        z0 = env.state()[6][:, 2].clone()
        replay = ReplayBuffer(16, 4, dev, state_dim=hoi.STATE_DIM)
        for _ in range(10):
            out = env.step(torch.zeros(4, 29, device=dev))
            replay.add(out)
            assert out["s"].shape == (4, 90) and torch.isfinite(out["s1"]).all()
        drop = z0 - env.state()[6][:, 2]
        assert (drop > 0.1).all() and (env.state()[6][:, 2] > 2).all(), drop
        assert replay.sample_windows(8, 4)["s"].shape == (8, 4, 90)
        env.reset(ids, m.clip_start[0].repeat(4)); env.refresh()
        torch.testing.assert_close(env.state()[6], m.object_p[env.f], atol=1e-5, rtol=1e-5)
        report.update(free_box_drop_m=drop.mean().item(), physics_and_replay_checked=True)
print(json.dumps(report, indent=2))
