"""Run a checkpoint in a live Newton GL window, one motion at a time."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--motion_dir", required=True)
ap.add_argument("--clip", default=None, help="motion basename; default first motion")
ap.add_argument("--seconds", type=float, default=0, help="exit after this wall time, 0 = until window closes")
ap.add_argument("--no_realtime", action="store_true")
add_launcher_args(ap)
args = ap.parse_args()
if args.visualizer is None:
    args.visualizer = ["newton_gl"]

import torch
import supermimic.networks as networks
from supermimic.env import make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.robot import default_joint_pos_list
from supermimic.tasks import get_task

dev = "cuda:0"
ck = torch.load(args.checkpoint, map_location=dev)
task = get_task(ck["args"].get("task", "lafan"))
kin = G1Kinematics(dev)
motion = task.MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = task.MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
c = motion.names.index(args.clip) if args.clip else 0
K, H = ck["args"]["goal_frames"], ck["args"]["hist"]
networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp") == "bf16" else None
pi = networks.Policy(task.obs_dim, task.goal_frame_dim * K, hist=H).to(dev)
pi.load_state_dict(ck["pi"]); pi.eval()
cfg = make_sim(dev, args)
with launch_simulation(cfg, args), torch.no_grad():
    env = task.TrackingSim(cfg, 1, motion, mdp, eval_mode=True)
    ids = torch.zeros(1, dtype=torch.long, device=dev)
    frames = motion.clip_start[c:c+1]
    hist = None
    env.reset(ids, frames)
    env.refresh()
    start = time.monotonic()
    print(f"Live policy: {motion.names[c]}; checkpoint iteration {ck['iteration']}; close the window to exit.", flush=True)
    while env.sim.is_running() and (args.seconds <= 0 or time.monotonic() - start < args.seconds):
        t0 = time.monotonic()
        if bool(env.done_eval.any()):
            print(f"Motion finished, failed={bool(env.failed_eval.any())}; restarting.", flush=True)
            env.done_eval[:] = False; env.failed_eval[:] = False
            env.reset(ids, frames); env.refresh(); hist = None
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o[:, None].repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], o[:, None]), 1)
        p = S[0][0].cpu().tolist()
        env.sim.set_camera_view((p[0]+3, p[1]-3, 2.1), (p[0], p[1], 0.8))
        env.step(pi(hist, motion.goal(env.f, K)))
        if not args.no_realtime:
            time.sleep(max(0, 0.02 - (time.monotonic() - t0)))
