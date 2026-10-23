"""G1-box interaction task for the same learned-dynamics/BPTT training loop.

The box is a free dynamic rigid body. Reference poses are used only at episode resets
and in goals/rewards; they never overwrite the simulated box during a rollout.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass, replace

from . import mdp as base_mdp
from . import networks
from .env import TrackingSceneCfg, TrackingSim
from .motion import CONTROL_DT, MotionLib, _gradient
from .robot import DATA_DIR, G1_CFG
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, so3_exp, so3_log, yaw_mat

STATE_DIM = base_mdp.STATE_DIM + 13 + 6
OBS_DIM = base_mdp.OBS_DIM + 23
CRITIC_DIM = base_mdp.INV_DIM + base_mdp.CRITIC_EXTRA + 26
GOAL_FRAME_DIM = 92
MAX_STEP_REWARD = 9 * CONTROL_DT


def initialize_policy(pi, ck, K, hist):
    """Embed a LAFAN actor exactly; initially ignore the new object channels."""
    a = ck["args"]
    if a.get("task", "lafan") != "lafan" or a["goal_frames"] != K or a["hist"] != hist:
        raise ValueError("LAFAN initialization requires matching goal_frames and hist")
    old, new = ck["pi"], pi.state_dict()
    for k, value in old.items():
        if value.shape == new[k].shape:
            new[k].copy_(value)
    gi = torch.tensor([j * GOAL_FRAME_DIM + i for j in range(K) for i in range(73)], device=new["enc.0.weight"].device)
    oi = torch.tensor([i for i in range(32)] + [32 + j * OBS_DIM + i for j in range(hist) for i in range(base_mdp.OBS_DIM)], device=gi.device)
    new["enc.0.weight"].zero_(); new["enc.0.weight"][:, gi] = old["enc.0.weight"]
    new["dec.0.weight"].zero_(); new["dec.0.weight"][:, oi] = old["dec.0.weight"]
    for suffix in ("mean", "var"):
        new[f"goal_norm.{suffix}"][gi] = old[f"goal_norm.{suffix}"]
        new[f"obs_norm.{suffix}"][:base_mdp.OBS_DIM] = old[f"obs_norm.{suffix}"]
    new["goal_norm.count"].zero_(); new["obs_norm.count"].zero_()
    pi.load_state_dict(new)


def flatten_state(S):
    op, oR, ov, ow, tables = S[6:]
    return torch.cat((base_mdp.flatten_state(S[:6]), op, mat_to_quat(oR), ov, ow, tables.flatten(-2)), -1)


def unflatten_state(s):
    return (*base_mdp.unflatten_state(s[..., :71]), s[..., 71:74], quat_to_mat(s[..., 74:78]),
            s[..., 78:81], s[..., 81:84], s[..., 84:90].reshape(*s.shape[:-1], 2, 3))


def object_features(S):
    p, R, v, w = S[:4]
    op, oR, ov, ow, tables = S[6:]
    Rt = R.transpose(-1, -2)
    local = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)
    return torch.cat((local(op - p), mat_to_6d(Rt @ oR), local(ov - v), local(ow - w),
                      (Rt.unsqueeze(-3) @ (tables - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1).flatten(-2)), -1)


class HOIMotionLib(MotionLib):
    def __init__(self, path, kin, device):
        files = sorted(glob.glob(str(Path(path) / "*.csv")))
        if not files:
            raise ValueError(f"No prepared HOI clips in {path}; run tools/prepare_hoi.py")
        super().__init__(files, kin, device, src_fps=50.0)
        self.object_cfg = json.loads((Path(path) / "object.json").read_text())
        fields = {k: [] for k in ("object_p", "object_R", "object_v", "object_w", "contact", "table_p")}
        for i, filename in enumerate(files):
            with np.load(Path(filename).with_suffix(".npz"), allow_pickle=False) as d:
                op = torch.tensor(d["object_p"], device=device)
                oR = quat_to_mat(torch.tensor(d["object_quat"], device=device))
                if len(op) != int(self.clip_len[i]):
                    raise ValueError(f"Robot/object frame mismatch in {filename}")
                ow = torch.zeros_like(op)
                ow[1:-1] = so3_log(oR[2:] @ oR[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
                ow[0] = so3_log(oR[1] @ oR[0].T) / CONTROL_DT
                ow[-1] = so3_log(oR[-1] @ oR[-2].T) / CONTROL_DT
                tables = torch.tensor(d["table_p"], device=device)
                if self.object_cfg["tables"] and (tables - tables[:1]).abs().max() > 1e-4:
                    raise ValueError("This first HOI task supports stationary support tables only")
                if not self.object_cfg["tables"]:
                    tables.zero_()
                for k, value in zip(fields, (op, oR, _gradient(op, CONTROL_DT), ow,
                                             torch.tensor(d["contact"], device=device), tables)):
                    fields[k].append(value)
        for k, values in fields.items():
            setattr(self, k, torch.cat(values))

    @staticmethod
    def from_dir(path, kin, device):
        return HOIMotionLib(path, kin, device)

    def goal(self, f, K):
        g = super().goal(f, K).reshape(*f.shape, K, 73)
        fut = self.future_frames(f, K)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        local = lambda x: (Ht @ x.unsqueeze(-1)).squeeze(-1)
        obj = torch.cat((local(self.object_p[fut] - self.root_p[f].unsqueeze(-2)),
                         mat_to_6d(Ht @ self.object_R[fut]), local(self.object_v[fut]),
                         local(self.object_w[fut]), self.contact[fut]), -1)
        return torch.cat((g, obj), -1).flatten(-2)


class HOIMDP(base_mdp.MDP):
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, kin, motion, default_q):
        super().__init__(kin, motion, default_q)
        dev = default_q.device
        self.half_size = torch.tensor(motion.object_cfg["half_size"], device=dev)
        self.hand_offsets = torch.tensor([[0.11, -0.01, 0], [0.11, 0.01, 0]], device=dev)
        self.obs_noise = torch.cat((self.obs_noise, torch.full((23,), 0.005, device=dev)))

    def hand_geometry(self, S):
        bp, bR = self.kin.forward(*S[:6])[:2]
        hands = bp[..., [10, 13], :] + (bR[..., [10, 13], :, :] @ self.hand_offsets.unsqueeze(-1)).squeeze(-1)
        local = (S[7].transpose(-1, -2).unsqueeze(-3) @ (hands - S[6].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        delta = local.abs() - self.half_size
        distance = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(-1).clamp_max(0) - 0.035
        return local, distance

    def obs(self, S, a_prev, xi, f):
        return torch.cat((super().obs(S[:6], a_prev, xi, f), object_features(S), self.hand_geometry(S)[1]), -1)

    def reward(self, S, f, a, a_prev, terms=False):
        r, detail = super().reward(S[:6], f, a, a_prev, terms=True)
        m = self.motion
        ep = (S[6] - m.object_p[f]).square().sum(-1)
        er = rot_angle_sq(m.object_R[f], S[7])
        ev = (S[8] - m.object_v[f]).square().sum(-1)
        local, distance = self.hand_geometry(S)
        ref_hands = m.body_p[f][..., [10, 13], :] + (m.body_R[f][..., [10, 13], :, :] @ self.hand_offsets.unsqueeze(-1)).squeeze(-1)
        ref_local = (m.object_R[f].transpose(-1, -2).unsqueeze(-3) @ (ref_hands - m.object_p[f].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        contact = m.contact[f][..., 2:4]
        count = contact.sum(-1).clamp_min(1)
        relation = torch.exp(-(local - ref_local).square().sum(-1) / 0.15**2).mean(-1)
        touch = (torch.exp(-distance.square() / 0.06**2) * contact).sum(-1) / count
        touch = torch.where(contact.sum(-1) > 0, touch, torch.ones_like(touch))
        obj = torch.exp(-ep / 0.25**2) + 0.5 * torch.exp(-er / 0.5**2) + 0.5 * torch.exp(-ev / 1.0**2)
        r = r + (obj + relation + touch) * CONTROL_DT
        detail.update(e_object_pos=ep, object_track=obj * CONTROL_DT, hand_relation=relation, contact_proxy=touch)
        return (r, detail) if terms else r

    def termination(self, S, f):
        return super().termination(S[:6], f) | ((S[6] - self.motion.object_p[f]).square().sum(-1) > 0.75**2)

    def soft_termination(self, S, f, tz=0.02, tt=0.04):
        robot = super().soft_termination(S[:6], f, tz, tt)
        error = (S[6] - self.motion.object_p[f]).square().sum(-1)
        obj = torch.sigmoid((error - 0.75**2) / 0.05)
        return 1 - (1 - robot) * (1 - obj)

    def critic_features(self, S, f, xi):
        delta = (S[1].transpose(-1, -2) @ (self.motion.object_p[f] - S[6]).unsqueeze(-1)).squeeze(-1)
        return torch.cat((super().critic_features(S[:6], f, xi), object_features(S), delta, self.hand_geometry(S)[1]), -1)


class HOIWorldModel(networks.WorldModel):
    def __init__(self, hidden=(2048, 2048, 2048)):
        super().__init__(hidden)
        self.in_dim += 21
        self.net = networks.mlp(self.in_dim, list(hidden), networks.WM_OUT + 12, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.in_norm = networks.RunningNorm(self.in_dim)
        self.out_norm = networks.RunningNorm(networks.WM_OUT + 12, eps=1e-4)

    def inputs(self, S, a, xi_n):
        return torch.cat((super().inputs(S[:6], a, xi_n), object_features(S)), -1)

    @staticmethod
    def targets(S, S1):
        Ht, Rt = yaw_mat(S[1]).transpose(-1, -2), S[1].transpose(-1, -2)
        local = lambda R, x: (R @ x.unsqueeze(-1)).squeeze(-1)
        obj = torch.cat((local(Ht, S1[6] - S[6]), so3_log(S[7].transpose(-1, -2) @ S1[7]),
                         local(Rt, S1[8] - S[8]), local(Rt, S1[9] - S[9])), -1)
        return torch.cat((networks.WorldModel.targets(S[:6], S1[:6]), obj), -1)

    def step(self, S, a, xi_n):
        y = networks.run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dp, dr, dv, dw, dq, dqd, dop, dor, dov, dow = d.split((3, 3, 3, 3, 29, 29, 3, 3, 3, 3), -1)
        H, R = yaw_mat(S[1]), S[1]
        world = lambda rot, x: (rot @ x.unsqueeze(-1)).squeeze(-1)
        return (S[0] + world(H, dp), R @ so3_exp(dr), S[2] + world(R, dv), S[3] + world(R, dw),
                S[4] + dq, S[5] + dqd, S[6] + world(H, dop), S[7] @ so3_exp(dor),
                S[8] + world(R, dov), S[9] + world(R, dow), S[10])

    @staticmethod
    def state_error(S_hat, S):
        return HOIWorldModel.targets(S, S_hat)


@configclass
class HOISceneCfg(TrackingSceneCfg):
    box: RigidObjectCfg = None
    table1: RigidObjectCfg | None = None
    table2: RigidObjectCfg | None = None


def scene_config(num_envs, motion):
    cfg = HOISceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True)
    usd = str(Path(DATA_DIR) / "g1_hoi_usd/g1_29dof_hoi/g1_29dof_hoi.usda")
    if not Path(usd).exists():
        raise FileNotFoundError(f"Convert the HOI MJCF first: {usd}")
    cfg.robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot", spawn=replace(G1_CFG.spawn, usd_path=usd))
    o = motion.object_cfg
    cfg.box = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Box", spawn=sim_utils.CuboidCfg(
        size=tuple(2 * x for x in o["half_size"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=o["friction"], dynamic_friction=o["friction"]),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.35, 0.15))))
    if o["tables"]:
        for name in ("table1", "table2"):
            setattr(cfg, name, RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.CuboidCfg(
                size=tuple(2 * x for x in o["table_half_size"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=1.0))))
    return cfg


class HOITrackingSim(TrackingSim):
    flatten_state = staticmethod(flatten_state)

    def __init__(self, sim_cfg, num_envs, motion, mdp, **kwargs):
        super().__init__(sim_cfg, num_envs, motion, mdp, scene_cfg=scene_config(num_envs, motion), **kwargs)

    def _setup_friction(self):
        super()._setup_friction()
        model = self.sim.backend_manager.get_model()
        wp.to_torch(model.shape_material_mu)[wp.to_torch(model.shape_body) < 0] = 0.5

    def state(self):
        S = super().state()
        pose = self.scene["box"].data.root_link_pose_w.torch.clone()
        vel = self.scene["box"].data.root_link_vel_w.torch.clone()
        return (*S, pose[:, :3] - self.scene.env_origins, quat_to_mat(pose[:, [6, 3, 4, 5]]),
                vel[:, :3], vel[:, 3:], self.table_positions)

    def reset(self, env_ids, frames=None):
        if not len(env_ids):
            return
        super().reset(env_ids, frames)
        f, m = self.f[env_ids], self.motion
        if not hasattr(self, "table_positions"):
            self.table_positions = torch.zeros(self.N, 2, 3, device=self.device)
        self.table_positions[env_ids] = m.table_p[f]
        q = mat_to_quat(m.object_R[f])
        pose = torch.cat((m.object_p[f] + self.scene.env_origins[env_ids], q[:, [1, 2, 3, 0]]), -1)
        box = self.scene["box"]
        box.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        box.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((m.object_v[f], m.object_w[f]), -1), env_ids=env_ids)
        if m.object_cfg["tables"]:
            for i, name in enumerate(("table1", "table2")):
                pose = torch.zeros(len(env_ids), 7, device=self.device)
                pose[:, :3] = m.table_p[f, i] + self.scene.env_origins[env_ids]
                pose[:, 6] = 1
                self.scene[name].write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.scene.reset(env_ids)
