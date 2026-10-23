"""Task-specific models and dimensions; the training algorithm is shared."""
from types import SimpleNamespace

from . import mdp, networks
from .env import TrackingSim
from .motion import MotionLib


def get_task(name):
    floor = [3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3 + [3e-3] * 29 + [5e-2] * 29
    if name == "lafan":
        return SimpleNamespace(MotionLib=MotionLib, MDP=mdp.MDP, WorldModel=networks.WorldModel,
            TrackingSim=TrackingSim, unflatten_state=mdp.unflatten_state, state_dim=mdp.STATE_DIM,
            obs_dim=mdp.OBS_DIM, critic_dim=mdp.INV_DIM + mdp.CRITIC_EXTRA, goal_frame_dim=73,
            max_step_reward=mdp.MAX_STEP_REWARD, error_floor=floor)
    if name == "hoi":
        from . import hoi
        return SimpleNamespace(MotionLib=hoi.HOIMotionLib, MDP=hoi.HOIMDP, WorldModel=hoi.HOIWorldModel,
            TrackingSim=hoi.HOITrackingSim, unflatten_state=hoi.unflatten_state, state_dim=hoi.STATE_DIM,
            obs_dim=hoi.OBS_DIM, critic_dim=hoi.CRITIC_DIM, goal_frame_dim=hoi.GOAL_FRAME_DIM,
            max_step_reward=hoi.MAX_STEP_REWARD, error_floor=floor + [3e-3]*3 + [5e-3]*3 + [2e-2]*3 + [5e-2]*3)
    raise ValueError(f"Unknown task: {name}")
