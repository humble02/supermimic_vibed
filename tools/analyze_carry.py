"""Measure lifted transport in a recorded HOI evaluation, including early failures.

Hand proximity is a geometric proxy, not a contact-force measurement. Expected carry
time is computed over the entire reference clip, so early termination cannot inflate it.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def quat_matrix(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def clearance(p, R, tables, half_size, table_half_size):
    ext = np.abs(R) @ np.asarray(half_size)
    overlap = (np.abs(p[..., None, :2] - tables[..., :2]) <=
               ext[..., None, :2] + np.asarray(table_half_size[:2])).all(-1)
    support = np.maximum(0, np.where(overlap, tables[..., 2] + table_half_size[2], 0).max(-1))
    return p[..., 2] - ext[..., 2] - support


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def transport_span(p, mask, min_frames=15):
    """Largest XY displacement from a continuous segment's start, excluding short segments."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    spans = [np.linalg.norm(p[start:end, :2] - p[start, :2], axis=-1).max()
             for start, end in zip(starts, ends) if end - start >= min_frames]
    return float(max(spans, default=0))


def hand_proximity(model, data, root_p, root_q, joint_q, object_p, object_R, half_size):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:36] = root_p, root_q, joint_q
    mujoco.mj_kinematics(model, data)
    near = []
    for side, y in (("left", -0.02), ("right", 0.02)):
        body = model.body(side + "_wrist_yaw_link").id
        samples = np.array([[0.07, 0, 0], [0.11, y/2, 0], [0.15, y, 0]])
        world = samples @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
        local = (world - object_p) @ object_R
        delta = np.abs(local) - np.asarray(half_size)
        sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0) - 0.035
        near.append(bool(np.min(np.abs(sdf)) <= 0.06))
    return all(near)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.trajectory, allow_pickle=False) as loaded:
        d = dict(loaded)
    cfg = json.loads(str(d["object_cfg"]))
    if not cfg["tables"]:
        raise ValueError("Use a carry-box trajectory with its support tables")
    if "ref_contact" not in d:
        raise ValueError("This trajectory lacks contact references; record it again with the current evaluate.py")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/g1_description/g1_29dof_hoi.xml"))
    data = mujoco.MjData(model)
    result = {"thresholds": {"support_clearance_m": 0.03, "object_tracking_error_m": 0.15,
                              "hand_capsule_surface_distance_m": 0.06,
                              "minimum_continuous_carry_seconds": 0.3,
                              "minimum_segment_xy_displacement_m": 0.1,
                              "minimum_lift_above_initial_center_m": 0.03},
              "note": "Geometric hand proximity is not measured contact force; carry evidence alone is not full task success.",
              "per_clip": {}}
    for c, name in enumerate(d["names"]):
        start, length = int(d["clip_start"][c]), int(d["clip_len"][c])
        full = np.arange(start, start + length - 1)
        tables = d["table_p"][0, c]
        ref_clear = clearance(d["ref_object_p"][full], d["ref_object_R"][full], tables,
                              cfg["half_size"], cfg["table_half_size"])
        expected = (ref_clear > 0.03) & (d["ref_contact"][full, 2:4] > 0.5).all(-1)
        p, R = d["object_p"][:, c], quat_matrix(d["object_quat"][:, c])
        actual_clear = clearance(p, R, d["table_p"][:, c], cfg["half_size"], cfg["table_half_size"])
        frames = d["f"][:, c]
        alive = d["alive"][:, c].astype(bool)
        expected_here = expected[np.clip(frames - start, 0, len(expected) - 1)]
        error = np.linalg.norm(p - d["ref_object_p"][frames], axis=-1)
        good = alive & expected_here & (actual_clear > 0.03) & (error <= 0.15)
        for t in np.flatnonzero(good):
            good[t] &= hand_proximity(model, data, d["root_p"][t, c], d["root_quat"][t, c],
                                      d["q"][t, c], p[t], R[t], cfg["half_size"])
        travel = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=-1)
        airborne_path = float(travel[good[1:] & good[:-1]].sum())
        # Requiring a rise from the initial support height excludes sliding off a table
        # and falling while the hands happen to remain nearby. Displacement (not summed
        # path length) also prevents stationary jitter from counting as transport.
        lifted = good & (p[:, 2] - p[0, 2] > 0.03)
        lifted_span = transport_span(p, lifted)
        valid_p = p[alive]
        result["per_clip"][str(name)] = {
            "reference_expected_carry_seconds": round(float(expected.sum()) / 50, 3),
            "tracked_airborne_with_both_hands_seconds": round(float(good.sum()) / 50, 3),
            "required_carry_fraction_matched": round(float(good.sum()) / max(int(expected.sum()), 1), 4),
            "longest_tracked_airborne_segment_seconds": round(longest_run(good) / 50, 3),
            "tracked_airborne_xy_path_m": round(airborne_path, 4),
            "longest_lifted_segment_seconds": round(longest_run(lifted) / 50, 3),
            "max_lifted_segment_xy_displacement_m": round(lifted_span, 4),
            "reference_peak_height_above_start_m": round(float(d["ref_object_p"][full, 2].max() - d["ref_object_p"][start, 2]), 4),
            "actual_peak_height_above_start_m": round(float(valid_p[:, 2].max() - p[0, 2]), 4),
            "observed_lift_and_transport": bool(lifted_span >= 0.1),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
