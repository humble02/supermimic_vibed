"""Build a provenance-tracked, 50 Hz G1/object dataset from local OmniContact motions.

Raw third-party data stays under ignored data/. Splits are by entire source motion.
ULTRA and I-BFM are inventoried for follow-up work, not silently mixed into this dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def normal_quat(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 0.9) or np.any(norm > 1.1):
        raise ValueError("Invalid wxyz quaternion")
    return q / norm


def prepare_assets():
    """Keep baseline assets intact; add collision capsules to the HOI robot's palms."""
    src = ROOT / "data/g1_description/g1_29dof.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(src.parent))
    for side, y in (("left", -0.02), ("right", 0.02)):
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(body, "geom", name=f"{side}_hoi_palm", type="capsule",
                      fromto=f"0.07 0 0 0.15 {y} 0", size="0.035", density="0",
                      friction="0.9 0.01 0.001", rgba="0.2 0.3 0.5 1")
    out = ROOT / "data/g1_description/g1_29dof_hoi.xml"
    tree.write(out, encoding="unicode")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omnicontact", type=Path, default=Path.home() / "OmniContact_sim2sim")
    ap.add_argument("--ultra", type=Path, default=Path.home() / "ULTRA")
    ap.add_argument("--ibfm", type=Path, default=Path.home() / "I-BFM-Code")
    ap.add_argument("--out", type=Path, default=ROOT / "data/hoi/omnicontact")
    args = ap.parse_args()
    from supermimic.robot import G1_JOINT_NAMES
    cfg = yaml.safe_load((args.omnicontact / "policy/omnicontact/config/OmniContact.yaml").read_text())
    perm = np.asarray(cfg["lab2mj"], dtype=int)
    if sorted(perm.tolist()) != list(range(29)):
        raise ValueError("Invalid source joint permutation")
    manifest = {"schema": "supermimic-hoi-v1", "fps": 50, "quaternion_order": "wxyz",
                "joint_names": G1_JOINT_NAMES, "contact_order": ["left_foot", "right_foot", "left_hand", "right_hand"],
                "source": "https://github.com/Ingrid789/OmniContact_sim2sim",
                "source_commit": revision(args.omnicontact), "source_license": "CC-BY-NC-SA-4.0",
                "split_rule": "hold out the last 20% (at least one) complete source motions per family; no window leakage",
                "clips": [], "reviewed_projects": {}}
    for name, path in (("ULTRA", args.ultra), ("I-BFM", args.ibfm)):
        if path.is_dir():
            info = {"commit": revision(path), "included_in_training": False}
            if name == "ULTRA":
                info["g1_augmented_clips"] = len(list((path / "InterAct/OMOMO_retarget_aug").glob("*.pt")))
                info["reason"] = "Different 630D layout, object assets/scales and contact conventions need a separate adapter."
            else:
                info["reason"] = "Reviewed object context, capsule geometry and task metrics; no policy or replay buffer imported."
            manifest["reviewed_projects"][name] = info
    for family in ("pushbox", "carrybox"):
        files = sorted((args.omnicontact / "data" / family).glob("*.npz"))
        if len(files) < 2:
            raise ValueError(f"Need at least two independent {family} motions")
        nval = max(1, int(np.ceil(0.2 * len(files))))
        xml = args.omnicontact / "g1_description" / ("omnicontact_push_box_npz.xml" if family == "pushbox" else "omnicontact_carry_box.xml")
        box = ET.parse(xml).getroot().find(".//body[@name='box']")
        obj = {"half_size": [float(x) for x in box.find("geom").get("size").split()],
               "mass": float(box.find("inertial").get("mass")),
               "friction": float(box.find("geom").get("friction").split()[0]),
               "tables": family == "carrybox", "table_half_size": [0.2, 0.2, 0.01],
               "robot_palm_capsule_radius": 0.035,
               "physics_note": "Box inertia recomputed from size/mass; baseline G1 plus collision palms. Not an exact OmniContact simulator reproduction."}
        for index, src in enumerate(files):
            split = "val" if index >= len(files) - nval else "train"
            dst = args.out / family / split
            dst.mkdir(parents=True, exist_ok=True)
            with np.load(src, allow_pickle=False) as d:
                if not np.isclose(float(d["fps"].item()), 50):
                    raise ValueError(f"Expected 50 Hz: {src}")
                p = d["body_pos_w"][:, 0].astype(np.float32)
                quat = normal_quat(d["body_quat_w"][:, 0])
                q = d["joint_pos"][:, perm].astype(np.float32)
                T = len(q)
                offset = p[0].copy(); offset[2] = 0
                contact = d["contact_info"].reshape(T, 4).astype(np.float32)
                if not np.isin(contact, (0, 1)).all():
                    raise ValueError(f"Non-binary contacts: {src}")
                arrays = {"root_p": p - offset, "root_quat": quat, "q": q,
                          "object_p": d["object_pos_w"].astype(np.float32) - offset,
                          "object_quat": normal_quat(d["object_quat_w"]), "contact": contact,
                          "table_p": np.stack((d["table1_pos_w"], d["table2_pos_w"]), 1).astype(np.float32) - offset}
                for key, a in arrays.items():
                    if len(a) != T or not np.isfinite(a).all():
                        raise ValueError(f"Invalid {key} in {src}")
                if T <= 33:
                    raise ValueError(f"Clip too short for imagination: {src}")
                # Baseline MotionLib CSV convention is xyzw; the sidecar uses explicit wxyz.
                csv = np.concatenate((arrays["root_p"], quat[:, [1, 2, 3, 0]], q), -1)
                np.savetxt(dst / f"{src.stem}.csv", csv, delimiter=",", fmt="%.9g")
                np.savez_compressed(dst / src.name, **arrays)
                manifest["clips"].append({"name": src.stem, "family": family, "split": split,
                    "frames": T, "duration_s": (T - 1) / 50, "source_file": str(src.relative_to(args.omnicontact)),
                    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                    "hand_contact_fraction": float(contact[:, 2:].mean()),
                    "object_displacement_m": float(np.linalg.norm(arrays["object_p"][-1] - arrays["object_p"][0]))})
            (dst / "object.json").write_text(json.dumps(obj, indent=2) + "\n")
    manifest["robot_asset"] = str(prepare_assets().relative_to(ROOT))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"out": str(args.out), "clips": len(manifest["clips"]),
               "duration_s": sum(x["duration_s"] for x in manifest["clips"]),
               "splits": {f"{f}/{s}": sum(x["family"] == f and x["split"] == s for x in manifest["clips"])
                          for f in ("pushbox", "carrybox") for s in ("train", "val")}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
