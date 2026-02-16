#!/usr/bin/env python3
"""
Combine dataset/ and dataset2/ into a single combined_dataset/ directory.

dataset/  has iter_0000 .. iter_0099  → copied as iter_0000 .. iter_0099
dataset2/ has iter_0000 .. iter_0054  → copied as iter_0100 .. iter_0154

Uses shutil.copytree so the combined directory is fully self-contained.
"""

import os
import shutil
from tqdm import tqdm

BASE = "/Users/darshiljariwala/Desktop/Robot-Sim/pybullet/old_setup"

DATASET1 = os.path.join(BASE, "dataset")
DATASET2 = os.path.join(BASE, "dataset2")
COMBINED = os.path.join(BASE, "combined_dataset")

# ── Discover trajectories ────────────────────────────────────────────────────
trajs1 = sorted(
    d for d in os.listdir(DATASET1)
    if os.path.isdir(os.path.join(DATASET1, d)) and d.startswith("iter_")
)
trajs2 = sorted(
    d for d in os.listdir(DATASET2)
    if os.path.isdir(os.path.join(DATASET2, d)) and d.startswith("iter_")
)

offset = len(trajs1)  # dataset2 indices start after dataset1
total = len(trajs1) + len(trajs2)

print(f"dataset  : {len(trajs1)} trajectories")
print(f"dataset2 : {len(trajs2)} trajectories")
print(f"combined : {total} trajectories (offset for dataset2 = {offset})")

# ── Create combined directory ────────────────────────────────────────────────
if os.path.exists(COMBINED):
    print(f"Removing existing {COMBINED}")
    shutil.rmtree(COMBINED)

os.makedirs(COMBINED)

# ── Copy dataset1 trajectories (keep original numbering) ────────────────────
print("\nCopying dataset trajectories ...")
for traj in tqdm(trajs1, desc="dataset"):
    src = os.path.join(DATASET1, traj)
    dst = os.path.join(COMBINED, traj)
    shutil.copytree(src, dst)

# ── Copy dataset2 trajectories (renumber starting from offset) ──────────────
print("\nCopying dataset2 trajectories (renumbered) ...")
for i, traj in enumerate(tqdm(trajs2, desc="dataset2")):
    new_idx = offset + i
    new_name = f"iter_{new_idx:04d}"
    src = os.path.join(DATASET2, traj)
    dst = os.path.join(COMBINED, new_name)
    shutil.copytree(src, dst)

print(f"\n✅ Combined dataset created at {COMBINED}")
print(f"   Total trajectories: {total}")
