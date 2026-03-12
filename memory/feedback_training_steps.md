---
name: Per-task total_timesteps
description: How many training steps to use per task in REP experiments
type: feedback
---

StackCube-v1 uses --total_timesteps 40000000 (40M steps).
All other tasks (PickCube-v1, PushCube-v1, LiftPegUpright-v1, PlaceSphere-v1) use --total_timesteps 5000000 (5M steps).

**Why:** StackCube is a harder hierarchical task that requires more steps to converge.

**How to apply:** Always use 20M for StackCube, 5M for everything else, unless the user says otherwise.
