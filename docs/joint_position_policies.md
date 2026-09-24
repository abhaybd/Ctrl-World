# Joint position policies in Ctrl-World (tabled)

`scripts/rollout_interact_molmobot_pi0.py` only supports joint velocity policies (`pi0_droid`, `pi05_droid`,
`pi0_fast_droid`). It raises `NotImplementedError` for joint position policies like MolmoBot-Pi0-DROID. This doc
collects the options we explored for them.

## The problem

Ctrl-World is conditioned on end effector poses at 5hz. Joint velocity policies go through Ctrl-World's action adapter
(`models/action_adapter/model2_15_9.pth`). Given the current joints and 15 steps of DROID joint velocity, the adapter
predicts the joint positions a real DROID robot reaches, which are then turned into poses with forward kinematics.

MolmoBot-Pi0 predicts absolute joint position targets instead. On the real robot the arm lags behind its targets by
about 0.023 rad on average, so the targets aren't where the robot actually is.

DROID's joint velocity is the delta from the *current* joint position divided by 0.2, scaled down so each joint is at
most 1 (`joint_velocity_to_delta` in `droid/robot_ik/robot_ik_solver.py`). RoboRollout logs it the same way, as
`arm_vel = (cmd - obs) / 0.2`. So converting a chunk of targets to velocity needs the robot state at every step of the
chunk, which isn't known ahead of time.

## Options

1. **Direct FK.** Assume each target is reached on the next step (`q[k+1] = cmd[k]`) and use FK on the targets directly,
   without the adapter. This is simple, but ignores the tracking lag.
2. **Adapter, differencing.** Convert with `v[k] = (cmd[k] - cmd[k-1]) / 0.2`, i.e. take each delta from the previous
   target, then use the adapter. Because the targets move smoothly, these velocities are about 2.5x smaller than the
   ones DROID commands on the robot, so the adapter predicts too little motion.
3. **Adapter, fixed point iteration.** Solve for velocities that agree with the positions the adapter predicts for them.
   Start from the differencing guess, then repeat `q = [q0, adapter(q0, v)[:-1]]` and
   `v = 0.5 * v + 0.5 * scale((cmd - q) / 0.2)` 20 times. Without the damping the updates oscillate. With it they
   converge (the 90th percentile of the last update's change is 0.003), but there's no guarantee: the adapter is a
   non-causal MLP over the whole chunk, and the clipping isn't smooth.
4. **Direct FK with a fitted lag.** Like 1, but `q[k+1] = cmd[k - L]` with a lag `L` fit to real rollouts. Not evaluated.
5. **Causal adapter.** Retrain the adapter to predict one step at a time from `(q[k], v[k])`, then roll it out with
   `v[k] = scale((cmd[k] - q[k]) / 0.2)`, exactly like the real control loop. This is the principled fix, but needs
   retraining on DROID.

Option 3 is the implementation that was removed:

```python
def joint_targets_to_vel(dynamics_model, joints, joint_targets, num_iters=20, damping=0.5):
    # joints (7,) current joint position, joint_targets (15, 7) absolute joint targets
    joint_vel = droid_scale_vel((joint_targets - np.concatenate([joints[None], joint_targets[:-1]], axis=0)) / 0.2)
    for _ in range(num_iters):
        joint_pos = np.concatenate([joints[None], dynamics_model(joints[None], joint_vel, None, training=False)[:-1]], axis=0)
        joint_vel = (1 - damping) * joint_vel + damping * droid_scale_vel((joint_targets - joint_pos) / 0.2)
    return joint_vel
```

## Evaluation

The evaluation used 12 real rollouts from `adeshpande-princeton-university/synthetic-wm-evals`, all driven by
MolmoBot-DROID, which also commands absolute joint positions. It covered 1215 windows with the arm moving (starting
every 3 steps). Each method got the logged commands and the joint state at the start of the window. Its predictions
were compared with the logged joint positions after 3, 6, 9 and 12 policy steps; 12 steps is one Ctrl-World interaction.

End effector position error (cm), mean [median]:

| method                  | +3          | +6          | +9          | +12         |
|-------------------------|-------------|-------------|-------------|-------------|
| direct FK               | 1.36 [1.24] | 1.34 [1.23] | 1.33 [1.22] | 1.31 [1.19] |
| adapter, fixed point    | 0.82 [0.56] | 0.90 [0.54] | 0.91 [0.58] | 0.97 [0.63] |
| adapter, differencing   | 0.65 [0.54] | 1.15 [0.89] | 2.07 [1.73] | 2.92 [2.49] |
| no motion (`q = q0`)    | 1.46 [1.06] | 2.68 [2.06] | 3.82 [2.99] | 4.88 [3.79] |

End effector rotation error (deg), mean [median]:

| method                  | +3          | +6          | +9          | +12         |
|-------------------------|-------------|-------------|-------------|-------------|
| direct FK               | 1.72 [1.00] | 1.73 [1.00] | 1.74 [0.99] | 1.72 [0.98] |
| adapter, fixed point    | 1.02 [0.73] | 1.08 [0.82] | 1.11 [0.82] | 1.09 [0.78] |
| adapter, differencing   | 0.78 [0.61] | 1.47 [1.05] | 2.18 [1.27] | 2.94 [1.67] |
| no motion (`q = q0`)    | 1.47 [0.62] | 2.61 [1.16] | 3.67 [1.73] | 4.66 [2.33] |

Joint error (rad, mean over joints), mean:

| method                  | +3    | +6    | +9    | +12   |
|-------------------------|-------|-------|-------|-------|
| direct FK               | 0.021 | 0.021 | 0.021 | 0.021 |
| adapter, fixed point    | 0.010 | 0.012 | 0.013 | 0.014 |
| adapter, differencing   | 0.008 | 0.015 | 0.027 | 0.038 |
| no motion (`q = q0`)    | 0.017 | 0.031 | 0.044 | 0.057 |

On two of those runs, stepping every 5 steps and measuring over the full 15 step adapter horizon:

- Feeding the logged `arm_vel` through the adapter has 0.0168 rad joint error.
- The fixed point has 0.0128 rad, and its velocities are about as large as the logged ones (mean |v| 0.144 vs 0.131).
- Differencing has 0.0252 rad, with mean |v| 0.057.

Takeaways:

- Direct FK's error is roughly constant: it's the tracking lag, and it doesn't compound within a chunk.
- The fixed point is 30-50% more accurate, and is the only method that models the lag.
- Differencing is fine for the first few steps, then drifts.
- Caveat: the data is from MolmoBot-DROID, not MolmoBot-Pi0. The action space is the same, but the lag may differ
  somewhat.
