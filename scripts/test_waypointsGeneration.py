import numpy as np
import matplotlib.pyplot as plt

start_dist = 107.2
max_dist = 107.2 + 180  # 287.2

def generate_waypoints(calibrated_poses, start_base_pos, speed, dir, total_time):
    # calibrated poses is in the form of [pose1, pose2, ...], where each pos is [base_mm, vert_mm, cable_lengths]
    # start_base_pos is between start_dist to max_dist
    # return: list of lists in the form [[t1, cmd1], [t2, cmd2], ...], where cmd is in the form of [base_cmd, vert_cmd, cable_cmds]
    def get_interpolated_cmd(base_pos):
        n_cali = len(calibrated_poses)
        if base_pos >= calibrated_poses[-1][0]:
            return n_cali - 1, calibrated_poses[-1]
        if base_pos <= calibrated_poses[0][0]:
            return 0, calibrated_poses[0]
        for i in range(n_cali - 1):
            if calibrated_poses[i][0] <= base_pos <= calibrated_poses[i + 1][0]:
                t = (base_pos - calibrated_poses[i][0]) / (calibrated_poses[i + 1][0] - calibrated_poses[i][0])
                base_cmd = calibrated_poses[i][0] + t * (calibrated_poses[i + 1][0] - calibrated_poses[i][0])
                vert_cmd = calibrated_poses[i][1] + t * (calibrated_poses[i + 1][1] - calibrated_poses[i][1])
                cable_cmds = [calibrated_poses[i][2][j] + t * (calibrated_poses[i + 1][2][j] - calibrated_poses[i][2][j]) for j in range(4)]
                return i, [base_cmd, vert_cmd, cable_cmds]

    # --- Step 1: compute all segments by simulating bouncing ---
    segments = []
    cur_pos = start_base_pos
    cur_dir = dir
    dist_remaining = speed * total_time
    while dist_remaining > 1e-9:
        dist_to_wall = (max_dist - cur_pos) if cur_dir == 1 else (cur_pos - start_dist)
        if dist_remaining <= dist_to_wall:
            end_pos = cur_pos + cur_dir * dist_remaining
            segments.append((cur_pos, end_pos, cur_dir))
            dist_remaining = 0
        else:
            end_pos = max_dist if cur_dir == 1 else start_dist
            segments.append((cur_pos, end_pos, cur_dir))
            dist_remaining -= dist_to_wall
            cur_pos = end_pos
            cur_dir = -cur_dir

    # --- Step 2: build waypoints by walking calibrated poses through each segment ---
    _, cmd_start = get_interpolated_cmd(start_base_pos)
    waypoints = [[0, cmd_start]]
    t_cur = 0.0

    for seg_start, seg_end, seg_dir in segments:
        seg_dist = abs(seg_end - seg_start)
        idx_seg_start, _ = get_interpolated_cmd(seg_start)
        idx_seg_end, cmd_seg_end = get_interpolated_cmd(seg_end)

        if seg_dir == 1:
            for j in range(idx_seg_start + 1, idx_seg_end + 1):
                dist_from_seg_start = calibrated_poses[j][0] - seg_start
                t_cur_wp = t_cur + dist_from_seg_start / speed
                waypoints.append([t_cur_wp, calibrated_poses[j]])
        else:
            for j in range(idx_seg_start, idx_seg_end, -1):
                dist_from_seg_start = seg_start - calibrated_poses[j][0]
                t_cur_wp = t_cur + dist_from_seg_start / speed
                waypoints.append([t_cur_wp, calibrated_poses[j]])

        t_cur += seg_dist / speed
        waypoints.append([t_cur, cmd_seg_end])

    return waypoints

def generate_trajectory(waypoints, Hz=50, accel=20.0):
        # waypoints: [[t, [base_mm, vert_mm, cable_cmds]], ...]
        # Returns list of [base_mm, vert_mm, cable_cmds] at 1/Hz intervals,
        # with a trapezoidal speed profile (0 -> cruise -> 0) applied to arc length.

        bases  = [w[1][0] for w in waypoints]
        verts  = [w[1][1] for w in waypoints]
        cables = [w[1][2] for w in waypoints]

        # cumulative arc length along the base axis
        s_vals = [0.0]
        for i in range(1, len(waypoints)):
            s_vals.append(s_vals[-1] + abs(bases[i] - bases[i - 1]))
        s_total = s_vals[-1]

        # cruise speed derived from waypoints (constant-speed total time)
        v_cruise = s_total / waypoints[-1][0]

        # trapezoidal profile parameters
        t_ramp = v_cruise / accel
        d_ramp = 0.5 * v_cruise * t_ramp

        if 2 * d_ramp >= s_total:
            # distance too short for full ramp: use triangular profile
            v_peak = (accel * s_total) ** 0.5
            t_ramp = v_peak / accel
            t_total = 2 * t_ramp
            def s_at_t(t):
                t = min(t, t_total)
                if t <= t_ramp:
                    return 0.5 * accel * t ** 2
                return s_total - 0.5 * accel * (t_total - t) ** 2
        else:
            d_cruise = s_total - 2 * d_ramp
            t_cruise = d_cruise / v_cruise
            t_total = 2 * t_ramp + t_cruise
            def s_at_t(t):
                t = min(t, t_total)
                if t <= t_ramp:
                    return 0.5 * accel * t ** 2
                if t <= t_ramp + t_cruise:
                    return d_ramp + v_cruise * (t - t_ramp)
                return s_total - 0.5 * accel * (t_total - t) ** 2

        def cmd_at_s(s):
            s = max(0.0, min(s, s_total))
            for i in range(len(s_vals) - 1):
                if s_vals[i] <= s <= s_vals[i + 1]:
                    ds = s_vals[i + 1] - s_vals[i]
                    alpha = (s - s_vals[i]) / ds if ds > 1e-12 else 0.0
                    base = bases[i]  + alpha * (bases[i + 1]  - bases[i])
                    vert = verts[i]  + alpha * (verts[i + 1]  - verts[i])
                    cl   = [cables[i][j] + alpha * (cables[i + 1][j] - cables[i][j]) for j in range(4)]
                    return [base, vert, cl]
            return [bases[-1], verts[-1], cables[-1]]

        n_samples = int(np.ceil(t_total * Hz)) + 1
        trajectory = []
        for k in range(n_samples):
            s = s_at_t(k / Hz)
            trajectory.append(cmd_at_s(s))

        return trajectory
# --- Build calibrated poses ---
start_cali = 107.2
end_cali = 107.2 + 180
n_cali = 5

base_positions = np.linspace(start_cali, end_cali, n_cali)
# vert_mm: dips in the middle (bowl shape), 100 at ends, 80 at center
vert_values = [100.0, 90.0, 80.0, 90.0, 100.0]
# cable lengths: cables 1&4 shorten toward center, cables 2&3 lengthen toward center
cable_values = [
    [110.0, 120.0, 140.0, 110.0],  # pose 0
    [105.0, 115.0, 125.0, 105.0],  # pose 1
    [120.0, 130.0, 130.0, 120.0],  # pose 2 (center)
    [105.0, 125.0, 125.0, 125.0],  # pose 3
    [110.0, 120.0, 115.0, 110.0],  # pose 4
]
calibrated_poses = [
    [base_positions[i], vert_values[i], cable_values[i]]
    for i in range(n_cali)
]

print("Calibrated poses:")
for p in calibrated_poses:
    print(f"  base={p[0]:.1f}  vert={p[1]:.1f}  cables={p[2]}")

# --- Test cases ---
test_cases = [
    {"label": "No turn (fwd)",            "start": 107.2, "speed": 10, "dir":  1, "total_time": 5},
    {"label": "No turn (bwd)",            "start": 287.2, "speed": 10, "dir": -1, "total_time": 5},
    {"label": "One turn (fwd→bwd)",       "start": 200.0, "speed": 10, "dir":  1, "total_time": 20},
    {"label": "One turn (bwd→fwd)",       "start": 200.0, "speed": 10, "dir": -1, "total_time": 20},
    {"label": "Multi turn",               "start": 150.0, "speed": 10, "dir":  1, "total_time": 60},
    # start=170 is between pose1(152.2) and pose2(197.2); total_dist=30 ends at 200, between pose2 and pose3(242.2)
    {"label": "Start & end between poses","start": 170.0, "speed": 5, "dir":  1, "total_time": 1},
]

n_cols = 3 + 4  # base, vert, + 4 cables
fig, axes = plt.subplots(len(test_cases), n_cols, figsize=(4 * n_cols, 3 * len(test_cases)))
fig.suptitle("Waypoints: base_mm | vert_mm | cable lengths", fontsize=13)

col_labels = ["base_mm", "vert_mm", "cable 1", "cable 2", "cable 3", "cable 4", "overview"]

for row, tc in enumerate(test_cases):
    wp = generate_waypoints(calibrated_poses, tc["start"], tc["speed"], tc["dir"], tc["total_time"])
    times   = [w[0] for w in wp]
    bases   = [w[1][0] for w in wp]
    verts   = [w[1][1] for w in wp]
    cables  = [[w[1][2][i] for w in wp] for i in range(4)]

    n_turns = sum(1 for i in range(1, len(times) - 1)
                  if abs(bases[i] - start_dist) < 1e-6 or abs(bases[i] - max_dist) < 1e-6)
    total_t = times[-1]
    dt_min  = min(times[i+1] - times[i] for i in range(len(times)-1))
    dt_max  = max(times[i+1] - times[i] for i in range(len(times)-1))

    print(f"\n[{tc['label']}]  speed={tc['speed']} mm/s  dir={tc['dir']}  total_time={tc['total_time']}s")
    print(f"  waypoints: {len(wp)}  |  turns: {n_turns}  |  total_time: {total_t:.3f}s  |  dt range: [{dt_min:.3f}, {dt_max:.3f}]s")
    for w in wp:
        print(f"  t={w[0]:.3f}s  base={w[1][0]:.2f}  vert={w[1][1]:.2f}  cables={[round(c,2) for c in w[1][2]]}")

    # base_mm
    ax = axes[row][0]
    ax.plot(times, bases, 'o-', markersize=4, color='tab:blue')
    ax.axhline(start_dist, color='r', linestyle='--', linewidth=0.8)
    ax.axhline(max_dist,   color='g', linestyle='--', linewidth=0.8)
    ax.set_ylabel(tc["label"], fontsize=8)
    ax.set_title("base_mm" if row == 0 else "")
    ax.set_xlabel("time (s)")
    ax.set_ylim(start_dist - 10, max_dist + 10)

    # vert_mm
    ax = axes[row][1]
    ax.plot(times, verts, 's-', markersize=4, color='tab:orange')
    ax.set_title("vert_mm" if row == 0 else "")
    ax.set_xlabel("time (s)")

    # cable lengths
    colors = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for ci in range(4):
        ax = axes[row][2 + ci]
        ax.plot(times, cables[ci], 'd-', markersize=4, color=colors[ci])
        ax.set_title(f"cable {ci+1}" if row == 0 else "")
        ax.set_xlabel("time (s)")

    # overview: all channels on one plot
    ax = axes[row][6]
    ax.plot(times, bases,  label='base',   color='tab:blue')
    ax.plot(times, verts,  label='vert',   color='tab:orange')
    for ci in range(4):
        ax.plot(times, cables[ci], label=f'cl{ci+1}', color=colors[ci], linestyle='--', linewidth=0.8)
    ax.set_title("overview" if row == 0 else "")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=6, ncol=2)

plt.tight_layout()
plt.show()

# ── Trajectory figure (one case) ──────────────────────────────────────────
Hz    = 50
accel = 20.0
tc = test_cases[4]  # change index to pick a different case

wp   = generate_waypoints(calibrated_poses, tc["start"], tc["speed"], tc["dir"], tc["total_time"])
traj = generate_trajectory(wp, Hz=Hz, accel=accel)

n         = len(traj)
t_traj    = np.arange(n) / Hz
b_traj    = np.array([s[0] for s in traj])
v_traj    = np.array([s[1] for s in traj])
cl_traj   = np.array([s[2] for s in traj])   # (n, 4)
speed_traj = np.abs(np.gradient(b_traj, t_traj))

wp_times  = [w[0] for w in wp]
wp_bases  = [w[1][0] for w in wp]
wp_verts  = [w[1][1] for w in wp]
wp_cables = [[w[1][2][i] for w in wp] for i in range(4)]

print(f"\n[TRAJ {tc['label']}]  samples={n}  t_total={t_traj[-1]:.3f}s  "
      f"peak_speed={speed_traj.max():.2f} mm/s  "
      f"mean_speed={speed_traj[5:-5].mean():.2f} mm/s (excl. ramps)")

def annotate_wp(ax, times, values, color, fmt="{:.1f}", y_offset=4):
    for i, (t, v) in enumerate(zip(times, values)):
        ax.annotate(f"wp{i}\n{fmt.format(v)}",
                    xy=(t, v), xytext=(0, y_offset),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6, color=color,
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.5))

cable_colors = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
fig2, axes2 = plt.subplots(1, 6, figsize=(26, 4))
fig2.suptitle(f"Trajectory — {tc['label']}  (Hz={Hz}, accel={accel} mm/s²)", fontsize=11)

# base_mm + speed
ax = axes2[0]
ax.plot(t_traj, b_traj, color='tab:blue', linewidth=1, label='base_mm')
ax.scatter(wp_times, wp_bases, color='tab:blue', s=40, zorder=5, label='waypts')
annotate_wp(ax, wp_times, wp_bases, 'tab:blue')
ax.axhline(start_dist, color='r', linestyle='--', linewidth=0.7, label='start_dist')
ax.axhline(max_dist,   color='g', linestyle='--', linewidth=0.7, label='max_dist')
ax.set_ylim(start_dist - 20, max_dist + 20)
ax_spd = ax.twinx()
ax_spd.plot(t_traj, speed_traj, color='gray', linewidth=0.8, linestyle='--', label='speed')
ax_spd.set_ylabel("speed (mm/s)", fontsize=7, color='gray')
ax.set_title("base_mm + speed")
ax.set_xlabel("time (s)")
ax.legend(fontsize=6, loc='upper left')

# vert_mm
ax = axes2[1]
ax.plot(t_traj, v_traj, color='tab:orange', linewidth=1)
ax.scatter(wp_times, wp_verts, color='tab:orange', s=40, zorder=5)
annotate_wp(ax, wp_times, wp_verts, 'tab:orange')
ax.set_title("vert_mm")
ax.set_xlabel("time (s)")

# cable lengths
for ci in range(4):
    ax = axes2[2 + ci]
    ax.plot(t_traj, cl_traj[:, ci], color=cable_colors[ci], linewidth=1)
    ax.scatter(wp_times, wp_cables[ci], color=cable_colors[ci], s=40, zorder=5)
    annotate_wp(ax, wp_times, wp_cables[ci], cable_colors[ci])
    ax.set_title(f"cable {ci+1}")
    ax.set_xlabel("time (s)")

plt.tight_layout()
plt.show()
