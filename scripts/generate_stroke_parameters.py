import openpyxl
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import matplotlib.pyplot as plt
calibrated_pos = [[107.17, 60.05, [108.46095915113699, 124.69735596866342, 123.8492159154188, 107.34437453563503]], [143.15, 59.65, [108.26844456225734, 125.35190557085421, 126.7369347486135, 107.8449124667221]], [179.13, 58.18, [108.92299416444814, 128.00860689739335, 129.8556710884638, 108.88449124667221]], [215.12, 55.52, [110.27059628660568, 131.3198578261233, 133.01291034609002, 109.77005835551859]], [251.1, 52.47, [109.80856127329452, 131.12734323724365, 131.08776445729356, 109.11550875332779]]]
start_dist = 107.2
max_dist = 107.2 + 180  # 287.2

def read_stroke_params(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}

    headers = rows[0]
    result = {h: [] for h in headers if h is not None}

    for row in rows[1:]:
        for h, val in zip(headers, row):
            if h is not None and val is not None:
                result[h].append(float(val))

    return result

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

if __name__ == "__main__":
    path = "./stroke_params.xlsx"
    params = read_stroke_params(path)
    for k, v in params.items():
        print(f"{k}: {v}")
    start_mm_list = params['start']
    start_mm_list = [x+start_dist for x in start_mm_list]
    dir_list = params['dir']
    # convert to integer
    dir_list = [int(d) for d in dir_list]
    speed_list = params['speed']
    total_time_list = params['time']
    

    n_trial = len(start_mm_list)
    fig, axes = plt.subplots(n_trial, 6, figsize=(24, 3 * n_trial))
    if n_trial == 1:
        axes = [axes]  # keep consistent indexing
    col_titles = ["base_mm", "vert_mm", "cable 1", "cable 2", "cable 3", "cable 4"]
    cable_colors = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

    for i in range(n_trial):
        wp   = generate_waypoints(calibrated_pos, start_mm_list[i], speed_list[i], dir_list[i], total_time_list[i])
        traj = generate_trajectory(wp, Hz=50, accel=20.0)
        print(f"Trial {i}: start={start_mm_list[i]}  dir={dir_list[i]}  "
              f"speed={speed_list[i]}  time={total_time_list[i]}  "
              f"waypoints={len(wp)}  samples={len(traj)}")
        # print(f"Waypoints for trial {i}:")
        # for j, wp_point in enumerate(wp):
        #     print(f"  {j}: t={wp_point[0]}, pos={wp_point[1]}")

        n      = len(traj)
        t_traj = np.arange(n) / 50
        b_traj = np.array([s[0] for s in traj])
        v_traj = np.array([s[1] for s in traj])
        cl_traj = np.array([s[2] for s in traj])

        wp_bases  = [w[1][0] for w in wp]
        wp_verts  = [w[1][1] for w in wp]
        wp_cables = [[w[1][2][ci] for w in wp] for ci in range(4)]

        # Map each waypoint to its trajectory time by matching closest base_mm in traj.
        # wp_traj_times are constant-speed times; traj uses trapezoidal profile, so times differ.
        wp_traj_times = []
        for wb in wp_bases:
            idx = int(np.argmin(np.abs(b_traj - wb)))
            wp_traj_times.append(t_traj[idx])

        row = axes[i]

        # base_mm
        ax = row[0]
        ax.plot(t_traj, b_traj, color='tab:blue', linewidth=1)
        ax.scatter(wp_traj_times, wp_bases, color='tab:blue', s=30, zorder=5)
        ax.axhline(start_dist, color='r', linestyle='--', linewidth=0.7)
        ax.axhline(max_dist,   color='g', linestyle='--', linewidth=0.7)
        ax.set_ylabel(f"trial {i}", fontsize=8)
        ax.set_title(col_titles[0] if i == 0 else "")
        ax.set_xlabel("time (s)")

        # vert_mm
        ax = row[1]
        ax.plot(t_traj, v_traj, color='tab:orange', linewidth=1)
        ax.scatter(wp_traj_times, wp_verts, color='tab:orange', s=30, zorder=5)
        ax.set_title(col_titles[1] if i == 0 else "")
        ax.set_xlabel("time (s)")

        # cable lengths
        for ci in range(4):
            ax = row[2 + ci]
            ax.plot(t_traj, cl_traj[:, ci], color=cable_colors[ci], linewidth=1)
            ax.scatter(wp_traj_times, wp_cables[ci], color=cable_colors[ci], s=30, zorder=5)
            ax.set_title(col_titles[2 + ci] if i == 0 else "")
            ax.set_xlabel("time (s)")

    plt.suptitle("Stroke trajectories", fontsize=13)
    plt.tight_layout()
    plt.show()

