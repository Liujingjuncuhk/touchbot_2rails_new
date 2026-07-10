import sys
import os
import time

import openpyxl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import touchbot_controller
import numpy
import pickle

# calibrated_pos = [[107.17, 60.05, [108.46095915113699, 124.69735596866342, 123.8492159154188, 107.34437453563503]], [143.15, 59.65, [108.26844456225734, 125.35190557085421, 126.7369347486135, 107.8449124667221]], [179.13, 58.18, [108.92299416444814, 128.00860689739335, 129.8556710884638, 108.88449124667221]], [215.12, 55.52, [110.27059628660568, 131.3198578261233, 133.01291034609002, 109.77005835551859]], [251.1, 52.47, [109.80856127329452, 131.12734323724365, 131.08776445729356, 109.11550875332779]]]
# xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stroke_params.xlsx")
xlsx_path = "./stroke_params.xlsx"


def plot_force_logs(force_list):
    _, axes = plt.subplots(len(force_list), 1, figsize=(10, 3 * len(force_list)), squeeze=False)
    for i, force_log in enumerate(force_list):
        t0 = force_log[0][0]
        ts = [entry[0] - t0 for entry in force_log]
        forces = [entry[1] for entry in force_log]
        axes[i][0].plot(ts, forces)
        axes[i][0].set_title(f"Trial {i + 1}")
        axes[i][0].set_xlabel("Time (s)")
        axes[i][0].set_ylabel("Touch Force (N)")
        axes[i][0].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    touchbot = touchbot_controller.touchbotController()
    time.sleep(0.1)
    touchbot.calibrate_motor_pos()

    touchbot.send_length_cmd_rel_timed([0, -30, -30, 0], 3)
    time.sleep(0.1)
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    time.sleep(0.1)
    touchbot.send_stepper_cmd_abs(0, 110, 10, 20)
    touchbot.mc.wait_until_reached(0, 110)
    params = touchbot.read_stroke_params(xlsx_path)
    for k, v in params.items():
        print(f"{k}: {v}")
    start_mm_list = params['start']
    start_mm_list = [x+touchbot.start_dist for x in start_mm_list]
    dir_list = params['dir']
    # convert to integer
    dir_list = [int(d) for d in dir_list]
    speed_list = params['speed']
    total_time_list = params['time']
    nTraj = len(start_mm_list)

    force_list = []
    touchbot.checkpoint("put your arm on the touch sensor")

    pts_all = touchbot.take_arm_contour()
    calibrated_pos = touchbot.perform_calibration(pts_all, 1)

    wp_list = []
    traj_list = []
    for i in range(nTraj):
        wp   = touchbot.generate_waypoints(calibrated_pos, start_mm_list[i], speed_list[i], dir_list[i], total_time_list[i])
        traj = touchbot.generate_trajectory(wp, Hz=50, accel=40.0)
        wp_list.append(wp)
        traj_list.append(traj)
    for i in range(nTraj):
        wp = wp_list[i]
        traj = traj_list[i]
        traj_start_pos = wp[0][1]
        # move to the start position
        touchbot.send_cmd_abs_timed(traj_start_pos[0], traj_start_pos[1]+20, traj_start_pos[2], 1)
        time.sleep(1)
        
        arm_weight = touchbot.calibrate_arm_weight(total_time=1)
        touchbot.send_cmd_abs_timed(traj_start_pos[0], traj_start_pos[1], traj_start_pos[2], 1)
        touchbot.mc.wait_until_reached(traj_start_pos[0], traj_start_pos[1])
        # touchbot.execute_waypoints(wp)
        force_log = touchbot.execute_trajectory_record_force(traj, arm_weight)
        force_list.append(force_log)

        time.sleep(0.1)
        touchbot.send_stepper_cmd_rel_timed(0, 20, 1)
        time.sleep(1)
        # touchbot.checkpoint("go to next trajectory")

        # execute the trajectory
    touchbot.close_all()
    plot_force_logs(force_list)
