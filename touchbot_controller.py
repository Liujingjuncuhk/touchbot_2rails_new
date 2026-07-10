from random import random
import motor_commander
import cable_tension_sensor
import touch_force_sensor_single
import touch_force_sensor_array
import flexx2_driver
import palm_simulator
import numpy as np
import time
import threading
import pickle
import pyvista as pv
import openpyxl

class touchbotController():
    def __init__(self):
        self.mc = motor_commander.MotorCommander()
        self.cts = cable_tension_sensor.CableTensionSensor()
        # self.tfs = touch_force_sensor_single.TouchForceSensor()
        self.tfs_array = touch_force_sensor_array.TouchForceSensorArray()
        self.camera = flexx2_driver.Flexx2Driver()
        self.camera.connect()
        self.palm_file = 'models/palm_size3.pickle'
        self.palm = palm_simulator.PalmSimulator(self.palm_file)
        self.step_per_mm = 4096/(50.2*np.pi)
        self.moving_dir = [1, 1, -1, -1] # for motor 1,2: decrease means shorten, for 3,4: increase means shorten
        self.nCables = 4
        # self.motorPos_cali = [1540, 1615, 3281, 2980] # position
        self.motorPos_cali = [1580, 1062, 3167, 2522]
        # self.cableLength_cali = [112, 133, 131, 113] # mm
        self.cableLength_cali = [109, 120, 121 , 109]
        self.initial_state = self.get_state()
        self.initial_cl = self.initial_state["cable_lengths"]
        self.initial_motorPos = self.initial_state["servo_pos"]
        self.dist_2_first_loadcell = 108
        self.dist_between_loadcell = 57.5
        self.start_dist = 107.2
        self.max_dist = 107.2+180 # 187.2
        
        print("initial cl is: {}".format(self.initial_cl))

    def send_cmd_abs(self, base_mm: float, vert_mm: float, cable_lengths: list, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0, cl_speed: list = None):
        motor_cmd_abs = self.mc.initial_servo_pos.copy()
        for i in range(self.nCables):
            d_cl = cable_lengths[i] - self.cableLength_cali[i]
            motor_cmd_abs[i] = self.motorPos_cali[i] + d_cl * self.step_per_mm * self.moving_dir[i]
        servo_speeds = None
        if cl_speed is not None:
            servo_speeds = [abs(cl_speed[i] * self.step_per_mm) for i in range(self.nCables)]
        self.mc.send_command(
            base_mm=base_mm,
            vert_mm=vert_mm,
            servo_pos=motor_cmd_abs,
            base_speed_mm_s=base_speed_mm_s,
            vert_speed_mm_s=vert_speed_mm_s,
            servo_speeds=servo_speeds,
        )

    def send_cmd_abs_timed(self, base_mm: float, vert_mm: float, cable_lengths: list, consumed_time: float):
        # Implementation for sending absolute commands with timing
        # calculate the velocity
        cur_state = self.get_state()
        cur_base_mm = cur_state['base_mm']
        cur_vert_mm = cur_state['vert_mm']
        base_speed = (abs(cur_base_mm-base_mm) / consumed_time)
        vert_speed = (abs(cur_vert_mm-vert_mm) / consumed_time)
        cl_speed = [abs(cable_lengths[i] - cur_state['cable_lengths'][i]) / consumed_time for i in range(self.nCables)]
        self.send_cmd_abs(base_mm, vert_mm, cable_lengths, base_speed, vert_speed, cl_speed)

    def send_cmd_rel(self, base_mm: float, vert_mm: float, cable_lengths: list, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        cur_state = self.get_state()
        abs_base = cur_state['base_mm'] + base_mm
        abs_vert = cur_state['vert_mm'] + vert_mm
        abs_cl = [cur_state['cable_lengths'][i] + cable_lengths[i] for i in range(self.nCables)]
        self.send_cmd_abs(abs_base, abs_vert, abs_cl, base_speed_mm_s, vert_speed_mm_s)

    def send_stepper_cmd_abs(self, base_mm: float, vert_mm: float, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        # Implementation for sending absolute stepper commands
        # print("moving stepper abs: base_mm={}, vert_mm={}".format(base_mm, vert_mm))
        self.mc.send_stepper_only(base_mm=base_mm, vert_mm=vert_mm, base_speed_mm_s=base_speed_mm_s, vert_speed_mm_s=vert_speed_mm_s)

    def send_stepper_cmd_abs_timed(self, base_mm: float, vert_mm: float, consumed_time: float):
        # Implementation for sending absolute stepper commands with timing
        cur_state = self.get_state()
        cur_base_mm = cur_state['base_mm']
        cur_vert_mm = cur_state['vert_mm']
        base_speed = (abs(cur_base_mm-base_mm) / consumed_time)
        vert_speed = (abs(cur_vert_mm-vert_mm) / consumed_time)
        self.mc.send_stepper_only(base_mm=base_mm, vert_mm=vert_mm, base_speed_mm_s=base_speed, vert_speed_mm_s=vert_speed)

    def send_stepper_cmd_rel(self, base_mm: float, vert_mm: float, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        # Implementation for sending relative stepper commands
        self.mc.send_stepper_only_rel(base_mm=base_mm, vert_mm=vert_mm, base_speed_mm_s=base_speed_mm_s, vert_speed_mm_s=vert_speed_mm_s)

    def send_stepper_cmd_rel_timed(self, base_mm: float, vert_mm: float, consumed_time: float):
        # Implementation for sending relative stepper commands with timing
        base_speed = abs(base_mm / consumed_time)
        vert_speed = abs(vert_mm / consumed_time)
        self.mc.send_stepper_only_rel(base_mm=base_mm, vert_mm=vert_mm, base_speed_mm_s=base_speed, vert_speed_mm_s=vert_speed)

    def send_length_cmd_abs(self, cable_lengths: list, cl_speed: list = None):
        # Implementation for sending absolute length commands
        motor_cmd_abs = self.mc.initial_servo_pos.copy()
        for i in range(4):
            d_cl = cable_lengths[i] - self.cableLength_cali[i]
            motor_cmd_abs[i] = self.motorPos_cali[i] + d_cl * self.step_per_mm * self.moving_dir[i]
        if cl_speed is None:
            self.mc.send_servos_only(motor_cmd_abs)
        else:
            motor_speed = []
            for i in range(4):
                motor_speed.append(abs(cl_speed[i] * self.step_per_mm))
            self.mc.send_servos_only(motor_cmd_abs, motor_speed)

    def send_length_cmd_abs_timed(self, cable_lengths: list, consumed_time: float):
        # Implementation for sending absolute length commands with timing
        cur_cl = self.get_state()["cable_lengths"]
        d_cl = [0, 0, 0, 0]
        for i in range(4):
            d_cl[i] = cable_lengths[i] - cur_cl[i]
        self.send_length_cmd_rel_timed(d_cl, consumed_time)
        # time.sleep(consumed_time)

    def send_length_cmd_rel(self, cable_lengths: list, cl_speed: list = None):
        # Implementation for sending relative length commands
        motor_cmd_rel = [0 for _ in range(4)]
        for i in range(4):
            motor_cmd_rel[i] = cable_lengths[i] * self.step_per_mm * self.moving_dir[i]

        if cl_speed is None:
            self.mc.send_servos_only_rel(motor_cmd_rel)
        else:
            motor_speed = []
            for i in range(4):
                motor_speed.append(abs(cl_speed[i] * self.step_per_mm))
            self.mc.send_servos_only_rel(motor_cmd_rel, motor_speed)

    def get_cable_force(self, n = 10):
        forces = []
        for _ in range(n):
            forces.append(self.cts.read_forces())
        return [sum(f[i] for f in forces)/len(forces) for i in range(4)]

    def get_touch_force_array(self, calibrated_arm_weight):
        # Implementation for getting touch force array
        touch_force_array = self.tfs_array.read_forces()
        touch_force = 0
        for i in range(4):
            touch_force += touch_force_array[i]-calibrated_arm_weight[i]
        return touch_force

    def read_stroke_params(self, xlsx_path):
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

    def send_length_cmd_rel_timed(self, cable_lengths: list, consumed_time = 1.0):
        # Implementation for sending relative length commands
        motor_cmd_rel = [0 for _ in range(4)]
        for i in range(4):
            motor_cmd_rel[i] = cable_lengths[i] * self.step_per_mm * self.moving_dir[i]

        motor_speed = [0 for _ in range(4)]
        for i in range(4):
            motor_speed[i] = abs(cable_lengths[i] / consumed_time * self.step_per_mm)
        self.mc.send_servos_only_rel(motor_cmd_rel, motor_speed)
        time.sleep(consumed_time)

    def get_state(self):
        cur_cl = self.cableLength_cali.copy()
        all_feedbacks = self.mc.get_positions()
        servo_positions = all_feedbacks["servo_pos"]
        for i in range(self.nCables):
            cur_cl[i] = self.cableLength_cali[i] + (servo_positions[i] - self.motorPos_cali[i]) / (self.step_per_mm * self.moving_dir[i])
        return_dict = {
            'base_mm':   all_feedbacks['base_mm'],
            'vert_mm':   all_feedbacks['vert_mm'],
            'servo_pos': servo_positions,
            'cable_lengths': cur_cl,
            'timestamp': all_feedbacks['timestamp'],
        }
        return return_dict

    def calibrate_motor_pos(self):
        offset_unit = -2.0 # shorten 2mm every time
        offsets_list = [0.0 for i in range(self.nCables)]
        tightened = [0 for i in range(self.nCables)]
        calibrated = [0 for i in range(self.nCables)]
        threshold = 0.1
        while True:
            loadcell_reading = self.cts.read_forces()
            # print(loadcell_reading)
            offsets_list = [0 for i in range(self.nCables)]
            if not all(cal == 1 for cal in calibrated):
                if not all(t == 1 for t in tightened):
                    for i in range(self.nCables):
                        if not tightened[i]:
                            if loadcell_reading[i] > threshold:
                                tightened[i] = 1
                            else:
                                offsets_list[i] = offset_unit
                        else:
                            offsets_list[i] = 0
                    # print(offsets_list)
                    self.send_length_cmd_rel(offsets_list, [4,4,4,4])
                    time.sleep(0.5)
                else:
                    print("all tightened, loosing a little now")

                    calibrated = [1,1,1,1]
                    offsets_list = [1 for _ in range(self.nCables)]
                    self.send_length_cmd_rel(offsets_list, [4,4,4,4])
                    time.sleep(0.5)
            else:
                print("all calibrated")
                break
    

    def moveDownUntilTouched(self, initial_forces):
        # initial forces are cable forces
        touch_threshold = 0.2
        while 1:
            self.send_stepper_cmd_rel(0, -1, 0, 2)
            time.sleep(0.55)
            cur_force = self.cts.read_forces()
            numContact = 0
            for i in range(4):
                if initial_forces[i] - cur_force[i] > touch_threshold:
                    numContact += 1
            if numContact >= 2:
                print("contact detected")
                break
            if self.get_state()['vert_mm'] <= 1:
                print("reached bottom")
                break

    def adjust_touch_force(self, calibrated_weight, target_force):
        # this function is when the palm is in contact with human forearm, and adjust the touch force
        diff_range = 0.3
        K_ratio = 1
        Ke_list = [-1,-1,-1,-1]
        min_cable_force = [0.1,0.1,0.1,0.1]
        max_loop = 20
        for _ in range(max_loop):
            self.move_to_slack()
            self.enforce_min_cable_forces(min_cable_force)
            cur_contact_force = self.get_touch_force_array(calibrated_weight)
            print(f"cur force is: {cur_contact_force}")
            cur_state = self.get_state()
            cur_cable_length = cur_state["cable_lengths"]
            if abs(cur_contact_force - target_force) < diff_range:
                break
            if cur_contact_force > target_force+diff_range:
                # Adjust cable lengths to reduce contact force
                self.send_stepper_cmd_rel_timed(0,1,1)
                time.sleep(1.1)
            elif cur_contact_force < target_force-diff_range:
                # Adjust cable lengths to increase contact force
                self.send_stepper_cmd_rel_timed(0,-0.5,1)
                time.sleep(1.1)

    def perform_calibration(self, pts_all, target_force, n_cali = 5):
        if isinstance(pts_all, list):
            pts_all = np.vstack(pts_all)
        miny = np.min(pts_all[:,1])
        dis_between_cali = 180/5
        self.send_stepper_cmd_abs(self.start_dist, 110, 20,20)
        self.mc.wait_until_reached(self.start_dist, 110)
        self.send_stepper_cmd_rel_timed(0, -(max(80+miny,30)),2)
        time.sleep(2.2)
        calibrated_info = []
        
        for i in range(n_cali):
            initial_cable_force = self.get_cable_force()
            self.moveDownUntilTouched(initial_cable_force)
            self.send_stepper_cmd_rel_timed(0, 10,1)
            time.sleep(1.1)
            initial_cable_force = self.get_cable_force()
            arm_weight = self.tfs_array.read_forces()
            self.moveDownUntilTouched(initial_cable_force)
            self.adjust_touch_force(arm_weight, target_force)
            time.sleep(0.5)
            cur_state = self.get_state()
            calibrated_info.append([cur_state['base_mm'], cur_state['vert_mm'], cur_state['cable_lengths']])
            self.send_stepper_cmd_rel_timed(0, 20, 1)
            time.sleep(1)
            if i != n_cali-1:
                self.send_stepper_cmd_rel_timed(dis_between_cali, 0, 2)
                time.sleep(2.2)

        
        print(calibrated_info)
        return calibrated_info


    # def collectData_length_force_nocontact(self):
    #     num_iter = 3
    #     shorten_dist = 3 # each iteration for cable 1,4
    #     shorten_dist_23 = 2 # shoten distance for cable 2,3
    #     default_speed = 5
    #     cl_list = []
    #     motorPos_list = []
    #     force_list = []
    #     def append_data():
    #         cur_state = self.get_state()
    #         cl_list.append(cur_state["cable_lengths"])
    #         motorPos_list.append(cur_state["servo_pos"])
    #         force_list.append(self.get_cable_force())
    #     for i in range(num_iter):
    #         self.send_length_cmd_rel_timed([-shorten_dist, 0, 0, -shorten_dist],1)
    #         time.sleep(0.1)
    #         self.enforce_min_cable_forces([0.1]*4)
    #         time.sleep(0.1)
    #         cl_thisiter_1 = self.get_state()["cable_lengths"]
    #         append_data()
    #         # start perturbation
    #         for _ in range(20):
    #             self.send_length_cmd_rel_timed([0, 0, -shorten_dist_23, 0], 1)
    #             time.sleep(0.1)
    #             # check force
    #             cur_force = self.get_cable_force()
    #             if cur_force[0] < 0.1 or cur_force[3] < 0.1:
    #                 self.send_length_cmd_abs_timed(cl_thisiter_1, 1)
    #                 time.sleep(0.1)
    #                 break
    #             self.enforce_min_cable_forces([0.1]*4)
    #             append_data()
    #             cl_thisiter_2 = self.get_state()["cable_lengths"]
    #             for _ in range(20):
    #                 self.send_length_cmd_rel_timed([0, -shorten_dist_23, 0, 0], 1)
    #                 time.sleep(0.1)
    #                 cur_force = self.get_cable_force()
    #                 if cur_force[0] < 0.1 or cur_force[3] < 0.1 or cur_force[1] < 0.1:
    #                     self.send_length_cmd_abs_timed(cl_thisiter_2, 1)
    #                     time.sleep(0.1)
    #                     break
    #                 self.enforce_min_cable_forces([0.1]*4)
    #                 append_data()
    #     return cl_list, motorPos_list, force_list

    def generate_waypoints(self, calibrated_poses, start_base_pos, speed, dir, total_time):
        # calibrated poses is in the form of [pose1, pose2, ...], where each pos is [base_mm, vert_mm, cable_lengths]
        # start_base_pos is between self.start_dist to self.max_dist
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
        # Each segment: (seg_start_pos, seg_end_pos, seg_dir)
        segments = []
        cur_pos = start_base_pos
        cur_dir = dir
        dist_remaining = speed * total_time
        while dist_remaining > 1e-9:
            dist_to_wall = (self.max_dist - cur_pos) if cur_dir == 1 else (cur_pos - self.start_dist)
            if dist_remaining <= dist_to_wall:
                end_pos = cur_pos + cur_dir * dist_remaining
                segments.append((cur_pos, end_pos, cur_dir))
                dist_remaining = 0
            else:
                end_pos = self.max_dist if cur_dir == 1 else self.start_dist
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

            # visit intermediate calibrated poses within this segment
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


    def execute_waypoints(self, waypoints):
        # waypoints: [[t, [base_mm, vert_mm, cable_cmds]], ...]
        # Between consecutive waypoints, speed is constant: dist / dt
        if not waypoints:
            return

        BUSY_WAIT_MARGIN = 0.002

        t_start = time.perf_counter()
        for k in range(1, len(waypoints)):
            cl_speed = [0,0,0,0]
            t_cmd = waypoints[k][0]
            base_mm = waypoints[k][1][0]
            vert_mm = waypoints[k][1][1]
            cl_cmd = waypoints[k][1][2]
            dt = t_cmd-waypoints[k-1][0]
            if dt > 1e-9:
                base_speed = abs(base_mm - waypoints[k-1][1][0]) / dt 
                vert_speed = abs(vert_mm - waypoints[k-1][1][1]) / dt 
                for i in range(4):
                    cl_speed[i] = abs(cl_cmd[i] - waypoints[k-1][1][2][i]) / dt
            else:
                base_speed = 0
                vert_speed = 0
                cl_speed = [0, 0, 0, 0]
            self.send_cmd_abs(base_mm, vert_mm, cl_cmd,
                              base_speed_mm_s=base_speed,
                              vert_speed_mm_s=vert_speed,
                              cl_speed=cl_speed)

            t_next = t_start + waypoints[k][0]
            wait = t_next - time.perf_counter()
            if wait > BUSY_WAIT_MARGIN:
                time.sleep((wait - BUSY_WAIT_MARGIN)*0.8)
            while time.perf_counter() < t_next:
                pass

    def generate_trajectory(self, waypoints, Hz=50, accel=20.0):
        # waypoints: [[t, [base_mm, vert_mm, cable_cmds]], ...]
        # Returns list of [t, [base_mm, vert_mm, cable_cmds]] at 1/Hz intervals,
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
            t = k / Hz
            trajectory.append([t, cmd_at_s(s_at_t(t))])

        return trajectory

    def execute_trajectory(self, traj_list, cl_speed=None):
        # traj_list: [[t, [base_mm, vert_mm, cable_cmds]], ...]
        # Sends position commands at the scheduled times, using wall-clock to stay on schedule.
        if not traj_list:
            return
        # if cl_speed is None:
        #     cl_speed = [20.0] * self.nCables
        BUSY_WAIT_MARGIN = 0.002  # spin for the last 2ms to avoid sleep overshoot

        # Robot is already at traj_list[0].
        # Send cmd[k] at time t[k-1] so the robot has the full interval [t_{k-1}, t_k] to travel.
        t_start = time.perf_counter()
        for k in range(1, len(traj_list)):
            t_cmd, cmd = traj_list[k]
            base_mm, vert_mm, cable_cmds = cmd[0], cmd[1], cmd[2]

            # constant speed over this interval: dist / dt
            dt = t_cmd - traj_list[k - 1][0]
            d_base = abs(base_mm - traj_list[k - 1][1][0])
            d_vert = abs(vert_mm - traj_list[k - 1][1][1])
            base_speed = d_base / dt if dt > 1e-9 else 0.0
            vert_speed = d_vert / dt if dt > 1e-9 else 0.0

            # fire at t[k-1]: robot then has [t_{k-1}, t_k] to reach position k
            t_fire = t_start + traj_list[k - 1][0]
            wait = t_fire - time.perf_counter()
            if wait > BUSY_WAIT_MARGIN:
                time.sleep((wait - BUSY_WAIT_MARGIN) * 0.8)
            while time.perf_counter() < t_fire:
                pass

            self.send_cmd_abs(base_mm, vert_mm, cable_cmds,
                              base_speed_mm_s=base_speed,
                              vert_speed_mm_s=vert_speed,
                              cl_speed=cl_speed)


    def execute_trajectory_record_force(self, traj_list, calibrated_arm_weight):
        force_log = []  # [(timestamp, force), ...]
        stop_flag = threading.Event()

        def record_loop():
            while not stop_flag.is_set():
                t = time.perf_counter()
                force = self.get_touch_force_array(calibrated_arm_weight)
                force_log.append((t, force))

        recorder = threading.Thread(target=record_loop, daemon=True)
        recorder.start()
        self.execute_trajectory(traj_list)
        stop_flag.set()
        recorder.join()
        return force_log

    def collectData_length_force_nocontact(self):
        num_iter = 12
        shorten_dist_14 = 2 # each iteration for cable 4
        shorten_dist_23 = 4 # shoten distance for cable 2,3
        max_shorten_14 = 25
        max_shorten_23 = 50
        cur_cl = self.get_state()["cable_lengths"]
        min_l1 = cur_cl[0] - max_shorten_14
        min_l2 = cur_cl[1] - max_shorten_23
        min_l3 = cur_cl[2] - max_shorten_23
        min_l4 = cur_cl[3] - max_shorten_14
        default_speed = 5
        cl_list = []
        motorPos_list = []
        force_list = []
        num_data_collected = 0
        def append_data():
            nonlocal num_data_collected
            num_data_collected += 1
            print(f"Data collected: {num_data_collected}")
            cur_state = self.get_state()
            cl_list.append(cur_state["cable_lengths"])
            motorPos_list.append(cur_state["servo_pos"])
            force_list.append(self.get_cable_force())
        for i in range(num_iter): # for cable 4
            self.send_length_cmd_rel_timed([0, 0, 0, -shorten_dist_14],1)
            time.sleep(0.1)
            self.enforce_min_cable_forces([0.1]*4)
            time.sleep(0.1)
            cl_thisiter_0 = self.get_state()["cable_lengths"]
            append_data()
            for iter_c1 in range(10):
                cur_c1 = self.get_state()["cable_lengths"][0]
                if cur_c1-shorten_dist_14 <= min_l1:
                    break
                self.send_length_cmd_rel_timed([-shorten_dist_14, 0, 0, 0], 1)
                time.sleep(0.1)
                cur_force = self.get_cable_force()
                if cur_force[3] < 0.1:
                    self.send_length_cmd_abs_timed(cl_thisiter_0, 1)
                    time.sleep(0.1)
                    break
                self.enforce_min_cable_forces([0.1]*4)
                append_data()
                cl_thisiter_1 = self.get_state()["cable_lengths"]
            # start perturbation
                for _ in range(15):
                    cur_c3 = self.get_state()["cable_lengths"][2]
                    if cur_c3-shorten_dist_23 <= min_l3:
                        break
                    self.send_length_cmd_rel_timed([0, 0, -shorten_dist_23, 0], 1)
                    time.sleep(0.1)
                    # check force
                    cur_force = self.get_cable_force()
                    if cur_force[0] < 0.1 or cur_force[3] < 0.1:
                        self.send_length_cmd_abs_timed(cl_thisiter_1, 1)
                        time.sleep(0.1)
                        break
                    self.enforce_min_cable_forces([0.1]*4)
                    append_data()
                    cl_thisiter_2 = self.get_state()["cable_lengths"]
                    for _ in range(15):
                        cur_c2 = self.get_state()["cable_lengths"][1]
                        if cur_c2 - shorten_dist_23 <= min_l3:
                            break
                        self.send_length_cmd_rel_timed([0, -shorten_dist_23, 0, 0], 1)
                        time.sleep(0.1)
                        cur_force = self.get_cable_force()
                        if cur_force[0] < 0.1 or cur_force[3] < 0.1 or cur_force[1] < 0.1:
                            self.send_length_cmd_abs_timed(cl_thisiter_2, 1)
                            time.sleep(0.1)
                            break
                        self.enforce_min_cable_forces([0.1]*4)
                        append_data()
        return cl_list, motorPos_list, force_list

    def get_touch_force_single(self, count = 10):
        forces = []
        for _ in range(count):
            forces.append(self.tfs.read_force())
        return sum(forces) / len(forces)

    def collect_contact_data(self, initial_cable_force, pickle_file):
        # Implementation for collecting contact data
        cable_length_list = []
        cable_force_list = []
        contact_force_list = []
        vert_mm_list = []
        initial_state = self.get_state()
        initial_vert_mm = initial_state['vert_mm']
        initial_cable_length = initial_state['cable_lengths']

        def collect_force_data():
            cur_state = self.get_state()
            cable_length_list.append(cur_state["cable_lengths"])
            vert_mm_list.append(cur_state['vert_mm'])
            cable_force_list.append(self.get_cable_force())
            contact_force_list.append(self.get_touch_force())

        # move down vertical rail until all cable force is < 0.1
        self.move_down_vertical_until_slack()
        # tighten cables
        self.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
        base_cl = self.get_state()["cable_lengths"]
        total_collect = 0
        while 1:
            collect_target_num = 20
            collected_count = 0
            total_shorten = self.shorten_cable_until_no_contact()
            print("total_shorten: ", total_shorten)
            while 1:
                # randomly sample a list from [0,0,0,0] to total_shorten
                sample = [random() * total_shorten[i] for i in range(4)]
                tar_cl_this = [base_cl[i] - sample[i] for i in range(4)]
                self.send_length_cmd_abs_timed(tar_cl_this, 0.5)
                time.sleep(0.5)
                touch_force = self.get_touch_force()
                if touch_force > 0.1:
                    collect_force_data()
                    collected_count += 1
                    total_collect += 1
                    print("collected ", collected_count)
                    if collected_count >= collect_target_num:
                        break
                else:
                    continue
            self.send_stepper_cmd_rel(0, 1, 1,1)
            time.sleep(1.5)
            self.move_to_slack()
            time.sleep(1)
            cur_vert = self.get_state()['vert_mm']
            if cur_vert >= initial_vert_mm:
                break
        print("total collected data: ", total_collect)
        dump_file = {"initial_cable_force": initial_cable_force,
                     "initial_vert_mm": initial_vert_mm,
                     "initial_cable_length": initial_cable_length,
                     "cable_length_list": cable_length_list,
                     "cable_force_list": cable_force_list,
                     "contact_force_list": contact_force_list,
                     "vert_mm_list": vert_mm_list}
        with open(pickle_file, "wb") as f:
            pickle.dump(dump_file, f)
        return dump_file
        
    def calibrate_arm_weight(self, total_time = 2, has_break = False):
        if has_break:
            testkey = input("Press Enter to start calibration...")
        while 1:
            readings = [[] for _ in range(4)]
            timestamps = []
            t_start = time.time()

            while time.time() - t_start < total_time:
                forces = self.tfs_array.read_forces()
                if forces is not None:
                    timestamps.append(time.time() - t_start)
                    for i in range(4):
                        readings[i].append(forces[i]) 

            ave_reading = [sum(readings[i]) / len(readings[i]) for i in range(4)]
            std_array = [np.array(readings[i]).std() for i in range(4)]
            print("\n--- Results ---")
            for i in range(4):
                arr = np.array(readings[i])
                print(f"CH{i+1}: mean={arr.mean():.4f}  sd={arr.std():.4f}")
            # if any of the sd is bigger than 0.3, then re-calibrate
            if any(std > 0.3 for std in std_array):
                print("Standard deviation is too high. Stay still and re-calibrating...")
                continue
            break
        return ave_reading

    def get_pts(self):
        # get the pts from camera driver
        cur_state = self.get_state()
        base_mm = cur_state['base_mm']
        vert_mm = cur_state['vert_mm']
        pts_global = self.camera.get_pointcloud_in_world()
        pts_global[:,2] += base_mm
        pts_global[:,1] -= vert_mm
        return pts_global

    def take_arm_contour(self, holder_angle = 5, total_pic = 5):
        # Implementation for taking arm data
        holder_angle = 5/180*np.pi
        self.send_stepper_cmd_abs(base_mm=0, vert_mm=110, base_speed_mm_s=20, vert_speed_mm_s=20)
        self.mc.wait_until_reached(base_mm=0, vert_mm=110)
        dist_between_pic = 180/(total_pic-1) if total_pic > 1 else 0
        xmin = -50
        xmax = 150
        ymin = -110
        ymax = 40
        time.sleep(0.1)
        pts_all = []
        for i in range(total_pic):
            self.send_stepper_cmd_abs(base_mm=dist_between_pic*i, vert_mm=110, base_speed_mm_s=20, vert_speed_mm_s=20)
            self.mc.wait_until_reached(base_mm=dist_between_pic*i, vert_mm=110)
            time.sleep(0.1)
            pts_global = self.get_pts()
            zmin = self.start_dist+i*dist_between_pic
            zmax = zmin+70
            ymax = -abs(zmax-self.start_dist)*np.tan(holder_angle)
            print("ymax:", ymax)
            pts_filtered = self.filter_pts(pts_global, xmin, xmax, ymin, ymax, zmin, zmax)
            print("number of filtered pts: ", pts_filtered.shape[0])
            # pts_filtered = pts_global.copy()
            pts_all.append(pts_filtered)
            time.sleep(1)
        return pts_all

    def filter_pts(self, pts, xmin, xmax, ymin, ymax, zmin, zmax):
        print("number of captured pts: ", pts.shape[0])
        return pts[(pts[:,0] >= xmin) & (pts[:,0] <= xmax) & (pts[:,1] >= ymin) & (pts[:,1] <= ymax) & (pts[:,2] >= zmin) & (pts[:,2] <= zmax)]

    def visualize_arm_contour(self, pts_all):
        # visualize the current state 
            

        # visualize pointcloud, annotate global axis
        # if pts_all is a list, stack into a np array
        if isinstance(pts_all, list):
            pts_all = np.vstack(pts_all)

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

        print("Coordinate frame axis colors:")
        print("  X axis — Red")
        print("  Y axis — Green")
        print("  Z axis — Blue")

        o3d.visualization.draw_geometries([pcd, frame], window_name="Arm Contour")

    def visualize_cur_setting(self, pts_all):

        cur_state = self.get_state()
        base_mm = cur_state['base_mm']
        vert_mm = cur_state['vert_mm']
        cur_cl = cur_state['cable_lengths']

        tar_cable_length = [cl * 1e-3 for cl in cur_cl]
        print(tar_cable_length)
        verts, _ = self.palm.FKD_static(  # second return is Q, not cable_tension
            tar_cable_length, self.palm.Q0, np.zeros((3 * self.palm.num_vertices, 1)))
        print("cable length in sim: ", self.palm.get_cable_length(self.palm.vertices_2_Q(verts), self.palm.pulley_location))
        verts = verts * 1e3                                   # m -> mm
        cur_pulley_location = self.palm.pulley_location.copy() * 1e3  # m -> mm
        verts[:, 1] -= vert_mm
        verts[:, 2] += base_mm
        cur_pulley_location[:, 1] -= vert_mm
        cur_pulley_location[:, 2] += base_mm

        if isinstance(pts_all, list):
            pts_all = np.vstack(pts_all)

        plotter = pv.Plotter()
        mesh = pv.PolyData(verts)
        mesh.faces = np.hstack([[3, *tri] for tri in self.palm.triangle_list])  # surface triangles, not quads

        # plot the cable from pp to pulley
        for i in range(self.palm.nCable):
            plotter.add_lines(np.array([cur_pulley_location[i], verts[self.palm.pp_idx[i]]]), color='blue')

        plotter.add_mesh(mesh, show_edges=True, opacity=0.5)
        # annotate origin
        plotter.add_mesh(pv.Sphere(radius=1.0, center=[0, 0, 0]), color='red')
        cloud = pv.PolyData(pts_all)  # StructuredGrid requires meshgrid arrays; PolyData handles scattered points
        plotter.add_mesh(cloud, color='orange', point_size=3, render_points_as_spheres=True, opacity=0.5)
        # make axis equal
        plotter.show_grid()
        plotter.show_axes()

        plotter.set_scale(1, 1, 1)
        plotter.show()


    def move_down_vertical_until_slack(self, force_threshold: float = 0.1, step_mm: float = 0.5, speed_mm_s: float = 2.0):
        step_duration = step_mm / speed_mm_s
        while True:
            forces = self.cts.read_forces()
            if all(f < force_threshold for f in forces):
                break
            if self.get_state()['vert_mm'] <= 1:
                print("reached bottom without all cables going slack")
                break
            self.send_stepper_cmd_rel(0, -step_mm, 0, speed_mm_s)
            time.sleep(step_duration + 0.05)

    def shorten_cable_until_no_contact(self, force_threshold: float = 0.1, speed_mm_s: float = 4.0):
        total_shorten_time = 0
        step_cable = [1, 2, 2, 1]
        while True:
            if self.get_touch_force() < force_threshold:
                break
            self.send_length_cmd_rel([-s for s in step_cable], [speed_mm_s] * self.nCables)
            time.sleep(max(step_cable) / speed_mm_s + 0.05)
            total_shorten_time += 1
        return [total_shorten_time*step for step in step_cable]


    def tighten_selected_motor(self, motor_ids):
        # print("start calibrating motor: ", motor_ids)
        offset_unit = -1 # about 0.76mm*5
        offsets_list = [0,0,0,0]
        tightened = [0,0,0,0]
        calibrated = [0,0,0,0]
        threshold = 0.1
        for i in range(4):
            if (i+1) not in motor_ids:
                calibrated[i] = 1
                tightened[i] = 1
        while True:
            loadcell_reading = self.cts.read_forces()
            offsets_list = [0,0,0,0]
            if tightened != [1,1,1,1]:
                for i in range(4):
                    loadcell_reading[i] = round(loadcell_reading[i], 3)
                    # print(loadcell_reading[i])
                    if not tightened[i]:
                        if abs(loadcell_reading[i]) > threshold:
                            tightened[i] = 1
                        else:
                            offsets_list[i] = offset_unit
                    else:
                        offsets_list[i] = 0
                self.send_length_cmd_rel(offsets_list, [4,4,4,4])
                time.sleep(0.5)
            else:
                # print("all calibrated")
                break

    def enforce_min_cable_forces(self, min_forces: list, step_mm: float = 0.5, speed_mm_s: float = 4.0, poll_interval: float = 0.1):
        """Drive motors until each cable force is >= the corresponding value in min_forces."""
        speeds = [speed_mm_s] * self.nCables
        stable_count_goal = 10
        stable_count = 0
        while True:
            forces = self.cts.read_forces()
            # print("forces: ", forces, " stable count: ", stable_count)
            offsets = [0.0] * self.nCables
            all_satisfied = True
            for i in range(self.nCables):
                if forces[i] < min_forces[i]:
                    offsets[i] = -step_mm  # negative shortens cable, increasing tension
                    all_satisfied = False
                    stable_count = 0
            if all_satisfied:
                stable_count += 1
                if stable_count >= stable_count_goal:
                    break
                else:
                    continue
                
            self.send_length_cmd_rel(offsets, speeds)
            time.sleep(poll_interval)

    def move_to_slack(self, force_threshold: float = 0.02, step_mm: float = 1,
                      speed_mm_s: float = 4.0, poll_interval: float = 0.1):
        """Lengthen cables until every cable force is below force_threshold (N),
        without letting any cable exceed its initial_cl."""
        speeds = [speed_mm_s] * self.nCables
        stable_count_goal = 10
        stable_count = 0
        while True:
            forces = self.get_cable_force()
            cur_cl = self.get_state()["cable_lengths"]
            offsets = [0.0] * self.nCables
            all_slack = True
            for i in range(self.nCables):
                if forces[i] >= force_threshold and cur_cl[i] < self.initial_cl[i]:
                    offsets[i] = min(step_mm, self.initial_cl[i] - cur_cl[i])
                    all_slack = False
                    stable_count = 0
            if all_slack:
                stable_count += 1
                if stable_count >= stable_count_goal:
                    break
                else:
                    continue
            self.send_length_cmd_rel(offsets, speeds)
            time.sleep(poll_interval)

    def enforce_force(self, target_force: list,
                      Kp: float = 0.5, Kd: float = 0.05,
                      max_delta_mm: float = 1.0,
                      speed_mm_s: float = 4.0,
                      poll_interval: float = 0.1,
                      tol: float = 0.05,
                      stable_count_goal: int = 10):
        """
        PD-controlled cable adjustment to reach target_force (N) on each cable.

        Error = target - current. Positive error means force is below target,
        so we shorten the cable (negative delta_length) to increase tension.
        The derivative term damps the approach to reduce overshoot.
        Each iteration the commanded delta is clamped to [-max_delta_mm, max_delta_mm].
        """
        speeds = [speed_mm_s] * self.nCables
        prev_errors = [0.0] * self.nCables
        prev_time = time.time()
        stable_count = 0

        while True:
            now = time.time()
            dt = max(now - prev_time, 1e-6)  # avoid division by zero
            prev_time = now

            forces = self.cts.read_forces()
            errors = [target_force[i] - forces[i] for i in range(self.nCables)]
            print(f"forces: {[round(f,3) for f in forces]}  "
                  f"errors: {[round(e,3) for e in errors]}  "
                  f"stable: {stable_count}")

            if all(abs(e) < tol for e in errors):
                stable_count += 1
                if stable_count >= stable_count_goal:
                    break
            else:
                stable_count = 0

            deltas = []
            for i in range(self.nCables):
                d_error = (errors[i] - prev_errors[i]) / dt
                # Negative sign: positive error -> shorten cable (negative delta)
                raw_delta = -(Kp * errors[i] + Kd * d_error)
                deltas.append(float(np.clip(raw_delta, -max_delta_mm, max_delta_mm)))

            prev_errors = errors
            self.send_length_cmd_rel(deltas, speeds)
            time.sleep(poll_interval)

    def checkpoint(self, info = ""):
        print("checkpoint: ", info)
        key = input("press enter to continue or q to quit")
        if key == 'q':
            self.close_all()
            exit()

    def close_all(self):
        # self.mc.return_zeros()
        # self.send_stepper_cmd_abs(0,0,10,10)
        # self.send_length_cmd_abs(self.initial_cl,[10,10,10,10])
        self.send_cmd_abs(0, 60, self.initial_cl, 10, 10, [10,10,10,10])
        self.mc.wait_until_reached(0, 60)
        self.send_cmd_abs(0, 0, self.initial_cl, 10, 10, [10,10,10,10])
        self.camera.stop()
        # self.mc._running = False
        # self.mc._sock.close()

if __name__ == '__main__':
    touchbot = touchbotController()
    time.sleep(0.1)
    # get current motor pos
    print("Current motor positions:")
    print(touchbot.get_state())
    print(touchbot.mc.get_positions())
    touchbot.checkpoint("press to do the calibration")
    touchbot.calibrate_motor_pos()
    touchbot.checkpoint("Calibration complete")
    print("cur state: ", touchbot.get_state())
    touchbot.checkpoint("Initial state recorded")

    touchbot.send_length_cmd_rel_timed([-0,-30,-30,0],3)
    time.sleep(0.1)
    # touchbot.tighten_selected_motor([1,4])
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    touchbot.checkpoint("print current motor positions and cable tensions, do the measurement")
    # print(touchbot.mc.get_positions())
    touchbot.move_to_slack()
    touchbot.checkpoint("Target cable lengths set")
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    # print cable tensions
    print("Current cable tensions:")
    print(touchbot.cts.read_forces())
    
    touchbot.checkpoint("will return zero")
    touchbot.close_all()
    start_t = time.time()
    while time.time() - start_t < 5:
        print(touchbot.cts.read_forces())
        time.sleep(0.1)



