from random import random
import motor_commander
import cable_tension_sensor
import touch_force_sensor
import palm_simulator
import numpy as np
import time

class touchbotController():
    def __init__(self):
        self.mc = motor_commander.MotorCommander()
        self.cts = cable_tension_sensor.CableTensionSensor()
        self.tfs = touch_force_sensor.TouchForceSensor()
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

    def send_stepper_cmd_rel(self, base_mm: float, vert_mm: float, base_speed_mm_s: float = 0.0, vert_speed_mm_s: float = 0.0):
        # Implementation for sending relative stepper commands
        self.mc.send_stepper_only_rel(base_mm=base_mm, vert_mm=vert_mm, base_speed_mm_s=base_speed_mm_s, vert_speed_mm_s=vert_speed_mm_s)

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

    def collectData_length_force_nocontact(self):
        num_iter = 25
        shorten_dist_14 = 1 # each iteration for cable 4
        shorten_dist_23 = 2 # shoten distance for cable 2,3
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

    def get_touch_force(self, count = 10):
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
        while 1:
            collect_target_num = 10
            collected_count = 0
            total_shorten = self.shorten_cable_until_no_contact()
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
                    print("collected ", collected_count)
                    if collected_count >= collect_target_num:
                        break
                else:
                    continue
            break


            



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

    def move_to_slack(self, force_threshold: float = 0.1, step_mm: float = 1,
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
        self.send_cmd_abs(0, 0, self.initial_cl, 10, 10, [10,10,10,10])
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



