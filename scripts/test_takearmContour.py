import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import touchbot_controller
import numpy
import pickle


if __name__ == "__main__":
    touchbot = touchbot_controller.touchbotController()
    time.sleep(0.1)
    touchbot.calibrate_motor_pos()
    # touchbot.checkpoint("Calibration complete, start collecting data")
    time.sleep(1)
    initial_cl = touchbot.get_state()["cable_lengths"]
    initial_cl_sim = touchbot.palm.initial_cable_length
    print("initial_cl in simulation: {}".format(initial_cl_sim))
    initial_motorPos = touchbot.mc.get_positions()["servo_pos"]
    # touchbot.send_length_cmd_rel_timed([0, -30, -30, 0], 3)
    # time.sleep(0.1)
    # touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    # time.sleep(0.1)
    touchbot.send_stepper_cmd_abs(0, 110, 20,20)
    touchbot.mc.wait_until_reached(0, 110)
    touchbot.checkpoint('put your arm in the designated position')
    pts_list = touchbot.take_arm_contour()
    touchbot.send_stepper_cmd_abs(150, 80, 20,20)
    touchbot.mc.wait_until_reached(150, 80)
    touchbot.visualize_cur_setting(pts_list)
    touchbot.send_stepper_cmd_abs(0, 80, 20,20)
    touchbot.mc.wait_until_reached(0, 80)
    time.sleep(1)
    touchbot.close_all()