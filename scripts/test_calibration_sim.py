import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import touchbot_controller
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    touchbot = touchbot_controller.touchbotController()
    time.sleep(0.1)
    touchbot.calibrate_motor_pos()
    initial_motorPos = touchbot.mc.get_positions()["servo_pos"]
    touchbot.send_length_cmd_rel_timed([0, -30, -30, 0], 3)
    time.sleep(0.1)
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    time.sleep(0.1)
    touchbot.send_stepper_cmd_abs(0, 110, 10, 20)
    touchbot.mc.wait_until_reached(0, 110)
    initial_cl = touchbot.get_state()["cable_lengths"]
    
    touchbot.checkpoint("put your arm and press enter")
    time.sleep(1)
    pts_all = touchbot.take_arm_contour()
    pts_all = np.vstack(pts_all)
    miny = np.min(pts_all[:, 1])
    print(f"miny: {miny}")
    # touchbot.visualize_arm_contour(pts_all)
    touchbot.perform_calibration(pts_all, 1.5)
    # touchbot.send_stepper_cmd_abs(touchbot.start_dist+80, 110, 30, 20)
    # touchbot.mc.wait_until_reached(touchbot.start_dist+80, 110)
    
    # touchbot.send_stepper_cmd_rel(0, -(max(80+miny,30)),10,10)
    # time.sleep(3.5)
    # cur_cable_force = touchbot.get_cable_force()
    # print(f"Current cable force: {cur_cable_force}")
    # touchbot.moveDownUntilTouched(cur_cable_force)
    # time.sleep(2)
    # touchbot.send_stepper_cmd_rel(0,10, 10,10)
    # time.sleep(1)
    # arm_weight = touchbot.calibrate_arm_weight()
    # print(f"Calibrated arm weight: {arm_weight}")
    # cur_cable_force = touchbot.get_cable_force()
    # touchbot.moveDownUntilTouched(cur_cable_force)
    # # touchbot.calibrate_to_target_force(arm_weight, target_force = 2)
    # touchbot.adjust_touch_force(cur_cable_force, arm_weight, target_force=1.5)
    touchbot.checkpoint("Test completed. Press enter to exit.")
    touchbot.close_all()
