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
    
    touchbot.send_stepper_cmd_abs(0, 110, 10, 20)
    touchbot.mc.wait_until_reached(0, 110)
    touchbot.checkpoint("put your arm and press enter")
    time.sleep(3)
    calibrated_weight = touchbot.calibrate_arm_weight()
    print("calibrated weight: {}".format(calibrated_weight))
    try:
        while True:
            force = touchbot.get_touch_force_array(calibrated_weight)
            if force is not None:
                print("contact force: {:.2f}".format(force))
    except KeyboardInterrupt:
        pass

    touchbot.checkpoint("Test completed. Press enter to exit.")
    touchbot.close_all()
