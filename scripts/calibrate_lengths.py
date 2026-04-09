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
    touchbot.send_length_cmd_rel_timed([-0,-30,-30,0],3)
    time.sleep(0.1)
    # touchbot.tighten_selected_motor([1,4])
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    time.sleep(0.1)
    print("current motor positions:", touchbot.mc.get_positions())
    touchbot.checkpoint("measure the lengths now")
    touchbot.close_all()