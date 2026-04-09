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

    touchbot.send_stepper_cmd_abs(0, 10, 10, 10)
    # time.sleep(13.5)
    touchbot.mc.wait_until_reached(0, 10, timeout= 20)
    touchbot.send_stepper_cmd_abs(10, 10, 10, 10)
    touchbot.mc.wait_until_reached(10,10, timeout=20)
    touchbot.send_stepper_cmd_abs(0, 0, 10, 10)
    # time.sleep(13.5)
    touchbot.mc.wait_until_reached(0, 0, timeout= 20)
    print(touchbot.get_state())

