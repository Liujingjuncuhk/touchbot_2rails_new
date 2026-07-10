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
    print(touchbot.get_state())
    touchbot.checkpoint("Calibration complete, start collecting data")

    initial_cl = touchbot.get_state()["cable_lengths"]
    initial_motorPos = touchbot.mc.get_positions()["servo_pos"]
    touchbot.send_length_cmd_rel_timed([0, -30, -30, 0], 3)
    time.sleep(0.1)
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    time.sleep(0.1)

    touchbot.send_stepper_cmd_abs(260, 100, 30, 30)
    touchbot.mc.wait_until_reached(260, 100, timeout= 20)
    touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    # time.sleep(13.5)
    time.sleep(0.5)
    initial_cable_force = touchbot.get_cable_force()
    print("initial cable force: ", initial_cable_force)
    touchbot.moveDownUntilTouched(initial_cable_force)
    print("current force in cable: ", touchbot.get_cable_force())
    print("current touch force: ", touchbot.get_touch_force())
    touchbot.checkpoint("contact detected, move down until slack")
    touchbot.collect_contact_data(initial_cable_force, "test.pkl")
    # touchbot.move_to_slack()
    # time.sleep(0.5)
    # print("cur cable force: ", touchbot.get_cable_force())
    # touchbot.checkpoint("move to slack done")
    # touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    # time.sleep(0.5)
    # print("cur cable force: ", touchbot.get_cable_force())
    # touchbot.checkpoint("test collect contact data done")
    # time.sleep(0.1)
    touchbot.send_stepper_cmd_abs(100, 110, 30, 30)
    # time.sleep(2)
    touchbot.mc.wait_until_reached(100,110,timeout=20)
    time.sleep(0.1)
    touchbot.close_all()
