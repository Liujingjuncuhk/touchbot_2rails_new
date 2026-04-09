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
    touchbot.checkpoint("Calibration complete, start collecting data")

    initial_cl = touchbot.get_state()["cable_lengths"]
    initial_motorPos = touchbot.mc.get_positions()["servo_pos"]

    cl_list, motorPos_list, force_list = touchbot.collectData_length_force_nocontact()

    print("number of data collected: ", len(cl_list))

    touchbot.checkpoint("Data collection complete")
    data_noContact = {
        "initial_cable_lengths": initial_cl,
        "initial_servo_positions": initial_motorPos,
        "cable_lengths": cl_list,
        "servo_positions": motorPos_list,
        "forces": force_list
    }

    with open("data_noContact.pkl", "wb") as f:
        pickle.dump(data_noContact, f)

    touchbot.close_all()