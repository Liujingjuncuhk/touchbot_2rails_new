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
# [0.587, 1.751, 1.481, 0.159]
# [0.47199999999999986, 2.0500000000000003, 1.7670000000000001, 0.24000000000000005]
# touchbot = touchbot_controller.touchbotController()
# time.sleep(1)
def collect_repetitive_data():
    tar_cable_length = [106.76683076899609, 118.99892413782582, 117.8812636601497, 106.9208424400998]
    num_collect = 10
    cable_forces = []
    for i in range(num_collect):
        touchbot.send_length_cmd_abs_timed(tar_cable_length, 3)
        time.sleep(2)
        cable_forces.append(touchbot.get_cable_force())
        touchbot.send_length_cmd_abs_timed(touchbot.initial_cl,3)
        time.sleep(2)
    with open("cable_forces.pkl", "wb") as f:
        pickle.dump(cable_forces, f)
    return cable_forces


def visualize_data(pkl_file):
    with open(pkl_file, "rb") as f:
        cable_forces = pickle.load(f)
    # Implement visualization logic here
    each_forces = [list(group) for group in zip(*cable_forces)]
    print("diff of max forces")
    print("cable 1: ", max(each_forces[0]) - min(each_forces[0]))
    print("cable 2: ", max(each_forces[1]) - min(each_forces[1]))
    print("cable 3: ", max(each_forces[2]) - min(each_forces[2]))
    print("cable 4: ", max(each_forces[3]) - min(each_forces[3]))
    # plot each forces
    for i, forces in enumerate(each_forces):
        plt.plot(forces, label=f'Force {i+1}')
        plt.xlabel('exp id')
        plt.ylabel('force')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    touchbot = touchbot_controller.touchbotController()
    time.sleep(1)
    # cable_forces = collect_repetitive_data()
    visualize_data("cable_forces.pkl")
    