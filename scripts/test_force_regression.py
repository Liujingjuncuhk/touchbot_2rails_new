import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import time
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import touchbot_controller
import numpy
import pickle
model_pickleFile = 'models/force_regression_model.pkl'
with open(model_pickleFile, 'rb') as f:
    model = pickle.load(f)

def predict_new_data(model, new_data):
    predicted_data = model.predict(new_data)
    return predicted_data

def plot_two_vals():
    predicted_vals = [0.5637243881720235, 0.4847409935569092, 0.5888395603015478, 0.5119020907211301, 0.5378221112245777, 0.4644838866801117, 0.5398569858788331, 0.5107196137973581, 0.5380272511749083, 0.6030948479266421, 0.5330313891062004, 0.49827296057311743, 0.5632910391465772, 0.46110841338421904, 0.5215872900019344, 0.534740933617633, 0.5518819596030471, 0.5711445234449001, 0.5089744541657749, 0.5701020095371873, 0.5645980736674827, 0.5609851586170169, 0.5041233241901601, 0.5421503508387389, 0.5612991972941801, 0.5253798187147442, 0.5350843555048377, 0.5463665953736346, 0.49424780852221917, 0.5595491013170651, 0.5340928930798309, 0.5173394573578893, 0.5172040448857952, 0.536873016443089, 0.5447686564874881, 0.5367429208860673, 0.570290493112422, 0.5290601628930806, 0.5529388867470015, 0.5330893000332325]
    truth_vals = [0.9200000000000002, 0.33, 0.7199999999999999, 0.97, 2.1299999999999994, 0.89, 2.2380000000000004, 0.7599999999999999, 1.5, 0.6500000000000001, 2.069, 0.5439999999999999, 1.2340000000000002, 0.28300000000000003, 0.45000000000000007, 0.8300000000000001, 1.1150000000000002, 0.41000000000000003, 1.07, 0.7400000000000001, 0.11000000000000001, 0.14000000000000004, 0.14899999999999997, 0.4880000000000001, 0.6100000000000001, 0.5999999999999999, 0.266, 0.33, 0.11000000000000001, 0.14000000000000004, 0.17999999999999997, 0.24000000000000005, 0.21000000000000002, 0.30499999999999994, 0.4600000000000001, 0.131, 0.23000000000000004, 0.10700000000000001, 0.461, 0.19999999999999998]

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
    data_all = touchbot.collect_contact_data(initial_cable_force, "force_exp_testdata.pkl")
    initial_cable_force = data_all['initial_cable_force']
    cable_force_list = data_all['cable_force_list']
    contact_force_list = data_all['contact_force_list']
    test_force_list = []
    for i in range(len(cable_force_list)):
        fl_input = []
        for j in range(4):
            fl_input.append(cable_force_list[i][j] - initial_cable_force[j])
        test_force_list.append(fl_input)
    
    predicted_forces = predict_new_data(model, np.array(test_force_list))
    print("predicted_forces: ", predicted_forces.tolist())
    print("contact_forces_truth: ", contact_force_list)
    touchbot.checkpoint("Prediction complete")
    touchbot.send_stepper_cmd_abs(100, 110, 30, 30)
    # time.sleep(2)
    touchbot.mc.wait_until_reached(100,110,timeout=20)
    time.sleep(0.1)
    touchbot.close_all()

