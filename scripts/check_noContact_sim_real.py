import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import palm_simulator
import touchbot_controller
import numpy as np
import pickle
import matplotlib.pyplot as plt

def process_saved_data():
    pass

if __name__ == "__main__":
    palm_file = 'models/palm_size3.pickle'
    palm = palm_simulator.PalmSimulator(palm_file)
    data_file = 'data/size3_data_noContact.pkl'
    with open(data_file, 'rb') as f:
        data_noContact = pickle.load(f)

    #     data_noContact = {
    #     "initial_cable_lengths": initial_cl,
    #     "initial_servo_positions": initial_motorPos,
    #     "cable_lengths": cl_list,
    #     "servo_positions": motorPos_list,
    #     "forces": force_list
    # }
    initial_cable_length = data_noContact["initial_cable_lengths"]
    cable_lengths = data_noContact["cable_lengths"]
    forces_collected = data_noContact["forces"]
    print("type of forces_collect: ", type(forces_collected))
    print("initial cable length: ", initial_cable_length)
    print("a random cable length data: ", cable_lengths[0])
    # print the minimum vale of each cable for the collected data
    # for c in range(len(forces_collected[0])):
    #     min_force = min(forces_collected[i][c] for i in range(len(forces_collected)))
    #     print(f"Minimum force for cable {c+1}: {min_force}")
    # exit(0)
    forces_sim = []
    forces_real_filtered = []
    cable_length_filtered = []
    nSample = len(cable_lengths)
    nDropped = 0
    for i in range(nSample):
        test_cable_length = [cl*1e-3 for cl in cable_lengths[i]]
        print(f"Processing sample {i+1}/{nSample}")
        print("input cable length: ", test_cable_length)
        vert_length, cable_tension = palm.FKD_static(test_cable_length, palm.vertices_2_Q(palm.vertices),np.zeros((3 * palm.num_vertices, 1)))
        print("force sim: ", cable_tension.tolist())
        print("force real: ", forces_collected[i])
        if np.any(cable_tension < 0.0001):
            nDropped += 1
            continue
        cable_length_filtered.append(cable_lengths[i])
        forces_sim.append(cable_tension.tolist())
        forces_real_filtered.append(forces_collected[i])
        
    print(f"Dropped {nDropped}/{nSample} samples with any cable force < 0.001")
    # Save sim and real forces together with the cable lengths
    

    forces_sim = np.array(forces_sim)              # (nKept, nCable)
    forces_real = np.array(forces_real_filtered)   # (nKept, nCable)
    cable_lengths_filtered = np.array(cable_length_filtered)
    nCable = forces_sim.shape[1]
    out_file = 'data/sim_real_forces_noContact.pkl'
    save_data = {
        "cable_lengths": cable_lengths_filtered,
        "forces_sim": forces_sim,
        "forces_real": forces_real_filtered,
    }
    with open(out_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Data saved to {out_file}")

    fig, axes = plt.subplots(2, nCable, figsize=(4 * nCable, 8))
    fig.suptitle("Sim-to-Real Gap: Cable Forces (No Contact)", fontsize=14)

    for c in range(nCable):
        sim_c  = forces_sim[:, c]
        real_c = forces_real[:, c]

        # Linear fit: real = a * sim + b
        coeffs = np.polyfit(sim_c, real_c, 1)
        a, b = coeffs
        fit_x = np.linspace(sim_c.min(), sim_c.max(), 200)
        fit_y = a * fit_x + b
        residuals = real_c - (a * sim_c + b)
        r2 = 1 - np.var(residuals) / np.var(real_c)

        # Top row: scatter + linear fit
        ax_top = axes[0, c]
        ax_top.scatter(sim_c, real_c, s=10, alpha=0.6, label="samples")
        ax_top.plot(fit_x, fit_y, 'r-', linewidth=1.5,
                    label=f"fit: {a:.3f}·x + {b:.3f}\n$R^2$={r2:.4f}")
        ax_top.set_xlabel("Sim force")
        ax_top.set_ylabel("Real force")
        ax_top.set_title(f"Cable {c+1}")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, linestyle='--', alpha=0.5)

        # Bottom row: residuals (real - linear_fit)
        ax_bot = axes[1, c]
        ax_bot.scatter(sim_c, residuals, s=10, alpha=0.6, color='orange')
        ax_bot.axhline(0, color='k', linewidth=1)
        ax_bot.set_xlabel("Sim force")
        ax_bot.set_ylabel("Residual (real − fit)")
        ax_bot.set_title(f"Cable {c+1} residuals")
        ax_bot.grid(True, linestyle='--', alpha=0.5)

        print(f"Cable {c+1}: slope={a:.4f}, intercept={b:.4f}, R²={r2:.4f}, "
              f"residual std={residuals.std():.4f}")

    plt.tight_layout()
    plt.savefig("sim_real_gap_noContact.png", dpi=150)
    plt.show()
    print("Plot saved to sim_real_gap_noContact.png")

    
