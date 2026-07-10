import os
import pickle
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_kernel import GaussianKernelRegression

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/size3_data_noContact.pkl"
SAVE_PATH   = "NN_models/size3_length_force_noContact_gaussian.npz"

N_KERNELS   = 40
NOISE_VAR   = 0.05
TEST_SPLIT  = 0.2
RANDOM_SEED = 42

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Returns raw X (cable lengths) and raw Y (forces)."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["cable_lengths"])   # (N, 4)
    Y = np.array(data["forces"])          # (N, 4)
    # round X and Y to 0.1
    # X = np.round(X / 0.1) * 0.1
    # Y = np.round(Y / 0.1) * 0.1

    print(f"Samples : {len(X)}")
    print(f"Input   : cable_lengths  range=[{X.min():.3f}, {X.max():.3f}]")
    print(f"Output  : forces         range=[{Y.min():.3f}, {Y.max():.3f}]")
    return X, Y


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_raw, Y_raw = load_data()

    # Train / test split
    np.random.seed(RANDOM_SEED)
    N = len(X_raw)
    idx = np.random.permutation(N)
    split = int((1.0 - TEST_SPLIT) * N)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, Y_train = X_raw[train_idx], Y_raw[train_idx]
    X_test,  Y_test  = X_raw[test_idx],  Y_raw[test_idx]

    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # Train model (normalization is handled internally)
    model = GaussianKernelRegression(
        n_kernels=N_KERNELS,
        input_dim=X_raw.shape[1],
        output_dim=Y_raw.shape[1],
        noise_var=NOISE_VAR,
    )
    model.train_kernels(X_train, Y_train, verbose=True)

    # Evaluate
    Y_pred = model.predict(X_test)
    rmse_per_output = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=0))
    rmse_overall    = rmse_per_output.mean()
    print(f"\nTest RMSE per output : {rmse_per_output}")
    print(f"Test RMSE overall    : {rmse_overall:.4f}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)




if __name__ == "__main__":
    main()
