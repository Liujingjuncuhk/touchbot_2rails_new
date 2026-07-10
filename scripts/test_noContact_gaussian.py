import os
import pickle
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_kernel import GaussianKernelRegression
import matplotlib.pyplot as plt

DATA_PATH   = "data/size3_noContact_lengthForce_test.pkl"
MODEL_PATH  = "NN_models/size3_length_force_noContact_gaussian.npz"

ADAPT_RATIO = 0.3    # fraction of new-env data used for adaptation
RANDOM_SEED = 0


def load_data(path=DATA_PATH):
    """Returns raw X (cable lengths) and raw Y (forces)."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["cable_lengths"])   # (N, 4)
    Y = np.array(data["forces"])          # (N, 4)

    print(f"Samples : {len(X)}")
    print(f"Input   : cable_lengths  range=[{X.min():.3f}, {X.max():.3f}]")
    print(f"Output  : forces         range=[{Y.min():.3f}, {Y.max():.3f}]")
    return X, Y


def print_metrics(label, Y_pred, Y_test):
    errors = Y_pred - Y_test
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mae  = np.mean(np.abs(errors), axis=0)
    print(f"\n  [{label}]")
    for c in range(Y_test.shape[1]):
        print(f"    Cable {c+1}:  RMSE={rmse[c]:.4f}  MAE={mae[c]:.4f}")
    print(f"    Overall RMSE: {rmse.mean():.4f}")
    return rmse

def diagnose(model, X_new, Y_new):
    """Figure out what's going wrong in the new environment."""
    
    Y_pred = model.predict(X_new)
    residuals = Y_new - Y_pred
    
    # 1. Per-output RMSE — is one output much worse?
    rmse_per_output = np.sqrt(np.mean(residuals**2, axis=0))
    print("Per-output RMSE:")
    for m in range(Y_new.shape[1]):
        print(f"  Output {m}: {rmse_per_output[m]:.4f}")
    
    # 2. Is the error systematic or random?
    mean_error = np.mean(residuals, axis=0)
    print(f"\nMean error (systematic bias): {mean_error}")
    print(f"  If large → relationship has shifted")
    print(f"  If near zero → model has right trend but wrong details")
    
    # 3. Is the error uniform or concentrated in certain regions?
    abs_errors = np.sqrt(np.sum(residuals**2, axis=1))
    worst_10pct = np.percentile(abs_errors, 90)
    median_error = np.median(abs_errors)
    print(f"\nMedian error:     {median_error:.4f}")
    print(f"90th percentile:  {worst_10pct:.4f}")
    print(f"Max error:        {abs_errors.max():.4f}")
    if worst_10pct > 3 * median_error:
        print("  → Errors concentrated in specific regions")
        print("  → Kernels may not cover new input range")
    else:
        print("  → Errors spread uniformly")
        print("  → General shift in relationship")
    
    # 4. Are new inputs within the training range?
    print(f"\nInput ranges:")
    print(f"  Training normalization X_mean: {model.X_mean}")
    print(f"  Training normalization X_std:  {model.X_std}")
    X_norm = (X_new - model.X_mean) / model.X_std
    for j in range(X_new.shape[1]):
        lo, hi = X_norm[:, j].min(), X_norm[:, j].max()
        flag = " ← OUTSIDE TRAINING RANGE!" if (lo < -3 or hi > 3) else ""
        print(f"  Dim {j}: normalized [{lo:.2f}, {hi:.2f}]{flag}")
    
    # 5. Kernel activation — are kernels firing in new environment?
    Phi = model._compute_phi_matrix(model._normalize_X(X_new))
    max_activation = Phi.max(axis=1)  # strongest kernel per point
    print(f"\nKernel activation (max per point):")
    print(f"  Mean:   {max_activation.mean():.4f}")
    print(f"  Min:    {max_activation.min():.4f}")
    if max_activation.min() < 0.01:
        n_dead = np.sum(max_activation < 0.01)
        print(f"  → {n_dead} points have almost no kernel activation!")
        print(f"  → These points are too far from any kernel center")

def main():
    # ── Load model trained on original environment ────────────────────────────
    model = GaussianKernelRegression()
    model.load(MODEL_PATH)

    # ── Load newly collected data (new environment) ───────────────────────────
    X_new, Y_new = load_data()

    # ── Filter: keep only samples within the training input range ────────────
    X_norm_all = (X_new - model.X_mean) / model.X_std
    in_range = np.all((X_norm_all >= -3) & (X_norm_all <= 3), axis=1)
    n_dropped = np.sum(~in_range)
    X_new, Y_new = X_new[in_range], Y_new[in_range]
    print(f"\nFiltered {n_dropped} out-of-range samples  →  {len(X_new)} remaining")

    # ── Diagnose: understand the gap before touching weights ──────────────────
    print("\n" + "=" * 55)
    print("DIAGNOSIS (new environment, pre-adaptation)")
    print("=" * 55)
    diagnose(model, X_new, Y_new)
    print("=" * 55)

    # Split: small adaptation set + held-out test set
    np.random.seed(RANDOM_SEED)
    N = len(X_new)
    idx = np.random.permutation(N)
    n_adapt = max(1, int(ADAPT_RATIO * N))
    adapt_idx = idx[:n_adapt]
    test_idx  = idx[n_adapt:]

    X_adapt, Y_adapt = X_new[adapt_idx], Y_new[adapt_idx]
    X_test,  Y_test  = X_new[test_idx],  Y_new[test_idx]

    print(f"\nAdaptation samples : {n_adapt}  |  Test samples : {len(test_idx)}")

    # ── Evaluate BEFORE adaptation ────────────────────────────────────────────
    Y_pred_before, Y_var_before = model.predict(X_test, return_uncertainty=True)

    # ── Adapt to new environment ──────────────────────────────────────────────
    model.adapt_to_new_environment(X_adapt, Y_adapt, use_prior_from_old=True)

    # ── Evaluate AFTER adaptation ─────────────────────────────────────────────
    Y_pred_after, Y_var_after = model.predict(X_test, return_uncertainty=True)

    # ── Print metrics ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    rmse_before = print_metrics("Before adaptation", Y_pred_before, Y_test)
    rmse_after  = print_metrics("After  adaptation", Y_pred_after,  Y_test)
    print("=" * 55)

    # ── Plot ──────────────────────────────────────────────────────────────────
    nCable = Y_test.shape[1]
    fig, axes = plt.subplots(3, nCable, figsize=(4 * nCable, 11))
    fig.suptitle(
        f"Gaussian Kernel: Adaptation to New Environment  "
        f"(adapt={n_adapt}, test={len(test_idx)})",
        fontsize=13
    )

    for c in range(nCable):
        real_c  = Y_test[:, c]
        pre_c   = Y_pred_before[:, c]
        post_c  = Y_pred_after[:, c]
        std_pre = np.sqrt(Y_var_before[:, c])
        std_post= np.sqrt(Y_var_after[:, c])

        lim_min = min(real_c.min(), pre_c.min(), post_c.min())
        lim_max = max(real_c.max(), pre_c.max(), post_c.max())
        ideal   = [lim_min, lim_max]

        sort_idx = np.argsort(real_c)

        # Row 0: before adaptation
        ax = axes[0, c]
        ax.errorbar(real_c[sort_idx], pre_c[sort_idx], yerr=2 * std_pre[sort_idx],
                    fmt='o', markersize=3, alpha=0.5, linewidth=0.5, label="pred ± 2σ")
        ax.plot(ideal, ideal, 'r--', linewidth=1.5, label="ideal")
        ax.set_title(f"Cable {c+1}  Before  (RMSE={rmse_before[c]:.4f})")
        ax.set_xlabel("Actual force")
        ax.set_ylabel("Predicted force")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Row 1: after adaptation
        ax = axes[1, c]
        ax.errorbar(real_c[sort_idx], post_c[sort_idx], yerr=2 * std_post[sort_idx],
                    fmt='o', markersize=3, alpha=0.5, linewidth=0.5,
                    color='tab:green', label="pred ± 2σ")
        ax.plot(ideal, ideal, 'r--', linewidth=1.5, label="ideal")
        ax.set_title(f"Cable {c+1}  After   (RMSE={rmse_after[c]:.4f})")
        ax.set_xlabel("Actual force")
        ax.set_ylabel("Predicted force")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Row 2: residual comparison
        ax = axes[2, c]
        ax.scatter(real_c, pre_c  - real_c, s=10, alpha=0.5, label="before", color='tab:blue')
        ax.scatter(real_c, post_c - real_c, s=10, alpha=0.5, label="after",  color='tab:green')
        ax.axhline(0, color='k', linewidth=1)
        ax.set_title(f"Cable {c+1} residuals")
        ax.set_xlabel("Actual force")
        ax.set_ylabel("Residual (pred − actual)")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_fig = "gaussian_noContact_adapt.png"
    plt.savefig(out_fig, dpi=150)
    plt.show()
    print(f"Plot saved to {out_fig}")


if __name__ == "__main__":
    main()
