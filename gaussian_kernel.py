"""
Gaussian Kernel Basis Function Regression with Normalization & Meta-Learning

Key addition: proper normalization that stores training statistics
and applies them consistently to all future data.

Usage:
  See the __main__ block at the bottom for a complete example.
"""

import numpy as np
from scipy.optimize import minimize


class GaussianKernelRegression:
    """
    Gaussian Kernel Basis Function Regression with built-in normalization.
    
    Parameters
    ----------
    n_kernels : int
        Number of Gaussian kernel basis functions (K).
    input_dim : int
        Dimensionality of input (d).
    output_dim : int
        Dimensionality of output (M).
    noise_var : float
        Assumed noise variance (sigma^2) in normalized space.
    """
    
    def __init__(self, n_kernels=20, input_dim=4, output_dim=4, noise_var=0.01):
        self.K = n_kernels
        self.d = input_dim
        self.M = output_dim
        self.noise_var = noise_var
        
        # Kernel parameters
        self.centers = None
        self.log_lengthscales = None
        
        # Weight parameters
        self.w_mean = None
        self.w_cov = None
        
        # Normalization parameters (computed once, reused forever)
        self.X_mean = None   # (d,)
        self.X_std = None    # (d,)
        self.Y_mean = None   # (M,)
        self.Y_std = None    # (M,)
        self.is_fitted = False
    
    # =========================================================================
    # Normalization
    # =========================================================================
    
    def _compute_normalization(self, X, Y):
        """
        Compute and store normalization parameters from training data.
        Called ONCE during initial training. Never recomputed.
        """
        self.X_mean = np.mean(X, axis=0)   # (d,)
        self.X_std = np.std(X, axis=0)     # (d,)
        self.Y_mean = np.mean(Y, axis=0)   # (M,)
        self.Y_std = np.std(Y, axis=0)     # (M,)
        
        # Prevent division by zero if a dimension is constant
        self.X_std[self.X_std < 1e-10] = 1.0
        self.Y_std[self.Y_std < 1e-10] = 1.0
        
        self.is_fitted = True
    
    def _normalize_X(self, X):
        """Normalize inputs using STORED training statistics."""
        return (X - self.X_mean) / self.X_std
    
    def _normalize_Y(self, Y):
        """Normalize outputs using STORED training statistics."""
        return (Y - self.Y_mean) / self.Y_std
    
    def _denormalize_Y(self, Y_norm):
        """Convert normalized predictions back to original scale."""
        return Y_norm * self.Y_std + self.Y_mean
    
    def _denormalize_Y_var(self, var_norm):
        """Convert normalized variance back to original scale."""
        # var(aX + b) = a^2 * var(X)
        return var_norm * (self.Y_std ** 2)
    
    # =========================================================================
    # Kernel computations (all in normalized space)
    # =========================================================================
    
    def _get_lengthscales_sq(self):
        return np.exp(2.0 * self.log_lengthscales)
    
    def _compute_phi_matrix(self, X_norm):
        """Compute kernel matrix. X_norm must already be normalized."""
        ell_sq = self._get_lengthscales_sq()
        diff = X_norm[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        exponent = -0.5 * np.sum(diff**2 / ell_sq, axis=2)
        return np.exp(exponent)
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def _init_centers(self, X_norm):
        from scipy.cluster.vq import kmeans2
        centers, _ = kmeans2(X_norm.astype(np.float64), self.K, minit='points')
        self.centers = centers
    
    def _pack_kernel_params(self):
        return np.concatenate([self.centers.ravel(), self.log_lengthscales])
    
    def _unpack_kernel_params(self, params):
        n_center = self.K * self.d
        centers = params[:n_center].reshape(self.K, self.d)
        log_ls = params[n_center:]
        return centers, log_ls
    
    def _neg_log_marginal_likelihood(self, params, X_norm, Y_norm):
        centers, log_ls = self._unpack_kernel_params(params)
        
        old_c, old_l = self.centers, self.log_lengthscales
        self.centers, self.log_lengthscales = centers, log_ls
        Phi = self._compute_phi_matrix(X_norm)
        self.centers, self.log_lengthscales = old_c, old_l
        
        N = X_norm.shape[0]
        K_y = Phi @ Phi.T + self.noise_var * np.eye(N)
        
        try:
            L = np.linalg.cholesky(K_y)
        except np.linalg.LinAlgError:
            return 1e10
        
        neg_lml = 0.0
        for m in range(self.M):
            alpha = np.linalg.solve(L, Y_norm[:, m])
            neg_lml += 0.5 * np.dot(alpha, alpha)
        
        neg_lml += self.M * np.sum(np.log(np.diag(L)))
        neg_lml += 0.5 * self.M * N * np.log(2 * np.pi)
        return neg_lml
    
    def train_kernels(self, X_raw, Y_raw, max_iter=200, verbose=True):
        """
        Train kernel parameters on raw (unnormalized) data.
        
        This method:
          1. Computes and saves normalization parameters
          2. Normalizes the data
          3. Optimizes kernel centers and lengthscales
          4. Estimates weights
        
        Parameters
        ----------
        X_raw : array, shape (N, d) - raw training inputs
        Y_raw : array, shape (N, M) - raw training outputs
        """
        assert X_raw.shape[1] == self.d
        assert Y_raw.shape[1] == self.M
        
        # Step 1: compute and save normalization (ONLY done here)
        self._compute_normalization(X_raw, Y_raw)
        
        if verbose:
            print(f"\nNormalization parameters (saved for all future use):")
            print(f"  X_mean: {self.X_mean}")
            print(f"  X_std:  {self.X_std}")
            print(f"  Y_mean: {self.Y_mean}")
            print(f"  Y_std:  {self.Y_std}")
        
        # Step 2: normalize
        X_norm = self._normalize_X(X_raw)
        Y_norm = self._normalize_Y(Y_raw)
        
        if verbose:
            print(f"\nTraining on {X_raw.shape[0]} samples...")
            print(f"  Raw X range:  [{X_raw.min():.2f}, {X_raw.max():.2f}]")
            print(f"  Norm X range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
            print(f"  Raw Y range:  [{Y_raw.min():.2f}, {Y_raw.max():.2f}]")
            print(f"  Norm Y range: [{Y_norm.min():.2f}, {Y_norm.max():.2f}]")
        
        # Step 3: optimize kernel parameters in normalized space
        self._init_centers(X_norm)
        avg_dist = np.mean(np.std(X_norm, axis=0))
        self.log_lengthscales = np.log(np.ones(self.d) * avg_dist)
        
        params0 = self._pack_kernel_params()
        result = minimize(
            self._neg_log_marginal_likelihood,
            params0,
            args=(X_norm, Y_norm),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        self.centers, self.log_lengthscales = self._unpack_kernel_params(result.x)
        
        if verbose:
            print(f"  Optimized lengthscales: {np.exp(self.log_lengthscales)}")
        
        # Step 4: estimate weights
        self.estimate_weights(X_norm, Y_norm)
        
        if verbose:
            Y_pred = self.predict(X_raw)
            rmse = np.sqrt(np.mean((Y_pred - Y_raw)**2, axis=0))
            print(f"  Training RMSE per output: {rmse}")
    
    # =========================================================================
    # Weight estimation (in normalized space)
    # =========================================================================
    
    def estimate_weights(self, X_norm, Y_norm, prior_mean=None, prior_cov=None):
        """Bayesian linear regression for weights. Inputs must be normalized."""
        Phi = self._compute_phi_matrix(X_norm)
        
        if prior_mean is None:
            prior_mean = np.zeros((self.K, self.M))
        if prior_cov is None:
            prior_cov = np.eye(self.K)
        
        prior_cov_inv = np.linalg.inv(prior_cov)
        
        self.w_cov = np.linalg.inv(
            Phi.T @ Phi / self.noise_var + prior_cov_inv
        )
        self.w_mean = self.w_cov @ (
            Phi.T @ Y_norm / self.noise_var + prior_cov_inv @ prior_mean
        )
    
    # =========================================================================
    # Prediction (accepts raw data, returns raw-scale results)
    # =========================================================================
    
    def predict(self, X_raw, return_uncertainty=False):
        """
        Predict outputs for new raw (unnormalized) inputs.
        
        The method internally:
          1. Normalizes X using saved training statistics
          2. Computes kernels and prediction in normalized space
          3. Denormalizes the output back to original scale
        
        Parameters
        ----------
        X_raw : array, shape (N, d) or (d,) - raw inputs
        
        Returns
        -------
        y_pred : array - predictions in original scale
        y_var  : array - prediction variance in original scale (if requested)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train_kernels() first.")
        
        single = X_raw.ndim == 1
        if single:
            X_raw = X_raw.reshape(1, -1)
        
        # Normalize input using SAVED parameters
        X_norm = self._normalize_X(X_raw)
        
        # Predict in normalized space
        Phi = self._compute_phi_matrix(X_norm)
        Y_norm_pred = Phi @ self.w_mean
        
        # Denormalize prediction
        Y_pred = self._denormalize_Y(Y_norm_pred)
        
        if return_uncertainty:
            Y_var_norm = np.zeros((X_raw.shape[0], self.M))
            for n in range(X_raw.shape[0]):
                phi_n = Phi[n, :]
                model_var = phi_n @ self.w_cov @ phi_n
                Y_var_norm[n, :] = model_var + self.noise_var
            
            # Denormalize variance
            Y_var = self._denormalize_Y_var(Y_var_norm)
            
            if single:
                return Y_pred.ravel(), Y_var.ravel()
            return Y_pred, Y_var
        
        if single:
            return Y_pred.ravel()
        return Y_pred
    
    # =========================================================================
    # Adaptation (accepts raw data)
    # =========================================================================
    
    def adapt_to_new_environment(self, X_new_raw, Y_new_raw, use_prior_from_old=True):
        """
        Adapt to a new environment using a small raw dataset.
        
        Keeps kernel parameters AND normalization parameters fixed.
        Only re-estimates weights.
        
        Parameters
        ----------
        X_new_raw : array, shape (N_new, d) - raw inputs from new environment
        Y_new_raw : array, shape (N_new, M) - raw outputs from new environment
        use_prior_from_old : bool
            If True, use old weights as prior (recommended for small datasets).
        
        Returns
        -------
        old_w_mean : array - previous weights (for switching back if needed)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train_kernels() first.")
        
        # Normalize using the ORIGINAL training statistics (not new data stats!)
        X_norm = self._normalize_X(X_new_raw)
        Y_norm = self._normalize_Y(Y_new_raw)
        
        old_w_mean = self.w_mean.copy()
        old_w_cov = self.w_cov.copy()
        
        if use_prior_from_old:
            self.estimate_weights(X_norm, Y_norm,
                                  prior_mean=old_w_mean,
                                  prior_cov=old_w_cov)
        else:
            self.estimate_weights(X_norm, Y_norm)
        
        return old_w_mean
    
    # =========================================================================
    # Save / Load (so you can deploy the model later)
    # =========================================================================
    
    def save(self, filepath):
        """Save the entire model including normalization parameters."""
        np.savez(filepath,
                 # Normalization
                 X_mean=self.X_mean, X_std=self.X_std,
                 Y_mean=self.Y_mean, Y_std=self.Y_std,
                 # Kernel parameters
                 centers=self.centers,
                 log_lengthscales=self.log_lengthscales,
                 # Weights
                 w_mean=self.w_mean, w_cov=self.w_cov,
                 # Config
                 config=np.array([self.K, self.d, self.M, self.noise_var]))
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a previously saved model."""
        data = np.load(filepath)
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        self.Y_mean = data['Y_mean']
        self.Y_std = data['Y_std']
        self.centers = data['centers']
        self.log_lengthscales = data['log_lengthscales']
        self.w_mean = data['w_mean']
        self.w_cov = data['w_cov']
        config = data['config']
        self.K, self.d, self.M = int(config[0]), int(config[1]), int(config[2])
        self.noise_var = config[3]
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


# =============================================================================
# Demo
# =============================================================================

def generate_data(n, env, noise=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, 4))
    s, b, c = env['scale'], env['shift'], env['coupling']
    Y = np.column_stack([
        s[0] * (X[:, 0]**2 + c * X[:, 1] * X[:, 2]) + b[0],
        s[1] * (np.sin(2*np.pi*X[:, 1]) + c * X[:, 0]) + b[1],
        s[2] * (X[:, 2] * X[:, 3] + c * X[:, 0]**2) + b[2],
        s[3] * (X[:, 3]**2 + c * X[:, 1] * X[:, 3]) + b[3],
    ]) + np.random.randn(n, 4) * noise
    return X, Y


if __name__ == "__main__":
    
    # =================================================================
    # Environment A: large dataset (your main training data)
    # =================================================================
    env_A = {'scale': [2.0, 1.5, 1.8, 2.2],
             'shift': [0.5, -0.3, 0.1, 0.4],
             'coupling': 0.6}
    
    X_train, Y_train = generate_data(200, env_A, seed=42)
    X_test_A, Y_test_A = generate_data(50, env_A, seed=99)
    
    # Show raw data ranges (these can be anything)
    print("Raw data ranges:")
    print(f"  X: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Y: [{Y_train.min():.2f}, {Y_train.max():.2f}]")
    
    # =================================================================
    # Train the model (normalization happens automatically inside)
    # =================================================================
    model = GaussianKernelRegression(
        n_kernels=20, input_dim=4, output_dim=4, noise_var=0.01
    )
    model.train_kernels(X_train, Y_train, verbose=True)
    
    # =================================================================
    # Test on Environment A (just pass raw data, normalization is internal)
    # =================================================================
    Y_pred_A = model.predict(X_test_A)
    rmse_A = np.sqrt(np.mean((Y_pred_A - Y_test_A)**2))
    print(f"\nEnv A test RMSE: {rmse_A:.4f}")
    
    # =================================================================
    # Save the model (includes normalization params)
    # =================================================================
    model.save("/tmp/my_model.npz")
    
    # =================================================================
    # Later: load and use on completely new data
    # =================================================================
    model2 = GaussianKernelRegression()
    model2.load("/tmp/my_model.npz")
    
    # Predict with the loaded model — just pass raw data
    x_new = np.array([0.3, 0.7, 0.2, 0.5])
    y_pred, y_unc = model2.predict(x_new, return_uncertainty=True)
    print(f"\nLoaded model prediction:")
    print(f"  Input (raw):   {x_new}")
    print(f"  Output (raw):  {y_pred}")
    print(f"  Uncertainty:   ±{np.sqrt(y_unc)}")
    
    # =================================================================
    # Environment B: small dataset, slightly different conditions
    # =================================================================
    env_B = {'scale': [2.3, 1.2, 2.0, 1.9],
             'shift': [0.8, -0.1, 0.3, 0.6],
             'coupling': 0.5}
    
    X_adapt, Y_adapt = generate_data(15, env_B, seed=123)
    X_test_B, Y_test_B = generate_data(50, env_B, seed=456)
    
    # Test BEFORE adaptation
    Y_pred_no = model2.predict(X_test_B)
    rmse_no = np.sqrt(np.mean((Y_pred_no - Y_test_B)**2))
    
    # Adapt — just pass raw data, normalization uses saved parameters
    model2.adapt_to_new_environment(X_adapt, Y_adapt, use_prior_from_old=True)
    
    # Test AFTER adaptation
    Y_pred_yes = model2.predict(X_test_B)
    rmse_yes = np.sqrt(np.mean((Y_pred_yes - Y_test_B)**2))
    
    # From scratch with only 15 points
    model3 = GaussianKernelRegression(
        n_kernels=10, input_dim=4, output_dim=4, noise_var=0.01
    )
    model3.train_kernels(X_adapt, Y_adapt, verbose=False)
    rmse_scratch = np.sqrt(np.mean((model3.predict(X_test_B) - Y_test_B)**2))
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Env A test:                    {rmse_A:.4f}")
    print(f"  Env B (no adaptation):         {rmse_no:.4f}")
    print(f"  Env B (adapted, 15 samples):   {rmse_yes:.4f}")
    print(f"  Env B (from scratch, 15 pts):  {rmse_scratch:.4f}")
    print(f"{'='*50}")