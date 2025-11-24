import numpy as np
from scipy.stats import binom

def calibrate_load_uncertainty(d_forecast, d_real, epsilon=0.05, delta=0.05):
    """
    Implements Eq. 5-9: Data-Driven Uncertainty Set Calibration.
    Returns: mu, Sigma_sqrt, rho
    """
    # 1. Calculate Errors xi
    errors = d_real - d_forecast
    
    # ---【修复】强制确保形状为 (Time, Buses) ---
    # 假设时间维度通常远大于节点维度 (2880 >> 39)
    if errors.shape[0] < errors.shape[1]:
        print(f"  [Debug] Transposing errors from {errors.shape} to (Time, Buses)")
        errors = errors.T
        
    n_samples, n_buses = errors.shape
    print(f"  [Debug] Error matrix shape: {errors.shape} (Time={n_samples}, Buses={n_buses})")
    
    # Split D1 (Shape) and D2 (Calibration)
    n_cal = int(n_samples * 0.5)
    errors_d1 = errors[:n_cal]
    errors_d2 = errors[n_cal:]
    
    # 2. Shape Construction (Eq. 7)
    mu = np.mean(errors_d1, axis=0) # (Buses,)
    
    # rowvar=False means each column is a variable (Bus), each row is an observation (Time)
    # Resulting Sigma should be (Buses, Buses) -> (39, 39)
    Sigma = np.cov(errors_d1, rowvar=False) + 1e-6 * np.eye(n_buses) 
    
    if Sigma.shape != (n_buses, n_buses):
        raise ValueError(f"Sigma shape wrong: {Sigma.shape}, expected ({n_buses}, {n_buses})")

    # Eigen decomposition for Sigma^{1/2}
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 1e-10] = 1e-10
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    # 3. Calibration (Eq. 8-9)
    diff = errors_d2 - mu
    # Mahalanobis distance squared
    # (Time, Buses) @ (Buses, Buses) * (Time, Buses) -> Sum over Buses
    scores = np.sum((diff @ Sigma_inv) * diff, axis=1)
    
    # Binomial sorting index (Eq. 9)
    n2 = len(scores)
    i_star = n2 # Default to max
    for r in range(1, n2 + 1):
        prob = binom.cdf(r-1, n2, 1-delta)
        if prob >= 1 - epsilon:
            i_star = r
            break
            
    sorted_scores = np.sort(scores)
    rho = sorted_scores[i_star-1]
    
    print(f"Load Uncertainty Calibrated: rho = {rho:.4f}")
    return mu, Sigma_sqrt, rho

def build_robust_constraints(data, mu, Sigma_sqrt, rho, tau=0.5):
    """
    Converts Eq. 10-12 into standard Polytope form Ax <= b for the CVX layer.
    """
    H = data['PTDF']
    # PTDF is (Lines, Buses). Make sure we align with Buses dimension
    # If PTDF is (46, 39), shape[1] is 39.
    
    num_buses_ptdf = H.shape[1]
    num_buses_sigma = Sigma_sqrt.shape[0]
    
    # ---【修复】维度安全检查 ---
    if num_buses_ptdf != num_buses_sigma:
        print(f"Warning: PTDF buses ({num_buses_ptdf}) != Sigma buses ({num_buses_sigma})")
        if H.shape[0] == num_buses_sigma:
            print("  -> Transposing PTDF to match Sigma")
            H = H.T
        else:
            raise ValueError("Dimension mismatch between PTDF and Sigma that cannot be fixed by transpose.")

    vec_ones = np.ones(num_buses_sigma)
    
    # --- 1. Power Balance (AGC) ---
    # Margin_sys = sqrt(rho) * || Sigma^0.5 * 1 ||_2
    # Sigma_sqrt: (39, 39), vec_ones: (39,) -> Result (39,)
    # norm -> scalar
    margin_sys = np.sqrt(rho) * np.linalg.norm(Sigma_sqrt @ vec_ones)
    
    # --- 2. Line Flows ---
    # Margin_line_i = sqrt(rho) * || Sigma^0.5 * h_i^T ||_2
    F = data['F_limits']
    margin_lines = np.zeros(len(F))
    
    for i in range(len(F)):
        # H[i, :] is (Buses,). Sigma_sqrt is (Buses, Buses).
        margin_lines[i] = np.sqrt(rho) * np.linalg.norm(Sigma_sqrt @ H[i, :])
        
    # Effective limits
    tau_eff = tau - margin_sys
    
    print(f"System Balance Margin: {margin_sys:.4f}, Effective Tau: {tau_eff:.4f}")
    
    return margin_sys, margin_lines, tau_eff