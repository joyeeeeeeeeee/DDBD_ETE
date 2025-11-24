import h5py
import numpy as np
import torch

# =========================================================
#  Updated Data Loader for "v2_" Files
# =========================================================

def ensure_shape_time_first(tensor, expected_time_dim=2880):
    """Helper to fix dimension mismatches (h5py transpose issue)"""
    if tensor.shape[0] == expected_time_dim:
        return tensor
    elif tensor.shape[1] == expected_time_dim:
        return tensor.T
    else:
        # Heuristic: Time dim is usually larger than Bus/Gen dim
        if tensor.shape[0] < tensor.shape[1]:
            return tensor.T
        return tensor

def load_mat_data(base_path=''):
    print("--- Loading Data (Version 2) ---")
    
    # 1. Load Learning Dataset (New Filename: v2_learning_dataset.mat)
    try:
        with h5py.File(base_path + 'v2_learning_dataset.mat', 'r') as f:
            X_raw = np.array(f['X_features'])
            if X_raw.shape[0] < X_raw.shape[1]: X_raw = X_raw.T
            num_hours = X_raw.shape[0]
            
            Y_raw = np.array(f['Y_labels'])
            Y_labels = ensure_shape_time_first(Y_raw, num_hours)
            
            d_fcst_raw = np.array(f['d_forecast'])
            d_forecast = ensure_shape_time_first(d_fcst_raw, num_hours)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find 'v2_learning_dataset.mat'. Please run the new MATLAB scripts.")

    # 2. Load Topology (New Filename: v2_network_topology.mat)
    with h5py.File(base_path + 'v2_network_topology.mat', 'r') as f:
        ptdf_raw = np.array(f['PTDF'])
        num_buses = d_forecast.shape[1]
        
        # Fix PTDF shape -> (Lines, Buses)
        if ptdf_raw.shape[1] == num_buses:
            PTDF = ptdf_raw
        elif ptdf_raw.shape[0] == num_buses:
            PTDF = ptdf_raw.T
        else:
            # Fallback guess
            if ptdf_raw.shape[0] > ptdf_raw.shape[1]:
                 PTDF = ptdf_raw
            else:
                 PTDF = ptdf_raw.T
        
        num_lines = PTDF.shape[0]
            
        branch = np.array(f['mpc']['branch'])
        # Robustly find Rate A (Column 5 in 0-indexed Python, Col 6 in MATLAB)
        if branch.shape[0] == num_lines:
            F_limits = branch[:, 5]
        elif branch.shape[1] == num_lines:
            F_limits = branch[5, :]
        else:
            # Last resort guessing
            if branch.shape[0] > branch.shape[1]: F_limits = branch[:, 5]
            else: F_limits = branch[5, :]

        F_limits[F_limits == 0] = 9999 
        baseMVA = np.array(f['mpc']['baseMVA']).item()
    
    # 3. Load Generator Params (New Filename: v2_generator_parameters.mat)
    with h5py.File(base_path + 'v2_generator_parameters.mat', 'r') as f:
        p_min = np.array(f['Pmin']).flatten() / baseMVA
        p_max = np.array(f['Pmax']).flatten() / baseMVA
        c_base = np.array(f['c_base']).flatten()
        gen_bus_idx = np.array(f['gen_bus_indices']).flatten().astype(int) - 1
        
        num_gens = len(p_min)
        C_gen = np.zeros((num_buses, num_gens))
        for i, bus in enumerate(gen_bus_idx):
            if bus < num_buses:
                C_gen[bus, i] = 1.0

    # 4. Load Real Load (New Filename: v2_balanced_load_data.mat)
    with h5py.File(base_path + 'v2_balanced_load_data.mat', 'r') as f:
        d_real_raw = np.array(f['d_real'])
        d_real = ensure_shape_time_first(d_real_raw, num_hours)

    print(f"Data Loaded Successfully (v2).")
    print(f"  Time: {num_hours}, Buses: {num_buses}, Lines: {num_lines}")
    
    return {
        'PTDF': PTDF,
        'F_limits': F_limits / baseMVA,
        'p_min': p_min,
        'p_max': p_max,
        'C_gen': C_gen,
        'X': torch.FloatTensor(X_raw),
        'Y': torch.FloatTensor(Y_labels),
        'd_forecast': d_forecast / baseMVA,
        'd_real': d_real / baseMVA,
        'baseMVA': baseMVA
    }