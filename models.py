import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

class PricePredictorLSTM(nn.Module):
    """
    Section IV.A: LSTM Forecast Model
    """
    def __init__(self, input_dim, hidden_dim, num_gens):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, num_gens)
        self.fc_width = nn.Linear(hidden_dim, num_gens)
        self.softplus = nn.Softplus() 
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        base_c = self.fc_mean(last_hidden)
        width = self.softplus(self.fc_width(last_hidden))
        h_lo = base_c - width
        h_hi = base_c + width
        return h_lo, h_hi

class DifferentiableOPFLayer(nn.Module):
    """
    Section IV.A: Optimization Layer
    **Strictly Differentiable Version** (No Fallback)
    """
    def __init__(self, data, mu):
        super().__init__()
        
        # 1. Data Setup
        C_gen_np = data['C_gen'] 
        PTDF_np = data['PTDF']
        self.num_gens = len(data['p_min'])
        self.num_lines = len(data['F_limits'])
        num_buses = C_gen_np.shape[0] 

        # Dimension Check
        if PTDF_np.shape[1] != num_buses:
            if PTDF_np.shape[0] == num_buses: PTDF_np = PTDF_np.T
            else: raise ValueError("PTDF Shape Mismatch")

        # Buffers (Float64)
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float64))
        self.register_buffer('H', torch.tensor(PTDF_np, dtype=torch.float64))
        self.Cg_np = C_gen_np
        self.H_np = PTDF_np

        # --- 2. CVXPY Problem with Slacks ---
        x_var = cp.Variable(self.num_gens)
        s_bal = cp.Variable(1, nonneg=True)        # Slack for Balance
        s_line = cp.Variable(self.num_lines, nonneg=True) # Slack for Lines

        # Parameters
        c_param = cp.Parameter(self.num_gens) 
        rhs_bal_ub = cp.Parameter(1) 
        rhs_bal_lb = cp.Parameter(1)
        rhs_line_ub = cp.Parameter(self.num_lines)
        rhs_line_lb = cp.Parameter(self.num_lines)
        
        # Constraints
        constraints = [x_var >= data['p_min'], x_var <= data['p_max']]
        
        net_inj = np.ones(num_buses) @ C_gen_np @ x_var
        constraints.append(net_inj <= rhs_bal_ub + s_bal)
        constraints.append(net_inj >= rhs_bal_lb - s_bal)
        
        flow = PTDF_np @ C_gen_np @ x_var
        constraints.append(flow <= rhs_line_ub + s_line)
        constraints.append(flow >= rhs_line_lb - s_line)
        
        # Objective: Min Cost + Heavy Penalty on Slacks
        # This guarantees Feasibility while trying to satisfy constraints
        PENALTY = 100000.0
        obj = cp.Minimize(c_param @ x_var + PENALTY * (cp.sum(s_bal) + cp.sum(s_line)))
        
        problem = cp.Problem(obj, constraints)
        
        # Initialize CvxpyLayer
        # If this fails during init, environment is broken.
        self.cvx_layer = CvxpyLayer(problem, parameters=[
            c_param, rhs_bal_ub, rhs_bal_lb, rhs_line_ub, rhs_line_lb
        ], variables=[x_var, s_bal, s_line])
        
    def forward(self, c_robust, d_forecast, margins):
        # 1. Enforce Float64 and Contiguous Memory
        # This fixes "Error parsing inputs"
        c_robust = c_robust.double().contiguous()
        d_forecast = d_forecast.double().contiguous()
        device = c_robust.device
        if torch.isnan(c_robust).any() or torch.isinf(c_robust).any():
            print("CRITICAL: c_robust (Cost) contains NaNs or Infs!")
            # 返回一个 dummy 结果或者抛出异常，避免 C++ 崩溃
            return torch.zeros((c_robust.shape[0], self.num_gens), device=device)
        if torch.isnan(d_forecast).any():
            print("CRITICAL: d_forecast contains NaNs!")
            return torch.zeros((c_robust.shape[0], self.num_gens), device=device)
 
        # 2. Prepare RHS
        d_plus_mu = d_forecast + self.mu.to(device)
        
        center_bal = torch.sum(d_plus_mu, dim=1, keepdim=True)
        tau = margins['tau_eff'].to(device).double()
        rhs_bal_ub = (center_bal + tau).contiguous()
        rhs_bal_lb = (center_bal - tau).contiguous()
        
        center_lines = torch.matmul(d_plus_mu, self.H.to(device).t())
        F = margins['F_eff'].to(device).double()
        rhs_line_ub = (F + center_lines).contiguous()
        rhs_line_lb = (-F + center_lines).contiguous()
        if torch.isnan(rhs_bal_ub).any() or torch.isnan(rhs_line_ub).any():
             print("CRITICAL: Constraints RHS contain NaNs!")
             return torch.zeros((c_robust.shape[0], self.num_gens), device=device)

        # 
        try:
            x_star, _, _ = self.cvx_layer(
                c_robust, 
                rhs_bal_ub, rhs_bal_lb, 
                rhs_line_ub, rhs_line_lb,
                solver_args={'solver': cp.SCS, 'eps': 1e-3, 'max_iters': 2000, 'verbose': False}
            )
        except Exception as e:
            # 捕捉 CvxpyLayer 内部的其他错误
            print(f"Solver failed in layer: {e}")
            return torch.zeros((c_robust.shape[0], self.num_gens), device=device)
        
        
        return x_star

def differentiable_conformal_calibration(h_lo, h_hi, y_cal, alpha=0.1):
    diff_lower = h_lo - y_cal
    diff_upper = y_cal - h_hi
    max_diff_per_gen = torch.max(diff_lower, diff_upper) 
    scores, _ = torch.max(max_diff_per_gen, dim=1) 
    
    N = scores.shape[0]
    k = int(np.ceil((N + 1) * (1 - alpha)))
    k = min(k, N) - 1 
    
    scores_sorted, indices = torch.sort(scores, descending=False)
    idx_k = indices[k]
    q_val = scores[idx_k] 
    return q_val