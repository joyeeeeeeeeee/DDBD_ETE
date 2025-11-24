import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

def test_cvx_layer():
    print(f"Testing Environment...")
    print(f"NumPy: {np.__version__}")
    import scipy
    print(f"SciPy: {scipy.__version__}")
    import cvxpy
    print(f"CVXPY: {cvxpy.__version__}")

    # 定义一个最简单的优化问题： min (x - y)^2 s.t. x >= 0
    x = cp.Variable(1)
    y = cp.Parameter(1)
    
    obj = cp.Minimize(cp.sum_squares(x - y))
    cons = [x >= 0]
    prob = cp.Problem(obj, cons)
    
    print("\nInitializing CvxpyLayer...")
    try:
        layer = CvxpyLayer(prob, parameters=[y], variables=[x])
    except Exception as e:
        print(f"❌ Layer Initialization Failed: {e}")
        return

    print("Running Forward Pass (Double Precision)...")
    # 模拟一个 Batch 的输入
    y_batch = torch.tensor([[-5.0], [2.0], [10.0]], dtype=torch.float64, requires_grad=True)
    
    try:
        # 强制使用 SCS 求解器（最稳健）
        x_star, = layer(y_batch, solver_args={'solver': cp.SCS, 'eps': 1e-4})
        print("✅ Forward Pass Successful!")
        print(f"Input: {y_batch.detach().numpy().flatten()}")
        print(f"Output: {x_star.detach().numpy().flatten()}")
        print("(Expected: [0., 2., 10.])")
        
        print("\nRunning Backward Pass...")
        loss = x_star.sum()
        loss.backward()
        print("✅ Backward Pass Successful!")
        print(f"Grads: {y_batch.grad.numpy().flatten()}")
        
    except Exception as e:
        print(f"❌ Execution Failed: {e}")

if __name__ == '__main__':
    test_cvx_layer()