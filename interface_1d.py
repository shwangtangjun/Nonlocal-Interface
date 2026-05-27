import argparse
import numpy as np
import scipy.sparse as sp
from utils import (
    build_interp_matrix,
    build_quadrature,
    build_subdomain_simpson_weights,
    build_subdomain_stiffness,
    build_subdomain_kernel_coeffs_vector,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--interface", type=float, default=0.5, help="interface * n must be an integer")
args = parser.parse_args()

n = args.n
factor = 2 / np.sqrt(np.pi)  # The constant to ensure ∫ R(|z|^2) |z_1|^2 dz = 1
lambda1 = 1.0
lambda2 = 9.0
interface_cell = int(args.interface * n)
x_basis = np.linspace(0, 1, 3 * n + 1)

Phi_simpson_left = build_subdomain_simpson_weights(n, 0, interface_cell)
Phi_simpson_right = build_subdomain_simpson_weights(n, interface_cell, n)
Phi_simpson = Phi_simpson_left + Phi_simpson_right

u_func = lambda x: np.cos(np.pi * x)
grad_func = lambda x: -np.pi * np.sin(np.pi * x)
f_base_func = lambda x: np.pi**2 * np.cos(np.pi * x)
g_value = lambda1 * grad_func(args.interface) - lambda2 * grad_func(args.interface)

for delta in np.logspace(-5, -2, num=50, base=10):
    print("delta:", delta)

    quadrature_nodes, quadrature_weights = build_quadrature(n, delta, num_leggauss=10)

    B_left, A_left = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, 0, interface_cell)
    B_right, A_right = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, interface_cell, n)
    K_left = (B_left - A_left + (B_left - A_left).T) / delta**2
    K_right = (B_right - A_right + (B_right - A_right).T) / delta**2

    left_coeffs = build_subdomain_kernel_coeffs_vector(n, delta, args.interface, 0, interface_cell)
    right_coeffs = build_subdomain_kernel_coeffs_vector(n, delta, args.interface, interface_cell, n)
    avg_left = left_coeffs / left_coeffs.sum()
    avg_right = right_coeffs / right_coeffs.sum()
    avg_jump = avg_left - avg_right

    penalty = 2 / delta * sp.csr_matrix(np.outer(avg_jump, avg_jump))
    LHS = (factor * (lambda1 * K_left + lambda2 * K_right) + penalty).tocsr()
    RHS = (lambda1 * Phi_simpson_left + lambda2 * Phi_simpson_right) * f_base_func(x_basis) + g_value * avg_left

    # Impose constraint \int_\Omega u dx=0 using Lagrange multiplier
    LHS_mod = sp.bmat([[LHS, Phi_simpson[:, None]], [Phi_simpson[None, :], None]], format="csr")
    RHS_mod = np.concatenate([np.asarray(RHS).ravel(), np.array([0.0])])
    solution = sp.linalg.spsolve(LHS_mod, RHS_mod)
    u_basis = solution[:-1]  # The last variable is the Lagrange multiplier, not needed

    x_test = np.linspace(0, 1, 5000, endpoint=False)
    test_nodes_interp_matrix = build_interp_matrix(n, x_test)
    error = np.sqrt(np.mean((test_nodes_interp_matrix * u_basis - u_func(x_test)) ** 2))
    print("Error:", error)
    print()
