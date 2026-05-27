import argparse
import numpy as np
import scipy.sparse as sp
from utils import (
    build_interp_matrix,
    build_quadrature,
    build_subdomain_simpson_weights,
    build_subdomain_simpson_weights_2d,
    build_subdomain_kernel_coeffs_vector,
    build_subdomain_stiffness,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--x-left", type=float, default=0.3, help="x-left * n must be an integer")
parser.add_argument("--x-right", type=float, default=0.6, help="x-right * n must be an integer")
parser.add_argument("--y-bottom", type=float, default=0.4, help="y-bottom * n must be an integer")
parser.add_argument("--y-top", type=float, default=0.7, help="y-top * n must be an integer")
args = parser.parse_args()


def build_interface_average_matrix(n, delta, rectangles, x_left_cell, x_right_cell, y_bottom_cell, y_top_cell):
    x_left = x_left_cell / n
    x_right = x_right_cell / n
    y_bottom = y_bottom_cell / n
    y_top = y_top_cell / n
    basis_1d = np.linspace(0, 1, 3 * n + 1)
    x_interface_nodes = basis_1d[3 * x_left_cell : 3 * x_right_cell + 1]
    y_interface_nodes = basis_1d[3 * y_bottom_cell : 3 * y_top_cell + 1]

    avg_left = sp.csr_matrix((len(y_interface_nodes), (3 * n + 1) ** 2))
    avg_right = sp.csr_matrix((len(y_interface_nodes), (3 * n + 1) ** 2))
    avg_bottom = sp.csr_matrix((len(x_interface_nodes), (3 * n + 1) ** 2))
    avg_top = sp.csr_matrix((len(x_interface_nodes), (3 * n + 1) ** 2))
    for x_start, x_end, y_start, y_end in rectangles:
        coeffs_x_left = sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, x_left, x_start, x_end))
        coeffs_x_right = sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, x_right, x_start, x_end))
        coeffs_y_bottom = sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, y_bottom, y_start, y_end))
        coeffs_y_top = sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, y_top, y_start, y_end))
        coeffs_x_line = sp.vstack(
            [sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, x, x_start, x_end)) for x in x_interface_nodes],
            format="csr",
        )
        coeffs_y_line = sp.vstack(
            [sp.csr_matrix(build_subdomain_kernel_coeffs_vector(n, delta, y, y_start, y_end)) for y in y_interface_nodes],
            format="csr",
        )

        avg_left += sp.kron(coeffs_x_left, coeffs_y_line).tocsr()
        avg_right += sp.kron(coeffs_x_right, coeffs_y_line).tocsr()
        avg_bottom += sp.kron(coeffs_x_line, coeffs_y_bottom).tocsr()
        avg_top += sp.kron(coeffs_x_line, coeffs_y_top).tocsr()

    avg = sp.vstack([avg_left, avg_right, avg_bottom, avg_top], format="csr")
    return sp.diags(1 / avg.sum(axis=1).A1) * avg


n = args.n
lambda1 = 1.0
lambda2 = 9.0
x_left, x_right = args.x_left, args.x_right
y_bottom, y_top = args.y_bottom, args.y_top
x_left_cell, x_right_cell = int(x_left * n), int(x_right * n)
y_bottom_cell, y_top_cell = int(y_bottom * n), int(y_top * n)
factor = 2 / np.pi

omega1_rectangles = [(x_left_cell, x_right_cell, y_bottom_cell, y_top_cell)]
omega2_rectangles = [
    (0, x_left_cell, 0, n),
    (x_right_cell, n, 0, n),
    (x_left_cell, x_right_cell, 0, y_bottom_cell),
    (x_left_cell, x_right_cell, y_top_cell, n),
]

basis_1d = np.linspace(0, 1, 3 * n + 1)
X, Y = np.meshgrid(basis_1d, basis_1d, indexing="ij")
x_basis, y_basis = X.ravel(), Y.ravel()

q = lambda x: 3 * x**2 - 2 * x**3 - 0.5
dq = lambda x: 6 * x - 6 * x**2
d2q = lambda x: 6 - 12 * x
alpha = 0.35
beta = 0.25
u_func = lambda x, y: q(x) + alpha * q(y) + beta * q(x) * q(y)
grad_func = lambda x, y: (dq(x) * (1 + beta * q(y)), dq(y) * (alpha + beta * q(x)))
laplace_func = lambda x, y: d2q(x) * (1 + beta * q(y)) + d2q(y) * (alpha + beta * q(x))
f_base_func = lambda x, y: -laplace_func(x, y)

Phi_simpson_1 = build_subdomain_simpson_weights_2d(n, omega1_rectangles)
Phi_simpson_2 = build_subdomain_simpson_weights_2d(n, omega2_rectangles)
Phi_simpson = Phi_simpson_1 + Phi_simpson_2

for delta in np.logspace(-5, -2, num=50, base=10):
    print("delta:", delta)

    quadrature_nodes, quadrature_weights = build_quadrature(n, delta, num_leggauss=10)

    K_1 = sp.csr_matrix(((3 * n + 1) ** 2, (3 * n + 1) ** 2))
    for x0, x1, y0, y1 in omega1_rectangles:
        for x_int0, x_int1, y_int0, y_int1 in omega1_rectangles:
            Bx, Ax = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, x0, x1, x_int0, x_int1)
            By, Ay = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, y0, y1, y_int0, y_int1)
            K_1 += sp.kron(Bx, By).tocsr() - sp.kron(Ax, Ay).tocsr()
    K_1 = ((K_1 + K_1.T) / delta**2).tocsr()

    K_2 = sp.csr_matrix(((3 * n + 1) ** 2, (3 * n + 1) ** 2))
    for x0, x1, y0, y1 in omega2_rectangles:
        for x_int0, x_int1, y_int0, y_int1 in omega2_rectangles:
            Bx, Ax = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, x0, x1, x_int0, x_int1)
            By, Ay = build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, y0, y1, y_int0, y_int1)
            K_2 += sp.kron(Bx, By).tocsr() - sp.kron(Ax, Ay).tocsr()
    K_2 = ((K_2 + K_2.T) / delta**2).tocsr()

    x_interface_nodes = basis_1d[3 * x_left_cell : 3 * x_right_cell + 1]
    y_interface_nodes = basis_1d[3 * y_bottom_cell : 3 * y_top_cell + 1]
    x_interface_weights = build_subdomain_simpson_weights(n, x_left_cell, x_right_cell)[3 * x_left_cell : 3 * x_right_cell + 1]
    y_interface_weights = build_subdomain_simpson_weights(n, y_bottom_cell, y_top_cell)[3 * y_bottom_cell : 3 * y_top_cell + 1]

    interface_points = np.vstack(
        [
            np.column_stack([np.full_like(y_interface_nodes, x_left), y_interface_nodes]),
            np.column_stack([np.full_like(y_interface_nodes, x_right), y_interface_nodes]),
            np.column_stack([x_interface_nodes, np.full_like(x_interface_nodes, y_bottom)]),
            np.column_stack([x_interface_nodes, np.full_like(x_interface_nodes, y_top)]),
        ]
    )
    interface_weights = np.concatenate([y_interface_weights, y_interface_weights, x_interface_weights, x_interface_weights])
    interface_normals = np.vstack(
        [
            np.tile([-1.0, 0.0], (len(y_interface_nodes), 1)),
            np.tile([1.0, 0.0], (len(y_interface_nodes), 1)),
            np.tile([0.0, -1.0], (len(x_interface_nodes), 1)),
            np.tile([0.0, 1.0], (len(x_interface_nodes), 1)),
        ]
    )

    avg1 = build_interface_average_matrix(n, delta, omega1_rectangles, x_left_cell, x_right_cell, y_bottom_cell, y_top_cell)
    avg2 = build_interface_average_matrix(n, delta, omega2_rectangles, x_left_cell, x_right_cell, y_bottom_cell, y_top_cell)

    avg_jump = avg1 - avg2
    Phi_interface = sp.diags(interface_weights)

    ux, uy = grad_func(interface_points[:, 0], interface_points[:, 1])
    g_values = (lambda1 - lambda2) * (ux * interface_normals[:, 0] + uy * interface_normals[:, 1])

    penalty = 2 / delta * (avg_jump.T * Phi_interface * avg_jump)
    LHS = (factor * (lambda1 * K_1 + lambda2 * K_2) + penalty).tocsr()
    RHS = (lambda1 * Phi_simpson_1 + lambda2 * Phi_simpson_2) * f_base_func(x_basis, y_basis) + avg1.T * (interface_weights * g_values)

    LHS_mod = sp.bmat([[LHS, Phi_simpson[:, None]], [Phi_simpson[None, :], None]], format="csr")
    RHS_mod = np.concatenate([np.asarray(RHS).ravel(), np.array([0.0])])
    solution = sp.linalg.spsolve(LHS_mod, RHS_mod)
    u_basis = solution[:-1]

    test_1d = np.linspace(0, 1, 1000, endpoint=False)
    X_test, Y_test = np.meshgrid(test_1d, test_1d, indexing="ij")
    x_test, y_test = X_test.ravel(), Y_test.ravel()
    test_interp = build_interp_matrix(n, test_1d)
    u_test = sp.kron(test_interp, test_interp) * u_basis
    error = np.sqrt(np.mean((u_test - u_func(x_test, y_test)) ** 2))
    print("Error:", error)
    print()
