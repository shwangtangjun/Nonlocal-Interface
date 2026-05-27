import numpy as np
from scipy.special import erf
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss


def cubic_lagrange_basis_coeffs(left, n):
    """
    Return the coefficients of cubic Lagrange basis polynomials on one cell [left, left + 1 / n],
    with four interpolation nodes

        left,
        left + 1 / (3 * n),
        left + 2 / (3 * n),
        left + 1 / n.

    Parameters
    ----------
    left : float
        Left endpoint of the cell.
    n : int
        Number of cells with cell size 1 / n.

    Returns
    -------
    coeff : np.ndarray of shape (4, 4)
        Coefficients of the four cubic Lagrange basis polynomials.

        The i-th row contains the coefficients of the basis polynomial
        associated with the i-th interpolation node. The coefficients are
        ordered by descending powers:

            coeff[i] = [a3, a2, a1, a0],

        meaning that the i-th basis polynomial is

            a3 * x^3 + a2 * x^2 + a1 * x + a0.
    """
    scale = 3 * n

    t0 = left * scale
    t1 = t0 + 1
    t2 = t1 + 1
    t3 = t2 + 1

    return np.array(
        [
            -1 / 6 * np.array([scale**3, -(scale**2) * (t1 + t2 + t3), scale * (t1 * t2 + t1 * t3 + t2 * t3), -t1 * t2 * t3]),
            1 / 2 * np.array([scale**3, -(scale**2) * (t0 + t2 + t3), scale * (t0 * t2 + t0 * t3 + t2 * t3), -t0 * t2 * t3]),
            -1 / 2 * np.array([scale**3, -(scale**2) * (t0 + t1 + t3), scale * (t0 * t1 + t0 * t3 + t1 * t3), -t0 * t1 * t3]),
            1 / 6 * np.array([scale**3, -(scale**2) * (t0 + t1 + t2), scale * (t0 * t1 + t0 * t2 + t1 * t2), -t0 * t1 * t2]),
        ]
    )


def f0(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2) dx

    using the analytical formula based on the error function.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return np.sqrt(np.pi) / 2 * (erf(eta * b) - erf(eta * a)) / eta


def f1(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return (np.exp(-eta * eta * a * a) - np.exp(-eta * eta * b * b)) / (2 * eta * eta)


def f2(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x^2 dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return a * np.exp(-eta * eta * a * a) / (2 * eta * eta) - b * np.exp(-eta * eta * b * b) / (2 * eta * eta) + f0(eta, a, b) / (2 * eta * eta)


def f3(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x^3 dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return a * a * np.exp(-eta * eta * a * a) / (2 * eta * eta) - b * b * np.exp(-eta * eta * b * b) / (2 * eta * eta) + f1(eta, a, b) / (eta * eta)


def build_quadrature(n, delta, num_leggauss=5):
    """
    Build Gauss-Legendre quadrature nodes and weights on the cell [0, 1 / n].

    If the cell is sufficiently longer than the nonlocal scale, namely
    1 / n > 6 * delta, the cell is split into three subintervals:

        [0, 3 * delta],
        [3 * delta, 1 / n - 3 * delta],
        [1 / n - 3 * delta, 1 / n].

    Gauss-Legendre quadrature with `num_leggauss` points is then applied
    on each subinterval.

    Otherwise, Gauss-Legendre quadrature with `3 * num_leggauss` points
    is applied directly on [0, 1 / n].

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    delta : float
        Nonlocal interaction radius.
    num_leggauss : int, optional
        Number of Gauss-Legendre points used on each subinterval.
        Default is 5.

    Returns
    -------
    quadrature_nodes : np.ndarray of shape (3 * num_leggauss,)
        Quadrature nodes on [0, 1 / n].
    quadrature_weights : np.ndarray of shape (3 * num_leggauss,)
        Quadrature weights corresponding to `quadrature_nodes`.
    """
    singular_width = 3 * delta
    if 1 / n > 2 * singular_width:
        leggauss_nodes, leggauss_weights = leggauss(num_leggauss)

        a = 0
        b = singular_width
        left_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        left_weights = 0.5 * (b - a) * leggauss_weights

        a = singular_width
        b = 1 / n - singular_width
        middle_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        middle_weights = 0.5 * (b - a) * leggauss_weights

        right_nodes = 1 / n - np.flip(left_nodes)
        right_weights = np.flip(left_weights)

        quadrature_nodes = np.concatenate([left_nodes, middle_nodes, right_nodes])
        quadrature_weights = np.concatenate([left_weights, middle_weights, right_weights])
    else:
        leggauss_nodes, leggauss_weights = leggauss(3 * num_leggauss)
        a = 0
        b = 1 / n
        quadrature_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        quadrature_weights = 0.5 * (b - a) * leggauss_weights

    return quadrature_nodes, quadrature_weights


def compute_relevant_cells_coeffs(n, quadrature_node, delta, tol=1e-12):
    """
    Compute kernel-weighted basis integrals for all relevant cells.
    For a fixed quadrature node x_quad in [0, 1 / n), this function computes

        ∫_{k/n}^{(k+1)/n} R_delta(x_quad, y) psi_i(y) dy,

    for integer cell offsets k = ..., -2, -1, 0, 1, 2, ... and
    i = 0, 1, 2, 3, where psi_i are the four cubic Lagrange basis
    polynomials on the cell [k / n, (k + 1) / n].

    For the computation on each single cell, see
    `compute_single_cell_coeffs`.

    Since the nonlocal interaction radius delta is typically small, the
    coefficients decay rapidly as |k| increases. The cell offsets are
    therefore truncated adaptively once the contributions on both sides
    are smaller than `tol`.

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    quadrature_node : float
        A single quadrature node in [0, 1 / n).
    delta : float
        Nonlocal interaction radius.
    tol : float, optional
        Truncation tolerance for ignoring far-away cell contributions.
        Default is 1e-12.

    Returns
    -------
    relevant_cells_coeffs : np.ndarray of shape (num_relevant_cells, 4)
        Kernel-weighted integrals over the relevant neighboring cells.
        The row `relevant_cells_coeffs[j]` corresponds to one cell offset (j - center_cell_index), and the
        four columns correspond to the four cubic basis functions on that cell.
    center_cell_index : int
        Index of the row in `relevant_cells_coeffs` corresponding to cell offset k = 0.
        In other words, the position of the cell that contains the quadrature node.
    """

    def compute_single_cell_coeffs(k):
        """
        Compute kernel-weighted basis integrals over a single cell.

        For a fixed quadrature node x_quad, this function evaluates

            I_i(k) = ∫_{k/n}^{(k+1)/n} R_delta(x_quad, y) psi_i(y) dy,

        for i = 0, 1, 2, 3, where R_delta is the scaled Gaussian kernel and
        psi_i are cubic Lagrange basis polynomials.

        The integral is computed analytically by combining the coefficients
        of each cubic basis polynomial with Gaussian moments.

        Parameters
        ----------
        k : int
            Cell index. The integration domain is [k / n, (k + 1) / n].

        Returns
        -------
        single_cell_coeffs : np.ndarray of shape (4,)
            Values [I_0(k), I_1(k), I_2(k), I_3(k)].
        """
        eta = 1 / delta

        # change of variable, simply shift
        left = k / n - quadrature_node
        right = (k + 1) / n - quadrature_node
        gaussian_moments = np.array([f3(eta, left, right), f2(eta, left, right), f1(eta, left, right), f0(eta, left, right)])
        basis_coeffs = cubic_lagrange_basis_coeffs(k / n - quadrature_node, n)
        single_cell_coeffs = gaussian_moments @ basis_coeffs.T / delta
        return single_cell_coeffs

    coeffs_by_cell = {0: compute_single_cell_coeffs(0)}

    cell_offset = 1
    while True:
        right_cell_impacts = compute_single_cell_coeffs(cell_offset)
        left_cell_impacts = compute_single_cell_coeffs(-cell_offset)

        if max(np.linalg.norm(right_cell_impacts), np.linalg.norm(left_cell_impacts)) < tol:
            break

        if np.linalg.norm(right_cell_impacts) >= tol:
            coeffs_by_cell[cell_offset] = right_cell_impacts

        if np.linalg.norm(left_cell_impacts) >= tol:
            coeffs_by_cell[-cell_offset] = left_cell_impacts

        cell_offset += 1

    cell_offsets = sorted(coeffs_by_cell.keys())
    relevant_cells_coeffs = np.array([coeffs_by_cell[k] for k in cell_offsets])
    center_cell_index = cell_offsets.index(0)

    return relevant_cells_coeffs, center_cell_index


def build_interp_matrix(n, x_interp):
    """
    Compute the sparse matrix A such that

        u(x_interp)= A @ u_basis,

    where `x_interp` denotes points of shape (num_interp, ) to be interpolated using cubic basis polynomials,
    `u_basis` contains the values of u (to be solved) at the global cubic basis nodes of shape (3 * n + 1, ),
    A is the sparse matrix of shape (num_interp, 3 * n + 1).

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    x_interp : np.ndarray of shape (num_interp, )
        Points in [0, 1].

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (num_interp, 3 * n + 1)
        Sparse matrix whose rows correspond to x_interp and
        whose columns correspond to global cubic basis nodes.
    """
    num_interp = len(x_interp)

    # x_interp = q * (1 / n) + r
    q = np.floor(n * x_interp).astype(int)
    q = np.clip(q, 0, n - 1)  # corner case x_interp == 1
    r = x_interp - q / n

    interp_coeffs = np.stack([np.polyval(coeff, r) for coeff in cubic_lagrange_basis_coeffs(0, n)], axis=1).ravel()
    i, j = np.indices((num_interp, 4))
    row_indices = i.ravel()
    col_indices = (q[:, None] * 3 + j).ravel()
    A = sp.coo_matrix((interp_coeffs, (row_indices, col_indices)), shape=(num_interp, 3 * n + 1))

    return A.tocsr()


def build_subdomain_coeffs_one_row(relevant_cells_coeffs, center_cell_index, cell_idx, cell_start, cell_end):
    """
    Assemble the nonzero entries of one subdomain kernel-matrix row.

    For a fixed local quadrature node and a fixed global cell index `cell_idx`,
    this function accumulates the contributions from nearby relevant cells and
    maps them to the corresponding global cubic basis nodes. The valid integration
    cells are clipped to the subdomain [cell_start, cell_end), which is needed
    for sharp interface models where interactions do not cross the interface.

    The input `relevant_cells_coeffs` stores the kernel-weighted coefficients
    associated with relative cell offsets around the quadrature node. Near the
    subdomain boundary, only the subset of offsets whose global cell indices
    remain in [cell_start, cell_end) is active.

    Parameters
    ----------
    relevant_cells_coeffs : np.ndarray of shape (num_relevant_cells, 4)
        Kernel-weighted coefficients for relevant cell offsets. Each row
        contains the four contributions from one nearby cell.
    center_cell_index : int
        Index of the row in `relevant_cells_coeffs` corresponding to the
        cell offset k = 0, i.e. the cell containing the quadrature node.
    cell_idx : int
        Global index of the cell containing the quadrature node.
    cell_start : int
        First global cell index of the subdomain, inclusive.
    cell_end : int
        Last global cell index of the subdomain, exclusive.

    Returns
    -------
    values : np.ndarray of shape (num_active_basis_nodes,)
        Nonzero values in the matrix row for this subdomain quadrature node.
        Shared cubic basis nodes from adjacent active cells are already
        accumulated.
    cols : np.ndarray of shape (num_active_basis_nodes,)
        Column indices of the global cubic basis nodes corresponding to
        `values`.
    """
    offset_start = max(-center_cell_index, cell_start - cell_idx)
    offset_end = min(len(relevant_cells_coeffs) - center_cell_index, cell_end - cell_idx)
    num_active_cells = offset_end - offset_start
    num_active_basis_nodes = 3 * num_active_cells + 1

    values = np.zeros(num_active_basis_nodes)
    base_idx = 3 * np.arange(num_active_cells)[:, None] + np.arange(4)
    np.add.at(values, base_idx, relevant_cells_coeffs[center_cell_index + offset_start : center_cell_index + offset_end])

    col_start = 3 * (cell_idx + offset_start)
    col_end = col_start + num_active_basis_nodes
    cols = np.arange(col_start, col_end, dtype=np.int64)
    return values, cols


def build_subdomain_coeffs_matrix(n, quadrature_nodes, delta, cell_start, cell_end, integral_cell_start=None, integral_cell_end=None):
    """
    Compute the sparse subdomain kernel matrix A.

    For quadrature points x located in the subdomain
    [cell_start / n, cell_end / n), this function builds A such that

        ∫_{integral_cell_start/n}^{integral_cell_end/n} R_delta(x, y) u(y) dy = A @ u_basis,

    where `u_basis` contains the values of u at the global cubic basis nodes on
    [0, 1]. The rows correspond only to quadrature points inside
    [cell_start / n, cell_end / n), while the columns still use the global
    basis numbering. If `integral_cell_start` and `integral_cell_end` are not
    provided, the integration interval is the same as the row subdomain. Thus
    the usual subdomain matrix is a special case of this cross-interval matrix.

    Parameters
    ----------
    n : int
        Number of cells on [0, 1], with cell size 1 / n.
    quadrature_nodes : np.ndarray of shape (num_quad,)
        Local quadrature nodes on [0, 1 / n).
    delta : float
        Nonlocal interaction radius.
    cell_start : int
        First global cell index of the subdomain, inclusive.
    cell_end : int
        Last global cell index of the row subdomain, exclusive.
    integral_cell_start : int, optional
        First global cell index of the integration interval, inclusive. Defaults
        to `cell_start`.
    integral_cell_end : int, optional
        Last global cell index of the integration interval, exclusive. Defaults
        to `cell_end`.

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape ((cell_end - cell_start) * num_quad, 3 * n + 1)
        Sparse matrix whose rows correspond to subdomain quadrature nodes and
        whose columns correspond to global cubic basis nodes.
    """
    if integral_cell_start is None:
        integral_cell_start = cell_start
    if integral_cell_end is None:
        integral_cell_end = cell_end

    row_blocks, col_blocks, data_blocks = [], [], []
    num_quad = len(quadrature_nodes)
    num_cells = cell_end - cell_start

    if num_cells == 0 or integral_cell_start == integral_cell_end:
        return sp.csr_matrix((num_cells * num_quad, 3 * n + 1))

    for quad_idx, quadrature_node in enumerate(quadrature_nodes):
        relevant_cells_coeffs, center_cell_index = compute_relevant_cells_coeffs(n, quadrature_node, delta)

        for cell_idx in range(cell_start, cell_end):
            offset_start = max(-center_cell_index, integral_cell_start - cell_idx)
            offset_end = min(len(relevant_cells_coeffs) - center_cell_index, integral_cell_end - cell_idx)
            if offset_end <= offset_start:
                continue

            global_coeffs, cols = build_subdomain_coeffs_one_row(
                relevant_cells_coeffs, center_cell_index, cell_idx, integral_cell_start, integral_cell_end
            )
            rows = np.full(len(cols), quad_idx + (cell_idx - cell_start) * num_quad, dtype=np.int64)

            row_blocks.append(rows)
            col_blocks.append(cols)
            data_blocks.append(global_coeffs)

    if not row_blocks:
        return sp.csr_matrix((num_cells * num_quad, 3 * n + 1))

    row_indices = np.concatenate(row_blocks)
    col_indices = np.concatenate(col_blocks)
    data_entries = np.concatenate(data_blocks)

    A = sp.coo_matrix((data_entries, (row_indices, col_indices)), shape=(num_cells * num_quad, 3 * n + 1))
    return A.tocsr()


def build_subdomain_stiffness(n, quadrature_nodes, quadrature_weights, delta, cell_start, cell_end, integral_cell_start=None, integral_cell_end=None):
    """
    Assemble the stiffness matrix for one subdomain nonlocal Dirichlet energy.

    This function discretizes the variational derivative of

        1 / (2 * delta^2) ∫_{Omega_i} ∫_{Omega_i} R_delta(x, y) |u(y) - u(x)|^2 dy dx,

    Usually, Omega_i = [cell_start / n, cell_end / n). More generally, the
    quadrature points x may lie in [cell_start / n, cell_end / n), while the
    integral variable y may lie in
    [integral_cell_start / n, integral_cell_end / n). If the latter interval is
    not provided, it defaults to the same interval as x. The coefficient lambda
    is not included here; callers can multiply the returned matrix by the
    desired subdomain coefficient.

    The implementation first constructs A for

        ∫_{Omega_i} R_delta(x, y) u(y) dy,

    and B for

        u(x) ∫_{Omega_i} R_delta(x, y) dy.

    Then L = B - A represents the nonlocal Laplacian on quadrature points,
    and the final symmetric stiffness is

        (L + L.T) / delta^2.

    Parameters
    ----------
    n : int
        Number of cells on [0, 1], with cell size 1 / n.
    quadrature_nodes : np.ndarray of shape (num_quad,)
        Local quadrature nodes on [0, 1 / n).
    quadrature_weights : np.ndarray of shape (num_quad,)
        Quadrature weights corresponding to `quadrature_nodes`.
    delta : float
        Nonlocal interaction radius.
    cell_start : int
        First global cell index of the x interval, inclusive.
    cell_end : int
        Last global cell index of the x interval, exclusive.
    integral_cell_start : int, optional
        First global cell index of the y integration interval, inclusive.
        Defaults to `cell_start`.
    integral_cell_end : int, optional
        Last global cell index of the y integration interval, exclusive.
        Defaults to `cell_end`.

    Returns
    -------
    B_part : scipy.sparse.csr_matrix of shape (3 * n + 1, 3 * n + 1)
        Matrix for the local-in-x part of the one-dimensional contribution.
    A_part : scipy.sparse.csr_matrix of shape (3 * n + 1, 3 * n + 1)
        Matrix for the kernel-integral part of the one-dimensional contribution.
        Their difference `B_part - A_part` is the unsymmetrized contribution
        before the final `delta**2` scaling.
    """
    if integral_cell_start is None:
        integral_cell_start = cell_start
    if integral_cell_end is None:
        integral_cell_end = cell_end

    cells = np.arange(cell_start, cell_end)
    subdomain_quadrature_nodes = ((cells / n)[:, None] + quadrature_nodes[None, :]).ravel()
    subdomain_quadrature_weights = np.tile(quadrature_weights, cell_end - cell_start)

    A = build_subdomain_coeffs_matrix(n, quadrature_nodes, delta, cell_start, cell_end, integral_cell_start, integral_cell_end)
    quadrature_nodes_interp_matrix = build_interp_matrix(n, subdomain_quadrature_nodes)
    Phi = sp.diags(subdomain_quadrature_weights)

    B_part = quadrature_nodes_interp_matrix.T * Phi * sp.diags(A.sum(axis=1).A1) * quadrature_nodes_interp_matrix
    A_part = quadrature_nodes_interp_matrix.T * Phi * A
    return B_part.tocsr(), A_part.tocsr()


def build_subdomain_kernel_coeffs_vector(n, delta, x0, cell_start, cell_end):
    """
    Build an unnormalized one-sided kernel coefficient vector on a 1D cell interval.

    For a fixed point x0 and a subdomain [cell_start / n, cell_end / n), this
    function constructs a vector `coeffs` such that

        coeffs @ u_basis

    approximates

        ∫_{cell_start/n}^{cell_end/n} R_delta(x0 , y) u(y) dy.

    Parameters
    ----------
    n : int
        Number of cells on [0, 1], with cell size 1 / n.
    delta : float
        Nonlocal interaction radius.
    x0 : float
        Point where the kernel is centered.
    cell_start : int
        First global cell index of the integration interval, inclusive.
    cell_end : int
        Last global cell index of the integration interval, exclusive.

    Returns
    -------
    coeffs : np.ndarray of shape (3 * n + 1,)
        Unnormalized kernel-weighted coefficients on the global cubic basis.
        Entries are zero outside the requested integration interval and outside
        the effective kernel support.
    """
    q = np.floor(n * x0).astype(int)
    q = np.clip(q, 0, n - 1)
    r = x0 - q / n
    relevant_cells_coeffs, center_cell_index = compute_relevant_cells_coeffs(n, r, delta)

    offset_start = max(-center_cell_index, cell_start - q)
    offset_end = min(len(relevant_cells_coeffs) - center_cell_index, cell_end - q)
    coeffs = np.zeros(3 * n + 1)
    if offset_end <= offset_start:
        return coeffs

    values, cols = build_subdomain_coeffs_one_row(relevant_cells_coeffs, center_cell_index, q, cell_start, cell_end)
    coeffs[cols] = values
    return coeffs


def build_subdomain_simpson_weights(n, cell_start, cell_end):
    """
    Construct composite Simpson (3/8 rule) weights on a uniform cubic grid on a subdomain.

    The grid consists of (3n + 1) nodes corresponding to n cells with
    cubic Lagrange basis (3 degrees of freedom per cell plus one endpoint).

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    cell_start : int
        Starting cell index.
    cell_end : int
        Ending cell index.

    Returns
    -------
    weights : np.ndarray of shape (3 * n + 1,)
        Quadrature weights corresponding to the nodes in subdomain cells. Zero otherwise.
    """
    weights = np.zeros(3 * n + 1)
    local_weights = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8]) / n

    for cell_idx in range(cell_start, cell_end):
        weights[3 * cell_idx : 3 * cell_idx + 4] += local_weights

    return weights


def build_subdomain_simpson_weights_2d(n, rectangles):
    """
    Construct tensor-product composite Simpson (3/8 rule) weights on 2D cell rectangles.

    The grid consists of (3n + 1)^2 nodes corresponding to an n by n Cartesian
    cell mesh with tensor-product cubic Lagrange basis functions. Each rectangle
    is specified in cell indices, and the returned weights are nonzero only on
    the union of the listed rectangles.

    Parameters
    ----------
    n : int
        Number of cells in each coordinate direction on [0, 1]^2.
    rectangles : list of tuple[int, int, int, int]
        Cell rectangles of the form (x_start, x_end, y_start, y_end), where
        starts are inclusive and ends are exclusive.

    Returns
    -------
    weights : np.ndarray of shape ((3 * n + 1) ** 2,)
        Tensor-product quadrature weights on the global 2D cubic grid. The
        flattening order is consistent with np.kron(weights_x, weights_y).
    """
    weights = np.zeros((3 * n + 1) ** 2)
    for x0, x1, y0, y1 in rectangles:
        weights_x = build_subdomain_simpson_weights(n, x0, x1)
        weights_y = build_subdomain_simpson_weights(n, y0, y1)
        weights += np.kron(weights_x, weights_y)
    return weights
