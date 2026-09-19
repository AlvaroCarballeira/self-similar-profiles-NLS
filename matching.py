"""Implementation of matching map and Brouwer inclusion bounds."""
from flint import arb, arb_mat

from helpers import require, matrix_infinity_norm, encode


def matching_map(data, s_m):
    """Evaluate H(q,q') and its parameter Jacobian at the matching radius s_m."""
    q, q_prime = data[0]
    require(q.abs_lower() > 0, "Matching denominator can vanish")
    scaled_log_deriv = s_m*q_prime/q
    matching_coords = [
        abs(q).log(), scaled_log_deriv.real, scaled_log_deriv.imag]
    jacobian_cols = []
    for q_deriv, q_prime_deriv in data[1:]:
        relative_q_deriv = q_deriv/q
        log_deriv_sensitivity = s_m*(
            q_prime_deriv/q-q_prime*q_deriv/q**2
        )
        jacobian_cols.append([
            relative_q_deriv.real,
            log_deriv_sensitivity.real,
            log_deriv_sensitivity.imag,
        ])
    matching_vector = arb_mat([[value] for value in matching_coords])
    jacobian = arb_mat([
        [jacobian_cols[col_index][row_index] for col_index in range(3)]
        for row_index in range(3)
    ])
    return matching_vector, jacobian


def matching_boundary_error(flow, s_m):
    r"""If Q is an exact solution and \tilde{Q} is the truncated-data solution,
    this bounds |H(Q)-H(\tilde{Q})| coordinate by coordinate.
    
    The lower bound |q|-b keeps the exact quotient well defined.  The same
    bound for s_m q'/q controls both real and imaginary components.
    """
    q, q_prime = flow["family"][0]
    boundary_error = flow["boundary_error"]
    q_lower = q.abs_lower()
    require(q_lower > boundary_error, "True matching denominator can vanish")
    log_amplitude_error = (
        boundary_error/(q_lower-boundary_error)).upper()
    log_deriv_error = (
        s_m*boundary_error*(1+q_prime.abs_upper()/q_lower)
        / (q_lower-boundary_error)).upper()
    return [log_amplitude_error, log_deriv_error, log_deriv_error]


def brouwer_bounds(inner_flow, outer_flow, s_m, rho):
    """Verify condition Y + E + Z*rho < rho in the manuscript. 
    """
    inner_coords, inner_jacobian = matching_map(inner_flow["center"], s_m)
    outer_coords, outer_jacobian = matching_map(outer_flow["center"], s_m)
    M = inner_coords-outer_coords
    J = inner_jacobian-outer_jacobian
    _, inner_box_jacobian = matching_map(inner_flow["family"], s_m)
    _, outer_box_jacobian = matching_map(outer_flow["family"], s_m)
    J_box = inner_box_jacobian-outer_box_jacobian

    # The midpoint of the interval inverse is the fixed real matrix B.
    inverse = J.inv()
    B = arb_mat([[inverse[i, j].mid() for j in range(3)] for i in range(3)])
    determinant = B.det()
    require(not determinant.contains(0), "Preconditioner not certified invertible")

    Y = matrix_infinity_norm(B*M)
    identity = arb_mat([[int(i == j) for j in range(3)] for i in range(3)])
    Z = matrix_infinity_norm(identity-B*J_box)
    e_inner, e_outer = [
        matching_boundary_error(flow, s_m)
        for flow in (inner_flow, outer_flow)]
    e_sum = [
        (inner+outer).upper()
        for inner, outer in zip(e_inner, e_outer)]
    E = max(sum((B[i, j].abs_upper()*e_sum[j]
                 for j in range(3)), arb(0)).upper()
            for i in range(3))
    total = (Y+E+Z*rho).upper()

    record = {
        "Y": encode(Y), "E": encode(E), "Z": encode(Z),
        "total": encode(total),
        "inclusion_ratio": encode((total/rho).upper()),
    }
    require(Z < 1 and total < rho, "Brouwer self-map inequality failed")
    return record
