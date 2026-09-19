"""Search for and refine a high-precision numerical candidate for ``prove.py``.

This is a standalone, non-rigorous shooting calculation. It uses ordinary
multiprecision numbers. The first step is to find a coarse shooting root at a
fixed dimension and exponent, then the script continues that solution branch to
the requested parameters. This is done in double precission. Finally, the
script shoots high-precision origin and tail expansions to a common radius,
applies Newton corrections to match them, and writes the refined approximation
to JSON.
"""
import argparse
import json
from pathlib import Path

from mpmath import mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from helpers import profile_spec

S_M = 8  # Matching radius s_m for the numerical search.
S_OUT = 48  # Outer radius s_out.
ORIGIN_SERIES_ORDER = 48
TAIL_SERIES_ORDER = 128
NEWTON_ODE_TOLERANCE = "1e-60"
FINAL_ODE_TOLERANCE = "1e-145"

# The branch search uses cheaper boundaries and ordinary double precision.
COARSE_S_IN = 0.02
COARSE_S_OUT = 32.0
COARSE_ORIGIN_ORDER = 12
COARSE_TAIL_ORDER = 18


def origin_data(params, s_in, series_order, d,
                high_precision=True, p=None):
    """Build origin data using ordinary or multiprecision arithmetic.
    Uses the recurrence in Section 3.2 of the manuscript.
    Output is (q, q') at the inner radius s_in.
    """
    A, omega, _ = params
    if p is None:
        p = profile_spec(d).p
    if high_precision:
        similarity_exp = mp.one/(p-1)
        conjugate_exp = (p-1)//2
        conjugate = mp.conj
        real_part = mp.re
    else:
        similarity_exp = 1/(p-1)
        conjugate_exp = (p-1)/2
        conjugate = np.conj
        real_part = np.real
    # Normalize by keeping c_0 = 1. Multiply by A later.
    c_coeffs = [mp.one if high_precision else 1+0j]
    modulus_squared_coeffs = []
    modulus_power_coeffs = []
    for n in range(series_order):
        modulus_squared_coeffs.append(real_part(sum(
            c_coeffs[j]*conjugate(c_coeffs[n-j])
            for j in range(n+1))))
        if n == 0:
            modulus_power_coeffs.append(
                modulus_squared_coeffs[0]**conjugate_exp)
        else:
            modulus_power_coeffs.append(
                sum(((conjugate_exp+1)*j-n)
                    * modulus_squared_coeffs[j]
                    * modulus_power_coeffs[n-j]
                    for j in range(1, n+1))
                / (n*modulus_squared_coeffs[0]))
        nonlinearity_coeff = sum(
            modulus_power_coeffs[j]*c_coeffs[n-j] for j in range(n+1))
        c_coeffs.append(
            (A**(p-1)*nonlinearity_coeff
             -(omega+1j*(similarity_exp+n))*c_coeffs[n])
            / (2*(n+1)*(2*n+d)))
    c_coeffs = [A*z for z in c_coeffs]
    q = sum(z*s_in**(2*n) for n, z in enumerate(c_coeffs))
    q_prime = sum(2*n*c_coeffs[n]*s_in**(2*n-1)
                  for n in range(1, len(c_coeffs)))
    if high_precision:
        return q, q_prime
    return np.array([q, q_prime])


def tail_data(params, s_out, series_order, d,
              high_precision=True, p=None):
    """Build tail data using ordinary or multiprecision arithmetic.
    Uses the recurrence in Section 3.1 of the manuscript.
    Output is (q, q') at the outer radius s_out.
    """
    _, omega, K = params
    if p is None:
        p = profile_spec(d).p
    if high_precision:
        similarity_exp = mp.one/(p-1)
        l_coeffs = [mp.zero]
        E_coeffs = [mp.one]
        real_part = mp.re
        logarithm = mp.log
        exponential = mp.exp
    else:
        similarity_exp = 1/(p-1)
        l_coeffs = [0j]
        E_coeffs = [1+0j]
        real_part = np.real
        logarithm = np.log
        exponential = np.exp
    beta = -2*similarity_exp+2j*omega
    f_coeffs = [beta]
    for n in range(series_order):
        if n:
            E_coeffs.append(
                sum((p-1)*j*real_part(l_coeffs[j])
                    * E_coeffs[n-j] for j in range(1, n+1))/n)
        H_n = (sum(f_coeffs[j]*f_coeffs[n-j] for j in range(n+1))
               +(d-2-2*n)*f_coeffs[n])
        l_coeffs.append(1j*(K*E_coeffs[n]-H_n)/(n+1))
        f_coeffs.append(-2*(n+1)*l_coeffs[-1])
    log_q = (similarity_exp*logarithm(K)+beta*logarithm(s_out)
             +sum(l_coeffs[n]*s_out**(-2*n) for n in range(1, series_order+1)))
    q = exponential(log_q)
    scaled_log_deriv = sum(
        z*s_out**(-2*n) for n, z in enumerate(f_coeffs))
    if high_precision:
        return q, q*scaled_log_deriv/s_out
    return np.array([q, q*scaled_log_deriv/s_out])


def find_coarse_root(param_guess, d, p):
    """Try one branch point; return physical parameters or ``None``."""
    if param_guess[0] <= 0 or param_guess[2] <= 0:
        return None
    log_A = np.log(param_guess[0])
    omega = param_guess[1]
    log_K = np.log(param_guess[2])
    solver_coords = np.array([(p-1)*log_A, (p-1)*omega, log_K])
    shot_count = 0

    def propagate_to_match(shooting_state, s_start, omega):
        """Integrate one coarse shot to the matching radius."""
        def profile_rhs(radius, state_values):
            q, q_prime = state_values
            return np.array([
                q_prime,
                -((d-1)/radius+0.5j*radius)*q_prime
                -(omega+1j/(p-1))*q+abs(q)**(p-1)*q])

        integration_result = solve_ivp(
            profile_rhs, (s_start, S_M), shooting_state,
            method="DOP853", rtol=2e-10,
            atol=2e-12*min(abs(shooting_state[0]), 1))
        if (not integration_result.success
                or integration_result.t[-1] != S_M):
            raise RuntimeError("search shot did not reach the matching radius")
        return integration_result.y[:, -1]

    def matching_residual(solver_coords):
        """Evaluate the conditioned matching residual for the root solver."""
        nonlocal shot_count
        shot_count += 1
        log_A, omega, log_K = (
            solver_coords[0]/(p-1),
            solver_coords[1]/(p-1),
            solver_coords[2])
        within_bounds = (-1000 < log_A < 1
                         and -100 < omega < 0
                         and -50 < log_K < 25)
        if not within_bounds:
            raise ValueError("search left its parameter bounds")
        params = np.array([np.exp(log_A), omega, np.exp(log_K)])
        inner_data = propagate_to_match(
            origin_data(params, COARSE_S_IN,
                        COARSE_ORIGIN_ORDER, d,
                        high_precision=False, p=p),
            COARSE_S_IN, omega)
        outer_data = propagate_to_match(
            tail_data(params, COARSE_S_OUT,
                      COARSE_TAIL_ORDER, d,
                      high_precision=False, p=p),
            COARSE_S_OUT, omega)
        inner_q, inner_q_prime = inner_data
        outer_q, outer_q_prime = outer_data
        inner_scaled_log_deriv = (
            S_M*inner_q_prime/inner_q)
        outer_scaled_log_deriv = (
            S_M*outer_q_prime/outer_q)
        inner_matching_coords = np.array([
            np.log(abs(inner_q)), inner_scaled_log_deriv.real,
            inner_scaled_log_deriv.imag])
        outer_matching_coords = np.array([
            np.log(abs(outer_q)), outer_scaled_log_deriv.real,
            outer_scaled_log_deriv.imag])
        return (p-1)*(inner_matching_coords-outer_matching_coords)

    try:
        root_result = root(
            matching_residual, solver_coords, method="hybr",
            options={"xtol": 1e-10, "eps": 1e-9, "maxfev": 240})
        residual_norm = np.linalg.norm(matching_residual(root_result.x))
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        return None
    if not np.isfinite(residual_norm) or residual_norm >= 2e-7:
        return None
    params = np.array([np.exp(root_result.x[0]/(p-1)),
                       root_result.x[1]/(p-1),
                       np.exp(root_result.x[2])])
    print(f"  accepted d={d:g}, p={p:g} after {shot_count} shots",
          flush=True)
    return params


def continue_branch(params, continuation_start, continuation_target,
                    fixed_param, vary_dimension):
    """Adaptive continuation in dimension d or exponent p."""
    current_value = float(continuation_start)
    target_value = float(continuation_target)
    continuation_direction = 1 if target_value > current_value else -1
    step_size = 0.05 if vary_dimension else 2.0
    minimum_step_size = 0.001
    maximum_step_size = 0.10 if vary_dimension else 2.0
    previous_point = None
    while continuation_direction*(target_value-current_value) > 1e-12:
        # Use smaller exponent steps below p=13, where the branch is harder
        # to track reliably.
        if (not vary_dimension and continuation_direction < 0
                and current_value <= 13):
            step_size = min(step_size, 0.1)
        next_value = current_value+continuation_direction*min(
            step_size, abs(target_value-current_value))
        param_guess = params.copy()
        if previous_point is not None:
            previous_value, previous_params = previous_point
            #Use linear extrapolation to improve next guess
            param_guess += (
                (next_value-current_value)*(params-previous_params)
                / (current_value-previous_value))
        d, p = ((next_value, fixed_param) if vary_dimension
                else (fixed_param, next_value))
        continued_params = find_coarse_root(param_guess, d, p)
        # Step halves after fail and enlarges after success
        if continued_params is None:
            step_size /= 2
            if step_size < minimum_step_size:
                message = f"continuation stalled before d={d:g}, p={p:g}"
                raise RuntimeError(message)
            continue
        previous_point = (current_value, params.copy())
        current_value, params = next_value, continued_params
        step_size = min(maximum_step_size, step_size*1.5)
    return params


def find_seed(d, p):
    """Find the anchor root, then continue its branch to ``(d, p)``."""
    A_guess = 0.6  # Guesses tuned by trial and error.
    omega_guess = -0.30
    K_guess = 8.0
    param_guess = (A_guess, omega_guess, K_guess)

    # Search for a root at (d,p) = (4,33). The reason behind this choice is
    # that this is the first solution that was found using parameter sweeps.
    # The sweeps are not in this codebase.
    print("Searching for a d=4, p=33 branch point...", flush=True)
    params = find_coarse_root(param_guess, 4.0, 33.0)
    if params is None:
        raise RuntimeError("coarse branch search found no seed")

    # Continue the branch from (4,33) to the requested (d,p) parameters.
    params = continue_branch(params, 4, d, 33, vary_dimension=True)
    params = continue_branch(params, 33, p, d, vary_dimension=False)
    formatted_params = (f"{value:.12g}" for value in params)
    print("Seed search result:", *formatted_params, flush=True)
    return params


def shooting_rhs(radius, shooting_state, d, omega):
    """Evaluate the profile ODE and its three parameter variations."""
    p = profile_spec(d).p
    similarity_exp = mp.one/(p-1)
    conjugate_exp = (p-1)//2
    q, q_prime = shooting_state[:2]
    modulus_squared = q*mp.conj(q)
    modulus_power = modulus_squared**conjugate_exp
    conjugate_deriv = (conjugate_exp*q**2
                       * modulus_squared**(conjugate_exp-1))
    radial_drift = (d-1)/radius+0.5j*radius
    linear_q_coeff = omega+1j*similarity_exp
    derivatives = [
        q_prime,
        -radial_drift*q_prime-linear_q_coeff*q+modulus_power*q,
    ]
    for state_index in range(1, 4):
        sensitivity, sensitivity_prime = (
            shooting_state[2*state_index:2*state_index+2])
        sensitivity_second_deriv = (
            -radial_drift*sensitivity_prime
            -linear_q_coeff*sensitivity
            +(conjugate_exp+1)*modulus_power*sensitivity
            +conjugate_deriv*mp.conj(sensitivity))
        if state_index == 2:  # Omega column.
            sensitivity_second_deriv -= q
        derivatives.extend((sensitivity_prime, sensitivity_second_deriv))
    return derivatives


def shoot_to_match(params, boundary_data, s_start, series_order, d,
                   taylor_degree, ode_tolerance):
    """Build boundary data, propagate it, and evaluate the matching map."""
    # Append the derivatives of (q, q') with respect to A, Omega, and K.
    shooting_state = list(boundary_data(params, s_start, series_order, d))
    for param_ind in range(3):
        for component_index in range(2):
            def boundary_component(
                    x, param_index=param_ind,component_ind=component_index):
                varied_params = params[:]
                varied_params[param_index] = x
                return boundary_data(
                    varied_params, s_start, series_order,d)[component_ind]
            shooting_state.append(
                mp.diff(boundary_component, params[param_ind]))

    # Propagate in a nonnegative travel variable, including for inward shots.
    s_m = mp.mpf(S_M)
    integration_direction = 1 if s_m > s_start else -1

    def directed_rhs(travel, state):
        radius = s_start+integration_direction*travel
        return [integration_direction*value for value in
                shooting_rhs(radius, state, d, params[1])]

    # Use mpmath's Taylor method to integrate the ODE to the matching radius.
    solution = mp.odefun(
        directed_rhs, mp.zero, shooting_state,
        tol=mp.mpf(ode_tolerance), degree=taylor_degree)
    shooting_state = solution(abs(s_m-s_start))

    # Evaluate the phase-invariant matching values.
    q, q_prime = shooting_state[:2]
    scaled_log_deriv = s_m*q_prime/q
    matching_coords = mp.matrix([
        mp.log(abs(q)), mp.re(scaled_log_deriv), mp.im(scaled_log_deriv)])

    # Evaluate Jacobian of the matching map with respect to (A, Omega, K).
    matching_jacobian = mp.matrix(3, 3)
    for param_ind in range(3):
        q_sensitivity, q_prime_sensitivity = (
            shooting_state[2+2*param_ind:4+2*param_ind])
        relative_q_sensitivity = q_sensitivity/q
        log_deriv_sensitivity = s_m*(
            q_prime_sensitivity/q-q_prime*q_sensitivity/q**2)
        jacobian_column = (
            mp.re(relative_q_sensitivity),
            mp.re(log_deriv_sensitivity),
            mp.im(log_deriv_sensitivity))
        for row_ind in range(3):
            matching_jacobian[row_ind, param_ind] = (jacobian_column[row_ind])
    return matching_coords, matching_jacobian


def main():
    """Generate the candidates for d."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True, choices=(3, 4, 5))
    d = parser.parse_args().d
    p = profile_spec(d).p
    mp.dps = 150    # Set the decimal precision for mpmath.
    params = [mp.mpf(str(value)) for value in find_seed(d, p)]

    # Use Newton's method to match the inward and outward shots.
    # Use Taylor expansions of degree 40 in the shooting.
    for newton_iteration in range(8):
        inner_matching_coords, inner_matching_jacobian = shoot_to_match(
            params, origin_data, mp.mpf(1)/8, ORIGIN_SERIES_ORDER,
            d, 40, NEWTON_ODE_TOLERANCE)
        outer_matching_coords, outer_matching_jacobian = shoot_to_match(
            params, tail_data, mp.mpf(S_OUT),
            TAIL_SERIES_ORDER, d, 40, NEWTON_ODE_TOLERANCE)
        matching_residual = (
            inner_matching_coords-outer_matching_coords)
        residual_norm = max(abs(value) for value in matching_residual)
        print(
            f"Newton {newton_iteration}: matching residual "
            f"{mp.nstr(residual_norm, 6)}", flush=True)
        if residual_norm < mp.mpf("1e-50"):
            break
        newton_step = mp.lu_solve(
            inner_matching_jacobian-outer_matching_jacobian,
            matching_residual)
        params = [value-newton_step[j] for j, value in enumerate(params)]
    else:
        raise RuntimeError("candidate refinement did not converge")

    # Recompute both shots with Taylor expansions of degree 72 for one final,
    # higher-accuracy approximate solution.
    inner_matching_coords, inner_matching_jacobian = shoot_to_match(
        params, origin_data, mp.mpf(1)/8, ORIGIN_SERIES_ORDER,
        d, 72, FINAL_ODE_TOLERANCE)
    outer_matching_coords, outer_matching_jacobian = shoot_to_match(
        params, tail_data, mp.mpf(S_OUT), TAIL_SERIES_ORDER,
        d, 72, FINAL_ODE_TOLERANCE)
    matching_residual = (
        inner_matching_coords-outer_matching_coords)
    newton_step = mp.lu_solve(
        inner_matching_jacobian-outer_matching_jacobian, matching_residual)
    params = [value-newton_step[j] for j, value in enumerate(params)]
    print("Last residual:", mp.nstr(max(map(abs, matching_residual)), 6))

    # Generate the JSON output file
    candidate_data = {"d": d, "p": p, **{
        key: mp.nstr(value, 140)
        for key, value in zip(("A", "omega", "K"), params)}}
    output_path = Path(__file__).with_name(f"candidate-d{d}-p{p}.json")
    output_path.write_text(json.dumps(candidate_data, indent=2)+"\n")


if __name__ == "__main__":
    main()
