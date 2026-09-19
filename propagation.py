"""Rigorous polynomial shooting from r_in or r_out to the matching radius r_m.
Returns the four state pairs P, P_A, P_Omega, P_K approximating the finite
profile state (Q, Q') and its A-, Omega-, and K-derivatives on the step.
"""
from flint import arb, acb, acb_poly, acb_series, ctx
from helpers import (require, norm, pair_radius, polynomial_norm,
                     conjugate_poly, encode, inflate, profile_spec)

I = acb(0, 1)


def polynomial_segment(r_star, h, initial_columns, omega_0, taylor_degree, d):
    r"""Build midpoint Taylor polynomials on r=r_star+h*u, 0<=u<=1 to Q, Q' and
    their parameter derivatives. It implements the approximate polynomials
    P and P_xi in the manuscript.

    Returns the four state pairs P, P_A, P_Omega, P_K approximating the
    finite profile state (Q, Q') and its A-, Omega-, and K-derivatives on
    the step. Output polynomials have coefficients given by the midpoints
    of Arb balls. Enclosures for the full ODE residuals are computed later.
    """
    spec = profile_spec(d)
    conjugate_exp = spec.conjugate_exp
    old_cap = ctx.cap
    try: 
        ctx.cap = taylor_degree+1 # Number of Taylor coefficients stored
        q_series = [
            acb_series([pair[0]], prec=taylor_degree+1)
            for pair in initial_columns]
        v_series = [
            acb_series([pair[1]], prec=taylor_degree+1)
            for pair in initial_columns]
        r_series = acb_series([r_star, h], prec=taylor_degree+1)
        radial_drift = (d-1)/r_series+I*r_series/2
        linear_q_coeff = acb(omega_0, spec.similarity_exp)
        for n in range(taylor_degree):
            # Conjugation means conjugation of coefficients, for REAL u.
            q_conjugate = acb_series(
                [z.conjugate() for z in q_series[0].coeffs()],
                prec=taylor_degree+1)
            modulus_squared = q_series[0]*q_conjugate
            lower_modulus_power = modulus_squared**(conjugate_exp-1)
            modulus_power = lower_modulus_power*modulus_squared
            conjugate_deriv_term = (
                conjugate_exp*q_series[0]*q_series[0]*lower_modulus_power)
            nonlinearity = modulus_power*q_series[0]
            for column_index in range(4):
                rhs = (-radial_drift*v_series[column_index]
                       - linear_q_coeff*q_series[column_index])
                if column_index == 0:
                    rhs += nonlinearity
                else:
                    sensitivity_conjugate = acb_series(
                        [z.conjugate()
                         for z in q_series[column_index].coeffs()],
                        prec=taylor_degree+1)
                    rhs += ((conjugate_exp+1)*modulus_power
                            * q_series[column_index]
                            + conjugate_deriv_term*sensitivity_conjugate)
                    # Columns are Q, partial_A Q, partial_Omega Q, partial_K Q,
                    # where each Q denotes the two-component profile state.
                    if column_index == 2:
                        rhs -= q_series[0]
                q_next = (h*v_series[column_index][n]/(n+1)).mid()
                v_next = (h*rhs[n]/(n+1)).mid()
                q_series[column_index][n+1] = q_next
                v_series[column_index][n+1] = v_next
        return [(acb_poly(q.coeffs()), acb_poly(v.coeffs()))
                for q, v in zip(q_series, v_series)]
    finally:
        ctx.cap = old_cap


def defect_bounds(r_star, h, step_polynomials, omega_0, d):
    r"""Implementation of the upper bounds \delta_0, \delta_A, \delta_\Omega
    and \delta_K in the manuscript. Substitute midpoint Taylor polynomials 
    into the ODE and bound the residuals on 0<=u<=1.
    """
    spec = profile_spec(d)
    conjugate_exp = spec.conjugate_exp
    r_poly = acb_poly([r_star, h])
    r_min = min(r_star, r_star+h)
    p_q = step_polynomials[0][0]
    q_conjugate = conjugate_poly(p_q)
    modulus_squared = p_q*q_conjugate
    lower_modulus_power = modulus_squared**(conjugate_exp-1)
    modulus_power = lower_modulus_power*modulus_squared
    conjugate_deriv_term = conjugate_exp*p_q*p_q*lower_modulus_power
    radial_drift_numerator = (d-1)+(I/2)*r_poly*r_poly
    linear_q_coeff = acb(omega_0, spec.similarity_exp)
    deltas = []
    for column_index, (p_column_q, p_column_v) in enumerate(step_polynomials):
        q_defect = p_column_q.derivative()-h*p_column_v
        v_defect = (r_poly*p_column_v.derivative()
                    + h*radial_drift_numerator*p_column_v
                    + h*r_poly*linear_q_coeff*p_column_q)
        if column_index == 0:
            v_defect -= h*r_poly*modulus_power*p_q
        else:
            v_defect -= h*r_poly*(
                (conjugate_exp+1)*modulus_power*p_column_q
                + conjugate_deriv_term*conjugate_poly(p_column_q))
            if column_index == 2:
                v_defect += h*r_poly*p_q
        deltas.append(norm([
            polynomial_norm(q_defect),
            (polynomial_norm(v_defect)/r_min).upper()]))
    return deltas


def initialize(center_data, parameter_box_data, delta_bd):
    """Separate midpoint rounding, parameter spread, and boundary error."""
    polynomial_columns = [[z.mid() for z in pair] for pair in center_data]
    epsilon_poly = [pair_radius(pair) for pair in center_data]
    parameter_spread = [norm([box-center for box, center in zip(
        parameter_pair, center_pair)])
        for parameter_pair, center_pair
        in zip(parameter_box_data, center_data)]
    return {"polynomial_columns": polynomial_columns,
            "epsilon_poly": epsilon_poly,
            "parameter_spread": parameter_spread,
            "boundary_error": delta_bd}


def propagate(initial_data, r_start, r_end, omega_0, omega_interval, rho, d,
              taylor_degree=88):
    r"""Rigorously integrate (q, q') and its (A, \Omega, K) derivatives from
    r_start to r_end. Apply the polynomial error-propagation estimates to
    every step.
    """
    spec = profile_spec(d)
    p = spec.p
    max_step = arb(1)/16
    epsilon = arb("1e-8")
    polynomial_columns = initial_data["polynomial_columns"]
    epsilon_poly = initial_data["epsilon_poly"][:]
    parameter_spread = initial_data["parameter_spread"][:]
    delta_bd = initial_data["boundary_error"]
    require(epsilon_poly[0]+parameter_spread[0]+delta_bd < epsilon,
            "Initial tube exceeded")
    direction = 1 if r_end > r_start else -1
    r_star = r_start
    minimum_modulus = None
    while direction*(r_end-r_star) > 0:
        step_length = min(max_step, r_star/8, (r_end-r_star).abs_upper())
        h = direction*step_length
        step_polynomials = polynomial_segment(
            r_star, h, polynomial_columns, omega_0, taylor_degree, d)
        deltas = defect_bounds(r_star, h, step_polynomials, omega_0, d)
        p_q_sup = polynomial_norm(step_polynomials[0][0])
        M_step = (p_q_sup+epsilon).upper()
        r_min = min(r_star, r_star+h)
        logarithmic_norm = (
            (1+omega_interval.abs_upper()+spec.similarity_exp
             + p*M_step**(p-1))/2
            + ((d-1)/r_min if direction < 0 else 0)).upper()
        g = (logarithmic_norm*step_length).exp().upper()
        nonlinear_curvature = (p*(p-1)*M_step**(p-2)).upper()

        epsilon_poly_plus = (g*(epsilon_poly[0]+deltas[0])).upper()
        r_Q_plus = (
            g*(parameter_spread[0]+step_length*rho*M_step)).upper()
        delta_bd_plus = (g*delta_bd).upper()
        # Hypothesis of Lemma 3.9
        require(
            epsilon_poly_plus+r_Q_plus+delta_bd_plus < epsilon,
            f"Propagation tube failed at r_*={r_star}, "
            f"epsilon_poly={epsilon_poly_plus}, "
            f"r_Q={r_Q_plus}, delta_bd={delta_bd_plus}")
 
        epsilon_poly_xi_plus, r_xi_plus = [], []
        for column_index in range(1, 4):
            V_xi = norm([
                polynomial_norm(poly)
                for poly in step_polynomials[column_index]])
            omega_source = int(column_index == 2)
            epsilon_xi_plus = (g*(
                epsilon_poly[column_index]+deltas[column_index]
                + step_length*(nonlinear_curvature*V_xi+omega_source)
                * epsilon_poly_plus)).upper()
            parameter_xi_plus = (g*(
                parameter_spread[column_index]
                + step_length*((nonlinear_curvature*r_Q_plus+rho)
                               * (V_xi+epsilon_xi_plus)
                               + omega_source*r_Q_plus))).upper()
            epsilon_poly_xi_plus.append(epsilon_xi_plus)
            r_xi_plus.append(parameter_xi_plus)

        p_q = step_polynomials[0][0]
        q_min = (p_q[0].abs_lower()
                 - sum((z.abs_upper() for z in p_q.coeffs()[1:]), arb(0))
                 - epsilon).lower()
        require(q_min > 0,
                f"Propagation nonvanishing failed at r_*={r_star}")
        minimum_modulus = (q_min if minimum_modulus is None
                           else min(minimum_modulus, q_min))

        # Convert the current state endpoint into initial data for next step.
        endpoint_data = [
            [poly(1) for poly in pair] for pair in step_polynomials]
        epsilon_poly = [
            (epsilon_poly_plus+pair_radius(endpoint_data[0])).upper()]
        epsilon_poly += [
            (epsilon_xi+pair_radius(pair)).upper()
            for epsilon_xi, pair
            in zip(epsilon_poly_xi_plus, endpoint_data[1:])]
        parameter_spread = [r_Q_plus]+r_xi_plus
        delta_bd = delta_bd_plus
        polynomial_columns = [
            [z.mid() for z in pair] for pair in endpoint_data]
        require(epsilon_poly[0]+r_Q_plus+delta_bd < epsilon,
                "Endpoint recentering exceeded tube")
        r_star += h

    center_enclosures = [
        [inflate(z, error) for z in pair]
        for pair, error in zip(polynomial_columns, epsilon_poly)]
    family_enclosures = [
        [inflate(z, (error+difference).upper()) for z in pair]
        for pair, error, difference
        in zip(polynomial_columns, epsilon_poly, parameter_spread)]
    return {"center": center_enclosures, "family": family_enclosures,
            "boundary_error": delta_bd,
            "record": {
                "center_error": encode(epsilon_poly[0]),
                "parameter_spread": encode(parameter_spread[0]),
                "boundary_error": encode(delta_bd),
                "minimum_modulus": encode(minimum_modulus),
            }}
