"""Regular-origin recurrence and enclosure.

The recurrence computes a polynomial P_in and its parameter derivatives.  A
bound on its full ODE residual then certifies existence of nearby exact 
solution with error bounds at r_in. The returned error is used in the outward
rigorous shooting.
"""
from flint import arb, acb, acb_poly

from helpers import (Jet, param_jets, norm, require, polynomial_norm,
                     conjugate_poly, encode, unpack_pair, profile_spec)

I = acb(0, 1)


def origin_coeffs(params, series_order, d):
    """Solve the regular even-power recurrence coefficient by coefficient.
    Uses the recurrence in Section 3.2 of the manuscript.
    Jets carry the three real parameter derivatives.
    """
    spec = profile_spec(d)
    similarity_exp = spec.similarity_exp
    p, conjugate_exp = spec.p, spec.conjugate_exp
    A, omega, _ = param_jets(params)
    # Normalize by keeping c_0 = 1. Multiply by A later.
    c_coeffs = [Jet(1)]
    modulus_squared_coeffs = []
    modulus_power_coeffs = []
    for n in range(series_order):
        modulus_squared_coeffs.append(
            sum(c_coeffs[j]*c_coeffs[n-j].conjugate()
                for j in range(n+1)).real())
        if n == 0:
            modulus_power_coeffs.append(
                modulus_squared_coeffs[0]**conjugate_exp)
        else:
            modulus_power_coeffs.append(
                sum(((conjugate_exp+1)*j-n)*modulus_squared_coeffs[j]
                    * modulus_power_coeffs[n-j]
                    for j in range(1, n+1))
                / (n*modulus_squared_coeffs[0]))
        nonlinearity_coeff = sum(
            modulus_power_coeffs[j]*c_coeffs[n-j]
            for j in range(n+1))
        c_coeffs.append(
            (A**(p-1)*nonlinearity_coeff
             -(omega+I*(similarity_exp+n))*c_coeffs[n])
            / (2*(n+1)*(2*n+d)))
    return [A*z for z in c_coeffs]


def origin_data(params, r_in, series_order, d):
    """Evaluate the truncated regular-origin (P_in, P_in') and variations."""
    c_coeffs = origin_coeffs(params, series_order, d)
    q = sum(c_coeffs[n]*r_in**(2*n) for n in range(series_order+1))
    q_prime = sum(2*n*c_coeffs[n]*r_in**(2*n-1)
                  for n in range(1, series_order+1))
    return unpack_pair([q, q_prime])


def certify_origin(param_box, r_in, series_order, d):
    """Checks the hypotheses of Lemma 3.7 in the manuscript and returns
    rigorous error bounds.
    """
    spec = profile_spec(d)
    similarity_exp = spec.similarity_exp
    p, conjugate_exp = spec.p, spec.conjugate_exp
    c_coeffs = origin_coeffs(param_box, series_order, d)
    P_in = acb_poly([
        c_coeffs[n].v*r_in**(2*n) for n in range(series_order+1)])
    nonlinear_poly = (
        P_in**(conjugate_exp+1)*conjugate_poly(P_in)**conjugate_exp)
    omega = param_box[1]
    # The recurrence should cancel every residual coefficient below `order`.
    # Arb's contains(0) verifies this over the entire parameter box.
    for n in range(series_order):
        check = (2*(n+1)*(2*n+d)*c_coeffs[n+1].v*r_in**(2*n)
                 +(omega+I*(similarity_exp+n))*P_in[n]
                 -nonlinear_poly[n])
        require(check.real.contains(0) and check.imag.contains(0),
                "Origin recurrence consistency failed")
    H_0 = (
        ((omega+I*(similarity_exp+series_order))*P_in[series_order]
         - nonlinear_poly[series_order]).abs_upper()
        + sum((nonlinear_poly[n].abs_upper()
               for n in range(series_order+1, nonlinear_poly.length())),
              arb(0))).upper()
    amplitude_cap = arb(1)/2
    lipschitz = (acb(omega, similarity_exp).abs_upper()
                 + p*amplitude_cap**(p-1)).upper()
    contraction = (r_in**2*lipschitz/(2*d)).upper()
    require(contraction < 1, "Origin contraction failed")
    delta_in = (2*r_in**2*H_0/(2*d*(1-contraction))).upper()
    require(polynomial_norm(P_in)+delta_in < amplitude_cap,
            "Origin amplitude ball failed")
    require(P_in[0].abs_lower()
            - sum((z.abs_upper() for z in P_in.coeffs()[1:]), arb(0))-delta_in > 0,
            "Origin nonvanishing failed")
    inclusion = (r_in**2*H_0/(2*d)+contraction*delta_in).upper()
    require(inclusion <= delta_in, "Origin ball inclusion failed")
    q_prime_error = (r_in*(H_0+lipschitz*delta_in)/d).upper()
    state_error = norm([delta_in, q_prime_error])
    return state_error, {
        "contraction": encode(contraction),
        "inclusion_ratio": encode((inclusion/delta_in).upper()),
        "boundary_error": encode(state_error),
    }
