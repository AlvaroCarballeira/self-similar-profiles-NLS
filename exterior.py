"""Pure-slow exterior expansion, linear-mode bounds, and exact enclosure.

The recurrence gives data at s_out.  The residual bound and contraction enclose
an exact solution with no fast mode. Its recorded error starts the inward
rigorous shooting.
"""
from flint import arb, acb, acb_poly

from helpers import (Jet, param_jets, norm, require, encode, unpack_pair,
                     profile_spec)

I = acb(0, 1)


def certify_mode_bounds(d, s_out, omega_interval):
    """Bound the exterior linear modes uniformly over the frequency box
    This is an implementaion of the estimates in Lemma 3.3 of the manuscript.
    """
    spec = profile_spec(d)
    similarity_exp, omega_limit = spec.similarity_exp, spec.omega_limit
    require(s_out >= 32 and omega_interval.abs_upper() <= omega_limit,
            "Exterior mode parameter range")
    bounds = []
    for x in (similarity_exp, arb(d)/2-similarity_exp):
        product = arb(1)
        for n in range(16):
            product *= 1+omega_limit**2/(x+n)**2
        t = x+16
        ratio = (product.sqrt()
                 * (omega_limit**2*(1/t**2+1/t)/2).exp()).upper()
        # The differentiated power is d/2-2-z; for d=5 its slow
        # real part is positive. Include this growth in the damping.
        loss = (omega_limit
                + max(arb(0), (arb(d)/2-2-x).upper())).upper()
        damping = 1-4*loss/32**2
        require(damping > 0, "Gamma integral damping must be positive")
        length = ((arb(d)/2-1-x)**2+omega_limit**2).sqrt()
        bounds.append((4*length*x*ratio*damping**(-x-1)).upper())
    D_s, D_f = bounds
    B_s = (2*(similarity_exp**2+omega_limit**2).sqrt()).upper()
    B_f = (2*((arb(d)/2-similarity_exp)**2+omega_limit**2).sqrt()).upper()
    # This is the constant A_{s_out} in the manuscript
    mode_bound = max(
        (1+D_s/s_out**2).upper(), (1+D_f/s_out**2).upper(),
        (B_s*(1+D_s/s_out**2)+2*D_s/s_out**2).upper(),
        ((arb(1)/2+B_f/s_out**2)*(1+D_f/s_out**2)
         + 2*D_f/s_out**4).upper())
    return mode_bound


def tail_coeffs(params, series_order, d):
    """Compute the recurrence l_n, f_n, E_n in Section 3.1 in the manuscript.
    Jets carry derivatives in (A, Omega, K) through the same recurrence,
    providing Cauchy-data variations for the finite matching map.
    """
    spec = profile_spec(d)
    similarity_exp, p = spec.similarity_exp, spec.p
    _, omega, K = param_jets(params)
    beta = -2*similarity_exp+2*I*omega
    l, f, E = [Jet(0)], [beta], [Jet(1)]
    for n in range(series_order):
        if n:
            E.append(sum((p-1)*j*l[j].real()*E[n-j] for j in range(1, n+1))/n)
        H = sum(f[j]*f[n-j] for j in range(n+1))+(d-2-2*n)*f[n]
        l.append(I*(K*E[n]-H)/(n+1))
        f.append(-2*(n+1)*l[-1])
    return l, f


def tail_data(params, s_out, series_order, d):
    """Evaluate the truncated pure-slow (P_out, P_out') at the outer radius s_out."""
    similarity_exp = profile_spec(d).similarity_exp
    l, f = tail_coeffs(params, series_order, d)
    _, omega, K = param_jets(params)
    beta = -2*similarity_exp+2*I*omega
    log_q = (similarity_exp*K.log()+beta*s_out.log()
             + sum(l[n]*s_out**(-2*n)
                   for n in range(1, series_order+1)))
    q = log_q.exp()
    scaled_log_deriv = sum(
        f[n]*s_out**(-2*n) for n in range(series_order+1))
    return unpack_pair([q, q*scaled_log_deriv/s_out])


def certify_exterior(param_box, s_out, series_order, d):
    """Enclose exact pure-slow Cauchy data near the finite expansion.
    This is an implementation of the estimates in Lemma 3.2 and Proposition 3.4
    in the manuscript.
    """
    spec = profile_spec(d)
    a, p = spec.similarity_exp, spec.p  # a = 1/(p-1)
    J = series_order
    A_s_out = certify_mode_bounds(d, s_out, param_box[1])
    l, f = tail_coeffs(param_box, J, d)
    K = param_box[2]
    require(K > 0, "Exterior construction requires K > 0")

    # Lemma 3.2: form L_n = l_n s_out^(-2n), F, and G, then compute
    # the constant B.
    L = [l[n].v*s_out**(-2*n) for n in range(J+1)]
    F = acb_poly([f[n].v*s_out**(-2*n) for n in range(J+1)])
    u = acb_poly([0, 1])
    G = F*F+(d-2)*F-2*u*F.derivative()
    sum_abs_Re_L = sum(
        (L[n].real.abs_upper() for n in range(1, J+1)), arb(0)).upper()
    exponent_coefficients = [arb(0)] + [
        ((p-1)*L[n].real.abs_upper()).upper()
        for n in range(1, J+1)]
    # For chi>1, the majorant in the lemma has the term 
    # chi^(-J) exp((p-1) sum_n |Re L_n| chi^n). The loop chooses the best
    # bound from a finite grid of chi.
    exponential_tail = None
    for j in range(4, 65):
        chi = arb(j)/2
        exponent = sum(
            (exponent_coefficients[n]*chi**n for n in range(1, J+1)),
            arb(0)).upper()
        if exponent > 10000:
            continue
        candidate = (exponent.exp()/chi**J).upper()
        if exponential_tail is None or candidate < exponential_tail:
            exponential_tail = candidate
    require(exponential_tail is not None, "No finite exponential majorant")
    polynomial_tail = sum(
        (G[n].abs_upper() for n in range(J, G.length())),
        arb(0)).upper()
    B = (polynomial_tail+K*exponential_tail).upper()

    # Proposition 3,4: construct inverse constant C_nu_s_out = mathfrak C_{nu,s_out}
    # and bound M.
    nu = 2*a+2*J+2
    C_nu_s_out = (2*A_s_out**2*(1/(nu-2*a)+1/(nu+2*a-d))).upper()
    M = (K**a*sum_abs_Re_L.exp()).upper()
    require(sum_abs_Re_L < 1, "Exterior lower amplitude estimate")
    # This equals K^(2/(p-1))/M, the lower bound in the third test.
    lower_amplitude = (K**a*(-sum_abs_Re_L).exp()).lower()
    image_at_zero = (C_nu_s_out*M*B/s_out**2).upper()

    # Choose epsilon_ext, then check contraction and invariance of the
    # correction ball (the first two proposition hypothesis).
    preliminary_contraction = (p*C_nu_s_out*M**(p-1)/s_out**2).upper()
    require(preliminary_contraction < 1,
            "Exterior preliminary contraction failed")
    epsilon_ext = (2*image_at_zero/(1-preliminary_contraction)).upper()
    contraction = (p*C_nu_s_out*(M+epsilon_ext)**(p-1)/s_out**2).upper()
    inclusion = image_at_zero+contraction*epsilon_ext
    require(contraction < 1 and inclusion <= epsilon_ext,
            "Exterior contraction/inclusion failed")

    # Third proposition hypothesis
    require(epsilon_ext < lower_amplitude, "Exterior nonvanishing failed")

    Q_error_at_s_out = (epsilon_ext*s_out**(-2*a)).upper()
    Q_prime_error_at_s_out = (s_out*Q_error_at_s_out).upper()
    state_error = norm([Q_error_at_s_out, Q_prime_error_at_s_out])
    return state_error, {
        "contraction": encode(contraction),
        "inclusion_ratio": encode((inclusion/epsilon_ext).upper()),
        "boundary_error": encode(state_error),
    }
