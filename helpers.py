"""Data and small Arb constructions shared by the rest of the codebase.
"""
from dataclasses import dataclass

from flint import arb, acb


# Retain enough decimal digits to save the high-precision Arb enclosure.
ARB_JSON_DIGITS = 300


class CertificationFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CertificationFailure(message)

# Stores the equation parameters and run settings for each dimension.
@dataclass(frozen=True)
class ProfileSpec:
    """Equation parameters and validated run defaults for one dimension."""
    p: int
    omega_bound: str       # Bound |Omega| \leq Omega_bd in the manuscript.
    rho_bits: int          # Parameter-cube radius is rho = 2^(-rho_bits).
    r_m: int              # Matching radius where the origin and exterior shots meet.
    inner_order: int       # Truncation order of the regular-origin expansion.
    outer_order: int       # Truncation order of the pure-slow tail expansion.
    inner_degree: int      # Taylor degree for propagation from the origin.
    outer_degree: int      # Taylor degree for propagation from the exterior.

    @property
    def similarity_exp(self):
        return arb(1)/(self.p-1)

    @property
    def conjugate_exp(self):
        return (self.p-1)//2

    @property
    def omega_limit(self):
        return arb(self.omega_bound)

# These parameters have been tuned through trial and error
PROFILES = {
    3: ProfileSpec(27, "7/5", 240, 16, 48, 128, 112, 64),
    4: ProfileSpec(9, "59/20", 320, 8, 32, 224, 88, 88),
    5: ProfileSpec(7, "7/4", 220, 12, 48, 128, 112, 64),
}


def profile_spec(d):
    """Return the supported equation and run defaults for dimension ``d``."""
    return PROFILES[d]


def norm(xs):
    """Upper-bound the Euclidean norm."""
    return sum((x.abs_upper()**2 for x in xs), arb(0)).sqrt().upper()


def pair_radius(xs):
    """Bound the enclosure radius of a pair (Q,Q')."""
    return norm([z.rad() for z in xs])


def inflate(z, radius):
    """A rectangular superset of the complex disk of radius `radius`."""
    return z + acb(arb(0, radius), arb(0, radius))


def polynomial_norm(poly):
    """Uniform complex modulus on 0 <= u <= 1."""
    return sum((z.abs_upper() for z in poly.coeffs()), arb(0)).upper()


def conjugate_poly(poly):
    return type(poly)([z.conjugate() for z in poly.coeffs()])


def matrix_infinity_norm(M):
    """Bound the infinity norm by upward-rounded row sums."""
    return max(sum((M[i, j].abs_upper() for j in range(M.ncols())),
                   arb(0)).upper()
               for i in range(M.nrows()))


def encode(value):
    """Store a real Arb ball as an outward decimal enclosure."""
    if not isinstance(value, arb):
        raise TypeError(f"expected arb, got {type(value).__name__}")
    return value.str(ARB_JSON_DIGITS)


class Jet:
    """A value and derivatives in the three real parameters (A, Omega, K).

    The recurrence algebra in :mod:`origin` and :mod:`exterior` acts on jets
    just as on complex values. Product and quotient rules propagate the
    derivatives needed by the finite matching map.
    """
    def __init__(self, value=0, derivs=None):
        if isinstance(value, Jet):
            self.v, self.derivs = value.v, value.derivs
        else:
            self.v = acb(value)
            self.derivs = tuple(acb(x) for x in (derivs or (0, 0, 0)))

    def __add__(self, other):
        other = Jet(other)
        return Jet(self.v + other.v,
                   [a+b for a, b in zip(self.derivs, other.derivs)])
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, [-a for a in self.derivs])

    def __sub__(self, other):
        return self + -Jet(other)

    def __mul__(self, other):
        other = Jet(other)
        return Jet(self.v * other.v,
                   [a*other.v + self.v*b
                    for a, b in zip(self.derivs, other.derivs)])
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Jet(other)
        return Jet(self.v / other.v,
                   [(a*other.v-self.v*b)/other.v**2
                    for a, b in zip(self.derivs, other.derivs)])

    def __pow__(self, n):
        require(isinstance(n, int) and n >= 0, 
                "Only nonnegative integer Jet powers")
        if n == 0:
            return Jet(1)
        return Jet(self.v**n, [n*self.v**(n-1)*deriv
                               for deriv in self.derivs])

    def conjugate(self):
        return Jet(self.v.conjugate(),
                   [deriv.conjugate() for deriv in self.derivs])

    def real(self):
        return Jet(self.v.real, [deriv.real for deriv in self.derivs])

    def exp(self):
        e = self.v.exp()
        return Jet(e, [e*deriv for deriv in self.derivs])

    def log(self):
        return Jet(self.v.log(), [deriv/self.v for deriv in self.derivs])


def param_jets(params):
    # Lift (A, Omega, K) to jets whose derivatives are the standard basis.
    return [Jet(value, [int(i == j) for j in range(3)])
            for i, value in enumerate(params)]


def unpack_pair(pair):
    # Separate each jet into one row for (Q,Q') and three rows for
    # (\partial_A Q, \partial_A Q'), 
    # (\partial_\Omega Q, \partial_\Omega Q'),
    # (\partial_K Q, \partial_K Q').
    return ([[z.v for z in pair]]
            + [[z.derivs[j] for z in pair] for j in range(3)])
