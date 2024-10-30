import numpy as np
from numpy import inf

from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution,
    _RealDomain,
    _RealParameter,
    _Parameterization,
)


class Uniform(ContinuousDistribution):
    r"""Uniform distribution.

    The probability density function of the uniform distribution is:

    .. math::

        f(x; a, b) = \frac{1}
                          {b - a}

    """

    _a_domain = _RealDomain(endpoints=(-inf, inf))
    _b_domain = _RealDomain(endpoints=("a", inf))
    _x_support = _RealDomain(endpoints=("a", "b"), inclusive=(False, False))

    _a_param = _RealParameter("a", domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter("b", domain=_b_domain, typical=(1.1, 1e3))
    _x_param = _RealParameter("x", domain=_x_support, typical=("a", "b"))

    _b_domain.define_parameters(_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=None, b=None, **kwargs):
        super().__init__(a=a, b=b, **kwargs)

    def _process_parameters(self, a=None, b=None, ab=None, **kwargs):
        ab = b - a
        kwargs.update(dict(a=a, b=b, ab=ab))
        return kwargs

    def _pdf_formula(self, x, *, ab, **kwargs):
        return np.full(x.shape, 1 / ab)

    def _icdf_formula(self, x, a, b, ab, **kwargs):
        return a + ab * x

    def _mode_formula(self, *, a, b, ab, **kwargs):
        return a + 0.5 * ab
