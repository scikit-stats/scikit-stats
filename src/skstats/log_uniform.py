import numpy as np
from numpy import inf

from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution,
    _RealDomain,
    _RealParameter,
    _Parameterization,
)


def _log_diff(log_p, log_q):
    return special.logsumexp([log_p, log_q + np.pi * 1j], axis=0)


class _LogUniform(ContinuousDistribution):
    r"""Log-uniform distribution.

    The probability density function of the log-uniform distribution is:

    .. math::

        f(x; a, b) = \frac{1}
                          {x (\log(b) - \log(a))}

    If :math:`\log(X)` is a random variable that follows a uniform distribution
    between :math:`\log(a)` and :math:`\log(b)`, then :math:`X` is log-uniformly
    distributed with shape parameters :math:`a` and :math:`b`.

    """

    _a_domain = _RealDomain(endpoints=(0, inf))
    _b_domain = _RealDomain(endpoints=("a", inf))
    _log_a_domain = _RealDomain(endpoints=(-inf, inf))
    _log_b_domain = _RealDomain(endpoints=("log_a", inf))
    _x_support = _RealDomain(endpoints=("a", "b"), inclusive=(True, True))

    _a_param = _RealParameter("a", domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter("b", domain=_b_domain, typical=(1.1, 1e3))
    _log_a_param = _RealParameter(
        "log_a", symbol=r"\log(a)", domain=_log_a_domain, typical=(-3, -0.1)
    )
    _log_b_param = _RealParameter(
        "log_b", symbol=r"\log(b)", domain=_log_b_domain, typical=(0.1, 3)
    )
    _x_param = _RealParameter("x", domain=_x_support, typical=("a", "b"))

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [
        _Parameterization(_log_a_param, _log_b_param),
        _Parameterization(_a_param, _b_param),
    ]
    _variable = _x_param

    def __init__(self, *, a=None, b=None, log_a=None, log_b=None, **kwargs):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b, **kwargs)

    def _process_parameters(self, a=None, b=None, log_a=None, log_b=None, **kwargs):
        a = np.exp(log_a) if a is None else a
        b = np.exp(log_b) if b is None else b
        log_a = np.log(a) if log_a is None else log_a
        log_b = np.log(b) if log_b is None else log_b
        kwargs.update(dict(a=a, b=b, log_a=log_a, log_b=log_b))
        return kwargs

    # def _logpdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return -np.log(x) - np.log(log_b - log_a)

    def _pdf_formula(self, x, *, log_a, log_b, **kwargs):
        return ((log_b - log_a) * x) ** -1

    # def _cdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)

    def _moment_raw_formula(self, order, log_a, log_b, **kwargs):
        if order == 0:
            return self._one
        t1 = self._one / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2
