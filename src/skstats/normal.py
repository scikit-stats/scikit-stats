import numpy as np
from numpy import inf

from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution,
    _RealDomain,
    _RealParameter,
    _Parameterization,
)


__all__ = ["Normal", "StandardNormal"]


class Normal(ContinuousDistribution):
    r"""Normal distribution with prescribed mean and standard deviation.

    The probability density function of the normal distribution is:

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp {
            \left( -\frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^2 \right)}

    """

    # `ShiftedScaledDistribution` allows this to be generated automatically from
    # an instance of `StandardNormal`, but the normal distribution is so frequently
    # used that it's worth a bit of code duplication to get better performance.
    _mu_domain = _RealDomain(endpoints=(-inf, inf))
    _sigma_domain = _RealDomain(endpoints=(0, inf))
    _x_support = _RealDomain(endpoints=(-inf, inf))

    _mu_param = _RealParameter("mu", symbol=r"\mu", domain=_mu_domain, typical=(-1, 1))
    _sigma_param = _RealParameter(
        "sigma", symbol=r"\sigma", domain=_sigma_domain, typical=(0.5, 1.5)
    )
    _x_param = _RealParameter("x", domain=_x_support, typical=(-1, 1))

    _parameterizations = [_Parameterization(_mu_param, _sigma_param)]

    _variable = _x_param
    _normalization = 1 / np.sqrt(2 * np.pi)
    _log_normalization = np.log(2 * np.pi) / 2

    def __new__(cls, mu=None, sigma=None, **kwargs):
        if mu is None and sigma is None:
            return super().__new__(StandardNormal)
        return super().__new__(cls)

    def __init__(self, *, mu=0.0, sigma=1.0, **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def _logpdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logpdf_formula(self, (x - mu) / sigma) - np.log(sigma)

    def _pdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._pdf_formula(self, (x - mu) / sigma) / sigma

    def _logcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logcdf_formula(self, (x - mu) / sigma)

    def _cdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._cdf_formula(self, (x - mu) / sigma)

    def _logccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logccdf_formula(self, (x - mu) / sigma)

    def _ccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ccdf_formula(self, (x - mu) / sigma)

    def _icdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._icdf_formula(self, x) * sigma + mu

    def _ilogcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogcdf_formula(self, x) * sigma + mu

    def _iccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._iccdf_formula(self, x) * sigma + mu

    def _ilogccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogccdf_formula(self, x) * sigma + mu

    def _entropy_formula(self, *, mu, sigma, **kwargs):
        return StandardNormal._entropy_formula(self) + np.log(abs(sigma))

    def _logentropy_formula(self, *, mu, sigma, **kwargs):
        lH0 = StandardNormal._logentropy_formula(self)
        lls = np.log(np.log(abs(sigma)) + 0j)
        return special.logsumexp(np.broadcast_arrays(lH0, lls), axis=0)

    def _median_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _mode_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _moment_raw_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order == 1:
            return mu
        else:
            return None

    _moment_raw_formula.orders = [0, 1]  # type: ignore[attr-defined]

    def _moment_central_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order % 2:
            return np.zeros_like(mu)
        else:
            # exact is faster (and obviously more accurate) for reasonable orders
            return sigma**order * special.factorial2(int(order) - 1, exact=True)

    def _sample_formula(self, sample_shape, full_shape, rng, *, mu, sigma, **kwargs):
        return rng.normal(loc=mu, scale=sigma, size=full_shape)[()]


class StandardNormal(Normal):
    r"""Standard normal distribution.

    The probability density function of the standard normal distribution is:

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{1}{2} x^2 \right)

    """

    _x_support = _RealDomain(endpoints=(-inf, inf))
    _x_param = _RealParameter("x", domain=_x_support, typical=(-5, 5))
    _variable = _x_param
    _parameterizations = []
    _normalization = 1 / np.sqrt(2 * np.pi)
    _log_normalization = np.log(2 * np.pi) / 2
    mu = np.float64(0.0)
    sigma = np.float64(1.0)

    def __init__(self, **kwargs):
        ContinuousDistribution.__init__(self, **kwargs)

    def _logpdf_formula(self, x, **kwargs):
        return -(self._log_normalization + x**2 / 2)

    def _pdf_formula(self, x, **kwargs):
        return self._normalization * np.exp(-(x**2) / 2)

    def _logcdf_formula(self, x, **kwargs):
        return special.log_ndtr(x)

    def _cdf_formula(self, x, **kwargs):
        return special.ndtr(x)

    def _logccdf_formula(self, x, **kwargs):
        return special.log_ndtr(-x)

    def _ccdf_formula(self, x, **kwargs):
        return special.ndtr(-x)

    def _icdf_formula(self, x, **kwargs):
        return special.ndtri(x)

    def _ilogcdf_formula(self, x, **kwargs):
        return special.ndtri_exp(x)

    def _iccdf_formula(self, x, **kwargs):
        return -special.ndtri(x)

    def _ilogccdf_formula(self, x, **kwargs):
        return -special.ndtri_exp(x)

    def _entropy_formula(self, **kwargs):
        return (1 + np.log(2 * np.pi)) / 2

    def _logentropy_formula(self, **kwargs):
        return np.log1p(np.log(2 * np.pi)) - np.log(2)

    def _median_formula(self, **kwargs):
        return 0

    def _mode_formula(self, **kwargs):
        return 0

    def _moment_raw_formula(self, order, **kwargs):
        raw_moments = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 0}
        return raw_moments.get(order, None)

    def _moment_central_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _moment_standardized_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _sample_formula(self, sample_shape, full_shape, rng, **kwargs):
        return rng.normal(size=full_shape)[()]
