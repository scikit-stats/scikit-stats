import numpy as np
import mpmath
from mpmath import mp


class ReferenceDistribution:
    """Minimalist distribution infrastructure for generating reference data.

    The purpose is to generate reference values for unit tests of SciPy
    distribution accuracy and robustness.

    Handles array input with standard broadcasting rules, and method
    implementations are easily compared against their mathematical definitions.
    No attempt is made to handle edge cases or be fast, and arbitrary precision
    arithmetic is trusted for accuracy rather than making the method
    implementations "smart".

    Notes
    -----

    In this infrastructure, distributions families are classes, and
    fully-specified distributions (i.e. with definite values of all family
    parameters) are instances of these classes. Typically, the public methods
    accept as input only the argument at which the at which the function is to
    be evaluated. Unlike SciPy distributions, they never accept values of
    distribution family shape, location, or scale parameters. A few
    other parameters are noteworthy:

    - All methods accept `dtype` to control the output data type. The default
      is `np.float64`, but `object` or `mp.mpf` may be
      specified to output the full `mpf`.
    - `ppf`/`isf` accept a `guess` because they use a scalar rootfinder
      to invert the `cdf`/`sf`. This is passed directly into the `x0` method
      of `mpmath.findroot`; see its documentation for details.
    - moment accepts `order`, an integer that specifies the order of the (raw)
      moment, and `center`, which is the value about which the moment is
      taken. The default is to calculate the mean and use it to calculate
      central moments; passing ``0`` results in a noncentral moment. For
      efficiency, the mean can be passed explicitly if it is already known.

    Follow the example of SkewNormal to generate new reference distributions,
    overriding only `__init__` and `_pdf`*. Use the reference distributions to
    generate reference values for unit tests of SciPy distribution method
    precision and robustness (e.g. for extreme arguments). If the a SciPy
    methods implementation is independent and yet the output matches reference
    values generated with this infrastructure, it is unlikely that the SciPy
    and reference values are both inaccurate.

    * If the SciPy output *doesn't* match and the cause appears to be
    inaccuracy of the reference values (e.g. due to numerical issues that
    mpmath's arbitrary precision arithmetic doesn't handle), then it may be
    appropriate to override a method of the reference distribution rather than
    relying on the generic implementation. Otherwise, hesitate to override
    methods: the generic implementations are mathematically correct and easy
    to verify, whereas an override introduces many possibilities of mistakes,
    requires more time to write, and requires more time to review.

    In general, do not create custom unit tests to ensure that
    SciPy distribution methods are *correct* (in the sense of being consistent
    with the rest of the distribution methods); generic tests take care of
    that.
    """

    def __init__(self, **kwargs):
        try:
            if mpmath.dps is not None:
                message = (
                    "`mpmath.dps` has been assigned. This is not "
                    "intended usage; instead, assign the desired "
                    "precision to `mpmath.mp.dps` (e.g. `from mpmath "
                    "as mp; mp.dps = 50."
                )
                raise RuntimeError(message)
        except AttributeError:
            mpmath.dps = None

        if mp.dps <= 15:
            message = (
                "`mpmath.mp.dps <= 15`. Set a higher precision (e.g."
                "`50`) to use this distribution."
            )
            raise RuntimeError(message)

        self._params = {key: self._make_mpf_array(val) for key, val in kwargs.items()}

    def _make_mpf_array(self, x):
        shape = np.shape(x)
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.asarray([mp.mpf(xi) for xi in x]).reshape(shape)[()]

    def _pdf(self, x):
        raise NotImplementedError("_pdf must be overridden.")

    def _cdf(self, x, **kwargs):
        if (self._cdf.__func__ is ReferenceDistribution._cdf) and (
            self._sf.__func__ is not ReferenceDistribution._sf
        ):
            return mp.one - self._sf(x, **kwargs)

        a, b = self._support(**kwargs)
        res = mp.quad(lambda x: self._pdf(x, **kwargs), (a, x))
        res = res if res < 0.5 else mp.one - self._sf(x, **kwargs)
        return res

    def _sf(self, x, **kwargs):
        if (self._sf.__func__ is ReferenceDistribution._sf) and (
            self._cdf.__func__ is not ReferenceDistribution._cdf
        ):
            return mp.one - self._cdf(x, **kwargs)

        a, b = self._support(**kwargs)
        res = mp.quad(lambda x: self._pdf(x, **kwargs), (x, b))
        res = res if res < 0.5 else mp.one - self._cdf(x, **kwargs)
        return res

    def _ppf(self, p, guess=0, **kwargs):
        if (self._ppf.__func__ is ReferenceDistribution._ppf) and (
            self._isf.__func__ is not ReferenceDistribution._isf
        ):
            return self._isf(mp.one - p, guess, **kwargs)

        def f(x):
            return self._cdf(x, **kwargs) - p

        return mp.findroot(f, guess)

    def _isf(self, p, guess=0, **kwargs):
        if (self._isf.__func__ is ReferenceDistribution._isf) and (
            self._ppf.__func__ is not ReferenceDistribution._ppf
        ):
            return self._ppf(mp.one - p, guess, **kwargs)

        def f(x):
            return self._sf(x, **kwargs) - p

        return mp.findroot(f, guess)

    def _logpdf(self, x, **kwargs):
        return mp.log(self._pdf(x, **kwargs))

    def _logcdf(self, x, **kwargs):
        return mp.log(self._cdf(x, **kwargs))

    def _logsf(self, x, **kwargs):
        return mp.log(self._sf(x, **kwargs))

    def _support(self, **kwargs):
        return -mp.inf, mp.inf

    def _entropy(self, **kwargs):
        def integrand(x):
            logpdf = self._logpdf(x, **kwargs)
            pdf = mp.exp(logpdf)
            return -pdf * logpdf

        a, b = self._support(**kwargs)
        return mp.quad(integrand, (a, b))

    def _mean(self, **kwargs):
        return self._moment(order=1, center=0, **kwargs)

    def _var(self, **kwargs):
        mu = self._mean(**kwargs)
        return self._moment(order=2, center=mu, **kwargs)

    def _skew(self, **kwargs):
        mu = self._mean(**kwargs)
        u2 = self._moment(order=2, center=mu, **kwargs)
        sigma = mp.sqrt(u2)
        u3 = self._moment(order=3, center=mu, **kwargs)
        return u3 / sigma**3

    def _kurtosis(self, **kwargs):
        mu = self._mean(**kwargs)
        u2 = self._moment(order=2, center=mu, **kwargs)
        u4 = self._moment(order=4, center=mu, **kwargs)
        return u4 / u2**2 - 3

    def _moment(self, order, center, **kwargs):
        def integrand(x):
            return self._pdf(x, **kwargs) * (x - center) ** order

        if center is None:
            center = self._mean(**kwargs)

        a, b = self._support(**kwargs)
        return mp.quad(integrand, (a, b))

    def pdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._pdf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def cdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._cdf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def sf(self, x, dtype=np.float64):
        fun = np.vectorize(self._sf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def ppf(self, x, guess=0, dtype=np.float64):
        fun = np.vectorize(self._ppf, excluded={1})  # don't vectorize guess
        x = self._make_mpf_array(x)
        res = fun(x, guess, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def isf(self, x, guess=0, dtype=np.float64):
        fun = np.vectorize(self._isf, excluded={1})  # don't vectorize guess
        x = self._make_mpf_array(x)
        res = fun(x, guess, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def logpdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._logpdf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def logcdf(self, x, dtype=np.float64):
        fun = np.vectorize(self._logcdf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def logsf(self, x, dtype=np.float64):
        fun = np.vectorize(self._logsf)
        x = self._make_mpf_array(x)
        res = fun(x, **self._params)
        return np.asarray(res, dtype=dtype)[()]

    def support(self, dtype=np.float64):
        fun = np.vectorize(self._support)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def entropy(self, dtype=np.float64):
        fun = np.vectorize(self._entropy)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def mean(self, dtype=np.float64):
        fun = np.vectorize(self._mean)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def var(self, dtype=np.float64):
        fun = np.vectorize(self._var)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def skew(self, dtype=np.float64):
        fun = np.vectorize(self._skew)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def kurtosis(self, dtype=np.float64):
        fun = np.vectorize(self._kurtosis)
        res = fun(**self._params)
        return np.asarray(res, dtype=dtype)[()]

    def moment(self, order, center=None, dtype=np.float64):
        fun = np.vectorize(self._moment)
        order = self._make_mpf_array(order)
        res = fun(order, **self._params)
        return np.asarray(res, dtype=dtype)[()]


class LogNormal(ReferenceDistribution):
    def __init__(self, *, s):
        super().__init__(s=s)

    def _support(self, s):
        return 0, mp.inf

    def _pdf(self, x, s):
        return (
            mp.one
            / (s * x * mp.sqrt(2 * mp.pi))
            * mp.exp(-mp.one / 2 * (mp.log(x) / s) ** 2)
        )

    def _cdf(self, x, s):
        return mp.ncdf(mp.log(x) / s)


class Normal(ReferenceDistribution):
    def _pdf(self, x):
        return mp.npdf(x)