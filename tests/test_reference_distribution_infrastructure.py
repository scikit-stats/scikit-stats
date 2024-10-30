import mpmath
from mpmath import mp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import skstats

import tests.reference_distributions as rd


mp.dps = 20  # high enough to pass, not unreasonably slow


def test_precision():
    mp.dps = 8

    message = "`mpmath.mp.dps <= 15`. Set a higher precision..."
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()

    mpmath.dps = 20
    message = "`mpmath.dps` has been assigned. This is not intended usage..."
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()
    del mpmath.dps

    mp.dps = 20


def test_complementary_method_use():
    # Show that complementary methods are used as expected.
    # E.g., use 1 - CDF to compute SF if CDF is overridden but SF is not

    mp.dps = 50
    x = np.linspace(0, 1, 10)

    class MyDist(rd.ReferenceDistribution):
        def _cdf(self, x):
            return x

    dist = MyDist()
    assert_allclose(dist.sf(x), 1 - dist.cdf(x))

    class MyDist(rd.ReferenceDistribution):
        def _sf(self, x):
            return 1 - x

    dist = MyDist()
    assert_allclose(dist.cdf(x), 1 - dist.sf(x))

    class MyDist(rd.ReferenceDistribution):
        def _ppf(self, x, guess):
            return x

    dist = MyDist()
    assert_allclose(dist.isf(x), dist.ppf(1 - x))

    class MyDist(rd.ReferenceDistribution):
        def _isf(self, x, guess):
            return 1 - x

    dist = MyDist()
    assert_allclose(dist.ppf(x), dist.isf(1 - x))


@pytest.mark.parametrize("dist, dist_ref", ((skstats.Normal(), rd.Normal()),))
def test_mpmath_vs_skstats(dist, dist_ref):
    # Basic tests of the mpmath distribution infrastructure using a skstats
    # distribution as a reference. The intent is just to make sure that the
    # implementations do not have *mistakes* and that broadcasting is working
    # as expected. The accuracy is what it is.

    rng = np.random.default_rng(6716188855217730280)

    x = rng.random(size=3)
    rtol = 1e-15

    assert_allclose(dist.pdf(x), dist_ref.pdf(x), rtol=rtol)
    assert_allclose(dist.cdf(x), dist_ref.cdf(x), rtol=rtol)
    # assert_allclose(dist.sf(x), dist_ref.sf(x), rtol=rtol)
    # assert_allclose(dist.ppf(x), dist_ref.ppf(x), rtol=rtol)
    # assert_allclose(dist.isf(x), dist_ref.isf(x), rtol=rtol)
    assert_allclose(dist.logpdf(x), dist_ref.logpdf(x), rtol=rtol)
    assert_allclose(dist.logcdf(x), dist_ref.logcdf(x), rtol=rtol)
    # assert_allclose(dist.logsf(x), dist_ref.logsf(x), rtol=rtol)
    assert_allclose(dist.support(), dist_ref.support(), rtol=rtol)
    assert_allclose(dist.entropy(), dist_ref.entropy(), rtol=rtol)
    assert_allclose(dist.mean(), dist_ref.mean(), rtol=rtol)
    # assert_allclose(dist.var(), dist_ref.var(), rtol=rtol)
    # assert_allclose(dist.skew(), dist_ref.stats('s'), rtol=rtol)
    # assert_allclose(dist.kurtosis(), dist_ref.stats('k'), rtol=rtol)
