
def test_standard_absolute_deviation():
    from pysad.models import StandardAbsoluteDeviation
    import numpy as np
    from numpy.testing import assert_raises
    from pysad.utils import fix_seed
    from scipy.optimize import least_squares

    fix_seed(61)
    X = np.random.rand(150, 1)

    model = StandardAbsoluteDeviation(substracted_statistic="mean")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)
    assert y_pred.choose == least_squares(y_pred)

    model = StandardAbsoluteDeviation(substracted_statistic="median")
    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(substracted_statistic="asd")

    with assert_raises(ValueError):
        StandardAbsoluteDeviation(substracted_statistic=None)
