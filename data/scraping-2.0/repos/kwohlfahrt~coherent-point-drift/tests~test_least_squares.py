import numpy as np
from coherent_point_drift.least_squares import *
from coherent_point_drift.geometry import RigidXform, randomRotations, rotationMatrix
from coherent_point_drift.util import last
from itertools import islice


def test_least_squares():
    rng = np.random.RandomState(4)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim, rng)))
    t = rng.normal(size=ndim)
    s = rng.lognormal(size=1)[0]
    xform = RigidXform(R, t, s)

    X = rng.normal(size=(10, ndim))
    Y = xform @ X

    alignment = align(X, Y)
    expected = xform.inverse

    np.testing.assert_almost_equal(alignment.R, expected.R)
    np.testing.assert_almost_equal(alignment.t, expected.t)
    np.testing.assert_almost_equal(alignment.s, expected.s)
    np.testing.assert_almost_equal(alignment @ Y, X)


def test_cpd_prior():
    from coherent_point_drift.align import driftRigid
    rng = np.random.RandomState(4)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim, rng)))
    t = rng.normal(size=ndim)
    s = rng.normal(size=1)[0]

    X = rng.normal(size=(10, ndim))
    Y = RigidXform(R, t, s) @ X

    _, cpd = last(islice(driftRigid(X, Y, w=np.eye(len(X))), 200))
    ls = align(X, Y)
    np.testing.assert_almost_equal(cpd.R, ls.R)
    np.testing.assert_almost_equal(cpd.t, ls.t)
    np.testing.assert_almost_equal(cpd.s, ls.s)


def test_mirror():
    # L-shape
    X = np.array([[1, 0], [0, 0], [0, 1], [0, 2], [0, 3]])
    Y = X * np.array([[-1, 1]])
    np.testing.assert_almost_equal(align(X, Y, mirror=True) @ Y, X)
