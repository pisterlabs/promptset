"""Rotation utility functions from OpenAI's mujoco-worldgen repository.

Functions have been extended to our 6D embedding.

https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/rotation.py.
"""
# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Many methods borrow heavily or entirely from transforms3d:
# https://github.com/matthew-brett/transforms3d
# They have mostly been modified to support batched operations.

import numpy as np
"""
Rotations
=========
Note: these have caused many subtle bugs in the past.
Be careful while updating these methods and while using them in clever ways.
See MuJoCo documentation here: http://mujoco.org/book/modeling.html#COrientation
Conventions
-----------
    - All functions accept batches as well as individual rotations
    - All rotation conventions match respective MuJoCo defaults
    - All angles are in radians
    - Matricies follow LR convention
    - Euler Angles are all relative with 'xyz' axes ordering
    - See specific representation for more information
Representations
---------------
Euler
    There are many euler angle frames -- here we will strive to use the default
        in MuJoCo, which is eulerseq='xyz'.
    This frame is a relative rotating frame, about x, y, and z axes in order.
        Relative rotating means that after we rotate about x, then we use the
        new (rotated) y, and the same for z.
Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:
            q0 = cos(angle / 2)
            q1 = sin(angle / 2) * x
            q2 = sin(angle / 2) * y
            q3 = sin(angle / 2) * z
        This is also sometimes called qw, qx, qy, qz.
    Note that quaternions are ambiguous, because we can represent a rotation by
        angle about vector <x, y, z> and -angle about vector <-x, -y, -z>.
        To choose between these, we pick "first nonzero positive", where we
        make the first nonzero element of the quaternion positive.
    This can result in mismatches if you're converting an quaternion that is not
        "first nonzero positive" to a different representation and back.
Axis Angle
    (Not currently implemented)
    These are very straightforward.  Rotation is angle about a unit vector.
XY Axes
    (Not currently implemented)
    We are given x axis and y axis, and z axis is cross product of x and y.
Z Axis
    This is NOT RECOMMENDED.  Defines a unit vector for the Z axis,
        but rotation about this axis is not well defined.
    Instead pick a fixed reference direction for another axis (e.g. X)
        and calculate the other (e.g. Y = Z cross-product X),
        then use XY Axes rotation instead.
SO3
    (Not currently implemented)
    While not supported by MuJoCo, this representation has a lot of nice features.
    We expect to add support for these in the future.
TODO / Missing
--------------
    - Rotation integration or derivatives (e.g. velocity conversions)
    - More representations (SO3, etc)
    - Random sampling (e.g. sample uniform random rotation)
    - Performance benchmarks/measurements
    - (Maybe) define everything as to/from matricies, for simplicity
"""

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def mat2euler(mat: np.ndarray) -> np.ndarray:
    """Convert Rotation Matrix to Euler Angles.

    Args:
        mat: Rotation matrix.

    See rotation.py for notes
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler


def euler2quat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler Angles to Quaternions.

    See rotation.py for notes.

    Args:
        euler: Array of euler angles.

    Returns:
        An array of quaternions.
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2quat(mat: np.ndarray) -> np.ndarray:
    """Convert Rotation Matrices to Quaternions.

    See rotation.py for notes.

    Args:
        mat: Array of rotations matrices.

    Returns:
        An array of quaternions.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def fastmat2quat(mat: np.ndarray) -> np.ndarray:
    """Faster matrix to quaternion conversion.

    See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    tr0 = 1.0 + mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    tr1 = 1.0 + mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]
    tr2 = 1.0 - mat[..., 0, 0] + mat[..., 1, 1] - mat[..., 2, 2]
    tr3 = 1.0 - mat[..., 0, 0] - mat[..., 1, 1] + mat[..., 2, 2]

    # Calculate which conversion to take for which matrix for best numeric stability
    q = np.empty(mat.shape[:-2] + (4,))
    # idx1 = np.logical_and(tr1 > tr2, tr1 > tr3)
    # idx2 = np.logical_and(tr2 > tr1, tr2 > tr3)
    # idx3 = np.logical_not(np.logical_or(idx1, idx2))

    idx0 = tr0 > 0
    nidx0 = np.logical_not(idx0)
    idx1 = np.logical_and(np.logical_and(tr1 > tr2, tr1 > tr3), nidx0)
    idx2 = np.logical_and(np.logical_and(tr2 > tr1, tr2 > tr3), nidx0)
    idx3 = np.logical_and(np.logical_not(np.logical_or(idx1, idx2)), nidx0)

    s0 = np.sqrt(tr0[idx0]) * 2
    s1 = np.sqrt(tr1[idx1]) * 2
    s2 = np.sqrt(tr2[idx2]) * 2
    s3 = np.sqrt(tr3[idx3]) * 2

    q[idx0, 0] = 0.25 * s0
    q[idx0, 1] = (mat[idx0, 2, 1] - mat[idx0, 1, 2]) / s0
    q[idx0, 2] = (mat[idx0, 0, 2] - mat[idx0, 2, 0]) / s0
    q[idx0, 3] = (mat[idx0, 1, 0] - mat[idx0, 0, 1]) / s0

    q[idx1, 0] = (mat[idx1, 2, 1] - mat[idx1, 1, 2]) / s1
    q[idx1, 1] = 0.25 * s1
    q[idx1, 2] = (mat[idx1, 0, 1] + mat[idx1, 1, 0]) / s1
    q[idx1, 3] = (mat[idx1, 0, 2] + mat[idx1, 2, 0]) / s1

    q[idx2, 0] = (mat[idx2, 0, 2] - mat[idx2, 2, 0]) / s2
    q[idx2, 1] = (mat[idx2, 0, 1] + mat[idx2, 1, 0]) / s2
    q[idx2, 2] = 0.25 * s2
    q[idx2, 3] = (mat[idx2, 1, 2] + mat[idx2, 2, 1]) / s2

    q[idx3, 0] = (mat[idx3, 1, 0] - mat[idx3, 0, 1]) / s3
    q[idx3, 1] = (mat[idx3, 0, 2] + mat[idx3, 2, 0]) / s3
    q[idx3, 2] = (mat[idx3, 1, 2] + mat[idx3, 2, 1]) / s3
    q[idx3, 3] = 0.25 * s3

    q[q[..., 0] < 0, :] *= -1  # Prefer quaternion with positive w
    return q


def quat2mat(quat: np.ndarray) -> np.ndarray:
    """Convert Quaternions to Euler Angles.

    See rotation.py for notes.

    Args:
        quat: Array of quaternions.

    Returns:
        An array of euler angles.
    """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, f"Invalid shape quat {quat}"

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Multiply Quaternions.

    Args:
        q0: First array of quaternions.
        q1: Second array of quaternions.

    Returns:
        The multiplied quaternions.
    """
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_conjugate(q: np.array) -> np.array:
    """Conjugate Quaternions.

    Args:
        q: Array of quaternions.

    Returns:
        The conjugated quaternions.
    """
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def vec2quat(x: np.ndarray) -> np.ndarray:
    """Convert vectors to UnitQuaternions.

    Args:
        x: Vector or tensor of vectors.

    Returns:
        The normalized quaternions.
    """
    assert x.shape[-1] == 4
    q = x / np.linalg.norm(x, axis=-1, keepdims=True)
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    q[q[..., 0] < 0] *= -1
    return q


def axisangle2quat(x: float, y: float, z: float, a: float) -> np.ndarray:
    """Convert a single axis-angle to a Quaternion.

    Args:
        x: X-axis component.
        y: Y-axis component.
        z: Z-axis component.
        a: Angle around the axis in radians.

    Returns:
        The quaternion.
    """
    sin_a = np.sin(a / 2.)
    x *= sin_a
    y *= sin_a
    z *= sin_a
    q = np.array([np.cos(a / 2.), x, y, z])
    return q / np.linalg.norm(q)


def quat2embedding(quat: np.ndarray) -> np.ndarray:
    """Convert Quaternions to Embeddings.

    Args:
        quat: An array of quaternions.

    Returns:
        The embeddings.
    """
    assert quat.shape[-1] == 4
    return mat2embedding(quat2mat(quat))


def embedding2quat(embedding: np.ndarray, regularize: bool = False) -> np.ndarray:
    """Convert Embeddings to Quaternions.

    The embeddings are assumed to have the form [a11, a12, a13, a21, a22, a23].

    Args:
        embedding: An array of embeddings.
        regularize: If True, the embedding is regularized to a proper embedding before conversion.

    Returns:
        The quaternions.
    """
    assert embedding.shape[-1] == 6
    if regularize:
        b1 = embedding[..., 0:3] / np.linalg.norm(embedding[..., 0:3], axis=-1, keepdims=True)
        # np.sum for batched dot product
        b2 = embedding[..., 3:6] - (np.sum(b1 * embedding[..., 3:6], axis=-1, keepdims=True) * b1)
        b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    else:
        b1 = embedding[..., 0:3]
        b2 = embedding[..., 3:6]
    b3 = np.cross(b1, b2, axis=-1)

    Qxx, Qyx, Qzx = b1[..., 0], b2[..., 0], b3[..., 0]
    Qxy, Qyy, Qzy = b1[..., 1], b2[..., 1], b3[..., 1]
    Qxz, Qyz, Qzz = b1[..., 2], b2[..., 2], b3[..., 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(embedding.shape[:-1] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def fastembedding2quat(embedding: np.ndarray, regularize: bool = False) -> np.ndarray:
    """Faster embedding to quaternion conversion.

    See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    mat = embedding2mat(embedding, regularize=regularize)

    tr0 = 1.0 + mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    tr1 = 1.0 + mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]
    tr2 = 1.0 - mat[..., 0, 0] + mat[..., 1, 1] - mat[..., 2, 2]
    tr3 = 1.0 - mat[..., 0, 0] - mat[..., 1, 1] + mat[..., 2, 2]

    # Calculate which conversion to take for which matrix for best numeric stability
    q = np.empty(mat.shape[:-2] + (4,))

    idx0 = tr0 > 0
    nidx0 = np.logical_not(idx0)
    idx1 = np.logical_and(np.logical_and(tr1 > tr2, tr1 > tr3), nidx0)
    idx2 = np.logical_and(np.logical_and(tr2 > tr1, tr2 > tr3), nidx0)
    idx3 = np.logical_and(np.logical_not(np.logical_or(idx1, idx2)), nidx0)

    s0 = np.sqrt(tr0[idx0]) * 2
    s1 = np.sqrt(tr1[idx1]) * 2
    s2 = np.sqrt(tr2[idx2]) * 2
    s3 = np.sqrt(tr3[idx3]) * 2

    q[idx0, 0] = 0.25 * s0
    q[idx0, 1] = (mat[idx0, 2, 1] - mat[idx0, 1, 2]) / s0
    q[idx0, 2] = (mat[idx0, 0, 2] - mat[idx0, 2, 0]) / s0
    q[idx0, 3] = (mat[idx0, 1, 0] - mat[idx0, 0, 1]) / s0

    q[idx1, 0] = (mat[idx1, 2, 1] - mat[idx1, 1, 2]) / s1
    q[idx1, 1] = 0.25 * s1
    q[idx1, 2] = (mat[idx1, 0, 1] + mat[idx1, 1, 0]) / s1
    q[idx1, 3] = (mat[idx1, 0, 2] + mat[idx1, 2, 0]) / s1

    q[idx2, 0] = (mat[idx2, 0, 2] - mat[idx2, 2, 0]) / s2
    q[idx2, 1] = (mat[idx2, 0, 1] + mat[idx2, 1, 0]) / s2
    q[idx2, 2] = 0.25 * s2
    q[idx2, 3] = (mat[idx2, 1, 2] + mat[idx2, 2, 1]) / s2

    q[idx3, 0] = (mat[idx3, 1, 0] - mat[idx3, 0, 1]) / s3
    q[idx3, 1] = (mat[idx3, 0, 2] + mat[idx3, 2, 0]) / s3
    q[idx3, 2] = (mat[idx3, 1, 2] + mat[idx3, 2, 1]) / s3
    q[idx3, 3] = 0.25 * s3

    q[q[..., 0] < 0, :] *= -1  # Prefer quaternion with positive w
    return q


def embedding2mat(embedding: np.ndarray, regularize: bool = False) -> np.ndarray:
    """Convert Embeddings to Rotation Matrices.

    The embeddings are assumed to have the form [a11, a12, a13, a21, a22, a23].

    Args:
        embedding: An array of embeddings.
        regularize: If True, the embedding is regularized to a proper embedding before conversion.

    Returns:
        The rotation matrices.
    """
    assert embedding.shape[-1] == 6
    if regularize:
        b1 = embedding[..., 0:3] / np.linalg.norm(embedding[..., 0:3], axis=-1, keepdims=True)
        # np.sum for batched dot product
        b2 = embedding[..., 3:6] - (np.sum(b1 * embedding[..., 3:6], axis=-1, keepdims=True) * b1)
        b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    else:
        b1 = embedding[..., 0:3]
        b2 = embedding[..., 3:6]
    b3 = np.cross(b1, b2, axis=-1)

    mat = np.empty(embedding.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2] = b1[..., 0], b2[..., 0], b3[..., 0]
    mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2] = b1[..., 1], b2[..., 1], b3[..., 1]
    mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2] = b1[..., 2], b2[..., 2], b3[..., 2]
    return mat


def mat2embedding(mat: np.ndarray) -> np.ndarray:
    """Convert Rotation Matrices to Embeddings.

    Args:
        mat: An array of rotation matrices.

    Returns:
        The embeddings.
    """
    assert mat.shape[-2:] == (3, 3)
    return np.concatenate((mat[..., :, 0], mat[..., :, 1]), axis=-1)


def map2pi(theta: np.ndarray) -> np.ndarray:
    """Map an angle to the interval of [-np.pi, np.pi].

    Args:
        theta: An array of angles (in radians).

    Returns:
        The mapped angles.
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi
