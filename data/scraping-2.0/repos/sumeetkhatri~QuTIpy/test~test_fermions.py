#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2023 Sumeet Khatri.
#                       --.- ..- - .. .--. -.--
#
#
# SPDX-License-Identifier: AGPL-3.0
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np

from qutipy.fermions import coherent_state_fermi, cov_matrix_fermi, jordan_wigner

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

H = np.dot(np.sqrt(1 / 2), np.array([[1, 1], [1, -1]]))


def test_jordan_wigner():
    jw = jordan_wigner(2)
    assert len(jw) == 2
    assert len(jw[0]) == 2
    assert len(jw[1]) == 4

    # 0
    assert np.all(
        jw[0][1]
        == np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        jw[0][2]
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.0, -1.0],
                [0.0, 0.0, -0.0, -0.0],
            ]
        )
    )

    # 1
    assert np.all(
        jw[1][1]
        == np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        jw[1][2]
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.0, -1.0],
                [0.0, 0.0, -1.0, -0.0],
            ]
        )
    )
    assert np.all(
        jw[1][3]
        == np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, -0.0 - 1.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.0 - 1.0j],
                [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        )
    )
    assert np.all(
        jw[1][4]
        == np.array(
            [
                [0.0 + 0.0j, -0.0 - 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -0.0 - 1.0j, 0.0 + 0.0j],
            ]
        )
    )


def test_coherent_state_fermi():
    assert np.all(
        np.round(coherent_state_fermi(X).flatten(), 5)
        == np.array(
            [
                0.10608 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.15912 + 0.0j,
                0.0 + 0.0j,
                -0.31824 + 0.0j,
                -0.15912 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.47736 + 0.0j,
                -0.31824 + 0.0j,
                0.0 + 0.0j,
                -0.15912 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )
    )


def test_cov_matrix_fermi():
    assert np.all(
        cov_matrix_fermi(X, 2)
        == np.array(
            [
                [0.0 + 0.0j, -0.0 - 12.0j, 20.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 12.0j, 0.0 + 0.0j, 34.0 + 0.0j, 10.0 + 0.0j],
                [-20.0 + 0.0j, -34.0 + 0.0j, 0.0 + 0.0j, 0.0 + 6.0j],
                [0.0 + 0.0j, -10.0 + 0.0j, -0.0 - 6.0j, 0.0 + 0.0j],
            ]
        )
    )
