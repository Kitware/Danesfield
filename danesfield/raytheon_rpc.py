#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright 2017 by Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
"""Parse the Raytheon file format for RPC parameters
"""

import os.path
import numpy
from danesfield.rpc import RPCModel


def parse_raytheon_rpc_file(fp):
    """Parse the Raytheon RPC file format from an open file pointer
    """
    def parse_rational_poly(fp):
        """Parse coefficients for a two polynomials from the file stream
        """
        coeff = numpy.zeros((2, 20), dtype='float64')
        idx = 0
        powers = True
        # The expected exponent order matrix.  Currently we only support this
        # default.  If what is in the file doesn't match, raise an exception
        exp_exp_mat = [[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
                       [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [2, 0, 0, 1],
                       [0, 2, 0, 1], [0, 0, 2, 1], [1, 1, 1, 1], [3, 0, 0, 1],
                       [1, 2, 0, 1], [1, 0, 2, 1], [2, 1, 0, 1], [0, 3, 0, 1],
                       [0, 1, 2, 1], [2, 0, 1, 1], [0, 2, 1, 1], [0, 0, 3, 1]]
        for line in fp:
            if line.strip() == '20':
                data = []
                for i in range(20):
                    data.append(fp.readline())
                if powers:
                    powers = False
                    exp_mat = numpy.array([d.split() for d in data],
                                          dtype='int')
                    if not numpy.array_equal(exp_mat, exp_exp_mat):
                        raise ValueError
                else:
                    powers = True
                    coeff[idx, :] = numpy.array(data, dtype='float64')
                    idx = idx + 1
                if idx > 1:
                    break
        return coeff

    rpc = RPCModel()
    for line in fp:
        if line.startswith('# uvOffset_'):
            line = fp.readline()
            rpc.image_offset = numpy.array(line.split(), dtype='float64')
        if line.startswith('# uvScale_'):
            line = fp.readline()
            rpc.image_scale = numpy.array(line.split(), dtype='float64')
        if line.startswith('# xyzOffset_'):
            line = fp.readline()
            rpc.world_offset = numpy.array(line.split(), dtype='float64')
        if line.startswith('# xyzScale_'):
            line = fp.readline()
            rpc.world_scale = numpy.array(line.split(), dtype='float64')
        if line.startswith('# u=sample'):
            rpc.coeff[0:2, :] = parse_rational_poly(fp)
        if line.startswith('# v=line'):
            rpc.coeff[2:4, :] = parse_rational_poly(fp)
    return rpc


def read_raytheon_rpc_file(filename):
    """Read a Raytheon RPC file
    """
    if os.path.isfile(filename):
        print("Reading RPC from ", filename)
        with open(filename, 'r') as f:
            return parse_raytheon_rpc_file(f)
