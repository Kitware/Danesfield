#!/usr/bin/env python
#ckwg +28
# Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Parse the Raytheon file format for RPC parameters
"""

import os.path
import numpy
from danesfield.rpc import *


def parse_raytheon_rpc_file(fp):
    """Parse the Raytheon RPC file format from an open file pointer
    """
    def parse_rational_poly(fp):
        """Parse coefficients for a two polynomials from the file stream
        """
        coeff = numpy.zeros((2,20), dtype='float64')
        idx = 0
        powers = True
        # The expected exponent order matrix.  Currently we only support this default
        # if what is in the file doesn't match, raise an exception
        expected_exp_mat =[[0, 0, 0, 1],[1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],
                           [1, 1, 0, 1],[1, 0, 1, 1],[0, 1, 1, 1],[2, 0, 0, 1],
                           [0, 2, 0, 1],[0, 0, 2, 1],[1, 1, 1, 1],[3, 0, 0, 1],
                           [1, 2, 0, 1],[1, 0, 2, 1],[2, 1, 0, 1],[0, 3, 0, 1],
                           [0, 1, 2, 1],[2, 0, 1, 1],[0, 2, 1, 1],[0, 0, 3, 1]]
        for line in fp:
            if line.strip() == '20':
                data = []
                for i in range(20):
                    data.append(fp.next())
                if powers:
                    powers = False
                    exp_mat = numpy.array([d.split() for d in data], dtype='int')
                    if not numpy.array_equal(exp_mat, expected_exp_mat):
                        raise ValueError
                else:
                    powers = True
                    coeff[idx,:] = numpy.array(data, dtype='float64')
                    idx = idx + 1
                if idx > 1:
                    break
        return coeff

    rpc = RPCModel()
    for line in fp:
        if line.startswith('# uvOffset_'):
            line = fp.next()
            rpc.image_offset = numpy.array(line.split(), dtype='float64')
        if line.startswith('# uvScale_'):
            line = fp.next()
            rpc.image_scale = numpy.array(line.split(), dtype='float64')
        if line.startswith('# xyzOffset_'):
            line = fp.next()
            rpc.world_offset = numpy.array(line.split(), dtype='float64')
        if line.startswith('# xyzScale_'):
            line = fp.next()
            rpc.world_scale = numpy.array(line.split(), dtype='float64')
        if line.startswith('# u=sample'):
            rpc.coeff[0:2,:] = parse_rational_poly(fp)
        if line.startswith('# v=line'):
            rpc.coeff[2:4,:] = parse_rational_poly(fp)
    return rpc


def read_raytheon_rpc_file(filename):
    """Read a Raytheon RPC file
    """
    if os.path.isfile(filename):
        print "Reading RPC from ", filename
        with open(filename, 'r') as f:
            return parse_raytheon_rpc_file(f)
