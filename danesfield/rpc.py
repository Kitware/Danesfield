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
"""Model RPC (Rational Polynomial Camera) projections
"""

import numpy


class RPCModel(object):
    """Represents a Rational Polynomial Camera (RPC) model
    """

    def __init__(self):
        """Constructor from a GDAL RPC dictionary
        """
        # This constructs something like an identity camera
        # (Lon, Lat, Alt) ==> (Lon, Lat)
        dtype = 'float64'
        self.coeff = numpy.zeros((4, 20), dtype=dtype)
        self.coeff[0, 1] = 1
        self.coeff[2, 2] = 1
        self.world_offset = numpy.zeros((1, 3), dtype=dtype)
        self.world_scale = numpy.ones((1, 3), dtype=dtype)
        self.image_offset = numpy.zeros((1, 2), dtype=dtype)
        self.image_scale = numpy.ones((1, 2), dtype=dtype)

    @staticmethod
    def power_vector(point):
        """Compute the vector of polynomial terms
        """
        x, y, z = point
        xx = x * x
        xy = x * y
        xz = x * z
        yy = y * y
        yz = y * z
        zz = z * z
        xxx = xx * x
        xxy = xx * y
        xxz = xx * z
        xyy = xy * y
        xyz = xy * z
        xzz = xz * z
        yyy = yy * y
        yyz = yy * z
        yzz = yz * z
        zzz = zz * z
        # This is the standard order of terms used in NITF metadata
        return numpy.array([1, x, y, z, xy, xz, yz, xx, yy, zz,
                            xyz, xxx, xyy, xzz, xxy, yyy, yzz, xxz, yyz, zzz])

    def project(self, point):
        """Project a long, lat, alt point into image coordinates
        """
        norm_pt = (numpy.array(point) - self.world_offset) / self.world_scale
        polys = numpy.dot(self.coeff, self.power_vector(norm_pt))
        img_pt = numpy.array([polys[0] / polys[1], polys[2] / polys[3]])
        return img_pt * self.image_scale + self.image_offset


def rpc_from_gdal_dict(md_dict):
    """Construct a RPCModel from a GDAL RPC meta-data dictionary

    The format of the dictionary matches what GDAL returns and contains
    the standard fields from the RPC00B standard
    """
    def from_keys(keys):
        """Extract from the data dictionary by list of keys"""
        if keys in md_dict:
            return numpy.array(md_dict[keys].split(), dtype='float64')
        if all(k in md_dict for k in keys):
            return numpy.array([md_dict[k] for k in keys], dtype='float64')
        raise KeyError("Unable to find "+str(keys)+" in the dictionary")

    rpc = RPCModel()
    rpc.world_offset = from_keys(('LONG_OFF', 'LAT_OFF', 'HEIGHT_OFF'))
    rpc.world_scale = from_keys(('LONG_SCALE', 'LAT_SCALE', 'HEIGHT_SCALE'))
    rpc.image_offset = from_keys(('SAMP_OFF', 'LINE_OFF'))
    rpc.image_scale = from_keys(('SAMP_SCALE', 'LINE_SCALE'))
    rpc.coeff[0, :] = from_keys('SAMP_NUM_COEFF')
    rpc.coeff[1, :] = from_keys('SAMP_DEN_COEFF')
    rpc.coeff[2, :] = from_keys('LINE_NUM_COEFF')
    rpc.coeff[3, :] = from_keys('LINE_DEN_COEFF')
    return rpc


def rpc_to_gdal_dict(rpc, precision=12):
    """Construct a GDAL RPC meta-data dictionary from a RPCModel

    The format of the dictionary matches what GDAL returns and contains
    the standard fields from the RPC00B standard
    """
    md_dict = dict()
    fmt = '%0.' + str(int(precision)) + 'f'
    md_dict['LONG_OFF'] = fmt % rpc.world_offset[0]
    md_dict['LAT_OFF'] = fmt % rpc.world_offset[1]
    md_dict['HEIGHT_OFF'] = fmt % rpc.world_offset[2]
    md_dict['LONG_SCALE'] = fmt % rpc.world_scale[0]
    md_dict['LAT_SCALE'] = fmt % rpc.world_scale[1]
    md_dict['HEIGHT_SCALE'] = fmt % rpc.world_scale[2]
    md_dict['SAMP_OFF'] = fmt % rpc.image_offset[0]
    md_dict['LINE_OFF'] = fmt % rpc.image_offset[1]
    md_dict['SAMP_SCALE'] = fmt % rpc.image_scale[0]
    md_dict['LINE_SCALE'] = fmt % rpc.image_scale[1]
    md_dict['SAMP_NUM_COEFF'] = ' '.join(fmt % c for c in rpc.coeff[0, :])
    md_dict['SAMP_DEN_COEFF'] = ' '.join(fmt % c for c in rpc.coeff[1, :])
    md_dict['LINE_NUM_COEFF'] = ' '.join(fmt % c for c in rpc.coeff[2, :])
    md_dict['LINE_DEN_COEFF'] = ' '.join(fmt % c for c in rpc.coeff[3, :])
    return md_dict
