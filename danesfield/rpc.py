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

    def compute_partial_deriv_coeffs(self):
        """Compute the coefficients of the partial derivatives of the
        polynomials in the RPC with respect to X and Y
        """
        self.dx_coeff = numpy.zeros((4, 20), dtype=self.coeff.dtype)
        dx_ind = [1, 7, 4, 5, 14, 17, 10, 11, 12, 13]
        self.dx_coeff[:, :10] = self.coeff[:, dx_ind]
        self.dx_coeff[:, [1, 4, 5]] *= 2
        self.dx_coeff[:, 7] *= 3

        self.dy_coeff = numpy.zeros((4, 20), dtype=self.coeff.dtype)
        dy_ind = [2, 4, 8, 6, 12, 10, 18, 14, 15, 16]
        self.dy_coeff[:, :10] = self.coeff[:, dy_ind]
        self.dy_coeff[:, [2, 4, 6]] *= 2
        self.dy_coeff[:, 8] *= 3

    def jacobian(self, point):
        """Compute the Jacobian of the RPC at the given normalized world point

        Currently this only computes the 2x2 Jacobian for X and Y parameters.
        This funciton also returns the projected point in normalized coordinates
        """
        pv = self.power_vector(point)
        polys = numpy.dot(self.coeff, pv)
        dx_polys = numpy.dot(self.dx_coeff, pv)
        dy_polys = numpy.dot(self.dy_coeff, pv)
        J = numpy.empty((2,2), dtype=self.coeff.dtype)
        J[0,0] = (polys[1]*dx_polys[0] - polys[0]*dx_polys[1]) / (polys[1]**2)
        J[0,1] = (polys[1]*dy_polys[0] - polys[0]*dy_polys[1]) / (polys[1]**2)
        J[1,0] = (polys[3]*dx_polys[2] - polys[2]*dx_polys[3]) / (polys[3]**2)
        J[1,1] = (polys[3]*dy_polys[2] - polys[2]*dy_polys[3]) / (polys[3]**2)
        norm_pt = numpy.array([polys[0] / polys[1], polys[2] / polys[3]])
        return J, norm_pt

    @staticmethod
    def power_vector(point):
        """Compute the vector of polynomial terms

        Also applies to an (n,3) matrix where each row is a point.
        """
        x, y, z = point.transpose()
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
        try:
            w = numpy.ones(x.shape)
        except TypeError:
            w = 1
        # This is the standard order of terms used in NITF metadata
        return numpy.array([w, x, y, z, xy, xz, yz, xx, yy, zz,
                            xyz, xxx, xyy, xzz, xxy, yyy, yzz, xxz, yyz, zzz])

    def project(self, point):
        """Project a long, lat, elev point into image coordinates

        This function can also project an (n,3) matrix where each row of the
        matrix is a point to project.  The result is an (n,2) matrix of image
        coordinates.
        """
        norm_pt = (numpy.array(point) - self.world_offset) / self.world_scale
        polys = numpy.dot(self.coeff, self.power_vector(norm_pt))
        img_pt = numpy.array([polys[0] / polys[1], polys[2] / polys[3]])
        return img_pt.transpose() * self.image_scale + self.image_offset

    def back_project(self, image_point, elev):
        """Back project an image point with known elevation to long, lat

        This is the inverse of the project() function assuming that the
        elevation to project to is known.  This function requires an iterative
        solver to find the solution and is more expensive to compute than the
        forward projection
        """
        norm_img_pt = (numpy.array(image_point) - self.image_offset) \
                    / self.image_scale
        norm_elev = (numpy.array(elev) - self.world_offset[2]) \
                  /  self.world_scale[2]

        x = norm_img_pt.transpose()[0]
        y = norm_img_pt.transpose()[1]
        h = norm_elev

        # Use a first order approximate to the RPC to solve for an initial guess
        Bx = self.coeff[0,1:3] - numpy.outer(x, self.coeff[1,1:3])
        lx = x * (self.coeff[1,0] + self.coeff[1,3] * h) \
               - (self.coeff[0,0] + self.coeff[0,3] * h)
        lx = numpy.reshape(lx, (-1))
        By = self.coeff[2,1:3] - numpy.outer(y, self.coeff[3,1:3])
        ly = y * (self.coeff[3,0] + self.coeff[3,3] * h) \
               - (self.coeff[2,0] + self.coeff[2,3] * h)
        ly = numpy.reshape(ly, (-1))

        init_pt = numpy.empty((len(lx), 3))
        init_pt[:,2] = h
        self.compute_partial_deriv_coeffs()
        for i in range(len(lx)):
            B = numpy.stack((Bx[i], By[i]))
            l = numpy.stack((lx[i], ly[i]))
            init_pt[i,0:2] = numpy.linalg.solve(B,l)
            nip = numpy.reshape(norm_img_pt, (-1,2))[i]
            # Apply gradient descent until convergence
            # typically this only takes 2 or 3 iterations
            for k in range(10):
                J, pt = self.jacobian(init_pt[i])
                step = numpy.linalg.solve(J, nip - pt)
                init_pt[i,0:2] += step
                if numpy.max(numpy.abs(step)) < 1e-16:
                    break
        return init_pt * self.world_scale + self.world_offset



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
