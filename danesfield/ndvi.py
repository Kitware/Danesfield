###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import numpy as np


def chanNmask(msi_file, ndx):
    band = msi_file.GetRasterBand(ndx)
    chan = band.ReadAsArray()
    mask = chan != band.GetNoDataValue()
    return chan, mask

def normalize(arr, rng=[0,1]): # return array scaled to range
    a, b = np.min(arr), np.max(arr)
    c, d = rng[0], rng[1]
    e, g = b-a, d-c
    if e:
        k = g/e    
        l = c - k*a
        res = k*arr + l
    else:
        res = .5*g*np.ones_like(arr)
    return res


def compute_ndvi(msi_file, visible=False):
    """
    Compute a normalized difference vegetation index (NVDI) image from an MSI file.
    For visible NDVI (vNDVI) use the method described in
    Costa et al. A new visible band index (vNDVI) for estimating NDVI values on RGB images utilizing genetic algorithms. CEA 2020
    """
    num_bands = msi_file.RasterCount
    # Guess band indices based on the number of bands
    if num_bands == 8:
        # for 8-band MSI from WV3 and WV2 the RGB bands have these indices
        blue_idx, green_idx, red_idx, nir_idx = 2, 3, 5, 7
    elif num_bands == 4:
        # assume the bands are B,G,R,N (where N is near infrared)
        blue_idx, green_idx, red_idx, nir_idx = 1,2,3,4
    else:
        raise RuntimeError("Unknown Red/NIR channels in {}-band image".format(num_bands))

    red, red_mask = chanNmask(msi_file, red_idx)
    nir, nir_mask = chanNmask(msi_file, nir_idx)

    if visible:
        blue, blue_mask = chanNmask(msi_file, blue_idx)
        green, green_mask = chanNmask(msi_file, green_idx)
        mask = np.logical_and(red_mask, green_mask, blue_mask)
        # constants from the paper
        C, rp, gp, bp = 0.5268, -0.1294, 0.3389, -0.3118
        R = np.power(red, rp, where=mask)
        G = np.power(green, gp, where=mask)
        B = np.power(blue, bp, where=mask)
        V = np.power(C*R*G*B, 1./3, where=mask)
        res = normalize(V, [-1,1])
    else:
        mask = np.logical_and(red_mask, nir_mask)
        red = red.astype(np.float)
        nir = nir.astype(np.float)
        res = np.divide(nir - red, nir + red, where=mask)

    return res
