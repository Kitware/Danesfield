###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import numpy


def compute_ndvi(msi_file):
    """
    Compute a normalized difference vegetation index (NVDI) image from an MSI file
    """
    num_bands = msi_file.RasterCount
    # Guess band indices based on the number of bands
    if num_bands == 8:
        # for 8-band MSI from WV3 and WV2 the RGB bands have these indices
        red_idx = 5
        nir_idx = 7
    elif num_bands == 4:
        # assume the bands are B,G,R,N (where N is near infrared)
        red_idx = 3
        nir_idx = 4
    else:
        raise RuntimeError("Unknown Red/NIR channels in {}-band image".format(num_bands))

    red_band = msi_file.GetRasterBand(red_idx)
    red = red_band.ReadAsArray()
    red_mask = red != red_band.GetNoDataValue()

    nir_band = msi_file.GetRasterBand(nir_idx)
    nir = nir_band.ReadAsArray()
    nir_mask = nir != nir_band.GetNoDataValue()

    mask = numpy.logical_and(red_mask, nir_mask)

    red = red.astype(numpy.float)
    nir = nir.astype(numpy.float)

    return numpy.divide(nir - red, nir + red, where=mask)
