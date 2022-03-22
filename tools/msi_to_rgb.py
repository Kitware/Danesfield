#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import gdal
import logging
import numpy


def main(args):
    parser = argparse.ArgumentParser(
        description='Convert a multispectral (MSI) image to RGB')
    parser.add_argument("msi_image", help="Source MSI image file name")
    parser.add_argument("rgb_image", help="Destination RGB image file name")
    parser.add_argument('-b', "--byte", action="store_true",
                        help="Enable this option to stretch intensity range and "
                             "convert to a byte image")
    parser.add_argument('-a', "--alpha", action="store_true",
                        help="Create an alpha channel for an RGBA image instead "
                             "of using zero as a special no-value marker")
    parser.add_argument('-p', "--range-percentile", default=0.1, type=float,
                        help="The percent of largest and smallest intensities to "
                             "ignore when computing range for intensity scaling")
    parser.add_argument("--big", action="store_true",
                        help="Needed when the rgb_image is bigger than 4GB")
    args = parser.parse_args(args)

    if args.range_percentile < 0.0 or args.range_percentile >= 50.0:
        raise RuntimeError("Error: range percentile must be between 0 and 50")

    # open the source image
    msi_image = gdal.Open(args.msi_image, gdal.GA_ReadOnly)
    if not msi_image:
        raise RuntimeError("Error: Failed to open MSI {}".format(args.msi_image))

    driver = msi_image.GetDriver()
    driver_metadata = driver.GetMetadata()
    rgb_image = None
    if driver_metadata.get(gdal.DCAP_CREATE) != "YES":
        raise RuntimeError("Error: driver {} does not supports Create().".format(driver))

    num_bands = msi_image.RasterCount
    rgb_bands = [0, 0, 0]
    for b in range(1, num_bands+1):
        label = msi_image.GetRasterBand(b).GetRasterColorInterpretation()
        if label == gdal.GCI_RedBand:
            rgb_bands[0] = b
        elif label == gdal.GCI_GreenBand:
            rgb_bands[1] = b
        elif label == gdal.GCI_BlueBand:
            rgb_bands[2] = b

    if not numpy.all(rgb_bands):
        # Guess which band indices are RGB based on the number of bands
        if num_bands == 8:
            # for 8-band MSI from WV3 and WV2 the RGB bands have these indices
            rgb_bands = [5, 3, 2]
        elif num_bands == 4:
            # assume the bands are B,G,R,N (where N is near infrared)
            rgb_bands = [3, 2, 1]
        elif num_bands == 3:
            # assume that we have already converted to RGB
            rgb_bands = [1, 2, 3]
        else:
            raise RuntimeError("Error: unknown RGB channels in {}-band "
                               "image".format(num_bands))
        print("Assuming RGB labels on bands {}".format(rgb_bands))
    else:
        print("Found RGB labels on bands {}".format(rgb_bands))

    # Set the output data type, either match the input or use byte (uint8)
    if args.byte:
        eType = gdal.GDT_Byte
    else:
        eType = msi_image.GetRasterBand(1).DataType

    # Create the output RGB image
    print("Create destination image (format {}), "
          "size:({}, {}) ...".format(gdal.GetDataTypeName(eType),
                                     msi_image.RasterXSize,
                                     msi_image.RasterYSize))
    projection = msi_image.GetProjection()
    transform = msi_image.GetGeoTransform()
    gcpProjection = msi_image.GetGCPProjection()
    gcps = msi_image.GetGCPs()
    options = ["PHOTOMETRIC=RGB", "COMPRESS=DEFLATE"]
    if (args.big):
        options.append("BIGTIFF=YES")
    else:
        options.append("BIGTIFF=IF_SAFER")
    num_out_bands = 3
    if args.alpha:
        options.append("ALPHA=YES")
        num_out_bands = 4
    # ensure that space will be reserved for geographic corner coordinates
    # (in DMS) to be set later
    if (driver.ShortName == "NITF" and not projection):
        options.append("ICORDS=G")
    rgb_image = driver.Create(args.rgb_image, xsize=msi_image.RasterXSize,
                              ysize=msi_image.RasterYSize, bands=num_out_bands,
                              eType=eType, options=options)

    # Copy over georeference information
    if (projection):
        # georeference through affine geotransform
        rgb_image.SetProjection(projection)
        rgb_image.SetGeoTransform(transform)
    else:
        # georeference through GCPs
        rgb_image.SetGCPs(gcps, gcpProjection)

    # Copy the image data
    band_names = ["red", "green", "blue"]
    band_types = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand]
    alpha = None
    dtype = numpy.uint8
    for out_idx, in_idx in enumerate(rgb_bands, 1):
        in_band = msi_image.GetRasterBand(in_idx)
        out_band = rgb_image.GetRasterBand(out_idx)
        out_band.SetRasterColorInterpretation(band_types[out_idx-1])

        in_data = in_band.ReadAsArray()
        # mask out pixels with no valid value
        in_no_data_val = in_band.GetNoDataValue()
        mask = in_data == in_no_data_val
        valid = numpy.logical_not(mask)
        if args.alpha:
            if alpha is None:
                alpha = numpy.logical_and(alpha, valid)
            else:
                alpha = valid

        # if not stretching to byte range, just copy the data
        if not args.byte or in_band.DataType == gdal.GDT_Byte:
            out_band.WriteArray(in_data)
            if in_no_data_val is not None:
                out_band.SetNoDataValue(in_no_data_val)
            dtype = in_data.dtype
            continue

        valid_data = in_data[valid]

        # robustly find a range for intensity scaling
        min_p = args.range_percentile
        max_p = 100.0 - min_p
        min_val = numpy.percentile(valid_data, min_p)
        max_val = numpy.percentile(valid_data, max_p)
        print("{} band detect range: [{}, {}]".format(band_names[out_idx-1],
                                                      min_val, max_val))

        # clip and scale the data to fit the range and cast to byte
        in_data = in_data.clip(min_val, max_val)
        if args.alpha or (in_no_data_val is None):
            # use the full 8-bit range
            scale = 255.0 / float(max_val - min_val)
            in_data = (in_data - min_val) * scale
        else:
            # use 1-255 and reserve 0 as a no data flag
            scale = 254.0 / float(max_val - min_val)
            in_data = (in_data - min_val) * scale + 1
            out_band.SetNoDataValue(0)
        out_data = in_data.astype(numpy.uint8)
        # set the masked out invalid pixels to 0
        out_data[mask] = 0
        out_band.WriteArray(out_data)
        dtype = out_data.dtype

    if args.alpha:
        out_band = rgb_image.GetRasterBand(4)
        out_band.SetRasterColorInterpretation(gdal.GCI_AlphaBand)
        alpha = alpha.astype(dtype)
        if dtype == numpy.uint8:
            alpha *= 255
        elif dtype == numpy.uint16:
            alpha *= 65535
        out_band.WriteArray(alpha)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
