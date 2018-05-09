
import argparse
import gdal
import gdalnumeric
import numpy
import sys
import scipy.ndimage.measurements as ndm
import scipy.ndimage.morphology as morphology

import cv2

def compute_ndvi(msi_file):
    """
    Compute a normalized difference vegetation index (NVDI) image from an MSI file
    """
    num_bands = msi_file.RasterCount
    red_idx = None
    nir_idx = None
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
        print("Unknown Red/NIR channels in {}-band image".format(num_bands))
        sys.exit(1)
    
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


def save_ndvi(ndvi, msi_file, filename):
    driver = msi_file.GetDriver()
    driver_metadata = driver.GetMetadata()
    ndvi_image = None
    if driver_metadata.get(gdal.DCAP_CREATE) == "YES":
        options = []
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (driver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        ndvi_file = driver.Create(
            filename, xsize=ndvi.shape[1],
            ysize=ndvi.shape[0],
            bands=1, eType=gdal.GDT_Float32,
            options=options)

        gdalnumeric.CopyDatasetInfo(msi_file, ndvi_file)
    else:
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)

    ndvi_band = ndvi_file.GetRasterBand(1)
    ndvi_band.WriteArray(ndvi)


def save_ndsm(ndsm, dsm_file, filename):
    driver = dsm_file.GetDriver()
    driver_metadata = driver.GetMetadata()
    ndsm_image = None
    if driver_metadata.get(gdal.DCAP_CREATE) == "YES":
        options = []
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (driver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        ndsm_file = driver.Create(
            filename, xsize=ndsm.shape[1],
            ysize=ndsm.shape[0],
            bands=1, eType=gdal.GDT_Float32,
            options=options)

        gdalnumeric.CopyDatasetInfo(dsm_file, ndsm_file)
    else:
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)

    ndsm_band = ndsm_file.GetRasterBand(1)
    ndsm_band.WriteArray(ndsm)
    no_data_val = dsm_file.GetRasterBand(1).GetNoDataValue()
    ndsm_band.SetNoDataValue(no_data_val)




def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Segment buildings by comparing a DSM to a DTM')
    parser.add_argument("source_dsm",
                        help="Digital surface model (DSM) image file name")
    parser.add_argument("source_dtm",
                        help="Digital terrain model (DTM) image file name")
    parser.add_argument("--msi",
                        help="Optional MSI image from which to compute NDVI "
                             "for vegetation removal")
    parser.add_argument("--ndsm",
                        help="Write out the normalized DSM image")
    parser.add_argument("destination_mask",
                        help="Building mask output image")
    args = parser.parse_args()

    # For now assume the input DSM and DTM are in the same resolution,
    # aligned, and in the same coordinates.  Later we can warp the DTM
    # to the DSM, if needed.

    # open the DSM
    dsm_file = gdal.Open(args.source_dsm, gdal.GA_ReadOnly)
    if not dsm_file:
        print("Unable to open {}".format(args.source_dsm))
        exit(1)
    dsm_band = dsm_file.GetRasterBand(1)
    dsm = dsm_band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=dsm_file.RasterXSize, win_ysize=dsm_file.RasterYSize)
    dsm_nodata_value = dsm_band.GetNoDataValue()
    print("DSM raster shape {}".format(dsm.shape))

    # open the DTM
    dtm_file = gdal.Open(args.source_dtm, gdal.GA_ReadOnly)
    if not dtm_file:
        print("Unable to open {}".format(args.source_dtm))
        exit(1)
    dtm_band = dtm_file.GetRasterBand(1)
    dtm = dtm_band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=dtm_file.RasterXSize, win_ysize=dtm_file.RasterYSize)
    dtm_nodata_value = dtm_band.GetNoDataValue()
    print("DTM raster shape {}".format(dtm.shape))

    mask = (dsm - dtm) > 2
    seeds = (dsm - dtm) > 4

    if args.msi:
        msi_file = gdal.Open(args.msi, gdal.GA_ReadOnly)
        if not msi_file:
            print("Unable to open {}".format(args.msi))
            exit(1)
        ndvi = compute_ndvi(msi_file)
        save_ndvi(ndvi, msi_file, "NDVI.tiff")
        mask[ndvi > 0.2] = False
        seeds[ndvi > 0.1] = False

    mask = morphology.binary_opening(mask, numpy.ones((3,3)), iterations=1)
    mask = morphology.binary_closing(mask, numpy.ones((3,3)), iterations=1)
    seeds = morphology.binary_opening(seeds, numpy.ones((3,3)), iterations=1)
    seeds = morphology.binary_closing(seeds, numpy.ones((3,3)), iterations=1)

    label_img = ndm.label(seeds)[0]
    labels, counts = numpy.unique(label_img, return_counts=True)
    for idx, count in zip(labels, counts):
        if idx == 0:
            continue
        if count < 500:
            seeds[label_img == idx] = False
        print("index {} has size {}".format(idx, count))

    cv2.imshow('mask1', mask.astype(numpy.uint8)*127 + seeds.astype(numpy.uint8)*127)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    seed_img, num_seeds = ndm.label(seeds)
    label_img = ndm.label(mask)[0]
    label_set = set()
    for idx in range(1, num_seeds+1):
        print(numpy.unique(label_img[seed_img == idx]))
        label_set.update(set(numpy.unique(label_img[seed_img == idx]).tolist()))
    print("labels")
    print(label_set)
    good_mask = numpy.full(mask.shape, False)
    for idx in label_set:
        if idx == 0:
            continue
        print(numpy.count_nonzero(label_img == idx))
        good_mask[label_img == idx] = True

    good_mask = morphology.binary_closing(good_mask, numpy.ones((3,3)), iterations=3)

    cv2.imshow('mask2', good_mask.astype(numpy.uint8)*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convert the mask to label map with value 2 (background) and 6 (building)
    cls = numpy.full(good_mask.shape, 2)
    cls[good_mask] = 6

    if args.ndsm:
        ndsm = dsm - dtm
        ndsm[dsm == dsm_nodata_value] = dsm_nodata_value
        save_ndsm(ndsm, dsm_file, args.ndsm) 


    # create the mask image
    driver = dsm_file.GetDriver()
    driver_metadata = driver.GetMetadata()
    destImage = None
    if driver_metadata.get(gdal.DCAP_CREATE) == "YES":
        print("Create destination mask of "
              "size:({}, {}) ...".format(dsm_file.RasterXSize,
                                         dsm_file.RasterYSize))
        options = []
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (driver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        dest_file = driver.Create(
            args.destination_mask, xsize=cls.shape[1],
            ysize=cls.shape[0],
            bands=1, eType=gdal.GDT_Byte,
            options=options)

        gdalnumeric.CopyDatasetInfo(dsm_file, dest_file)
    else:
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)

    dest_band = dest_file.GetRasterBand(1)
    dest_band.WriteArray(cls)


if __name__ == '__main__':
    main(sys.argv[1:])
