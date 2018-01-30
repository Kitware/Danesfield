import argparse
import gdal
import numpy
import sys

parser = argparse.ArgumentParser(
    description='Convert a multispectral (MSI) image to RGB')
parser.add_argument("msi_image", help="Source MSI image file name")
parser.add_argument("rgb_image", help="Destination RGB image file name")
parser.add_argument('-b', "--byte", action="store_true",
                    help="Enable this option to stretch intensity range and "
                         "convert to a byte image")
parser.add_argument('-p', "--range-percentile", default=0.1, type=float,
                    help="The percent of largest and smallest intensities to "
                         "ignore when computing range for intensity scaling")
args = parser.parse_args()

if args.range_percentile < 0.0 or args.range_percentile >= 50.0:
    print("range percentile must be between 0 and 50")
    sys.exit(1)

# open the source image
msi_image = gdal.Open(args.msi_image, gdal.GA_ReadOnly)
if not msi_image:
    exit(1)

driver = msi_image.GetDriver()
driver_metadata = driver.GetMetadata()
rgb_image = None
if driver_metadata.get(gdal.DCAP_CREATE) != "YES":
    print("Driver {} does not supports Create().".format(driver))
    sys.exit(1)

# Guess which band indices are RGB based on the number of bands
num_bands = msi_image.RasterCount
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
    print("This tool does not support images with {} bands".format(num_bands))
    sys.exit(1)

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
options = ["PHOTOMETRIC=RGB"]
# ensure that space will be reserved for geographic corner coordinates
# (in DMS) to be set later
if (driver.ShortName == "NITF" and not projection):
    options.append("ICORDS=G")
rgb_image = driver.Create(args.rgb_image, xsize=msi_image.RasterXSize,
                          ysize=msi_image.RasterYSize, bands=3, eType=eType,
                          options=options)

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
for out_idx, in_idx in enumerate(rgb_bands, 1):
    in_band = msi_image.GetRasterBand(in_idx)
    out_band = rgb_image.GetRasterBand(out_idx)
    # if not stretching to byte range, just copy the data
    if not args.byte:
        out_band.WriteArray(in_band.ReadAsArray())
        out_band.SetNoDataValue(in_band.GetNoDataValue())
        continue

    in_data = in_band.ReadAsArray()
    # mask out pixels with no valid value
    in_no_data_val = in_band.GetNoDataValue()
    mask = in_data == in_no_data_val
    valid_data = in_data[numpy.logical_not(mask)]

    # robustly find a range for intensity scaling
    min_p = args.range_percentile
    max_p = 100.0 - min_p
    min_val = numpy.percentile(valid_data, min_p)
    max_val = numpy.percentile(valid_data, max_p)
    print("{} band detect range: [{}, {}]".format(band_names[out_idx-1],
                                                  min_val, max_val))

    # clip and scale the data to fit the range 1-255 and cast to byte
    in_data = in_data.clip(min_val, max_val)
    scale = 254.0 / float(max_val - min_val)
    in_data = (in_data - min_val) * scale + 1
    out_data = in_data.astype(numpy.uint8)
    # set the masked out invalid pixels to 0
    out_data[mask] = 0
    out_band.SetNoDataValue(0)
    out_band.WriteArray(out_data)
