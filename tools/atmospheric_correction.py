import argparse
import os
import tarfile
import math
import gdal
import subprocess
import numpy as np
import shutil


def read_tar(file_path):
    with tarfile.open(file_path) as tar:

        filenames = tar.getnames()

        # Find IMD file in tar
        IMD_path = [
            filename for filename in filenames if filename[-3:] == 'IMD']
        if len(IMD_path) >= 1:
            IMD_path = IMD_path[0]
        else:
            raise RuntimeError('No IMD file found in tar file.')

        # Read contents of tarfile
        imd_member = tar.getmember(IMD_path)
        imd_file = tar.extractfile(imd_member)

        content = []
        for line in imd_file:
            content.append(line.decode("utf-8"))
        return content


def read_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        return content


def get_info_value(keyword, file_content):
    data = []
    for word in file_content:
        if keyword == 'absCalFactor' or keyword == 'effectiveBandwidth':
            if word.split()[0] == keyword:
                data.append(word.split()[-1][:-1])
        else:
            if word.split()[0] == keyword:
                return word.split()[-1][:-1]
    return data


def calc_sun_earth_dist(time):
    year = int(time[:4])
    month = int(time[5:7])
    day = int(time[8:10])
    UT = int(time[11:13]) + float(time[14:16]) / 60 \
        + float(time[17:26])/3600
    if month <= 2:
        year -= 1
        month += 12
    A = int(year/100)
    B = 2-A + int(A/4)
    JD = int(365.25*(year+4716)) + int(30.6001 *
                                       (month+1)) + day + UT/24 + B - 1524.5
    g = 357.529 + 0.98560028 * (JD - 2451545)
    dES = 1.00014-0.01671 * \
        math.cos(math.radians(g))-0.00014 * \
        math.cos(math.radians(2*g))

    return dES


def get_metadata(info_path):
    file_ext = os.path.splitext(info_path)[1]

    if file_ext == '.tar':
        content = read_tar(info_path)
    elif file_ext == '.IMD':
        content = read_txt(info_path)
    else:
        # Should I raise an error or just skip pair
        # raise RuntimeError('{} is not a .tar or .imd file.'.format(info_path))
        print('{} is not a .tar or .IMD file.'.format(info_path))
        return None

    # Find keyword values in content
    absCalFactor = get_info_value('absCalFactor', content)
    effectiveBandwidth = get_info_value('effectiveBandwidth', content)
    firstLineTime = get_info_value('firstLineTime', content)
    meanSunEl = get_info_value('meanSunEl', content)
    cloudCover = get_info_value('cloudCover', content)

    # Calculate the sun off-nadir angle
    theta = 90 - float(meanSunEl)

    # Calculate sun-earth distance
    dist_ES = calc_sun_earth_dist(firstLineTime)

    metadata = {'absCalFactor': absCalFactor, 'effectiveBandwidth': effectiveBandwidth,
                'firstLineTime': firstLineTime, 'meanSunEl': meanSunEl,
                'cloudCover': cloudCover, 'theta': theta, 'dES': dist_ES}

    return metadata


def tile_coords(height, width, tile_sz):
    x_steps = math.ceil(height / tile_sz)
    y_steps = math.ceil(width / tile_sz)

    coords = []
    for x in range(x_steps):
        for y in range(y_steps):
            x0 = x*tile_sz
            y0 = y*tile_sz
            x1 = (x+1)*tile_sz
            y1 = (y+1)*tile_sz
            if x1 > height:
                x1 = height - x0
            else:
                x1 = tile_sz
            if y1 > width:
                y1 = width - y0
            else:
                y1 = tile_sz
            tile = [x0, x1, y0, y1]
            coords.append(tile)
    return coords


def split_image(image_path, tile_sz=1500):
    # Create temporay folder to save tiles
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'temp')
    os.makedirs(save_dir)

    # Get size of image
    in_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    height = in_dataset.RasterXSize
    width = in_dataset.RasterYSize

    in_dataset = None

    # Get tile coordinates
    coords = tile_coords(height, width, tile_sz)

    # Call gdal translate to crop tiles
    output_image_paths = []
    for i, (x0, x1, y0, y1) in enumerate(coords):
        save_path = save_dir + '/tile_' + str(i) + '.tif'
        call_args = ['gdal_translate', '-srcwin',
                     str(x0), str(y0), str(x1), str(y1),
                     image_path, save_path]
        subprocess.run(call_args)
        output_image_paths.append(save_path)
    return output_image_paths


def get_zero_mask(img):
    # Return occluded pixel mask
    x, y = np.where((img[:, :, 0] == 0) & (
        img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    mask = np.ones((img.shape[0], img.shape[1]))
    mask[x, y] = 0
    return mask


def apply_zero_mask(img, mask):
    # Zero pixels that were occluded from orthorectification
    x, y = np.where(mask == 0)
    for i in range(img.shape[2]):
        img[x, y, i] = 0
    return img


def absolute_radiometric_correction(img, metadata):
    # The absolute radiometric correction follows this equation
    # L = GAIN * DN * abscalfactor / effective bandwidth + OFFSET
    # absCalFactor and effective Bandwidth are in the image metafile (IMD)
    GAIN = [0.905, 0.940, 0.938, 0.962, 0.964, 1.0, 0.961, 0.978,
            1.20, 1.227, 1.199, 1.196, 1.262, 1.314, 1.346, 1.376]
    OFFSET = [-8.604, -5.809, -4.996, -3.646, -3.021, -4.521, -5.522,
              -2.992, -5.546, -2.6, -2.309, -1.676, -0.705, -0.669,
              -0.512, -0.372]

    absCalFactor = metadata['absCalFactor']
    effectiveBandwidth = metadata['effectiveBandwidth']

    corrected_img = img.copy()
    for i in range(img.shape[2]):
        corrected_img[:, :, i] = GAIN[i] * img[:, :, i] * \
            (float(absCalFactor[i]) /
             float(effectiveBandwidth[i])) + OFFSET[i]

    return corrected_img


def top_of_atmosphere_reflectance(img, metadata):
    spectral_irradiance = [1757.89, 2004.61, 1830.18, 1712.07, 1535.33, 1348.08, 1055.94, 858.77,
                           479.02, 263.797, 225.28, 197.55, 90.41, 85.06, 76.95, 68.10]

    theta = metadata['theta']
    D = metadata['dES']

    corr_img = img.copy()
    for i in range(img.shape[2]):
        corr_img[:, :, i] = img[:, :, i] * D**2 * math.pi / \
            (spectral_irradiance[i] *
             math.cos(math.radians(theta)))

    return corr_img


def transfer_metadata(corrected_image_path, original_image_path):
    corrected_dataset = gdal.Open(corrected_image_path, gdal.GA_Update)
    original_dataset = gdal.Open(original_image_path, gdal.GA_ReadOnly)

    corrected_dataset.SetMetadata(original_dataset.GetMetadata())
    rpcs = original_dataset.GetMetadata('RPC')
    corrected_dataset.SetMetadata(rpcs, 'RPC')
    corrected_dataset.SetGeoTransform(original_dataset.GetGeoTransform())
    corrected_dataset.SetProjection(original_dataset.GetProjection())

    corrected_dataset = None
    original_dataset = None


def calibrate(image_path, metadata, save_dir, tile=False):
    # Load image
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    image = np.transpose(dataset.ReadAsArray(), (1, 2, 0)).astype(float)
    dataset = None

    # Check if images has zero pixels
    zero_mask = get_zero_mask(image)

    # Absolute radiometic correction
    arc_img = absolute_radiometric_correction(image, metadata)
    del image

    # Conversion to TOA reflectance
    toa_img = top_of_atmosphere_reflectance(arc_img, metadata)
    del arc_img

    # Zero out occluded pixels
    toa_img = apply_zero_mask(toa_img, zero_mask)

    # Use gdal_translate to change the datatype
    if tile:
        out_path = os.path.splitext(image_path)[0] + '_F.tif'
    else:
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]
        out_path = os.path.join(save_dir, image_name+'_cal.tif')  # May need to change name

    call_args = ['gdal_translate', '-ot',
                 'float32', image_path, out_path]
    subprocess.run(call_args)

    nBands = toa_img.shape[2]

    # Add reflectance values to output dataset
    out_dataset = gdal.Open(out_path, gdal.GA_Update)
    for i in range(nBands):
        out_dataset.GetRasterBand(i+1).WriteArray(toa_img[:, :, i])
    out_dataset = None

    transfer_metadata(out_path, image_path)


def merge_tiles(image_path, tile_paths, save_dir):
    # Find temp folder with image tiles
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    out_path = os.path.join(save_dir, image_name+'_cal.tif')  # May need to change name

    call_args = ['gdal_merge.py', '-o', out_path]

    for tile_path in tile_paths:
        call_args.append(tile_path[:-4]+'_F.tif')  # Need to get corrected tiles
    subprocess.run(call_args)

    transfer_metadata(out_path, image_path)

    # Delete all of the old files
    tile_dir = os.path.split(tile_paths[0])[0]
    shutil.rmtree(tile_dir)


def main(args):
    # For each pair of images and info files convert pixel values to reflectance
    size_thresh = 5000 ** 2 * 8
    for image_path, info_path in zip(args.image_paths, args.info_paths):
        # Read metadata from image file
        metadata = get_metadata(info_path)

        if metadata is None:
            print('Skipping {}'.format(image_path))
            continue

        # Check size of image
        dataset = gdal.Open(image_path)
        height = dataset.RasterXSize
        width = dataset.RasterYSize
        nBands = dataset.RasterCount

        # If image too large break into smaller tiles
        if size_thresh < height * width * nBands:
            tile_paths = split_image(image_path, tile_sz=1500)
            # For each tile calibrate
            for tile_path in tile_paths:
                calibrate(tile_path, metadata, None, tile=True)
            merge_tiles(image_path, tile_paths, args.save_dir)
        else:
            calibrate(image_path, metadata, args.save_dir, tile=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert digital values of raw satellite image to reflectance. \
                     Input image paths should be orthorectified but are not required.')

    parser.add_argument('--image_paths', nargs='*', required=True,
                        help='List of image paths. (.tif)')

    parser.add_argument('--info_paths', nargs='*', required=True,
                        help='List of metadata file paths for image files. (.tar or .imd)')

    parser.add_argument('--save_dir', required=True,
                        help='Directory where calibrated images will be saved.')

    args = parser.parse_args()

    main(args)
