import math
import tarfile
import os
import numpy as np


def read_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        return content


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


class Image_Calibration(object):
    def __init__(self, image, imd_path, norm=False):
        super(Image_Calibration).__init__()
        self.image = image
        self.imd_path = imd_path
        self.norm = norm

    def calibrate(self):
        # Get necessary metadata
        metadata = self._get_metadata(self.imd_path)

        # Get orthorectification mask
        # zero_mask = self._get_zero_mask(self.image)

        # Absolute radiometic correction
        arc_img = self._absolute_radiometric_correction(self.image, metadata)

        # Top of atmosphere reflectance
        toa_img = self._top_of_atmosphere_reflectance(arc_img, metadata)

        # if self.norm:
        #     toa_img = self._normalize_image(toa_img)

        # Zero out occluded pixels
        # toa_img = self._apply_mask(toa_img, zero_mask)

        return toa_img

    def _get_metadata(self, imd_path):
        imd_ext = os.path.splitext(imd_path)[1]
        if imd_ext == '.IMD':
            content = read_txt(imd_path)
        elif imd_ext == '.tar':
            content = read_tar(imd_path)
        else:
            raise RuntimeError(
                'IMD file extension {} is not supported.'.format(imd_ext))

        # Get calibration parameters
        absCalFactor = []
        effectiveBandwidth = []
        firstLineTime = []
        meanSunEl = []
        cloudCover = []
        for i in range(len(content)):
            if 'absCalFactor' in content[i]:
                absCalFactor.append(content[i].split()[-1][:-1])
            elif 'effectiveBandwidth' in content[i]:
                effectiveBandwidth.append(content[i].split()[-1][:-1])
            elif 'firstLineTime' in content[i]:
                firstLineTime.append(content[i].split()[-1][:-1])
            elif 'meanSunEl' in content[i]:
                meanSunEl.append(content[i].split()[-1][:-1])
            elif 'cloudCover' in content[i]:
                cloudCover.append(content[i].split()[-1][:-1])

        # Calculate theta
        theta = 90 - float(meanSunEl[0])

        # Calculate sun-earth distance
        firstLineTime = firstLineTime[0]
        year = int(firstLineTime[:4])
        month = int(firstLineTime[5:7])
        day = int(firstLineTime[8:10])
        UT = int(firstLineTime[11:13]) + float(firstLineTime[14:16]) / 60 \
            + float(firstLineTime[17:26])/3600
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

        metadata = {'absCalFactor': absCalFactor, 'effectiveBandwidth': effectiveBandwidth,
                    'firstLineTime': firstLineTime, 'meanSunEl': meanSunEl,
                    'cloudCover': cloudCover, 'theta': theta, 'dES': dES}
        return metadata

    def _get_zero_mask(self, img):
        # Zero mask equal to 1 where there are 0 values
        img_zero = img.any(axis=-1)
        x, y = np.where(img_zero == False)  # noqa: E712
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask[x, y] = 1
        return mask

    def _absolute_radiometric_correction(self, img, metadata):
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

    def _top_of_atmosphere_reflectance(self, img, metadata):
        spectral_irradiance = [1757.89, 2004.61, 1830.18, 1712.07, 1535.33,
                               1348.08, 1055.94, 858.77, 479.02, 263.797,
                               225.28, 197.55, 90.41, 85.06, 76.95, 68.10]

        theta = metadata['theta']
        D = metadata['dES']

        corr_img = img.copy()
        for i in range(img.shape[2]):
            corr_img[:, :, i] = img[:, :, i] * D**2 * math.pi / \
                (spectral_irradiance[i] *
                 math.cos(math.radians(theta)))

        return corr_img

    def _normalize_image(self, img):
        # Image range becomes 0-1
        img /= img.max()
        return img

    def _apply_mask(self, img, mask):
        # Zero pixels that were occluded from orthorectification
        x, y = np.where(mask == 1)
        for i in range(img.shape[2]):
            img[x, y, i] = 0
        return img
