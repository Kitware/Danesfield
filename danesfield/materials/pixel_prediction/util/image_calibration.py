import numpy as np
import math


def BandStats(img, img_name):
    print(img_name)
    # img_float = img.astype(float)
    img_float = img
    for i in range(img.shape[2]):
        print("Band {0:1.0f}: Min {1:0.3f}, Max {2:0.3f}, Mean {3:}, \
         std {4:0.3f}".format(i, img_float[:, :, i].min(),
                              img_float[:, :, i].max(),
                              np.mean(img_float[:, :, i]),
                              np.std(img_float[:, :, i])))


class Image_Calibration():
    def __init__(self, raw_img, imd_path, type, zero_one=False):
        super(Image_Calibration).__init__()
        self.imd_path = imd_path
        self.sat_type = type
        self.zero_one = zero_one

        # Load image
        self.raw_img = raw_img.astype(float)

        # Get metadata
        self.metadata = self.get_metadata()
        theta, dES = self.calc_correction_parameters()

        rad_corr_img = self.Absolute_Radiometric_Correction()
        self.norm_img = self.TOA(rad_corr_img, dES, theta)
        if(self.norm_img.min() < 0):
            self.norm_img -= self.norm_img.min()

    def Absolute_Radiometric_Correction(self):
        # The absolute radiometric correction follows this equation
        # L = GAIN * DN * abscalfactor / effective bandwidth + OFFSET
        # absCalFactor and effective Bandwidth are in the image metafile (IMD)
        # Create matrices to make this faster
        GAIN = [0.863, 0.905, 0.907, 0.938, 0.945, 0.98, 0.982, 0.954,
                1.160, 1.184, 1.173, 1.187, 1.286, 1.336, 1.340, 1.392]
        OFFSET = [-7.154, -4.189, -3.287, -1.816, -1.350, -2.617, -3.752,
                  -1.507, -4.479, -2.248, -1.806, -1.507, -0.622, -0.605,
                  -0.423, -0.302]
        absCalFactor, effectiveBandwidth = self.metadata[0], self.metadata[1]

        corrected_img = self.raw_img.copy()
        for i in range(self.raw_img.shape[2]):
            if(self.sat_type == 'SWIR'):
                corrected_img[:, :, i] = GAIN[i+8] * self.raw_img[:, :, i] * \
                    (float(absCalFactor[i]) /
                     float(effectiveBandwidth[i])) + OFFSET[i+8]
            else:
                corrected_img[:, :, i] = GAIN[i] * self.raw_img[:, :, i] * \
                    (float(absCalFactor[i]) /
                     float(effectiveBandwidth[i])) + OFFSET[i]

        return corrected_img

    def TOA(self, img, D, theta):
        spectral_irradiance = [1757.89, 2004.61, 1830.18, 1712.07, 1535.33,
                               1348.08, 1055.94, 858.77, 479.02, 263.797,
                               225.28, 197.55, 90.41, 85.06, 76.95, 68.10]

        corr_img = img.copy()
        for i in range(img.shape[2]):
            if(self.sat_type == 'SWIR'):
                corr_img[:, :, i] = img[:, :, i] * D**2 * math.pi / \
                    (spectral_irradiance[i+8] *
                     math.cos(math.radians(theta)))
            else:
                corr_img[:, :, i] = img[:, :, i] * D**2 * math.pi / \
                    (spectral_irradiance[i] *
                     math.cos(math.radians(theta)))

        return corr_img

    def get_metadata(self):
        # Find the corresponding IMD file
        # Parse the file
        # Get the absCalFactor, effectiveBandwidth for each band
        # Get the earth-sun distance (first line time),
        # solar irradiance (have already),
        # solar zenith angle (meanSunEl)
        if self.imd_path is None:
            print("IMD file was not found!")
        content = self.read_txt(self.imd_path)
        params = self.get_parameters(content)

        absCalFactor, effectiveBandwidth, firstLineTime, meanSunEl = params[
            0], params[1], params[2], params[3]
        return absCalFactor, effectiveBandwidth, firstLineTime, meanSunEl

    def get_parameters(self, content):
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

        return absCalFactor, effectiveBandwidth, firstLineTime, meanSunEl, \
            cloudCover

    def read_txt(self, path):
        with open(path, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            # content = [i.split() for i in content]
            return content

    def calc_correction_parameters(self):
        # absCalFactor, effectiveBandwidth, firstLineTime,
        # meanSunEl = self.metadata
        _, _, firstLineTime, meanSunEl = self.metadata

        # Calculate theta
        theta = 90 - float(meanSunEl[0])

        # Calculate sun-earth distance
        firstLineTime = firstLineTime[0]
        year = int(firstLineTime[:4])
        month = int(firstLineTime[5:7])
        day = int(firstLineTime[8:10])
        UT = int(firstLineTime[11:13]) + \
            float(firstLineTime[14:16])/60 + \
            float(firstLineTime[17:26])/3600
        if month <= 2:
            year -= 1
            month += 12
        A = int(year/100)
        B = 2-A + int(A/4)
        JD = int(365.25*(year+4716)) + int(30.6001 * (month+1)) + \
            day + UT/24 + B - 1524.5
        g = 357.529 + 0.98560028*(JD - 2451545)
        dES = 1.00014-0.01671 * \
            math.cos(math.radians(g))-0.00014 * \
            math.cos(math.radians(2*g))

        return theta, dES

    def norm(self):
        if self.zero_one is False:
            return self.norm_img
        else:
            min_num = self.norm_img.min()
            if min_num < 0:
                self.norm_img -= min_num
            else:
                self.norm_img += min_num
            max_num = self.norm_img.max()
            self.norm_img /= max_num
            return self.norm_img
