import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .image_calibration import Image_Calibration as IC
from scipy import stats
from PIL import Image
import gdal


class TestDataLoader(Dataset):
    def __init__(self, img):
        img = np.reshape(
            img, [img.shape[0]*img.shape[1], 1, img.shape[2]])
        self.signals = torch.from_numpy(img).float()

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, index):
        signal = self.signals[index, :, :]
        return signal


def getTestDataLoader(img, batch_size):
    loader = TestDataLoader(img)
    return DataLoader(loader, shuffle=False, num_workers=4,
                      drop_last=False, batch_size=batch_size)


def ColorImage(img):
    color_image = np.zeros((img.shape[0], img.shape[1], 3))
    color = [[0, 0, 0], [78, 78, 79], [161, 161, 163], [255, 186, 250],
             [16, 119, 14], [105, 249, 102], [
                 240, 252, 103], [214, 0, 60],
             [153, 120, 59], [179, 74, 239], [124, 161, 255], [255, 255, 255]]

    for c in range(0, 12):
        print("Status: {0:2d}/{1:2d}".format(c+1, 12), end="\r")
        x, y = np.where(c == img)
        color_image[x, y, :] = np.array(color[c])
    print(" "*100, end="\r")
    return color_image


def calibrate_img(img, imd_path):
    cal_obj = IC(img, imd_path)
    img = cal_obj.calibrate()
    return img


def create_mode_img(stacked_img, save_path):
    print("Generating mode image ...")

    # Get modes for each pixel
    mode_matrix, _ = stats.mode(stacked_img, axis=2)
    mode_matrix = np.squeeze(mode_matrix)

    # Find where mode is 0 and check if there is another value
    x, y = np.where(mode_matrix == 0)

    for i in range(len(x)):
        # Check if only zeros for pixel
        if np.any(stacked_img[x[i], y[i], :]):
            pixel = stacked_img[x[i], y[i], :]
            j = np.where(pixel != 0)
            non_zero_vals = pixel[j]
            pixel_mode, _ = stats.mode(non_zero_vals)
            if len(pixel_mode) > 1:
                pixel_mode = pixel_mode[0]
            mode_matrix[x[i], y[i]] = pixel_mode

    # Save results
    print("Saving mode image")
    pil_image = Image.fromarray(mode_matrix.astype(np.uint8))
    pil_image.save(save_path+'mode.tif')

    color_image = Image.fromarray(
        ColorImage(mode_matrix).astype('uint8'))
    color_image.save(save_path+'mode_color.png')


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


def save_output(img, out_path):
    img = img.astype(int)

    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(out_path,
                                img.shape[1],
                                img.shape[0],
                                1,
                                gdal.GDT_Byte)

    out_dataset.GetRasterBand(1).WriteArray(img)
    out_dataset = None

    Image.fromarray(ColorImage(img).astype(
        'uint8')).save(out_path + '_color.png')


class Combine_Result(object):
    def __init__(self, merge_type):
        self.merge_type = merge_type
        self.update_count = 0

        if self.merge_type != 'max_prob':
            raise RuntimeError('Other type of merge {} is not added yet.'.format(self.merge_type))

    def update(self, result):
        if self.merge_type == 'max_prob':
            if self.update_count == 0:
                self.merge_result = result
            else:
                self.merge_result += result
                self.update_count += 1

    def call(self):
        if self.merge_type == 'max_prob':
            class_probs = self.merge_result[:, :, 1:]
            mean_image = np.argmax(class_probs, axis=2) + 1
            return mean_image
