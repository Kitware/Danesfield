import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .image_calibration import Image_Calibration as IC
from scipy import stats
from PIL import Image
import pickle
import os


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


def save_output(img, out_path, aux=False):
    if aux:
        out_path = os.path.splitext(out_path)[0] + '.p'
        pickle.dump(img, open(out_path, 'wb'))
    else:
        Image.fromarray(img.astype('uint8')).save(out_path + '.png')
        Image.fromarray(ColorImage(img).astype(
            'uint8')).save(out_path + '_color.png')
