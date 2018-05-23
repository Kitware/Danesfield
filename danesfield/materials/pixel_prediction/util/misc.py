import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .image_calibration import Image_Calibration as IC
from scipy import stats
from PIL import Image


class TestDataLoader(Dataset):
    def __init__(self, data):
        self.signals = data

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, index):
        signal = self.signals[index, :, :]
        return signal


def image_to_tensor(img):
    img = np.reshape(
        img, [img.shape[0]*img.shape[1], 1, img.shape[2]])
    interp_points = torch.from_numpy(img).float()
    return interp_points


def getTestDataLoader(img, batch_size):
    loader = TestDataLoader(img)
    return DataLoader(loader, shuffle=False, num_workers=4,
                      drop_last=False, batch_size=batch_size)


def ColorImage(img):
    color_image = np.zeros((img.shape[0], img.shape[1], 3))
    for c in range(0, 12):
        print("Status: {0:2d}/{1:2d}".format(c+1, 12), end="\r")
        x, y = np.where(c == img)
        if c == 0:
            # Unclassified
            color_image[x, y, :] = np.array([0, 0, 0])
        elif c == 1:
            # Asphalt
            color_image[x, y, :] = np.array([78, 78, 79])
        elif c == 2:
            # Concrete
            color_image[x, y, :] = np.array([161, 161, 163])
        elif c == 3:
            # Glass
            color_image[x, y, :] = np.array([255, 186, 250])
        elif c == 4:
            # Tree
            color_image[x, y, :] = np.array([16, 119, 14])
        elif c == 5:
            # Non tree veg
            color_image[x, y, :] = np.array([105, 249, 102])
        elif c == 6:
            # Metal
            color_image[x, y, :] = np.array([240, 252, 103])
        elif c == 7:
            # Red Ceramic
            color_image[x, y, :] = np.array([214, 0, 60])
        elif c == 8:
            # Soil
            color_image[x, y, :] = np.array([153, 120, 59])
        elif c == 9:
            # Solar panel
            color_image[x, y, :] = np.array([179, 74, 239])
        elif c == 10:
            # Water
            color_image[x, y, :] = np.array([124, 161, 255])
        elif c == 11:
            # Polymer
            color_image[x, y, :] = np.array([255, 255, 255])
    print(" "*100, end="\r")
    return color_image


def get_mask(img):
    zero_mask = np.ones((img.shape[0], img.shape[1]))
    x, y = np.where(~img.any(axis=2))
    zero_mask[x, y] = 0
    return zero_mask


def apply_mask(result, mask):
    x, y = np.where(mask == 0)
    result[x, y] = 0
    return result


def calibrate_img(img, imd_path):
    norm_obj = IC(img, imd_path, 'MSI')
    img = norm_obj.norm()
    return img


def create_mode_img(stacked_img, save_path):
    print("Generating mode image...")

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
