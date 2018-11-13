###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .image_calibration import Image_Calibration as IC
from PIL import Image
import gdal
import os
import tarfile
import math


class Dataset_Test(Dataset):
    def __init__(self, data):
        self.data_size = data.shape[0]
        data = torch.from_numpy(data.astype(float))
        self.data = torch.unsqueeze(data, 1).float()

    def __len__(self):
        num_pixels = self.data.shape[0]
        return num_pixels

    def __getitem__(self, index):
        sample = self.data[index]
        return sample


def normalize_data(image, stats):
    image = np.reshape(image, [image.shape[0]*image.shape[1], image.shape[2]])
    image = image.astype(np.float64)
    image -= stats['mean']
    image = image / stats['std']
    return image


def create_dataloader(stacked_image, train_stats, batch_size):
    normalized_data = normalize_data(stacked_image, train_stats)
    dataset = Dataset_Test(normalized_data)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, drop_last=False)
    return loader


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


def calibrate_img(image, info_path):
    cal_obj = IC(image, info_path)
    img = cal_obj.calibrate()
    return img


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
                                gdal.GDT_Byte,
                                options=["COMPRESS=DEFLATE"])

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


def order_images(image_list, info_file_list, viewing_angle=True):
    # Given a set of images
    # Find the metadata for each image
    # Sort images based on viewing angle or illumination angle

    if len(image_list) == 1:
        return image_list, info_file_list

    image_info = []
    for info_file_path in info_file_list:
        if viewing_angle:
            metadata = get_metadata(info_file_path, 'meanOffNadirViewAngle')
        else:
            metadata = get_metadata(info_file_path, 'meanSunEl')
        image_info.append(metadata)

    # Sort images and info files according to metadata
    img_paths = [x for _, x in sorted(zip(image_info, image_list))]
    info_file_list = [x for _, x in sorted(zip(image_info, info_file_list))]

    return img_paths, info_file_list


def read_imd(file_path):
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


def get_metadata(info_file_path, data_name):
    file_ext = os.path.splitext(info_file_path)[1]

    if file_ext == '.tar':
        content = read_tar(info_file_path)
    elif file_ext == '.IMD':
        content = read_imd(info_file_path)

    for i in range(len(content)):
        if data_name in content[i]:
            data = content[i].split()[-1][:-1]

    return float(data)


def coordinate_set_generator(img_path, sub_factor, tile_sz=1000):
    # Creates a list of coordinate pairs that can be parsed as follows:
    # [(x_min, y_min, x_size, y_size), (...), ...]
    dataset = gdal.Open(img_path)
    width, height = dataset.RasterYSize, dataset.RasterXSize
    width, height = width // sub_factor, height // sub_factor

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

            tile = [x0, y0, x1, y1]
            coords.append(tile)
    return coords


def get_train_data_stats(model_path):
    data_stats = {'mean': torch.load(model_path)['data_mean'],
                  'std': torch.load(model_path)['data_std']}
    return data_stats
