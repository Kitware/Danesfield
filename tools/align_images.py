import os
import gdal
import cv2
import numpy as np
import argparse


def align_images(ref_image, base_image):
    ref_image = ref_image.astype(np.float32)
    base_image = base_image.astype(np.float32)

    height, width, bands = ref_image.shape

    # Second band works but can be changed
    base_band = ref_image[:, :, 2]
    compare_band = base_image[:, :, 2]

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    num_of_iter = 5000
    term_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_of_iter, term_eps)
    (cc, warp_matrix) = cv2.findTransformECC(compare_band,
                                             base_band,
                                             warp_matrix,
                                             warp_mode,
                                             criteria)

    registered_img = ref_image.copy()
    for i in range(8):
        registered_img[:, :, i] = cv2.warpAffine(ref_image[:, :, i], warp_matrix,
                                                 (width, height),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return registered_img


def get_name(image_path):
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    return image_name


def save_image(image, save_path, overwrite):
    if overwrite:
        # Save path is original path if overwrite is True
        dataset = gdal.Open(save_path, gdal.GA_Update)

        for b in range(8):
            dataset.GetRasterBand(b+1).WriteArray(image[:, :, b])
        dataset = None
    else:
        driver = gdal.GetDriverByName('GTiff')
        dst = driver.Create(save_path,
                            image.shape[1],
                            image.shape[0],
                            8,
                            gdal.GDT_Float32)

        for b in range(8):
            dst.GetRasterBand(b+1).WriteArray(image[:, :, b])

        dst = None


def main(args):
    num_images = len(args.image_paths)
    if num_images == 1:
        print('Only one image is used for alignment.')
        print('Exiting alignment script without action.')
    else:
        # If base_image_path not given then select the first image in image path list
        if args.base_image_path is None:
            base_image_path = args.image_paths[0]
        else:
            base_image_path = args.base_image_path

        base_image = np.transpose(gdal.Open(base_image_path).ReadAsArray(), (1, 2, 0))

        for i, image_path in enumerate(args.image_paths):
            print('Alignment progress: {0:2d}/{1:2d}'.format(i+1, num_images))
            dataset = gdal.Open(image_path)
            image = np.transpose(dataset.ReadAsArray(), (1, 2, 0))
            dataset = None  # Close dataset

            aligned_image = align_images(image, base_image)

            # If save_dir not given then overwrite the input images
            if args.save_dir is None:
                save_image(aligned_image, image_path, overwrite=True)
            else:
                image_name = get_name(image_path)
                print(image_name)
                # TODO: Check that name of output aligned image is fine
                save_path = os.path.join(args.save_dir, image_name+'_aligned.tif')
                save_image(aligned_image, save_path, overwrite=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Align images using a simple translation transform.')

    parser.add_argument('--image_paths', nargs='*', required=True,
                        help='List of image paths. (.tif)')

    parser.add_argument('--base_image_path', required=False,
                        help='All other images will be aligned to this image.')

    parser.add_argument('--save_dir', required=False,
                        help='Where aligned images will be saved.'
                             'If not used, then images will be overwritten.')

    args = parser.parse_args()

    main(args)
