#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import argparse
import logging
from danesfield.segmentation.building import inception_v1
import cv2

import gdal
import gdalnumeric


def combine_imagery(img_data, depth_data, NDVI_data):
    """Combine the various source of imagery into a multi-channel image
    """
    if not NDVI_data:
        img = np.zeros([1, img_data.shape[0], img_data.shape[1], 4], dtype=np.float32)
    else:
        img = np.zeros([1, img_data.shape[0], img_data.shape[1], 5], dtype=np.float32)
    float_img = (np.float32(img_data.copy())/255.0-0.5)*2.0
    float_depth = (np.float32(depth_data.copy())/255.0 - 0.5)*2.0

    img[0, :, :, :3] = float_img
    img[0, :, :, 3] = float_depth

    if NDVI_data:
        float_NDVI = (np.float32(NDVI_data.copy())/255.0-0.5)*2.0
        img[0, :, :, 4] = float_NDVI

    return img


# Read input image data
def read_data_test(args):
    sourceRGB = gdal.Open(args.rgb_image, gdal.GA_ReadOnly)
    sourceMSI = gdal.Open(args.msi_image, gdal.GA_ReadOnly)
    sourceDSM = gdal.Open(args.dsm, gdal.GA_ReadOnly)
    sourceDTM = gdal.Open(args.dtm, gdal.GA_ReadOnly)

    # get RGB image
    color_image = sourceRGB.ReadAsArray()
    color_image = np.transpose(color_image, (1, 2, 0))
    color_image = color_image[:, :, [2, 1, 0]]

    # get DHM
    dsm_image = sourceDSM.ReadAsArray()
    dtm_image = sourceDTM.ReadAsArray()
    dhm = dsm_image - dtm_image
    dhm[dsm_image < -1000] = 0
    dhm[dhm < 0] = 0
    dhm[dhm > 40] = 40
    dhm = dhm/40*255

    # get NDVI
    msi_image = sourceMSI.ReadAsArray()
    msi_image = np.transpose(msi_image, (1, 2, 0))
    red_map = msi_image[:, :, 4].astype(np.float)
    nir_map = msi_image[:, :, 6].astype(np.float)
    NDVI = (nir_map-red_map)/(nir_map+red_map+1e-7)
    NDVI[NDVI < 0] = 0
    NDVI[NDVI > 1] = 1
    NDVI = (NDVI*255).astype(np.uint8)
    return color_image, dhm, NDVI, sourceDSM


def main(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
            '--rgb_image', default='ortho_rgb_D2_07NOV16WV031100016NOV07165023.tif',
            help='File path to orthorectified image')
    parser.add_argument(
            '--msi_image', default='ortho_ps_D2_07NOV16WV031100016NOV07165023.tif',
            help='File path to MSI image (8 channels)')
    parser.add_argument(
            '--dsm', default='/dvmm-filer2/projects/Core3D/D2_WPAFB/DSMs/D2_P3D_DSM.tif',
            help='File path to DSM')
    parser.add_argument(
            '--dtm', default='/dvmm-filer2/projects/Core3D/D2_WPAFB/DTMs/D2_DTM.tif',
            help='File path to DTM')
    parser.add_argument('--model_path', default='../Inception_model/Dayton_best', help='')
    parser.add_argument('--save_dir', default='../data/thres_img/', help='folder to save result')
    parser.add_argument("--no_NDVI", action="store_true", help="use NDVI or not")
    parser.add_argument("--output_tif", action="store_true", help="save to geotif or not")
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    np.set_printoptions(precision=2)

    args = parser.parse_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    training_stride = 4

    test_img_data, test_dhm, test_NDVI, dsm = read_data_test(args)
    # resize image
    test_img_data = cv2.resize(test_img_data, (2048, 2048))
    test_dhm_data = cv2.resize(test_dhm, (test_img_data.shape[1], test_img_data.shape[0]))
    if not args.no_NDVI:
        test_NDVI_data = cv2.resize(test_NDVI, (test_img_data.shape[1], test_img_data.shape[0]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Build the model
    if not args.no_NDVI:
        img_place_holder = tf.placeholder(tf.float32, [None, None, None, 5])
    else:
        img_place_holder = tf.placeholder(tf.float32, [None, None, None, 4])

    with tf.contrib.slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, _ = inception_v1.inception_v1(img_place_holder, no_bn=False)
        test_logits, _ = inception_v1.inception_v1(
                img_place_holder, reuse=True, is_training=False, no_bn=False)

    restorer = tf.train.Saver()
    restorer.restore(sess, '{}'.format(args.model_path))

    # show with different threshold
    img = combine_imagery(test_img_data, test_dhm_data, test_NDVI_data)
    tmp_logits = sess.run([test_logits], feed_dict={img_place_holder: img})

    pred = tmp_logits[0]  # np.argmax(tmp_logits[0],axis = 1)
    pred = pred.reshape(int(test_img_data.shape[0]/training_stride),
                        int(test_img_data.shape[1]/training_stride))

    resize_depth_data = cv2.resize(
            test_dhm_data, (pred.shape[0], pred.shape[1]),
            interpolation=cv2.INTER_NEAREST)

    predict_img = np.zeros((
            int(test_img_data.shape[0]/training_stride),
            int(test_img_data.shape[1]/training_stride)), dtype=np.uint8)

    predict_img[pred > 0.0] = 255
    pred[resize_depth_data > 200] = 1.0
    predict_img[resize_depth_data > 200] = 255

    try:
        os.stat(args.save_dir)
    except OSError:
        os.makedirs(args.save_dir)

    cv2.imwrite('{}/predict.png'.format(args.save_dir), predict_img)

    if args.output_tif:
        driver = dsm.GetDriver()
        destImage = driver.Create(
            '{}/CU_CLS_Float.tif'.format(args.save_dir),
            xsize=dsm.RasterXSize,
            ysize=dsm.RasterYSize,
            bands=1, eType=gdal.GDT_Float32)

        gdalnumeric.CopyDatasetInfo(dsm, destImage)

        full_predict = cv2.resize(
                pred, (dsm.RasterXSize, dsm.RasterYSize),
                interpolation=cv2.INTER_NEAREST)

        destBand = destImage.GetRasterBand(1)
        destBand.WriteArray(full_predict)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
