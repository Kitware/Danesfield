#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import numpy as np
import tensorflow as tf
import os
import sys
# from Loggers import Logger

from tqdm import tqdm

from danesfield.geon_fitting.tensorflow import roof_type_segmentation

from mpl_toolkits.mplot3d import Axes3D
import pcl
import scipy.spatial
import matplotlib as mpl
# Force 'Agg' backend
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read_txt_pc(filename):
    point_list = []
    with open(filename, 'r') as pc_file:
        for line in pc_file:
            point_coordinate = line.split(' ')
            point_list.append(
                [float(point_coordinate[0]),
                    float(point_coordinate[1]),
                    float(point_coordinate[2])])
    return np.array(point_list)


def save_png_file(point_matrix, label, filename, original_flag=False):
    fig = plt.figure(figsize=(4, 4), dpi=160)
    ax = Axes3D(fig)

    num_category = int(np.max(label))
    for i in range(num_category+1):
        if np.sum(label == i) > 0:
            ax.scatter(point_matrix[label == i, 0],
                       point_matrix[label == i, 1], point_matrix[label == i, 2],
                       color='C{}'.format(i), s=0.2, alpha=1)

    plt.xlim((-2.0, 2.0))  # set the xlim to xmin, xmax
    plt.ylim((-2.0, 2.0))  # set the xlim to xmin, xmax
    ax.set_zlim((-2.0, 2.0))  # set the xlim to xmin, xmax
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def log_string(out_str):
    print(out_str)


def draw_classification_result(ax, point_matrix, label):
    num_category = int(np.max(label))
    for i in range(num_category+1):
        if np.sum(label == i) > 0:
            ax.scatter(point_matrix[label == i, 0],
                       point_matrix[label == i, 1], point_matrix[label == i, 2],
                       color='C{}'.format(i), s=0.2, alpha=1)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU to use [default: GPU 0]')
    parser.add_argument(
        '--model_prefix',
        type=str,
        help='Classification model prefix (e.g. dayton_geon).')
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Classification model directory.')
    parser.add_argument(
        '--model_path',
        type=str,
        help='Full path to classification model.')
    parser.add_argument(
        '--input_pc',
        type=str,
        required=True,
        help='Input p3d point cloud. The point cloud has to be clipped by building mask and \
        smoothed by mls. ')
    parser.add_argument(
        '--output_png',
        type=str,
        default='../segmentation_graph/out.png',
        help='Output png result file.')
    parser.add_argument(
        '--output_txt',
        type=str,
        help='Output txt result file.')
    parser.add_argument(
        '--num_point',
        type=int,
        default=3500,
        help='Point Number [default: 3500]')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch Size during training [default: 32]')
    args = parser.parse_args(args)

    # Accept either combined model directory/prefix or separate directory and prefix
    if args.model_path is None:
        if args.model_dir is None or args.model_prefix is None:
            raise RuntimeError('Model directory and prefix are required')
        args.model_path = os.path.join(args.model_dir, args.model_prefix)
    elif args.model_dir is not None or args.model_prefix is not None:
        raise RuntimeError('The model_dir and model_prefix arguments cannot be specified when '
                           'model_path is specified')

    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)

    NUM_CLASSES = 4

    center_of_mess = np.array([0, 0, 0])

    dataset_point_list = []
    show_point_list = []
    original_point_list = []
    choice_list = []

    point_list = read_txt_pc('{}'.format(args.input_pc))
    point_list = point_list.astype(np.float32)

    center_of_mess = np.mean(point_list, axis=0)
    point_list = point_list - center_of_mess

    cloud = pcl.PointCloud()
    cloud.from_array(point_list)

    remaining_cloud = cloud
    # nr_points = remaining_cloud.size
    tree = remaining_cloud.make_kdtree()

    ec = remaining_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(2)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(600000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    tmp_point_list = []
    tmp_show_point_list = []
    tmp_original_point_list = []
    tmp_choice_list = []

    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)
        # point_label = np.zeros((len(indices),), dtype=np.int32)

        for i, indice in enumerate(indices):
            points[i][0] = remaining_cloud[indice][0]
            points[i][1] = remaining_cloud[indice][1]
            points[i][2] = remaining_cloud[indice][2]

        tmp_original_point_list.append((points+center_of_mess).copy())

        point_data = points.copy()
        max_x = np.amax(point_data[:, 0])
        min_x = np.amin(point_data[:, 0])
        max_y = np.amax(point_data[:, 1])
        min_y = np.amin(point_data[:, 1])
        max_z = np.amax(point_data[:, 2])
        min_z = np.amin(point_data[:, 2])
        max_scale = max((max_x-min_x), (max_y-min_y), (max_z-min_z))
        center = np.mean(point_data, axis=0)
        point_data[:, 0] = (point_data[:, 0]-center[0])/max_scale*2
        point_data[:, 1] = (point_data[:, 1]-center[1])/max_scale*2
        point_data[:, 2] = (point_data[:, 2]-center[2])/max_scale*2

        choice = np.random.choice(points.shape[0], NUM_POINT, replace=True)
        tmp_choice_list.append(choice.copy())

        normed_points = point_data[choice, :]
        show_point = points[choice, :]

        tmp_point_list.append(normed_points)
        tmp_show_point_list.append(show_point)

    dataset_point_list.append(tmp_point_list)
    show_point_list.append(tmp_show_point_list)
    original_point_list.append(tmp_original_point_list)
    choice_list.append(tmp_choice_list)

    with tf.Graph().as_default():
        pointclouds_pl, labels_pl = roof_type_segmentation.placeholder_inputs(
            NUM_POINT, NUM_CLASSES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment
        # the 'batch' parameter for you every time it trains.
        batch = tf.Variable(0)

        print("--- Get model and loss")
        # Get model and loss
        pred, end_points = roof_type_segmentation.get_segmentation_model(
            pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=0.0)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path='{}'.format(args.model_path))

        # Init variables
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'step': batch,
               'end_points': end_points}
        test_pc(sess,
                ops,
                dataset_point_list,
                show_point_list,
                original_point_list,
                center_of_mess,
                BATCH_SIZE,
                NUM_POINT,
                args.output_png,
                args.output_txt)


def get_pc_batch(dataset, start_idx, end_idx, NUM_POINT):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    for i in range(bsize):
        batch_data[i, :] = dataset[start_idx+i]
    return batch_data


def test_pc(sess,
            ops,
            dataset_point_list,
            show_point_list,
            original_point_list,
            center_of_mess,
            BATCH_SIZE,
            NUM_POINT,
            output_png,
            output_txt=None):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    for index in range(len(dataset_point_list)):

        if output_txt:
            fout = open('{}'.format(output_txt), mode='w')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        num_batches = int(len(dataset_point_list[index]) / BATCH_SIZE)+1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(dataset_point_list[index]))
            if start_idx >= end_idx:
                break

            batch_data = get_pc_batch(dataset_point_list[index], start_idx, end_idx, NUM_POINT)
            batch_show_data = get_pc_batch(show_point_list[index], start_idx, end_idx, NUM_POINT)

            aug_data = batch_data
            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['is_training_pl']: is_training}
            step, pred_val = sess.run([ops['step'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 2)

            for i in tqdm(range(end_idx-start_idx)):
                draw_classification_result(ax, batch_show_data[i], pred_val[i, :])
                if output_txt:
                    tmp_original_points = original_point_list[index][start_idx+i]
                    tmp_show_points = show_point_list[index][start_idx+i]
                    tmp_label = np.zeros((tmp_original_points.shape[0],))
                    matrix_distance = scipy.spatial.distance_matrix(
                        tmp_original_points-center_of_mess,
                        tmp_show_points)
                    best_idx = np.argmin(matrix_distance, axis=1)
                    # if np.sum(pred_val[i,:]==2)>200:
                    #    print(start_idx+i)
                    #    print(np.sum(pred_val[i,:]==2))
                    #    print(tmp_original_points.shape)
                    #    print(tmp_show_points.shape)
                    #    print(best_idx[:100])
                    for point_idx in range(tmp_original_points.shape[0]):
                        tmp_label[point_idx] = pred_val[i, best_idx[point_idx]]
                        fout.write('{} {} {} {} {}\n'.format(tmp_original_points[point_idx, 0],
                                                             tmp_original_points[point_idx, 1],
                                                             tmp_original_points[point_idx, 2],
                                                             start_idx+i,
                                                             pred_val[i, best_idx[point_idx]]))
        if output_txt:
            fout.close()

        axisEqual3D(ax)
        plt.savefig(output_png, bbox_inches='tight')
        plt.close()

    return


if __name__ == "__main__":
    main(sys.argv[1:])
