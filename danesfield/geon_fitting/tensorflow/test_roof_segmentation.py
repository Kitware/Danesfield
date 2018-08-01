import argparse
import numpy as np
import tensorflow as tf
import os
from Loggers import Logger

from tqdm import tqdm

import roof_type_segmentation

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pcl
import scipy.spatial
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='GPU to use [default: GPU 0]')
parser.add_argument(
        '--dataset', default='dayton',
        help='GPU to use [default: GPU 0]')
parser.add_argument(
        '--model', default='roof_segmentation',
        help='Model name [default: model]')
parser.add_argument(
        '--root_dir', default='../data/',
        help='Log dir [default: log]')
parser.add_argument(
        '--log_dir', default='../segmentation_log/',
        help='Log dir [default: log]')
parser.add_argument(
        '--model_dir', default='../segmentation_model/',
        help='Log dir [default: log]')
parser.add_argument(
        '--num_point', type=int,
        default=3500, help='Point Number [default: 3500]')
parser.add_argument(
        '--max_epoch', type=int, default=100,
        help='Epoch to run [default: 201]')
parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch Size during training [default: 32]')
parser.add_argument(
        '--learning_rate', type=float,
        default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument(
        '--momentum', type=float, default=0.9,
        help='Initial learning rate [default: 0.9]')
parser.add_argument(
        '--optimizer', default='adam',
        help='adam or momentum [default: adam]')
parser.add_argument(
        '--decay_step', type=int, default=200000,
        help='Decay step for lr decay [default: 200000]')
parser.add_argument(
        '--decay_rate', type=float, default=0.7,
        help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--appendix', type=str, default="", help='')
parser.add_argument(
        "--flip", action="store_true",
        help="use feature mean to initialize or not")
parser.add_argument(
        "--da", action="store_true",
        help="use feature mean to initialize or not")
parser.add_argument(
        "--scale", action="store_true",
        help="use feature mean to initialize or not")
parser.add_argument(
        "--shift", action="store_true",
        help="use feature mean to initialize or not")
args = parser.parse_args()


def read_txt_pc(filename):
    point_list = []
    with open(filename, 'r') as pc_file:
        for line in pc_file:
            point_coordinate = line.split(',')
            point_list.append(
                [float(point_coordinate[0]),
                    float(point_coordinate[1]),
                    float(point_coordinate[2])])
    return np.array(point_list)


BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
GPU_INDEX = args.gpu_id
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)

appendix = args.appendix
if args.flip:
    appendix = appendix + '_fl'
if args.da:
    appendix = appendix + '_da'
if args.scale:
    appendix = appendix + '_sc'
if args.shift:
    appendix = appendix + '_shift'

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir + args.dataset)
logger = Logger(args.log_dir + args.dataset + '/' + args.dataset + appendix)

if not os.path.isdir('{}{}/{}{}/'.format(
        args.model_dir, args.dataset, args.dataset, appendix)):
    os.makedirs('{}{}/{}{}/'.format(
        args.model_dir, args.dataset,
        args.dataset, appendix))

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 4

dataset_list = ['D1', 'D2', 'D3', 'D4']

center_of_mess = np.array([0, 0, 0])

dataset_point_list = []
show_point_list = []
original_point_list = []
choice_list = []

for dataset in dataset_list:
    point_list = read_txt_pc('../../core3d-columbia/data/{}_mls_building.txt'.format(dataset))
    print('opened')
    point_list = point_list.astype(np.float32)

    center_of_mess = np.mean(point_list, axis=0)
    point_list = point_list - center_of_mess
    print(len(point_list))

    cloud = pcl.PointCloud()
    cloud.from_array(point_list)

    remaining_cloud = cloud
    nr_points = remaining_cloud.size
    tree = remaining_cloud.make_kdtree()

    ec = remaining_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(2)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(550000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    tmp_point_list = []
    tmp_show_point_list = []
    tmp_original_point_list = []
    tmp_choice_list = []

    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)
        point_label = np.zeros((len(indices),), dtype=np.int32)

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


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


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


def main():
    with tf.Graph().as_default():
        pointclouds_pl, labels_pl = roof_type_segmentation.placeholder_inputs(
                NUM_POINT, NUM_CLASSES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment
        # the 'batch' parameter for you every time it trains.
        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)

        print("--- Get model and loss")
        # Get model and loss
        pred, end_points = roof_type_segmentation.get_segmentation_model(
                pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path='{}/{}/{}{}/'.format(args.model_dir, args.dataset,
                      args.dataset, appendix)+"epoch_{:03d}.ckpt".format(95))

        # Init variables
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'step': batch,
               'end_points': end_points}
        test_pc(sess, ops)


def get_pc_batch(dataset, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    for i in range(bsize):
        batch_data[i, :] = dataset[start_idx+i]
    return batch_data


def test_pc(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    for index in range(len(dataset_point_list)):
        print('Process AOI #{}'.format(index))

        try:
            os.stat('../out_las/')
        except OSError:
            os.makedirs('../out_las/')

        fout = open('../out_las/{}_with_label.txt'.format(dataset_list[index]), mode='w')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        num_batches = int(len(dataset_point_list[index]) / BATCH_SIZE)+1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(dataset_point_list[index]))
            if start_idx >= end_idx:
                break

            batch_data = get_pc_batch(dataset_point_list[index], start_idx, end_idx)
            batch_show_data = get_pc_batch(show_point_list[index], start_idx, end_idx)

            aug_data = batch_data
            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['is_training_pl']: is_training}
            step, pred_val = sess.run([ops['step'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 2)

            for i in tqdm(range(end_idx-start_idx)):
                draw_classification_result(ax, batch_show_data[i], pred_val[i, :])
                tmp_original_points = original_point_list[index][start_idx+i]
                tmp_show_points = show_point_list[index][start_idx+i]
                tmp_label = np.zeros((tmp_original_points.shape[0],))
                matrix_distance = scipy.spatial.distance_matrix(tmp_original_points,
                                                                tmp_show_points)
                best_idx = np.argmax(matrix_distance, axis=1)

                for point_idx in range(tmp_original_points.shape[0]):
                    tmp_label[point_idx] = pred_val[i, best_idx[point_idx]]
                    fout.write('{},{},{},{}\n'.format(tmp_original_points[point_idx, 0],
                                                      tmp_original_points[point_idx, 1],
                                                      tmp_original_points[point_idx, 2],
                                                      pred_val[i, best_idx[point_idx]]))
        fout.close()

        try:
            os.stat('../segmentation_graph/{}{}/'.format(args.dataset, appendix))
        except OSError:
            os.makedirs('../segmentation_graph/{}{}/'.format(args.dataset, appendix))

        axisEqual3D(ax)
        plt.savefig('../segmentation_graph/{}{}/{}_test.png'.format(
            args.dataset, appendix, dataset_list[index]), bbox_inches='tight')
        plt.close()

    return


if __name__ == "__main__":
    main()
