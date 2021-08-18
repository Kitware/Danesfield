#!/usr/bin/env python
'''
Load mesh or point-cloud, and visualize,
applying optional statistical trimming to the latter.
3D viewer: Use typical 3D mouse manipulation gestures. ESC to exit.
'''

import os, argparse, logging
import open3d as o3d
import numpy as np


class Model: # mesh/cloud visualization
    def __init__(self, args):
        path=args.input
        ext = os.path.splitext(args.input)[1].lower()
        min_track_len=args.min_track_len # min len(point3D.point2D_idxs)
        nb_neighbors=args.outlier_neighbors
        std_ratio=args.outlier_std_ratio # for remove_statistical_outlier
        if not path: return
        self.create_window()
        allowed = ['.ply']
        assert ext in allowed, f'allowed formats: {allowed}'
        self.pcd = o3d.io.read_point_cloud(path)
        if nb_neighbors:
            logging.info('remove point outliers: nb_neighbors={}, std_ratio={}'.format(nb_neighbors, std_ratio))
            [self.pcd, _] = self.pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        # o3d.visualization.draw_geometries([pcd])
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
    def add_frames(self, scale=1):
        frames = []
        for img in self.images.values():
            # rotation
            R = qvec2rotmat(img.qvec)
            # translation
            t = img.tvec
            # pose
            t = -R.T.dot(t)
            R = R.T
            # intrinsics
            cam = self.cameras[img.camera_id]
            if cam.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE'):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception('unsupported camera model: {}'.format(cam.model))
            K = np.identity(3)
            K[0,0] = fx
            K[1,1] = fy
            K[0,2] = cx
            K[1,2] = cy
            # create axis, plane and pyramid geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
            frames.extend(cam_model)
        logging.info('add {} frames'.format(len(frames)))
        for i in frames:
            self.vis.add_geometry(i)
    def create_window(self, bkgClr=[0.9,0.9,0.9], ptSize=1.0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(bkgClr)
        opt.point_size = ptSize
    def show(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()
        self.vis.destroy_window()
    def write(self, path): # write out the (trimmed) point cloud
        logging.info('write point cloud: {}'.format(path))
        o3d.io.write_point_cloud(path, self.pcd, compressed=True)


def draw_camera(K, R, t, w, h, scale=1, color=[.8,.2,.8]):
    '''Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    '''
    # intrinsics
    K = K.copy()/scale
    Kinv = np.linalg.inv(K)
    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))
    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5*scale)
    axis.transform(T)
    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]
    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]
    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)
    # pyramid
    points_in_world = [(R.dot(p)+t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # return as list in Open3D format
    return [axis, plane, line_set]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='path/to/input.ply mesh or point cloud')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-m', '--min_track_len', metavar='len', type=int, default=3, help='min len(point3D.point2D_idxs); 0 = skip point filter; default=%(default)s')
    parser.add_argument('-o:n', '--outlier_neighbors', metavar='cnt', type=int, default=20, help='outlier number of neighbors; 0 = skip outlier removal; default=%(default)s')
    parser.add_argument('-o:r', '--outlier_std_ratio', metavar='num', type=float, default=2.0, help='outlier std ratio; default=%(default)s')
    parser.add_argument('-o', '--output', metavar='path', help='path/to/output.ply mesh or point cloud')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    return args


def CLI(argv=None):
    args = parse_args(argv)
    model = Model(args)
    model.show()
    if args.output: model.write(args.output)

if __name__ == '__main__': CLI()
