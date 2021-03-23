#from lib.ply_np_converter import ply2np, np2ply
import open3d as o3d
import numpy as np
import csv
import sys

def convert(args):
	ply = o3d.io.read_point_cloud(args[0])
	points = np.asarray(ply.points)
	with open(args[1], 'w') as f:
		csv.writer(f, delimiter=' ').writerows(points)


if __name__ == '__main__':
	convert(sys.argv[1:])