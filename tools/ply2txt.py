#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import open3d as o3d
import numpy as np
import csv
import sys

def convert(args):
	ply = o3d.io.read_point_cloud(args[0])
	points = np.asarray(ply.points)
	with open(args[1], 'w') as f:
		csv.writer(f, delimiter=' ').writerows(points)
