import subprocess
import numpy
import json

def getMinMax(json_string):
    j = json.loads(json_string)
    j = j["stats"]["statistic"]
    minX = j[0]["minimum"]
    maxX = j[0]["maximum"]
    minY = j[1]["minimum"]
    maxY = j[1]["maximum"]
    return minX, maxX, minY, maxY

def get_pc_bounds(source_points):
    print("Computing the bounding box for {} point cloud files ...".format(
            len(source_points)))
    minX = numpy.inf
    maxX = - numpy.inf
    minY = numpy.inf
    maxY = - numpy.inf
    pdal_info_template = ["pdal", "info", "--stats", "--dimensions", "X,Y"]
    for i, s in enumerate(source_points):
        pdal_info_args = pdal_info_template + [s]
        out = subprocess.check_output(pdal_info_args)
        tempMinX, tempMaxX, tempMinY, tempMaxY = getMinMax(out)
        if tempMinX < minX:
            minX = tempMinX
        if tempMaxX > maxX:
            maxX = tempMaxX
        if tempMinY < minY:
            minY = tempMinY
        if tempMaxY > maxY:
            maxY = tempMaxY
        if i % 10 == 0:
            print("Iteration {}".format(i))

    return minX, maxX, minY, maxY
    