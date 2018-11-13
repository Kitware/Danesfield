#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import gdal
import logging
import numpy
from osgeo import ogr
from osgeo import osr
import vtk
from vtk.numpy_interface import dataset_adapter as dsa


def main(args):
    parser = argparse.ArgumentParser(
        description='Generate building outlines given a segmentation map')
    parser.add_argument("segmentation", help="Input image file (tif) with labeled buildings")
    parser.add_argument("outlines", help="Output vector file (geojson) with building outlines")
    parser.add_argument('-l', "--label", type=int, nargs="*",
                        help="Label value(s) used for buildings outlines."
                             "If not specified, all values are used.")
    parser.add_argument("--no_decimation", action="store_true",
                        help="Do not decimate the contours")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate results")
    args = parser.parse_args(args)

    # Read the segmentation of buildings
    segmentationReader = vtk.vtkGDALRasterReader()
    segmentationReader.SetFileName(args.segmentation)
    segmentationReader.Update()
    # MAP_PROJECTION is not available for some reason.
    # proj4Srs = segmentationReader.GetOutput().GetFieldData().
    #                GetArray('MAP_PROJECTION').GetValue(0)
    segmentationReaderGdal = gdal.Open(args.segmentation, gdal.GA_ReadOnly)
    segmentationSrsWkt = segmentationReaderGdal.GetProjection()

    segmentationC2p = vtk.vtkCellDataToPointData()
    segmentationC2p.SetInputConnection(segmentationReader.GetOutputPort())
    segmentationC2p.PassCellDataOn()
    segmentationC2p.Update()
    segmentation = segmentationC2p.GetOutput()

    scalarName = segmentation.GetCellData().GetScalars().GetName()
    segmentationNp = dsa.WrapDataObject(segmentation)
    scalars = segmentationNp.CellData[scalarName]
    labels = numpy.unique(scalars)

    if (args.debug):
        segmentationWriter = vtk.vtkXMLImageDataWriter()
        segmentationWriter.SetFileName("segmentation.vti")
        segmentationWriter.SetInputConnection(segmentationReader.GetOutputPort())
        segmentationWriter.Update()

        sb = segmentation.GetBounds()
        print("segmentation bounds: \t{}".format(sb))

    # Extract polygons
    contours = vtk.vtkDiscreteFlyingEdges2D()
    contours.SetInputConnection(segmentationC2p.GetOutputPort())
    if (args.label):
        print("Contouring on {} of {}".format(args.label, labels))
        labels = args.label
    else:
        print("Contouring on {} of {}".format(labels, labels))
    contours.SetNumberOfContours(len(labels))
    for i in range(len(labels)):
        contours.SetValue(i, labels[i])

    if (args.debug):
        contoursWriter = vtk.vtkXMLPolyDataWriter()
        contoursWriter.SetFileName("contours.vtp")
        contoursWriter.SetInputConnection(contours.GetOutputPort())
        contoursWriter.Update()
        contoursData = contours.GetOutput()
        cb = contoursData.GetBounds()
        print("contours bounds: \t{}".format(cb))

    if (not args.no_decimation):
        # combine lines into a polyline
        stripperContours = vtk.vtkStripper()
        stripperContours.SetInputConnection(contours.GetOutputPort())
        stripperContours.SetMaximumLength(3000)

        if (args.debug):
            stripperWriter = vtk.vtkXMLPolyDataWriter()
            stripperWriter.SetFileName("stripper.vtp")
            stripperWriter.SetInputConnection(stripperContours.GetOutputPort())
            stripperWriter.Update()

        # decimate polylines
        decimateContours = vtk.vtkDecimatePolylineFilter()
        decimateContours.SetMaximumError(0.01)
        decimateContours.SetInputConnection(stripperContours.GetOutputPort())

        if (args.debug):
            decimateWriter = vtk.vtkXMLPolyDataWriter()
            decimateWriter.SetFileName("decimate.vtp")
            decimateWriter.SetInputConnection(decimateContours.GetOutputPort())
            decimateWriter.Update()

        contours = decimateContours

    # Create loops
    loopsExtraction = vtk.vtkContourLoopExtraction()
    loopsExtraction.SetInputConnection(contours.GetOutputPort())
    loopsExtraction.Update()
    loops = loopsExtraction.GetOutput()

    if (args.debug):
        loopsWriter = vtk.vtkXMLPolyDataWriter()
        loopsWriter.SetFileName("loops.vtp")
        loopsWriter.SetInputConnection(loopsExtraction.GetOutputPort())
        loopsWriter.Update()

    # Create the output Driver
    outDriver = ogr.GetDriverByName('GeoJSON')

    # spatial reference
    # outSrs = OGRSpatialReference()

    # Create the output GeoJSON
    outDataSource = outDriver.CreateDataSource(args.outlines)
    outSrs = osr.SpatialReference(segmentationSrsWkt)
    outLayer = outDataSource.CreateLayer("buildings", srs=outSrs, geom_type=ogr.wkbPolygon)

    # Get the output Layer's Feature Definition
    featureDefn = outLayer.GetLayerDefn()

    polys = loops.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i in range(idList.GetNumberOfIds()):
            point = loops.GetPoint(idList.GetId(i))
            ring.AddPoint(point[0], point[1])
        idList.Initialize()
        # create a polygon geometry
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        # create a new feature that contains the geometry
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(poly)
        # add the feature to the layer
        outLayer.CreateFeature(outFeature)
        outFeature = None

    # Save and close DataSources
    outDataSource = None


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
