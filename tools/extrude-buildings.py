#!/usr/bin/env python

import argparse
import numpy
import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Demonstrate generation of extruded objects from a segmentation map, where
# the extrusion is trimmed by a terrain surface.
parser = argparse.ArgumentParser(
    description='Generate extruded buildings given a segmentation map, DSM and DTM')
parser.add_argument("segmentation", help="Image with labeled buildings")
parser.add_argument("dsm", help="Digital surface model (DSM)")
parser.add_argument("dtm", help="Digital terain model (DTM)")
parser.add_argument("destination", help="Extruded buildings polygonal file (.vtp)")
parser.add_argument('-l', "--label", type=int, nargs="*",
                    help="Label value(s) used for buildings outlines."
                         "If not specified, [6, 17] (buildings, roads) are used.")
parser.add_argument("--no_decimation", action="store_true",
                    help="Do not decimate the contours")
parser.add_argument("--no_render", action="store_true",
                    help="Do not render")
parser.add_argument("--debug", action="store_true",
                    help="Save intermediate results")
args = parser.parse_args()

#!/usr/bin/env python

# Read the terrain data
print("Reading and warping the DTM ...")
dtmReader = vtk.vtkGDALRasterReader()
dtmReader.SetFileName(args.dtm)

dtmC2p = vtk.vtkCellDataToPointData()
dtmC2p.SetInputConnection(dtmReader.GetOutputPort())
dtmC2p.Update()

# Range of terrain data
lo = dtmC2p.GetOutput().GetScalarRange()[0]
hi = dtmC2p.GetOutput().GetScalarRange()[1]
bds = dtmC2p.GetOutput().GetBounds()
#print("Bounds: {0}".format(bds))
extent = dtmC2p.GetOutput().GetExtent()
#print("Extent: {0}".format(extent))
origin = dtmC2p.GetOutput().GetOrigin()
#print("Origin: {0}".format(origin))
spacing = dtmC2p.GetOutput().GetSpacing()
#print("Spacing: {0}".format(spacing))

# Convert the terrain into a polydata.
surface = vtk.vtkImageDataGeometryFilter()
surface.SetInputConnection(dtmC2p.GetOutputPort())

# Make sure the polygons are planar, so need to use triangles.
tris = vtk.vtkTriangleFilter()
tris.SetInputConnection(surface.GetOutputPort())

# Warp the surface by scalar values
warp = vtk.vtkWarpScalar()
warp.SetInputConnection(tris.GetOutputPort())
warp.SetScaleFactor(1)
warp.UseNormalOn()
warp.SetNormal(0, 0, 1)
warp.Update()

# Read the segmentation of buildings. Original data in GDAL is cell data.
# Point data is interpolated.
print("Reading the segmentation ...")
segmentationReader = vtk.vtkGDALRasterReader()
segmentationReader.SetFileName(args.segmentation)

segmentationC2p = vtk.vtkCellDataToPointData()
segmentationC2p.SetInputConnection(segmentationReader.GetOutputPort())
segmentationC2p.PassCellDataOn()
segmentationC2p.Update()
segmentation = segmentationC2p.GetOutput()

if (args.debug):
    segmentationWriter = vtk.vtkXMLImageDataWriter()
    segmentationWriter.SetFileName("segmentation.vti")
    segmentationWriter.SetInputConnection(segmentationC2p.GetOutputPort())
    segmentationWriter.Update()

    segmentation = segmentationC2p.GetOutput()
    sb = segmentation.GetBounds()
    print("segmentation bounds: \t{}".format(sb))

# Extract polygons
contours = vtk.vtkDiscreteFlyingEdges2D()
#contours = vtk.vtkMarchingSquares()
contours.SetInputConnection(segmentationC2p.GetOutputPort())
# default labels
# 2 -- no building
# 6 -- building
# 17 -- road or highway
# 65 -- don't score
labels = [6, 17]
if (args.label):
    labels = args.label
if (args.debug):
    scalarName = segmentation.GetCellData().GetScalars().GetName()
    segmentationNp = dsa.WrapDataObject(segmentation)
    scalars = segmentationNp.CellData[scalarName]
    allLabels = numpy.unique(scalars)
    print("Contouring on labels: {} of {}".format(labels, allLabels))
else:
    print("Contouring on labels: {}".format(labels))
contours.SetNumberOfContours(len(labels))
for i in range(len(labels)):
    contours.SetValue(i, labels[i])
#print("DFE: {0}".format(contours.GetOutput()))

if (args.debug):
    contoursWriter = vtk.vtkXMLPolyDataWriter()
    contoursWriter.SetFileName("contours.vtp")
    contoursWriter.SetInputConnection(contours.GetOutputPort())
    contoursWriter.Update()
    contoursData = contours.GetOutput()
    cb = contoursData.GetBounds()
    print("contours bounds: \t{}".format(cb))

if (not args.no_decimation):
    print("Decimating the contours ...")
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
print("Creating the loops ...")
loops = vtk.vtkContourLoopExtraction()
loops.SetInputConnection(contours.GetOutputPort())

if (args.debug):
    loopsWriter = vtk.vtkXMLPolyDataWriter()
    loopsWriter.SetFileName("loops.vtp")
    loopsWriter.SetInputConnection(loops.GetOutputPort())
    loopsWriter.Update()

# Read the DSM
print("Reading the DSM ...")
dsmReader = vtk.vtkGDALRasterReader()
dsmReader.SetFileName(args.dsm)

dsmC2p = vtk.vtkCellDataToPointData()
dsmC2p.SetInputConnection(dsmReader.GetOutputPort())
dsmC2p.Update()

print("Extruding the buildings ...")
fit = vtk.vtkFitToHeightMapFilter()
fit.SetInputConnection(loops.GetOutputPort())
fit.SetHeightMapConnection(dsmC2p.GetOutputPort())
fit.UseHeightMapOffsetOn()
fit.SetFittingStrategyToPointMaximumHeight()

if (args.debug):
    fitWriter = vtk.vtkXMLPolyDataWriter()
    fitWriter.SetFileName("fit.vtp")
    fitWriter.SetInputConnection(fit.GetOutputPort())
    fitWriter.Update()

# Extrude polygon down to surface
extrude = vtk.vtkTrimmedExtrusionFilter()
#extrude.SetInputData(polygons)
extrude.SetInputConnection(fit.GetOutputPort())
extrude.SetTrimSurfaceConnection(warp.GetOutputPort())
extrude.SetExtrusionDirection(0,0,1)
extrude.CappingOn()

extrudeWriter = vtk.vtkXMLPolyDataWriter()
extrudeWriter.SetFileName(args.destination)
extrudeWriter.SetInputConnection(extrude.GetOutputPort())
extrudeWriter.Update()

if (not args.no_render):
    # Create the RenderWindow, Renderer
    #
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer( ren )

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create pipeline. Load terrain data.
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6, 0)
    lut.SetSaturationRange(1.0, 0)
    lut.SetValueRange(0.5, 1.0)


    # Show the terrain
    dtmMapper = vtk.vtkPolyDataMapper()
    dtmMapper.SetInputConnection(warp.GetOutputPort())
    dtmMapper.SetScalarRange(lo, hi)
    dtmMapper.SetLookupTable(lut)

    dtmActor = vtk.vtkActor()
    dtmActor.SetMapper(dtmMapper)

    # show the buildings
    trisExtrude = vtk.vtkTriangleFilter()
    trisExtrude.SetInputConnection(extrude.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(trisExtrude.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Render it
    ren.AddActor(dtmActor)
    ren.AddActor(actor)

    ren.GetActiveCamera().Elevation(-60)
    ren.ResetCamera()

    renWin.Render()
    iren.Start()
