import argparse
import sys
import vtk

# Demonstrate generation of extruded objects from a segmentation map, where
# the extrusion is trimmed by a terrain surface.
parser = argparse.ArgumentParser(
    description='Generate extruded given the DSM')
parser.add_argument("segmentation", help="Image with labeled buildings")
parser.add_argument("dsm", help="Digital surface model (DSM)")
parser.add_argument("dtm", help="Digital terain model (DTM)")
parser.add_argument("destination", help="Extruded buildings")
parser.add_argument('-l', "--label", type=int, default=6,
                    help="Label value used for buildings")
args = parser.parse_args()

#!/usr/bin/env python

# Create the RenderWindow, Renderer
#
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer( ren )

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Create pipeline. Load terrain data.
#
lut = vtk.vtkLookupTable()
lut.SetHueRange(0.6, 0)
lut.SetSaturationRange(1.0, 0)
lut.SetValueRange(0.5, 1.0)

# Read the data: after processing a height field results (i.e., terrain)
dtmReader = vtk.vtkGDALRasterReader()
dtmReader.SetFileName(args.dtm)
dtmReader.Update()

# Range of terrain data
lo = dtmReader.GetOutput().GetScalarRange()[0]
hi = dtmReader.GetOutput().GetScalarRange()[1]
bds = dtmReader.GetOutput().GetBounds()
#print("Bounds: {0}".format(bds))
extent = dtmReader.GetOutput().GetExtent()
#print("Extent: {0}".format(extent))
origin = dtmReader.GetOutput().GetOrigin()
#print("Origin: {0}".format(origin))
spacing = dtmReader.GetOutput().GetSpacing()
#print("Spacing: {0}".format(spacing))

# Convert the terrain into a polydata.
surface = vtk.vtkImageDataGeometryFilter()
surface.SetInputConnection(dtmReader.GetOutputPort())

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

# Show the terrain
dtmMapper = vtk.vtkPolyDataMapper()
dtmMapper.SetInputConnection(warp.GetOutputPort())
dtmMapper.SetScalarRange(lo, hi)
dtmMapper.SetLookupTable(lut)

dtmActor = vtk.vtkActor()
dtmActor.SetMapper(dtmMapper)

# Read the segmentation of buildings
segmentationReader = vtk.vtkGDALRasterReader()
segmentationReader.SetFileName(args.segmentation)
segmentationReader.Update()
segmentation = segmentationReader.GetOutput()

# Extract polygons
discrete = vtk.vtkDiscreteFlyingEdges2D()
discrete.SetInputData(segmentation)
discrete.SetValue(0,args.label)
discrete.Update()
#print("DFE: {0}".format(discrete.GetOutput()))

# Create polygons
# Create polgons
polyLoops = vtk.vtkContourLoopExtraction()
polyLoops.SetInputConnection(discrete.GetOutputPort())

# Extrude polygon down to surface
extrude = vtk.vtkTrimmedExtrusionFilter()
#extrude.SetInputData(polygons)
extrude.SetInputConnection(polyLoops.GetOutputPort())
extrude.SetTrimSurfaceConnection(warp.GetOutputPort())
extrude.SetExtrusionDirection(0,0,1)
extrude.CappingOn()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(extrude.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Show generating polygons
polyMapper = vtk.vtkPolyDataMapper()
polyMapper.SetInputConnection(polyLoops.GetOutputPort())
polyMapper.ScalarVisibilityOff()

# Offset slightly to avoid zbuffer issues
polyActor = vtk.vtkActor()
polyActor.SetMapper(polyMapper)
polyActor.GetProperty().SetColor(1,0,0)
polyActor.AddPosition(0,0,-5)

# Render it
ren.AddActor(dtmActor)
ren.AddActor(actor)
ren.AddActor(polyActor)

ren.GetActiveCamera().SetPosition( 560752, 5110002, 2110)
ren.GetActiveCamera().SetFocalPoint( 560750, 5110000, 2100)
ren.ResetCamera()
ren.GetActiveCamera().SetClippingRange(269.775, 34560.4)
ren.GetActiveCamera().SetFocalPoint(562026, 5.1135e+006, -400.794)
ren.GetActiveCamera().SetPosition(556898, 5.10151e+006, 7906.19)
ren.ResetCamera()

# added these unused default arguments so that the prototype
# matches as required in python.
def reportCamera (a=0,b=0,__vtk__temp0=0,__vtk__temp1=0):
    print("Camera: {}".format(ren.GetActiveCamera()))

picker = vtk.vtkCellPicker()
picker.AddObserver("EndPickEvent",reportCamera)
iren.SetPicker(picker)

renWin.Render()
iren.Start()
