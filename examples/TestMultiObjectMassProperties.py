#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


# Demonstrate generation of extruded objects from a segmentation map, where
# the extrusion is trimmed by a terrain surface.
import vtk
from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()

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
demReader = vtk.vtkDEMReader()
#demReader.SetFileName(VTK_DATA_ROOT + "/Data/SainteHelens.dem")
demReader.SetFileName("SainteHelens.dem")
demReader.Update()

# Range of terrain data
lo = demReader.GetOutput().GetScalarRange()[0]
hi = demReader.GetOutput().GetScalarRange()[1]
bds = demReader.GetOutput().GetBounds()
#print("Bounds: {0}".format(bds))
extent = demReader.GetOutput().GetExtent()
#print("Extent: {0}".format(extent))
origin = demReader.GetOutput().GetOrigin()
#print("Origin: {0}".format(origin))
spacing = demReader.GetOutput().GetSpacing()
#print("Spacing: {0}".format(spacing))

# Convert the terrain into a polydata.
surface = vtk.vtkImageDataGeometryFilter()
surface.SetInputConnection(demReader.GetOutputPort())

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
demMapper = vtk.vtkPolyDataMapper()
demMapper.SetInputConnection(warp.GetOutputPort())
demMapper.SetScalarRange(lo, hi)
demMapper.SetLookupTable(lut)

demActor = vtk.vtkActor()
demActor.SetMapper(demMapper)

# Generate polygons to extrude. In application this should come from a reader
# (of a segmentation mask).
mask = vtk.vtkImageData()
mask.SetExtent(extent)
mask.SetOrigin(origin[0],origin[1],3000)
mask.SetSpacing(spacing)
dims = mask.GetDimensions()
#print("Mask Dims: {0}".format(dims))
extent = mask.GetExtent()
#print("Mask Extent: {0}".format(extent))
origin = mask.GetOrigin()
#print("Mask Origin: {0}".format(origin))
spacing = mask.GetSpacing()
#print("Mask Spacing: {0}".format(spacing))

# Initialize the pixel values to 0. Use VTK_CHAR=2 type.
mask.AllocateScalars(3,1)
imgData = mask.GetPointData().GetScalars()
npts = mask.GetNumberOfPoints()
#print("NPts: {0}".format(npts))
for i in range(0,npts):
    imgData.SetTuple1(i, 0)

# Now create some synthetic building masks
for j in range(250,275):
    jIdx = j*dims[0]
    for i in range(350,375):
        idx = i + jIdx
        imgData.SetTuple1(idx,255)

for j in range(125,140):
    jIdx = j*dims[0]
    for i in range(50,90):
        idx = i + jIdx
        imgData.SetTuple1(idx,255)

for j in range(290,315):
    jIdx = j*dims[0]
    for i in range(425,440):
        idx = i + jIdx
        imgData.SetTuple1(idx,255)

for j in range(25,50):
    jIdx = j*dims[0]
    for i in range(425,440):
        idx = i + jIdx
        imgData.SetTuple1(idx,255)

for j in range(25,50):
    jIdx = j*dims[0]
    for i in range(10,30):
        idx = i + jIdx
        imgData.SetTuple1(idx,255)

# Extract polygons
discrete = vtk.vtkDiscreteFlyingEdges2D()
discrete.SetInputData(mask)
discrete.SetValue(0,255)
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
polyActor.AddPosition(0,0,10)

# Render it
ren.AddActor(demActor)
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
