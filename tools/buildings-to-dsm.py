import argparse
import gdal
import glob
import numpy
import os
import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support

parser = argparse.ArgumentParser(
    description='Render a DSM from a DTM and polygons representing buildings.')
parser.add_argument("buildings", help="Buildings polygonal file")
parser.add_argument("dtm", help="Digital terain model (DTM)")
parser.add_argument("dsm", help="Resulting digital surface model (DSM)")
parser.add_argument("--render_png", action="store_true",
                    help="Do not save the DSM, render into a PNG instead.")
parser.add_argument("--buildings_only", action="store_true",
                    help="Do not use the DTM, use only the buildings.")
args = parser.parse_args()

# open the DTM
dtm = gdal.Open(args.dtm, gdal.GA_ReadOnly)
if not dtm:
    sys.exit(1)

dtmDriver = dtm.GetDriver()
dtmDriverMetadata = dtmDriver.GetMetadata()
dsm = None
dtmBounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
flipX = False
flipY = False
if dtmDriverMetadata.get(gdal.DCAP_CREATE) == "YES":
    print("Create destination image "
          "size:({}, {}) ...".format(dtm.RasterXSize,
                                     dtm.RasterYSize))
    # georeference information
    projection = dtm.GetProjection()
    transform = dtm.GetGeoTransform()
    gcpProjection = dtm.GetGCPProjection()
    gcps = dtm.GetGCPs()
    options = []
    # ensure that space will be reserved for geographic corner coordinates
    # (in DMS) to be set later
    if (dtmDriver.ShortName == "NITF" and not projection):
        options.append("ICORDS=G")
    eType = gdal.GDT_Float32
    dtype = numpy.float32
    dsm = dtmDriver.Create(
        args.dsm, xsize=dtm.RasterXSize,
        ysize=dtm.RasterYSize, bands=1, eType=eType,
        options=options)
    if (projection):
        # georeference through affine geotransform
        dsm.SetProjection(projection)
        dsm.SetGeoTransform(transform)
        corners = [[0, 0], [0, dtm.RasterYSize],
                   [dtm.RasterXSize, dtm.RasterYSize], [dtm.RasterXSize, 0]]
        geoCorners = numpy.zeros((4,2))
        for i, corner in enumerate(corners):
            geoCorners[i] = [
                transform[0] + corner[0] * transform[1] +\
                corner[1] * transform[2],
                transform[3] + corner[0] * transform[4] +\
                corner[1] * transform[5]]
        dtmBounds[0] = numpy.min(geoCorners[:,0])
        dtmBounds[1] = numpy.max(geoCorners[:,0])
        dtmBounds[2] = numpy.min(geoCorners[:,1])
        dtmBounds[3] = numpy.max(geoCorners[:,1])
    else:
        # georeference through GCPs
        dsm.SetGCPs(gcps, gcpProjection)
        # compute dtmBounds before
        # ...
        print("Error: we need to compute DTM bounds from GCPs")
        sys.exit(1)
    print("Reading the DTM {} size: ({}, {})\n"
          "\tbounds: ({}, {}), ({}, {})...".format(
        args.dtm, dtm.RasterXSize, dtm.RasterYSize, dtmBounds[0], dtmBounds[1],
        dtmBounds[2], dtmBounds[3]))
    dtmRaster = dtm.GetRasterBand(1).ReadAsArray()
else:
    print("Driver {} does not supports Create().".format(dtmDriver))
    sys.exit(1)


# read the buildings polydata, set Z as a scalar and project to XY plane
print("Reading the buildings ...")
if (os.path.isfile(args.buildings)):
    polyReader = vtk.vtkXMLPolyDataReader()
    polyReader.SetFileName(args.buildings)
    polyReader.Update()
    polyBuildingsVtk = polyReader.GetOutput()
else:
    # assume a folder with OBJ files
    buildingsFiles = glob.glob(args.buildings + "*.obj")
    append = vtk.vtkAppendPolyData()
    for i, buildingsFileName in enumerate(buildingsFiles):
        objReader = vtk.vtkOBJReader()
        objReader.SetFileName(buildingsFileName)
        append.AddInputConnection(objReader.GetOutputPort())
    append.Update()
    polyBuildingsVtk = append.GetOutput()

polyBuildings = dsa.WrapDataObject(polyBuildingsVtk)
polyElevation = polyBuildings.Points[:, 2]
polyElevationVtk = numpy_support.numpy_to_vtk(polyElevation)
polyElevationVtk.SetName("Elevation")
polyBuildings.PointData.SetScalars(polyElevationVtk)


# Create the RenderWindow, Renderer
#
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.OffScreenRenderingOn()
renWin.SetSize(dtm.RasterXSize, dtm.RasterYSize)
renWin.SetMultiSamples(0)
renWin.AddRenderer( ren )

# Show the terrain. This is not really neccessary but
# this is how we get best allignment.
print("Converting the DTM into a surface ...")
# read the DTM as a VTK object
dtmReader = vtk.vtkGDALRasterReader()
dtmReader.SetFileName(args.dtm)
dtmReader.Update()
dtmVtk = dtmReader.GetOutput()

# Convert the terrain into a polydata.
surface = vtk.vtkImageDataGeometryFilter()
surface.SetInputDataObject(dtmVtk)

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
dsmScalarRange = warp.GetOutput().GetPointData().GetScalars().GetRange()

dtmMapper = vtk.vtkPolyDataMapper()
dtmMapper.SetInputConnection(warp.GetOutputPort())
dtmActor = vtk.vtkActor()
dtmActor.SetMapper(dtmMapper)
ren.AddActor(dtmActor)

# show the buildings
trisBuildingsFilter = vtk.vtkTriangleFilter()
trisBuildingsFilter.SetInputDataObject(polyBuildingsVtk)
trisBuildingsFilter.Update()
trisBuildings = trisBuildingsFilter.GetOutput()

p2cBuildings = vtk.vtkPointDataToCellData()
p2cBuildings.SetInputConnection(trisBuildingsFilter.GetOutputPort())
p2cBuildings.PassPointDataOn()
p2cBuildings.Update()
buildingsScalarRange = p2cBuildings.GetOutput().GetCellData().GetScalars().GetRange()


polyWriter = vtk.vtkXMLPolyDataWriter()
polyWriter.SetFileName("p2c.vtp")
polyWriter.SetInputConnection(p2cBuildings.GetOutputPort())
polyWriter.Write()

buildingsMapper = vtk.vtkPolyDataMapper()
buildingsMapper.SetInputDataObject(p2cBuildings.GetOutput())

buildingsActor = vtk.vtkActor()
buildingsActor.SetMapper(buildingsMapper)
ren.AddActor(buildingsActor)

ren.ResetCamera()
camera = ren.GetActiveCamera()
camera.ParallelProjectionOn()
camera.SetParallelScale((dtmBounds[3] - dtmBounds[2])/2)
#camera.SetFocalPoint([(dtmBounds[0] + dtmBounds[1])/2, (dtmBounds[3] + dtmBounds[2])/2, 0])

if (args.render_png):
    print("Render into a PNG ...")
    if (args.buildings_only):
        scalarRange = buildingsScalarRange
    else:
        scalarRange = [min(dsmScalarRange[0], buildingsScalarRange[0]),
                       max(dsmScalarRange[0], buildingsScalarRange[0])]
    lut = vtk.vtkColorTransferFunction()
    lut.AddRGBPoint(scalarRange[0], 0.23, 0.30, 0.75)
    lut.AddRGBPoint((scalarRange[0] + scalarRange[1]) / 2, 0.86, 0.86, 0.86)
    lut.AddRGBPoint(scalarRange[1], 0.70, 0.02, 0.15)

    dtmMapper.SetLookupTable(lut)
    dtmMapper.SetColorModeToMapScalars()
    buildingsMapper.SetLookupTable(lut)
    if (args.buildings_only):
        ren.RemoveActor(dtmActor)

    renWin.Render()
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.SetInputBufferTypeToRGBA()
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    writerPng = vtk.vtkPNGWriter()
    writerPng.SetFileName(args.dsm + ".png");
    writerPng.SetInputConnection(windowToImageFilter.GetOutputPort())
    writerPng.Write();
else:
    print("Render into a floating point buffer ...")
    valuePass = vtk.vtkValuePass()
    valuePass.SetRenderingMode(vtk.vtkValuePass.FLOATING_POINT)
    valuePass.SetInputComponentToProcess(0)
    # use the default scalar for point data
    valuePass.SetInputComponentToProcess(0);
    valuePass.SetInputArrayToProcess(vtk.VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
                                     "Elevation");
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(valuePass)
    sequence = vtk.vtkSequencePass()
    sequence.SetPasses(passes)
    cameraPass = vtk.vtkCameraPass()
    cameraPass.SetDelegatePass(sequence)
    ren.SetPass(cameraPass)
    # We have to render the points first, otherwise we get a segfault.
    renWin.Render()
    ren.RemoveActor(dtmActor)
    valuePass.SetInputArrayToProcess(vtk.VTK_SCALAR_MODE_USE_CELL_FIELD_DATA, "Elevation");
    renWin.Render()
    elevationFlatVtk = valuePass.GetFloatImageDataArray(ren)
    valuePass.ReleaseGraphicsResources(renWin)

    print("Writing the DSM ...")
    elevationFlat = numpy_support.vtk_to_numpy(elevationFlatVtk)
    # VTK X,Y corresponds to numpy cols,rows. VTK stores arrays
    # in Fortran order.
    elevationTranspose = numpy.reshape(
        elevationFlat, dtmRaster.shape[::-1], "F")
    # changes from cols, rows to rows,cols.
    elevation = numpy.transpose(elevationTranspose)
    # not sure why is this needed
    elevation = numpy.flip(elevation, 0)
    if (args.buildings_only):
        dsmElevation = elevation
    else:
        # elevation has nans in places other than buildings
        dsmElevation = numpy.fmax(dtmRaster, elevation)
    dsm.GetRasterBand(1).WriteArray(dsmElevation)
