#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import gdal
import numpy
import logging
import os
import re
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
from danesfield import gdal_utils
import glob

def activate_virtual_framebuffer():
    '''
    Activates a virtual (headless) framebuffer for rendering 3D
    scenes via VTK.

    Most critically, this function is useful when this code is being run
    in a Dockerized notebook, or over a server without X forwarding.

    * Requires the following packages:
      * `sudo apt-get install libgl1-mesa-dev xvfb`
    '''

    import subprocess
    import vtk

    vtk.OFFSCREEN = True
    os.environ['DISPLAY']=':99.0'

    commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
                'sleep 3',
                'exec "$@"']

    for command in commands:
        subprocess.call(command,shell=True)

def main(args):
    parser = argparse.ArgumentParser(
        description='Render a DSM from a DTM and polygons representing buildings.')
    parser.add_argument("--input_vtp_path", type=str,
                        help="Input buildings polygonal file (.vtp)")
    parser.add_argument("--input_obj_paths", nargs="*",
                        help="List of input building (.obj) file paths.  "
                             "Building object files start "
                             "with a digit, road object files start with \"Road\". "
                             "All obj files start with comments specifying the offsets "
                             "that are added the coordinats. There are three comment lines, "
                             "one for each coordinate: \"#c offset: value\" where c is x, y and z.")
    parser.add_argument("input_dtm", help="Input digital terain model (DTM)")
    parser.add_argument("output_dsm", help="Output digital surface model (DSM)")
    parser.add_argument("--render_png", action="store_true",
                        help="Do not save the DSM, render into a PNG instead.")
    parser.add_argument("--render_cls", action="store_true",
                        help="Render a buildings mask: render buildings label (6), "
                             "background (2) and no DTM.")
    parser.add_argument("--buildings_only", action="store_true",
                        help="Do not use the DTM, use only the buildings.")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate results")
    args = parser.parse_args(args)

    activate_virtual_framebuffer()

    # open the DTM
    dtm = gdal.Open(args.input_dtm, gdal.GA_ReadOnly)
    if not dtm:
        raise RuntimeError("Error: Failed to open DTM {}".format(args.input_dtm))

    dtmDriver = dtm.GetDriver()
    dtmDriverMetadata = dtmDriver.GetMetadata()
    dsm = None
    dtmBounds = [0.0, 0.0, 0.0, 0.0]
    if dtmDriverMetadata.get(gdal.DCAP_CREATE) == "YES":
        print("Create destination image "
              "size:({}, {}) ...".format(dtm.RasterXSize,
                                         dtm.RasterYSize))
        # georeference information
        projection = dtm.GetProjection()
        transform = dtm.GetGeoTransform()
        gcpProjection = dtm.GetGCPProjection()
        gcps = dtm.GetGCPs()
        options = ["COMPRESS=DEFLATE"]
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (dtmDriver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        if args.render_cls:
            eType = gdal.GDT_Byte
        else:
            eType = gdal.GDT_Float32
        dsm = dtmDriver.Create(
            args.output_dsm, xsize=dtm.RasterXSize,
            ysize=dtm.RasterYSize, bands=1, eType=eType,
            options=options)
        if (projection):
            # georeference through affine geotransform
            dsm.SetProjection(projection)
            dsm.SetGeoTransform(transform)
        else:
            # georeference through GCPs
            dsm.SetGCPs(gcps, gcpProjection)
            gdal.GCPsToGeoTransform(gcps, transform)
        corners = [[0, 0], [0, dtm.RasterYSize],
                   [dtm.RasterXSize, dtm.RasterYSize], [dtm.RasterXSize, 0]]
        geoCorners = numpy.zeros((4, 2))
        for i, corner in enumerate(corners):
            geoCorners[i] = [
                transform[0] + corner[0] * transform[1] +
                corner[1] * transform[2],
                transform[3] + corner[0] * transform[4] +
                corner[1] * transform[5]]
        dtmBounds[0] = numpy.min(geoCorners[:, 0])
        dtmBounds[1] = numpy.max(geoCorners[:, 0])
        dtmBounds[2] = numpy.min(geoCorners[:, 1])
        dtmBounds[3] = numpy.max(geoCorners[:, 1])

        if args.render_cls:
            # label for no building
            dtmRaster = numpy.full([dtm.RasterYSize, dtm.RasterXSize], 2)
            nodata = 0
        else:
            print("Reading the DTM {} size: ({}, {})\n"
                  "\tbounds: ({}, {}), ({}, {})...".format(
                      args.input_dtm, dtm.RasterXSize, dtm.RasterYSize,
                      dtmBounds[0], dtmBounds[1],
                      dtmBounds[2], dtmBounds[3]))
            dtmRaster = dtm.GetRasterBand(1).ReadAsArray()
            nodata = dtm.GetRasterBand(1).GetNoDataValue()
        print("Nodata: {}".format(nodata))
    else:
        raise RuntimeError("Driver {} does not supports Create().".format(dtmDriver))

    # read the buildings polydata, set Z as a scalar and project to XY plane
    print("Reading the buildings ...")
    # labels for buildings and elevated roads
    labels = [6, 17]
    if (args.input_vtp_path and os.path.isfile(args.input_vtp_path)):
        polyReader = vtk.vtkXMLPolyDataReader()
        polyReader.SetFileName(args.input_vtp_path)
        polyReader.Update()
        polyVtkList = [polyReader.GetOutput()]
    elif (args.input_obj_paths):
        # buildings start with numbers
        # optional elevated roads start with Road*.obj
        bldg_re = re.compile(".*/?[0-9][^/]*\\.obj")
        bldg_files = [f for f in args.input_obj_paths
                      if bldg_re.match(f)]
        road_re = re.compile(".*/?Road[^/]*\\.obj")
        road_files = [f for f in args.input_obj_paths
                      if road_re.match(f)]
        files = [bldg_files,
                 road_files]
        files = [x for x in files if x]
        print(road_files)
        if len(files) >= 2:
            print("Found {} buildings and {} roads".format(len(files[0]), len(files[1])))
        elif len(files) == 1:
            print("Found {} buildings".format(len(files[0])))
        else:
            raise RuntimeError("No OBJ files found in {}".format(args.input_obj_paths))
        polyVtkList = []
        for category in range(len(files)):
            append = vtk.vtkAppendPolyData()
            for i, fileName in enumerate(files[category]):
                offset = [0.0, 0.0, 0.0]
                gdal_utils.read_offset(fileName, offset)
                print("Offset: {}".format(offset))
                transform = vtk.vtkTransform()
                transform.Translate(offset[0], offset[1], offset[2])

                objReader = vtk.vtkOBJReader()
                objReader.SetFileName(fileName)
                transformFilter = vtk.vtkTransformFilter()
                transformFilter.SetTransform(transform)
                transformFilter.SetInputConnection(objReader.GetOutputPort())
                append.AddInputConnection(transformFilter.GetOutputPort())
            append.Update()
            polyVtkList.append(append.GetOutput())
    else:
        raise RuntimeError("Must provide either --input_vtp_path, or --input_obj_paths")

    arrayName = "Elevation"
    append = vtk.vtkAppendPolyData()
    for category in range(len(polyVtkList)):
        poly = dsa.WrapDataObject(polyVtkList[category])
        polyElevation = poly.Points[:, 2]
        if args.render_cls:
            # label for buildings
            polyElevation[:] = labels[category]
        polyElevationVtk = numpy_support.numpy_to_vtk(polyElevation)
        polyElevationVtk.SetName(arrayName)
        poly.PointData.SetScalars(polyElevationVtk)
        append.AddInputDataObject(polyVtkList[category])
    append.Update()

    # Create the RenderWindow, Renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.OffScreenRenderingOn()
    renWin.SetOffScreenRendering(1)
    renWin.SetSize(dtm.RasterXSize, dtm.RasterYSize)
    renWin.SetMultiSamples(0)
    renWin.AddRenderer(ren)

    # show the buildings
    trisBuildingsFilter = vtk.vtkTriangleFilter()
    trisBuildingsFilter.SetInputDataObject(append.GetOutput())
    trisBuildingsFilter.Update()

    p2cBuildings = vtk.vtkPointDataToCellData()
    p2cBuildings.SetInputConnection(trisBuildingsFilter.GetOutputPort())
    p2cBuildings.PassPointDataOn()
    p2cBuildings.Update()
    buildingsScalarRange = p2cBuildings.GetOutput().GetCellData().GetScalars().GetRange()

    if (args.debug):
        polyWriter = vtk.vtkXMLPolyDataWriter()
        polyWriter.SetFileName("p2c.vtp")
        polyWriter.SetInputConnection(p2cBuildings.GetOutputPort())
        polyWriter.Write()

    buildingsMapper = vtk.vtkPolyDataMapper()
    buildingsMapper.SetInputDataObject(p2cBuildings.GetOutput())

    buildingsActor = vtk.vtkActor()
    buildingsActor.SetMapper(buildingsMapper)
    ren.AddActor(buildingsActor)

    if (args.render_png):
        print("Render into a PNG ...")
        # Show the terrain.
        print("Converting the DTM into a surface ...")
        # read the DTM as a VTK object
        dtmReader = vtk.vtkGDALRasterReader()
        dtmReader.SetFileName(args.input_dtm)
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

        ren.ResetCamera()
        camera = ren.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetParallelScale((dtmBounds[3] - dtmBounds[2])/2)

        if (args.buildings_only):
            scalarRange = buildingsScalarRange
        else:
            scalarRange = [min(dsmScalarRange[0], buildingsScalarRange[0]),
                           max(dsmScalarRange[1], buildingsScalarRange[1])]
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
        writerPng.SetFileName(args.output_dsm + ".png")
        writerPng.SetInputConnection(windowToImageFilter.GetOutputPort())
        writerPng.Write()
    else:
        print("Render into a floating point buffer ...")

        try:

            ren.ResetCamera()
            camera = ren.GetActiveCamera()
            camera.ParallelProjectionOn()
            camera.SetParallelScale((dtmBounds[3] - dtmBounds[2])/2)
            distance = camera.GetDistance()
            focalPoint = [(dtmBounds[0] + dtmBounds[1]) * 0.5,
                          (dtmBounds[3] + dtmBounds[2]) * 0.5,
                          (buildingsScalarRange[0] + buildingsScalarRange[1]) * 0.5]
            position = [focalPoint[0], focalPoint[1], focalPoint[2] + distance]
            camera.SetFocalPoint(focalPoint)
            camera.SetPosition(position)

            valuePass = vtk.vtkValuePass()
            valuePass.SetRenderingMode(vtk.vtkValuePass.FLOATING_POINT)
            # use the default scalar for point data
            valuePass.SetInputComponentToProcess(0)
            valuePass.SetInputArrayToProcess(vtk.VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
                                             arrayName)
            passes = vtk.vtkRenderPassCollection()
            passes.AddItem(valuePass)
            sequence = vtk.vtkSequencePass()
            sequence.SetPasses(passes)
            cameraPass = vtk.vtkCameraPass()
            cameraPass.SetDelegatePass(sequence)
            ren.SetPass(cameraPass)
            # We have to render the points first, otherwise we get a segfault.
            renWin.Render()
            valuePass.SetInputArrayToProcess(vtk.VTK_SCALAR_MODE_USE_CELL_FIELD_DATA, arrayName)
            renWin.Render()
            elevationFlatVtk = valuePass.GetFloatImageDataArray(ren)
            valuePass.ReleaseGraphicsResources(renWin)

            print("Writing the DSM ...")
            elevationFlat = numpy_support.vtk_to_numpy(elevationFlatVtk)
            # VTK X,Y corresponds to numpy cols,rows. VTK stores arrays
            # in Fortran order.
            elevationTranspose = numpy.reshape(
                elevationFlat, [dtm.RasterXSize, dtm.RasterYSize], "F")
            # changes from cols, rows to rows,cols.
            elevation = numpy.transpose(elevationTranspose)
            # numpy rows increase as you go down, Y for VTK images increases as you go up
            elevation = numpy.flip(elevation, 0)
            if args.buildings_only:
                dsmElevation = elevation
            else:
                # elevation has nans in places other than buildings
                dsmElevation = numpy.fmax(dtmRaster, elevation)
            dsm.GetRasterBand(1).WriteArray(dsmElevation)
            if nodata:
                dsm.GetRasterBand(1).SetNoDataValue(nodata)

        except:
            pass


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
