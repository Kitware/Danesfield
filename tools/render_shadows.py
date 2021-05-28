#!/usr/bin/env python

import argparse
from danesfield import gdal_utils
from danesfield import ortho
import gdal
import logging
import numpy
import os
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import vtkAlgorithm as vta


class ParallelProjectionModel(object):
    '''Implements a parallel projection: (lon, lat, alt) ==> (lon, lat)
    '''
    def __init__(self, m, dims):
        '''Initialize the class from a 16 element row order matrix.
        '''
        self.mat = numpy.array([[m[0], m[1], m[2], m[3]],
                                [m[4], m[5], m[6], m[7]],
                                [m[8], m[9], m[10], m[11]],
                                [m[12], m[13], m[14], m[15]]])
        self.dims = dims

    def project(self, point):
        '''Project a long, lat, elev point into image coordinates using parallel projection
        This function can also project an (n,3) matrix where each row of the
        matrix is a point to project.  The result is an (n,2) matrix of image
        coordinates.
        '''
        pointCount = point.shape[0]
        ones = numpy.repeat(1, pointCount)
        ones = numpy.reshape(ones, (pointCount, 1))
        point = numpy.hstack((point, ones))
        viewPoint = numpy.dot(point, self.mat)
        viewPoint = numpy.delete(viewPoint, [2, 3], axis=1)
        imagePoint = (viewPoint + 1.0) / 2.0 * self.dims
        imagePoint[:, 1] = self.dims[1] - imagePoint[:, 1]
        return imagePoint


class FillNoData(vta.VTKPythonAlgorithmBase):
    ''' Algorithm with a DSM and DTM as inputs, DTM is optional. It produces
        an output the same as the DSM with the NoData values replaced with correspoinding
        values from the DTM.
    '''
    def __init__(self):
        vta.VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, outputType='vtkUniformGrid')

    def RequestInformation(self, request, inInfo, outInfo):
        origin = inInfo[0].GetInformationObject(0).Get(vtk.vtkDataObject.ORIGIN())
        spacing = inInfo[0].GetInformationObject(0).Get(vtk.vtkDataObject.SPACING())
        outInfo.GetInformationObject(0).Set(
            vtk.vtkDataObject.ORIGIN(), origin, 3)
        outInfo.GetInformationObject(0).Set(
            vtk.vtkDataObject.SPACING(), spacing, 3)
        return 1

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        if (port == 1):
            info.Set(vtk.vtkAlgorithm.INPUT_IS_OPTIONAL(), 1)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        i0 = self.GetInputData(inInfo, 0, 0)
        dsm = dsa.WrapDataObject(i0)
        dtmInfo = inInfo[1].GetInformationObject(0)
        if (dtmInfo):
            i1 = self.GetInputData(inInfo, 1, 0)
            dtm = dsa.WrapDataObject(i1)
        o = self.GetOutputData(outInfo, 0)
        o.ShallowCopy(i0)
        output = dsa.WrapDataObject(o)

        dsmData = dsm.CellData["Elevation"]
        if (dtmInfo):
            dtmData = dtm.CellData["Elevation"]
            ghostArray = dsm.CellData["vtkGhostType"]
            blankedIndex = ghostArray == 32
            dsmData[blankedIndex] = dtmData[blankedIndex]
        output.VTKObject.GetCellData().RemoveArray("vtkGhostType")
        output.CellData.append(dsmData, "Elevation")
        return 1


def main(args):
    parser = argparse.ArgumentParser(
        description='Render a shadow mask from a sun position (stored in the input_image), '
                    'a DSM and an optional DTM')
    parser.add_argument("input_image", help="Source image with sun position")
    parser.add_argument("dsm", help="Digital surface model (DSM) image")
    parser.add_argument("output_image", help="Image with shadow mask")
    parser.add_argument("--dtm", type=str,
                        help="Optional DTM parameter used to fill nodata areas "
                             "in the dsm")
    parser.add_argument("--render_png", action="store_true",
                        help="Do not save shadow mask, render a PNG instead.")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate results")
    args = parser.parse_args(args)

    sourceImage = gdal_utils.gdal_open(args.input_image)
    metaData = sourceImage.GetMetadata()
    azimuth = float(metaData["NITF_CSEXRA_SUN_AZIMUTH"])
    elevation = float(metaData["NITF_CSEXRA_SUN_ELEVATION"])
    print("azimuth = {}, elevation = {}".format(azimuth, elevation))
    sourceImage = None

    dsm = vtk.vtkGDALRasterReader()
    dsm.SetFileName(args.dsm)

    fillNoData = FillNoData()
    fillNoData.SetInputConnection(0, dsm.GetOutputPort())
    if (args.dtm):
        dtm = vtk.vtkGDALRasterReader()
        dtm.SetFileName(args.dtm)
        fillNoData.SetInputConnection(1, dtm.GetOutputPort())

    cellToPoint = vtk.vtkCellDataToPointData()
    cellToPoint.SetInputConnection(fillNoData.GetOutputPort())

    warp = vtk.vtkWarpScalar()
    warp.SetInputConnection(cellToPoint.GetOutputPort())
    warp.Update()
    warpOutput = warp.GetOutput()

    scalarRange = warpOutput.GetPointData().GetScalars().GetRange()
    warpBounds = warpOutput.GetBounds()
    dims = warpOutput.GetDimensions()

    if (args.debug):
        writerVts = vtk.vtkXMLStructuredGridWriter()
        writerVts.SetFileName("warp.vts")
        writerVts.SetInputDataObject(warpOutput)
        writerVts.Write()

    print("warpBounds: {}".format(warpBounds))
    print("scalarRange: {}".format(scalarRange))
    print("dims: {}".format(dims))

    # vtkValuePass works only with vtkPolyData
    warpSurface = vtk.vtkDataSetSurfaceFilter()
    warpSurface.SetInputConnection(warp.GetOutputPort())
    warpSurface.Update()

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    # VTK specifies dimensions in points, GDAL in cells
    renWin.SetSize(dims[0] - 1, dims[1] - 1)
    renWin.AddRenderer(ren)

    warpMapper = vtk.vtkPolyDataMapper()
    warpMapper.SetInputConnection(warpSurface.GetOutputPort())
    warpActor = vtk.vtkActor()
    warpActor.SetMapper(warpMapper)
    ren.AddActor(warpActor)

    camera = ren.GetActiveCamera()
    camera.SetViewUp(0, 1, 0)
    camera.ParallelProjectionOn()
    camera.Roll(azimuth - 180)
    camera.Elevation(elevation - 90)
    ren.ResetCamera()

    if (args.render_png):
        print("Render into a PNG ...")
        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(scalarRange[0], 0.23, 0.30, 0.75)
        lut.AddRGBPoint((scalarRange[0] + scalarRange[1]) / 2, 0.86, 0.86, 0.86)
        lut.AddRGBPoint(scalarRange[1], 0.70, 0.02, 0.15)
        warpMapper.SetLookupTable(lut)
        warpMapper.SetColorModeToMapScalars()

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        renWin.Render()

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.SetInputBufferTypeToRGBA()
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()

        writerPng = vtk.vtkPNGWriter()
        writerPng.SetFileName(args.output_image + ".png")
        writerPng.SetInputConnection(windowToImageFilter.GetOutputPort())
        writerPng.Write()

        iren.Start()
    else:
        print("Render into a floating point buffer ...")
        renWin.OffScreenRenderingOn()
        arrayName = "Elevation"
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
        renWin.Render()
        elevationFlatVtk = valuePass.GetFloatImageDataArray(ren)
        elevationFlatVtk.SetName("Elevation")
        valuePass.ReleaseGraphicsResources(renWin)

        # GDAL dataset
        heightMapFileName = "_temp_heightMap.tif"
        elevation = gdal_utils.vtk_to_numpy_order(elevationFlatVtk,
                                                  [dims[0] - 1, dims[1] - 1])
        driver = gdal.GetDriverByName("GTiff")
        heightMap = driver.Create(
            heightMapFileName, xsize=dims[0] - 1, ysize=dims[1] - 1,
            bands=1, eType=gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
        heightMap.GetRasterBand(1).WriteArray(elevation)
        hasNoData = 0
        nodata = dsm.GetInvalidValue(0, hasNoData)
        if (not args.dtm and hasNoData):
            heightMap.GetRasterBand(1).SetNoDataValue(nodata)
        corners = [[0, 0], [dims[0] - 1, 0],
                   [dims[0] - 1, dims[1] - 1],
                   [0, dims[1] - 1]]
        gcps = []
        for corner in corners:
            worldPoint = [0.0, 0.0, 0.0]
            warpOutput.GetPoint(corner[0], corner[1], 0, worldPoint)
            ren.SetWorldPoint(worldPoint[0], worldPoint[1], worldPoint[2], 1.0)
            ren.WorldToView()
            viewPoint = numpy.array(ren.GetViewPoint())
            pixelCoord = (viewPoint + 1.0) / 2.0 * numpy.array(dims)
            pixelCoord[1] = dims[1] - pixelCoord[1]
            gcp = gdal.GCP(worldPoint[0], worldPoint[1], worldPoint[2],
                           pixelCoord[0], pixelCoord[1])
            gcps.append(gcp)
        wkt = dsm.GetProjectionWKT()
        heightMap.SetGCPs(gcps, wkt)
        heightMap = None

        if (args.debug):
            # VTK dataset
            heightMapVtk = vtk.vtkImageData()
            heightMapVtk.SetDimensions(dims[0], dims[1], 1)
            heightMapVtk.SetOrigin(dsm.GetOutput().GetOrigin())
            heightMapVtk.SetSpacing(dsm.GetOutput().GetSpacing())
            heightMapVtk.GetCellData().SetScalars(elevationFlatVtk)

            writerVti = vtk.vtkXMLImageDataWriter()
            writerVti.SetFileName("heightMap.vti")
            writerVti.SetInputDataObject(heightMapVtk)
            writerVti.Write()

        # create the parallel projection model
        mat = [ren.GetActiveCamera().GetCompositeProjectionTransformMatrix(
            ren.GetTiledAspectRatio(), 0, 1).GetElement(i, j)
               for j in range(0, 4)
               for i in range(0, 4)]
        model = ParallelProjectionModel(mat, numpy.array(dims[0:2]))
        ortho.orthorectify(heightMapFileName, args.dsm, args.output_image,
                           1.0, 2.0, model, args.dtm, False)
        if (not args.debug):
            os.remove(heightMapFileName)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
