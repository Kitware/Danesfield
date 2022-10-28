import argparse
import logging
import os
import re
from typing import (
    List,
    Tuple)

from vtkmodules.util import numpy_support
from vtkmodules.vtkIOImage import (
    vtkTIFFReader,
    vtkTIFFWriter)
from vtkmodules.vtkCommonCore import (
    vtkDataArray)
from vtkmodules.vtkCommonDataModel import (
    vtkImageData)
from vtkmodules.vtkImagingCore import vtkImageFlip

def read_tiff(tiff_file_name: str) -> Tuple[vtkDataArray, vtkImageData, bool]:
    '''
    Reads a tiff file and returns its values, data and if values are
    point or cell attributes.
    '''
    tiff_reader = vtkTIFFReader()
    tiff_reader.SetFileName(tiff_file_name)
    tiff_reader.Update()
    tiff_data = tiff_reader.GetOutput()
    # try if data is in points
    tiff_array = tiff_data.GetPointData().GetArray(0)
    dims = tiff_data.GetDimensions()
    is_point_array = True
    if tiff_array is None:
        # try if data is in cells
        tiff_array = tiff_data.GetCellData().GetArray(0)
        dims = (dims[0] - 1, dims[1] - 1, dims[2])
        is_point_array = False
    return (tiff_array, tiff_data, is_point_array)


def get_tiffs(d: str) -> List[str]:
    '''
    Read all tiffs file names in the directory 'd'
    '''
    print("Reading files from {}".format(d))
    tiff_files = []
    tiff_re = r'^.*\.tiff$'
    for f in os.listdir(d):
        m = re.search(tiff_re, f, re.IGNORECASE)
        if m:
            tiff_files.append(f)
    tiff_files.sort(key=lambda f: int(re.sub('\\D', '', f)))
    return tiff_files


def main(args):
    """
    Adds tiffs in two folders and write the result in a third folder.
    Tiffs in the input folder and in the output folder have the same name.
    """
    parser = argparse.ArgumentParser(
        description="We read all tiffs from input1 add them to the tiffs in input2 "
        "(the file names should match) and write the resulting tiffs in the output directory")
    parser.add_argument("dirs", nargs=3,
                        help="input1 input2 and output directories")
    args = parser.parse_args(args)
    tiff_files = get_tiffs(args.dirs[0]);
    for fileName in tiff_files:
        (tiff_array0, tiff_data0, is_point_array0) = read_tiff(args.dirs[0] + "/" + fileName)
        ta0 = numpy_support.vtk_to_numpy(tiff_array0)

        (tiff_array1, _, _) = read_tiff(args.dirs[1] + "/" + fileName)
        ta1 = numpy_support.vtk_to_numpy(tiff_array1)
        if len(ta0) != len(ta1):
            logging.error("Error: different size array for {}: {} vs {}".format(
                fileName, len(ta0), len(ta1)))
            break

        ta = ta0 + ta1
        data = vtkImageData()
        data.ShallowCopy(tiff_data0)
        tiff_array = numpy_support.numpy_to_vtk(ta)
        tiff_array.SetName(tiff_array0.GetName())
        if is_point_array0:
            data.GetPointData().SetScalars(tiff_array)
        else:
            data.GetCellData().SetScalars(tiff_array)
        outputFileName = args.dirs[2] + "/" + fileName
        print("Write {}".format(outputFileName))

        flip_y_filter = vtkImageFlip()
        flip_y_filter.SetFilteredAxis(1)
        flip_y_filter.SetInputDataObject(data)

        writer = vtkTIFFWriter()
        writer.SetInputConnection(flip_y_filter.GetOutputPort())
        writer.SetFileName(outputFileName)
        writer.Write()


if __name__ == '__main__':
    import sys
    try:
        logging.basicConfig(format='(%(relativeCreated)03d ms) %(levelname)s:%(message)s',
                            level=logging.DEBUG,
                            datefmt='%I:%M:%S %p')
        main(sys.argv[1:])
    except Exception as ex:
        logging.exception(ex)
        sys.exit(1)
