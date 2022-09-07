#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
from glob import glob
import json
import logging
import math
import os
import re
from subprocess import run
from typing import (
    List,
    Tuple,
    Optional)

from vtkmodules.vtkIOCityGML import vtkCityGMLReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkIOGDAL import vtkGDALRasterReader
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkIOPDAL import vtkPDALReader
from vtkmodules.vtkIOImage import (
    vtkPNGWriter)
from vtkmodules.vtkIOCesium3DTiles import vtkCesium3DTilesWriter
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkFieldData,
    vtkImageData,
    vtkMultiBlockDataSet)
from vtkmodules.vtkCommonCore import (
    vtkStringArray,
    vtkUnsignedCharArray,
    vtkVersion)
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkImagingCore import vtkImageFlip
from danesfield import gdal_utils


def get_obj_texture_file_names(path: str) -> Tuple[List[str], List[str], List[str]]:
    """Given an OBJ file name return a file names for the textures
    (colors or properties) by removing .obj and looking for
    file_name.png, file_name_1.png, ..., file_name_x.tiff, ...  PNGs
    have to be specified before TIFFs. The third return lists property
    names inferred from the postfix of each TIFF file.

    """
    file_no_ext = os.path.splitext(os.path.basename(path))[0]
    dir = os.path.dirname(path)
    png_re = r'^' + re.escape(file_no_ext) + r'.*\.png$'
    png_files = [f for f in os.listdir(dir) if re.search(png_re, f, re.IGNORECASE)]
    tiff_re = r'^' + re.escape(file_no_ext) + r'_(.*)\.tiff$'
    tiff_files = []
    property_names = []
    for f in os.listdir(dir):
        m = re.search(tiff_re, f, re.IGNORECASE)
        if m:
            tiff_files.append(f)
            property_names.append(m.group(1))
    # remove from the list of PNGs files that have the same basename as TIFFs. Assume
    # those are quantized files previously generated
    png_files = [f + ".png" for f in set([os.path.splitext(f)[0] for f in png_files]).difference([os.path.splitext(f)[0] for f in tiff_files])]
    tiff_files.sort(key=lambda f: int(re.sub('\\D', '', f)))
    property_names.sort(key=lambda f: int(re.sub('\\D', '', f)))
    print("pngs: {}, tiffs: {}, names: {}".format(png_files, tiff_files, property_names))
    return (png_files, tiff_files, property_names)


def set_field(obj: vtkDataObject, name: str, values: List[str]):
    """
    Adds to an obj an field array name with 1 element value.
    """
    field_data = obj.GetFieldData()
    if not field_data:
        newfd = vtkFieldData()
        obj.SetFieldData(newfd)
        field_data = newfd
    string_array = vtkStringArray()
    string_array.SetNumberOfTuples(len(values))
    print("{}: ".format(name), end='')
    for i, value in enumerate(values):
        print(value, end=', ')
        string_array.SetValue(i, value)
    print()
    string_array.SetName(name)
    field_data.AddArray(string_array)


UNINITIALIZED : int = 2147483647


property_texture_template = """{
        "EXT_structural_metadata": {
            "schema": {
                "classes": {
                    "mesh": {
                        "name": "Mesh GPM error",
                        "properties": {
                        }
                    }
                }
            },
            "propertyTextures": [
                {
                    "class": "mesh",
                    "properties": {
                    }
                }
            ]
        }
    }"""

def quantize(obj_path: str, tiff_files: List[str], property_names: List[str], texture_index: int, features_range: List[Tuple[float, float]], generate_json: bool, property_texture_flip_y: bool) -> Tuple[List[str], Optional[str]]:
    """
    Quantizes a float array in each tiff file to one component in a RGBA png file and
    generates a json describing property textures encoded if generate_json is true.
    The json file is described by
    https://github.com/CesiumGS/3d-tiles/tree/main/specification/Metadata
    and
    https://github.com/CesiumGS/glTF/tree/3d-tiles-next/extensions/2.0/Vendor/EXT_structural_metadata
    """
    dir = os.path.dirname(obj_path)
    file_no_ext = os.path.splitext(os.path.basename(obj_path))[0]
    png_count = math.ceil(len(tiff_files) / 4)
    quantized_files = []
    pt = json.loads(property_texture_template)
    schema_properties = pt['EXT_structural_metadata']['schema']['classes']\
        ['mesh']['properties']
    propertyTextures_properties = pt['EXT_structural_metadata']['propertyTextures'][0]\
        ['properties']
    for i in range(png_count):
        # read 4 tiff files per png file
        png_data = vtkImageData()
        png_array = vtkUnsignedCharArray()
        png_array.SetNumberOfComponents(4)
        for j in range(4):
            tiff_index = i*4 + j
            if tiff_index < len(tiff_files):
                print("reading tiff: {}".format(dir + "/" + tiff_files[tiff_index]))
                tiff_reader = vtkGDALRasterReader()
                tiff_reader.SetFileName(dir + "/" + tiff_files[tiff_index])
                tiff_reader.Update()
                tiff_data = tiff_reader.GetOutput()
                tiff_array = tiff_data.GetPointData().GetArray(0)
                tiff_data_in_cells = False
                if tiff_array is None:
                    tiff_array = tiff_data.GetCellData().GetArray(0)
                    tiff_data_in_cells = True
                xRange = tiff_array.GetRange()
                features_range[tiff_index] = (min(features_range[tiff_index][0], xRange[0]),
                                              max(features_range[tiff_index][1], xRange[1]))
                xRange = features_range[tiff_index]
                if generate_json:
                    property_name = "c" + property_names[tiff_index]
                    schema_properties[property_name] = {}
                    schema_properties[property_name]["name"] = "Covariance " + property_name
                    schema_properties[property_name]["type"] = "SCALAR"
                    schema_properties[property_name]["componentType"] = "UINT8"
                    schema_properties[property_name]["normalized"] = True
                    schema_properties[property_name]["offset"] = xRange[0]
                    schema_properties[property_name]["scale"] = xRange[1] - xRange[0]

                    propertyTextures_properties[property_name] = {}
                    propertyTextures_properties[property_name]["index"] = texture_index + i
                    propertyTextures_properties[property_name]["texCoord"] = 0
                    propertyTextures_properties[property_name]["channels"] = [ j ]

                dims = tiff_data.GetDimensions()
                if tiff_data_in_cells:
                    dims = (dims[0] - 1, dims[1] - 1, 1)
                if j == 0:
                    png_data.SetDimensions(dims)
                    png_array.SetNumberOfTuples(dims[0] * dims[1])
                else:
                    prev_dims = png_data.GetDimensions()
                    if not prev_dims == dims:
                        logging.error("TIFF files with different dimensions. 0: {} {}: {}".format(prev_dims, j, dims))
                for k in range(png_array.GetNumberOfTuples()):
                    x = tiff_array.GetComponent(k, 0)
                    png_array.SetComponent(
                        k, j,
                        (x - xRange[0]) / (xRange[1] - xRange[0]) * 255)
            else:
                for k in range(png_array.GetNumberOfTuples()):
                    png_array.SetComponent(k, j, 0)
        png_data.GetPointData().SetScalars(png_array)

        # we have to flip the PNG because how TIFFs are saved in VTK
        flip_y_filter = None
        png_writer = vtkPNGWriter()
        if property_texture_flip_y:
            flip_y_filter = vtkImageFlip()
            flip_y_filter.SetFilteredAxis(1)
            flip_y_filter.SetInputDataObject(png_data)
            png_writer.SetInputConnection(flip_y_filter.GetOutputPort())
        else:
            png_writer.SetInputDataObject(png_data)
        quantized_file_basename = file_no_ext + "_" + str(texture_index + i)
        quantized_files.append( quantized_file_basename + ".png")
        print("writing png: {}".format(dir + "/" + quantized_file_basename + ".png"))
        png_writer.SetFileName(dir + "/" + quantized_file_basename + ".png")
        png_writer.Write()
    property_texture_file = None
    if generate_json:
        property_texture_file = dir + "/property_texture.json"
        with open(property_texture_file, "w") as outfile:
            json.dump(pt, outfile, indent=4)
    return (quantized_files, property_texture_file)


def read_buildings_obj(
        number_of_features : int, begin_feature_index : int, end_feature_index : int,
        _lod : int, files : List[str], file_offset : List[float], property_texture_flip_y: bool) ->  Tuple[Optional[vtkMultiBlockDataSet], Optional[str]]:
    """
    Builds a multiblock dataset (similar with one built by the CityGML reader)
    from a list of OBJ files. It can generate quantized png textures from property
    textures stored in tiff files. It generates a property texture description
    for the first OBJ - the assumption is that the property textures are the same for
    all textures in the dataset.
    It expects textures in order PNGs first and then TIFFs.
    """
    if number_of_features == UNINITIALIZED and \
       begin_feature_index != UNINITIALIZED and end_feature_index != UNINITIALIZED:
        feature_index_range = range(begin_feature_index, end_feature_index)
    else:
        feature_index_range = range(0, min(len(files), number_of_features))
    root = vtkMultiBlockDataSet()
    property_texture_file = None
    feature_range: List[Tuple[float, float]] = []
    for i in feature_index_range:
        reader = vtkOBJReader()
        reader.SetFileName(files[i])
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            logging.warning("Empty OBJ file: %s", files[i])
            continue
        (png_files, tiff_files, property_names) = get_obj_texture_file_names(files[i])
        if i == 0:
            gdal_utils.read_offset(files[i], file_offset)
            for _ in range(len(tiff_files)):
                feature_range.append((sys.float_info.max, sys.float_info.min))
        if len(feature_range) != len(tiff_files):
            logging.error("Number of properties for feature {} is different than "
                          "for feature 0: {} versus {}".format(
                              i, len(tiff_files), len(feature_range)))
        (quantized_files, ptf) = quantize(files[i], tiff_files, property_names, len(png_files), feature_range, i == len(feature_index_range) - 1, property_texture_flip_y)
        if ptf:
            property_texture_file = ptf
        set_field(polydata, "texture_uri", png_files + quantized_files)
        building = vtkMultiBlockDataSet()
        building.SetBlock(0, polydata)
        root.SetBlock(root.GetNumberOfBlocks(), building)
    return (root, property_texture_file)


def read_points_obj(
        number_of_features : int, begin_feature_index : int, end_feature_index : int,
        _lod : int, files : List[str], file_offset : List[float], property_texture_flip_y: bool) -> Tuple[Optional[vtkMultiBlockDataSet], Optional[str]]:
    """
    Builds a  pointset from a list of OBJ files.
    """
    append = vtkAppendPolyData()
    for i, file_name in enumerate(files):
        reader = vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        if i == 0:
            gdal_utils.read_offset(file_name, file_offset)
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            logging.warning("Empty OBJ file: %s", files[i])
            continue
        append.AddInputDataObject(polydata)
    append.Update()
    return (append.GetOutput(), None)


def read_points_vtp(
        number_of_features : int, begin_feature_index : int, end_feature_index : int,
        _lod : int, files : List[str], file_offset : List[float], property_texture_flip_y: bool) -> Tuple[Optional[vtkMultiBlockDataSet], Optional[str]]:
    """
    Builds a  pointset from a list of VTP files.
    """
    append = vtkAppendPolyData()
    for i, file_name in enumerate(files):
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            logging.warning("Empty VTP file: %s", files[i])
            continue
        append.AddInputDataObject(polydata)
    append.Update()
    return (append.GetOutput(), None)


def read_points_pdal(
        number_of_features : int, begin_feature_index : int, end_feature_index : int,
        _lod : int, files : List[str], file_offset : List[float], property_texture_flip_y: bool) -> Tuple[Optional[vtkMultiBlockDataSet], Optional[str]]:
    """
    Reads a point set from a list of pdal files
    """
    append = vtkAppendPolyData()
    for i, file_name in enumerate(files):
        reader = vtkPDALReader()
        reader.SetFileName(file_name)
        print("Reading {} ...".format(file_name))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            logging.warning("Empty PDAL file: %s", files[i])
            continue
        append.AddInputDataObject(polydata)
    append.Update()
    for i in range(3):
        file_offset[i] = 0
    return (append.GetOutput(), None)


def read_buildings_citygml(
        number_of_features : int, begin_feature_index : int, end_feature_index : int,
        lod : int, files : List[str], file_offset : List[float], property_texture_flip_y: bool) -> Tuple[Optional[vtkMultiBlockDataSet], Optional[str]]:
    """
    Reads a lod from a citygml file files[0], max number of buildins and sets
    the file_offset to 0.
    """
    for i in range(3):
        file_offset[i] = 0
    allBuildings = vtkMultiBlockDataSet()
    for i, file_name in enumerate(files):
        print("Reading {} ...".format(file_name))
        reader = vtkCityGMLReader()
        reader.SetFileName(file_name)
        reader.SetBeginBuildingIndex(begin_feature_index)
        reader.SetEndBuildingIndex(end_feature_index)
        reader.SetNumberOfBuildings(number_of_features)
        reader.SetLOD(lod)
        reader.Update()
        mb = reader.GetOutput()
        if not mb:
            logging.error("Expecting vtkMultiBlockDataSet")
            return (None, None)
        currentNumberOfBlocks = mb.GetNumberOfBlocks()
        allNumberOfBlocks = allBuildings.GetNumberOfBlocks()
        allBuildings.SetNumberOfBlocks(allNumberOfBlocks + currentNumberOfBlocks)
        # add all buildings to all
        buildingIt = mb.NewTreeIterator()
        buildingIt.VisitOnlyLeavesOff()
        buildingIt.TraverseSubTreeOff()
        buildingIt.InitTraversal()
        j = 0
        while not buildingIt.IsDoneWithTraversal():
            allBuildings.SetBlock(allNumberOfBlocks + j, buildingIt.GetCurrentDataObject())
            j = j + 1
            buildingIt.GoToNextItem()
    return (allBuildings, None)


READER = {
    ".obj": {vtkCesium3DTilesWriter.Buildings: read_buildings_obj,
             vtkCesium3DTilesWriter.Points: read_points_obj,
             vtkCesium3DTilesWriter.Mesh: read_buildings_obj},
    ".gml": {vtkCesium3DTilesWriter.Buildings: read_buildings_citygml},
    ".las": {vtkCesium3DTilesWriter.Points: read_points_pdal},
    ".laz": {vtkCesium3DTilesWriter.Points: read_points_pdal},
    ".vtp": {vtkCesium3DTilesWriter.Points: read_points_vtp}
}


def is_supported_path(filename: str):
    """
    Returns true if the filename is support as input in the converter.
    OBJ and CityGML are the only valid file types for now.
    """
    ext = os.path.splitext(filename)[1]
    return ext in READER


def get_files(input_list: List[str]) -> List[str]:
    """
    converts a list of files and directories into a list of files by reading
    the directories and appending all valid files inside to the list.
    """
    files = []
    for name in input_list:
        if os.path.exists(name):
            if os.path.isdir(name):
                # add all supported files from the directory
                dirname = os.fsencode(name)
                for file in os.listdir(dirname):
                    filename = os.fsdecode(file)
                    if not os.path.isdir(filename) and is_supported_path(filename):
                        files.append(name + "/" + filename)
            else:
                files.append(name)
        else:
            logging.warning("No such file or directory: %s", name)
    return files


def tiler(
        input_list: List[str], output_dir: str, number_of_features: int,
        begin_feature_index: int, end_feature_index: int,
        features_per_tile: int, lod: int, input_offset : List[float],
        dont_save_tiles: bool, dont_save_textures: bool, merge_tile_polydata: bool, input_type: int,
        content_gltf: bool, content_gltf_save_gltf:bool, points_color_array: str, crs: str,
        utm_zone: int, utm_hemisphere: str, property_texture_flip_y: bool):
    """
    Reads the input and converts it to 3D Tiles and saves it
    to output.
    """
    files = get_files(input_list)
    if len(files) == 0:
        raise Exception("No valid input files")

    file_offset:List[float] = [0, 0, 0]
    ext = os.path.splitext(files[0])[1]
    if ext not in READER or input_type not in READER[ext]:
        raise Exception("No valid reader for extension {} and input_type {}".format(
            ext, input_type))
    (root, property_texture_file) = READER[ext][input_type](
        number_of_features, begin_feature_index,
        end_feature_index, lod, files, file_offset, property_texture_flip_y)
    if root is None:
        return
    if points_color_array:
        root.GetPointData().SetActiveScalars(points_color_array)
    logging.info("Done parsing files")
    file_offset = list(a + b for a, b in zip(file_offset, input_offset))
    texture_path = os.path.dirname(files[0])
    writer = vtkCesium3DTilesWriter()
    writer.SetInputDataObject(root)
    writer.SetDirectoryName(output_dir)
    writer.SetTextureBaseDirectory(texture_path)
    writer.SetPropertyTextureFile(property_texture_file)
    writer.SetInputType(input_type)
    writer.SetContentGLTF(content_gltf)
    writer.SetContentGLTFSaveGLB(not content_gltf_save_gltf)
    writer.SetOffset(file_offset)
    writer.SetSaveTextures(not dont_save_textures)
    writer.SetSaveTiles(not dont_save_tiles)
    writer.SetMergeTilePolyData(merge_tile_polydata)
    writer.SetNumberOfFeaturesPerTile(features_per_tile)
    if not crs:
        crs = "+proj=utm +zone={}".format(utm_zone)
        if utm_hemisphere == "S":
            crs = crs + " +south"
        crs = crs + " datum=WGS84"
    writer.SetCRS(crs)
    writer.Write()
    logging.info("Done Write()")


class SmartFormatter(argparse.HelpFormatter):
    """
    Enable newline for options that start with N|
    """
    def _split_lines(self, text, width):
        if text.startswith('N|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def check_input_type(value: str) -> int:
    """
    Content type can be only 0: Buildings, 1: Points, 2: Mesh
    """
    ivalue = int(value)
    if ivalue < 0 or ivalue > 2:
        raise argparse.ArgumentTypeError("%s is an invalid input_type" % value)
    return ivalue


def main(args):
    """
    Converts large 3D geospatial datasets to the 3D Tiles format.
    """
    # conda adds PROJ_LIB if proj is part of installed packages which points
    # to the conda proj. Construct VTK_PROJ_LIB to point to the VTK proj.
    proj_lib = os.environ.get("PROJ_LIB")
    if proj_lib:
        proj_lib_dir = os.path.dirname(proj_lib)
        proj_lib_file = os.path.basename(proj_lib)
        version = vtkVersion()
        os.environ["VTK_PROJ_LIB"] = "{}/vtk-{}.{}/{}".format(
            proj_lib_dir, version.GetVTKMajorVersion(), version.GetVTKMinorVersion(),
            proj_lib_file)
    parser = argparse.ArgumentParser(
        description="Converts large 3D geospatial datasets to the 3D Tiles "
        "format.", formatter_class=SmartFormatter)
    parser.add_argument("-t", "--features_per_tile", type=int,
                        help="Maximum number of features (buildings or points) per tile.",
                        default=10)
    parser.add_argument("input", nargs="+", help="Input files (obj or citygml) or directories. "
                        "We read all files of a known type from each directory "
                        "and add them to the list.")
    parser.add_argument("-b", "--begin_feature_index", type=int,
                        default=0,
                        help="Begin feature index. Read [begin, end) range.")
    parser.add_argument("-e", "--end_feature_index", type=int,
                        default=UNINITIALIZED,
                        help="End feature index. Read [begin, end) range.")
    parser.add_argument("--dont_save_tiles", action="store_true",
                        help="Create only tileset.json not the B3DM files")
    parser.add_argument("--property_texture_flip_y", action="store_true",
                        help="Flip property textures along Y axis")
    parser.add_argument("--input_type",
                        help="Select input type Buildings (0), Points(1) or Mesh(2). ",
                        type=check_input_type, default=0)
    parser.add_argument("--content_gltf", action="store_true",
                        help="Store tile content using B3DM (or PNTS) or GLB/GLTF."
                        "GLB use the 3DTILES_content_gltf extension.")
    parser.add_argument("--content_gltf_save_gltf", action="store_true",
                        help="Store tile content as GLTF or GLB."
                        "Use the 3DTILES_content_gltf extension.")
    parser.add_argument("-l", "--lod", type=int,
                        help="Level of detail to be read (if available)",
                        default=2)
    parser.add_argument("--dont_save_textures", action="store_true",
                        help="Don't save textures even if available",)
    parser.add_argument("-m", "--merge_tile_polydata", action="store_true",
                        help="Merge tile polydata in one large mesh.",)
    parser.add_argument("-n", "--number_of_features", type=int,
                        default=UNINITIALIZED,
                        help="Maximum number of features read.")
    parser.add_argument("-o", "--output", required=True,
                        help="A directory where the 3d-tiles dataset is created. ")
    parser.add_argument("--crs",
                        help="Coordinate reference system (CRS) or spatial reference system (SRS)")
    parser.add_argument("--points_color_array",
                        help="Name of the array containing the RGB or RGBA. The values\n"
                        "in the array can be unsigned char and unsigned short.")
    parser.add_argument("--utm_hemisphere",
                        help="UTM hemisphere for the OBJ file coordinates.",
                        choices=["N", "S"], default="N")
    parser.add_argument("--translation", nargs=3, type=float,
                        default=[0, 0, 0],
                        help="N|Translation for x,y,z. "
                        "The translation can be also read as a comment in the OBJ file\n"
                        "using the following format at the top of the file:\n"
                        "#x offset: ...\n"
                        "#y offset: ...\n"
                        "#z offset: ...\n"
                        "When both are available, they are added up.")
    parser.add_argument("--utm_zone", type=int,
                        help="UTM zone for the OBJ file coordinates.",
                        choices=range(1, 61))

    args = parser.parse_args(args)

    if ((args.utm_zone is None or args.utm_hemisphere is None) and
            args.crs is None):
        raise Exception("Error: crs or utm_zone/utm_hemisphere are missing.")
    if UNINITIALIZED not in (args.number_of_features, args.end_feature_index):
        logging.warning("Cannot use both number_of_features and begin_feature_index, using later.")
        args.number_of_features = UNINITIALIZED
    tiler(
        args.input, args.output, args.number_of_features,
        args.begin_feature_index, args.end_feature_index,
        args.features_per_tile, args.lod, args.translation,
        args.dont_save_tiles, args.dont_save_textures, args.merge_tile_polydata,
        args.input_type,
        args.content_gltf, args.content_gltf_save_gltf, args.points_color_array, args.crs,
        args.utm_zone, args.utm_hemisphere, args.property_texture_flip_y)

    if args.input_type == 1:  # Points
        # meshoptimizer does not support points so convert to glb directly.
        # meshoptimizer prints the following error:
        # ignoring primitive 0 of mesh 0 because indexed points are not supported
        log_text = "Converting to glb ..."
        cmd_args_prefix = ["node", "/gltf-pipeline/bin/gltf-pipeline.js", "-i"]
    else:
        log_text = "Optimizing gltf and converting to glb ..."
        # noq is no quantization as Cesium 1.84 does not support it.
        cmd_args_prefix = ["/meshoptimizer/build/gltfpack", "-noq", "-i"]
    if args.input_type != 1 and (not args.content_gltf or
                                 (args.content_gltf and not args.content_gltf_save_gltf)):
        logging.info(log_text)
        gltf_files = glob(args.output + "/*/*.gltf")
        for gltf_file in gltf_files:
            cmd_args = cmd_args_prefix.copy()
            cmd_args.append(gltf_file)
            cmd_args.append("-o")
            cmd_args.append(os.path.splitext(gltf_file)[0] + '.glb')
            logging.info("Execute: %s", " ".join(cmd_args))
            run(cmd_args, check=True)
            os.remove(gltf_file)
        bin_files = glob(args.output + "/*/*.bin")
        for bin_file in bin_files:
             os.remove(bin_file)

    if args.input_type != 1 and not args.content_gltf:  # B3DM
        logging.info("Converting to b3dm ...")
        glb_files = glob(args.output + "/*/*.glb")
        for glb_file in glb_files:
            cmd_args = ["node", "/3d-tiles-validator/tools/bin/3d-tiles-tools.js",
                        "glbToB3dm", "-f"]
            cmd_args.append(glb_file)
            cmd_args.append(os.path.splitext(glb_file)[0] + '.b3dm')
            logging.info("Converting: %s", " ".join(cmd_args))
            run(cmd_args, check=True)
            os.remove(glb_file)
    logging.info("Done")


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
