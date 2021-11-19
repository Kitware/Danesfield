#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
from glob import glob
import logging
import os
from subprocess import run

from danesfield import gdal_utils
import vtk


def get_obj_texture_file_name(filename):
    """
    Given an OBJ file name return a file name for the texture by removing .obj and
    adding .png
    """
    file_no_ext = os.path.splitext(os.path.basename(filename))[0]
    return file_no_ext + ".png"


def set_field(obj, name, value):
    """
    Adds to an obj an field array name with 1 element value.
    """
    field_data = obj.GetFieldData()
    if not field_data:
        newfd = vtk.vtkFieldData()
        obj.SetFieldData(newfd)
        field_data = newfd
    string_array = vtk.vtkStringArray()
    string_array.SetNumberOfTuples(1)
    string_array.SetValue(0, value)
    string_array.SetName(name)
    field_data.AddArray(string_array)


UNINITIALIZED = 2147483647


def read_obj_files(
        number_of_buildings,
        begin_building_index, end_building_index, lod,
        files, file_offset):
    """
    Builds a multiblock dataset (similar with one built by the CityGML reader)
    from a list of OBJ files.
    """
    if number_of_buildings == UNINITIALIZED and begin_building_index != UNINITIALIZED and end_building_index != UNINITIALIZED:
        building_range = range(begin_building_index, end_building_index)
    else:
        building_range = range(0, min(len(files), number_of_buildings))
    root = vtk.vtkMultiBlockDataSet()
    for i in building_range:
        reader = vtk.vtkOBJReader()
        reader.SetFileName(files[i])
        reader.Update()
        if i == 0:
            gdal_utils.read_offset(files[i], file_offset)
        polydata = reader.GetOutput()
        texture_file = get_obj_texture_file_name(files[i])
        set_field(polydata, "texture_uri", texture_file)
        building = vtk.vtkMultiBlockDataSet()
        building.SetBlock(0, polydata)
        root.SetBlock(root.GetNumberOfBlocks(), building)
    return root


def read_city_gml_files(
        number_of_buildings,
        begin_building_index, end_building_index, lod,
        files, file_offset):
    """
    Reads a lod from a citygml file files[0], max number of buildins and sets
    the file_offset to 0.
    """
    logging.info("Parsing: %s", files[0])
    if len(files) > 1:
        logging.warning("Can only process one CityGML file for now.")
    reader = vtk.vtkCityGMLReader()
    reader.SetFileName(files[0])
    reader.SetBeginBuildingIndex(begin_building_index)
    reader.SetEndBuildingIndex(end_building_index)
    reader.SetNumberOfBuildings(number_of_buildings)
    reader.SetLOD(lod)
    reader.Update()
    root = reader.GetOutput()
    if not root:
        logging.error("Expecting vtkMultiBlockDataSet")
        return None
    file_offset[:] = [0, 0, 0]
    return root


READER = {
    ".obj": read_obj_files,
    ".gml": read_city_gml_files
}


def is_supported(filename):
    """
    Returns true if the filename is support as input in the converter.
    OBJ and CityGML are the only valid file types for now.
    """
    ext = os.path.splitext(filename)[1]
    return ext in READER


def get_files(input_list):
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
                    if not os.path.isdir(filename) and is_supported(filename):
                        files.append(name + "/" + filename)
            else:
                files.append(name)
        else:
            logging.warning("No such file or directory: %s", name)
    return files


def tiler(
        input_list, output, number_of_buildings,
        begin_building_index, end_building_index,
        buildings_per_tile, lod, input_offset,
        dont_save_gltf, dont_save_textures, content_type, crs,
        utm_zone, utm_hemisphere):
    """
    Reads the input and converts it to 3D Tiles and saves it
    to output.
    """
    files = get_files(input_list)
    if len(files) == 0:
        raise Exception("No valid input files")

    file_offset = [0, 0, 0]
    ext = os.path.splitext(files[0])[1]
    root = READER[ext](
        number_of_buildings, begin_building_index,
        end_building_index, lod, files, file_offset)
    logging.info("Done parsing files")
    file_offset = [a + b for a, b in zip(file_offset, input_offset)]
    texture_path = os.path.dirname(files[0])

    writer = vtk.vtkCesium3DTilesWriter()
    writer.SetInputDataObject(root)
    writer.SetDirectoryName(output)
    writer.SetTexturePath(texture_path)
    writer.SetContentType(content_type)
    writer.SetOffset(file_offset)
    writer.SetSaveTextures(not dont_save_textures)
    writer.SetSaveGLTF(not dont_save_gltf)
    writer.SetNumberOfBuildingsPerTile(buildings_per_tile)
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


def check_content_type(value):
    """
    Content type can be only 0: B3DM, 1:GLB, 2:GLTF
    """
    ivalue = int(value)
    if ivalue < 0 or ivalue > 2:
        raise argparse.ArgumentTypeError("%s is an invalid content_type" % value)
    return ivalue


def main(args):
    """
    Converts large 3D geospatial datasets to the 3D Tiles format.
    """
    parser = argparse.ArgumentParser(
        description="Converts large 3D geospatial datasets to the 3D Tiles "
        "format.", formatter_class=SmartFormatter)
    parser.add_argument("-t", "--buildings_per_tile", type=int,
                        help="Maximum number of buildings per tile.",
                        default=10)
    parser.add_argument("input", nargs="+", help="Input files (obj or citygml) or directories. "
                        "We read all files of a known type from each directory "
                        "and add them to the list.")
    parser.add_argument("-b", "--begin_building_index", type=int,
                        default=0,
                        help="Begin building index. Read [begin, end) range.")
    parser.add_argument("-e", "--end_building_index", type=int,
                        default=UNINITIALIZED,
                        help="End building index. Read [begin, end) range.")
    parser.add_argument("--dont_save_gltf", action="store_true",
                        help="Create only tileset.json not the B3DM files")
    parser.add_argument("--content_type",
                        help="Store tile content using B3DM (0), GLB(1) or GLTF(2). "
                        "GLTF and GLB use the 3DTILES_content_gltf extension.",
                        type=check_content_type, default=0)
    parser.add_argument("-l", "--lod", action="store_true",
                        help="Level of detail to be read (if available)",
                        default=2)
    parser.add_argument("--dont_save_textures", action="store_true",
                        help="Don't save textures even if available",)
    parser.add_argument("-n", "--number_of_buildings", type=int,
                        default=UNINITIALIZED,
                        help="Maximum number of buildings read.")
    parser.add_argument("-o", "--output", required=True,
                        help="A directory where the 3d-tiles dataset is created. ")
    parser.add_argument("--crs",
                        help="Coordinate reference system (CRS) or spatial reference system (SRS)")
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
    if args.number_of_buildings != UNINITIALIZED and args.begin_building_index != UNINITIALIZED:
        logging.warning("Cannot use both number_of_buildings and begin_building_index, using later.")
        args.number_of_buildings = UNINITIALIZED
    tiler(
        args.input, args.output, args.number_of_buildings,
        args.begin_building_index, args.end_building_index,
        args.buildings_per_tile, args.lod, args.translation,
        args.dont_save_gltf, args.dont_save_textures, args.content_type, args.crs,
        args.utm_zone, args.utm_hemisphere)

    if args.content_type < 2:  # B3DM or GLB
        logging.info("Optimizing gltf and converting to glb ...")
        gltf_files = glob(args.output + "/*/*.gltf")
        for gltf_file in gltf_files:
            # noq is no quantization as Cesium 1.84 does not support it.
            cmd_args = ["/meshoptimizer/build/gltfpack", "-noq", "-i"]
            cmd_args.append(gltf_file)
            cmd_args.append("-o")
            cmd_args.append(os.path.splitext(gltf_file)[0] + '.glb')
            logging.info("Optimizing: %s", " ".join(cmd_args))
            run(cmd_args, check=True)
            os.remove(gltf_file)
        bin_files = glob(args.output + "/*/*.bin")
        for bin_file in bin_files:
            os.remove(bin_file)

        if args.content_type < 1:  # B3DM
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
